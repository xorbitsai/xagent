"""Native Amazon Bedrock Converse/ConverseStream chat adapter."""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import logging
import threading
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import urlsplit

from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from ....runtime_performance import run_in_thread_with_telemetry
from ....utils.security import redact_sensitive_text
from ..error import matches_context_length_error
from ..exceptions import LLMContextLengthError, LLMRetryableError
from ..token_context import add_token_usage
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 4096
_RETRYABLE_CODES = frozenset(
    {
        "InternalServerException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "ServiceUnavailableException",
        "ThrottlingException",
    }
)
_RETRYABLE_STREAM_EVENTS = frozenset(
    {
        "internalServerException",
        "modelStreamErrorException",
        "modelTimeoutException",
        "serviceUnavailableException",
        "throttlingException",
    }
)
_IMAGE_MEDIA_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _usage_payload(usage: Mapping[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    input_tokens = _safe_int(usage.get("inputTokens"))
    output_tokens = _safe_int(usage.get("outputTokens"))
    payload = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _safe_int(usage.get("totalTokens"))
        or input_tokens + output_tokens,
    }
    cache_read = _safe_int(usage.get("cacheReadInputTokens"))
    cache_write = _safe_int(usage.get("cacheWriteInputTokens"))
    if cache_read:
        payload["cached_input_tokens"] = cache_read
    if cache_write:
        payload["cache_write_input_tokens"] = cache_write
    return payload


def _stop_reason(value: Any) -> str:
    reason = str(value or "")
    return "tool_calls" if reason == "tool_use" else reason


def _tool_call(block: Mapping[str, Any], index: int) -> dict[str, Any]:
    tool_use = block.get("toolUse")
    if not isinstance(tool_use, Mapping):
        raise ValueError("Bedrock toolUse content must be an object")
    tool_id = str(tool_use.get("toolUseId") or "")
    name = str(tool_use.get("name") or "")
    if not tool_id or not name:
        raise ValueError("Bedrock toolUse content requires toolUseId and name")
    return {
        "id": tool_id,
        "index": index,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                tool_use.get("input", {}), ensure_ascii=False, separators=(",", ":")
            ),
        },
    }


class BedrockLLM(BaseLLM):
    """Call Bedrock's provider-neutral Converse APIs through boto3.

    boto3 is synchronous, so every network operation and stream read runs in
    the shared worker pool. An explicit Bedrock API key is attached to this
    client only; it is never written to ``AWS_BEARER_TOKEN_BEDROCK`` or any
    other process-global environment variable.
    """

    def __init__(
        self,
        model_name: str,
        *,
        region_name: str | None = None,
        api_key: str | None = None,
        endpoint_url: str | None = None,
        auth_mode: str = "auto",
        default_temperature: float | None = None,
        default_max_tokens: int | None = None,
        timeout: float = 180.0,
        abilities: list[str] | None = None,
        client: Any = None,
    ) -> None:
        self._model_name = model_name
        self.region_name = region_name or None
        self.api_key = api_key or None
        self.endpoint_url = endpoint_url or None
        self.auth_mode = auth_mode.strip().lower() if auth_mode else "auto"
        if self.auth_mode not in {"auto", "api_key", "aws_credentials"}:
            raise ValueError(
                "Bedrock auth_mode must be auto, api_key, or aws_credentials"
            )
        if self.auth_mode == "api_key" and not self.api_key:
            raise ValueError("Bedrock auth_mode=api_key requires an api_key")
        if self.auth_mode == "aws_credentials" and self.api_key:
            raise ValueError(
                "Bedrock auth_mode=aws_credentials cannot include an api_key"
            )
        self._validate_endpoint(self.endpoint_url)
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        self._abilities = list(abilities or ["chat", "tool_calling", "vision"])
        self._client = client
        self._client_lock = threading.Lock()

    @staticmethod
    def _validate_endpoint(endpoint_url: str | None) -> None:
        if not endpoint_url:
            return
        parsed = urlsplit(endpoint_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Bedrock endpoint_url must be an absolute HTTP(S) URL")
        if parsed.hostname.lower().startswith("bedrock-mantle."):
            raise ValueError(
                "Bedrock Converse requires a bedrock-runtime endpoint; "
                "bedrock-mantle does not support Converse"
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def abilities(self) -> list[str]:
        return self._abilities

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    @property
    def supports_json_schema_response_format(self) -> bool:
        return False

    @property
    def supports_json_object_response_format(self) -> bool:
        return False

    def _build_client(self) -> Any:
        import boto3

        key_auth = self.auth_mode == "api_key" or (
            self.auth_mode == "auto" and self.api_key is not None
        )
        config = Config(
            signature_version=UNSIGNED if key_auth else None,
            connect_timeout=self.timeout,
            read_timeout=self.timeout,
            retries={"total_max_attempts": 1, "mode": "standard"},
        )
        client = boto3.client(
            "bedrock-runtime",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            config=config,
        )
        if key_auth:
            api_key = self.api_key
            assert api_key is not None

            def add_bearer_header(request: Any, **_: Any) -> None:
                request.headers["Authorization"] = f"Bearer {api_key}"

            client.meta.events.register(
                "before-send.bedrock-runtime.*", add_bearer_header
            )
        return client

    def _ensure_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._build_client()
        return self._client

    async def _get_client(self) -> Any:
        return await run_in_thread_with_telemetry(
            "bedrock.create_client", self._ensure_client
        )

    @staticmethod
    def _decode_image(part: Mapping[str, Any]) -> dict[str, Any]:
        image_url = part.get("image_url")
        if isinstance(image_url, Mapping):
            url = image_url.get("url")
        else:
            url = image_url
        if not isinstance(url, str) or not url.startswith("data:"):
            raise ValueError(
                "Bedrock image input requires an inline data URL; remote image "
                "URLs are not silently downloaded"
            )
        try:
            header, encoded = url.split(",", 1)
            media_type, marker = header[5:].split(";", 1)
            image_format = _IMAGE_MEDIA_TYPES[media_type.lower()]
            if marker.lower() != "base64":
                raise ValueError
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, KeyError, binascii.Error) as exc:
            raise ValueError(
                "Bedrock image input must be a base64 data URL in PNG, JPEG, "
                "GIF, or WebP format"
            ) from exc
        return {"image": {"format": image_format, "source": {"bytes": data}}}

    @classmethod
    def _content_blocks(cls, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"text": content}]
        if not isinstance(content, list):
            raise ValueError("Bedrock message content must be text or a content list")
        blocks: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise ValueError("Bedrock content parts must be objects")
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                blocks.append({"text": str(part.get("text") or "")})
            elif part_type in {"image_url", "input_image"}:
                blocks.append(cls._decode_image(part))
            else:
                raise ValueError(f"Unsupported Bedrock content part: {part_type!r}")
        return blocks

    @staticmethod
    def _tool_result_content(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, (dict, list, int, float, bool)) or content is None:
            return [{"json": content}]
        text = str(content)
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            return [{"text": text}]
        return [{"json": decoded}]

    @classmethod
    def _convert_messages(
        cls, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        system: list[dict[str, str]] = []
        converted: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                converted.append(
                    {"role": "user", "content": list(pending_tool_results)}
                )
                pending_tool_results.clear()

        for message in messages:
            role = message.get("role")
            if role == "system":
                blocks = cls._content_blocks(message.get("content", ""))
                if any("text" not in block for block in blocks):
                    raise ValueError("Bedrock system messages only support text")
                system.extend({"text": str(block["text"])} for block in blocks)
                continue
            if role == "tool":
                tool_id = message.get("tool_call_id")
                if not tool_id:
                    raise ValueError("Bedrock tool result requires tool_call_id")
                result: dict[str, Any] = {
                    "toolUseId": str(tool_id),
                    "content": cls._tool_result_content(message.get("content")),
                }
                if message.get("status") in {"success", "error"}:
                    result["status"] = message["status"]
                pending_tool_results.append({"toolResult": result})
                continue

            flush_tool_results()
            if role not in {"user", "assistant"}:
                raise ValueError(f"Unsupported Bedrock message role: {role!r}")
            content = message.get("content", "")
            blocks = (
                []
                if role == "assistant" and content == "" and message.get("tool_calls")
                else cls._content_blocks(content)
            )
            if role == "assistant":
                for raw_call in message.get("tool_calls") or []:
                    function = raw_call.get("function") or {}
                    tool_id = str(raw_call.get("id") or "")
                    tool_name = str(function.get("name") or "")
                    if not tool_id or not tool_name:
                        raise ValueError(
                            "Bedrock assistant tool calls require id and function name"
                        )
                    arguments = function.get("arguments") or "{}"
                    try:
                        parsed_arguments = json.loads(arguments)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "Bedrock assistant tool arguments must be valid JSON"
                        ) from exc
                    blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tool_id,
                                "name": tool_name,
                                "input": parsed_arguments,
                            }
                        }
                    )
            if role == "assistant" and any("image" in block for block in blocks):
                raise ValueError("Bedrock images are only supported in user messages")
            converted.append({"role": role, "content": blocks})
        flush_tool_results()
        if not converted:
            raise ValueError(
                "Bedrock Converse requires at least one non-system message"
            )
        return system, converted

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for tool in tools:
            function = tool.get("function") if tool.get("type") == "function" else tool
            if not isinstance(function, Mapping) or not function.get("name"):
                raise ValueError("Bedrock tools require a function name")
            spec: dict[str, Any] = {
                "name": str(function["name"]),
                "inputSchema": {"json": function.get("parameters") or {}},
            }
            if function.get("description"):
                spec["description"] = str(function["description"])
            if "strict" in function:
                spec["strict"] = bool(function["strict"])
            converted.append({"toolSpec": spec})
        return converted

    @staticmethod
    def _convert_tool_choice(choice: Any) -> dict[str, Any] | None:
        if choice in {None, "auto"}:
            return {"auto": {}}
        if choice in {"required", "any"}:
            return {"any": {}}
        if isinstance(choice, str):
            if choice == "none":
                return None
            return {"tool": {"name": choice}}
        if isinstance(choice, Mapping):
            function = choice.get("function")
            name = function.get("name") if isinstance(function, Mapping) else None
            name = name or choice.get("name")
            if name:
                return {"tool": {"name": str(name)}}
        raise ValueError("Unsupported Bedrock tool_choice")

    def _request(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: dict[str, Any] | None,
        thinking: dict[str, Any] | None,
        output_config: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if response_format is not None or output_config is not None:
            raise ValueError(
                "Bedrock provider does not declare JSON response-format support"
            )
        if thinking is not None and thinking.get("type") != "disabled":
            raise ValueError(
                "Bedrock provider does not declare model-independent reasoning support"
            )
        system, converted_messages = self._convert_messages(
            self._strip_internal_message_keys(messages)
        )
        inference: dict[str, Any] = {
            "maxTokens": max_tokens
            if max_tokens is not None
            else self.default_max_tokens or _DEFAULT_MAX_TOKENS
        }
        selected_temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        if selected_temperature is not None:
            inference["temperature"] = selected_temperature
        top_p = kwargs.pop("top_p", None)
        if top_p is not None:
            inference["topP"] = top_p
        stop = kwargs.pop("stop", None)
        if stop is not None:
            inference["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
        additional = kwargs.pop("additional_model_request_fields", None)
        if kwargs:
            raise ValueError(
                "Unsupported Bedrock request parameters: " + ", ".join(sorted(kwargs))
            )
        request: dict[str, Any] = {
            "modelId": self._model_name,
            "messages": self._sanitize_unicode_content(converted_messages),
            "inferenceConfig": inference,
        }
        if system:
            request["system"] = self._sanitize_unicode_content(system)
        if additional is not None:
            request["additionalModelRequestFields"] = additional
        if tools and tool_choice != "none":
            request["toolConfig"] = {"tools": self._convert_tools(tools)}
            converted_choice = self._convert_tool_choice(tool_choice)
            if converted_choice is not None:
                request["toolConfig"]["toolChoice"] = converted_choice
        elif tool_choice not in {None, "none"}:
            raise ValueError("Bedrock tool_choice requires tools")
        return request

    def _record_usage(
        self, usage: Mapping[str, Any] | None, call_type: str
    ) -> dict[str, int]:
        payload = _usage_payload(usage)
        add_token_usage(
            input_tokens=payload["input_tokens"],
            output_tokens=payload["output_tokens"],
            model=self._model_name,
            model_id=self.model_id,
            call_type=call_type,
            cached_input_tokens=payload.get("cached_input_tokens", 0),
            cache_write_input_tokens=payload.get("cache_write_input_tokens", 0),
        )
        return payload

    def _raise_provider_error(self, error: BaseException) -> None:
        code = type(error).__name__
        status: int | None = None
        message = str(error)
        if isinstance(error, ClientError):
            response = error.response if isinstance(error.response, dict) else {}
            error_info = response.get("Error") or {}
            code = str(error_info.get("Code") or code)
            message = str(error_info.get("Message") or code)
            metadata = response.get("ResponseMetadata") or {}
            raw_status = metadata.get("HTTPStatusCode")
            status = raw_status if isinstance(raw_status, int) else None
        safe = redact_sensitive_text(message)
        detail = f"Bedrock API error [{code}]"
        if status is not None:
            detail += f" ({status})"
        detail += f": {safe}"
        if matches_context_length_error(error):
            raise LLMContextLengthError(detail) from error
        if (
            code in _RETRYABLE_CODES
            or (status is not None and (status in {408, 409, 429} or status >= 500))
            or isinstance(
                error,
                (
                    ConnectTimeoutError,
                    ConnectionClosedError,
                    EndpointConnectionError,
                    ReadTimeoutError,
                ),
            )
        ):
            raise LLMRetryableError(detail) from error
        raise RuntimeError(detail) from error

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request = self._request(
            messages,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            response_format,
            thinking,
            output_config,
            dict(kwargs),
        )
        client = await self._get_client()
        try:
            response = await run_in_thread_with_telemetry(
                "bedrock.converse", client.converse, **request
            )
        except (BotoCoreError, ClientError) as exc:
            self._raise_provider_error(exc)
        usage = self._record_usage(response.get("usage"), "chat")
        output = response.get("output") or {}
        message = output.get("message") or {}
        content = message.get("content") or []
        unsupported_blocks = [
            sorted(str(key) for key in block if key not in {"text", "toolUse"})
            for block in content
            if isinstance(block, Mapping)
            and any(key not in {"text", "toolUse"} for key in block)
        ]
        if unsupported_blocks:
            raise RuntimeError(
                "Bedrock returned unsupported content blocks: "
                + ", ".join("/".join(keys) for keys in unsupported_blocks)
            )
        text = "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, Mapping) and "text" in block
        )
        calls = [
            _tool_call(block, index)
            for index, block in enumerate(content)
            if isinstance(block, Mapping) and "toolUse" in block
        ]
        finish_reason = _stop_reason(response.get("stopReason"))
        raw = {
            key: response[key]
            for key in ("output", "stopReason", "usage", "metrics")
            if key in response
        }
        if calls:
            return {
                "type": "tool_call",
                "content": text,
                "tool_calls": calls,
                "finish_reason": finish_reason,
                "usage": usage,
                "raw": raw,
            }
        if not text:
            if response.get("stopReason") == "model_context_window_exceeded":
                raise LLMContextLengthError("Bedrock model context window was exceeded")
            if response.get("stopReason") in {
                "content_filtered",
                "guardrail_intervened",
                "malformed_model_output",
                "malformed_tool_use",
            }:
                raise RuntimeError(
                    "Bedrock stopped without content: "
                    + str(response.get("stopReason"))
                )
            raise LLMRetryableError("Bedrock returned empty content and no tool calls")
        return {
            "type": "text",
            "content": text,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw": raw,
        }

    async def vision_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        if not self.has_ability("vision"):
            raise RuntimeError(
                f"Model {self._model_name} does not support vision capabilities"
            )
        return await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )

    @staticmethod
    def _next_event(iterator: Any) -> tuple[bool, Any]:
        try:
            return True, next(iterator)
        except StopIteration:
            return False, None

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        request = self._request(
            messages,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            response_format,
            thinking,
            output_config,
            dict(kwargs),
        )
        client = await self._get_client()
        event_stream = None
        current_text = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        try:
            response = await run_in_thread_with_telemetry(
                "bedrock.converse_stream", client.converse_stream, **request
            )
            event_stream = response["stream"]
            iterator = iter(event_stream)
            while True:
                present, event = await run_in_thread_with_telemetry(
                    "bedrock.stream_next", self._next_event, iterator
                )
                if not present:
                    break
                if "contentBlockStart" in event:
                    start_event = event["contentBlockStart"]
                    index = int(start_event.get("contentBlockIndex", 0))
                    tool_use = (start_event.get("start") or {}).get("toolUse")
                    if tool_use:
                        tool_calls[index] = {
                            "id": str(tool_use.get("toolUseId") or ""),
                            "index": index,
                            "type": "function",
                            "function": {
                                "name": str(tool_use.get("name") or ""),
                                "arguments": "",
                                "arguments_mode": "delta",
                            },
                        }
                    elif start_event.get("start"):
                        raise RuntimeError(
                            "Bedrock stream returned an unsupported content block start"
                        )
                elif "contentBlockDelta" in event:
                    delta_event = event["contentBlockDelta"]
                    index = int(delta_event.get("contentBlockIndex", 0))
                    delta = delta_event.get("delta") or {}
                    if "text" in delta:
                        text_delta = str(delta.get("text") or "")
                        current_text += text_delta
                        yield StreamChunk(
                            type=ChunkType.TOKEN,
                            content=current_text,
                            delta=text_delta,
                            raw=event,
                        )
                    if "toolUse" in delta:
                        call = tool_calls.get(index)
                        if call is None:
                            raise RuntimeError(
                                "Bedrock stream emitted tool input before tool start"
                            )
                        fragment = str(delta["toolUse"].get("input") or "")
                        call["function"]["arguments"] += fragment
                        yield StreamChunk(
                            type=ChunkType.TOOL_CALL,
                            tool_calls=[copy.deepcopy(call)],
                            raw=event,
                        )
                    unsupported_delta = set(delta) - {"text", "toolUse"}
                    if unsupported_delta:
                        raise RuntimeError(
                            "Bedrock stream returned unsupported content delta: "
                            + ", ".join(sorted(unsupported_delta))
                        )
                elif "contentBlockStop" in event:
                    index = int(event["contentBlockStop"].get("contentBlockIndex", 0))
                    if index in tool_calls:
                        completed_call = copy.deepcopy(tool_calls[index])
                        # Argument fragments above are appended by the shared
                        # runtime. The stop event repeats the accumulated value,
                        # so expose it as a snapshot instead of a final delta.
                        completed_call["function"]["arguments_mode"] = "snapshot"
                        yield StreamChunk(
                            type=ChunkType.TOOL_CALL,
                            tool_calls=[completed_call],
                            raw=event,
                        )
                elif "messageStop" in event:
                    raw_stop_reason = event["messageStop"].get("stopReason")
                    finish_reason = _stop_reason(raw_stop_reason)
                elif "metadata" in event:
                    usage = self._record_usage(
                        event["metadata"].get("usage"), "stream_chat"
                    )
                    yield StreamChunk(
                        type=ChunkType.USAGE,
                        usage=usage,
                        finish_reason=finish_reason,
                        raw=event,
                    )
                elif any(key.endswith("Exception") for key in event):
                    key = next(key for key in event if key.endswith("Exception"))
                    message = (event[key] or {}).get("message") or key
                    detail = (
                        f"Bedrock stream error [{key}]: "
                        f"{redact_sensitive_text(str(message))}"
                    )
                    if key in _RETRYABLE_STREAM_EVENTS:
                        raise LLMRetryableError(detail)
                    raise RuntimeError(detail)
                elif "messageStart" not in event:
                    raise RuntimeError("Bedrock stream returned an unknown event")
            if not current_text and not tool_calls:
                if finish_reason == "model_context_window_exceeded":
                    raise LLMContextLengthError(
                        "Bedrock model context window was exceeded"
                    )
                if finish_reason in {
                    "content_filtered",
                    "guardrail_intervened",
                    "malformed_model_output",
                    "malformed_tool_use",
                }:
                    raise RuntimeError(
                        f"Bedrock stream stopped without content: {finish_reason}"
                    )
            yield StreamChunk(
                type=ChunkType.END,
                content=current_text,
                finish_reason=finish_reason,
            )
        except asyncio.CancelledError:
            raise
        except (BotoCoreError, ClientError) as exc:
            self._raise_provider_error(exc)
        finally:
            close = getattr(event_stream, "close", None)
            if callable(close):
                await run_in_thread_with_telemetry("bedrock.stream_close", close)

    async def close(self) -> None:
        client = self._client
        self._client = None
        close = getattr(client, "close", None)
        if callable(close):
            await run_in_thread_with_telemetry("bedrock.close_client", close)
