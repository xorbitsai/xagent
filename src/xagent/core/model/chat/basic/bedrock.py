"""Native Amazon Bedrock Converse/ConverseStream chat adapter."""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import logging
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

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
from ....utils.public_botocore import PublicBotocoreSession
from ....utils.security import redact_sensitive_text
from ...network_policy import ModelNetworkPolicy
from ...providers import validate_bedrock_endpoint
from ...public_http import get_model_proxy_url
from ..error import matches_context_length_error
from ..exceptions import LLMContextLengthError, LLMRetryableError
from ..stream_deadline import await_before_deadline
from ..timeout_config import TimeoutConfig
from ..token_context import add_token_usage
from ..types import PROVIDER_STATE_METADATA_KEY, ChunkType, StreamChunk
from .base import BaseLLM
from .tool_choice import supports_forced_tool_choice

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
# The HTTP status each retryable ConverseStream exception event stands for.
# An in-band event carries no response metadata, so the raised error is given
# this status for provider-failure classification (Auto failover, the call
# boundary); the event's text stays out of it.
_RETRYABLE_STREAM_EVENTS = {
    "internalServerException": 500,
    "modelStreamErrorException": 500,
    "modelTimeoutException": 408,
    "serviceUnavailableException": 503,
    "throttlingException": 429,
}
_IMAGE_MEDIA_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}
_BEDROCK_PROVIDER_STATE_NAMESPACE = "bedrock"
_BEDROCK_REASONING_BLOCKS_KEY = "reasoning_content_blocks"
_BEDROCK_BYTES_KEY = "_xagent_base64_bytes"


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


def _encode_provider_value(value: Any) -> Any:
    """Make provider-owned binary state safe for JSON checkpoints."""
    if isinstance(value, bytes):
        return {_BEDROCK_BYTES_KEY: base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(key): _encode_provider_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_provider_value(item) for item in value]
    return copy.deepcopy(value)


def _decode_provider_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {_BEDROCK_BYTES_KEY}:
            encoded = value[_BEDROCK_BYTES_KEY]
            if not isinstance(encoded, str):
                raise ValueError("Bedrock reasoning binary state must be base64 text")
            try:
                return base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError(
                    "Bedrock reasoning binary state is not valid base64"
                ) from exc
        return {str(key): _decode_provider_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_provider_value(item) for item in value]
    return copy.deepcopy(value)


def _validate_reasoning_content(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Bedrock reasoningContent must be an object")
    keys = set(value)
    if keys == {"reasoningText"}:
        reasoning_text = value["reasoningText"]
        if not isinstance(reasoning_text, Mapping):
            raise ValueError("Bedrock reasoningText must be an object")
        if set(reasoning_text) - {"text", "signature"} or not isinstance(
            reasoning_text.get("text"), str
        ):
            raise ValueError(
                "Bedrock reasoningText requires text and an optional signature"
            )
        if "signature" in reasoning_text and not isinstance(
            reasoning_text["signature"], str
        ):
            raise ValueError("Bedrock reasoningText signature must be text")
    elif keys == {"redactedContent"}:
        if not isinstance(value["redactedContent"], bytes):
            raise ValueError("Bedrock redactedContent must be bytes")
    else:
        raise ValueError(
            "Bedrock reasoningContent must contain exactly one of "
            "reasoningText or redactedContent"
        )
    return copy.deepcopy(dict(value))


def _reasoning_provider_state(content: list[Any]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for index, block in enumerate(content):
        if not isinstance(block, Mapping) or "reasoningContent" not in block:
            continue
        reasoning_content = _validate_reasoning_content(block["reasoningContent"])
        blocks.append(
            {
                "index": index,
                "reasoningContent": _encode_provider_value(reasoning_content),
            }
        )
    if not blocks:
        return {}
    return {
        _BEDROCK_PROVIDER_STATE_NAMESPACE: {
            _BEDROCK_REASONING_BLOCKS_KEY: blocks,
        }
    }


def _reasoning_blocks_from_message(
    message: Mapping[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    provider_state = message.get(PROVIDER_STATE_METADATA_KEY)
    if not isinstance(provider_state, Mapping):
        return []
    bedrock_state = provider_state.get(_BEDROCK_PROVIDER_STATE_NAMESPACE)
    if not isinstance(bedrock_state, Mapping):
        return []
    encoded_blocks = bedrock_state.get(_BEDROCK_REASONING_BLOCKS_KEY)
    if not isinstance(encoded_blocks, list):
        raise ValueError("Bedrock reasoning provider state must contain a block list")
    restored: list[tuple[int, dict[str, Any]]] = []
    for item in encoded_blocks:
        if not isinstance(item, Mapping) or not isinstance(item.get("index"), int):
            raise ValueError("Bedrock reasoning provider state has an invalid block")
        reasoning_content = _decode_provider_value(item.get("reasoningContent"))
        restored.append(
            (
                item["index"],
                {"reasoningContent": _validate_reasoning_content(reasoning_content)},
            )
        )
    return restored


def _insert_reasoning_blocks(
    blocks: list[dict[str, Any]], restored: list[tuple[int, dict[str, Any]]]
) -> None:
    for index, block in sorted(restored, key=lambda item: item[0]):
        blocks.insert(min(max(index, 0), len(blocks)), block)


def _shutdown_event_stream(event_stream: Any) -> bool:
    """Make a worker blocked reading ``event_stream`` return; report success.

    botocore's ``EventStream`` wraps the urllib3 response. Its ``shutdown()``
    (urllib3 >= 2.3) shuts the socket down for reading, which wakes a read
    blocked in another thread. ``EventStream.close()`` cannot: it waits for the
    lock that blocked read holds, so it returns only when the read does.
    """
    raw_stream = getattr(event_stream, "_raw_stream", None)
    shutdown = getattr(raw_stream, "shutdown", None)
    if not callable(shutdown):
        return False
    try:
        shutdown()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("Shutting down Bedrock stream failed (%s)", type(exc).__name__)
        return False
    return True


def _close_in_background(close: Callable[[], Any]) -> None:
    """Close without waiting, for a stream whose read may still be blocked."""

    def run() -> None:
        try:
            close()
        except Exception as exc:  # best effort
            logger.debug("Closing Bedrock stream failed (%s)", type(exc).__name__)

    try:
        threading.Thread(target=run, name="bedrock-stream-close", daemon=True).start()
    except RuntimeError as exc:
        # No thread to spare (or the interpreter is shutting down): leave the
        # response to be closed when collected rather than mask the error the
        # caller is about to see.
        logger.debug("Closing Bedrock stream failed (%s)", type(exc).__name__)


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
        timeout_config: TimeoutConfig | None = None,
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
        self.timeout_config = timeout_config or TimeoutConfig()
        self._abilities = list(abilities or ["chat", "tool_calling", "vision"])
        self._client = client
        self._client_lock = threading.Lock()

    @staticmethod
    def _validate_endpoint(endpoint_url: str | None) -> None:
        validate_bedrock_endpoint(endpoint_url)

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

    @property
    def supports_forced_tool_choice(self) -> bool:
        """Whether this Bedrock model accepts ``toolChoice.any/tool``.

        Anthropic documents Opus/Sonnet 5.5 as auto-only, and AWS also names
        Fable 5.1 and Mythos 5.1. The shared matcher accepts bare model ids
        and the model segment of Bedrock ARNs. Opaque application inference
        profiles can opt in through the ``auto_tool_choice_only`` ability.
        """
        return supports_forced_tool_choice(self._abilities, self._model_name)

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
        previous_session = client._endpoint.http_session
        try:
            public_session = PublicBotocoreSession(
                timeout=self.timeout,
                verify=previous_session._verify,
                proxy_url=get_model_proxy_url(),
                _policy=ModelNetworkPolicy(),
                max_pool_connections=getattr(
                    previous_session, "_max_pool_connections", 10
                ),
                socket_options=getattr(previous_session, "_socket_options", None),
                client_cert=(
                    getattr(previous_session, "_cert_file", None),
                    getattr(previous_session, "_key_file", None),
                )
                if getattr(previous_session, "_key_file", None)
                else getattr(previous_session, "_cert_file", None),
                proxies_config=getattr(
                    getattr(previous_session, "_proxy_config", None), "settings", None
                ),
            )
        except BaseException:
            previous_session.close()
            raise
        try:
            previous_session.close()
        except BaseException:
            public_session.close()
            raise
        client._endpoint.http_session = public_session
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
            reasoning_blocks = (
                _reasoning_blocks_from_message(message) if role == "assistant" else []
            )
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
                _insert_reasoning_blocks(blocks, reasoning_blocks)
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
        system, converted_messages = self._convert_messages(messages)
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
        tool_choice = self.compatible_tool_choice(tool_choice)
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
        supported_content_keys = {"text", "toolUse", "reasoningContent"}
        unsupported_blocks = []
        for block in content:
            if not isinstance(block, Mapping):
                unsupported_blocks.append([type(block).__name__])
                continue
            keys = set(block)
            if len(keys) != 1 or not keys <= supported_content_keys:
                unsupported_blocks.append(sorted(str(key) for key in keys))
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
        provider_state = _reasoning_provider_state(content)
        finish_reason = _stop_reason(response.get("stopReason"))
        raw = {
            key: response[key]
            for key in ("output", "stopReason", "usage", "metrics")
            if key in response
        }
        if calls:
            result = {
                "type": "tool_call",
                "content": text,
                "tool_calls": calls,
                "finish_reason": finish_reason,
                "usage": usage,
                "raw": raw,
            }
            if provider_state:
                result[PROVIDER_STATE_METADATA_KEY] = provider_state
            return result
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
        result = {
            "type": "text",
            "content": text,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw": raw,
        }
        if provider_state:
            result[PROVIDER_STATE_METADATA_KEY] = provider_state
        return result

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
        read_in_flight = False
        current_text = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        reasoning_deltas: dict[int, dict[str, Any]] = {}
        reasoning_blocks: dict[int, dict[str, Any]] = {}
        finish_reason = ""

        def provider_raw(event: dict[str, Any] | None = None) -> dict[str, Any]:
            raw = dict(event or {})
            ordered_content: list[dict[str, Any]] = []
            if reasoning_blocks:
                ordered_content = [{} for _ in range(max(reasoning_blocks) + 1)]
                for index, block in reasoning_blocks.items():
                    ordered_content[index] = block
            provider_state = _reasoning_provider_state(ordered_content)
            if provider_state:
                raw[PROVIDER_STATE_METADATA_KEY] = provider_state
            return raw

        try:
            # Both limits are deadlines on the wait itself (#3059). Each event
            # is read in a worker thread, and botocore's per-socket-read timeout
            # is the only other bound. The first-token deadline also covers
            # opening the stream. Expiry while ``converse_stream`` is still
            # waiting leaves no response to shut down: that worker runs on
            # until the headers arrive or botocore's connect/read timeout
            # (``self.timeout``) fires, and a stream it opens after expiry is
            # dropped and closed only when collected.
            loop = asyncio.get_running_loop()
            first_token_timeout = self.timeout_config.first_token_timeout
            first_token_deadline = loop.time() + first_token_timeout
            first_token_message = (
                f"First token timeout: no chunk within {first_token_timeout}s"
            )
            interval_timeout = self.timeout_config.token_interval_timeout
            interval_message = (
                f"Token interval timeout: no chunk within {interval_timeout}s "
                "of the previous one"
            )

            response = await await_before_deadline(
                run_in_thread_with_telemetry(
                    "bedrock.converse_stream", client.converse_stream, **request
                ),
                deadline=first_token_deadline,
                timeout_message=first_token_message,
            )
            event_stream = response["stream"]
            iterator = iter(event_stream)
            first_event = True
            while True:
                # The interval deadline starts when we begin waiting, so time
                # the caller spends on the previous chunk is not counted as
                # provider silence.
                if first_event:
                    deadline = first_token_deadline
                    timeout_message = first_token_message
                else:
                    deadline = loop.time() + interval_timeout
                    timeout_message = interval_message
                read_in_flight = True
                present, event = await await_before_deadline(
                    run_in_thread_with_telemetry(
                        "bedrock.stream_next", self._next_event, iterator
                    ),
                    deadline=deadline,
                    timeout_message=timeout_message,
                )
                read_in_flight = False
                if not present:
                    break
                first_event = False
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
                    if not isinstance(delta, Mapping) or len(delta) != 1:
                        raise RuntimeError(
                            "Bedrock stream content delta must contain exactly one member"
                        )
                    if "text" in delta:
                        text_delta = str(delta.get("text") or "")
                        current_text += text_delta
                        yield StreamChunk(
                            type=ChunkType.TOKEN,
                            content=current_text,
                            delta=text_delta,
                            raw=provider_raw(event),
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
                            raw=provider_raw(event),
                        )
                    if "reasoningContent" in delta:
                        reasoning_delta = delta["reasoningContent"]
                        if not isinstance(reasoning_delta, Mapping):
                            raise RuntimeError(
                                "Bedrock stream reasoningContent delta must be an object"
                            )
                        unsupported_reasoning = set(reasoning_delta) - {
                            "text",
                            "signature",
                            "redactedContent",
                        }
                        if unsupported_reasoning:
                            raise RuntimeError(
                                "Bedrock stream returned unsupported reasoning delta: "
                                + ", ".join(sorted(unsupported_reasoning))
                            )
                        if len(reasoning_delta) != 1:
                            raise RuntimeError(
                                "Bedrock stream reasoningContent delta must contain "
                                "exactly one member"
                            )
                        accumulated = reasoning_deltas.setdefault(index, {})
                        if "text" in reasoning_delta:
                            value = reasoning_delta["text"]
                            if not isinstance(value, str):
                                raise RuntimeError(
                                    "Bedrock stream reasoning text must be text"
                                )
                            accumulated["text"] = accumulated.get("text", "") + value
                        elif "signature" in reasoning_delta:
                            value = reasoning_delta["signature"]
                            if not isinstance(value, str):
                                raise RuntimeError(
                                    "Bedrock stream reasoning signature must be text"
                                )
                            accumulated["signature"] = (
                                accumulated.get("signature", "") + value
                            )
                        else:
                            value = reasoning_delta["redactedContent"]
                            if not isinstance(value, bytes):
                                raise RuntimeError(
                                    "Bedrock stream redactedContent must be bytes"
                                )
                            accumulated["redactedContent"] = (
                                accumulated.get("redactedContent", b"") + value
                            )
                    unsupported_delta = set(delta) - {
                        "text",
                        "toolUse",
                        "reasoningContent",
                    }
                    if unsupported_delta:
                        raise RuntimeError(
                            "Bedrock stream returned unsupported content delta: "
                            + ", ".join(sorted(unsupported_delta))
                        )
                elif "contentBlockStop" in event:
                    index = int(event["contentBlockStop"].get("contentBlockIndex", 0))
                    if index in reasoning_deltas:
                        accumulated = reasoning_deltas.pop(index)
                        if "redactedContent" in accumulated:
                            if set(accumulated) != {"redactedContent"}:
                                raise RuntimeError(
                                    "Bedrock stream mixed redacted and text reasoning"
                                )
                            reasoning_content = {
                                "redactedContent": accumulated["redactedContent"]
                            }
                        else:
                            if "text" not in accumulated:
                                raise RuntimeError(
                                    "Bedrock stream reasoning block is missing text"
                                )
                            reasoning_text: dict[str, Any] = {
                                "text": accumulated["text"]
                            }
                            if "signature" in accumulated:
                                reasoning_text["signature"] = accumulated["signature"]
                            reasoning_content = {"reasoningText": reasoning_text}
                        reasoning_blocks[index] = {
                            "reasoningContent": _validate_reasoning_content(
                                reasoning_content
                            )
                        }
                    if index in tool_calls:
                        completed_call = copy.deepcopy(tool_calls[index])
                        # Argument fragments above are appended by the shared
                        # runtime. The stop event repeats the accumulated value,
                        # so expose it as a snapshot instead of a final delta.
                        completed_call["function"]["arguments_mode"] = "snapshot"
                        yield StreamChunk(
                            type=ChunkType.TOOL_CALL,
                            tool_calls=[completed_call],
                            raw=provider_raw(event),
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
                        raw=provider_raw(event),
                    )
                elif any(key.endswith("Exception") for key in event):
                    key = next(key for key in event if key.endswith("Exception"))
                    message = (event[key] or {}).get("message") or key
                    detail = (
                        f"Bedrock stream error [{key}]: "
                        f"{redact_sensitive_text(str(message))}"
                    )
                    if key in _RETRYABLE_STREAM_EVENTS:
                        error = LLMRetryableError(detail)
                        error.status_code = _RETRYABLE_STREAM_EVENTS[key]  # type: ignore[attr-defined]
                        raise error
                    raise RuntimeError(detail)
                elif "messageStart" not in event:
                    raise RuntimeError("Bedrock stream returned an unknown event")
            if reasoning_deltas:
                raise RuntimeError(
                    "Bedrock stream ended with incomplete reasoning blocks"
                )
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
                raw=provider_raw(),
            )
        except asyncio.CancelledError:
            raise
        except (BotoCoreError, ClientError) as exc:
            self._raise_provider_error(exc)
        finally:
            close = getattr(event_stream, "close", None)
            if callable(close):
                # Only a read still pending in a worker (a deadline,
                # cancellation, or an error out of the await) needs the socket
                # shut down first: neither cancelling its wait nor close()
                # interrupts it. Without a shutdown, close() would wait for
                # that read, so it is left to run in the background instead.
                if not read_in_flight:
                    await run_in_thread_with_telemetry("bedrock.stream_close", close)
                elif _shutdown_event_stream(event_stream):
                    try:
                        await run_in_thread_with_telemetry(
                            "bedrock.stream_close", close
                        )
                    except Exception as exc:
                        # The woken worker's urllib3 cleanup can race this
                        # close (release_conn() clears _connection while
                        # close() checks it unlocked), and a close error must
                        # not replace the error already in flight.
                        logger.debug(
                            "Closing Bedrock stream failed (%s)", type(exc).__name__
                        )
                else:
                    _close_in_background(close)

    async def close(self) -> None:
        client = self._client
        self._client = None
        close = getattr(client, "close", None)
        if callable(close):
            await run_in_thread_with_telemetry("bedrock.close_client", close)
