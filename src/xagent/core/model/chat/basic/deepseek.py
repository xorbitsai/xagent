import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from ...providers import is_placeholder_api_key
from .base import StreamChunk
from .deepseek_tool_protocol import (
    adapt_deepseek_stream,
    normalize_deepseek_response,
)
from .openai import PROVIDER_STATE_METADATA_KEY, OpenAICompatibleLLM

logger = logging.getLogger(__name__)

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_PROVIDER_STATE_NAMESPACE = "deepseek"
DEEPSEEK_REASONING_CONTENT_STATE_KEY = "reasoning_content"
DEEPSEEK_SUPPORTED_MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
)


def resolve_deepseek_api_key(api_key: Optional[str] = None) -> str:
    """Resolve DeepSeek API key with OpenAI-compatible fallback."""

    resolved_api_key = api_key.strip() if isinstance(api_key, str) else api_key
    if is_placeholder_api_key(resolved_api_key):
        resolved_api_key = None

    if resolved_api_key is None:
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        deepseek_api_key = (
            deepseek_api_key.strip()
            if isinstance(deepseek_api_key, str)
            else deepseek_api_key
        )
        if not is_placeholder_api_key(deepseek_api_key):
            resolved_api_key = deepseek_api_key

    if resolved_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_key = (
            openai_api_key.strip()
            if isinstance(openai_api_key, str)
            else openai_api_key
        )
        if not is_placeholder_api_key(openai_api_key):
            resolved_api_key = openai_api_key

    return resolved_api_key or ""


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek v4 client using the OpenAI SDK with DeepSeek-specific options."""

    def __init__(
        self,
        model_name: str = "deepseek-v4-flash",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[Any] = None,
    ):
        self._validate_model_name(model_name)
        resolved_abilities = abilities or ["chat", "tool_calling", "thinking_mode"]
        resolved_api_key = resolve_deepseek_api_key(api_key)

        super().__init__(
            model_name=model_name,
            base_url=base_url or DEEPSEEK_DEFAULT_BASE_URL,
            api_key=resolved_api_key,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            timeout=timeout,
            abilities=resolved_abilities,
            timeout_config=timeout_config,
        )

    @staticmethod
    def _validate_model_name(model_name: str) -> None:
        if model_name not in DEEPSEEK_SUPPORTED_MODELS:
            supported = ", ".join(DEEPSEEK_SUPPORTED_MODELS)
            raise ValueError(
                f"Unsupported DeepSeek model '{model_name}'. Supported models: {supported}"
            )

    @property
    def supports_json_schema_response_format(self) -> bool:
        """DeepSeek supports JSON object mode, not OpenAI json_schema mode."""
        return False

    @property
    def supports_json_object_response_format(self) -> bool:
        """DeepSeek supports response_format={"type": "json_object"}."""
        return True

    def _prepare_provider_reasoning_extra_body(
        self,
        *,
        extra_body: Dict[str, Any],
        thinking: Optional[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
        output_config: Optional[Dict[str, Any]],
        is_streaming: bool,
    ) -> Dict[str, Any]:
        _ = tools, response_format, output_config, is_streaming
        updated_extra_body = dict(extra_body)

        if thinking is not None:
            if thinking.get("type") == "enabled" or thinking.get("enable", False):
                updated_extra_body["thinking"] = {"type": "enabled"}
            else:
                updated_extra_body["thinking"] = {"type": "disabled"}
        else:
            updated_extra_body["thinking"] = {"type": "disabled"}

        updated_extra_body.pop("enable_thinking", None)
        return updated_extra_body

    def _prepare_deepseek_kwargs(
        self,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        updated_kwargs = dict(kwargs)

        if "reasoning_effort" not in updated_kwargs:
            reasoning_effort = os.getenv("DEEPSEEK_REASONING_EFFORT")
            if reasoning_effort:
                updated_kwargs["reasoning_effort"] = reasoning_effort

        return updated_kwargs

    def _prepare_messages_for_request(
        self,
        messages: List[Dict[str, Any]],
        *,
        thinking: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Preserve DeepSeek thinking metadata on assistant tool-call history.

        DeepSeek V4 requires assistant messages in a tool-call chain to replay
        the exact ``reasoning_content`` returned by the provider. An explicit
        empty string is semantically different from a missing field, so this
        must use key presence rather than truthiness. If older context lacks
        captured provider state, use an empty string fallback to keep assistant
        tool-call history structurally valid for later DeepSeek requests.
        """
        prepared: List[Dict[str, Any]] = []
        for message in messages:
            prepared_message = dict(message)
            provider_state = prepared_message.get(PROVIDER_STATE_METADATA_KEY)
            if isinstance(provider_state, dict):
                deepseek_metadata = provider_state.get(
                    DEEPSEEK_PROVIDER_STATE_NAMESPACE
                )
                if (
                    isinstance(deepseek_metadata, dict)
                    and DEEPSEEK_REASONING_CONTENT_STATE_KEY in deepseek_metadata
                ):
                    prepared_message["reasoning_content"] = deepseek_metadata[
                        DEEPSEEK_REASONING_CONTENT_STATE_KEY
                    ]
            if (
                prepared_message.get("role") == "assistant"
                and prepared_message.get("tool_calls")
                and "reasoning_content" not in prepared_message
            ):
                prepared_message["reasoning_content"] = ""
            prepared.append(prepared_message)
        return prepared

    def _response_provider_state(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "reasoning_content" not in result:
            return {}
        return {
            DEEPSEEK_PROVIDER_STATE_NAMESPACE: {
                DEEPSEEK_REASONING_CONTENT_STATE_KEY: result["reasoning_content"],
            },
        }

    def _attach_reasoning_content_to_raw(
        self,
        raw_payload: Any,
        reasoning_content: str,
        *,
        has_reasoning_content: bool = False,
    ) -> Any:
        raw_payload = super()._attach_reasoning_content_to_raw(
            raw_payload,
            reasoning_content,
            has_reasoning_content=has_reasoning_content,
        )
        if has_reasoning_content and isinstance(raw_payload, dict):
            raw_payload[PROVIDER_STATE_METADATA_KEY] = {
                DEEPSEEK_PROVIDER_STATE_NAMESPACE: {
                    DEEPSEEK_REASONING_CONTENT_STATE_KEY: reasoning_content,
                },
            }
        return raw_payload

    def _normalize_response_format(
        self,
        response_format: Optional[Dict[str, Any]],
        output_config: Optional[Dict[str, Any]],
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if response_format and response_format.get("type") == "json_schema":
            logger.warning(
                "DeepSeek does not support json_schema response_format; using json_object instead."
            )
            return {"type": "json_object"}, output_config

        format_config = (output_config or {}).get("format") or {}
        if not response_format and format_config.get("type") == "json_schema":
            logger.warning(
                "DeepSeek does not support json_schema output_config; using json_object instead."
            )
            return {"type": "json_object"}, None

        return response_format, output_config

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        response_format, output_config = self._normalize_response_format(
            response_format=response_format,
            output_config=output_config,
        )
        kwargs = self._prepare_deepseek_kwargs(kwargs=kwargs)
        response = await super().chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )
        return normalize_deepseek_response(response, tools=tools)

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        response_format, output_config = self._normalize_response_format(
            response_format=response_format,
            output_config=output_config,
        )
        kwargs = self._prepare_deepseek_kwargs(kwargs=kwargs)
        stream = super().stream_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )
        async for chunk in adapt_deepseek_stream(stream, tools=tools):
            yield chunk

    @staticmethod
    async def list_available_models(
        api_key: str, base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        _ = api_key, base_url
        return [
            {
                "id": "deepseek-v4-flash",
                "created": 0,
                "owned_by": "deepseek",
                "abilities": ["chat", "tool_calling", "thinking_mode"],
            },
            {
                "id": "deepseek-v4-pro",
                "created": 0,
                "owned_by": "deepseek",
                "abilities": ["chat", "tool_calling", "thinking_mode"],
            },
        ]
