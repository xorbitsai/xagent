import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .....config import get_openrouter_official_providers_only
from ..exceptions import LLMRetryableError, LLMToolProtocolError
from ..timeout_config import TimeoutConfig
from ..tool_protocol import TOOL_PROTOCOL_ERROR_KEY, get_tool_protocol_error
from ..types import StreamChunk
from .deepseek_tool_protocol import (
    adapt_deepseek_stream,
    normalize_deepseek_response,
)
from .openai import OpenAILLM

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEEPSEEK_FUNCTION_PREFIX_ERROR = "function call should not be used with prefix"

_OPENROUTER_OFFICIAL_PROVIDERS_BY_AUTHOR: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "deepseek": ("deepseek",),
    "google": ("google-ai-studio", "google-vertex"),
    "minimax": ("minimax",),
    "openai": ("openai",),
    "z-ai": ("z-ai",),
}


def _openrouter_model_author(model_name: str) -> str:
    model_slug = model_name.strip().split(":", 1)[0]
    parts = [part for part in model_slug.split("/") if part]
    if len(parts) >= 3 and parts[0].lower() == "openrouter":
        return parts[1].lower()
    if len(parts) >= 2:
        return parts[0].lower()
    return ""


def _strip_assistant_tool_call_prefixes(
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], bool]:
    """Remove assistant prefixes that DeepSeek cannot combine with tools."""
    sanitized: List[Dict[str, Any]] = []
    changed = False
    for message in messages:
        sanitized_message = dict(message)
        if (
            sanitized_message.get("role") == "assistant"
            and sanitized_message.get("tool_calls")
            and str(sanitized_message.get("content") or "").strip()
        ):
            sanitized_message["content"] = ""
            changed = True
        sanitized.append(sanitized_message)

    # A non-blocking ``send_message`` control call records its result and may
    # leave a standalone assistant progress message at the end of older
    # checkpoints. OpenRouter treats that trailing assistant turn as prefix
    # completion, which DeepSeek rejects when tools are also present. The tool
    # result already contains the same progress text, so dropping only trailing
    # assistant-only turns preserves the tool chain and avoids replaying stale
    # progress as a generation prefix.
    has_completed_tool_chain = any(
        message.get("role") == "assistant" and message.get("tool_calls")
        for message in sanitized
    ) and any(message.get("role") == "tool" for message in sanitized)
    if has_completed_tool_chain:
        while (
            sanitized
            and sanitized[-1].get("role") == "assistant"
            and not sanitized[-1].get("tool_calls")
        ):
            sanitized.pop()
            changed = True
    return sanitized, changed


def _force_single_required_deepseek_tool(
    tools: Optional[List[Any]],
    tool_choice: Optional[str | Dict[str, Any]],
) -> Optional[str | Dict[str, Any]]:
    """Turn DeepSeek's ambiguous single-tool requirement into a named choice."""
    if tool_choice != "required" or not tools or len(tools) != 1:
        return tool_choice
    tool = tools[0]
    if not isinstance(tool, dict):
        return tool_choice
    function = tool.get("function")
    if not isinstance(function, dict):
        return tool_choice
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return tool_choice
    return {
        "type": "function",
        "function": {"name": name},
    }


def _deepseek_tool_protocol_retry_error(response: Any) -> LLMRetryableError | None:
    error = get_tool_protocol_error(response)
    if error is None:
        return None
    code = str(error.get("code") or "invalid_tool_protocol")
    # Replaying the same narrowed schema cannot make the requested tool
    # available. Surface this response to the agent pattern so it can restore
    # the appropriate tool set and re-decide with explicit feedback.
    if code == "unavailable_tool_call":
        return None
    message = str(error.get("message") or "DeepSeek returned an invalid tool call.")
    return LLMToolProtocolError(
        provider="deepseek",
        code=code,
        message=message,
        details=error.get("details")
        if isinstance(error.get("details"), dict)
        else None,
    )


class OpenRouterLLM(OpenAILLM):
    """OpenRouter client using the OpenAI SDK with OpenRouter-specific options."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url or OPENROUTER_BASE_URL,
            api_key=api_key,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            timeout=timeout,
            abilities=abilities,
            timeout_config=timeout_config,
        )

    @property
    def _uses_deepseek_tool_protocol(self) -> bool:
        return _openrouter_model_author(self._model_name) == "deepseek"

    def _is_official_openrouter_client(self) -> bool:
        return self.base_url.rstrip("/") == OPENROUTER_BASE_URL

    def _deepseek_function_prefix_retry_messages(
        self,
        exc: Exception,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]] | None:
        if _openrouter_model_author(self._model_name) != "deepseek":
            return None
        if _DEEPSEEK_FUNCTION_PREFIX_ERROR not in str(exc).lower():
            return None
        sanitized, changed = _strip_assistant_tool_call_prefixes(messages)
        return sanitized if changed else None

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if self._uses_deepseek_tool_protocol:
            tool_choice = _force_single_required_deepseek_tool(tools, tool_choice)
        try:
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
        except RuntimeError as exc:
            sanitized_messages = self._deepseek_function_prefix_retry_messages(
                exc, messages
            )
            if sanitized_messages is None:
                raise

            logger.info(
                "OpenRouter DeepSeek rejected function-call history with an "
                "assistant prefix; retrying once without tool-call prefixes"
            )
            response = await super().chat(
                messages=sanitized_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=thinking,
                output_config=output_config,
                **kwargs,
            )

        if not self._uses_deepseek_tool_protocol:
            return response
        response = normalize_deepseek_response(response, tools=tools)
        retry_error = _deepseek_tool_protocol_retry_error(response)
        if retry_error is not None:
            raise retry_error
        return response

    async def _stream_chat_with_prefix_retry(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        has_yielded = False
        try:
            async for chunk in super().stream_chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=thinking,
                output_config=output_config,
                **kwargs,
            ):
                has_yielded = True
                yield chunk
            return
        except RuntimeError as exc:
            sanitized_messages = self._deepseek_function_prefix_retry_messages(
                exc, messages
            )
            if has_yielded or sanitized_messages is None:
                raise

        logger.info(
            "OpenRouter DeepSeek rejected streaming function-call history with an "
            "assistant prefix; retrying once without tool-call prefixes"
        )
        async for chunk in super().stream_chat(
            messages=sanitized_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        ):
            yield chunk

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        if self._uses_deepseek_tool_protocol:
            tool_choice = _force_single_required_deepseek_tool(tools, tool_choice)
        stream = self._stream_chat_with_prefix_retry(
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
        if not self._uses_deepseek_tool_protocol:
            async for chunk in stream:
                yield chunk
            return
        adapted_stream = adapt_deepseek_stream(stream, tools=tools)

        # A named tool choice cannot validly finish as assistant text. Buffer
        # this narrow stream until DeepSeek's tool protocol has been validated,
        # so a malformed serialized call raises before any chunk escapes and the
        # shared LLM retry wrapper can safely replay the request.
        if isinstance(tool_choice, dict):
            buffered_chunks: list[StreamChunk] = []
            async for chunk in adapted_stream:
                if chunk.is_protocol_error():
                    retry_error = _deepseek_tool_protocol_retry_error(
                        {TOOL_PROTOCOL_ERROR_KEY: chunk.protocol_error}
                    )
                    if retry_error is not None:
                        raise retry_error
                buffered_chunks.append(chunk)
            for chunk in buffered_chunks:
                yield chunk
            return

        async for chunk in adapted_stream:
            if chunk.is_protocol_error():
                retry_error = _deepseek_tool_protocol_retry_error(
                    {TOOL_PROTOCOL_ERROR_KEY: chunk.protocol_error}
                )
                if retry_error is not None:
                    raise retry_error
            yield chunk

    def _prepare_extra_body(self, extra_body: Dict[str, Any]) -> Dict[str, Any]:
        if (
            not get_openrouter_official_providers_only()
            or not self._is_official_openrouter_client()
            or "provider" in extra_body
        ):
            return extra_body

        author = _openrouter_model_author(self._model_name)
        official_providers = _OPENROUTER_OFFICIAL_PROVIDERS_BY_AUTHOR.get(author)
        if not official_providers:
            return extra_body

        return {
            **extra_body,
            "provider": {
                "only": list(official_providers),
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }

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
        _ = tools, output_config
        updated_extra_body = dict(extra_body)
        should_disable = False

        if thinking is not None:
            should_enable = thinking.get("type") == "enabled" or thinking.get(
                "enable", False
            )
            should_disable = not should_enable and (
                thinking.get("type") == "disabled" or not thinking.get("enable", False)
            )
        elif is_streaming and response_format:
            should_disable = self.supports_thinking_mode
            should_enable = False
        else:
            should_enable = False

        if should_disable:
            updated_extra_body["reasoning"] = {"enabled": False}
            updated_extra_body["thinking"] = {"type": "disabled"}
        elif should_enable:
            updated_extra_body["reasoning"] = {"enabled": True}
            updated_extra_body["thinking"] = {"type": "enabled"}

        updated_extra_body.pop("enable_thinking", None)
        return updated_extra_body
