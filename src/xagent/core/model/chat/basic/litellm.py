import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from ..exceptions import LLMRetryableError, LLMTimeoutError
from ..timeout_config import TimeoutConfig
from ..token_context import add_token_usage
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)


class LiteLLM(BaseLLM):
    """
    LiteLLM client providing access to 100+ LLM providers through a unified interface.
    Uses provider-prefixed model names (e.g. openai/gpt-4o, anthropic/claude-sonnet-4-6).
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        self._model_name = model_name
        self._api_key = api_key
        self._api_base = api_base
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        self.timeout_config = timeout_config or TimeoutConfig()

        if abilities:
            self._abilities = abilities
        else:
            self._abilities = ["chat", "tool_calling"]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def abilities(self) -> List[str]:
        return self._abilities

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Perform a chat completion via LiteLLM."""
        import litellm

        completion_params: Dict[str, Any] = {
            "model": self._model_name,
            "messages": self._sanitize_unicode_content(messages),
            "drop_params": True,
            "timeout": self.timeout,
            **kwargs,
        }

        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens
        elif self.default_max_tokens is not None:
            completion_params["max_tokens"] = self.default_max_tokens

        if temperature is not None:
            completion_params["temperature"] = temperature
        elif self.default_temperature is not None:
            completion_params["temperature"] = self.default_temperature

        if tools:
            completion_params["tools"] = tools
        if tool_choice:
            completion_params["tool_choice"] = tool_choice
        if response_format:
            completion_params["response_format"] = response_format

        if self._api_key:
            completion_params["api_key"] = self._api_key
        if self._api_base:
            completion_params["api_base"] = self._api_base

        try:
            response = await litellm.acompletion(**completion_params)
        except litellm.Timeout as e:
            raise LLMTimeoutError(str(e)) from e
        except (
            litellm.RateLimitError,
            litellm.APIConnectionError,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
        ) as e:
            raise LLMRetryableError(str(e)) from e

        if not response.choices:
            raise LLMRetryableError("LiteLLM returned an empty response (no choices).")
        choice = response.choices[0]
        message = choice.message

        if hasattr(response, "usage") and response.usage:
            add_token_usage(
                input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
            )

        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            return {
                "type": "tool_call",
                "tool_calls": tool_calls,
                "raw": response.model_dump()
                if hasattr(response, "model_dump")
                else str(response),
            }

        return message.content or ""

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion via LiteLLM."""
        import litellm

        completion_params: Dict[str, Any] = {
            "model": self._model_name,
            "messages": self._sanitize_unicode_content(messages),
            "stream": True,
            "drop_params": True,
            "timeout": self.timeout,
            **kwargs,
        }

        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens
        elif self.default_max_tokens is not None:
            completion_params["max_tokens"] = self.default_max_tokens

        if temperature is not None:
            completion_params["temperature"] = temperature
        elif self.default_temperature is not None:
            completion_params["temperature"] = self.default_temperature

        if tools:
            completion_params["tools"] = tools
        if tool_choice:
            completion_params["tool_choice"] = tool_choice
        if response_format:
            completion_params["response_format"] = response_format

        if self._api_key:
            completion_params["api_key"] = self._api_key
        if self._api_base:
            completion_params["api_base"] = self._api_base

        try:
            response = await litellm.acompletion(**completion_params)
        except litellm.Timeout as e:
            raise LLMTimeoutError(str(e)) from e
        except (
            litellm.RateLimitError,
            litellm.APIConnectionError,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
        ) as e:
            raise LLMRetryableError(str(e)) from e

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            content = delta.content if hasattr(delta, "content") else None
            if content:
                yield StreamChunk(type=ChunkType.TOKEN, content=content, delta=content)

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                tool_calls = []
                for tc in delta.tool_calls:
                    tool_calls.append(
                        {
                            "index": tc.index
                            if hasattr(tc, "index") and tc.index is not None
                            else 0,
                            "id": tc.id if hasattr(tc, "id") else None,
                            "type": "function",
                            "function": {
                                "name": tc.function.name
                                if hasattr(tc, "function")
                                and hasattr(tc.function, "name")
                                else None,
                                "arguments": tc.function.arguments
                                if hasattr(tc, "function")
                                and hasattr(tc.function, "arguments")
                                else "",
                            },
                        }
                    )
                yield StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=tool_calls,
                    raw=chunk.model_dump()
                    if hasattr(chunk, "model_dump")
                    else str(chunk),
                )
