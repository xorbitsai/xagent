import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from ..types import StreamChunk
from .openai import OpenAILLM

logger = logging.getLogger(__name__)

_DISABLE_THINKING = {"type": "disabled", "enable": False}


class DashScopeLLM(OpenAILLM):
    """DashScope/OpenAI-compatible chat client with Qwen request policy.

    Qwen thinking mode rejects strict tool selection (``required`` or a
    function object). Xagent's agent patterns intentionally use strict tool
    choice for reliable tool execution, so this adapter disables thinking for
    those calls before falling back to ``auto`` only if the provider still
    rejects the request.
    """

    def __init__(
        self,
        model_name: str = "qwen-plus",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[Any] = None,
    ):
        if base_url is None:
            raise ValueError("DashScopeLLM requires a resolved base_url")

        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY"),
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            timeout=timeout,
            abilities=abilities,
            timeout_config=timeout_config,
        )

    @staticmethod
    def _is_strict_tool_choice(
        tool_choice: Optional[Union[str, Dict[str, Any]]],
    ) -> bool:
        return tool_choice == "required" or isinstance(tool_choice, dict)

    @staticmethod
    def _is_thinking_tool_choice_error(exc: Exception) -> bool:
        exc_msg = str(exc).lower()
        return "thinking" in exc_msg and "tool_choice" in exc_msg

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
        if thinking is None:
            return updated_extra_body

        if thinking.get("type") == "enabled" or thinking.get("enable", False):
            updated_extra_body["enable_thinking"] = True
        elif thinking.get("type") == "disabled" or not thinking.get("enable", False):
            updated_extra_body["enable_thinking"] = False

        return updated_extra_body

    @classmethod
    def _requires_thinking_disabled(
        cls,
        *,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
    ) -> bool:
        return bool(tools) and cls._is_strict_tool_choice(tool_choice)

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
        strict_tool_choice = self._requires_thinking_disabled(
            tools=tools,
            tool_choice=tool_choice,
        )
        call_thinking = _DISABLE_THINKING if strict_tool_choice else thinking

        try:
            return await super().chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=call_thinking,
                output_config=output_config,
                **kwargs,
            )
        except RuntimeError as exc:
            if not strict_tool_choice or not self._is_thinking_tool_choice_error(exc):
                raise

            logger.warning(
                "DashScope rejected strict tool_choice while thinking; retrying with tool_choice=auto"
            )
            return await super().chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice="auto",
                response_format=response_format,
                thinking=_DISABLE_THINKING,
                output_config=output_config,
                **kwargs,
            )

    async def stream_chat(
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
    ) -> AsyncIterator[StreamChunk]:
        strict_tool_choice = self._requires_thinking_disabled(
            tools=tools,
            tool_choice=tool_choice,
        )
        call_thinking = _DISABLE_THINKING if strict_tool_choice else thinking
        yielded = False

        try:
            async for chunk in super().stream_chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                thinking=call_thinking,
                output_config=output_config,
                **kwargs,
            ):
                yielded = True
                yield chunk
        except RuntimeError as exc:
            if (
                yielded
                or not strict_tool_choice
                or not self._is_thinking_tool_choice_error(exc)
            ):
                raise

            logger.warning(
                "DashScope rejected strict streaming tool_choice while thinking; retrying with tool_choice=auto"
            )
            async for chunk in super().stream_chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice="auto",
                response_format=response_format,
                thinking=_DISABLE_THINKING,
                output_config=output_config,
                **kwargs,
            ):
                yield chunk
