import json
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import openai
from openai import AsyncOpenAI

from ....utils.security import redact_sensitive_text
from ..exceptions import LLMEmptyContentError, LLMRetryableError, LLMTimeoutError
from ..timeout_config import TimeoutConfig
from ..token_context import add_token_usage, extract_cached_input_tokens
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)

PROVIDER_STATE_METADATA_KEY = "_xagent_provider_state"


def _truncate_error_detail(value: Any, limit: int = 4000) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _openai_error_body(error: BaseException) -> Any:
    body = getattr(error, "body", None)
    if body is not None:
        return body
    response = getattr(error, "response", None)
    if response is None:
        return None
    try:
        return response.json()
    except Exception:
        return None


def _openai_error_details(error: BaseException) -> list[str]:
    details: list[str] = []
    body = _openai_error_body(error)
    if isinstance(body, dict):
        error_payload = body.get("error")
        if isinstance(error_payload, dict):
            metadata = error_payload.get("metadata")
            if isinstance(metadata, dict):
                provider_name = metadata.get("provider_name")
                if provider_name:
                    details.append(f"provider_name={provider_name}")
                raw = metadata.get("raw")
                if raw:
                    details.append("provider_raw=" + _truncate_error_detail(raw))
                previous_errors = metadata.get("previous_errors")
                if previous_errors:
                    details.append(
                        "previous_errors=" + _truncate_error_detail(previous_errors)
                    )
            elif metadata is not None:
                details.append("metadata=" + _truncate_error_detail(metadata))
    return details


def _format_openai_error(prefix: str, error: BaseException) -> str:
    message = str(getattr(error, "message", None) or error)
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        formatted = f"{prefix} ({status_code}): {message}"
    else:
        formatted = f"{prefix}: {message}"

    details = _openai_error_details(error)
    if details:
        formatted = f"{formatted} | " + " | ".join(details)
    return formatted


def _message_reasoning_content(message: Any) -> tuple[bool, Any]:
    """Return whether a provider explicitly included reasoning content."""
    if isinstance(message, dict):
        if "reasoning_content" not in message:
            return False, None
        value = message.get("reasoning_content")
        return value is not None, value

    model_fields_set = getattr(message, "model_fields_set", None)
    if isinstance(model_fields_set, set) and "reasoning_content" in model_fields_set:
        value = getattr(message, "reasoning_content", None)
        return value is not None, value

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict) and "reasoning_content" in model_extra:
        value = model_extra.get("reasoning_content")
        return value is not None, value

    message_attrs = getattr(message, "__dict__", {})
    if not isinstance(message_attrs, dict) or "reasoning_content" not in message_attrs:
        return False, None

    value = message_attrs["reasoning_content"]
    return value is not None, value


def _is_retryable_stream_transport_error(error: BaseException) -> bool:
    retryable_messages = (
        "peer closed connection",
        "incomplete chunked read",
        "remoteprotocolerror",
        "server disconnected",
        "connection reset",
        "connection aborted",
        "connection lost",
    )
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                LLMRetryableError,
                openai.APIConnectionError,
                openai.APITimeoutError,
            ),
        ):
            return True
        detail = f"{type(current).__module__}.{type(current).__name__}: {current}"
        if any(message in detail.lower() for message in retryable_messages):
            return True
        current = current.__cause__ or current.__context__
    return False


class OpenAICompatibleLLM(BaseLLM):
    """
    Internal OpenAI-compatible chat client using the official OpenAI SDK.

    This base owns transport, streaming assembly, usage accounting, tool-call
    parsing, and retry/error handling. Provider classes layer policy on top:
    environment defaults, structured-output translation, thinking parameters,
    vision support, and model listing.
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str],
        api_key: Optional[str],
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        self._model_name = model_name
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        self.timeout_config = timeout_config or TimeoutConfig()

        # Use explicitly configured abilities
        if abilities:
            self._abilities = abilities
        else:
            self._abilities = ["chat", "tool_calling"]

        # Initialize the async OpenAI client
        self._client: Optional[AsyncOpenAI] = None

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this OpenAI LLM implementation."""
        return self._abilities

    def _ensure_client(self) -> None:
        """Ensure the OpenAI client is initialized."""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url
                if self.base_url != "https://api.openai.com/v1"
                else None,
                api_key=self.api_key,
                timeout=self.timeout,
            )

    def _prepare_extra_body(self, extra_body: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for OpenAI-compatible subclasses to customize extra_body."""
        return extra_body

    def _prepare_messages_for_request(
        self,
        messages: List[Dict[str, Any]],
        *,
        thinking: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return provider-ready messages before shared sanitization."""
        _ = thinking
        return messages

    def _response_provider_state(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Return opaque provider-owned message state for future LLM requests."""
        _ = result
        return {}

    def _strip_internal_message_keys(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove Xagent-only message metadata before sending provider calls."""
        sanitized: List[Dict[str, Any]] = []
        for message in messages:
            sanitized.append(
                {
                    key: value
                    for key, value in message.items()
                    if not key.startswith("_xagent_")
                }
            )
        return sanitized

    def _build_request_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        thinking: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prepared = self._prepare_messages_for_request(messages, thinking=thinking)
        sanitized_messages: List[Dict[str, Any]] = self._sanitize_unicode_content(
            self._strip_internal_message_keys(prepared)
        )
        return sanitized_messages

    def _apply_output_config(
        self,
        completion_params: Dict[str, Any],
        output_config: Optional[Dict[str, Any]],
    ) -> None:
        """Apply provider-specific structured-output request policy."""
        if output_config is None:
            return

        completion_params["output_config"] = output_config

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
        """Hook for subclasses to add provider-specific reasoning payloads."""
        _ = thinking, tools, response_format, output_config, is_streaming
        return dict(extra_body)

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
        """
        Perform a chat completion or trigger tool call.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            response_format: Response format specification (e.g., {"type": "json_object"})
            thinking: Provider-specific reasoning configuration for subclasses.
            output_config: Output configuration for structured outputs (e.g., {"format": {"type": "json_schema", "schema": {...}}})
            **kwargs: Additional parameters to pass to the OpenAI API

        Returns:
            - If normal text reply: return string
            - If tool call triggered: return dict with type "tool_call" and tool_calls list

        Raises:
            RuntimeError: If the API call fails
        """
        self._ensure_client()
        assert self._client is not None

        extra_body = self._prepare_extra_body(dict(kwargs.pop("extra_body", {}) or {}))

        # Prepare the completion parameters
        completion_params = {
            "model": self._model_name,
            "messages": self._build_request_messages(messages, thinking=thinking),
            **kwargs,
        }

        # Only add max_tokens if explicitly provided
        # Don't set default values - let API use its own defaults
        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens

        if temperature is not None:
            completion_params["temperature"] = temperature
        elif self.default_temperature is not None:
            completion_params["temperature"] = self.default_temperature

        # Add optional parameters
        if tools:
            completion_params["tools"] = tools
            if tool_choice:
                completion_params["tool_choice"] = tool_choice
        elif tool_choice:
            completion_params["tool_choice"] = tool_choice
        if response_format:
            completion_params["response_format"] = response_format

        self._apply_output_config(completion_params, output_config)

        extra_body = self._prepare_provider_reasoning_extra_body(
            extra_body=extra_body,
            thinking=thinking,
            tools=tools,
            response_format=response_format,
            output_config=output_config,
            is_streaming=False,
        )

        # Helper function to process response
        async def _make_api_call() -> Any:
            """Make the API call with current completion_params"""
            assert self._client is not None
            if extra_body:
                return await self._client.chat.completions.create(
                    extra_body=extra_body, **completion_params
                )
            else:
                return await self._client.chat.completions.create(**completion_params)

        # Helper function to process response
        def _process_response(resp: Any) -> Dict[str, Any]:
            """Process the API response and return the result"""
            # Validate response
            if not hasattr(resp, "choices") or not resp.choices:
                raise RuntimeError(
                    f"Invalid API response: no choices in response. Response: {resp}"
                )

            # Extract the choice
            choice = resp.choices[0]
            message = choice.message

            # Record token usage to context
            if hasattr(resp, "usage") and resp.usage:
                add_token_usage(
                    input_tokens=resp.usage.prompt_tokens,
                    output_tokens=resp.usage.completion_tokens,
                    model=self._model_name,
                    model_id=self.model_id,
                    cached_input_tokens=extract_cached_input_tokens(resp.usage),
                    call_type="chat",
                )

            # Check for tool calls
            if message.tool_calls:
                # Convert OpenAI tool calls to our format
                tool_calls = []
                for tool_call in message.tool_calls:
                    # Only handle function tool calls, not custom tool calls
                    if hasattr(tool_call, "function"):
                        func = tool_call.function
                        args = func.arguments if func.arguments else ""

                        # Validate arguments are not empty
                        if not args or args.strip() == "":
                            raise RuntimeError(
                                f"Tool '{func.name}' has empty arguments. "
                                f"This is a bug in the LLM provider's tool calling implementation. "
                                f"Model: {self._model_name}"
                            )

                        tool_calls.append(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": func.name,
                                    "arguments": args,
                                },
                            }
                        )

                result = {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                    "raw": resp.model_dump(),
                }
                has_reasoning_content, reasoning_content = _message_reasoning_content(
                    message
                )
                if has_reasoning_content:
                    result["reasoning_content"] = reasoning_content
                    result["reasoning"] = reasoning_content
                provider_state = self._response_provider_state(result)
                if provider_state:
                    result[PROVIDER_STATE_METADATA_KEY] = provider_state
                return result

            # Handle text content
            content = message.content
            has_reasoning_content, reasoning_content = _message_reasoning_content(
                message
            )
            finish_reason = getattr(choice, "finish_reason", None)

            # Handle None or empty content when no tool calls
            if not content or not content.strip():
                # Reasoning models (e.g. qwen3-thinking, deepseek-r1, served
                # via OpenAI-compatible endpoints like Xinference) can return
                # ``content=""`` while ``reasoning_content`` carries the
                # partial answer when the generation is truncated by
                # ``max_tokens`` (``finish_reason="length"``) before the
                # final answer is produced. Surface the reasoning text as
                # content so callers (notably the model connection test) do
                # not treat a truncated-but-otherwise-healthy response as
                # invalid. Mirror the ``content`` whitespace check so a
                # reasoning trace that is purely whitespace still falls
                # through to the empty-response error.
                #
                # Gate the fallback strictly on ``finish_reason == "length"``:
                # any other terminal reason (``"stop"``, ``"content_filter"``,
                # ``None`` …) means the model claims to be done but produced
                # no final answer, which is a real failure that callers
                # must see -- promoting the reasoning trace would silently
                # hide the bug.
                if (
                    finish_reason == "length"
                    and reasoning_content
                    and reasoning_content.strip()
                ):
                    return {
                        "type": "text",
                        "content": reasoning_content,
                        "reasoning_content": reasoning_content,
                        "reasoning": reasoning_content,
                        "raw": resp.model_dump(),
                    }
                # If there are no tool calls and no content, this is an error
                raise LLMEmptyContentError(
                    f"LLM returned {'empty' if content == '' else 'None'} content and no tool calls"
                )

            result = {
                "type": "text",
                "content": content,
                "raw": resp.model_dump(),
            }
            if has_reasoning_content:
                result["reasoning_content"] = reasoning_content
                result["reasoning"] = reasoning_content
            return result

        try:
            # Make the API call
            response = await _make_api_call()
            result = _process_response(response)

            # Provider reasoning can corrupt structured JSON on some compatible
            # endpoints. Subclasses can disable provider reasoning for a retry.
            if (
                response_format
                and "thinking_mode" in self.abilities
                and result.get("type") == "text"
                and hasattr(response, "choices")
                and response.choices
            ):
                message = response.choices[0].message
                # Check if response has reasoning_content (indicates thinking was active)
                has_reasoning_content, _ = _message_reasoning_content(message)
                if has_reasoning_content:
                    content = result.get("content", "")
                    # Try to parse as JSON
                    try:
                        json.loads(content)
                    except (json.JSONDecodeError, ValueError):
                        # Content is not valid JSON, retry with thinking disabled
                        logger.warning(
                            "Model returned non-JSON content with response_format while thinking was enabled. "
                            "Retrying with thinking disabled."
                        )
                        extra_body = self._prepare_provider_reasoning_extra_body(
                            extra_body=extra_body,
                            thinking={"type": "disabled", "enable": False},
                            tools=tools,
                            response_format=response_format,
                            output_config=output_config,
                            is_streaming=False,
                        )
                        response = await _make_api_call()
                        result = _process_response(response)

            return result

        except LLMRetryableError:
            raise

        except openai.BadRequestError as e:
            # Handle bad request errors
            error_msg = _format_openai_error("OpenAI bad request", e)

            # Check if error is related to response_format
            if (
                "response_format" in error_msg.lower()
                and "response_format" in completion_params
            ):
                # Remove response_format and retry
                logger.warning(
                    f"API doesn't support response_format, retrying without it. Error: {error_msg}"
                )
                completion_params.pop("response_format")

                # Retry the API call without response_format
                response = await _make_api_call()
                return _process_response(response)

            raise RuntimeError(error_msg) from e

        except openai.APITimeoutError as e:
            # Handle timeout errors
            raise RuntimeError(f"OpenAI API timeout: {str(e)}") from e

        except openai.RateLimitError as e:
            # Handle rate limit errors
            raise RuntimeError(f"OpenAI rate limit exceeded: {e.message}") from e

        except openai.AuthenticationError as e:
            # Handle authentication errors
            raise RuntimeError(f"OpenAI authentication failed: {e.message}") from e

        except openai.APIError as e:
            # Handle OpenAI API errors
            raise RuntimeError(_format_openai_error("OpenAI API error", e)) from e

        except Exception as e:
            # Handle any other unexpected errors
            raise RuntimeError(f"LLM chat failed: {str(e)}") from e

    @property
    def supports_thinking_mode(self) -> bool:
        """
        Check if this OpenAI LLM supports thinking mode.

        Returns:
            bool: True if the model has thinking_mode ability, False otherwise
        """
        return "thinking_mode" in self.abilities

    def _attach_reasoning_content_to_raw(
        self,
        raw_payload: Any,
        reasoning_content: str,
        *,
        has_reasoning_content: bool = False,
    ) -> Any:
        """Attach accumulated reasoning content to a raw payload when possible."""
        if not has_reasoning_content:
            return raw_payload

        if hasattr(raw_payload, "model_dump"):
            raw_payload = raw_payload.model_dump()

        if isinstance(raw_payload, dict):
            raw_payload = dict(raw_payload)
            raw_payload["reasoning_content"] = reasoning_content
            raw_payload["reasoning"] = reasoning_content

        return raw_payload

    async def vision_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Perform a vision-aware chat completion for OpenAI models that support vision.
        This method handles multimodal messages with image content.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
                      Content can be a string or list of multimodal content items
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            response_format: Response format specification
            thinking: Provider-specific reasoning configuration for subclasses.
            **kwargs: Additional parameters to pass to the OpenAI API

        Returns:
            - If normal text reply: return string
            - If tool call triggered: return dict with type "tool_call" and tool_calls list

        Raises:
            RuntimeError: If the model doesn't support vision or the API call fails
        """
        if not self.has_ability("vision"):
            raise RuntimeError(
                f"Model {self._model_name} does not support vision capabilities"
            )

        self._ensure_client()
        assert self._client is not None

        extra_body = self._prepare_extra_body(dict(kwargs.pop("extra_body", {}) or {}))

        # Prepare the completion parameters
        completion_params = {
            "model": self._model_name,
            "messages": self._build_request_messages(messages, thinking=thinking),
            **kwargs,
        }

        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens

        if temperature is not None:
            completion_params["temperature"] = temperature
        elif self.default_temperature is not None:
            completion_params["temperature"] = self.default_temperature

        # Add optional parameters
        if tools:
            completion_params["tools"] = tools
            if tool_choice:
                completion_params["tool_choice"] = tool_choice
        elif tool_choice:
            completion_params["tool_choice"] = tool_choice
        if response_format:
            completion_params["response_format"] = response_format

        self._apply_output_config(completion_params, output_config)

        extra_body = self._prepare_provider_reasoning_extra_body(
            extra_body=extra_body,
            thinking=thinking,
            tools=tools,
            response_format=response_format,
            output_config=output_config,
            is_streaming=False,
        )

        try:
            # Make the API call with extra_body if needed
            if extra_body:
                response = await self._client.chat.completions.create(
                    extra_body=extra_body, **completion_params
                )
            else:
                response = await self._client.chat.completions.create(
                    **completion_params
                )

            # Validate response
            if not hasattr(response, "choices") or not response.choices:
                raise RuntimeError(
                    f"Invalid API response: no choices in response. Response: {response}"
                )

            # Extract the choice
            choice = response.choices[0]
            message = choice.message

            # Record token usage to context
            if hasattr(response, "usage") and response.usage:
                add_token_usage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=self._model_name,
                    model_id=self.model_id,
                    cached_input_tokens=extract_cached_input_tokens(response.usage),
                    call_type="chat",
                )

            # Check for tool calls
            if message.tool_calls:
                # Convert OpenAI tool calls to our format
                tool_calls = []
                for tool_call in message.tool_calls:
                    # Only handle function tool calls, not custom tool calls
                    if hasattr(tool_call, "function"):
                        func = tool_call.function
                        args = func.arguments if func.arguments else ""

                        # Validate arguments are not empty
                        if not args or args.strip() == "":
                            raise RuntimeError(
                                f"Tool '{func.name}' has empty arguments. "
                                f"This is a bug in the LLM provider's tool calling implementation. "
                                f"Model: {self._model_name}"
                            )

                        tool_calls.append(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": func.name,
                                    "arguments": args,
                                },
                            }
                        )

                result = {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                    "raw": response.model_dump(),
                }
                has_reasoning_content, reasoning_content = _message_reasoning_content(
                    message
                )
                if has_reasoning_content:
                    result["reasoning_content"] = reasoning_content
                    result["reasoning"] = reasoning_content
                provider_state = self._response_provider_state(result)
                if provider_state:
                    result[PROVIDER_STATE_METADATA_KEY] = provider_state
                return result

            # Handle text content
            content = message.content
            has_reasoning_content, reasoning_content = _message_reasoning_content(
                message
            )
            finish_reason = getattr(choice, "finish_reason", None)

            # Handle None or empty content when no tool calls
            if not content or not content.strip():
                # See ``chat()``: reasoning models truncated by ``max_tokens``
                # may return ``content=""`` with the partial answer in
                # ``reasoning_content``. Surface it as content rather than
                # treating the response as invalid. Mirror the ``content``
                # whitespace check so a reasoning trace that is purely
                # whitespace still falls through to the empty-response error.
                # Gate strictly on ``finish_reason == "length"`` so a
                # ``"stop"``/``"content_filter"``/``None`` choice with no
                # final content still raises -- those mean the model claims
                # to be done but produced nothing, which is a real failure.
                if (
                    finish_reason == "length"
                    and reasoning_content
                    and reasoning_content.strip()
                ):
                    return {
                        "type": "text",
                        "content": reasoning_content,
                        "reasoning_content": reasoning_content,
                        "reasoning": reasoning_content,
                        "raw": response.model_dump(),
                    }
                # If there are no tool calls and no content, this is an error
                raise LLMEmptyContentError(
                    f"LLM returned {'empty' if content == '' else 'None'} content and no tool calls"
                )

            text_result: Dict[str, Any] = {
                "type": "text",
                "content": content,
                "raw": response.model_dump(),
            }
            if has_reasoning_content:
                text_result["reasoning_content"] = reasoning_content
                text_result["reasoning"] = reasoning_content
            return text_result

        except LLMRetryableError:
            raise

        except openai.APITimeoutError as e:
            # Handle timeout errors
            raise RuntimeError(f"OpenAI API timeout: {str(e)}") from e

        except openai.RateLimitError as e:
            # Handle rate limit errors
            raise RuntimeError(f"OpenAI rate limit exceeded: {e.message}") from e

        except openai.AuthenticationError as e:
            # Handle authentication errors
            raise RuntimeError(f"OpenAI authentication failed: {e.message}") from e

        except openai.BadRequestError as e:
            # Handle bad request errors
            error_msg = _format_openai_error("OpenAI bad request", e)

            # Check if error is related to response_format
            if (
                "response_format" in error_msg.lower()
                and "response_format" in completion_params
            ):
                # Remove response_format and retry
                logger.warning(
                    f"API doesn't support response_format, retrying without it. Error: {error_msg}"
                )
                completion_params.pop("response_format")

                # Retry the API call without response_format
                if extra_body:
                    response = await self._client.chat.completions.create(
                        extra_body=extra_body, **completion_params
                    )
                else:
                    response = await self._client.chat.completions.create(
                        **completion_params
                    )
            else:
                raise RuntimeError(error_msg) from e

        except openai.APIError as e:
            # Handle OpenAI API errors
            raise RuntimeError(_format_openai_error("OpenAI API error", e)) from e

        except Exception as e:
            # Handle any other unexpected errors
            raise RuntimeError(f"LLM vision chat failed: {str(e)}") from e

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
        """
        Stream chat completion with timeout controls and token tracking.

        Supports real-time token output, flexible timeout controls, and precise token statistics.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            response_format: Response format specification
            thinking: Provider-specific reasoning configuration for subclasses.
            **kwargs: Additional parameters to pass to the OpenAI API

        Yields:
            StreamChunk: Streaming response chunks

        Raises:
            RuntimeError: API call failed
            TimeoutError: First token timeout or token interval timeout
        """
        self._ensure_client()
        assert self._client is not None

        extra_body = self._prepare_extra_body(dict(kwargs.pop("extra_body", {}) or {}))

        # Prepare completion parameters
        completion_params = {
            "model": self._model_name,
            "messages": self._build_request_messages(messages, thinking=thinking),
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs,
        }

        # Only set max_tokens if explicitly provided
        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens

        if temperature is not None:
            completion_params["temperature"] = temperature
        elif self.default_temperature is not None:
            completion_params["temperature"] = self.default_temperature

        # Add tools if provided
        if tools:
            completion_params["tools"] = tools
            if tool_choice:
                completion_params["tool_choice"] = tool_choice
        elif tool_choice:
            completion_params["tool_choice"] = tool_choice

        if response_format:
            completion_params["response_format"] = response_format

        self._apply_output_config(completion_params, output_config)

        extra_body = self._prepare_provider_reasoning_extra_body(
            extra_body=extra_body,
            thinking=thinking,
            tools=tools,
            response_format=response_format,
            output_config=output_config,
            is_streaming=True,
        )

        try:
            # Create streaming response
            try:
                if extra_body:
                    stream = await self._client.chat.completions.create(
                        extra_body=extra_body, **completion_params
                    )
                else:
                    stream = await self._client.chat.completions.create(
                        **completion_params
                    )
            except openai.BadRequestError as e:
                # Check if error is related to response_format
                error_msg = _format_openai_error("OpenAI bad request", e)
                if (
                    "response_format" in error_msg.lower()
                    and "response_format" in completion_params
                ):
                    # Remove response_format and retry
                    logger.warning(
                        f"API doesn't support response_format, retrying without it. Error: {error_msg}"
                    )
                    completion_params.pop("response_format")

                    if extra_body:
                        stream = await self._client.chat.completions.create(
                            extra_body=extra_body, **completion_params
                        )
                    else:
                        stream = await self._client.chat.completions.create(
                            **completion_params
                        )
                else:
                    raise

            # Timeout control
            first_token = True
            last_token_time = None
            start_time = time.time()

            # Accumulate tool calls (across multiple chunks)
            accumulated_tool_calls: Dict[str, Dict] = {}
            accumulated_reasoning_content = ""
            has_reasoning_content = False
            last_raw_chunk = None  # Track last raw chunk for usage extraction
            usage_received = False

            async for raw_chunk in stream:
                current_time = time.time()

                # Check first token timeout
                if first_token:
                    elapsed = current_time - start_time
                    if elapsed > self.timeout_config.first_token_timeout:
                        logger.error(f"First token timeout after {elapsed}s")
                        raise LLMTimeoutError(
                            f"First token timeout: {elapsed}s > {self.timeout_config.first_token_timeout}s"
                        )
                    first_token = False
                    logger.debug(f"First token received after {elapsed:.2f}s")

                # Check token interval timeout
                if last_token_time is not None:
                    interval = current_time - last_token_time
                    if interval > self.timeout_config.token_interval_timeout:
                        logger.error(f"Token interval timeout: {interval}s")
                        raise LLMTimeoutError(
                            f"Token interval timeout: {interval}s > {self.timeout_config.token_interval_timeout}s"
                        )

                last_token_time = current_time

                # Store last raw chunk for potential usage extraction
                last_raw_chunk = raw_chunk

                # Parse chunk
                if hasattr(raw_chunk, "choices") and raw_chunk.choices:
                    delta = raw_chunk.choices[0].delta
                    delta_has_reasoning, delta_reasoning_content = (
                        _message_reasoning_content(delta)
                    )
                    if delta_has_reasoning:
                        has_reasoning_content = True
                        accumulated_reasoning_content += str(
                            delta_reasoning_content or ""
                        )

                chunk = self._parse_stream_chunk(
                    raw_chunk,
                    accumulated_tool_calls,
                    accumulated_reasoning_content,
                    has_reasoning_content=has_reasoning_content,
                )
                if chunk:
                    if chunk.is_usage():
                        usage_received = True
                    yield chunk

            # Fallback: Ensure usage chunk is always sent
            # If no usage chunk was received, try to extract from the last raw chunk
            if not usage_received and last_raw_chunk is not None:
                logger.warning(
                    "OpenAI stream ended without usage chunk, attempting to extract from last chunk"
                )
                if hasattr(last_raw_chunk, "usage") and last_raw_chunk.usage:
                    usage = last_raw_chunk.usage
                    input_tokens = getattr(usage, "prompt_tokens", 0)
                    output_tokens = getattr(usage, "completion_tokens", 0)

                    if input_tokens > 0 or output_tokens > 0:
                        # Record token usage
                        add_token_usage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            model=self._model_name,
                            model_id=self.model_id,
                            cached_input_tokens=extract_cached_input_tokens(usage),
                            call_type="stream_chat",
                        )

                        # Yield usage chunk
                        yield StreamChunk(
                            type=ChunkType.USAGE,
                            usage={
                                "prompt_tokens": input_tokens,
                                "completion_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens,
                            },
                            raw=last_raw_chunk,
                        )
                        logger.info(
                            f"Extracted usage from last chunk: {input_tokens} + {output_tokens} tokens"
                        )

        except LLMTimeoutError:
            # Re-raise timeout errors for retry
            raise

        except openai.APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {e}")
            raise LLMRetryableError(f"OpenAI API timeout: {str(e)}") from e

        except openai.RateLimitError as e:
            logger.error(
                "OpenAI rate limit exceeded: %s", redact_sensitive_text(str(e))
            )
            raise LLMRetryableError(f"OpenAI rate limit exceeded: {e.message}") from e

        except openai.APIConnectionError as e:
            logger.error(
                "OpenAI stream connection failed: %s", redact_sensitive_text(str(e))
            )
            raise LLMRetryableError(f"OpenAI stream connection failed: {str(e)}") from e

        except openai.AuthenticationError as e:
            logger.error(
                "OpenAI authentication failed: %s", redact_sensitive_text(str(e))
            )
            raise RuntimeError(f"OpenAI authentication failed: {e.message}") from e

        except openai.BadRequestError as e:
            logger.debug("OpenAI bad request: %s", redact_sensitive_text(str(e)))
            raise RuntimeError(_format_openai_error("OpenAI bad request", e)) from e

        except openai.APIError as e:
            logger.error("OpenAI API error: %s", redact_sensitive_text(str(e)))
            raise RuntimeError(_format_openai_error("OpenAI API error", e)) from e

        except TimeoutError:
            raise

        except Exception as e:
            logger.error("OpenAI stream chat failed: %s", redact_sensitive_text(str(e)))
            if _is_retryable_stream_transport_error(e):
                raise LLMRetryableError(
                    f"OpenAI stream connection failed: {str(e)}"
                ) from e
            raise RuntimeError(f"LLM stream chat failed: {str(e)}") from e

    def _parse_stream_chunk(
        self,
        raw_chunk: Any,
        accumulated_tool_calls: Dict,
        accumulated_reasoning_content: str = "",
        *,
        has_reasoning_content: bool = False,
    ) -> Optional[StreamChunk]:
        """
        Parse OpenAI streaming chunk

        Args:
            raw_chunk: Raw chunk returned by OpenAI SDK
            accumulated_tool_calls: Accumulated tool calls (across chunks)

        Returns:
            StreamChunk or None
        """
        # Check choices
        if not hasattr(raw_chunk, "choices") or not raw_chunk.choices:
            # Check usage information (in the final chunk)
            if hasattr(raw_chunk, "usage") and raw_chunk.usage:
                # Automatically record to token context
                add_token_usage(
                    input_tokens=raw_chunk.usage.prompt_tokens,
                    output_tokens=raw_chunk.usage.completion_tokens,
                    model=self._model_name,
                    model_id=self.model_id,
                    cached_input_tokens=extract_cached_input_tokens(raw_chunk.usage),
                    call_type="stream_chat",
                )

                return StreamChunk(
                    type=ChunkType.USAGE,
                    usage={
                        "prompt_tokens": raw_chunk.usage.prompt_tokens,
                        "completion_tokens": raw_chunk.usage.completion_tokens,
                        "total_tokens": raw_chunk.usage.total_tokens,
                    },
                    raw=raw_chunk,
                )
            return None

        choice = raw_chunk.choices[0]
        delta = choice.delta

        # Handle token content
        if hasattr(delta, "content") and delta.content:
            return StreamChunk(
                type=ChunkType.TOKEN,
                content=delta.content,
                delta=delta.content,
                raw=self._attach_reasoning_content_to_raw(
                    raw_chunk,
                    accumulated_reasoning_content,
                    has_reasoning_content=has_reasoning_content,
                ),
            )

        # Handle tool calls
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            tool_calls_list = []

            for tool_call in delta.tool_calls:
                call_id = tool_call.id
                index = tool_call.index
                func = tool_call.function if hasattr(tool_call, "function") else None

                # Handle Azure OpenAI's incremental tool call format
                # where later chunks may have null id but have arguments
                # Also handle qwen's empty string id format
                if call_id is None or call_id == "":
                    if accumulated_tool_calls and index is not None:
                        # Try to associate with the most recent tool call by index
                        for existing_id, existing_tc in accumulated_tool_calls.items():
                            if existing_tc.get("index") == index:
                                call_id = existing_id
                                break
                    if call_id is None or call_id == "":
                        if not self._stream_tool_call_has_payload(tool_call):
                            # Some OpenAI-compatible providers emit empty
                            # placeholder slots while streaming multiple tool
                            # calls. They carry no recoverable data and should
                            # not become a real accumulated call.
                            continue
                        if index is None:
                            # Cannot associate this chunk — skip it
                            continue
                        call_id = f"tool_call_{index}"

                if not self._stream_tool_call_has_payload(tool_call):
                    if call_id not in accumulated_tool_calls:
                        continue

                # Initialize or update accumulated tool call
                if call_id not in accumulated_tool_calls:
                    accumulated_tool_calls[call_id] = {
                        "index": index,
                        "id": call_id,
                        "type": getattr(tool_call, "type", None) or "function",
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    }

                # Update function information (even if call_id is empty string)
                if call_id is not None:
                    # Update function information
                    if func:
                        if hasattr(func, "name") and func.name:
                            accumulated_tool_calls[call_id]["function"]["name"] = (
                                func.name
                            )
                        # FIXED: Always accumulate arguments, even if empty string
                        # Some models send empty chunks before/after actual arguments
                        if hasattr(func, "arguments"):
                            args_to_add = func.arguments if func.arguments else ""
                            accumulated_tool_calls[call_id]["function"][
                                "arguments"
                            ] += args_to_add

            # Return current accumulated tool calls
            tool_calls_list = list(accumulated_tool_calls.values())
            if tool_calls_list:
                return StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=tool_calls_list,
                    raw=self._attach_reasoning_content_to_raw(
                        raw_chunk,
                        accumulated_reasoning_content,
                        has_reasoning_content=has_reasoning_content,
                    ),
                )

        # Check finish reason
        if hasattr(choice, "finish_reason") and choice.finish_reason:
            # If there are tool calls, return complete tool calls
            if accumulated_tool_calls:
                tool_calls_list = list(accumulated_tool_calls.values())

                # Validate all tool calls have non-empty arguments
                for tool_call_dict in tool_calls_list:
                    func_info = tool_call_dict.get("function", {})
                    args = func_info.get("arguments", "")
                    if not args or args.strip() == "":
                        tool_name = func_info.get("name", "unknown")
                        raise RuntimeError(
                            f"Tool '{tool_name}' has empty arguments in streaming response. "
                            f"This is a bug in the LLM provider's tool calling implementation. "
                            f"Model: {self._model_name}, raw tool call: {tool_call_dict}"
                        )

                return StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=tool_calls_list,
                    finish_reason=choice.finish_reason,
                    raw=self._attach_reasoning_content_to_raw(
                        raw_chunk,
                        accumulated_reasoning_content,
                        has_reasoning_content=has_reasoning_content,
                    ),
                )

            return StreamChunk(
                type=ChunkType.END,
                finish_reason=choice.finish_reason,
                raw=self._attach_reasoning_content_to_raw(
                    raw_chunk,
                    accumulated_reasoning_content,
                    has_reasoning_content=has_reasoning_content,
                ),
            )

        return None

    @staticmethod
    def _stream_tool_call_has_payload(tool_call: Any) -> bool:
        func = tool_call.function if hasattr(tool_call, "function") else None
        if func is None:
            return False
        name = getattr(func, "name", None)
        if isinstance(name, str) and name:
            return True
        arguments = getattr(func, "arguments", None)
        if isinstance(arguments, str):
            return bool(arguments)
        return arguments is not None

    async def close(self) -> None:
        """Close the OpenAI client and cleanup resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "OpenAICompatibleLLM":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @staticmethod
    async def list_available_models(
        api_key: str, base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch available models from OpenAI-compatible API using SDK.

        Args:
            api_key: API key for the OpenAI-compatible service
            base_url: Base URL for the API (optional).
                - If not provided, uses official OpenAI API: https://api.openai.com/v1
                - If provided, uses the specified endpoint (e.g., proxy or custom service)

        Returns:
            List of available models with their information

        Example:
            >>> # Use official OpenAI API
            >>> models = await OpenAILLM.list_available_models("sk-...")

            >>> # Use custom endpoint/proxy
            >>> models = await OpenAILLM.list_available_models(
            ...     "sk-...",
            ...     base_url="https://my-proxy.com/v1"
            ... )
        """
        # Create a client using SDK
        client = AsyncOpenAI(
            base_url=base_url if base_url != "https://api.openai.com/v1" else None,
            api_key=api_key,
            timeout=30.0,
        )

        try:
            # Use SDK's models.list() method
            models_pager = await client.models.list()

            models = []
            for model in models_pager.data:
                models.append(
                    {
                        "id": model.id,
                        "created": getattr(model, "created", None),
                        "owned_by": getattr(model, "owned_by", None),
                    }
                )

            # Sort by created date (newest first)
            models.sort(
                key=lambda x: (
                    (x.get("created") or 0) if x.get("created") is not None else 0
                ),
                reverse=True,
            )
            return models

        except openai.AuthenticationError as e:
            logger.error(
                "OpenAI authentication failed: %s", redact_sensitive_text(str(e))
            )
            raise ValueError("Invalid API key") from e
        except Exception as e:
            logger.error("Failed to fetch models: %s", redact_sensitive_text(str(e)))
            return []
        finally:
            await client.close()


class OpenAILLM(OpenAICompatibleLLM):
    """
    OpenAI LLM client using the official OpenAI SDK.

    This public provider class owns OpenAI defaults and request policy while
    inheriting OpenAI-compatible transport/parsing from ``OpenAICompatibleLLM``.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
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
            base_url=(
                base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            ),
            api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"),
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            timeout=timeout,
            abilities=abilities,
            timeout_config=timeout_config,
        )

    def _apply_output_config(
        self,
        completion_params: Dict[str, Any],
        output_config: Optional[Dict[str, Any]],
    ) -> None:
        if output_config is None:
            return

        format_config = output_config.get("format", {})
        if format_config.get("type") == "json_schema":
            schema = format_config.get("schema", {})
            completion_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title", "response").lower().replace(" ", "_"),
                    "strict": True,
                    "schema": schema,
                },
            }
            return

        completion_params["output_config"] = output_config
