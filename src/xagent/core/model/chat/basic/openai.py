import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import openai
from openai import AsyncOpenAI

from ....utils.security import redact_sensitive_text
from ..exceptions import LLMRetryableError, LLMTimeoutError
from ..timeout_config import TimeoutConfig
from ..token_context import add_token_usage
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM client using the official OpenAI SDK.
    Supports custom endpoints (e.g., Xinference) and all OpenAI API features.
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
        self._model_name = model_name
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
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
            thinking: Thinking mode configuration (enables thinking mode for supported models)
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

        # Prepare the completion parameters
        completion_params = {
            "model": self._model_name,
            "messages": self._sanitize_unicode_content(messages),
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

        # Handle output_config for structured outputs (JSON schema)
        if output_config is not None:
            # For OpenAI, we can pass output_config directly or convert to response_format
            # if it's using json_schema format
            format_config = output_config.get("format", {})
            if format_config.get("type") == "json_schema":
                # OpenAI supports json_schema through response_format
                completion_params["response_format"] = format_config
            else:
                # Pass through other output_config formats
                completion_params["output_config"] = output_config

        # Handle thinking mode using extra_body as specified in the requirements
        # Only add enable_thinking if the client supports this parameter (e.g., standard OpenAI)
        extra_body = {}

        # Check if this is a thinking-only model (only supports thinking_mode, not chat)
        is_thinking_only = (
            "thinking_mode" in self.abilities and "chat" not in self.abilities
        )

        # Check if this is a streaming call
        is_streaming = completion_params.get("stream", False)

        if not self.supports_enable_thinking_param:
            # Skip all enable_thinking logic for clients that don't support it (e.g., Azure OpenAI)
            pass
        elif is_thinking_only:
            # For thinking-only models, thinking mode is inherent - no extra_body needed
            # The model naturally thinks as part of its core functionality
            pass
        elif thinking is not None:
            # User explicitly specified thinking mode for hybrid models
            if thinking.get("type") == "enabled" or thinking.get("enable", False):
                # Only enable thinking for streaming calls
                if is_streaming:
                    extra_body["enable_thinking"] = True
                else:
                    # For non-streaming calls, enable_thinking must be false
                    extra_body["enable_thinking"] = False
            elif thinking.get("type") == "disabled" or not thinking.get(
                "enable", False
            ):
                # For hybrid models, allow disabling thinking mode
                extra_body["enable_thinking"] = False
        elif self.supports_thinking_mode and "thinking_mode" in self.abilities:
            # For hybrid models with thinking_mode ability, auto-enable thinking mode only for streaming
            if is_streaming:
                extra_body["enable_thinking"] = True
            else:
                # For non-streaming calls, enable_thinking must be false
                extra_body["enable_thinking"] = False

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
                    call_type="chat",
                )

            # Check for tool calls
            if message.tool_calls:
                # Convert OpenAI tool calls to our format
                tool_calls = []
                for tool_call in message.tool_calls:
                    # Only handle function tool calls, not custom tool calls
                    if hasattr(tool_call, "function"):
                        tool_calls.append(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )

                return {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                    "raw": resp.model_dump(),
                }

            # Handle text content
            content = message.content

            # Handle None or empty content when no tool calls
            if not content or not content.strip():
                # If there are no tool calls and no content, this is an error
                raise RuntimeError(
                    f"LLM returned {'empty' if content == '' else 'None'} content and no tool calls"
                )

            return {
                "type": "text",
                "content": content,
                "raw": resp.model_dump(),
            }

        try:
            # Make the API call
            response = await _make_api_call()
            return _process_response(response)

        except openai.BadRequestError as e:
            # Handle bad request errors
            error_msg = str(e.message) if hasattr(e, "message") else str(e)

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

            raise RuntimeError(f"OpenAI bad request: {error_msg}") from e

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
            error_msg = f"OpenAI API error: {e.message}"
            if (status_code := getattr(e, "status_code", None)) is not None:
                error_msg = f"OpenAI API error ({status_code}): {e.message}"
            raise RuntimeError(error_msg) from e

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

    @property
    def supports_enable_thinking_param(self) -> bool:
        """
        Check if this client supports the 'enable_thinking' parameter in extra_body.

        Standard OpenAI API supports this parameter for certain models.

        Returns:
            bool: True for standard OpenAI, can be overridden in subclasses
        """
        return True

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
            thinking: Thinking mode configuration (enables thinking mode for supported models)
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

        # Prepare the completion parameters
        completion_params = {
            "model": self._model_name,
            "messages": self._sanitize_unicode_content(messages),
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
            **kwargs,
        }

        # Add optional parameters
        if tools:
            completion_params["tools"] = tools
            if tool_choice:
                completion_params["tool_choice"] = tool_choice
        elif tool_choice:
            completion_params["tool_choice"] = tool_choice
        if response_format:
            completion_params["response_format"] = response_format

        # Handle output_config for structured outputs (JSON schema)
        if output_config is not None:
            # For OpenAI, we can pass output_config directly or convert to response_format
            # if it's using json_schema format
            format_config = output_config.get("format", {})
            if format_config.get("type") == "json_schema":
                # OpenAI supports json_schema through response_format
                completion_params["response_format"] = format_config
            else:
                # Pass through other output_config formats
                completion_params["output_config"] = output_config

        # Handle thinking mode using extra_body as specified in the requirements
        # Only add enable_thinking if the client supports this parameter (e.g., standard OpenAI)
        extra_body = {}

        # Check if this is a thinking-only model (only supports thinking_mode, not chat)
        is_thinking_only = (
            "thinking_mode" in self.abilities and "chat" not in self.abilities
        )

        # Check if this is a streaming call
        is_streaming = completion_params.get("stream", False)

        if not self.supports_enable_thinking_param:
            # Skip all enable_thinking logic for clients that don't support it (e.g., Azure OpenAI)
            pass
        elif is_thinking_only:
            # For thinking-only models, thinking mode is inherent - no extra_body needed
            # The model naturally thinks as part of its core functionality
            pass
        elif thinking is not None:
            # User explicitly specified thinking mode for hybrid models
            if thinking.get("type") == "enabled" or thinking.get("enable", False):
                # Only enable thinking for streaming calls
                if is_streaming:
                    extra_body["enable_thinking"] = True
                else:
                    # For non-streaming calls, enable_thinking must be false
                    extra_body["enable_thinking"] = False
            elif thinking.get("type") == "disabled" or not thinking.get(
                "enable", False
            ):
                # For hybrid models, allow disabling thinking mode
                extra_body["enable_thinking"] = False
        elif self.supports_thinking_mode and "thinking_mode" in self.abilities:
            # For hybrid models with thinking_mode ability, auto-enable thinking mode only for streaming
            if is_streaming:
                extra_body["enable_thinking"] = True
            else:
                # For non-streaming calls, enable_thinking must be false
                extra_body["enable_thinking"] = False

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
                    call_type="chat",
                )

            # Check for tool calls
            if message.tool_calls:
                # Convert OpenAI tool calls to our format
                tool_calls = []
                for tool_call in message.tool_calls:
                    # Only handle function tool calls, not custom tool calls
                    if hasattr(tool_call, "function"):
                        tool_calls.append(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )

                return {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                    "raw": response.model_dump(),
                }

            # Handle text content
            content = message.content

            # Handle None or empty content when no tool calls
            if not content or not content.strip():
                # If there are no tool calls and no content, this is an error
                raise RuntimeError(
                    f"LLM returned {'empty' if content == '' else 'None'} content and no tool calls"
                )

            return {
                "type": "text",
                "content": content,
                "raw": response.model_dump(),
            }

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
            error_msg = str(e.message) if hasattr(e, "message") else str(e)

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
                raise RuntimeError(f"OpenAI bad request: {error_msg}") from e

        except openai.APIError as e:
            # Handle OpenAI API errors
            error_msg = f"OpenAI API error: {e.message}"
            if (status_code := getattr(e, "status_code", None)) is not None:
                error_msg = f"OpenAI API error ({status_code}): {e.message}"
            raise RuntimeError(error_msg) from e

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
            thinking: Thinking mode configuration
            **kwargs: Additional parameters to pass to the OpenAI API

        Yields:
            StreamChunk: Streaming response chunks

        Raises:
            RuntimeError: API call failed
            TimeoutError: First token timeout or token interval timeout
        """
        self._ensure_client()
        assert self._client is not None

        # Prepare completion parameters
        completion_params = {
            "model": self._model_name,
            "messages": self._sanitize_unicode_content(messages),
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

        # Handle thinking mode
        extra_body = {}
        is_thinking_only = (
            "thinking_mode" in self.abilities and "chat" not in self.abilities
        )

        if not self.supports_enable_thinking_param:
            pass
        elif is_thinking_only:
            pass
        elif thinking is not None:
            if thinking.get("type") == "enabled" or thinking.get("enable", False):
                extra_body["enable_thinking"] = True
            elif thinking.get("type") == "disabled" or not thinking.get(
                "enable", False
            ):
                extra_body["enable_thinking"] = False
        elif self.supports_thinking_mode and "thinking_mode" in self.abilities:
            extra_body["enable_thinking"] = True

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
                error_msg = str(e.message) if hasattr(e, "message") else str(e)
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

                # Parse chunk
                chunk = self._parse_stream_chunk(raw_chunk, accumulated_tool_calls)
                if chunk:
                    yield chunk

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

        except openai.AuthenticationError as e:
            logger.error(
                "OpenAI authentication failed: %s", redact_sensitive_text(str(e))
            )
            raise RuntimeError(f"OpenAI authentication failed: {e.message}") from e

        except openai.BadRequestError as e:
            logger.error("OpenAI bad request: %s", redact_sensitive_text(str(e)))
            raise RuntimeError(f"OpenAI bad request: {e.message}") from e

        except openai.APIError as e:
            logger.error("OpenAI API error: %s", redact_sensitive_text(str(e)))
            error_msg = f"OpenAI API error: {e.message}"
            if (status_code := getattr(e, "status_code", None)) is not None:
                error_msg = f"OpenAI API error ({status_code}): {e.message}"
            raise RuntimeError(error_msg) from e

        except TimeoutError:
            raise

        except Exception as e:
            logger.error("OpenAI stream chat failed: %s", redact_sensitive_text(str(e)))
            raise RuntimeError(f"LLM stream chat failed: {str(e)}") from e

    def _parse_stream_chunk(
        self, raw_chunk: Any, accumulated_tool_calls: Dict
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
                raw=raw_chunk,
            )

        # Handle tool calls
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            tool_calls_list = []

            for tool_call in delta.tool_calls:
                call_id = tool_call.id

                # Handle Azure OpenAI's incremental tool call format
                # where later chunks may have null id but have arguments
                if call_id is None and accumulated_tool_calls:
                    # Try to associate with the most recent tool call by index
                    if tool_call.index is not None:
                        # Find the tool call with the same index
                        for existing_id, existing_tc in accumulated_tool_calls.items():
                            if existing_tc.get("index") == tool_call.index:
                                call_id = existing_id
                                break

                # Initialize or update accumulated tool call
                if call_id and call_id not in accumulated_tool_calls:
                    accumulated_tool_calls[call_id] = {
                        "index": tool_call.index,
                        "id": call_id,
                        "type": getattr(tool_call, "type", "function"),
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    }

                # Only process if we have a valid call_id
                if call_id:
                    # Update function information
                    if hasattr(tool_call, "function") and tool_call.function:
                        func = tool_call.function
                        if hasattr(func, "name") and func.name:
                            accumulated_tool_calls[call_id]["function"]["name"] = (
                                func.name
                            )
                        if hasattr(func, "arguments") and func.arguments:
                            accumulated_tool_calls[call_id]["function"][
                                "arguments"
                            ] += func.arguments

            # Return current accumulated tool calls
            tool_calls_list = list(accumulated_tool_calls.values())
            if tool_calls_list:
                return StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=tool_calls_list,
                    raw=raw_chunk,
                )

        # Check finish reason
        if hasattr(choice, "finish_reason") and choice.finish_reason:
            # If there are tool calls, return complete tool calls
            if accumulated_tool_calls:
                return StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=list(accumulated_tool_calls.values()),
                    finish_reason=choice.finish_reason,
                    raw=raw_chunk,
                )

            return StreamChunk(
                type=ChunkType.END,
                finish_reason=choice.finish_reason,
                raw=raw_chunk,
            )

        return None

    async def close(self) -> None:
        """Close the OpenAI client and cleanup resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "OpenAILLM":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @staticmethod
    async def list_available_models(
        api_key: str, base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch available models from OpenAI-compatible API.

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
        import httpx

        # Use official OpenAI API if base_url not provided
        url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                models = []
                for model in data.get("data", []):
                    models.append(
                        {
                            "id": model.get("id"),
                            "created": model.get("created"),
                            "owned_by": model.get("owned_by"),
                        }
                    )

                # Sort by created date (newest first)
                models.sort(key=lambda x: x.get("created", 0), reverse=True)
                return models

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching models: {e.response.status_code}")
            if e.response.status_code == 401:
                raise ValueError("Invalid API key") from e
            raise
        except Exception as e:
            logger.error("Failed to fetch models: %s", redact_sensitive_text(str(e)))
            return []
