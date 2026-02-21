import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

try:
    from zai import ZhipuAiClient  # type: ignore
except ImportError:
    # Fallback for when zai SDK is not available
    ZhipuAiClient = None

from ..exceptions import LLMRetryableError, LLMTimeoutError
from ..timeout_config import TimeoutConfig
from ..token_context import add_token_usage
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)


class ZhipuLLM(BaseLLM):
    """
    Zhipu AI LLM client using the official Zhipu SDK.
    Supports GLM-4.5 with thinking mode and all Zhipu API features.
    """

    def __init__(
        self,
        model_name: str = "glm-4.5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        thinking_mode: Optional[bool] = None,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        self._model_name = model_name
        self.api_key = (
            api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = base_url or os.getenv("ZHIPU_BASE_URL")
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        self.thinking_mode = thinking_mode
        self.timeout_config = timeout_config or TimeoutConfig()

        # Determine abilities based on model name or explicit configuration
        if abilities:
            self._abilities = abilities
        else:
            # Auto-detect abilities based on model name
            self._abilities = ["chat", "tool_calling"]
            if any(
                vision_keyword in model_name.lower()
                for vision_keyword in ["glm-4v", "glm-4.5v", "vision"]
            ):
                self._abilities.append("vision")
            if self.supports_thinking_mode:
                self._abilities.append("thinking_mode")

        # Initialize the Zhipu client
        self._client: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this Zhipu LLM implementation."""
        return self._abilities

    def _ensure_client(self) -> None:
        """Ensure the Zhipu client is initialized."""
        if self._client is None:
            if ZhipuAiClient is None:
                raise RuntimeError(
                    "zai SDK is not installed. Please install it with: pip install zai-sdk"
                )
            self._client = ZhipuAiClient(
                api_key=self.api_key,
                base_url=self.base_url,
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
            thinking: Thinking mode configuration (e.g., {"type": "disabled"})
            **kwargs: Additional parameters to pass to the Zhipu API

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
            "max_tokens": max_tokens or self.default_max_tokens,
            "stream": False,  # We handle streaming separately if needed
            **kwargs,
        }

        if temperature is not None:
            completion_params["temperature"] = temperature
        elif self.default_temperature is not None:
            completion_params["temperature"] = self.default_temperature

        # Add thinking mode if specified (parameter takes precedence over instance setting)
        if thinking is not None:
            completion_params["thinking"] = thinking
        elif self.thinking_mode is not None:
            thinking_type = "enabled" if self.thinking_mode else "disabled"
            completion_params["thinking"] = {"type": thinking_type}

        # Add tools if provided
        if tools:
            completion_params["tools"] = tools
            if tool_choice:
                completion_params["tool_choice"] = (
                    "auto" if tool_choice == "required" else tool_choice
                )

        # Add response format if provided
        if response_format:
            completion_params["response_format"] = response_format

        try:
            # Debug: Log the request parameters
            logger.debug("Zhipu API request parameters:")
            logger.debug(f"  - model: {completion_params.get('model')}")
            logger.debug(
                f"  - messages count: {len(completion_params.get('messages', []))}"
            )
            logger.debug(f"  - tools provided: {'tools' in completion_params}")
            logger.debug(f"  - tool_choice: {completion_params.get('tool_choice')}")
            logger.debug(f"  - thinking: {completion_params.get('thinking')}")
            logger.debug(f"  - temperature: {completion_params.get('temperature')}")

            # Make the API call in a thread pool since the SDK might be synchronous
            if self._client is None:
                raise RuntimeError("Zhipu client is not initialized")

            # Debug: Log before API call
            logger.debug("Making Zhipu API call...")

            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.chat.completions.create(**completion_params),  # type: ignore
                )
                logger.debug(
                    f"Zhipu API call completed, response type: {type(response)}"
                )
            except Exception as executor_error:
                logger.error(
                    f"Executor task failed: {type(executor_error).__name__}: {executor_error}"
                )
                raise executor_error

            # Check if response is None
            if response is None:
                logger.error("Zhipu API returned None response")
                raise RuntimeError("Zhipu API returned None response")

            # Check if response has choices
            if not hasattr(response, "choices") or not response.choices:
                logger.error(f"Zhipu API response missing choices: {response}")
                raise RuntimeError("Zhipu API response missing choices")

            # Record token usage
            if hasattr(response, "usage"):
                usage = response.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(
                    usage, "input_tokens", 0
                )
                output_tokens = getattr(usage, "completion_tokens", 0) or getattr(
                    usage, "output_tokens", 0
                )
                add_token_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model_name,
                    call_type="chat",
                )

            # Extract the choice
            choice = response.choices[0]
            message = choice.message

            # Debug: Print response structure
            logger.debug(f"Zhipu response choice: {choice}")
            logger.debug(f"Zhipu response message: {message}")
            logger.debug(
                f"Choice finish_reason: {getattr(choice, 'finish_reason', 'N/A')}"
            )
            logger.debug(f"Message tool_calls: {getattr(message, 'tool_calls', 'N/A')}")
            logger.debug(f"Message content: {message.content}")
            logger.debug(f"Message content type: {type(message.content)}")

            # Check for tool calls - handle both glm-4.5-air and glm-4.5 formats
            tool_calls = None

            # Try to access tool_calls directly first (glm-4.5 style)
            try:
                # Access choice.message.tool_calls directly
                if hasattr(choice, "message") and hasattr(choice.message, "tool_calls"):
                    tool_calls = choice.message.tool_calls
                    if tool_calls:
                        logger.debug(
                            f"Found tool_calls via direct access: {len(tool_calls)}"
                        )
            except Exception as e:
                logger.debug(f"Error accessing tool_calls directly: {e}")

            # Fallback: Check through message object
            if tool_calls is None:
                # Format 1: Direct tool_calls attribute (glm-4.5-air style)
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls = message.tool_calls
                    logger.debug(
                        f"Found tool_calls in message (glm-4.5-air style): {len(tool_calls)}"
                    )
                # Format 2: Tool calls in choice with finish_reason "tool_calls" (glm-4.5 style)
                elif (
                    hasattr(choice, "finish_reason")
                    and choice.finish_reason == "tool_calls"
                ):
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        tool_calls = message.tool_calls
                        logger.debug(
                            f"Found tool_calls with finish_reason (glm-4.5 style): {len(tool_calls)}"
                        )
                    else:
                        logger.debug(
                            "finish_reason is 'tool_calls' but no tool_calls found in message"
                        )

            logger.debug(f"Final tool_calls result: {tool_calls}")

            # Debug: Print tool_calls structure if found
            if tool_calls:
                logger.debug(f"Tool calls structure: {tool_calls}")
                for i, tool_call in enumerate(tool_calls):
                    logger.debug(f"Tool call {i}: {tool_call}")
                    if hasattr(tool_call, "function"):
                        logger.debug(f"Tool call {i} function: {tool_call.function}")
                        if hasattr(tool_call.function, "name"):
                            logger.debug(
                                f"Tool call {i} name: {tool_call.function.name}"
                            )
                        if hasattr(tool_call.function, "arguments"):
                            logger.debug(
                                f"Tool call {i} arguments: {tool_call.function.arguments}"
                            )

            if tool_calls:
                # Convert Zhipu tool calls to ReAct format
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        # Parse arguments as JSON
                        args = json.loads(tool_call.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    # Return ReAct-compatible tool call format
                    return {
                        "type": "tool_call",
                        "tool_calls": [
                            {
                                "id": getattr(
                                    tool_call, "id", f"call_{uuid.uuid4().hex}"
                                ),
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(args),
                                },
                                "type": "function",
                            }
                        ],
                        "raw": response.model_dump()
                        if hasattr(response, "model_dump")
                        else str(response),
                    }

            # Handle text content
            content = message.content

            # Handle None or empty content when no tool calls
            if not content or not content.strip():
                logger.warning("LLM returned None/empty content. Details:")
                logger.warning(
                    f"  - finish_reason: {getattr(choice, 'finish_reason', 'N/A')}"
                )
                logger.warning(f"  - has tool_calls: {tool_calls is not None}")
                logger.warning(
                    f"  - tool_calls count: {len(tool_calls) if tool_calls else 0}"
                )
                logger.warning(f"  - message attributes: {dir(message)}")

                # If there are no tool calls and no content, this is an error
                if not tool_calls:
                    raise RuntimeError(
                        f"LLM returned {'empty' if content == '' else 'None'} content and no tool calls"
                    )
                else:
                    logger.info(
                        "None/empty content but tool calls present, this is expected behavior"
                    )

            return content

        except Exception as e:
            # Handle any errors
            logger.error("Zhipu API exception details:")
            logger.error(f"  - Exception type: {type(e).__name__}")
            logger.error(f"  - Exception message: {str(e)}")
            logger.error(f"  - Exception args: {e.args}")
            raise RuntimeError(f"Zhipu API error: {str(e)}") from e

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

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            response_format: Response format specification
            thinking: Thinking mode configuration
            **kwargs: Additional parameters to pass to the Zhipu API

        Yields:
            StreamChunk objects with streaming response data

        Raises:
            RuntimeError: If the API call fails
            TimeoutError: If timeout thresholds are exceeded
        """
        self._ensure_client()
        assert self._client is not None

        # Timeout tracking
        first_token = True
        last_token_time = None
        start_time = time.time()

        # Accumulated content and tool calls
        current_content = ""
        accumulated_tool_calls: Dict[str, Dict] = {}

        # Prepare the completion parameters
        completion_params = {
            "model": self._model_name,
            "messages": self._sanitize_unicode_content(messages),
            "max_tokens": max_tokens or self.default_max_tokens,
            "stream": True,  # Enable streaming
            **kwargs,
        }

        if temperature is not None:
            completion_params["temperature"] = temperature
        elif self.default_temperature is not None:
            completion_params["temperature"] = self.default_temperature

        # Add thinking mode if specified
        if thinking is not None:
            completion_params["thinking"] = thinking
        elif self.thinking_mode is not None:
            thinking_type = "enabled" if self.thinking_mode else "disabled"
            completion_params["thinking"] = {"type": thinking_type}

        # Add tools if provided
        if tools:
            completion_params["tools"] = tools
            if tool_choice:
                completion_params["tool_choice"] = (
                    "auto" if tool_choice == "required" else tool_choice
                )

        # Add response format if provided
        if response_format:
            completion_params["response_format"] = response_format

        try:
            # Make the streaming API call in executor
            def create_stream() -> Any:
                if self._client is None:
                    raise RuntimeError("Zhipu client is not initialized")
                return self._client.chat.completions.create(**completion_params)

            stream = await asyncio.get_event_loop().run_in_executor(None, create_stream)

            # Process streaming response
            for chunk in stream:
                current_time = time.time()

                # Check first token timeout
                if first_token:
                    elapsed = current_time - start_time
                    if elapsed > self.timeout_config.first_token_timeout:
                        raise LLMTimeoutError(
                            f"First token timeout: {elapsed:.2f}s > "
                            f"{self.timeout_config.first_token_timeout}s"
                        )
                    first_token = False

                # Check token interval timeout
                if last_token_time is not None:
                    interval = current_time - last_token_time
                    if interval > self.timeout_config.token_interval_timeout:
                        raise LLMTimeoutError(
                            f"Token interval timeout: {interval:.2f}s > "
                            f"{self.timeout_config.token_interval_timeout}s"
                        )

                last_token_time = current_time

                # Parse the chunk (similar to OpenAI format)
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta if hasattr(choice, "delta") else None

                    if delta:
                        # Check for content
                        if hasattr(delta, "content") and delta.content:
                            text = delta.content
                            current_content += text
                            yield StreamChunk(
                                type=ChunkType.TOKEN,
                                content=current_content,
                                delta=text,
                                raw=chunk,
                            )

                        # Check for tool calls
                        if hasattr(delta, "tool_calls") and delta.tool_calls:
                            for tool_call in delta.tool_calls:
                                tool_id = getattr(tool_call, "id", None)

                                # Initialize tool call if not exists
                                if tool_id and tool_id not in accumulated_tool_calls:
                                    accumulated_tool_calls[tool_id] = {
                                        "id": tool_id,
                                        "name": "",
                                        "arguments": "",
                                    }

                                # Update tool call info
                                if tool_id:
                                    if hasattr(tool_call, "function"):
                                        func = tool_call.function
                                        if hasattr(func, "name") and func.name:
                                            accumulated_tool_calls[tool_id]["name"] = (
                                                func.name
                                            )
                                        if (
                                            hasattr(func, "arguments")
                                            and func.arguments
                                        ):
                                            accumulated_tool_calls[tool_id][
                                                "arguments"
                                            ] += func.arguments

                    # Check for finish reason
                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        finish_reason = choice.finish_reason

                        # Yield tool calls if accumulated
                        if accumulated_tool_calls:
                            tool_calls_list = []
                            for tool_call in accumulated_tool_calls.values():
                                tool_calls_list.append(
                                    {
                                        "id": tool_call["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tool_call["name"],
                                            "arguments": tool_call["arguments"],
                                        },
                                    }
                                )

                            yield StreamChunk(
                                type=ChunkType.TOOL_CALL,
                                tool_calls=tool_calls_list,
                                finish_reason=finish_reason,
                                raw=chunk,
                            )
                        else:
                            yield StreamChunk(
                                type=ChunkType.END,
                                finish_reason=finish_reason,
                                raw=chunk,
                            )

                # Check for usage information
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage
                    input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(
                        usage, "input_tokens", 0
                    )
                    output_tokens = getattr(usage, "completion_tokens", 0) or getattr(
                        usage, "output_tokens", 0
                    )

                    # Record token usage
                    add_token_usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=self._model_name,
                        call_type="stream_chat",
                    )

                    # Yield usage chunk
                    usage_dict = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                    yield StreamChunk(
                        type=ChunkType.USAGE,
                        usage=usage_dict,
                        raw=chunk,
                    )

        except LLMTimeoutError:
            # Re-raise timeout errors for retry
            raise

        except TimeoutError:
            # Re-raise timeout errors for retry
            raise LLMRetryableError("Streaming timeout exceeded")

        except Exception as e:
            logger.error(f"Zhipu streaming API error: {str(e)}")
            yield StreamChunk(
                type=ChunkType.ERROR,
                content=f"Zhipu streaming API error: {str(e)}",
                raw=e,
            )

    @property
    def supports_thinking_mode(self) -> bool:
        """
        Check if this Zhipu LLM supports thinking mode.

        Returns:
            bool: True since Zhipu GLM-4.5 supports thinking mode
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
        Perform a vision-aware chat completion for Zhipu models that support vision.
        This method handles multimodal messages with image content.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
                      Content can be a string or list of multimodal content items
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            response_format: Response format specification
            thinking: Thinking mode configuration
            **kwargs: Additional parameters to pass to the Zhipu API

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
            "stream": False,  # We handle streaming separately if needed
            **kwargs,
        }

        # Add thinking mode if specified (parameter takes precedence over instance setting)
        if thinking is not None:
            completion_params["thinking"] = thinking
        elif self.thinking_mode is not None:
            thinking_type = "enabled" if self.thinking_mode else "disabled"
            completion_params["thinking"] = {"type": thinking_type}

        # Add tools if provided
        if tools:
            completion_params["tools"] = tools
            if tool_choice:
                completion_params["tool_choice"] = (
                    "auto" if tool_choice == "required" else tool_choice
                )

        # Add response format if provided
        if response_format:
            completion_params["response_format"] = response_format

        try:
            # Debug: Log the request parameters
            logger.debug("Zhipu Vision API request parameters:")
            logger.debug(f"  - model: {completion_params.get('model')}")
            logger.debug(
                f"  - messages count: {len(completion_params.get('messages', []))}"
            )
            logger.debug(f"  - tools provided: {'tools' in completion_params}")
            logger.debug(f"  - tool_choice: {completion_params.get('tool_choice')}")
            logger.debug(f"  - thinking: {completion_params.get('thinking')}")
            logger.debug(f"  - temperature: {completion_params.get('temperature')}")

            # Make the API call in a thread pool since the SDK might be synchronous
            if self._client is None:
                raise RuntimeError("Zhipu client is not initialized")

            # Debug: Log before API call
            logger.debug("Making Zhipu Vision API call...")

            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.chat.completions.create(**completion_params),  # type: ignore
                )
                logger.debug(
                    f"Zhipu Vision API call completed, response type: {type(response)}"
                )
            except Exception as executor_error:
                logger.error(
                    f"Executor task failed: {type(executor_error).__name__}: {executor_error}"
                )
                raise executor_error

            # Check if response is None
            if response is None:
                logger.error("Zhipu Vision API returned None response")
                raise RuntimeError("Zhipu Vision API returned None response")

            # Check if response has choices
            if not hasattr(response, "choices") or not response.choices:
                logger.error(f"Zhipu Vision API response missing choices: {response}")
                raise RuntimeError("Zhipu Vision API response missing choices")

            # Record token usage
            if hasattr(response, "usage"):
                usage = response.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(
                    usage, "input_tokens", 0
                )
                output_tokens = getattr(usage, "completion_tokens", 0) or getattr(
                    usage, "output_tokens", 0
                )
                add_token_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model_name,
                    call_type="vision_chat",
                )

            # Extract the choice
            choice = response.choices[0]
            message = choice.message

            # Debug: Print response structure
            logger.debug(f"Zhipu Vision response choice: {choice}")
            logger.debug(f"Zhipu Vision response message: {message}")
            logger.debug(
                f"Choice finish_reason: {getattr(choice, 'finish_reason', 'N/A')}"
            )
            logger.debug(f"Message tool_calls: {getattr(message, 'tool_calls', 'N/A')}")
            logger.debug(f"Message content: {message.content}")
            logger.debug(f"Message content type: {type(message.content)}")

            # Check for tool calls - handle both glm-4.5-air and glm-4.5 formats
            tool_calls = None

            # Try to access tool_calls directly first (glm-4.5 style)
            try:
                # Access choice.message.tool_calls directly
                if hasattr(choice, "message") and hasattr(choice.message, "tool_calls"):
                    tool_calls = choice.message.tool_calls
                    if tool_calls:
                        logger.debug(
                            f"Found tool_calls via direct access: {len(tool_calls)}"
                        )
            except Exception as e:
                logger.debug(f"Error accessing tool_calls directly: {e}")

            # Fallback: Check through message object
            if tool_calls is None:
                # Format 1: Direct tool_calls attribute (glm-4.5-air style)
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls = message.tool_calls
                    logger.debug(
                        f"Found tool_calls in message (glm-4.5-air style): {len(tool_calls)}"
                    )
                # Format 2: Tool calls in choice with finish_reason "tool_calls" (glm-4.5 style)
                elif (
                    hasattr(choice, "finish_reason")
                    and choice.finish_reason == "tool_calls"
                ):
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        tool_calls = message.tool_calls
                        logger.debug(
                            f"Found tool_calls with finish_reason (glm-4.5 style): {len(tool_calls)}"
                        )
                    else:
                        logger.debug(
                            "finish_reason is 'tool_calls' but no tool_calls found in message"
                        )

            logger.debug(f"Final tool_calls result: {tool_calls}")

            # Debug: Print tool_calls structure if found
            if tool_calls:
                logger.debug(f"Tool calls structure: {tool_calls}")
                for i, tool_call in enumerate(tool_calls):
                    logger.debug(f"Tool call {i}: {tool_call}")
                    if hasattr(tool_call, "function"):
                        logger.debug(f"Tool call {i} function: {tool_call.function}")
                        if hasattr(tool_call.function, "name"):
                            logger.debug(
                                f"Tool call {i} name: {tool_call.function.name}"
                            )
                        if hasattr(tool_call.function, "arguments"):
                            logger.debug(
                                f"Tool call {i} arguments: {tool_call.function.arguments}"
                            )

            if tool_calls:
                # Convert Zhipu tool calls to ReAct format
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        # Parse arguments as JSON
                        args = json.loads(tool_call.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    # Return ReAct-compatible tool call format
                    return {
                        "type": "tool_call",
                        "tool_calls": [
                            {
                                "id": getattr(
                                    tool_call, "id", f"call_{uuid.uuid4().hex}"
                                ),
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(args),
                                },
                                "type": "function",
                            }
                        ],
                        "raw": response.model_dump()
                        if hasattr(response, "model_dump")
                        else str(response),
                    }

            # Handle text content
            content = message.content

            # Debug: Log detailed information about None content
            if content is None:
                logger.warning("Zhipu Vision LLM returned None content. Details:")
                logger.warning(
                    f"  - finish_reason: {getattr(choice, 'finish_reason', 'N/A')}"
                )
                logger.warning(f"  - has tool_calls: {tool_calls is not None}")
                logger.warning(
                    f"  - tool_calls count: {len(tool_calls) if tool_calls else 0}"
                )
                logger.warning(f"  - message attributes: {dir(message)}")

                # If there are no tool calls and no content, return empty string instead of error
                # This allows React pattern to handle the gracefully
                if not tool_calls:
                    logger.warning(
                        "No tool calls and None content, returning empty string"
                    )
                    return ""
                else:
                    logger.info(
                        "None content but tool calls present, this is expected behavior"
                    )

            return content

        except Exception as e:
            # Handle any errors
            logger.error("Zhipu Vision API exception details:")
            logger.error(f"  - Exception type: {type(e).__name__}")
            logger.error(f"  - Exception message: {str(e)}")
            logger.error(f"  - Exception args: {e.args}")
            raise RuntimeError(f"Zhipu Vision API error: {str(e)}") from e

    async def close(self) -> None:
        """Close the Zhipu client and cleanup resources."""
        # The Zhipu client doesn't have an explicit close method
        self._client = None

    async def __aenter__(self) -> "ZhipuLLM":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @staticmethod
    async def list_available_models(
        api_key: str, base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch available models from Zhipu AI using their SDK.

        Args:
            api_key: Zhipu API key
            base_url: Base URL for Zhipu API (optional).
                - If not provided, uses official Zhipu API: https://open.bigmodel.cn/v1
                - If provided, uses the specified endpoint

        Returns:
            List of available models with their information

        Example:
            >>> # Use official Zhipu API
            >>> models = await ZhipuLLM.list_available_models("your-api-key")

            >>> # Use custom endpoint/proxy
            >>> models = await ZhipuLLM.list_available_models(
            ...     "your-api-key",
            ...     base_url="https://my-proxy.com/v1"
            ... )
        """
        if ZhipuAiClient is None:
            raise ImportError(
                "zai SDK is not installed. Please install it with: pip install zai"
            )

        # Use official Zhipu API if base_url not provided
        if not base_url:
            base_url = "https://open.bigmodel.cn/v1"

        try:
            # Create client instance
            client = ZhipuAiClient(api_key=api_key)

            # Try to use SDK's models.list() method if available
            try:
                # zai SDK should have a models.list() method
                models_response = await asyncio.to_thread(client.models.list)

                models = []
                for model in models_response.data:
                    models.append(
                        {
                            "id": model.id,
                            "created": getattr(model, "created", None),
                            "owned_by": "zhipu",
                        }
                    )

                # Sort by created date (newest first)
                models.sort(key=lambda x: x.get("created", 0), reverse=True)
                return models

            except AttributeError:
                # If SDK doesn't have models.list(), use OpenAI-compatible endpoint
                import httpx

                url = base_url.rstrip("/") + "/models"
                headers = {"Authorization": f"Bearer {api_key}"}

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
                                "owned_by": "zhipu",
                            }
                        )

                    models.sort(key=lambda x: x.get("created", 0), reverse=True)
                    return models

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching Zhipu models: {e.response.status_code}")
            if e.response.status_code == 401:
                raise ValueError("Invalid API key") from e
            raise
        except Exception as e:
            logger.error(f"Failed to fetch Zhipu models: {e}")
            return []
