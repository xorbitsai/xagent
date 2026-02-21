import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Union

from json_repair import loads as repair_loads

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
else:
    try:
        from anthropic import Anthropic, AsyncAnthropic  # type: ignore
    except ImportError:
        Anthropic = None  # type: ignore
        AsyncAnthropic = None  # type: ignore

from ..exceptions import LLMRetryableError, LLMTimeoutError
from ..timeout_config import TimeoutConfig
from ..token_context import add_token_usage
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)


class ClaudeLLM(BaseLLM):
    """
    Anthropic Claude LLM client using the official Anthropic SDK.
    Supports Claude models with tool calling and vision capabilities.
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        self._model_name = model_name
        self.api_key = (
            api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        )
        self.base_url = base_url
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        self.timeout_config = timeout_config or TimeoutConfig()

        # Determine abilities based on model name or explicit configuration
        if abilities:
            self._abilities = abilities
        else:
            # Auto-detect abilities based on model name
            self._abilities = ["chat", "tool_calling"]
            if any(
                vision_keyword in model_name.lower()
                for vision_keyword in ["claude-3", "sonnet", "haiku", "opus"]
            ):
                self._abilities.append("vision")

        # Initialize the Anthropic client
        self._client: Optional[AsyncAnthropic] = None

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this Claude LLM implementation."""
        return self._abilities

    def _ensure_client(self) -> None:
        """Ensure the Anthropic client is initialized."""
        if self._client is None:
            if AsyncAnthropic is None:
                raise RuntimeError(
                    "anthropic SDK is not installed. Please install it with: pip install anthropic"
                )

            if not self.api_key:
                raise RuntimeError("ANTHROPIC_API_KEY or CLAUDE_API_KEY must be set")

            # Build client kwargs with proper type annotations
            client_kwargs: Dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }

            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self._client = AsyncAnthropic(**client_kwargs)

    def _convert_messages_to_anthropic_format(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert OpenAI format messages to Anthropic format.

        Args:
            messages: List of messages in OpenAI format

        Returns:
            Tuple of (system_message, list of messages in Anthropic format)
        """
        anthropic_messages = []
        system_message = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Handle system message - Anthropic separates it
            if role == "system":
                if isinstance(content, str):
                    system_message = content
                elif isinstance(content, list):
                    # Extract text from content list
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    system_message = "\n".join(text_parts)
                continue

            # Convert roles (Anthropic uses "user" and "assistant")
            anthropic_role = "user" if role == "user" else "assistant"

            # Handle content
            if isinstance(content, str):
                anthropic_messages.append({"role": anthropic_role, "content": content})
            elif isinstance(content, list):
                # Multimodal content
                anthropic_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            anthropic_content.append(
                                {"type": "text", "text": item.get("text", "")}
                            )
                        elif item.get("type") == "image_url":
                            # Handle image
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url

                            # For base64 data URLs
                            if url.startswith("data:image"):
                                # Parse the data URL
                                # Format: data:image/jpeg;base64,<base64_string>
                                try:
                                    mime_type = url.split(":")[1].split(";")[0]
                                    base64_data = url.split(",", 1)[1]
                                    anthropic_content.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": mime_type,
                                                "data": base64_data,
                                            },
                                        }
                                    )
                                except (IndexError, ValueError):
                                    logger.warning(
                                        f"Invalid data URL format: {url[:50]}..."
                                    )
                            else:
                                # For regular URLs, we'd need to fetch the image
                                # For now, log a warning
                                logger.warning(
                                    f"Direct image URLs not yet supported: {url[:50]}..."
                                )

                anthropic_messages.append(
                    {
                        "role": anthropic_role,
                        "content": anthropic_content if anthropic_content else [],  # type: ignore[dict-item]
                    }
                )

        return system_message, anthropic_messages

    def _convert_tools_to_anthropic_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI format tools to Anthropic format.

        Args:
            tools: List of tools in OpenAI format

        Returns:
            List of tools in Anthropic format
        """
        anthropic_tools = []

        for tool in tools:
            function = tool.get("function", {})
            name = function.get("name", "")
            description = function.get("description", "")
            parameters = function.get("parameters", {})
            strict = tool.get("strict", False)

            # Convert to Anthropic tool format
            anthropic_tool = {
                "name": name,
                "description": description,
                "input_schema": parameters,
            }

            # Add strict mode if specified
            if strict:
                anthropic_tool["strict"] = True

            anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    async def chat(
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
        Perform a chat completion or trigger tool call.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy ("auto", "any", "none", or tool name)
            response_format: Response format specification (e.g., {"type": "json_object"})
            thinking: Thinking mode configuration (enables extended thinking for supported models)
            output_config: Output configuration for structured outputs (e.g., {"format": {"type": "json_schema", "schema": {...}}})
            **kwargs: Additional parameters to pass to the Anthropic API

        Returns:
            - If normal text reply: return string
            - If tool call triggered: return dict with type "tool_call" and tool_calls list

        Raises:
            RuntimeError: If the API call fails
        """
        self._ensure_client()
        assert self._client is not None

        try:
            # Convert messages to Anthropic format
            system_message, anthropic_messages = (
                self._convert_messages_to_anthropic_format(messages)
            )

            # Prepare the completion parameters
            completion_params: Dict[str, Any] = {
                "model": self._model_name,
                "messages": self._sanitize_unicode_content(anthropic_messages),
                **kwargs,
            }

            # Set temperature
            if temperature is not None:
                completion_params["temperature"] = temperature
            elif self.default_temperature is not None:
                completion_params["temperature"] = self.default_temperature

            # Set max_tokens (required by Anthropic API)
            if max_tokens is not None:
                completion_params["max_tokens"] = max_tokens
            elif self.default_max_tokens is not None:
                completion_params["max_tokens"] = self.default_max_tokens
            else:
                # Anthropic requires max_tokens, use a reasonable default
                # Claude 3.5 Sonnet supports up to 8K output, 4K is a safe default
                completion_params["max_tokens"] = 4096

            # Set system message
            if system_message:
                completion_params["system"] = system_message

            # Handle tools
            if tools:
                completion_params["tools"] = self._convert_tools_to_anthropic_format(
                    tools
                )

                # Handle tool_choice
                if tool_choice:
                    if isinstance(tool_choice, str):
                        if tool_choice == "auto":
                            completion_params["tool_choice"] = {"type": "auto"}
                        elif tool_choice == "any" or tool_choice == "required":
                            completion_params["tool_choice"] = {"type": "any"}
                        elif tool_choice == "none":
                            # Don't include tools in the request
                            del completion_params["tools"]
                        else:
                            # Specific tool name
                            completion_params["tool_choice"] = {
                                "type": "tool",
                                "name": tool_choice,
                            }
                    elif isinstance(tool_choice, dict):
                        # Pass through the dict format
                        completion_params["tool_choice"] = tool_choice
            elif tool_choice and tool_choice != "none":
                # tool_choice specified but no tools - this is an error condition
                logger.warning(
                    f"tool_choice specified as {tool_choice} but no tools provided"
                )

            # Handle thinking mode (extended thinking)
            if thinking is not None:
                if thinking.get("type") == "enabled" or thinking.get("enable", False):
                    completion_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking.get("budget_tokens", 10240),
                    }
                elif thinking.get("type") == "disabled":
                    completion_params["thinking"] = {"type": "disabled"}

            # Handle output_config for structured outputs
            if output_config is not None:
                completion_params["output_config"] = output_config

            # Handle response_format for JSON mode
            # Anthropic doesn't have a response_format parameter, so we add instructions to system message
            if (
                response_format is not None
                and response_format.get("type") == "json_object"
            ):
                json_instruction = "You must respond with valid JSON only. Do not include any text outside the JSON structure."
                if system_message:
                    completion_params["system"] = (
                        f"{system_message}\n\n{json_instruction}"
                    )
                else:
                    completion_params["system"] = json_instruction

            # Make the API call
            response = await self._client.messages.create(**completion_params)

            # Record token usage
            if hasattr(response, "usage"):
                usage = response.usage
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                add_token_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model_name,
                    call_type="chat",
                )

            # Check for tool use in response
            if hasattr(response, "stop_reason") and response.stop_reason == "tool_use":
                # Extract tool calls
                tool_calls = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input)
                                    if hasattr(block, "input")
                                    else "{}",
                                },
                            }
                        )

                if tool_calls:
                    return {
                        "type": "tool_call",
                        "tool_calls": tool_calls,
                        "raw": response.model_dump()
                        if hasattr(response, "model_dump")
                        else str(response),
                    }

            # Extract text content
            text_content = []
            for block in response.content:
                if block.type == "text":
                    text_content.append(block.text)

            content = "".join(text_content).strip()

            if not content:
                # Empty response should trigger retry
                raise LLMRetryableError("LLM returned empty content and no tool calls")

            # If JSON format was requested, try to repair and validate JSON
            if response_format and response_format.get("type") == "json_object":
                try:
                    # Try to repair the JSON first
                    repaired_content = repair_loads(content, logging=False)
                    # If repair succeeded, return the repaired JSON as string
                    # to maintain consistency with normal text response
                    logger.info("JSON repair succeeded, returning repaired content")
                    return json.dumps(repaired_content, ensure_ascii=False)
                except Exception as repair_error:
                    # JSON repair failed - raise retryable error to trigger retry
                    logger.warning(
                        f"JSON repair failed, retrying LLM call: {repair_error}"
                    )
                    raise LLMRetryableError(
                        "LLM returned unrepairable JSON when response_format=json_object was requested"
                    )

            # Handle output_config with json_schema - content should already be valid JSON
            if output_config is not None:
                format_config = output_config.get("format", {})
                if format_config.get("type") == "json_schema":
                    # When using json_schema, the response is already validated JSON
                    # Return as-is since it's guaranteed to be valid
                    logger.info("Returning JSON schema validated response")
                    return content

            return content

        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            # Re-raise LLMRetryableError as-is (will be caught by retry wrapper)
            # Wrap other errors in RuntimeError
            if isinstance(e, LLMRetryableError):
                raise

            if self._client:
                # We can safely import these as _client exists implies SDK is installed
                from anthropic import (
                    APIConnectionError,
                    APIStatusError,
                    APITimeoutError,
                    RateLimitError,
                )

                if isinstance(e, (APITimeoutError, APIConnectionError, RateLimitError)):
                    raise LLMRetryableError(str(e)) from e

                if isinstance(e, APIStatusError):
                    raise LLMRetryableError(str(e)) from e

            raise RuntimeError(f"Claude API error: {str(e)}") from e

    async def stream_chat(
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
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream chat completion with timeout controls and token tracking.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            response_format: Response format specification (not used in streaming)
            thinking: Thinking mode configuration
            output_config: Output configuration for structured outputs
            **kwargs: Additional parameters to pass to the Claude API

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

        # Tool call accumulation
        accumulated_tool_calls: Dict[str, Dict] = {}
        current_content = ""

        try:
            # Convert messages to Anthropic format
            system_message, anthropic_messages = (
                self._convert_messages_to_anthropic_format(messages)
            )

            # Prepare the completion parameters
            completion_params: Dict[str, Any] = {
                "model": self._model_name,
                "messages": self._sanitize_unicode_content(anthropic_messages),
                **kwargs,
            }

            # Set temperature
            if temperature is not None:
                completion_params["temperature"] = temperature
            elif self.default_temperature is not None:
                completion_params["temperature"] = self.default_temperature

            # Set max_tokens (required by Anthropic API)
            if max_tokens is not None:
                completion_params["max_tokens"] = max_tokens
            elif self.default_max_tokens is not None:
                completion_params["max_tokens"] = self.default_max_tokens
            else:
                completion_params["max_tokens"] = 4096

            # Set system message
            if system_message:
                completion_params["system"] = system_message

            # Handle tools
            if tools:
                completion_params["tools"] = self._convert_tools_to_anthropic_format(
                    tools
                )

                # Handle tool_choice
                if tool_choice:
                    if isinstance(tool_choice, str):
                        if tool_choice == "auto":
                            completion_params["tool_choice"] = {"type": "auto"}
                        elif tool_choice == "any" or tool_choice == "required":
                            completion_params["tool_choice"] = {"type": "any"}
                        elif tool_choice == "none":
                            del completion_params["tools"]
                        else:
                            completion_params["tool_choice"] = {
                                "type": "tool",
                                "name": tool_choice,
                            }
                    elif isinstance(tool_choice, dict):
                        completion_params["tool_choice"] = tool_choice
            elif tool_choice and tool_choice != "none":
                logger.warning(
                    f"tool_choice specified as {tool_choice} but no tools provided"
                )

            # Handle thinking mode (extended thinking)
            if thinking is not None:
                if thinking.get("type") == "enabled" or thinking.get("enable", False):
                    completion_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking.get("budget_tokens", 10240),
                    }
                elif thinking.get("type") == "disabled":
                    completion_params["thinking"] = {"type": "disabled"}

            # Handle output_config for structured outputs
            if output_config is not None:
                completion_params["output_config"] = output_config

            # Handle response_format for JSON mode
            # Anthropic doesn't have a response_format parameter, so we add instructions to system message
            if (
                response_format is not None
                and response_format.get("type") == "json_object"
            ):
                json_instruction = "You must respond with valid JSON only. Do not include any text outside the JSON structure."
                if system_message:
                    completion_params["system"] = (
                        f"{system_message}\n\n{json_instruction}"
                    )
                else:
                    completion_params["system"] = json_instruction

            # Make the streaming API call
            stream = await self._client.messages.create(
                **completion_params, stream=True
            )

            # Process streaming response
            async for event in stream:
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

                # Parse event based on type
                if event.type == "content_block_start":
                    # Start of a new content block (text or tool_use)
                    if hasattr(event, "content_block"):
                        block = event.content_block
                        if block.type == "tool_use":
                            # Initialize tool call
                            tool_id = block.id if hasattr(block, "id") else ""
                            tool_name = block.name if hasattr(block, "name") else ""
                            accumulated_tool_calls[tool_id] = {
                                "id": tool_id,
                                "name": tool_name,
                                "arguments": "",
                            }

                elif event.type == "content_block_delta":
                    # Incremental content update
                    if hasattr(event, "delta"):
                        delta = event.delta
                        if delta.type == "text_delta":
                            # Text token
                            text = delta.text if hasattr(delta, "text") else ""
                            current_content += text
                            yield StreamChunk(
                                type=ChunkType.TOKEN,
                                content=current_content,
                                delta=text,
                                raw=event,
                            )
                        elif delta.type == "input_json_delta":
                            # Tool arguments update
                            tool_id = (
                                event.index
                                if hasattr(event, "index")
                                else list(accumulated_tool_calls.keys())[0]
                            )
                            if tool_id in accumulated_tool_calls:
                                args = (
                                    delta.partial_json
                                    if hasattr(delta, "partial_json")
                                    else ""
                                )
                                accumulated_tool_calls[tool_id]["arguments"] += args

                elif event.type == "content_block_stop":
                    # End of content block
                    pass

                elif event.type == "message_start":
                    # Message start (contains usage info)
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        usage = event.message.usage
                        # Record input tokens
                        if hasattr(usage, "input_tokens"):
                            add_token_usage(
                                input_tokens=usage.input_tokens,
                                model=self._model_name,
                                call_type="stream_chat",
                            )

                elif event.type == "message_delta":
                    # Message end (contains usage info)
                    if hasattr(event, "usage"):
                        usage = event.usage
                        # Record output tokens
                        if hasattr(usage, "output_tokens"):
                            add_token_usage(
                                output_tokens=usage.output_tokens,
                                model=self._model_name,
                                call_type="stream_chat",
                            )

                        # Yield usage chunk
                        usage_dict = {
                            "input_tokens": getattr(usage, "input_tokens", 0),
                            "output_tokens": getattr(usage, "output_tokens", 0),
                            "total_tokens": getattr(usage, "input_tokens", 0)
                            + getattr(usage, "output_tokens", 0),
                        }
                        yield StreamChunk(
                            type=ChunkType.USAGE,
                            usage=usage_dict,
                            raw=event,
                        )

                    # Check for stop reason
                    if hasattr(event, "delta"):
                        delta = event.delta
                        stop_reason = (
                            delta.stop_reason if hasattr(delta, "stop_reason") else None
                        )
                        if stop_reason:
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
                                    finish_reason=stop_reason,
                                    raw=event,
                                )
                            else:
                                yield StreamChunk(
                                    type=ChunkType.END,
                                    finish_reason=stop_reason,
                                    raw=event,
                                )

        except Exception as e:
            logger.error(f"Claude streaming API error: {str(e)}")
            # Re-raise LLMRetryableError for retry
            if isinstance(e, LLMRetryableError):
                raise

            # Check for retryable errors from anthropic
            if self._client:
                from anthropic import (
                    APIConnectionError,
                    APIStatusError,
                    APITimeoutError,
                    RateLimitError,
                )

                if isinstance(e, (APITimeoutError, APIConnectionError, RateLimitError)):
                    error_msg = f"Claude API retryable error: {str(e)}"
                    raise LLMRetryableError(error_msg) from e

                if isinstance(e, APIStatusError):
                    error_msg = f"Claude API status error: {str(e)}"
                    yield StreamChunk(type=ChunkType.ERROR, content=error_msg, raw=e)
                    return

            # Wrap other errors
            error_msg = f"Claude API error: {str(e)}"
            yield StreamChunk(type=ChunkType.ERROR, content=error_msg, raw=e)

    @property
    def supports_thinking_mode(self) -> bool:
        """
        Check if this Claude LLM supports thinking mode.

        Returns:
            bool: True - Claude supports extended thinking mode
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
        Perform a vision-aware chat completion for Claude models that support vision.
        This method handles multimodal messages with image content.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
                      Content can be a string or list of multimodal content items
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            response_format: Response format specification
            thinking: Thinking mode configuration (enables extended thinking)
            output_config: Output configuration for structured outputs
            **kwargs: Additional parameters to pass to the Claude API

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

        # Claude's chat method handles vision automatically
        return await self.chat(
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

    async def close(self) -> None:
        """Close the Anthropic client and cleanup resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "ClaudeLLM":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @staticmethod
    async def list_available_models(
        api_key: str, base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch available models from Anthropic Claude API.

        Args:
            api_key: Anthropic API key
            base_url: Base URL for Claude API (optional). If not provided,
                     uses the official Anthropic API.

        Returns:
            List of available models with their information

        Example:
            >>> # Using official API
            >>> models = await ClaudeLLM.list_available_models("sk-ant-...")
            >>> # Using custom endpoint
            >>> models = await ClaudeLLM.list_available_models(
            ...     "sk-ant-...",
            ...     base_url="https://custom-proxy.com"
            ... )
        """
        import httpx

        # Use official API if no custom base_url provided
        if base_url is None:
            base_url = "https://api.anthropic.com"

        # Remove trailing /v1 if present to avoid duplication
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        url = base_url + "/v1/models"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

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
                            "display_name": model.get("display_name"),
                            "created": model.get("created"),
                            "type": model.get("type"),
                        }
                    )

                # Sort by created date (newest first)
                models.sort(key=lambda x: x.get("created", 0), reverse=True)
                return models

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching Claude models: {e.response.status_code}")
            if e.response.status_code == 401:
                raise ValueError("Invalid Anthropic API key") from e
            raise
        except Exception as e:
            logger.error(f"Failed to fetch Claude models: {e}")
            return []
