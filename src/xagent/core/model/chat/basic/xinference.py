"""Xinference LLM provider implementation."""

import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from xinference_client import RESTfulClient as XinferenceClient

from ..exceptions import LLMTimeoutError
from ..timeout_config import TimeoutConfig
from ..token_context import add_token_usage
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)


class XinferenceLLM(BaseLLM):
    """
    Xinference LLM client using the xinference-client SDK.
    Supports chat, streaming, tool calling, and vision capabilities.
    """

    def __init__(
        self,
        model_name: str = "llama-3-8b-instruct",
        model_uid: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        """
        Initialize Xinference LLM client.

        Args:
            model_name: Name of the model (e.g., "llama-3-8b-instruct")
            model_uid: Unique model UID in Xinference (if model is already launched)
            base_url: Xinference server base URL (e.g., "http://localhost:9997")
            api_key: Optional API key for authentication
            default_temperature: Default sampling temperature
            default_max_tokens: Default max tokens for generation
            timeout: Request timeout in seconds
            abilities: List of model abilities (chat, vision, tool_calling, etc.)
            timeout_config: Timeout configuration for streaming
        """
        self._model_name = model_name
        self._model_uid = model_uid or model_name
        self.base_url = (base_url or "http://localhost:9997").rstrip("/")
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

        # Initialize the Xinference client (lazy initialization)
        self._client: Optional[XinferenceClient] = None
        self._model_handle: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this LLM implementation."""
        return self._abilities

    def _ensure_client(self) -> None:
        """Ensure the Xinference client and model handle are initialized."""
        if self._client is None:
            self._client = XinferenceClient(
                base_url=self.base_url, api_key=self.api_key
            )

        if self._model_handle is None:
            # Get the model handle (assumes model is already launched on the server)
            self._model_handle = self._client.get_model(self._model_uid)

    def _build_generate_config(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the generate_config dictionary for Xinference API.

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            generate_config dictionary
        """
        config: Dict[str, Any] = {"stream": stream}

        if temperature is not None:
            config["temperature"] = temperature
        elif self.default_temperature is not None:
            config["temperature"] = self.default_temperature

        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        elif self.default_max_tokens is not None:
            config["max_tokens"] = self.default_max_tokens

        # Add any additional kwargs to the config
        config.update(kwargs)

        return config

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
            response_format: Response format specification (not fully supported by Xinference)
            thinking: Thinking mode configuration (for models that support it)
            **kwargs: Additional parameters to pass to the Xinference API

        Returns:
            - If normal text reply: return dict with type "text" and content
            - If tool call triggered: return dict with type "tool_call" and tool_calls list

        Raises:
            RuntimeError: If the API call fails
        """
        self._ensure_client()
        assert self._model_handle is not None

        # Sanitize messages
        sanitized_messages = self._sanitize_unicode_content(messages)

        # Build generate config
        generate_config = self._build_generate_config(
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )

        # Handle thinking mode
        enable_thinking = None
        if thinking is not None:
            if thinking.get("type") == "enabled" or thinking.get("enable", False):
                enable_thinking = True
            elif thinking.get("type") == "disabled" or not thinking.get(
                "enable", False
            ):
                enable_thinking = False
        elif self.supports_thinking_mode:
            # Auto-enable thinking mode for models that support it
            enable_thinking = True

        try:
            # Make the chat call
            response = self._model_handle.chat(
                messages=sanitized_messages,
                tools=tools,
                enable_thinking=enable_thinking,
                generate_config=generate_config,
            )

            return self._process_chat_response(response)

        except Exception as e:
            logger.error(f"Xinference chat failed: {e}")
            raise RuntimeError(f"Xinference chat failed: {str(e)}") from e

    def _process_chat_response(self, response: Any) -> Dict[str, Any]:
        """Process the chat response from Xinference.

        Args:
            response: Raw response from Xinference

        Returns:
            Processed response dict
        """
        # Xinference returns a dict-like object with various fields
        response_dict = dict(response) if not isinstance(response, dict) else response

        # Record token usage if available
        usage = response_dict.get("usage", {})
        if usage:
            add_token_usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                model=self._model_name,
                call_type="chat",
            )

        # Check for tool calls
        choices = response_dict.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})

            # Check for tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                return {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                    "raw": response_dict,
                }

            # Handle text content
            content = message.get("content", "")
            if content:
                return {
                    "type": "text",
                    "content": content,
                    "raw": response_dict,
                }

        # Fallback: try to get content directly from response
        content = response_dict.get("content", "")
        if content:
            return {
                "type": "text",
                "content": content,
                "raw": response_dict,
            }

        raise RuntimeError(f"Invalid Xinference response: {response_dict}")

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
        Perform a vision-aware chat completion for Xinference models that support vision.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
                      Content can be a string or list of multimodal content items
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            response_format: Response format specification
            thinking: Thinking mode configuration
            **kwargs: Additional parameters

        Returns:
            - If normal text reply: return dict with type "text" and content
            - If tool call triggered: return dict with type "tool_call" and tool_calls list

        Raises:
            RuntimeError: If the model doesn't support vision or the API call fails
        """
        if not self.has_ability("vision"):
            raise RuntimeError(
                f"Model {self._model_name} does not support vision capabilities"
            )

        # Xinference handles vision through the same chat method
        # Just delegate to the regular chat method
        return await self.chat(
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
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
        """
        Stream chat completion from Xinference.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Tool definitions
            tool_choice: Tool choice strategy
            response_format: Response format specification
            thinking: Thinking mode configuration
            **kwargs: Additional parameters

        Yields:
            StreamChunk: Stream chunks

        Raises:
            RuntimeError: If API call fails
            LLMTimeoutError: If timeout occurs
        """
        self._ensure_client()
        assert self._model_handle is not None

        # Sanitize messages
        sanitized_messages = self._sanitize_unicode_content(messages)

        # Build generate config with streaming enabled
        generate_config = self._build_generate_config(
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        # Handle thinking mode
        enable_thinking = None
        if thinking is not None:
            if thinking.get("type") == "enabled" or thinking.get("enable", False):
                enable_thinking = True
            elif thinking.get("type") == "disabled" or not thinking.get(
                "enable", False
            ):
                enable_thinking = False
        elif self.supports_thinking_mode:
            enable_thinking = True

        try:
            # Make the streaming chat call
            stream = self._model_handle.chat(
                messages=sanitized_messages,
                tools=tools,
                enable_thinking=enable_thinking,
                generate_config=generate_config,
            )

            # Timeout control
            first_token = True
            last_token_time = None
            start_time = time.time()

            # Accumulated tool calls across chunks
            accumulated_tool_calls: Dict[str, Dict] = {}

            for chunk in stream:
                current_time = time.time()

                # First token timeout check
                if first_token:
                    elapsed = current_time - start_time
                    if elapsed > self.timeout_config.first_token_timeout:
                        logger.error(f"First token timeout after {elapsed}s")
                        raise LLMTimeoutError(
                            f"First token timeout: {elapsed}s > {self.timeout_config.first_token_timeout}s"
                        )
                    first_token = False
                    logger.debug(f"First token received after {elapsed:.2f}s")

                # Token interval timeout check
                if last_token_time is not None:
                    interval = current_time - last_token_time
                    if interval > self.timeout_config.token_interval_timeout:
                        logger.error(f"Token interval timeout: {interval}s")
                        raise LLMTimeoutError(
                            f"Token interval timeout: {interval}s > {self.timeout_config.token_interval_timeout}s"
                        )

                last_token_time = current_time

                # Parse and yield the chunk
                parsed_chunk = self._parse_stream_chunk(chunk, accumulated_tool_calls)
                if parsed_chunk:
                    yield parsed_chunk

        except LLMTimeoutError:
            raise

        except Exception as e:
            logger.error(f"Xinference stream chat failed: {e}")
            raise RuntimeError(f"Xinference stream chat failed: {str(e)}") from e

    def _parse_stream_chunk(
        self, raw_chunk: Any, accumulated_tool_calls: Dict
    ) -> Optional[StreamChunk]:
        """
        Parse a Xinference stream chunk.

        Args:
            raw_chunk: Raw chunk from Xinference
            accumulated_tool_calls: Accumulated tool calls across chunks

        Returns:
            StreamChunk or None
        """
        chunk_dict = dict(raw_chunk) if not isinstance(raw_chunk, dict) else raw_chunk

        # Check for usage information
        usage = chunk_dict.get("usage")
        if usage:
            add_token_usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                model=self._model_name,
                call_type="stream_chat",
            )

        # Check choices
        choices = chunk_dict.get("choices", [])
        if not choices:
            # No choices, might be a metadata chunk
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Handle token content
        content = delta.get("content", "")
        if content:
            return StreamChunk(
                type=ChunkType.TOKEN,
                content=content,
                delta=content,
                raw=chunk_dict,
            )

        # Handle tool calls
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                call_id = tool_call.get("id")
                if call_id and call_id not in accumulated_tool_calls:
                    accumulated_tool_calls[call_id] = {
                        "id": call_id,
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    }

                if call_id:
                    function = tool_call.get("function", {})
                    if function.get("name"):
                        accumulated_tool_calls[call_id]["function"]["name"] = function[
                            "name"
                        ]
                    if function.get("arguments"):
                        accumulated_tool_calls[call_id]["function"]["arguments"] += (
                            function["arguments"]
                        )

            # Return accumulated tool calls
            tool_calls_list = list(accumulated_tool_calls.values())
            if tool_calls_list:
                return StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=tool_calls_list,
                    raw=chunk_dict,
                )

        # Handle finish reason
        if finish_reason:
            if accumulated_tool_calls:
                return StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=list(accumulated_tool_calls.values()),
                    finish_reason=finish_reason,
                    raw=chunk_dict,
                )

            return StreamChunk(
                type=ChunkType.END,
                finish_reason=finish_reason,
                raw=chunk_dict,
            )

        return None

    @property
    def supports_thinking_mode(self) -> bool:
        """
        Check if this Xinference LLM supports thinking mode.

        Returns:
            bool: True if the model has thinking_mode ability, False otherwise
        """
        return "thinking_mode" in self.abilities

    async def close(self) -> None:
        """Close the Xinference client and cleanup resources."""
        if self._model_handle is not None:
            try:
                self._model_handle.close()
            except Exception:
                pass
            self._model_handle = None

        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    async def __aenter__(self) -> "XinferenceLLM":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @staticmethod
    async def list_available_models(
        base_url: str, api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch available models from Xinference server.

        Args:
            base_url: Xinference server base URL
            api_key: Optional API key for authentication

        Returns:
            List of available models with their information

        Example:
            >>> models = await XinferenceLLM.list_available_models(
            ...     base_url="http://localhost:9997"
            ... )
        """
        client = XinferenceClient(base_url=base_url, api_key=api_key)

        try:
            # Get list of running models
            # list_models returns Dict[str, Dict[str, Any]] where key is model_uid
            models_dict = client.list_models()

            result = []
            for model_uid, model_info in models_dict.items():
                result.append(
                    {
                        "id": model_info.get("model_name", model_uid),
                        "model_uid": model_uid,
                        "model_type": model_info.get("model_type", ""),
                        "model_ability": model_info.get("model_ability", []),
                        "description": model_info.get("model_description", ""),
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Failed to fetch models from Xinference: {e}")
            return []

        finally:
            client.close()
