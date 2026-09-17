"""Xinference LLM provider implementation."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from ..exceptions import LLMTimeoutError
from ..timeout_config import TimeoutConfig
from ..token_context import add_token_usage, extract_cached_input_tokens
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)

_MODEL_DISCOVERY_TIMEOUT_SECONDS = 30.0


async def _await_with_timeout(
    awaitable: Any, *, timeout: float, timeout_message: str
) -> Any:
    if timeout <= 0:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        logger.error(timeout_message)
        raise LLMTimeoutError(timeout_message) from asyncio.TimeoutError()

    async def preserve_transport_timeout() -> Any:
        try:
            return await awaitable
        except asyncio.TimeoutError as exc:
            logger.error("Xinference transport timeout: %s", exc)
            raise LLMTimeoutError("Xinference transport timeout") from exc

    try:
        return await asyncio.wait_for(preserve_transport_timeout(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        logger.error(timeout_message)
        raise LLMTimeoutError(timeout_message) from exc


async def _await_before_deadline(
    awaitable: Any, *, deadline: float, timeout_message: str
) -> Any:
    remaining = deadline - asyncio.get_running_loop().time()
    return await _await_with_timeout(
        awaitable,
        timeout=remaining,
        timeout_message=timeout_message,
    )


def _create_async_client(base_url: str, api_key: Optional[str]) -> Any:
    try:
        from xinference_client.client.restful.async_restful_client import (  # type: ignore
            AsyncClient,
        )
    except ImportError:
        from xinference.client.restful.async_restful_client import AsyncClient

    class NonBlockingAuthProbeAsyncClient(AsyncClient):  # type: ignore[valid-type,misc]
        """Avoid the SDK's synchronous auth probe during construction."""

        def __init__(self, client_base_url: str, client_api_key: Optional[str]) -> None:
            self._api_key_configured = client_api_key is not None
            super().__init__(client_base_url, api_key=client_api_key)

        def _check_cluster_authenticated(self) -> None:
            # The stock AsyncClient performs this probe with requests.get(),
            # which blocks the event loop. Sending the Authorization header
            # whenever an API key is configured works for both authenticated
            # clusters and clusters that ignore authentication.
            self._cluster_authed = self._api_key_configured

    return NonBlockingAuthProbeAsyncClient(base_url, api_key)


def _normalize_model_list_response(
    model_list: Any,
) -> List[tuple[str, dict[str, Any]]]:
    if isinstance(model_list, dict):
        return [
            (str(model_uid), model_info)
            for model_uid, model_info in model_list.items()
            if isinstance(model_info, dict)
        ]

    if isinstance(model_list, list):
        normalized: List[tuple[str, dict[str, Any]]] = []
        for model_info in model_list:
            if not isinstance(model_info, dict):
                continue
            model_uid = str(
                model_info.get("model_uid")
                or model_info.get("id")
                or model_info.get("model_name")
                or ""
            )
            normalized.append((model_uid, model_info))
        return normalized

    return []


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
        self._client: Optional[Any] = None
        self._model_handle: Optional[Any] = None
        self._client_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this LLM implementation."""
        return self._abilities

    async def _ensure_client(self) -> Any:
        """Ensure the Xinference client and model handle are initialized."""
        async with self._client_lock:
            if self._client is None:
                self._client = _create_async_client(self.base_url, self.api_key)

            client = self._client
            if client is None:
                raise RuntimeError("Failed to initialize Xinference client")

            if self._model_handle is None:
                # Get the model handle (assumes model is already launched on the server)
                self._model_handle = await client.get_model(self._model_uid)

            return self._model_handle

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
            LLMTimeoutError: If the request times out
        """
        # Sanitize messages
        sanitized_messages = self._sanitize_unicode_content(
            self._strip_internal_message_keys(messages)
        )

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

        async def call_model() -> Any:
            model_handle = await self._ensure_client()
            return await model_handle.chat(
                messages=sanitized_messages,
                tools=tools,
                enable_thinking=enable_thinking,
                generate_config=generate_config,
            )

        try:
            response = await _await_with_timeout(
                call_model(),
                timeout=self.timeout,
                timeout_message=f"Xinference chat timeout: exceeded {self.timeout}s",
            )

            return self._process_chat_response(response)

        except LLMTimeoutError:
            raise

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
                model_id=self.model_id,
                call_type="chat",
                cached_input_tokens=extract_cached_input_tokens(usage),
            )

        # Check for tool calls
        choices = response_dict.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            reasoning_content = message.get("reasoning_content") or ""
            finish_reason = choice.get("finish_reason")

            # Check for tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                result: Dict[str, Any] = {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                    "raw": response_dict,
                }
                if reasoning_content:
                    result["reasoning_content"] = reasoning_content
                    result["reasoning"] = reasoning_content
                return result

            # Handle text content
            content = message.get("content", "")
            if content:
                result = {
                    "type": "text",
                    "content": content,
                    "raw": response_dict,
                }
                if reasoning_content:
                    result["reasoning_content"] = reasoning_content
                    result["reasoning"] = reasoning_content
                return result

            # Reasoning models (e.g. qwen3-thinking, deepseek-r1) may emit
            # only ``reasoning_content`` and an empty ``content`` when the
            # generation is truncated by ``max_tokens`` (finish_reason="length")
            # before the final answer is produced. Surface the reasoning text
            # as content so callers (notably the model connection test) do
            # not treat a truncated-but-otherwise-healthy response as invalid.
            #
            # Gate strictly on ``finish_reason == "length"`` and require a
            # non-whitespace reasoning trace: any other terminal reason
            # (``"stop"``, ``"content_filter"``, ``None`` …) means the model
            # claims to be done but produced no final answer, which is a
            # real failure that callers must see -- promoting the reasoning
            # trace would silently hide the bug.
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
        # Sanitize messages
        sanitized_messages = self._sanitize_unicode_content(
            self._strip_internal_message_keys(messages)
        )

        # Build generate config with streaming enabled
        stream_options = dict(kwargs.pop("stream_options", {}) or {})
        stream_options["include_usage"] = True
        generate_config = self._build_generate_config(
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options=stream_options,
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

        iterator: Optional[AsyncIterator[Any]] = None
        try:
            loop = asyncio.get_running_loop()
            first_timeout = self.timeout_config.first_token_timeout
            first_deadline = loop.time() + first_timeout
            first_timeout_message = f"First token timeout: exceeded {first_timeout}s"
            model_handle = await _await_before_deadline(
                self._ensure_client(),
                deadline=first_deadline,
                timeout_message=first_timeout_message,
            )
            stream = await _await_before_deadline(
                model_handle.chat(
                    messages=sanitized_messages,
                    tools=tools,
                    enable_thinking=enable_thinking,
                    generate_config=generate_config,
                ),
                deadline=first_deadline,
                timeout_message=first_timeout_message,
            )
            if not hasattr(stream, "__aiter__"):
                raise RuntimeError(
                    "Xinference streaming response is not an async iterator"
                )
            iterator = stream.__aiter__()

            # Accumulated tool calls across chunks
            accumulated_tool_calls: Dict[str, Dict] = {}
            last_usage: Optional[Dict[str, Any]] = None
            usage_emitted = False
            first_payload_seen = False
            next_payload_deadline = first_deadline

            def parse_item(item: Any) -> Optional[StreamChunk]:
                nonlocal last_usage, usage_emitted
                item_dict = dict(item) if not isinstance(item, dict) else item
                usage = item_dict.get("usage")
                if usage:
                    last_usage = dict(usage)
                chunk = self._parse_stream_chunk(item_dict, accumulated_tool_calls)
                if chunk is not None and chunk.is_usage():
                    usage_emitted = True
                return chunk

            while True:
                if first_payload_seen:
                    timeout = self.timeout_config.token_interval_timeout
                    timeout_message = f"Token interval timeout: exceeded {timeout}s"
                else:
                    timeout = first_timeout
                    timeout_message = first_timeout_message
                try:
                    item = await _await_before_deadline(
                        iterator.__anext__(),
                        deadline=next_payload_deadline,
                        timeout_message=timeout_message,
                    )
                except StopAsyncIteration:
                    break

                parsed_chunk = parse_item(item)
                if parsed_chunk:
                    if parsed_chunk.type in {
                        ChunkType.TOKEN,
                        ChunkType.TOOL_CALL,
                        ChunkType.END,
                    }:
                        first_payload_seen = True
                        next_payload_deadline = (
                            loop.time() + self.timeout_config.token_interval_timeout
                        )
                    yield parsed_chunk

            if last_usage and not usage_emitted:
                self._record_stream_usage(last_usage)
                yield StreamChunk(
                    type=ChunkType.USAGE,
                    usage=last_usage,
                    raw={"choices": [], "usage": last_usage},
                )
        except LLMTimeoutError:
            raise

        except Exception as e:
            logger.error(f"Xinference stream chat failed: {e}")
            raise RuntimeError(f"Xinference stream chat failed: {str(e)}") from e

        finally:
            if iterator is not None:
                close_stream = getattr(iterator, "aclose", None)
                if callable(close_stream):
                    await close_stream()

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

        # Check choices
        choices = chunk_dict.get("choices", [])
        if not choices:
            if usage:
                self._record_stream_usage(usage)
                return StreamChunk(
                    type=ChunkType.USAGE,
                    usage=dict(usage),
                    raw=chunk_dict,
                )
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
                index = tool_call.get("index")
                if not call_id and isinstance(index, int):
                    for (
                        existing_id,
                        existing_tool_call,
                    ) in accumulated_tool_calls.items():
                        if existing_tool_call.get("index") == index:
                            call_id = existing_id
                            break
                if not call_id and len(accumulated_tool_calls) == 1:
                    existing_id, existing_tool_call = next(
                        iter(accumulated_tool_calls.items())
                    )
                    existing_index = existing_tool_call.get("index")
                    if not isinstance(index, int) or (
                        isinstance(existing_index, int) and existing_index == index
                    ):
                        call_id = existing_id
                if not call_id:
                    continue

                if call_id and call_id not in accumulated_tool_calls:
                    accumulated_tool_calls[call_id] = {
                        "index": index if isinstance(index, int) else None,
                        "id": call_id,
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    }
                elif isinstance(index, int):
                    accumulated_tool_calls[call_id]["index"] = index

                if tool_call.get("type"):
                    accumulated_tool_calls[call_id]["type"] = tool_call["type"]
                function = tool_call.get("function") or {}
                if function.get("name"):
                    accumulated_tool_calls[call_id]["function"]["name"] = function[
                        "name"
                    ]
                if "arguments" in function:
                    accumulated_tool_calls[call_id]["function"]["arguments"] += (
                        function.get("arguments") or ""
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

    def _record_stream_usage(self, usage: Dict[str, Any]) -> None:
        add_token_usage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=self._model_name,
            model_id=self.model_id,
            call_type="stream_chat",
            cached_input_tokens=extract_cached_input_tokens(usage),
        )

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
        async with self._client_lock:
            if self._model_handle is not None:
                try:
                    await self._model_handle.close()
                except Exception:
                    pass
                self._model_handle = None

            if self._client is not None:
                try:
                    await self._client.close()
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
        # Ensure base_url doesn't have trailing slash
        base_url = base_url.rstrip("/")

        # Map Xinference abilities to Xagent abilities
        ability_mapping = {
            "audio2text": "asr",
            "text2audio": "tts",
            "text2audio_zero_shot": "tts",
            "text2audio_voice_cloning": "tts",
            "chat": "chat",
            "vision": "vision",
            "tool_calling": "tool_calling",
            "embedding": "embedding",
        }

        # Retry logic for transient network issues
        max_retries = 3
        retry_delay = 1.0  # seconds

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Fetching models from Xinference: {base_url} (attempt {attempt + 1}/{max_retries})"
                )

                client = _create_async_client(base_url, api_key)
                try:
                    model_list = await _await_with_timeout(
                        client.list_models(),
                        timeout=_MODEL_DISCOVERY_TIMEOUT_SECONDS,
                        timeout_message=(
                            "Xinference model discovery timeout: exceeded "
                            f"{_MODEL_DISCOVERY_TIMEOUT_SECONDS}s"
                        ),
                    )
                finally:
                    try:
                        await client.close()
                    except Exception:
                        pass
                normalized_models = _normalize_model_list_response(model_list)

                result = []
                for model_uid, model_info in normalized_models:
                    if not model_uid:
                        continue

                    # Map abilities
                    xinference_abilities = model_info.get("model_ability", [])
                    mapped_abilities = []
                    for ability in xinference_abilities:
                        mapped_ability = ability_mapping.get(ability, ability)
                        # Only add core abilities (asr, tts, chat, vision, tool_calling)
                        # Filter out detailed capabilities like text2audio_emotion_control
                        if mapped_ability in [
                            "asr",
                            "tts",
                            "chat",
                            "vision",
                            "tool_calling",
                            "embedding",
                        ]:
                            if mapped_ability not in mapped_abilities:
                                mapped_abilities.append(mapped_ability)

                    result.append(
                        {
                            "id": model_info.get("model_name", model_uid),
                            "model_uid": model_uid,
                            "model_type": model_info.get("model_type", ""),
                            "model_ability": mapped_abilities,
                            "abilities": mapped_abilities,  # Add abilities field for xagent
                            "description": model_info.get("model_description", ""),
                        }
                    )

                logger.info(
                    f"Successfully fetched {len(result)} models from Xinference"
                )
                return result

            except Exception as e:
                # Network or connection error
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Error connecting to Xinference, retrying in {retry_delay}s: {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(
                        f"Failed to connect to Xinference after {max_retries} attempts: {e}"
                    )
                    raise RuntimeError(
                        f"Cannot connect to Xinference server at {base_url}: {e}"
                    ) from e

        # This should never be reached, but mypy needs it
        return []
