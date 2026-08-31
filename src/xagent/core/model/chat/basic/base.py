from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List

from ..response_shape import classify_chat_response
from ..types import ChunkType, StreamChunk


class BaseLLM(ABC):
    """
    Abstract base class for Large Language Models (LLMs) with ability-based support.
    This interface supports different capabilities like chat, vision, and tool calling.

    Implementations must define the supported abilities and implement corresponding methods.
    """

    # Total context window in tokens, set from model config by create_base_llm.
    # None means unknown; consumers fall back to their own default.
    context_window: int | None = None

    # Unique model id, set from model config by create_base_llm. Threaded into
    # token-usage details so identically-named models can be told apart.
    _model_id: str | None = None

    @property
    def model_id(self) -> str:
        """Unique model id (``""`` when unset), for token-usage attribution."""
        return self._model_id or ""

    @property
    @abstractmethod
    def abilities(self) -> List[str]:
        """
        Get the list of abilities supported by this LLM implementation.
        Possible abilities: ["chat", "vision", "video", "thinking_mode", "tool_calling"]

        Returns:
            List[str]: List of supported abilities
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the model name/identifier.

        Returns:
            str: The model name or identifier
        """
        pass

    @property
    @abstractmethod
    def supports_thinking_mode(self) -> bool:
        """
        Check if this LLM implementation supports thinking mode.

        Returns:
            bool: True if the model supports thinking mode, False otherwise
        """
        pass

    @property
    def supports_json_schema_response_format(self) -> bool:
        """
        Check if this LLM supports OpenAI-style json_schema response_format.

        Defaults to True to preserve existing provider behavior. Providers that
        are OpenAI-compatible at the transport layer but do not support
        json_schema, such as DeepSeek, should override this.
        """
        return True

    @property
    def supports_json_object_response_format(self) -> bool:
        """
        Check if this LLM supports JSON object response_format mode.

        Defaults to True to preserve existing structured-output routing.
        """
        return True

    @property
    def supports_native_video_input(self) -> bool:
        """Whether this model accepts a video as one multimodal input.

        Provider adapters override this when their public API has a known video
        contract.  OpenAI-compatible or self-hosted models can opt in explicitly
        by adding ``video`` to their configured abilities.
        """
        return self.has_ability("video")

    @property
    def supports_native_video_with_images(self) -> bool:
        """Whether native video and image parts may share one model request."""
        return False

    @property
    def supports_native_video_time_range(self) -> bool:
        """Whether native video input enforces start/end offsets."""
        return False

    def build_native_video_content(
        self,
        video_url: str,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, Any]:
        """Build the provider-ready content part for one native video input.

        The default is the OpenAI-compatible ``video_url`` shape used by a
        number of third-party providers.  Provider adapters can translate or
        enrich it without leaking provider checks into the vision tool.
        """
        _ = start_time, end_time
        return {"type": "video_url", "video_url": {"url": video_url}}

    def has_ability(self, ability: str) -> bool:
        """
        Check if this LLM implementation supports a specific ability.

        Args:
            ability: The ability to check

        Returns:
            bool: True if the ability is supported, False otherwise
        """
        return ability in self.abilities

    def _strip_internal_message_keys(
        self, messages: List[dict[str, Any]]
    ) -> List[dict[str, Any]]:
        """Remove Xagent-only message metadata before sending provider calls.

        Every provider that forwards a message dict's keys through to its SDK
        call largely unchanged (as opposed to rebuilding a provider-shaped
        message field-by-field) must call this before the request leaves the
        process, or an internal marker like ``_xagent_provider_state`` leaks
        onto the wire.
        """
        sanitized: List[dict[str, Any]] = []
        for message in messages:
            sanitized.append(
                {
                    key: value
                    for key, value in message.items()
                    if not key.startswith("_xagent_")
                }
            )
        return sanitized

    def _sanitize_unicode_content(self, content: Any) -> Any:
        """
        Sanitize content by removing or replacing invalid Unicode characters.

        Args:
            content: Content to sanitize (string, dict, or list)

        Returns:
            Sanitized content with invalid Unicode characters handled
        """
        if isinstance(content, str):
            # Remove or replace invalid Unicode surrogate pairs
            # This handles cases like \ud83d that can't be encoded in UTF-8
            try:
                # First try to encode/decode to catch any encoding issues
                content.encode("utf-8").decode("utf-8")
                return content
            except UnicodeEncodeError:
                # If encoding fails, remove invalid surrogate pairs
                # Pattern matches surrogate pairs: \ud800-\udfff
                sanitized = re.sub(r"[\ud800-\udfff]", "", content)
                return sanitized
            except UnicodeDecodeError:
                # If decoding fails, replace invalid characters
                sanitized = content.encode("utf-8", errors="replace").decode("utf-8")
                return sanitized
        elif isinstance(content, dict):
            # Recursively sanitize dictionary values
            return {
                key: self._sanitize_unicode_content(value)
                for key, value in content.items()
            }
        elif isinstance(content, list):
            # Recursively sanitize list items
            return [self._sanitize_unicode_content(item) for item in content]
        else:
            # Return as-is for other types
            return content

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        """
        Generate a chat completion from the model given the conversation history.

        Args:
            messages: A list of chat messages in the OpenAI format.
                      Each message must have a "role" ("user", "system", "assistant") and "content".
            temperature: Sampling temperature for generation (e.g. 0.7).
            max_tokens: Maximum number of tokens to generate.
            tools: Optional list of tools (functions) described in OpenAI function calling format.
            tool_choice: Specifies which tool to use.
                         - "auto": let the model decide.
                         - "none": disable tool calling.
                         - string: enforce a specific tool name.
                         - dict: force a structured tool_call selection.
            response_format: Optional response format specification (e.g., {"type": "json_object"}).
            thinking: Optional thinking mode configuration (e.g., {"type": "disabled"}).
            output_config: Optional output configuration for structured outputs (e.g., {"format": {"type": "json_schema", "schema": {...}}}).
            **kwargs: Additional parameters specific to the underlying model (e.g. top_p, user, stop).

        Returns:
            If the model returns a natural language response:
                -> dict envelope with fields:
                    - "type": "text"
                    - "content": the assistant reply content
                    - "usage": top-level provider usage stamp, when the provider
                      reported one (e.g. {"prompt_tokens": ..., "completion_tokens": ...})
                    - "raw": the provider's full response payload. Present on
                      some implementations' envelopes and absent on others,
                      so callers must not require it.
                (Legacy adapters may still return a plain string reply.)

            If the model triggers a tool call:
                -> dict envelope with fields:
                    - "type": "tool_call"
                    - "tool_calls": list of tool call objects
                    - "raw": the provider's full response payload -- optional,
                      provider-dependent (e.g. Gemini's envelope omits it);
                      callers must not require it
                    - "usage": top-level provider usage stamp, when reported

            Consumers needing the reply text must classify the envelope
            structurally (``classify_chat_response`` / ``unwrap_chat_text``)
            rather than stringifying it.

        Raises:
            RuntimeError if the model call fails or returns an unexpected format.
        """
        pass

    async def vision_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        """
        Generate a vision-aware chat completion from the model given the conversation history.
        This method supports multimodal inputs including images.

        Args:
            messages: A list of chat messages in the OpenAI format.
                      Each message must have a "role" ("user", "system", "assistant") and "content".
                      For vision support, content can be a list containing text and image objects.
            temperature: Sampling temperature for generation (e.g. 0.7).
            max_tokens: Maximum number of tokens to generate.
            tools: Optional list of tools (functions) described in OpenAI function calling format.
            tool_choice: Specifies which tool to use.
            response_format: Optional response format specification (e.g., {"type": "json_object"}).
            thinking: Optional thinking mode configuration.
            output_config: Optional output configuration for structured outputs.
            **kwargs: Additional parameters specific to the underlying model.

        Returns:
            Same envelope contract as ``chat()``: a text envelope
            (``{"type": "text", "content": ...}``, optionally with ``usage``
            and ``raw``) for a natural language response, a tool_call
            envelope (``{"type": "tool_call", "tool_calls": [...]}``) when
            the model triggers a tool call, or -- for legacy adapters -- a
            plain string reply. Never stringify the envelope to get the
            text; use ``classify_chat_response`` / ``unwrap_chat_text``.

        Raises:
            RuntimeError if the model doesn't support vision or the call fails.
        """
        if not self.has_ability("vision"):
            raise RuntimeError(
                f"Model {self.__class__.__name__} does not support vision capabilities"
            )

        # Default implementation delegates to chat method
        # Override in vision-capable implementations
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

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream LLM calls (optional implementation)

        Default implementation: uses chat() to return a single chunk
        Subclasses can override this method to provide true streaming response, supporting:
        - Real-time token output
        - More flexible timeout control (first token timeout, token interval timeout)
        - Precise token statistics

        Args:
            messages: Chat message list
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: Tool list
            tool_choice: Tool selection strategy
            response_format: Response format
            thinking: Thinking mode configuration
            output_config: Output configuration for structured outputs
            **kwargs: Other parameters

        Yields:
            StreamChunk: Streaming response chunk

        Raises:
            RuntimeError: If call fails
        """
        # Default implementation: uses non-streaming chat and returns single chunk
        # Subclasses can override this method to provide true streaming response
        result = await self.chat(
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

        if result is None:
            # None response - treat as error
            yield StreamChunk(
                type=ChunkType.ERROR,
                content="LLM returned None response",
            )
            return

        # Shape-discriminate via the shared classifier: a text envelope or
        # legacy plain string is a token chunk, a tool_call envelope is a
        # tool-call chunk, and anything else is an explicit error -- never a
        # silent empty TOOL_CALL chunk (the pre-contract fallback treated
        # every non-string as a tool call). Usage-bearing envelopes also
        # emit a USAGE chunk so the chat-to-stream adaptation keeps the
        # billing metadata the runtime reads from streams.
        shape = classify_chat_response(result)
        usage_payload = result.get("usage") if isinstance(result, dict) else None
        if shape.kind == "text":
            assert shape.text is not None  # classifier invariant
            yield StreamChunk(
                type=ChunkType.TOKEN,
                content=shape.text,
                delta=shape.text,
                raw=result,
            )
        elif shape.kind == "tool_call" and isinstance(result, dict):
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=result.get("tool_calls", []),
                raw=result,
            )
        else:
            yield StreamChunk(
                type=ChunkType.ERROR,
                content=(
                    "LLM returned an unusable chat response shape "
                    f"({shape.kind}); expected a text or tool_call envelope"
                ),
                raw=result,
            )
            return

        if isinstance(usage_payload, dict) and usage_payload:
            yield StreamChunk(
                type=ChunkType.USAGE,
                usage=dict(usage_payload),
                raw=result,
            )
