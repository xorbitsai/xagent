"""Gemini LLM implementation using official Google Generative AI SDK."""

import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from google import genai  # type: ignore[import-untyped]
from google.genai import errors as genai_errors  # type: ignore[import-untyped]

from ....utils.security import redact_sensitive_text
from ..exceptions import (
    LLMEmptyContentError,
    LLMInvalidResponseError,
    LLMRetryableError,
    LLMTimeoutError,
)
from ..timeout_config import TimeoutConfig
from ..token_context import add_token_usage
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)


def _contains_refs_or_defs(obj: Any) -> bool:
    """Recursively check if object contains $defs or $ref keys."""
    if isinstance(obj, dict):
        if "$ref" in obj or "$defs" in obj or "definitions" in obj:
            return True
        return any(_contains_refs_or_defs(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(_contains_refs_or_defs(item) for item in obj)
    return False


def _flatten_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a JSON schema by resolving $ref references and removing $defs."""
    if not _contains_refs_or_defs(schema):
        return schema

    schema = json.loads(json.dumps(schema))

    definitions: Dict[str, Any] = schema.get("$defs", {})
    definitions.update(schema.get("definitions", {}))

    def resolve_refs_and_filter(obj: Any, visited: Optional[set] = None) -> Any:
        if visited is None:
            visited = set()
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                if ref_path in visited:
                    return {"type": "string"}
                visited.add(ref_path)

                # Handle different reference formats
                if ref_path.startswith("#/$defs/") or ref_path.startswith(
                    "#/definitions/"
                ):
                    def_name = ref_path.split("/")[-1]
                    if def_name in definitions:
                        return resolve_refs_and_filter(definitions[def_name], visited)
                return {"type": "string"}

            result: Dict[str, Any] = {}
            for key, value in obj.items():
                if key not in ["$defs", "definitions"]:
                    result[key] = resolve_refs_and_filter(value, visited)
            return result
        elif isinstance(obj, list):
            return [resolve_refs_and_filter(item, visited) for item in obj]
        return obj

    flattened: Dict[str, Any] = resolve_refs_and_filter(schema)
    return flattened


class GeminiLLM(BaseLLM):
    """Google Gemini LLM client using the official Google Generative AI SDK."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
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
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
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
            self._abilities = ["chat", "tool_calling"]
            if any(
                vision_keyword in model_name.lower()
                for vision_keyword in ["vision", "pro-vision", "flash-vision", "2.5"]
            ):
                self._abilities.append("vision")

        # Initialize the Gemini client (lazy initialization)
        self._client: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this Gemini LLM implementation."""
        return self._abilities

    def _ensure_client(self) -> None:
        """Ensure the Gemini client is initialized using official SDK."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY must be set")

            try:
                # Prepare HTTP options for custom base URL if configured
                http_options = None
                if self.base_url:
                    # Remove trailing version paths (/v1beta, /v1) as SDK adds them automatically
                    base_url = self.base_url.rstrip("/")
                    for version_path in ["/v1beta", "/v1"]:
                        if base_url.endswith(version_path):
                            base_url = base_url[: -len(version_path)]
                            logger.info(
                                f"Removed version path from base_url: {self.base_url} -> {base_url}"
                            )
                            break

                    http_options = genai.types.HttpOptions(
                        base_url=base_url,
                        timeout=int(self.timeout * 1000),
                    )
                    logger.info(f"Using custom base URL: {base_url}")

                # Initialize the official Google Generative AI client
                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=http_options,
                )
                logger.info("Initialized Gemini using official SDK")

            except ImportError as e:
                logger.error(f"Google Generative AI SDK not available: {e}")
                raise RuntimeError(
                    "google-genai package is required. Install it with: pip install google-genai"
                ) from e
            except Exception as e:
                logger.error(f"Failed to initialize Gemini SDK: {e}")
                raise

    def _convert_messages_to_gemini_format(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert OpenAI format messages to Gemini format."""
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                if isinstance(content, str):
                    system_instruction = content
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    system_instruction = "\n".join(text_parts)
                continue

            gemini_role = "user" if role == "user" else "model"

            if isinstance(content, str):
                gemini_messages.append({"role": gemini_role, "parts": [content]})
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url

                            if url.startswith("data:image"):
                                try:
                                    mime_type = url.split(":")[1].split(";")[0]
                                    base64_data = url.split(",", 1)[1]
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": mime_type,
                                                "data": base64_data,
                                            }
                                        }
                                    )
                                except (IndexError, ValueError):
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": "image/jpeg",
                                                "data": url,
                                            }
                                        }
                                    )
                            else:
                                parts.append(
                                    {
                                        "inline_data": {
                                            "mime_type": "image/jpeg",
                                            "data": url,
                                        }
                                    }
                                )

                gemini_messages.append({"role": gemini_role, "parts": parts})

        return system_instruction, gemini_messages

    def _convert_tools_to_gemini_format(
        self, tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert OpenAI format tools to Gemini format."""
        gemini_tools: Dict[str, Any] = {"function_declarations": []}

        for tool in tools:
            function = tool.get("function", {})
            name = function.get("name", "")
            description = function.get("description", "")
            parameters = function.get("parameters", {})

            # Flatten JSON schema if it contains $defs or $ref
            if _contains_refs_or_defs(parameters):
                parameters = _flatten_json_schema(parameters)

            gemini_tools["function_declarations"].append(
                {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            )

        return gemini_tools

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
        """Perform a chat completion using official Google Generative AI SDK."""
        self._ensure_client()
        assert self._client is not None

        try:
            system_instruction, gemini_messages = (
                self._convert_messages_to_gemini_format(messages)
            )

            config: Dict[str, Any] = {}
            if temperature is not None:
                config["temperature"] = temperature
            elif self.default_temperature is not None:
                config["temperature"] = self.default_temperature

            if max_tokens is not None:
                config["max_output_tokens"] = max_tokens
            elif self.default_max_tokens is not None:
                config["max_output_tokens"] = self.default_max_tokens

            if response_format:
                response_type: str = response_format.get("type", "")
                if response_type == "json_object":
                    config["response_mime_type"] = "application/json"

            if output_config is not None:
                format_config = output_config.get("format", {})
                if format_config.get("type") == "json_schema":
                    schema = format_config.get("schema") or format_config.get(
                        "json_schema", {}
                    )
                    if schema:
                        config["response_mime_type"] = "application/json"
                        config["response_json_schema"] = schema

            # Combine all messages into contents for SDK
            contents = []
            if system_instruction:
                contents.append(system_instruction)

            for msg in gemini_messages:
                parts = msg.get("parts", [])
                if isinstance(parts, list):
                    contents.extend(parts)
                else:
                    contents.append(parts)

            # Make API call using SDK (async)
            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config if config else None,
            )

            # Extract token usage
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_metadata = response.usage_metadata
                input_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(usage_metadata, "candidates_token_count", 0)

                if input_tokens > 0 or output_tokens > 0:
                    add_token_usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=self._model_name,
                        call_type="chat",
                    )
            else:
                logger.warning("No usage_metadata in Gemini SDK response")

            # Extract response content
            if not response.candidates or len(response.candidates) == 0:
                raise LLMInvalidResponseError("No candidates in Gemini SDK response")

            candidate = response.candidates[0]

            # Check if candidate.content exists
            if not hasattr(candidate, "content") or candidate.content is None:
                raise LLMInvalidResponseError(
                    "Candidate content is None in Gemini SDK response"
                )

            content_parts = candidate.content.parts

            # Check if content_parts is None
            if content_parts is None:
                raise LLMInvalidResponseError(
                    "Content parts is None in Gemini SDK response"
                )

            tool_calls = []
            text_parts = []

            for part in content_parts:
                if hasattr(part, "function_call") and part.function_call:
                    func_call = part.function_call
                    func_name = func_call.name
                    func_args = func_call.args

                    tool_calls.append(
                        {
                            "id": f"call_{uuid.uuid4().hex[:16]}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(func_args),
                            },
                        }
                    )
                elif hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

            if tool_calls:
                return {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                }

            content = "".join(text_parts).strip()

            if not content:
                raise LLMEmptyContentError(
                    "LLM returned empty content and no tool calls"
                )

            return content

        except Exception as e:
            logger.error("Gemini SDK API error: %s", redact_sensitive_text(str(e)))

            # Check for timeout and network errors (string-based check for non-SDK exceptions)
            if "timeout" in str(e).lower() or "network" in str(e).lower():
                raise LLMRetryableError(str(e)) from e

            # Check for Google SDK specific errors
            if isinstance(e, genai_errors.ClientError):
                # 429: Rate limit (RESOURCE_EXHAUSTED)
                # 500, 502, 503: Server errors (retryable)
                if e.code in [429, 500, 502, 503]:
                    raise LLMRetryableError(
                        f"Gemini API error (code={e.code}): {str(e)}"
                    ) from e

            # Fallback: string-based check for rate limit/quota (for compatibility)
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                raise LLMRetryableError(str(e)) from e

            raise RuntimeError(f"Gemini SDK API error: {str(e)}") from e

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
        """Stream chat completion using official Google Generative AI SDK."""
        import time

        first_token = True
        last_token_time = None
        start_time = time.time()

        current_content = ""
        usage_received = False

        try:
            self._ensure_client()  # ← 这是关键！
            assert self._client is not None

            system_instruction, gemini_messages = (
                self._convert_messages_to_gemini_format(messages)
            )

            config: Dict[str, Any] = {}
            if temperature is not None:
                config["temperature"] = temperature
            elif self.default_temperature is not None:
                config["temperature"] = self.default_temperature

            if max_tokens is not None:
                config["max_output_tokens"] = max_tokens
            elif self.default_max_tokens is not None:
                config["max_output_tokens"] = self.default_max_tokens

            if response_format:
                response_type: str = response_format.get("type", "")
                if response_type == "json_object":
                    config["response_mime_type"] = "application/json"

            if output_config is not None:
                format_config = output_config.get("format", {})
                if format_config.get("type") == "json_schema":
                    schema = format_config.get("schema") or format_config.get(
                        "json_schema", {}
                    )
                    if schema:
                        config["response_mime_type"] = "application/json"
                        config["response_json_schema"] = schema

            # Combine all messages into contents for SDK
            contents = []
            if system_instruction:
                contents.append(system_instruction)

            for msg in gemini_messages:
                parts = msg.get("parts", [])
                if isinstance(parts, list):
                    contents.extend(parts)
                else:
                    contents.append(parts)

            # Make streaming API call using SDK (async)
            response_stream = await self._client.aio.models.generate_content_stream(
                model=self._model_name,
                contents=contents,
                config=config if config else None,
            )

            # Check if response_stream is None
            if response_stream is None:
                raise RuntimeError("Gemini SDK returned None response stream")

            # Process streaming response (async iteration)
            async for chunk in response_stream:
                current_time = time.time()

                if first_token:
                    elapsed = current_time - start_time
                    if elapsed > self.timeout_config.first_token_timeout:
                        raise LLMTimeoutError(
                            f"First token timeout: {elapsed:.2f}s > "
                            f"{self.timeout_config.first_token_timeout}s"
                        )
                    first_token = False
                    logger.debug(f"First token received after {elapsed:.2f}s")

                if last_token_time is not None:
                    interval = current_time - last_token_time
                    if interval > self.timeout_config.token_interval_timeout:
                        raise LLMTimeoutError(
                            f"Token interval timeout: {interval:.2f}s > "
                            f"{self.timeout_config.token_interval_timeout}s"
                        )

                last_token_time = current_time

                # Check for usage metadata
                if (
                    hasattr(chunk, "usage_metadata")
                    and chunk.usage_metadata
                    and not usage_received
                ):
                    usage_metadata = chunk.usage_metadata
                    input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens = (
                        getattr(usage_metadata, "candidates_token_count", 0) or 0
                    )

                    if input_tokens > 0 or output_tokens > 0:
                        usage_received = True

                        add_token_usage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            model=self._model_name,
                            call_type="stream_chat",
                        )

                        yield StreamChunk(
                            type=ChunkType.USAGE,
                            usage={
                                "prompt_tokens": input_tokens,
                                "completion_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens,
                            },
                            raw=chunk,
                        )

                # Extract content
                if not chunk.candidates or len(chunk.candidates) == 0:
                    continue

                candidate = chunk.candidates[0]

                # Check if candidate.content exists
                if not hasattr(candidate, "content") or candidate.content is None:
                    continue

                content_parts = candidate.content.parts

                # Check if content_parts is None
                if content_parts is None:
                    continue

                for part in content_parts:
                    if hasattr(part, "function_call") and part.function_call:
                        func_call = part.function_call
                        func_name = func_call.name
                        func_args = func_call.args

                        yield StreamChunk(
                            type=ChunkType.TOOL_CALL,
                            tool_calls=[
                                {
                                    "id": f"call_{uuid.uuid4().hex[:16]}",
                                    "type": "function",
                                    "function": {
                                        "name": func_name,
                                        "arguments": json.dumps(func_args),
                                    },
                                }
                            ],
                            finish_reason="tool_calls",
                            raw=chunk,
                        )

                    elif hasattr(part, "text") and part.text:
                        text = part.text
                        current_content += text

                        yield StreamChunk(
                            type=ChunkType.TOKEN,
                            content=current_content,
                            delta=text,
                            raw=chunk,
                        )

            # Yield end chunk
            yield StreamChunk(
                type=ChunkType.END,
                finish_reason="stop",
                raw=None,
            )

        except LLMTimeoutError:
            raise
        except Exception as e:
            logger.error(
                "Gemini SDK streaming error: %s", redact_sensitive_text(str(e))
            )

            # Check for timeout and network errors (string-based check for non-SDK exceptions)
            if "timeout" in str(e).lower() or "network" in str(e).lower():
                raise LLMRetryableError(str(e)) from e

            # Check for Google SDK specific errors
            if isinstance(e, genai_errors.ClientError):
                # 429: Rate limit (RESOURCE_EXHAUSTED)
                # 500, 502, 503: Server errors (retryable)
                if e.code in [429, 500, 502, 503]:
                    raise LLMRetryableError(
                        f"Gemini API error (code={e.code}): {str(e)}"
                    ) from e

            # Fallback: string-based check for rate limit/quota (for compatibility)
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                raise LLMRetryableError(str(e)) from e

            raise RuntimeError(f"Gemini SDK streaming error: {str(e)}") from e

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
        """Vision chat is same as regular chat for Gemini (supports multimodal by default)."""
        self._ensure_client()  # Ensure client is initialized
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

    @staticmethod
    async def list_available_models(
        api_key: str, base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch available models from Google Gemini API using SDK.

        Args:
            api_key: Google API key for Gemini
            base_url: Base URL for the API (optional).
                - If not provided, uses official Google Generative AI API
                - If provided, uses the specified endpoint (e.g., proxy or custom service)

        Returns:
            List of available Gemini models with their information

        Example:
            >>> # Use official Google API
            >>> models = await GeminiLLM.list_available_models("YOUR_API_KEY")
            >>> # Use custom endpoint/proxy
            >>> models = await GeminiLLM.list_available_models(
            ...     "YOUR_API_KEY",
            ...     base_url="https://my-proxy.com/v1beta"
            ... )
        """
        try:
            # Prepare HTTP options for custom base URL if configured
            http_options = None
            if base_url:
                # Remove trailing version paths (/v1beta, /v1) as SDK adds them automatically
                clean_base_url = base_url.rstrip("/")
                for version_path in ["/v1beta", "/v1"]:
                    if clean_base_url.endswith(version_path):
                        clean_base_url = clean_base_url[: -len(version_path)]
                        break

                http_options = genai.types.HttpOptions(
                    base_url=clean_base_url,
                    timeout=30000,
                )

            # Initialize client with API key and optional custom base URL
            client = genai.Client(
                api_key=api_key,
                http_options=http_options,
            )

            # Use SDK to list models (async)
            models_pager = await client.aio.models.list()
            models = []

            async for model in models_pager:
                models.append(
                    {
                        "id": model.name,
                        "name": model.display_name,
                        "description": model.description,
                        "base_url": base_url,
                    }
                )

            return models

        except Exception as e:
            logger.error(f"Failed to fetch Gemini models: {e}")
            return []

    async def close(self) -> None:
        """Close the Gemini client and cleanup resources."""
        self._client = None

    async def __aenter__(self) -> "GeminiLLM":
        """Async context manager entry."""
        self._ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def supports_thinking_mode(self) -> bool:
        """Check if this Gemini LLM supports thinking mode."""
        return "thinking_mode" in self.abilities
