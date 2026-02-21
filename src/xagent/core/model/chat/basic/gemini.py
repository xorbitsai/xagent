import copy
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union, cast

import httpx

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
    """Recursively check if object contains $defs or $ref keys.

    This is a structural check that avoids false positives from string values.
    """
    if isinstance(obj, dict):
        # Check if this dict has $ref or is a $defs/definitions container
        if "$ref" in obj or "$defs" in obj or "definitions" in obj:
            return True
        # Recursively check all values
        return any(_contains_refs_or_defs(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(_contains_refs_or_defs(item) for item in obj)
    return False


def _flatten_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a JSON schema by resolving $ref references and removing $defs.

    Gemini API doesn't support $defs or $ref in JSON schemas.
    This function inlines all referenced definitions and removes unsupported fields.

    Args:
        schema: The JSON schema that may contain $defs and $ref

    Returns:
        A flattened JSON schema with all $ref resolved, $defs removed, and unsupported fields filtered
    """
    # JSON Schema keywords that Gemini doesn't support
    # https://ai.google.dev/gemini-api/docs/json-mode
    GEMINI_UNSUPPORTED_KEYWORDS = {
        "$id",
        "$schema",
        "$defs",
        "definitions",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "patternProperties",
        "additionalProperties",
        "minProperties",
        "maxProperties",
        "const",
        "contains",
        "propertyNames",
        "if",
        "then",
        "else",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
    }

    # Extract definitions if present
    definitions = schema.get("$defs", {})
    # Also check for the older "definitions" key
    definitions.update(schema.get("definitions", {}))

    def resolve_refs_and_filter(obj: Any, visited: Optional[set] = None) -> Any:
        """Recursively resolve $ref references and filter unsupported fields."""
        if visited is None:
            visited = set()

        # Handle dict
        if isinstance(obj, dict):
            # Handle $ref
            if "$ref" in obj:
                ref_path = obj["$ref"]

                # Check for circular references
                if ref_path in visited:
                    logger.warning(f"Circular reference detected: {ref_path}")
                    return {"type": "string"}  # Fallback to string type

                # Parse the reference path (e.g., "#/$defs/MyType" or "#/definitions/MyType")
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("#/$defs/")[1]
                elif ref_path.startswith("#/definitions/"):
                    def_name = ref_path.split("#/definitions/")[1]
                elif ref_path.startswith("#/"):
                    # Generic JSON pointer reference
                    def_name = ref_path[2:]
                else:
                    logger.warning(f"Unsupported $ref format: {ref_path}")
                    return {"type": "string"}

                # Look up the definition
                if def_name not in definitions:
                    logger.warning(f"Definition not found: {def_name}")
                    return {"type": "string"}

                # Deep copy the definition to avoid modifying the original
                def_value = copy.deepcopy(definitions[def_name])

                # Recursively resolve references in the definition
                visited.add(ref_path)
                resolved = resolve_refs_and_filter(def_value, visited)
                visited.remove(ref_path)

                return resolved

            # Recursively process all values in the dict, filtering unsupported keys
            result = {}
            for key, value in obj.items():
                # Skip unsupported keywords
                if key in GEMINI_UNSUPPORTED_KEYWORDS:
                    logger.debug(
                        f"Filtering unsupported Gemini JSON Schema keyword: {key}"
                    )
                    continue

                # Recursively process the value
                result[key] = resolve_refs_and_filter(value, visited)

            return result

        # Handle list - recursively process each item
        elif isinstance(obj, list):
            return [resolve_refs_and_filter(item, visited) for item in obj]

        # Handle other types (string, number, bool, None)
        else:
            return obj

    if not isinstance(schema, dict):
        return schema

    # First, resolve all references and filter unsupported fields
    flattened = resolve_refs_and_filter(schema)

    # Ensure the result is a dict (for type checking)
    if not isinstance(flattened, dict):
        flattened = {}

    # Remove any remaining unsupported keys
    for key in GEMINI_UNSUPPORTED_KEYWORDS:
        if key in flattened:
            del flattened[key]

    return cast(Dict[str, Any], flattened)


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM client using the official Google Generative AI SDK.
    Supports Gemini models with tool calling and vision capabilities.
    """

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
            # Auto-detect abilities based on model name
            self._abilities = ["chat", "tool_calling"]
            if any(
                vision_keyword in model_name.lower()
                for vision_keyword in ["vision", "pro-vision", "flash-vision", "2.5"]
            ):
                self._abilities.append("vision")

        # Initialize the Gemini client
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
        """Ensure the Gemini client is initialized."""
        if self._client is None:
            # Configure the API key
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY must be set")

            # Use REST API directly (works with both proxy and official Google API)
            self._use_rest_api = True

            # Mark as initialized
            self._client = "rest_api"

    def _convert_messages_to_gemini_format(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert OpenAI format messages to Gemini format.

        Args:
            messages: List of messages in OpenAI format

        Returns:
            List of messages in Gemini format
        """
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Handle system message - Gemini uses separate system_instruction
            if role == "system":
                if isinstance(content, str):
                    system_instruction = content
                elif isinstance(content, list):
                    # Extract text from content list
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    system_instruction = "\n".join(text_parts)
                continue

            # Convert roles
            gemini_role = "user" if role == "user" else "model"

            # Handle content
            if isinstance(content, str):
                gemini_messages.append({"role": gemini_role, "parts": [content]})
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Handle image
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url

                            # For base64 data URLs
                            if url.startswith("data:image"):
                                # Extract mime type from data URL
                                # Format: data:image/jpeg;base64,<base64_string>
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
                                    logger.warning(
                                        f"Invalid data URL format: {url[:50]}..."
                                    )
                                    # Fallback to jpeg if we can't parse
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": "image/jpeg",
                                                "data": url,
                                            }
                                        }
                                    )
                            else:
                                # For regular URLs, we'd need to fetch the image
                                # For now, just pass the URL
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
        """
        Convert OpenAI format tools to Gemini format.

        Args:
            tools: List of tools in OpenAI format

        Returns:
            Tool declaration in Gemini REST API format with function_declarations wrapper
        """
        function_declarations = []

        for tool in tools:
            function = tool.get("function", {})
            name = function.get("name", "")
            description = function.get("description", "")
            parameters = function.get("parameters", {})

            # Flatten the JSON schema to remove $defs and resolve $ref
            # Gemini API doesn't support these JSON Schema keywords
            parameters = _flatten_json_schema(parameters)

            # Assert no $defs or $ref remain - fail fast if flattening didn't work
            # Use structural check instead of string-based check to avoid false positives
            if _contains_refs_or_defs(parameters):
                raise ValueError(
                    f"Tool '{name}' parameters still contain $defs or $ref after flattening. "
                    f"This is not supported by Gemini API. Parameters: {json.dumps(parameters, indent=2)}"
                )

            # Convert to Gemini function declaration format (dict for REST API)
            gemini_function = {
                "name": name,
                "description": description,
                "parameters": parameters,
            }

            function_declarations.append(gemini_function)

        # Wrap in function_declarations as per Gemini API spec
        return {"function_declarations": function_declarations}

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
            tool_choice: Tool choice strategy
            response_format: Response format specification (e.g., {"type": "json_object"})
            thinking: Thinking mode configuration (not supported by Gemini)
            output_config: Output configuration for structured outputs
            **kwargs: Additional parameters to pass to the Gemini API

        Returns:
            - If normal text reply: return string
            - If tool call triggered: return dict with type "tool_call" and tool_calls list

        Raises:
            RuntimeError: If the API call fails
        """
        self._ensure_client()
        assert self._client is not None

        try:
            # Convert messages to Gemini format
            system_instruction, gemini_messages = (
                self._convert_messages_to_gemini_format(messages)
            )

            # Build generation config
            gen_config: Dict[str, Any] = {}
            if temperature is not None:
                gen_config["temperature"] = temperature
            elif self.default_temperature is not None:
                gen_config["temperature"] = self.default_temperature

            if max_tokens is not None:
                gen_config["max_output_tokens"] = max_tokens
            elif self.default_max_tokens is not None:
                gen_config["max_output_tokens"] = self.default_max_tokens

            # Handle output_config for JSON schema (Gemini 3.0+)
            if output_config is not None:
                format_config = output_config.get("format", {})
                if format_config.get("type") == "json_schema":
                    # Gemini uses response_mime_type and response_json_schema
                    schema = format_config.get("schema") or format_config.get(
                        "json_schema", {}
                    )
                    if schema:
                        gen_config["response_mime_type"] = "application/json"
                        gen_config["response_json_schema"] = schema

            # Handle response_format for JSON mode (legacy)
            # Gemini uses "response_mime_type" instead of "response_format"
            if response_format:
                response_type: str = response_format.get("type", "")
                if response_type == "json_object":
                    # Only set if not already set by output_config
                    if "response_mime_type" not in gen_config:
                        gen_config["response_mime_type"] = "application/json"

            # Handle tools
            gemini_tools = None
            if tools:
                gemini_tools = self._convert_tools_to_gemini_format(tools)

            # Make the API call
            response = await self._generate_content(
                system_instruction=system_instruction,
                messages=gemini_messages,
                generation_config=gen_config,
                tools=gemini_tools,
            )

            # Extract token usage from response
            usage_metadata = response.get("usageMetadata", {})
            input_tokens = usage_metadata.get("promptTokenCount", 0)
            output_tokens = usage_metadata.get("candidatesTokenCount", 0)

            # Record token usage to tracker
            if input_tokens > 0 or output_tokens > 0:
                add_token_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model_name,
                    call_type="chat",
                )

            # Extract the response text and tool calls
            # REST API returns JSON dict with candidates array
            if isinstance(response, dict):
                candidates = response.get("candidates", [])
                if not candidates:
                    raise LLMInvalidResponseError("No candidates in response")
                result = candidates[0].get("content", {}).get("parts", [])
            else:
                # Fallback for unexpected format
                raise LLMInvalidResponseError(
                    f"Unexpected response format: {type(response)}"
                )

            # Check for function calls
            tool_calls = []
            text_parts = []

            for part in result:
                # REST API returns dict format
                if isinstance(part, dict):
                    # Check for function call
                    if "functionCall" in part:
                        func_call = part["functionCall"]
                        func_name = func_call.get("name", "")
                        func_args = func_call.get("args", {})

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
                    # Check for text
                    elif "text" in part:
                        text_parts.append(part["text"])

            if tool_calls:
                return {
                    "type": "tool_call",
                    "tool_calls": tool_calls,
                    "raw": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {"text": "".join(text_parts)}
                                        if text_parts
                                        else {},
                                    ]
                                }
                            }
                        ]
                    },
                }

            # Return text content
            content = "".join(text_parts).strip()

            if not content:
                raise LLMEmptyContentError(
                    "LLM returned empty content and no tool calls"
                )

            return content

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            # Re-raise LLMRetryableError as-is (will be caught by retry wrapper)
            # Wrap other errors in RuntimeError
            if isinstance(e, LLMRetryableError):
                raise

            if isinstance(e, (httpx.TimeoutException, httpx.NetworkError)):
                raise LLMRetryableError(str(e)) from e

            if isinstance(e, httpx.HTTPStatusError):
                # Retry on 400, 403, 422, 429 and 5xx
                # These errors from proxy services can be transient
                status = e.response.status_code
                if (
                    status == 400
                    or status == 403
                    or status == 422
                    or status == 429
                    or (500 <= status < 600)
                ):
                    raise LLMRetryableError(str(e)) from e

            raise RuntimeError(f"Gemini API error: {str(e)}") from e

    async def _call_gemini_rest_api(
        self,
        system_instruction: Optional[str],
        contents: List[Dict[str, Any]],
        generation_config: Dict[str, Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call Gemini REST API directly using async HTTP client (works with both proxy and official Google API)."""
        # Build the API URL
        base_url = (
            self.base_url.rstrip("/")
            if self.base_url
            else "https://generativelanguage.googleapis.com/v1beta"
        )
        model_name = self._model_name

        # Determine authentication method based on whether using proxy or official API
        # Official Google API uses ?key= in URL
        # Proxy services use Authorization header
        is_official_api = "googleapis.com" in base_url

        if is_official_api:
            api_url = (
                f"{base_url}/models/{model_name}:generateContent?key={self.api_key}"
            )
            headers = {}
        else:
            # For proxy services, try different URL patterns
            if "/v1beta" in base_url or "/v1" in base_url:
                # Base URL already includes the version path
                api_url = f"{base_url}/models/{model_name}:generateContent"
            else:
                # Add version path for other proxies
                api_url = f"{base_url}/v1beta/models/{model_name}:generateContent"
            headers = {"Authorization": f"Bearer {self.api_key}"}

        # Prepare request body
        request_body: Dict[str, Any] = {"contents": contents}

        # Add system instruction if provided
        # systemInstruction must be an object with parts array
        if system_instruction:
            request_body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        # Add generation config if provided
        if generation_config:
            # Map config keys to Gemini API format
            config_mapping = {
                "temperature": "temperature",
                "max_output_tokens": "maxOutputTokens",
                "top_p": "topP",
                "top_k": "topK",
                "response_mime_type": "responseMimeType",
            }
            gemini_config = {}
            for key, value in generation_config.items():
                if key in config_mapping and value is not None:
                    gemini_config[config_mapping[key]] = value
            if gemini_config:
                request_body["generationConfig"] = gemini_config

        # Add tools if provided
        if tools:
            request_body["tools"] = tools

        # Debug: log the request
        logger.info(f"Gemini REST API request URL: {api_url}")
        logger.debug(
            f"Gemini REST API request body: {json.dumps(request_body, indent=2)[:500]}"
        )

        # Make the async HTTP request
        timeout = httpx.Timeout(self.timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, json=request_body, headers=headers)

            # Debug: log the response status
            logger.info(f"Gemini REST API response status: {response.status_code}")

            if response.status_code != 200:
                # Log full error response
                logger.error(f"Gemini REST API error response (full):\n{response.text}")

            # Raise HTTPError for bad status codes (4xx, 5xx)
            # This will be caught by retry wrapper
            response.raise_for_status()

            # Return response in JSON format
            return response.json()

    async def _stream_gemini_rest_api(
        self,
        system_instruction: Optional[str],
        messages: List[Dict[str, Any]],
        generation_config: Dict[str, Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Call Gemini streaming REST API using Server-Sent Events.

        Yields parsed chunks with type, text, function_call, usage, etc.
        """
        # Build the API URL for streaming
        base_url = (
            self.base_url.rstrip("/")
            if self.base_url
            else "https://generativelanguage.googleapis.com/v1beta"
        )
        model_name = self._model_name

        # Determine authentication method
        is_official_api = "googleapis.com" in base_url

        if is_official_api:
            api_url = f"{base_url}/models/{model_name}:streamGenerateContent?key={self.api_key}"
            headers = {}
        else:
            # For proxy services, try different URL patterns
            if "/v1beta" in base_url or "/v1" in base_url:
                # Base URL already includes the version path
                api_url = f"{base_url}/models/{model_name}:streamGenerateContent"
            else:
                # Add version path for other proxies
                api_url = f"{base_url}/v1beta/models/{model_name}:streamGenerateContent"
            headers = {"Authorization": f"Bearer {self.api_key}"}

        # Prepare request body
        # Convert messages to proper Gemini format with object parts
        contents = []
        for msg in messages:
            role = msg.get("role")
            parts = msg.get("parts", [])

            # Convert parts to proper Gemini format (strings to {"text": "..."} objects)
            gemini_parts = self._convert_parts_to_gemini_format(parts)

            if gemini_parts:  # Only add if we have valid parts
                contents.append({"role": role, "parts": gemini_parts})

        request_body: Dict[str, Any] = {"contents": contents}

        # Add system instruction if provided
        if system_instruction:
            request_body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        # Add generation config
        if generation_config:
            config_mapping = {
                "temperature": "temperature",
                "max_output_tokens": "maxOutputTokens",
                "top_p": "topP",
                "top_k": "topK",
                "response_mime_type": "responseMimeType",
            }
            gemini_config = {}
            for key, value in generation_config.items():
                if key in config_mapping and value is not None:
                    gemini_config[config_mapping[key]] = value
            if gemini_config:
                request_body["generationConfig"] = gemini_config

        # Add tools if provided
        if tools:
            request_body["tools"] = tools

        # Make streaming request
        timeout = httpx.Timeout(self.timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"Gemini streaming API URL: {api_url}")
            logger.debug(
                f"Gemini streaming request body: {json.dumps(request_body, indent=2)[:1000]}"
            )

            async with client.stream(
                "POST", api_url, json=request_body, headers=headers
            ) as response:
                logger.info(f"Gemini streaming response status: {response.status_code}")

                if response.status_code != 200:
                    # Read the response content for error details
                    error_content = await response.aread()
                    logger.error(f"Gemini streaming API error: {response.status_code}")
                    logger.error(
                        f"Response: {error_content.decode('utf-8', errors='replace')[:1000]}"
                    )
                    response.raise_for_status()

                # Parse Server-Sent Events stream
                buffer = ""
                chunk_count = 0
                line_count = 0

                async for line in response.aiter_lines():
                    line_count += 1
                    logger.debug(
                        f"SSE line #{line_count}: {line[:200] if len(line) > 200 else line}"
                    )

                    # SSE format: "data: {...}\n\n"
                    if not line.strip():
                        # Empty line - this marks the end of a SSE message
                        continue

                    buffer += line

                    # Process complete SSE messages
                    # Some APIs don't use standard SSE format and return pure JSON
                    # So we also check if buffer contains valid JSON
                    while "\n\n" in buffer or (
                        buffer.strip().startswith("{") and chunk_count == 0
                    ):
                        # Check for standard SSE format first
                        if "\n\n" in buffer:
                            parts = buffer.split("\n\n", 1)
                            sse_msg: str = parts[0]
                            remaining = parts[1] if len(parts) > 1 else ""
                            buffer = remaining
                        else:
                            # No \n\n separator - treat entire buffer as one message
                            sse_msg = buffer
                            buffer = ""
                            logger.debug(
                                "Processing buffer as single SSE message (no \\n\\n separator)"
                            )

                        if not sse_msg.strip():
                            continue

                        # Remove "data: " prefix if present
                        # Remove "data: " prefix if present
                        if sse_msg.startswith("data: "):
                            data_str = sse_msg[6:].strip()
                        else:
                            # No prefix - treat as raw JSON
                            data_str = sse_msg.strip()
                            logger.debug("No 'data: ' prefix, treating as raw JSON")

                        chunk_count += 1
                        logger.debug(
                            f"SSE chunk #{chunk_count}: {data_str[:300] if len(data_str) > 300 else data_str}"
                        )

                        # Skip [DONE] message
                        if data_str == "[DONE]":
                            yield {"type": "end", "finish_reason": "stop"}
                            continue

                        try:
                            data = json.loads(data_str)

                            # Parse candidates
                            candidates = data.get("candidates", [])
                            if not candidates:
                                logger.warning(
                                    f"SSE chunk has no candidates: {data_str[:200]}"
                                )
                                continue

                            candidate = candidates[0]
                            content_parts = candidate.get("content", {}).get(
                                "parts", []
                            )

                            for part in content_parts:
                                if "text" in part:
                                    # Text token
                                    yield {
                                        "type": "text",
                                        "text": part["text"],
                                    }

                                elif "functionCall" in part:
                                    # Function call
                                    func_call = part["functionCall"]
                                    logger.info(
                                        f"SSE function call detected: {func_call.get('name', '')}"
                                    )
                                    yield {
                                        "type": "function_call",
                                        "function_call": {
                                            "id": str(uuid.uuid4()),
                                            "name": func_call.get("name", ""),
                                            "args": func_call.get("args", {}),
                                        },
                                    }

                            # Check for finish reason
                            finish_reason = candidate.get("finishReason")
                            if finish_reason:
                                logger.debug(f"SSE finish reason: {finish_reason}")
                                yield {
                                    "type": "end",
                                    "finish_reason": finish_reason,
                                }

                            # Check for usage metadata
                            usage_metadata = candidate.get("usageMetadata", {})
                            if usage_metadata:
                                logger.debug(f"SSE usage metadata: {usage_metadata}")
                                yield {
                                    "type": "usage",
                                    "usage": {
                                        "prompt_tokens": usage_metadata.get(
                                            "promptTokenCount", 0
                                        ),
                                        "candidates_token_count": usage_metadata.get(
                                            "candidatesTokenCount", 0
                                        ),
                                    },
                                }

                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse SSE data: {data_str[:150]}"
                            )
                            logger.warning(f"Parse error: {str(e)}")
                            yield {
                                "type": "error",
                                "error": f"Parse error: {str(e)}",
                            }

                # Handle case where stream ended but buffer has content
                # Some APIs return single data block without proper SSE formatting
                if buffer.strip() and chunk_count == 0:
                    logger.info(
                        f"Stream ended with buffer content, attempting to parse: {buffer[:200]}"
                    )
                    # Try to parse the remaining buffer
                    if buffer.startswith("data: "):
                        data_str = buffer[6:].strip()
                        try:
                            data = json.loads(data_str)
                            candidates = data.get("candidates", [])
                            if candidates:
                                candidate = candidates[0]
                                content_parts = candidate.get("content", {}).get(
                                    "parts", []
                                )

                                for part in content_parts:
                                    if "text" in part:
                                        yield {
                                            "type": "text",
                                            "text": part["text"],
                                        }
                                    elif "functionCall" in part:
                                        func_call = part["functionCall"]
                                        logger.info(
                                            f"Buffer function call detected: {func_call.get('name', '')}"
                                        )
                                        yield {
                                            "type": "function_call",
                                            "function_call": {
                                                "id": str(uuid.uuid4()),
                                                "name": func_call.get("name", ""),
                                                "args": func_call.get("args", {}),
                                            },
                                        }

                                finish_reason = candidate.get("finishReason")
                                if finish_reason:
                                    yield {
                                        "type": "end",
                                        "finish_reason": finish_reason,
                                    }

                                usage_metadata = candidate.get("usageMetadata", {})
                                if usage_metadata:
                                    yield {
                                        "type": "usage",
                                        "usage": {
                                            "prompt_tokens": usage_metadata.get(
                                                "promptTokenCount", 0
                                            ),
                                            "candidates_token_count": usage_metadata.get(
                                                "candidatesTokenCount", 0
                                            ),
                                        },
                                    }

                                logger.info(
                                    f"Successfully parsed buffer content: found {len(content_parts)} parts"
                                )
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse buffer as JSON: {buffer[:200]}"
                            )
                            logger.error(f"Parse error: {e}")
                            yield {
                                "type": "error",
                                "error": f"Parse error: {str(e)}",
                            }
                    else:
                        logger.warning(
                            f"Stream ended with unparsed buffer: {buffer[:200]}"
                        )

                logger.info(
                    f"SSE stream ended with {line_count} lines and {chunk_count} chunks"
                )

    def _convert_parts_to_gemini_format(self, parts: List[Any]) -> List[Any]:
        """Convert parts to Gemini-compatible format."""
        gemini_parts = []
        for part in parts:
            if isinstance(part, str):
                # Text part - wrap in dict for REST API
                gemini_parts.append({"text": part})
            elif isinstance(part, dict):
                if "text" in part:
                    # Already in correct format
                    gemini_parts.append(part)
                elif "inline_data" in part:
                    # Image part - keep inline_data format
                    gemini_parts.append(part)
                elif "function_call" in part:
                    # Function call part
                    gemini_parts.append(part)
                elif "function_response" in part:
                    # Function response part
                    gemini_parts.append(part)
                else:
                    # Unknown part type, try to use as-is
                    gemini_parts.append(part)
            else:
                gemini_parts.append({"text": str(part)})
        return gemini_parts

    async def _generate_content(
        self,
        system_instruction: Optional[str],
        messages: List[Dict[str, Any]],
        generation_config: Dict[str, Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Helper method to generate content using REST API (async)."""
        assert self._client is not None, "Client must be initialized"

        # Convert all messages to Gemini content format
        contents = []
        for msg in messages:
            role = msg.get("role")
            parts = msg.get("parts", [])

            # Convert parts to proper format
            gemini_parts = self._convert_parts_to_gemini_format(parts)

            # Skip empty parts
            if not gemini_parts:
                continue

            # Gemini uses 'user' and 'model' roles
            gemini_role = "user" if role == "user" else "model"

            # Build content object
            content_dict = {"role": gemini_role, "parts": gemini_parts}
            contents.append(content_dict)

        # Call REST API
        return await self._call_gemini_rest_api(
            system_instruction=system_instruction,
            contents=contents,
            generation_config=generation_config,
            tools=tools,
        )

    @property
    def supports_thinking_mode(self) -> bool:
        """
        Check if this Gemini LLM supports thinking mode.

        Returns:
            bool: False - Gemini does not have a thinking mode feature
        """
        return False

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
        """Stream chat completion with full streaming support.

        Uses Gemini's streamGenerateContent REST API endpoint with Server-Sent Events.
        """
        # Timeout tracking
        first_token = True
        last_token_time = None
        start_time = time.time()

        # Accumulated content
        current_content = ""

        try:
            # Convert messages to Gemini format
            system_instruction, gemini_messages = (
                self._convert_messages_to_gemini_format(messages)
            )

            # Build generation config
            gen_config: Dict[str, Any] = {}
            if temperature is not None:
                gen_config["temperature"] = temperature
            elif self.default_temperature is not None:
                gen_config["temperature"] = self.default_temperature

            if max_tokens is not None:
                gen_config["max_output_tokens"] = max_tokens
            elif self.default_max_tokens is not None:
                gen_config["max_output_tokens"] = self.default_max_tokens

            # Handle output_config for JSON schema (Gemini 3.0+)
            if output_config is not None:
                format_config = output_config.get("format", {})
                if format_config.get("type") == "json_schema":
                    # Gemini uses response_mime_type and response_json_schema
                    schema = format_config.get("schema") or format_config.get(
                        "json_schema", {}
                    )
                    if schema:
                        gen_config["response_mime_type"] = "application/json"
                        gen_config["response_json_schema"] = schema

            # Handle response_format for JSON mode (legacy)
            if response_format:
                response_type: str = response_format.get("type", "")
                if response_type == "json_object":
                    # Only set if not already set by output_config
                    if "response_mime_type" not in gen_config:
                        gen_config["response_mime_type"] = "application/json"

            # Handle tools
            gemini_tools = None
            if tools:
                gemini_tools = self._convert_tools_to_gemini_format(tools)

            # Call streaming API and process chunks
            chunk_count = 0
            text_chunks = 0
            function_call_chunks = 0
            usage_chunks = 0
            end_chunks = 0

            async for chunk in self._stream_gemini_rest_api(
                system_instruction=system_instruction,
                messages=gemini_messages,
                generation_config=gen_config,
                tools=gemini_tools,
            ):
                chunk_count += 1
                chunk_type = chunk.get("type", "unknown")
                logger.debug(
                    f"Gemini stream chunk #{chunk_count}: type={chunk_type}, content={chunk.get('text', '')[:50] if chunk_type == 'text' else ''}"
                )

                current_time = time.time()

                # Check first token timeout
                if first_token:
                    elapsed = current_time - start_time
                    if elapsed > self.timeout_config.first_token_timeout:
                        raise TimeoutError(
                            f"First token timeout: {elapsed:.2f}s > "
                            f"{self.timeout_config.first_token_timeout}s"
                        )
                    first_token = False

                # Check token interval timeout
                if last_token_time is not None:
                    interval = current_time - last_token_time
                    if interval > self.timeout_config.token_interval_timeout:
                        raise TimeoutError(
                            f"Token interval timeout: {interval:.2f}s > "
                            f"{self.timeout_config.token_interval_timeout}s"
                        )

                last_token_time = current_time

                # Process chunk based on type
                if chunk.get("type") == "text":
                    # Text token
                    text = chunk.get("text", "")
                    current_content += text
                    text_chunks += 1
                    yield StreamChunk(
                        type=ChunkType.TOKEN,
                        content=current_content,
                        delta=text,
                        raw=chunk,
                    )

                elif chunk.get("type") == "function_call":
                    # Function call
                    function_call_chunks += 1
                    func_call = chunk.get("function_call", {})
                    logger.info(f"Gemini function call: {func_call.get('name', '')}")
                    yield StreamChunk(
                        type=ChunkType.TOOL_CALL,
                        tool_calls=[
                            {
                                "id": func_call.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": func_call.get("name", ""),
                                    "arguments": json.dumps(func_call.get("args", {})),
                                },
                            }
                        ],
                        finish_reason="tool_calls",
                        raw=chunk,
                    )

                elif chunk.get("type") == "usage":
                    # Token usage
                    usage_data = chunk.get("usage", {})
                    input_tokens = usage_data.get("prompt_tokens", 0)
                    output_tokens = usage_data.get("candidates_token_count", 0)

                    # Record token usage
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

                elif chunk.get("type") == "end":
                    # End of stream
                    end_chunks += 1
                    finish_reason = chunk.get("finish_reason", "stop")
                    logger.debug(f"Gemini stream end: reason={finish_reason}")
                    yield StreamChunk(
                        type=ChunkType.END,
                        finish_reason=finish_reason,
                        raw=chunk,
                    )

                elif chunk.get("type") == "usage":
                    # Token usage
                    usage_chunks += 1
                    usage_data = chunk.get("usage", {})
                    input_tokens = usage_data.get("prompt_tokens", 0)
                    output_tokens = usage_data.get("candidates_token_count", 0)

                    # Record token usage
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

                elif chunk.get("type") == "error":
                    # Error
                    error_msg = chunk.get("error", "Unknown error")
                    logger.error(f"Gemini stream error chunk: {error_msg}")
                    yield StreamChunk(
                        type=ChunkType.ERROR,
                        content=error_msg,
                        raw=chunk,
                    )

            # Log streaming summary
            logger.info(
                f"Gemini streaming completed: "
                f"total_chunks={chunk_count}, "
                f"text_chunks={text_chunks}, "
                f"function_calls={function_call_chunks}, "
                f"usage_chunks={usage_chunks}, "
                f"end_chunks={end_chunks}, "
                f"content_length={len(current_content)}"
            )

        except LLMTimeoutError:
            # Re-raise timeout errors for retry
            raise
        except TimeoutError as e:
            # Convert to LLMTimeoutError for retry
            raise LLMTimeoutError(f"Streaming timeout: {str(e)}") from e
        except LLMRetryableError:
            # Re-raise retryable errors for retry
            raise
        except httpx.HTTPStatusError as e:
            # Retry on 400, 403, 422, 429 and 5xx
            status = e.response.status_code
            if (
                status == 400
                or status == 403
                or status == 422
                or status == 429
                or (500 <= status < 600)
            ):
                raise LLMRetryableError(f"Gemini streaming HTTP error: {str(e)}") from e
            # Other HTTP errors - return error chunk
            yield StreamChunk(
                type=ChunkType.ERROR,
                content=f"Gemini streaming error: {str(e)}",
                raw=e,
            )
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ) as e:
            # Network errors are retryable
            # RemoteProtocolError includes "Server disconnected without sending a response"
            raise LLMRetryableError(f"Gemini streaming network error: {str(e)}") from e
        except Exception as e:
            # Log and convert to error chunk for non-retryable errors
            logger.error(f"Gemini streaming error: {str(e)}")
            yield StreamChunk(
                type=ChunkType.ERROR,
                content=f"Gemini streaming error: {str(e)}",
                raw=e,
            )

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
        Perform a vision-aware chat completion for Gemini models that support vision.
        This method handles multimodal messages with image content.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
                      Content can be a string or list of multimodal content items
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            response_format: Response format specification
            thinking: Thinking mode configuration (not supported by Gemini)
            **kwargs: Additional parameters to pass to the Gemini API

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

        # Gemini's chat method handles vision automatically
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
        """Close the Gemini client and cleanup resources."""
        self._client = None

    async def __aenter__(self) -> "GeminiLLM":
        """Async context manager entry."""
        self._ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @staticmethod
    async def list_available_models(
        api_key: str, base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch available models from Google Gemini API.

        Args:
            api_key: Google API key
            base_url: Base URL for Gemini API (optional). If not provided,
                     uses the official Google API.

        Returns:
            List of available models with their information

        Example:
            >>> # Using official API
            >>> models = await GeminiLLM.list_available_models("AIza...")
            >>> # Using custom endpoint
            >>> models = await GeminiLLM.list_available_models(
            ...     "AIza...",
            ...     base_url="https://your-proxy.com/v1"
            ... )
        """
        import httpx

        # Use official API if no custom base_url provided
        if base_url is None:
            base_url = "https://generativelanguage.googleapis.com/v1beta"

        url = base_url.rstrip("/") + "/models"
        headers = {"x-goog-api-key": api_key}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                models = []
                for model in data.get("models", []):
                    models.append(
                        {
                            "id": model.get("name"),
                            "display_name": model.get("displayName"),
                            "description": model.get("description"),
                            "version": model.get("version"),
                            "capabilities": model.get("supportedGenerationMethods", []),
                        }
                    )

                # Sort by name
                models.sort(key=lambda x: x.get("id", ""))
                return models

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching Gemini models: {e.response.status_code}")
            if e.response.status_code == 401:
                raise ValueError("Invalid Google API key") from e
            if e.response.status_code == 403:
                raise ValueError(
                    "Google API key does not have permission to list models"
                ) from e
            raise
        except Exception as e:
            logger.error(f"Failed to fetch Gemini models: {e}")
            return []
