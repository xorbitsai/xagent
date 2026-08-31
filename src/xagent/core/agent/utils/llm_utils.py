"""Utility functions for LLM calls."""

import html
import logging
import re
from typing import Any, Dict, List

from ...model.chat.exceptions import LLMEmptyContentError, LLMNoTextContentError
from ...model.chat.response_shape import classify_chat_response

logger = logging.getLogger(__name__)


def unwrap_chat_text(response: Any) -> str:
    """Extract the text content of a ``chat()`` response, or raise.

    Adapters return either a plain string (legacy shape) or an envelope dict
    such as ``{"type": "text", "content": ...}`` /
    ``{"type": "tool_call", ...}``. Callers that need the text must never
    fall back to ``str(response)``: on a tool_call envelope that yields the
    dict's repr, which then leaks into compacted context or API responses as
    if it were model output (#1714). Shape reading delegates to
    ``classify_chat_response`` so every consumer shares one structural
    source of truth.

    Returns:
        The response itself for plain-string replies, or the ``content`` of
        a dict envelope carrying a usable non-empty string.

    Raises:
        LLMEmptyContentError: The response carries string content that is
            empty or whitespace-only (same transient class the adapters raise
            for an empty generation) -- envelope or legacy plain string.
        LLMNoTextContentError: The response is a tool_call envelope, carries
            non-string content, or has an unrecognized shape.
    """
    shape = classify_chat_response(response)
    if shape.kind == "text":
        assert shape.text is not None  # classifier invariant
        return shape.text
    if shape.kind == "empty":
        raise LLMEmptyContentError("Chat response content is empty")
    if isinstance(response, dict):
        raise LLMNoTextContentError(
            "Chat response has no usable text content "
            f"(type={response.get('type')!r}, keys={sorted(response.keys())})"
        )
    raise LLMNoTextContentError(
        "Chat response has no usable text content "
        f"(response type={type(response).__name__})"
    )


def clean_llm_content(content: str) -> str:
    """Clean content sent to LLM by removing characters that may cause API errors.

    Args:
        content: Original content string

    Returns:
        Cleaned content string
    """
    if not isinstance(content, str):
        return content

    # HTML decode
    content = html.unescape(content)

    # Remove control characters (except newline, tab, carriage return)
    content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", content)

    # Remove non-breaking spaces
    content = content.replace("\xa0", " ")
    content = content.replace("\u00a0", " ")

    # Normalize whitespace (but preserve paragraph structure)
    content = re.sub(r"[ \t]+", " ", content)  # Normalize spaces and tabs
    content = re.sub(r"\n{3,}", "\n\n", content)  # Keep at most 2 consecutive newlines

    # Limit length
    if len(content) > 50000:
        content = content[:50000] + "...\n[Content truncated due to length]"

    return content.strip()


def clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean all content in message list.

    Args:
        messages: List of messages, each containing content field

    Returns:
        Cleaned message list
    """
    cleaned_messages = []
    for message in messages:
        cleaned_message = message.copy()
        if "content" in cleaned_message:
            cleaned_message["content"] = clean_llm_content(cleaned_message["content"])
        cleaned_messages.append(cleaned_message)
    return cleaned_messages


def extract_json_from_markdown(content: str) -> str:
    """Extract JSON content from markdown code blocks.

    Handles markdown-formatted JSON returned by LLM, for example:
    ```json
    {"key": "value"}
    ```

    Args:
        content: String that may contain markdown code blocks

    Returns:
        Extracted JSON string, or original content if no code blocks found
    """
    if not content or not isinstance(content, str):
        return content

    # Check if content is already a JSON object (starts with { or [)
    # This prevents extracting inner code blocks from within JSON strings
    content_stripped = content.strip()
    if content_stripped.startswith("{") or content_stripped.startswith("["):
        # Content looks like raw JSON, don't try to extract from markdown
        logger.debug("Content appears to be raw JSON, skipping markdown extraction")
        return content

    # Try to match various markdown code block formats
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            logger.info("Extracted JSON from markdown code block")
            logger.debug(f"Extracted content (first 100 chars): {extracted[:100]}")
            logger.debug(f"Extracted starts with '[': {extracted.startswith('[')}")
            logger.debug(f"Extracted starts with '{{': {extracted.startswith('{')}")
            return extracted

    # No code block found, return original content
    return content


def clean_dict_content(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively clean all string values in dictionary.

    Args:
        data: Dictionary containing string values that may need cleaning

    Returns:
        Cleaned dictionary
    """
    if not isinstance(data, dict):
        return data

    cleaned_data: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            cleaned_data[key] = clean_llm_content(value)
        elif isinstance(value, dict):
            cleaned_data[key] = clean_dict_content(value)
        elif isinstance(value, list):
            cleaned_data[key] = [
                clean_llm_content(item)
                if isinstance(item, str)
                else clean_dict_content(item)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            cleaned_data[key] = value

    return cleaned_data


def try_extract_chat_response(content: str) -> tuple[str | None, dict | None]:
    """Extract structured chat response data from an LLM output string.

    Checks if the string represents a JSON block with `{"type": "chat", "chat": {...}}`.
    Returns (display_message, chat_response_data) if successful, otherwise (None, None).

    Args:
        content: The raw output string from LLM

    Returns:
        A tuple of (display_message, chat_response_data) or (None, None)
    """
    if not content:
        return None, None

    try:
        import json

        json_str = extract_json_from_markdown(content)
        if not json_str:
            return None, None

        parsed_content = json.loads(json_str)
        if isinstance(parsed_content, dict) and parsed_content.get("type") == "chat":
            chat_response_data = parsed_content.get("chat")
            if isinstance(chat_response_data, dict):
                display_message = chat_response_data.get("message")
                return display_message, chat_response_data

    except Exception:
        pass

    return None, None
