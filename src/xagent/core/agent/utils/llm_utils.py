"""Utility functions for LLM calls."""

import html
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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
