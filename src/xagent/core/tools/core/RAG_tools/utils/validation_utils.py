"""Validation utilities for RAG tools.

This module provides common validation and type conversion functions used across
RAG tool modules.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ..core.exceptions import ConfigurationError, DocumentValidationError

logger = logging.getLogger(__name__)


def validate_search_common_inputs(
    collection: str,
    model_tag: str,
    top_k: int,
) -> None:
    """Validate inputs shared by all public search modes."""
    if not collection or not isinstance(collection, str):
        raise DocumentValidationError("Collection must be a non-empty string")
    if not model_tag or not isinstance(model_tag, str):
        raise DocumentValidationError("model_tag must be a non-empty string")
    if not isinstance(top_k, int) or not 1 <= top_k <= 1000:
        raise DocumentValidationError("top_k must be between 1 and 1000")


def validate_search_query_text(query_text: str) -> None:
    """Validate the text query used by sparse and hybrid search."""
    if not query_text or not isinstance(query_text, str):
        raise DocumentValidationError("query_text must be a non-empty string")


def validate_and_convert_user_id(user_id: Any) -> Optional[int]:
    """Validate and convert user_id to int if provided.

    This function handles various input types for user_id:
    - int: Returns as-is
    - str: Attempts to convert to int, or extracts numeric part from patterns like "user_1"
    - None: Returns None
    - Other types: Raises ConfigurationError

    Args:
        user_id: User ID value (can be int, str, or None)

    Returns:
        Converted integer user_id, or None if input is None

    Raises:
        ConfigurationError: If user_id cannot be converted to int

    Examples:
        >>> validate_and_convert_user_id(123)
        123
        >>> validate_and_convert_user_id("123")
        123
        >>> validate_and_convert_user_id("user_1")
        1
        >>> validate_and_convert_user_id(None)
        None
        >>> validate_and_convert_user_id("invalid")
        ConfigurationError: user_id must be an integer or a string containing a number...
    """
    if user_id is None:
        return None

    original_user_id = user_id
    if isinstance(user_id, int):
        return user_id
    elif isinstance(user_id, str):
        # Try to extract numeric part from strings like "user_1" -> 1
        try:
            # First try direct conversion
            return int(user_id)
        except ValueError:
            # Try extracting number from patterns like "user_1", "user1", etc.
            match = re.search(r"\d+", user_id)
            if match:
                extracted_id = int(match.group())
                logger.warning(
                    "Extracted user_id from string '%s' -> %s. Please use integer user_id directly.",
                    original_user_id,
                    extracted_id,
                )
                return extracted_id
            else:
                raise ConfigurationError(
                    f"user_id must be an integer or a string containing a number, "
                    f"got: {original_user_id} (type: {type(original_user_id).__name__})",
                    details={
                        "provided_value": original_user_id,
                        "provided_type": type(original_user_id).__name__,
                        "expected_type": "int",
                    },
                )
    else:
        raise ConfigurationError(
            f"user_id must be an integer, got: {user_id} (type: {type(user_id).__name__})",
            details={
                "provided_value": user_id,
                "provided_type": type(user_id).__name__,
                "expected_type": "int",
            },
        )
