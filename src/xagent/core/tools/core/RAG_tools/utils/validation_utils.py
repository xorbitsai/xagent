"""Validation utilities for RAG tools.

This module provides common validation and type conversion functions used across
RAG tool modules.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


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
