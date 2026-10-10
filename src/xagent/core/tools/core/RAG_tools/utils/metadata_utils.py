"""Metadata serialization utilities for RAG tools.

This module provides common functions for serializing and deserializing
metadata dictionaries when storing them in LanceDB tables (which store
metadata as JSON strings).
"""

import json
from typing import Any, Dict, Optional

import pandas as pd


def serialize_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Serialize metadata dictionary to JSON string for database storage.

    LanceDB tables store metadata as string fields (pa.string()), so
    dictionary metadata must be serialized to JSON strings before writing.

    Args:
        metadata: Metadata dictionary to serialize. Can be None.

    Returns:
        JSON string representation of metadata, or None if metadata is None.
        Uses ensure_ascii=False for better Unicode support and sort_keys=True
        for consistent serialization.

    Examples:
        >>> serialize_metadata({"page": 1, "section": "intro"})
        '{"page": 1, "section": "intro"}'
        >>> serialize_metadata(None)
        None
    """
    if metadata is None:
        return None
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


CONTENT_HASH_KEY = "content_hash"  # file content a chunk was made from


def deserialize_metadata(metadata_json: Optional[str]) -> Optional[Dict[str, Any]]:
    """Deserialize metadata JSON string from database to dictionary.

    When reading metadata from LanceDB tables, the stored JSON strings
    need to be deserialized back to dictionaries for use in Python code.

    Args:
        metadata_json: JSON string to deserialize. Can be None or pandas NA.

    Returns:
        Deserialized metadata dictionary, or None if input is None/NA or empty.

    Examples:
        >>> deserialize_metadata('{"page": 1, "section": "intro"}')
        {'page': 1, 'section': 'intro'}
        >>> deserialize_metadata(None)
        None
        >>> deserialize_metadata(pd.NA)
        None
    """
    # Handle None
    if metadata_json is None:
        return None

    # Handle pandas NA (check type first to avoid errors with non-scalar types)
    if isinstance(metadata_json, (str, type(None))) and pd.isna(metadata_json):
        return None

    # Handle non-string types (e.g., list, dict, int)
    if not isinstance(metadata_json, str):
        return None

    # Handle empty string
    if not metadata_json.strip():
        return None

    try:
        result: Dict[str, Any] = json.loads(metadata_json)
        return result
    except (json.JSONDecodeError, TypeError):
        # Log error but return None to avoid breaking the pipeline
        # In production, you might want to log this
        return None
