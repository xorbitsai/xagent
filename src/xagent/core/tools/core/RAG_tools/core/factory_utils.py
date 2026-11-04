from __future__ import annotations

from .config import (
    CHUNK_PARAM_WHITELIST,
    DEFAULT_INDEX_POLICY,
    DEFAULT_INDEX_TYPE,
    MODEL_SYNONYMS,
    PARSE_PARAM_WHITELIST,
)


def canonicalize_model_name(model_name: str) -> str:
    """Return a canonical model name using synonyms mapping.

    Examples:
        - "text-embedding-v4" -> "QWEN/text-embedding-v4"
        - "bge-large-zh-v1.5" -> "BAAI/bge-large-zh-v1.5"
    """
    key = model_name.strip()
    return MODEL_SYNONYMS.get(key, key)


def get_parse_param_whitelist() -> tuple[str, ...]:
    return tuple(PARSE_PARAM_WHITELIST)


def get_chunk_param_whitelist() -> tuple[str, ...]:
    return tuple(CHUNK_PARAM_WHITELIST)


def get_default_index_policy() -> tuple[int, str]:
    """Return (threshold_rows, default_index_type).

    Note: Index type selection is now dynamic based on data scale.
    This function returns the default type for backward compatibility.
    """
    return DEFAULT_INDEX_POLICY.enable_threshold_rows, DEFAULT_INDEX_TYPE
