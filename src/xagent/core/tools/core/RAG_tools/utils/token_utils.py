"""Token counting utilities for RAG chunking and other modules.

Uses tiktoken for accurate token counts compatible with OpenAI-style models
(cl100k_base: GPT-4, GPT-3.5). Can be reused by chunk strategies and
context length estimation.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..core.config import DEFAULT_TIKTOKEN_ENCODING

logger = logging.getLogger(__name__)


def num_tokens_from_string(
    text: str,
    encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
) -> int:
    """Return the number of tokens in a string for the given encoding.

    Args:
        text: Input text to count.
        encoding_name: tiktoken encoding name (e.g. "cl100k_base", "o200k_base").

    Returns:
        Number of tokens.
    """
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception as e:
        logger.warning(
            "tiktoken count failed (%s), falling back to char//4: %s",
            encoding_name,
            e,
        )
        return max(0, len(text) // 4)


def get_token_counter(
    encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
) -> Callable[[str], int]:
    """Return a callable that counts tokens for the given encoding.

    Caches the encoding per callable so repeated calls do not re-load.
    Use this when counting many strings with the same encoding.

    Args:
        encoding_name: tiktoken encoding name.

    Returns:
        A function f(text: str) -> int.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(
            "tiktoken get_encoding failed (%s), using char//4 fallback: %s",
            encoding_name,
            e,
        )

        def _fallback(s: str) -> int:
            return max(0, len(s) // 4)

        return _fallback

    def _count(s: str) -> int:
        if not s:
            return 0
        return len(enc.encode(s))

    return _count


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
) -> list[str]:
    """Split text into segments of at most max_tokens with optional overlap.

    Used as fallback when a single semantic unit exceeds the chunk token limit.
    Boundaries are on token boundaries, not character boundaries.

    Args:
        text: Text to split.
        max_tokens: Maximum tokens per segment.
        overlap_tokens: Number of tokens to overlap between consecutive segments.
        encoding_name: tiktoken encoding name.

    Returns:
        List of text segments.
    """
    if not text:
        return []
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    try:
        import tiktoken

        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(text)
    except Exception as e:
        logger.warning(
            "tiktoken split failed (%s), falling back to character split: %s",
            encoding_name,
            e,
        )
        # Fallback: approximate token boundary by character (max_tokens * 4)
        approx_chars = max(1, max_tokens * 4)
        overlap_chars = max(0, overlap_tokens * 4)
        out: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + approx_chars)
            out.append(text[start:end])
            if end == n:
                break
            start = end - overlap_chars
        return out

    n = len(tokens)
    if n <= max_tokens:
        return [text]

    segments: list[str] = []
    start = 0
    while start < n:
        end = min(n, start + max_tokens)
        segment_tokens = tokens[start:end]
        segments.append(enc.decode(segment_tokens))
        if end == n:
            break
        start = end - overlap_tokens
    return segments
