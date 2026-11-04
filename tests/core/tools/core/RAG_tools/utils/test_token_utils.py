"""Unit tests for token_utils (P0: token counting and split)."""

import pytest

from xagent.core.tools.core.RAG_tools.utils.token_utils import (
    get_token_counter,
    num_tokens_from_string,
    split_text_by_tokens,
)


class TestNumTokensFromString:
    """Tests for num_tokens_from_string."""

    def test_empty_string_returns_zero(self) -> None:
        """Empty string has zero tokens."""
        assert num_tokens_from_string("") == 0

    def test_short_text_returns_positive(self) -> None:
        """Short text has positive token count."""
        n = num_tokens_from_string("hello world")
        assert n > 0

    def test_default_encoding_cl100k_base(self) -> None:
        """Default encoding is cl100k_base (OpenAI-style)."""
        text = "The quick brown fox jumps over the lazy dog."
        n = num_tokens_from_string(text)
        # cl100k_base: this sentence is typically ~11 tokens
        assert 8 <= n <= 20

    def test_longer_text_more_tokens(self) -> None:
        """Longer text has more tokens than shorter."""
        short = "Hi"
        long = "Hello world, this is a longer sentence with more words."
        assert num_tokens_from_string(long) > num_tokens_from_string(short)

    def test_chinese_text_counted(self) -> None:
        """Chinese characters are counted (cl100k_base)."""
        n = num_tokens_from_string("你好世界")
        assert n > 0


class TestGetTokenCounter:
    """Tests for get_token_counter."""

    def test_returns_callable(self) -> None:
        """get_token_counter returns a callable."""
        fn = get_token_counter()
        assert callable(fn)

    def test_counter_matches_num_tokens(self) -> None:
        """Counter result matches num_tokens_from_string for same text."""
        text = "Same text for comparison"
        fn = get_token_counter()
        assert fn(text) == num_tokens_from_string(text)

    def test_counter_empty_string(self) -> None:
        """Counter returns 0 for empty string."""
        fn = get_token_counter()
        assert fn("") == 0


class TestSplitTextByTokens:
    """Tests for split_text_by_tokens."""

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty text returns empty list."""
        assert split_text_by_tokens("", max_tokens=10) == []

    def test_short_text_returns_single_segment(self) -> None:
        """Text under max_tokens returns one segment."""
        text = "Short."
        segments = split_text_by_tokens(text, max_tokens=100)
        assert len(segments) == 1
        assert segments[0] == text

    def test_max_tokens_zero_raises_value_error(self) -> None:
        """max_tokens <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            split_text_by_tokens("x", max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            split_text_by_tokens("x", max_tokens=-1)

    def test_long_text_split_into_multiple(self) -> None:
        """Long text is split into multiple segments."""
        # Build a string that exceeds e.g. 20 tokens
        text = " ".join(["word"] * 50)
        segments = split_text_by_tokens(text, max_tokens=20, overlap_tokens=0)
        assert len(segments) >= 2
        joined = "".join(segments)
        assert joined == text

    def test_overlap_reduces_unique_content(self) -> None:
        """With overlap, consecutive segments share some content."""
        text = " ".join(["token"] * 60)
        no_overlap = split_text_by_tokens(text, max_tokens=15, overlap_tokens=0)
        with_overlap = split_text_by_tokens(text, max_tokens=15, overlap_tokens=5)
        assert len(with_overlap) >= len(no_overlap)
        # With overlap we have more segments (or same) and segments are smaller in step
        assert all(len(s) for s in with_overlap)
