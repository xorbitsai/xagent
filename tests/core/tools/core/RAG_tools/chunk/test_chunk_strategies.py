"""Unit tests for chunk strategies (P0 token merge, P1 protected content & headers, custom separators)."""

import pytest

from xagent.core.tools.core.RAG_tools.chunk.chunk_strategies import (
    _find_protected_ranges,
    _split_by_headers,
    _split_by_separators_core,
    _split_by_separators_with_metadata,
    _split_by_separators_with_metadata_and_protection,
    apply_markdown_strategy,
    apply_recursive_strategy,
    attach_media_context,
)
from xagent.core.tools.core.RAG_tools.utils.token_utils import num_tokens_from_string


class TestApplyRecursiveStrategyTokenMode:
    """Unit tests for apply_recursive_strategy with use_token_count=True (P0)."""

    def test_empty_paragraphs_returns_empty(self) -> None:
        """Empty paragraphs returns empty chunks."""
        result = apply_recursive_strategy(
            [],
            {
                "chunk_size": 100,
                "chunk_overlap": 20,
                "use_token_count": True,
            },
        )
        assert result == []

    def test_use_token_count_merges_by_token_limit(self) -> None:
        """With use_token_count=True, chunks respect token limit."""
        paragraphs = [
            {
                "text": "First sentence. Second sentence. Third sentence.",
                "metadata": {},
            },
            {"text": "Another paragraph with some content here.", "metadata": {}},
        ]
        params = {
            "chunk_size": 15,
            "chunk_overlap": 3,
            "use_token_count": True,
        }
        chunks = apply_recursive_strategy(paragraphs, params)
        assert len(chunks) >= 1
        for c in chunks:
            text = c.get("text", "")
            if text.strip():
                # Each chunk should be at or under ~15 tokens (allow small overshoot from merge)
                n = num_tokens_from_string(text)
                assert n <= 25, f"chunk token count {n} expected <= 25: {text[:80]}..."

    def test_use_token_count_preserves_metadata(self) -> None:
        """Chunks retain source paragraph metadata."""
        meta = {"page_number": 1, "section": "Intro"}
        paragraphs = [{"text": "Short.", "metadata": meta}]
        params = {
            "chunk_size": 100,
            "chunk_overlap": 0,
            "use_token_count": True,
        }
        chunks = apply_recursive_strategy(paragraphs, params)
        assert len(chunks) == 1
        assert chunks[0].get("metadata") == meta

    def test_no_chunk_size_trusts_semantic_splitting(self) -> None:
        """When chunk_size is None, no token/char limit (semantic only)."""
        paragraphs = [
            {"text": "A. B. C. D. E.", "metadata": {}},
        ]
        params = {
            "chunk_size": None,
            "chunk_overlap": 0,
            "use_token_count": True,
        }
        chunks = apply_recursive_strategy(paragraphs, params)
        # Should get units from separator split (e.g. "A.", "B.", ...) as separate or merged
        assert len(chunks) >= 1

    def test_token_mode_vs_char_mode_different_chunk_count(self) -> None:
        """Token mode can produce different chunk count than character mode for same text."""
        long_text = " ".join(["word"] * 80)
        paragraphs = [{"text": long_text, "metadata": {}}]
        token_params = {
            "chunk_size": 50,
            "chunk_overlap": 10,
            "use_token_count": True,
        }
        char_params = {
            "chunk_size": 100,
            "chunk_overlap": 20,
            "use_token_count": False,
        }
        token_chunks = apply_recursive_strategy(paragraphs, token_params)
        char_chunks = apply_recursive_strategy(paragraphs, char_params)
        assert len(token_chunks) >= 1
        assert len(char_chunks) >= 1
        # Both should produce valid chunks with text
        assert all(c.get("text", "").strip() for c in token_chunks)
        assert all(c.get("text", "").strip() for c in char_chunks)


class TestSplitBySeparatorsCore:
    """Tests for _split_by_separators_core (default and custom separators)."""

    def test_default_separators_splits_by_double_newline(self) -> None:
        text = "aaa\n\nbbb\n\nccc"
        result = _split_by_separators_core(text, None)
        assert len(result) == 3
        assert result[0].strip() == "aaa"
        assert result[1].strip() == "bbb"
        assert result[2].strip() == "ccc"

    def test_custom_separator_single_sentence_end(self) -> None:
        text = "第一句。第二句。第三句。"
        result = _split_by_separators_core(text, ["。"])
        # Split by 。 gives: "第一句", "第二句", "第三句", ""
        assert len(result) >= 3
        assert "第一句" in result[0]
        assert "第二句" in result[1]
        assert "第三句" in result[2]

    def test_custom_separators_double_newline_and_newline(self) -> None:
        text = "block1\n\nblock2\nblock3"
        result = _split_by_separators_core(text, ["\n\n", "\n"])
        assert len(result) >= 2
        assert "block1" in result[0]
        assert "block2" in result[1] or "block3" in result[1]

    def test_splitting_is_lossless(self) -> None:
        """The one invariant a splitter owes its caller: rejoining the pieces
        reproduces the input exactly. Nothing else here checked that, and every
        other case uses a single separator, so a bug that dropped every other
        delimiter across mixed separators went unseen."""
        texts = [
            "attach documents and Qualifications in bulk using our CSV integration.",
            "Line one\n\nLine two\nLine three with words",
            "第一句。第二句！第三句？结束",
            "a, b, c. d e f",
            "mixed\n\npara, with. all 。 separators！ present？ here",
        ]
        for text in texts:
            assert "".join(_split_by_separators_core(text, None)) == text

    def test_delimiters_attach_to_the_preceding_chunk(self) -> None:
        """The docstring's contract. One capturing group per separator makes
        re.split emit N elements per match, so a two-step read lands on the
        wrong element and delimiters end up leading the *next* chunk instead."""
        result = _split_by_separators_core("attach documents and more", None)

        assert result == ["attach ", "documents ", "and ", "more"]

    def test_adjacent_separators_do_not_form_a_punctuation_chunk(self) -> None:
        """A separator that follows another separator has a delimiter but no
        text of its own. Judging the piece by the whole string counts the
        punctuation as content and emits a chunk holding only that."""
        result = _split_by_separators_core("hello 。 world", None)

        assert result == ["hello 。 ", "world"]

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "x",
            "no separators present here at all",
            "。leading delimiter then text",
            "trailing delimiter then nothing。",
        ],
    )
    def test_splitting_is_lossless_at_the_edges(self, text: str) -> None:
        assert "".join(_split_by_separators_core(text, None)) == text

    def test_splitting_is_lossless_with_custom_separators(self) -> None:
        """Every other case here uses the defaults, so a bug that only showed
        up with a caller-supplied list would go unseen."""
        text = "one|two||three|four"
        assert "".join(_split_by_separators_core(text, ["||", "|"])) == text

    def test_empty_separators_list_uses_default(self) -> None:
        text = "a\n\nb"
        result = _split_by_separators_core(text, [])
        assert len(result) == 2
        assert "a" in result[0]
        assert "b" in result[1]


class TestSplitWithMetadataIsLossless:
    """The core splitter's guarantee has to survive the wrappers. It did not:
    the metadata wrapper dropped whitespace-only units, and the protected-content
    path - which apply_recursive_strategy uses by default - runs the wrapper once
    per segment between fenced regions, so the separator holding two code blocks
    apart was deleted."""

    def test_metadata_wrapper_keeps_a_blank_leading_unit(self) -> None:
        """Text opening with a separator makes the core splitter emit a blank
        first unit. Filtering blank units here is what deleted the separator
        between two protected regions, so the input has to contain one - a
        string with no blank unit passes whether or not the filter is there."""
        text = "\n\nalpha beta"
        units = _split_by_separators_with_metadata(text, None)

        assert [u["text"] for u in units] == ["\n\n", "alpha ", "beta"]
        assert "".join(u["text"] for u in units) == text

    @pytest.mark.parametrize(
        "text",
        [
            "intro text\n\n```\ncode here\n```\n\ntail words",
            "a\n\n```\nx\n```\n\n```\ny\n```\n\nb",
            "```\nonly a fence\n```",
            "no fence anywhere in this text",
        ],
    )
    def test_protected_regions_keep_the_text_between_them(self, text: str) -> None:
        units = _split_by_separators_with_metadata_and_protection(
            text, None, None, None
        )

        assert "".join(u["text"] for u in units) == text


class TestApplyRecursiveStrategyKeepsWordsApart:
    """The bug reached users through this entry point, not through the private
    helpers, so the regression guard belongs here too. Every other test around
    it asserts chunk counts and substring presence, neither of which notices
    words fusing together."""

    def test_words_are_not_fused(self) -> None:
        """chunk_size is a character budget, so it is set well above the text
        to keep the window from cutting words itself - what is under test is
        the separator handling, not where the window lands."""
        text = "attach documents and Qualifications in bulk using our CSV integration"
        chunks = apply_recursive_strategy(
            [{"text": text}], {"chunk_size": 500, "chunk_overlap": 0}
        )

        joined = " ".join(c["text"] for c in chunks)
        for glued in ("CSVintegration", "documentsand", "inbulk", "usingour"):
            assert glued not in joined
        assert "CSV integration" in joined


class TestApplyRecursiveStrategyCustomSeparators:
    """Tests for apply_recursive_strategy with custom separators."""

    def _paragraph(self, text: str) -> dict:
        return {"text": text, "metadata": {}}

    def test_custom_separator_sentence_only(self) -> None:
        """Split only by 。; use small chunk_size so sliding window yields multiple chunks."""
        paragraphs = [
            self._paragraph("第一句。第二句。第三句。"),
        ]
        params = {
            "separators": ["。"],
            "chunk_size": 4,
            "chunk_overlap": 0,
        }
        chunks = apply_recursive_strategy(paragraphs, params)
        assert len(chunks) >= 2
        texts = [c["text"].strip() for c in chunks if c["text"].strip()]
        assert any("第一句" in t for t in texts)
        assert any("第二句" in t for t in texts)
        assert any("第三句" in t for t in texts)

    def test_custom_separators_double_newline_and_newline(self) -> None:
        """Custom separators [\\n\\n, \\n] split by paragraph then line; small chunk_size."""
        paragraphs = [
            self._paragraph("A\n\nB\nC"),
        ]
        params = {
            "separators": ["\n\n", "\n"],
            "chunk_size": 2,
            "chunk_overlap": 0,
        }
        chunks = apply_recursive_strategy(paragraphs, params)
        assert len(chunks) >= 2
        texts = [c["text"].strip() for c in chunks if c["text"].strip()]
        assert any("A" in t for t in texts)
        assert any("B" in t or "C" in t for t in texts)

    def test_none_separators_uses_default(self) -> None:
        """When separators is None, DEFAULT_SEPARATORS is used; small chunk_size."""
        paragraphs = [
            self._paragraph("x\n\ny"),
        ]
        params = {
            "separators": None,
            "chunk_size": 2,
            "chunk_overlap": 0,
        }
        chunks = apply_recursive_strategy(paragraphs, params)
        assert len(chunks) >= 2
        texts = [c["text"].strip() for c in chunks if c["text"].strip()]
        assert any("x" in t for t in texts)
        assert any("y" in t for t in texts)

    def test_empty_paragraphs_returns_empty_list(self) -> None:
        chunks = apply_recursive_strategy(
            [], {"separators": ["\n"], "chunk_size": 10, "chunk_overlap": 0}
        )
        assert chunks == []

    def test_metadata_preserved_in_chunks(self) -> None:
        meta = {"page_number": 1, "section": "intro"}
        paragraphs = [self._paragraph("a\n\nb"), self._paragraph("c")]
        for p in paragraphs:
            p["metadata"] = meta
        params = {
            "separators": ["\n\n", "\n"],
            "chunk_size": 100,
            "chunk_overlap": 0,
        }
        chunks = apply_recursive_strategy(paragraphs, params)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.get("metadata") == meta


class TestProtectedContent:
    """P1: Unit tests for protected content (code blocks, formulas, etc.)."""

    def test_find_protected_ranges_empty(self) -> None:
        """No protected content returns empty ranges."""
        assert _find_protected_ranges("plain text only") == []

    def test_find_protected_ranges_code_block(self) -> None:
        """Fenced code block is detected as protected."""
        text = "before\n```py\nx=1\n```\nafter"
        ranges = _find_protected_ranges(text)
        assert len(ranges) == 1
        start, end = ranges[0]
        assert text[start:end] == "```py\nx=1\n```"

    def test_find_protected_ranges_latex(self) -> None:
        """LaTeX display math is detected as protected."""
        text = "Equation: $$E=mc^2$$ end"
        ranges = _find_protected_ranges(text)
        assert len(ranges) == 1
        start, end = ranges[0]
        assert "$$" in text[start:end]

    def test_recursive_with_protection_keeps_code_block_whole(self) -> None:
        """With enable_protected_content=True and token merge, code block stays one unit."""
        paragraphs = [
            {
                "text": "Intro. ```\ncode line one\ncode line two\n``` Outro.",
                "metadata": {},
            },
        ]
        params = {
            "chunk_size": 100,
            "chunk_overlap": 0,
            "use_token_count": True,
            "enable_protected_content": True,
        }
        chunks = apply_recursive_strategy(paragraphs, params)
        full_text = " ".join(c["text"] for c in chunks)
        assert "code line one" in full_text and "code line two" in full_text
        assert "```" in full_text

    def test_recursive_without_protection_can_split_anywhere(self) -> None:
        """With enable_protected_content=False, splitting is by separators/chars only."""
        paragraphs = [{"text": "A. B. C.", "metadata": {}}]
        params = {
            "chunk_size": 2,
            "chunk_overlap": 0,
            "enable_protected_content": False,
        }
        chunks = apply_recursive_strategy(paragraphs, params)
        assert len(chunks) >= 1


class TestMarkdownHeadersAndSection:
    """P1: Unit tests for headers_to_split_on and section metadata."""

    def test_split_by_headers_default_single_section(self) -> None:
        """Text with no # headers yields one section with empty header."""
        text = "No markdown headers here.\nJust lines."
        sections = _split_by_headers(text, None)
        assert len(sections) == 1
        assert sections[0][1] == ""

    def test_split_by_headers_default_atx_style(self) -> None:
        """Default atx-style # and ## split sections."""
        text = "# One\ncontent one\n\n## Two\ncontent two"
        sections = _split_by_headers(text, None)
        assert len(sections) >= 2
        headers = [s[1] for s in sections]
        assert "# One" in headers
        assert "## Two" in headers

    def test_split_by_headers_custom_headers(self) -> None:
        """Custom headers_to_split_on splits by given prefixes."""
        text = "## A\nbody A\n\n### B\nbody B"
        sections = _split_by_headers(
            text,
            [("#", "H1"), ("##", "H2"), ("###", "H3")],
        )
        assert len(sections) >= 2
        assert any("## A" in s[1] for s in sections)
        assert any("### B" in s[1] for s in sections)

    def test_markdown_strategy_preserves_section_metadata(self) -> None:
        """Chunks from markdown strategy have section in metadata."""
        paragraphs = [
            {
                "text": "# Intro\nThis is intro.\n\n## Body\nThis is body.",
                "metadata": {},
            },
        ]
        params = {
            "chunk_size": 100,
            "chunk_overlap": 0,
            "headers_to_split_on": [("# ", "H1"), ("## ", "H2")],
        }
        chunks = apply_markdown_strategy(paragraphs, params)
        assert len(chunks) >= 2
        sections = [
            c.get("section") or c.get("metadata", {}).get("section") for c in chunks
        ]
        sections = [s for s in sections if s]
        assert any("Intro" in (s or "") for s in sections)
        assert any("Body" in (s or "") for s in sections)

    def test_markdown_fallback_when_no_headers(self) -> None:
        """When no headers match, markdown falls back to recursive."""
        paragraphs = [{"text": "No headers. Just text.", "metadata": {}}]
        params = {"chunk_size": 50, "chunk_overlap": 0}
        chunks = apply_markdown_strategy(paragraphs, params)
        assert len(chunks) >= 1
        assert "No headers" in chunks[0].get("text", "")


class TestAttachMediaContext:
    """Unit tests for P2 attach_media_context (table/image context)."""

    def test_empty_chunks_no_op(self) -> None:
        """Empty list does nothing."""
        attach_media_context([], table_context_size=50, image_context_size=50)
        # no exception

    def test_zero_sizes_no_op(self) -> None:
        """Zero context sizes do not modify chunks."""
        chunks = [{"text": "| a | b |\n|---|---|"}]
        attach_media_context(chunks, table_context_size=0, image_context_size=0)
        assert chunks[0]["text"] == "| a | b |\n|---|---|"

    def test_table_chunk_gets_prev_next_context(self) -> None:
        """Table chunk gets last N of prev and first N of next chunk."""
        chunks = [
            {"text": "Introduction paragraph here."},
            {"text": "| col1 | col2 |\n|-----|-----|\n| v1  | v2  |"},
            {"text": "Conclusion paragraph here."},
        ]
        attach_media_context(
            chunks,
            table_context_size=10,
            image_context_size=0,
        )
        # Table chunk (index 1) should have prefix from chunk 0 and suffix from chunk 2
        text = chunks[1]["text"]
        assert "paragraph here." in text or "here." in text  # last 10 of intro
        assert (
            "Conclusion" in text or "Conclusion paragraph" in text
        )  # first 10 of conclusion
        assert "| col1 | col2 |" in text

    def test_image_chunk_gets_context(self) -> None:
        """Image chunk gets prev/next context when image_context_size > 0."""
        chunks = [
            {"text": "Before image."},
            {"text": "See ![alt](img.png) for details."},
            {"text": "After image."},
        ]
        attach_media_context(
            chunks,
            table_context_size=0,
            image_context_size=6,
        )
        text = chunks[1]["text"]
        assert "Before" in text or "image." in text
        assert "After" in text or "image." in text
        assert "![alt](img.png)" in text

    def test_non_table_non_image_unchanged(self) -> None:
        """Plain text chunks are not modified."""
        chunks = [
            {"text": "Just plain text."},
        ]
        attach_media_context(chunks, table_context_size=50, image_context_size=50)
        assert chunks[0]["text"] == "Just plain text."
