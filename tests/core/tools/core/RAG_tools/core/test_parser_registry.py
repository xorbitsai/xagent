"""Tests for parser registry functionality."""

from xagent.core.tools.core.document_parser import document_parser_registry
from xagent.core.tools.core.RAG_tools.core.parser_registry import (
    PARSER_COMPATIBILITY,
    _build_dynamic_compatibility,
    get_all_supported_extensions,
    get_supported_parsers,
    register_parser_support,
    validate_parser_compatibility,
)


def test_get_supported_parsers_uses_dynamic_mapping_for_pdf() -> None:
    """PDF extension should return parser set derived from parser registry."""
    parsers = get_supported_parsers(".pdf")

    # Core parsers from document_parser_registry
    assert "pypdf" in parsers
    assert "pdfplumber" in parsers
    assert "pymupdf" in parsers
    assert "unstructured" in parsers
    assert "deepdoc" in parsers


def test_get_supported_parsers_uses_dynamic_mapping_for_docx() -> None:
    """DOCX extension should use dynamic compatibility (Unstructured/DeepDoc)."""
    parsers = get_supported_parsers(".docx")

    # Unstructured and DeepDoc both declare support for .docx
    assert "unstructured" in parsers
    assert "deepdoc" in parsers


def test_get_supported_parsers_uses_dynamic_mapping_for_spreadsheets() -> None:
    """Spreadsheet extensions should surface non-PDF parsers ahead of fallback parsers."""
    xlsx_parsers = get_supported_parsers(".xlsx")
    csv_parsers = get_supported_parsers(".csv")

    assert "deepdoc" in xlsx_parsers
    assert "unstructured" in xlsx_parsers
    assert "deepdoc" in csv_parsers


class TestGetSupportedParsers:
    """Test getting supported parsers for file extensions."""

    def test_get_supported_parsers_known_extension(self):
        """Test getting parsers for known file extensions (only registered parsers)."""
        # PDF: all 5 registered parsers
        pdf_parsers = get_supported_parsers(".pdf")
        assert "deepdoc" in pdf_parsers
        assert "pymupdf" in pdf_parsers
        assert "pdfplumber" in pdf_parsers
        assert "unstructured" in pdf_parsers
        assert "pypdf" in pdf_parsers

        # Docx and text formats: unstructured (and deepdoc for docx)
        docx_parsers = get_supported_parsers(".docx")
        assert "unstructured" in docx_parsers
        assert "deepdoc" in docx_parsers

        # Markdown: only unstructured is registered for .md in this codebase
        md_parsers = get_supported_parsers(".md")
        assert "unstructured" in md_parsers

        # Extensions with no registered parser in this codebase
        py_parsers = get_supported_parsers(".py")
        assert py_parsers == []

    def test_get_supported_parsers_unknown_extension(self):
        """Test getting parsers for unknown file extensions."""
        result = get_supported_parsers(".unknown")
        assert result == []

    def test_get_supported_parsers_case_insensitive(self):
        """Test extension matching is case insensitive."""
        result1 = get_supported_parsers(".PDF")
        result2 = get_supported_parsers(".pdf")
        assert result1 == result2

    def test_get_supported_parsers_without_dot(self):
        """Test extension matching works without leading dot."""
        result1 = get_supported_parsers("pdf")
        result2 = get_supported_parsers(".pdf")
        assert result1 == result2


class TestValidateParserCompatibility:
    """Test parser compatibility validation."""

    def test_validate_parser_compatibility_mixed_allowed(self):
        """Test validation when mixed parsers are allowed."""
        # Should always return True when allow_mixed=True
        assert validate_parser_compatibility(".pdf", "any_parser", allow_mixed=True)
        assert validate_parser_compatibility(".unknown", "any_parser", allow_mixed=True)
        assert validate_parser_compatibility(".py", "web_parser", allow_mixed=True)

    def test_validate_parser_compatibility_mixed_not_allowed(self):
        """Test validation when mixed parsers are not allowed."""
        # Valid combinations (parser must be in registry and in compatibility list)
        assert validate_parser_compatibility(".pdf", "deepdoc", allow_mixed=False)
        assert validate_parser_compatibility(".md", "unstructured", allow_mixed=False)
        assert validate_parser_compatibility(".docx", "deepdoc", allow_mixed=False)

        # Invalid: "code" is not in document_parser_registry
        assert not validate_parser_compatibility(".py", "code", allow_mixed=False)
        # Invalid: extension/parser mismatch
        assert not validate_parser_compatibility(".pdf", "code", allow_mixed=False)
        assert not validate_parser_compatibility(".py", "deepdoc", allow_mixed=False)
        assert not validate_parser_compatibility(
            ".unknown", "any_parser", allow_mixed=False
        )

    def test_validate_parser_compatibility_unknown_extension(self):
        """Test validation with unknown file extension."""
        # Unknown extension should not be compatible with any parser when mixed=False
        assert not validate_parser_compatibility(
            ".xyz", "any_parser", allow_mixed=False
        )
        # But should be compatible when mixed=True
        assert validate_parser_compatibility(".xyz", "any_parser", allow_mixed=True)


class TestGetAllSupportedExtensions:
    """Test getting all supported extensions."""

    def test_get_all_supported_extensions(self):
        """Test getting all supported file extensions (static and dynamic)."""
        extensions = get_all_supported_extensions()

        # Should contain common extensions
        assert ".pdf" in extensions
        assert ".py" in extensions
        assert ".md" in extensions
        assert ".txt" in extensions
        assert ".html" in extensions

        # Should be a set
        assert isinstance(extensions, set)

        # Must include all static keys; may also include extensions from dynamic map
        assert set(PARSER_COMPATIBILITY.keys()) <= extensions


class TestRegisterParserSupport:
    """Test registering new parser support."""

    def test_register_parser_support_new_extension(self):
        """Test registering support for new file extension."""
        # Register support for a new extension
        register_parser_support(".xyz", "xyz_parser")

        # Should now be supported
        parsers = get_supported_parsers(".xyz")
        assert "xyz_parser" in parsers

        # Clean up
        if ".xyz" in PARSER_COMPATIBILITY:
            del PARSER_COMPATIBILITY[".xyz"]

    def test_register_parser_support_existing_extension(self):
        """Test registering additional parser for existing extension (static-only)."""
        # Use .py: only in static map, so register_parser_support affects get_supported_parsers
        original_parsers = get_supported_parsers(".py")
        original_count = len(original_parsers)

        register_parser_support(".py", "new_py_parser")

        new_parsers = get_supported_parsers(".py")
        assert len(new_parsers) == original_count + 1
        assert "new_py_parser" in new_parsers

        # Clean up
        if "new_py_parser" in PARSER_COMPATIBILITY[".py"]:
            PARSER_COMPATIBILITY[".py"].remove("new_py_parser")

    def test_register_parser_support_normalizes_extension(self):
        """Test that extension gets normalized with leading dot."""
        register_parser_support("xyz", "test_parser")

        parsers = get_supported_parsers(".xyz")
        assert "test_parser" in parsers

        # Clean up
        if ".xyz" in PARSER_COMPATIBILITY:
            del PARSER_COMPATIBILITY[".xyz"]

    def test_register_parser_support_no_duplicates(self):
        """Test that registering same parser twice doesn't create duplicates."""
        register_parser_support(".test", "duplicate_parser")
        register_parser_support(".test", "duplicate_parser")

        parsers = get_supported_parsers(".test")
        assert parsers.count("duplicate_parser") == 1

        # Clean up
        if ".test" in PARSER_COMPATIBILITY:
            del PARSER_COMPATIBILITY[".test"]


class TestParserCompatibilityData:
    """Test the PARSER_COMPATIBILITY data structure."""

    def test_parser_compatibility_structure(self):
        """Test that PARSER_COMPATIBILITY has correct structure."""
        assert isinstance(PARSER_COMPATIBILITY, dict)

        for ext, parsers in PARSER_COMPATIBILITY.items():
            # Extensions should start with dot
            assert ext.startswith("."), f"Extension {ext} should start with dot"

            # Parsers should be a list (may be empty for extensions with no registered parser)
            assert isinstance(parsers, list), f"Parsers for {ext} should be a list"

            # All parsers should be strings
            assert all(isinstance(p, str) for p in parsers), (
                f"All parsers for {ext} should be strings"
            )

    def test_common_file_types_supported(self):
        """Test that common file types are present; those with registered parsers have non-empty lists."""
        common_extensions = [
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".py",
            ".html",
            ".json",
            ".csv",
            ".xlsx",
            ".ppt",
            ".pptx",
        ]

        for ext in common_extensions:
            assert ext in PARSER_COMPATIBILITY, (
                f"Common extension {ext} should be in PARSER_COMPATIBILITY"
            )

        # Extensions that have at least one registered parser
        extensions_with_parsers = [
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".json",
            ".xlsx",
            ".xls",
            ".ppt",
            ".pptx",
            ".doc",
        ]
        for ext in extensions_with_parsers:
            if ext not in PARSER_COMPATIBILITY:
                continue
            assert len(PARSER_COMPATIBILITY[ext]) > 0, (
                f"Extension {ext} should have at least one registered parser"
            )

    def test_powerpoint_files_supported(self):
        """Test that PowerPoint files are supported with unstructured parser."""
        # Test .pptx support
        pptx_parsers = get_supported_parsers(".pptx")
        assert "unstructured" in pptx_parsers
        assert len(pptx_parsers) > 0

        # Test .ppt support
        ppt_parsers = get_supported_parsers(".ppt")
        assert "unstructured" in ppt_parsers
        assert len(ppt_parsers) > 0

        # Validate compatibility
        assert validate_parser_compatibility(".pptx", "unstructured", allow_mixed=False)
        assert validate_parser_compatibility(".ppt", "unstructured", allow_mixed=False)

        # deepdoc should not be compatible with PowerPoint
        assert not validate_parser_compatibility(".pptx", "deepdoc", allow_mixed=False)
        assert not validate_parser_compatibility(".ppt", "deepdoc", allow_mixed=False)


class TestDynamicCompatibilityEdgeCases:
    """Edge case tests for dynamic compatibility mapping."""

    def test_parser_without_supported_extensions_is_skipped(self, monkeypatch):
        """Parsers without supported_extensions attribute should be ignored."""

        class DummyNoAttr:
            """Parser without supported_extensions."""

            pass

        def fake_parsers():
            return {"dummy": DummyNoAttr}

        # Monkeypatch registry to only contain the dummy parser
        monkeypatch.setattr(document_parser_registry, "parsers", fake_parsers)

        mapping = _build_dynamic_compatibility()
        assert mapping == {}

    def test_parser_with_empty_supported_extensions_is_skipped(self, monkeypatch):
        """Parsers with empty supported_extensions should be ignored."""

        class DummyEmpty:
            """Parser with empty supported_extensions."""

            supported_extensions: list[str] = []

        def fake_parsers():
            return {"dummy": DummyEmpty}

        monkeypatch.setattr(document_parser_registry, "parsers", fake_parsers)

        mapping = _build_dynamic_compatibility()
        assert mapping == {}

    def test_build_dynamic_compatibility_respects_runtime_registry_changes(
        self, monkeypatch
    ):
        """_build_dynamic_compatibility reflects current registry state when called."""

        class DummyMarkdown:
            """Parser declaring support for .md at runtime."""

            supported_extensions = [".md"]

        original_parsers = document_parser_registry.parsers()

        def extended_parsers():
            data = dict(original_parsers)
            data["dummy_markdown"] = DummyMarkdown
            return data

        monkeypatch.setattr(document_parser_registry, "parsers", extended_parsers)

        mapping = _build_dynamic_compatibility()
        assert ".md" in mapping
        assert "dummy_markdown" in mapping[".md"]
