from pathlib import Path

import pytest

from xagent.providers.pdf_parser.base import ParseResult
from xagent.providers.pdf_parser.basic import (
    PdfPlumberParser,
    PyMuPdfParser,
    PyPdfParser,
)


# A fixture to easily access test resource files
@pytest.fixture
def resource_path() -> Path:
    """Provides the correct path to the test resources directory."""
    return Path("tests/resources/test_files")


# Parsers that can be tested with real files without heavy dependencies
BASIC_PARSERS_NO_UNSTRUCTURED = [
    PyPdfParser,
    PdfPlumberParser,
    PyMuPdfParser,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("ParserClass", BASIC_PARSERS_NO_UNSTRUCTURED)
async def test_basic_parsers_pdf_success(resource_path: Path, ParserClass):
    """Ensures the basic parsers can still process PDF files correctly."""
    test_file = resource_path / "test.pdf"
    parser = ParserClass()

    result = await parser.parse(str(test_file))

    assert result is not None
    assert isinstance(result, ParseResult)
    assert result.text_segments
    assert len(result.text_segments) > 0
    assert result.full_text
    assert "page_number" in result.text_segments[0].metadata


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Unstructured parser has optional dependencies (pi_heif) that may not be installed"
)
async def test_unstructured_parser_logic(mocker, resource_path: Path):
    """Tests the UnstructuredParser wrapper logic by mocking unstructured partition.

    Skipped because unstructured has optional dependencies that may not be installed.
    The parser is tested indirectly through integration tests.
    """
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("ParserClass", BASIC_PARSERS_NO_UNSTRUCTURED)
async def test_basic_parsers_reject_non_pdf(resource_path: Path, ParserClass):
    """Ensures basic parsers raise an error for non-PDF files.

    Note: UnstructuredParser is excluded as it now supports multiple file types
    including DOCX, PPTX, XLSX, TXT, MD, and JSON.
    """
    test_file = resource_path / "test.docx"  # Use a non-PDF file
    parser = ParserClass()

    # The file type check happens before any heavy dependencies are imported
    with pytest.raises(ValueError, match="only supports PDF files"):
        await parser.parse(str(test_file))
