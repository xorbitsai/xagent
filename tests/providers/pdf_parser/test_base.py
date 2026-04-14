import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from xagent.providers.pdf_parser.base import ParsedTable  # Import new model
from xagent.providers.pdf_parser.base import (
    DocumentParser,
    FigureParsing,
    FullTextResult,
    LocalParsing,
    ParsedFigures,
    ParsedTextSegment,
    ParseResult,
    RemoteParsing,
    SegmentedTextResult,
    TextParsing,
    detect_file_format,
    validate_office_file_format,
)


class MockParser(DocumentParser, TextParsing, FullTextResult, LocalParsing):
    """Mock parser for testing"""

    async def _parse_impl(self, file_path: str, **kwargs: Any) -> ParseResult:
        return ParseResult(full_text="Mocked text")


class MockSegmentedParser(
    DocumentParser, TextParsing, SegmentedTextResult, RemoteParsing
):
    """Mock segmented parser for testing"""

    async def _parse_impl(self, file_path: str, **kwargs: Any) -> ParseResult:
        return ParseResult(
            text_segments=[
                ParsedTextSegment(text="Segment 1"),
                ParsedTextSegment(text="Segment 2"),
            ]
        )


class MockFigureAndTableParser(
    DocumentParser, FigureParsing, FullTextResult, LocalParsing
):
    """Mock parser for testing figures and tables"""

    async def _parse_impl(self, file_path: str, **kwargs: Any) -> ParseResult:
        return ParseResult(
            figures=[
                ParsedFigures(text="Figure 1", metadata={"page": 1}),
            ],
            tables=[ParsedTable(html="<table>...</table>", metadata={"page": 2})],
        )


@pytest.mark.asyncio
async def test_parse_validates_file():
    parser = MockParser()
    with pytest.raises(FileNotFoundError):
        await parser.parse("/nonexistent/file.pdf")


@pytest.mark.asyncio
async def test_parse_calls_impl(tmp_path):
    test_file = tmp_path / "test.pdf"
    test_file.write_text("test")

    parser = MockParser()
    result = await parser.parse(str(test_file))

    assert result.full_text == "Mocked text"


@pytest.mark.asyncio
async def test_segmented_parser_builds_full_text(tmp_path):
    test_file = tmp_path / "test.pdf"
    test_file.write_text("test")

    parser = MockSegmentedParser()
    result = await parser.parse(str(test_file))

    assert result.full_text == "Segment 1\n\nSegment 2"
    assert result.text_segments
    assert len(result.text_segments) == 2


@pytest.mark.asyncio
async def test_segmented_parser_no_segments(tmp_path):
    test_file = tmp_path / "test.pdf"
    test_file.write_text("test")

    class EmptySegmentedParser(DocumentParser, SegmentedTextResult):
        async def _parse_impl(self, file_path: str, **kwargs: Any) -> ParseResult:
            return ParseResult(text_segments=[])  # Return empty list, not None

    parser = EmptySegmentedParser()
    result = await parser.parse(str(test_file))

    assert result.full_text == ""


def test_get_capabilities_basic():
    capabilities = MockParser.get_capabilities()
    capability_names = {cap.__name__ for cap in capabilities}

    assert "TextParsing" in capability_names
    assert "FullTextResult" in capability_names
    assert "LocalParsing" in capability_names
    assert "FigureParsing" not in capability_names


def test_get_capabilities_segmented():
    capabilities = MockSegmentedParser.get_capabilities()
    capability_names = {cap.__name__ for cap in capabilities}

    assert "TextParsing" in capability_names
    assert "SegmentedTextResult" in capability_names
    assert "FullTextResult" in capability_names
    assert "RemoteParsing" in capability_names
    assert "LocalParsing" not in capability_names


def test_get_capabilities_figure_parser():
    capabilities = MockFigureAndTableParser.get_capabilities()
    capability_names = {cap.__name__ for cap in capabilities}

    assert "FigureParsing" in capability_names
    assert "FullTextResult" in capability_names
    assert "LocalParsing" in capability_names


@pytest.mark.asyncio
async def test_figure_and_table_parser_returns_results(tmp_path):
    test_file = tmp_path / "test.pdf"
    test_file.write_text("test")

    parser = MockFigureAndTableParser()
    result = await parser.parse(str(test_file))

    assert result.figures
    assert len(result.figures) == 1
    assert result.figures[0].text == "Figure 1"

    assert result.tables
    assert len(result.tables) == 1
    assert result.tables[0].html == "<table>...</table>"
    assert result.tables[0].metadata == {"page": 2}


# Tests for detect_file_format
def test_detect_file_format_pdf(tmp_path: Path) -> None:
    """Test detecting PDF format."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n")
    assert detect_file_format(str(pdf_file)) == "pdf"


def test_detect_file_format_docx(tmp_path: Path) -> None:
    """Test detecting DOCX format (Open XML)."""
    docx_file = tmp_path / "test.docx"
    # Create a minimal ZIP file with DOCX structure
    with zipfile.ZipFile(docx_file, "w") as zf:
        zf.writestr("[Content_Types].xml", "")
        zf.writestr("word/document.xml", "")
    assert detect_file_format(str(docx_file)) == "docx"


def test_detect_file_format_xlsx(tmp_path: Path) -> None:
    """Test detecting XLSX format (Open XML)."""
    xlsx_file = tmp_path / "test.xlsx"
    # Create a minimal ZIP file with XLSX structure
    with zipfile.ZipFile(xlsx_file, "w") as zf:
        zf.writestr("[Content_Types].xml", "")
        zf.writestr("xl/workbook.xml", "")
    assert detect_file_format(str(xlsx_file)) == "xlsx"


def test_detect_file_format_pptx(tmp_path: Path) -> None:
    """Test detecting PPTX format (Open XML)."""
    pptx_file = tmp_path / "test.pptx"
    # Create a minimal ZIP file with PPTX structure
    with zipfile.ZipFile(pptx_file, "w") as zf:
        zf.writestr("[Content_Types].xml", "")
        zf.writestr("ppt/presentation.xml", "")
    assert detect_file_format(str(pptx_file)) == "pptx"


def test_detect_file_format_ole2_doc(tmp_path: Path) -> None:
    """Test detecting OLE2 format (.doc)."""
    doc_file = tmp_path / "test.docx"  # Wrong extension but OLE2 format
    # OLE2 file header: D0 CF 11 E0 A1 B1 1A E1
    doc_file.write_bytes(
        bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) + b"rest"
    )
    assert detect_file_format(str(doc_file)) == "doc"


def test_detect_file_format_ole2_xls(tmp_path: Path) -> None:
    """Test detecting OLE2 format (.xls)."""
    xls_file = tmp_path / "test.xlsx"  # Wrong extension but OLE2 format
    # OLE2 file header
    xls_file.write_bytes(
        bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) + b"rest"
    )
    assert detect_file_format(str(xls_file)) == "xls"


def test_detect_file_format_unknown(tmp_path: Path) -> None:
    """Test detecting unknown format falls back to extension."""
    unknown_file = tmp_path / "test.unknown"
    unknown_file.write_bytes(b"random content")
    assert detect_file_format(str(unknown_file)) == "unknown"


# Tests for validate_office_file_format
def test_validate_office_file_format_valid_docx(tmp_path: Path) -> None:
    """Test validation passes for valid DOCX file."""
    docx_file = tmp_path / "test.docx"
    with zipfile.ZipFile(docx_file, "w") as zf:
        zf.writestr("[Content_Types].xml", "")
        zf.writestr("word/document.xml", "")
    # Should not raise
    validate_office_file_format(str(docx_file), ".docx", strict=True)


def test_validate_office_file_format_invalid_docx_ole2(tmp_path: Path) -> None:
    """Test validation fails when DOCX extension but OLE2 format."""
    doc_file = tmp_path / "test.docx"
    # OLE2 file header
    doc_file.write_bytes(
        bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) + b"rest"
    )
    with pytest.raises(ValueError, match="legacy .doc \\(OLE2\\) format"):
        validate_office_file_format(
            str(doc_file), ".docx", strict=True, parser_name="test_parser"
        )


def test_validate_office_file_format_invalid_xlsx_ole2(tmp_path: Path) -> None:
    """Test validation fails when XLSX extension but OLE2 format."""
    xls_file = tmp_path / "test.xlsx"
    # OLE2 file header
    xls_file.write_bytes(
        bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) + b"rest"
    )
    with pytest.raises(ValueError, match="legacy .xls \\(OLE2\\) format"):
        validate_office_file_format(
            str(xls_file), ".xlsx", strict=True, parser_name="test_parser"
        )


def test_validate_office_file_format_warning_mode(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test validation logs warning instead of raising when strict=False."""
    doc_file = tmp_path / "test.docx"
    # OLE2 file header
    doc_file.write_bytes(
        bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) + b"rest"
    )
    with patch("xagent.providers.pdf_parser.base.logger.warning") as mock_warning:
        validate_office_file_format(
            str(doc_file), ".docx", strict=False, parser_name="test_parser"
        )
    assert mock_warning.called
    assert "legacy .doc (OLE2) format" in mock_warning.call_args.args[0]


def test_validate_office_file_format_non_office_file(tmp_path: Path) -> None:
    """Test validation skips non-Office files."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n")
    # Should not raise or log anything
    validate_office_file_format(str(pdf_file), ".pdf", strict=True)


def test_validate_office_file_format_valid_xlsx(tmp_path: Path) -> None:
    """Test validation passes for valid XLSX file."""
    xlsx_file = tmp_path / "test.xlsx"
    with zipfile.ZipFile(xlsx_file, "w") as zf:
        zf.writestr("[Content_Types].xml", "")
        zf.writestr("xl/workbook.xml", "")
    # Should not raise
    validate_office_file_format(str(xlsx_file), ".xlsx", strict=True)


def test_validate_office_file_format_valid_pptx(tmp_path: Path) -> None:
    """Test validation passes for valid PPTX file."""
    pptx_file = tmp_path / "test.pptx"
    with zipfile.ZipFile(pptx_file, "w") as zf:
        zf.writestr("[Content_Types].xml", "")
        zf.writestr("ppt/presentation.xml", "")
    # Should not raise
    validate_office_file_format(str(pptx_file), ".pptx", strict=True)
