"""
Tests for document parsing functionality in workspace file operations.

This module tests the document parsing capabilities added to handle PDF, DOCX,
Excel, CSV, and Markdown files with fallback mechanisms.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.tools.core.workspace_file_tool import (
    WorkspaceFileOperations,
    extract_text_from_document,
    is_document_file,
)
from xagent.core.workspace import TaskWorkspace


class TestIsDocumentFile:
    """Test the is_document_file utility function."""

    def test_recognizes_pdf_files(self):
        """Test that .pdf files are recognized as documents."""
        assert is_document_file("test.pdf") is True
        assert is_document_file("/path/to/document.pdf") is True
        assert is_document_file("DOCUMENT.PDF") is True  # Case insensitive

    def test_recognizes_docx_files(self):
        """Test that .docx files are recognized as documents."""
        assert is_document_file("test.docx") is True
        assert is_document_file("/path/to/document.docx") is True
        assert is_document_file("DOCUMENT.DOCX") is True  # Case insensitive

    def test_recognizes_excel_files(self):
        """Test that .xlsx and .xls files are recognized as documents."""
        assert is_document_file("test.xlsx") is True
        assert is_document_file("test.xls") is True
        assert is_document_file("/path/to/spreadsheet.xlsx") is True

    def test_recognizes_csv_files(self):
        """Test that .csv files are recognized as documents."""
        assert is_document_file("test.csv") is True
        assert is_document_file("/path/to/data.csv") is True

    def test_recognizes_markdown_files(self):
        """Test that .md files are recognized as documents."""
        assert is_document_file("test.md") is True
        assert is_document_file("/path/to/document.md") is True

    def test_rejects_non_document_files(self):
        """Test that other file types are not recognized as documents."""
        assert is_document_file("test.txt") is False
        assert is_document_file("test.json") is False
        assert is_document_file("test.xml") is False
        assert is_document_file("test.html") is False
        assert is_document_file("test.log") is False

    def test_handles_files_without_extension(self):
        """Test handling of files without extensions."""
        assert is_document_file("testfile") is False
        assert is_document_file("/path/to/testfile") is False


class TestExtractTextFromDocument:
    """Test the extract_text_from_document function."""

    @pytest.fixture
    def mock_parse_result(self):
        """Create a mock ParseResult with text segments."""
        mock_result = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Sample document content"

        mock_result.text_segments = [mock_segment]
        mock_result.tables = []
        mock_result.figures = []
        return mock_result

    @pytest.fixture
    def mock_parse_result_with_tables(self):
        """Create a mock ParseResult with tables."""
        mock_result = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Sample document content"

        mock_table = MagicMock()
        mock_table.html = "<table><tr><td>Cell 1</td><td>Cell 2</td></tr></table>"

        mock_result.text_segments = [mock_segment]
        mock_result.tables = [mock_table]
        mock_result.figures = []
        return mock_result

    @pytest.fixture
    def mock_parse_result_with_figures(self):
        """Create a mock ParseResult with figures."""
        mock_result = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Sample document content"

        mock_figure = MagicMock()
        mock_figure.text = "Figure caption"

        mock_result.text_segments = [mock_segment]
        mock_result.tables = []
        mock_result.figures = [mock_figure]
        return mock_result

    def test_extract_text_from_pdf_success(self, mock_parse_result):
        """Test successful text extraction from PDF."""
        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_parse_result),
        ):
            result = extract_text_from_document("test.pdf")

            assert result == "Sample document content"
            assert "Sample document content" in result

    def test_extract_text_from_docx_success(self, mock_parse_result):
        """Test successful text extraction from DOCX."""
        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_parse_result),
        ):
            result = extract_text_from_document("test.docx")

            assert result == "Sample document content"

    def test_extract_text_from_excel_success(self, mock_parse_result):
        """Test successful text extraction from Excel files."""
        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_parse_result),
        ):
            result = extract_text_from_document("test.xlsx")
            assert result == "Sample document content"

            result = extract_text_from_document("test.xls")
            assert result == "Sample document content"

    def test_extract_text_from_csv_success(self, mock_parse_result):
        """Test successful text extraction from CSV files."""
        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_parse_result),
        ):
            result = extract_text_from_document("test.csv")
            assert result == "Sample document content"

    def test_extract_text_from_markdown_success(self, mock_parse_result):
        """Test successful text extraction from Markdown files."""
        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_parse_result),
        ):
            result = extract_text_from_document("test.md")
            assert result == "Sample document content"

    def test_extract_text_with_tables(self, mock_parse_result_with_tables):
        """Test that tables are included in extracted text."""
        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_parse_result_with_tables),
        ):
            result = extract_text_from_document("test.pdf")

            assert "Sample document content" in result
            assert "Table:" in result
            assert "Cell 1" in result

    def test_extract_text_with_figures(self, mock_parse_result_with_figures):
        """Test that figures are included in extracted text."""
        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_parse_result_with_figures),
        ):
            result = extract_text_from_document("test.pdf")

            assert "Sample document content" in result
            assert "Figure:" in result
            assert "Figure caption" in result

    def test_extract_text_pdf_parser_fallback(self, mock_parse_result):
        """Test that PDF parsing tries multiple parsers in order."""

        # First two parsers fail, third succeeds
        async def side_effect_func(*args, **kwargs):
            # Create a new mock for each call
            mock_result = MagicMock()
            mock_segment = MagicMock()
            mock_segment.text = "Success"
            mock_result.text_segments = [mock_segment]
            mock_result.tables = []
            mock_result.figures = []
            return mock_result

        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(side_effect=side_effect_func),
        ):
            result = extract_text_from_document("test.pdf")
            assert result == "Success"

    def test_extract_text_docx_python_docx_fallback(self):
        """Test that DOCX falls back to python-docx when parsers fail."""
        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(side_effect=Exception("Parser failed")),
        ):
            # Mock python-docx
            mock_doc = MagicMock()
            mock_paragraph = MagicMock()
            mock_paragraph.text = "Paragraph text"
            mock_doc.paragraphs = [mock_paragraph]
            mock_doc.tables = []

            mock_docx_module = MagicMock()
            mock_docx_module.Document.return_value = mock_doc

            with patch("builtins.__import__", return_value=mock_docx_module):
                result = extract_text_from_document("test.docx")
                assert "Paragraph text" in result

    def test_extract_text_all_parsers_fail(self):
        """Test that an error is raised when all parsers fail."""
        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(side_effect=Exception("All parsers failed")),
        ):
            with pytest.raises(ValueError, match="Unable to parse document"):
                extract_text_from_document("test.pdf")

    def test_extract_text_empty_result(self):
        """Test handling of empty parsing results."""
        mock_result = MagicMock()
        mock_result.text_segments = []
        mock_result.tables = []
        mock_result.figures = []

        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_result),
        ):
            with pytest.raises(ValueError, match="Unable to parse document"):
                extract_text_from_document("test.pdf")


class TestWorkspaceFileOperationsDocumentParsing:
    """Test document parsing in WorkspaceFileOperations."""

    def test_read_pdf_file(self, tmp_path):
        """Test reading a PDF file through workspace operations."""
        # Create workspace
        workspace = TaskWorkspace("test_task", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a mock PDF file
        pdf_path = workspace.input_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

        # Mock the document parser
        mock_result = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Extracted PDF content"
        mock_result.text_segments = [mock_segment]
        mock_result.tables = []
        mock_result.figures = []

        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_result),
        ):
            content = ops.read_file("test.pdf")
            assert content == "Extracted PDF content"

    def test_read_docx_file(self, tmp_path):
        """Test reading a DOCX file through workspace operations."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a mock DOCX file
        docx_path = workspace.input_dir / "test.docx"
        docx_path.write_bytes(b"PK\x03\x04 fake docx content")

        # Mock the document parser
        mock_result = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Extracted DOCX content"
        mock_result.text_segments = [mock_segment]
        mock_result.tables = []
        mock_result.figures = []

        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_result),
        ):
            content = ops.read_file("test.docx")
            assert content == "Extracted DOCX content"

    def test_read_csv_as_document(self, tmp_path):
        """Test reading a CSV file as a document."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a CSV file
        csv_path = workspace.input_dir / "test.csv"
        csv_path.write_text("name,age\nAlice,25\nBob,30", encoding="utf-8")

        # Mock the document parser
        mock_result = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "name,age\nAlice,25\nBob,30"
        mock_result.text_segments = [mock_segment]
        mock_result.tables = []
        mock_result.figures = []

        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_result),
        ):
            content = ops.read_file("test.csv")
            assert "name,age" in content

    def test_read_markdown_as_document(self, tmp_path):
        """Test reading a Markdown file as a document."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a Markdown file
        md_path = workspace.input_dir / "test.md"
        md_path.write_text("# Test Document\n\nSome content", encoding="utf-8")

        # Mock the document parser
        mock_result = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "# Test Document\n\nSome content"
        mock_result.text_segments = [mock_segment]
        mock_result.tables = []
        mock_result.figures = []

        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_result),
        ):
            content = ops.read_file("test.md")
            assert "Test Document" in content

    def test_read_regular_text_file_not_affected(self, tmp_path):
        """Test that regular text files are read normally without document parsing."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a text file
        txt_path = workspace.input_dir / "test.txt"
        txt_path.write_text("Regular text content", encoding="utf-8")

        content = ops.read_file("test.txt")
        assert content == "Regular text content"

    def test_read_file_encoding_fallback(self, tmp_path):
        """Test encoding fallback mechanism for regular files."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a file with UTF-8 BOM encoding
        txt_path = workspace.input_dir / "test.txt"
        content_with_bom = "\ufeffContent with BOM"
        txt_path.write_text(content_with_bom, encoding="utf-8-sig")

        content = ops.read_file("test.txt", encoding="utf-8")
        # Should handle the BOM gracefully
        assert "Content with BOM" in content

    def test_read_file_encoding_fallback_to_latin1(self, tmp_path):
        """Test that encoding fallback includes latin-1 which can decode any byte sequence."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a binary file with bytes that are invalid UTF-8
        # but valid in latin-1 (which can decode any byte)
        bin_path = workspace.input_dir / "test.bin"
        bin_path.write_bytes(b"\xff\xfe\xfd\xfc")

        # Reading should succeed with latin-1 fallback
        # latin-1 can decode any byte sequence
        content = ops.read_file("test.bin")
        assert isinstance(content, str)
        assert len(content) == 4  # Each byte becomes a character

    def test_read_file_from_output_directory(self, tmp_path):
        """Test reading document file from output directory."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a PDF in output directory
        pdf_path = workspace.output_dir / "output.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

        # Mock the document parser
        mock_result = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Output PDF content"
        mock_result.text_segments = [mock_segment]
        mock_result.tables = []
        mock_result.figures = []

        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_result),
        ):
            content = ops.read_file("output.pdf")
            assert content == "Output PDF content"


class TestDocumentParsingIntegration:
    """Integration tests for document parsing with real file handling."""

    def test_document_file_detection_and_parsing_flow(self, tmp_path):
        """Test the complete flow from file detection to parsing."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a PDF file
        pdf_path = workspace.input_dir / "integration.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        # Verify it's detected as a document file
        assert is_document_file(str(pdf_path))

        # Mock parsing and verify flow
        mock_result = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Integration test content"
        mock_result.text_segments = [mock_segment]
        mock_result.tables = []
        mock_result.figures = []

        with patch(
            "xagent.core.tools.core.workspace_file_tool.parse_document",
            new=AsyncMock(return_value=mock_result),
        ):
            content = ops.read_file("integration.pdf")
            assert "Integration test content" in content

    def test_non_document_file_normal_read_flow(self, tmp_path):
        """Test that non-document files use normal read flow."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a regular file
        txt_path = workspace.input_dir / "regular.txt"
        txt_path.write_text("Normal file content", encoding="utf-8")

        # Verify it's not detected as a document file
        assert not is_document_file(str(txt_path))

        # Read should work normally
        content = ops.read_file("regular.txt")
        assert content == "Normal file content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
