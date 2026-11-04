"""Tests for file utility functions."""

import pytest

from xagent.core.tools.adapters.vibe.file_tool import read_file
from xagent.core.tools.core.RAG_tools.core.exceptions import DocumentValidationError
from xagent.core.tools.core.RAG_tools.utils.file_utils import (
    check_file_type,
    validate_file_path,
)


class TestCheckFileType:
    """Test check_file_type function."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("document.txt", "txt"),
            ("README.md", "md"),
            ("data.json", "json"),
            ("page.html", "html"),
            ("report.pdf", "pdf"),
            ("document.docx", "docx"),
            ("spreadsheet.xlsx", "xlsx"),
            ("image.jpg", "image"),
            ("photo.png", "image"),
        ],
    )
    def test_check_file_type_success(self, filename, expected):
        """Test check_file_type with valid file extensions."""
        result = check_file_type(f"/path/to/{filename}")
        assert result == expected

    def test_check_file_type_case_insensitive(self):
        """Test check_file_type is case insensitive."""
        assert check_file_type("/path/to/DOCUMENT.TXT") == "txt"
        assert check_file_type("/path/to/File.PDF") == "pdf"
        assert check_file_type("/path/to/image.JPG") == "image"

    def test_check_file_type_unsupported_extension(self):
        """Test check_file_type with unsupported extension."""
        with pytest.raises(DocumentValidationError, match="Unsupported file type"):
            check_file_type("/path/to/file.xyz")

    def test_check_file_type_no_extension(self):
        """Test check_file_type with no extension."""
        with pytest.raises(DocumentValidationError, match="Unsupported file type"):
            check_file_type("/path/to/file_no_ext")


class TestValidateFilePath:
    """Test validate_file_path function."""

    def test_validate_file_path_success(self, tmp_path):
        """Test validate_file_path with valid file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = validate_file_path(str(test_file))
        assert result is True

    def test_validate_file_path_empty_path(self):
        """Test validate_file_path with empty path."""
        with pytest.raises(DocumentValidationError, match="File path cannot be empty"):
            validate_file_path("")

    def test_validate_file_path_nonexistent(self):
        """Test validate_file_path with nonexistent file."""
        with pytest.raises(DocumentValidationError, match="File does not exist"):
            validate_file_path("/nonexistent/path/file.txt")

    def test_validate_file_path_directory(self, tmp_path):
        """Test validate_file_path with directory path."""
        with pytest.raises(DocumentValidationError, match="Path is not a file"):
            validate_file_path(str(tmp_path))


class TestFileReadWorkflow:
    """Test file reading workflow using file_tool.read_file."""

    def test_read_text_file(self, tmp_path):
        """Test reading text file content."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!\nThis is a test file."
        test_file.write_text(test_content)

        result = read_file(str(test_file))
        assert result == test_content

    def test_read_json_file(self, tmp_path):
        """Test reading JSON file content."""
        test_file = tmp_path / "test.json"
        test_content = '{"key": "value", "number": 42}'
        test_file.write_text(test_content)

        result = read_file(str(test_file))
        assert result == test_content

    def test_read_file_with_encoding(self, tmp_path):
        """Test reading file with specific encoding."""
        test_file = tmp_path / "test.txt"
        test_content = "测试内容"
        test_file.write_text(test_content, encoding="utf-8")

        result = read_file(str(test_file), encoding="utf-8")
        assert result == test_content

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_file("/nonexistent/path/file.txt")

    def test_read_and_check_workflow(self, tmp_path):
        """Test workflow of reading content and checking type."""
        test_file = tmp_path / "test.json"
        test_content = '{"key": "value"}'
        test_file.write_text(test_content)

        # Read content
        content = read_file(str(test_file))
        assert content == test_content

        # Check type
        file_type = check_file_type(str(test_file))
        assert file_type == "json"

    def test_read_and_validate_workflow(self, tmp_path):
        """Test workflow of reading content and validating path."""
        test_file = tmp_path / "test.txt"
        test_content = "test content"
        test_file.write_text(test_content)

        # Validate path
        is_valid = validate_file_path(str(test_file))
        assert is_valid is True

        # Read content
        content = read_file(str(test_file))
        assert content == test_content

    def test_error_handling_workflow(self, tmp_path):
        """Test error handling in file operations workflow."""
        # Create a file first
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        file_path = str(test_file)

        # Test successful operations
        validate_file_path(file_path)
        content = read_file(file_path)
        file_type = check_file_type(file_path)

        assert content == "test content"
        assert file_type == "txt"

        # Delete the file
        test_file.unlink()

        # Test that validate_file_path raises error
        with pytest.raises(DocumentValidationError):
            validate_file_path(file_path)

        # Test that read_file raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            read_file(file_path)
