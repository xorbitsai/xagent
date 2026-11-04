"""
Test intelligent file search functionality in workspace file tools.
"""

import tempfile
from pathlib import Path

import pytest

from xagent.core.tools.adapters.vibe.workspace_file_tool import WorkspaceFileTools
from xagent.core.workspace import TaskWorkspace


class TestIntelligentFileSearch:
    """Test intelligent file search functionality."""

    def setup_method(self):
        """Set up test workspace and tools."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace = TaskWorkspace("test_search", str(self.temp_dir))
        self.tools = WorkspaceFileTools(self.workspace)

    def test_read_file_search_input_first(self):
        """Test that read_file searches input directory first."""
        # Create file in input directory
        input_file = self.workspace.input_dir / "test.txt"
        input_file.write_text("input content")

        # Create same name file in output directory
        output_file = self.workspace.output_dir / "test.txt"
        output_file.write_text("output content")

        # Should read from input directory
        content = self.tools.read_file("test.txt")
        assert content == "input content"

    def test_read_file_fallback_to_output(self):
        """Test that read_file falls back to output directory."""
        # Create file only in output directory
        output_file = self.workspace.output_dir / "test.txt"
        output_file.write_text("output content")

        # Should read from output directory
        content = self.tools.read_file("test.txt")
        assert content == "output content"

    def test_read_file_fallback_to_temp(self):
        """Test that read_file falls back to temp directory."""
        # Create file only in temp directory
        temp_file = self.workspace.temp_dir / "test.txt"
        temp_file.write_text("temp content")

        # Should read from temp directory
        content = self.tools.read_file("test.txt")
        assert content == "temp content"

    def test_file_exists_search_all_directories(self):
        """Test that file_exists searches all directories."""
        # Test file exists in input
        input_file = self.workspace.input_dir / "input_test.txt"
        input_file.write_text("input content")

        assert self.tools.file_exists("input_test.txt") is True

        # Test file exists in output
        output_file = self.workspace.output_dir / "output_test.txt"
        output_file.write_text("output content")

        assert self.tools.file_exists("output_test.txt") is True

        # Test file doesn't exist
        assert self.tools.file_exists("nonexistent.txt") is False

    def test_get_file_info_search(self):
        """Test that get_file_info searches all directories."""
        # Create file in input directory
        input_file = self.workspace.input_dir / "test.txt"
        input_file.write_text("input content")

        file_info = self.tools.get_file_info("test.txt")
        assert file_info.name == "test.txt"
        assert file_info.path == str(input_file.resolve())
        assert file_info.size == len("input content")

    def test_append_file_search(self):
        """Test that append_file finds files in any directory."""
        # Create file in input directory
        input_file = self.workspace.input_dir / "test.txt"
        input_file.write_text("initial content")

        # Append to file
        result = self.tools.append_file("test.txt", " appended")
        assert result is True

        # Check content
        content = input_file.read_text()
        assert content == "initial content appended"

    def test_delete_file_search(self):
        """Test that delete_file finds files in any directory."""
        # Create file in input directory
        input_file = self.workspace.input_dir / "test.txt"
        input_file.write_text("test content")

        # Delete file
        result = self.tools.delete_file("test.txt")
        assert result is True

        # Check file is deleted
        assert input_file.exists() is False

    def test_absolute_path_not_affected(self):
        """Test that absolute paths are not affected by search logic."""
        # Create file in input directory
        input_file = self.workspace.input_dir / "test.txt"
        input_file.write_text("input content")

        # Use absolute path - should work normally
        content = self.tools.read_file(str(input_file))
        assert content == "input content"

    def test_search_priority(self):
        """Test search priority: input > output > temp."""
        # Create same file in all directories
        input_file = self.workspace.input_dir / "test.txt"
        input_file.write_text("input content")

        output_file = self.workspace.output_dir / "test.txt"
        output_file.write_text("output content")

        temp_file = self.workspace.temp_dir / "test.txt"
        temp_file.write_text("temp content")

        # Should read from input (highest priority)
        content = self.tools.read_file("test.txt")
        assert content == "input content"

        # Delete input file, should read from output
        input_file.unlink()
        content = self.tools.read_file("test.txt")
        assert content == "output content"

        # Delete output file, should read from temp
        output_file.unlink()
        content = self.tools.read_file("test.txt")
        assert content == "temp content"

    def test_nonexistent_file_error_message(self):
        """Test that error message includes all search directories."""
        with pytest.raises(FileNotFoundError) as exc_info:
            self.tools.read_file("nonexistent.txt")

        error_message = str(exc_info.value)
        assert "nonexistent.txt" in error_message
        assert (
            "input" in error_message
            or "output" in error_message
            or "temp" in error_message
        )

    def test_read_json_file_uses_search(self):
        """Test that read_json_file uses the search logic."""
        # Create JSON file in input directory
        input_file = self.workspace.input_dir / "test.json"
        input_file.write_text('{"key": "input_value"}')

        # Create same file in output directory
        output_file = self.workspace.output_dir / "test.json"
        output_file.write_text('{"key": "output_value"}')

        # Should read from input directory
        data = self.tools.read_json_file("test.json")
        assert data["key"] == "input_value"

    def test_read_csv_file_uses_search(self):
        """Test that read_csv_file uses the search logic."""
        # Create CSV file in input directory
        input_file = self.workspace.input_dir / "test.csv"
        input_file.write_text("name,age\nJohn,30")

        # Create same file in output directory
        output_file = self.workspace.output_dir / "test.csv"
        output_file.write_text("name,age\nJane,25")

        # Should read from input directory
        data = self.tools.read_csv_file("test.csv")
        assert data[0]["name"] == "John"
        assert data[0]["age"] == "30"

    def teardown_method(self):
        """Clean up test workspace."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
