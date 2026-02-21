"""
Tests for workspace file tool consistency between write and read operations.
"""

import pytest

from xagent.core.tools.adapters.vibe.workspace_file_tool import WorkspaceFileTools
from xagent.core.workspace import TaskWorkspace


class TestWorkspaceFileToolConsistency:
    """Test that write and read operations work consistently."""

    def test_write_then_read_consistency(self, tmp_path):
        """Test that a file written can be immediately read back."""
        # Create workspace
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        # Test content
        test_content = "Hello, workspace!"
        test_filename = "test_file.txt"

        # Write file
        write_result = tools.write_file(test_filename, test_content)
        assert write_result is True

        # Verify file exists in output directory
        output_file = workspace.output_dir / test_filename
        assert output_file.exists()
        assert output_file.read_text() == test_content

        # Read file back
        read_content = tools.read_file(test_filename)
        assert read_content == test_content

    def test_write_then_read_with_relative_path(self, tmp_path):
        """Test that relative paths work consistently."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Relative path test"
        test_filename = "subdir/test_file.txt"

        # Write file with relative path
        write_result = tools.write_file(test_filename, test_content)
        assert write_result is True

        # Verify file exists
        output_file = workspace.output_dir / "subdir" / "test_file.txt"
        assert output_file.exists()
        assert output_file.read_text() == test_content

        # Read file back with same relative path
        read_content = tools.read_file(test_filename)
        assert read_content == test_content

    def test_write_then_read_with_different_default_dirs(self, tmp_path):
        """Test that write and read use consistent default directories."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Default dir test"
        test_filename = "test_default.txt"

        # Write to output directory (default for write_file)
        write_result = tools.write_file(test_filename, test_content)
        assert write_result is True

        # Read from output directory (should be default for read_file too)
        read_content = tools.read_file(test_filename)
        assert read_content == test_content

        # Verify the file is in output directory
        output_file = workspace.output_dir / test_filename
        assert output_file.exists()

    def test_file_not_found_error(self, tmp_path):
        """Test proper error when file doesn't exist."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        with pytest.raises(
            FileNotFoundError,
            match="File 'nonexistent.txt' not found in workspace directories",
        ):
            tools.read_file("nonexistent.txt")

    def test_write_with_output_prefix(self, tmp_path):
        """Test that writing with 'output/' prefix doesn't create duplicate directories."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Test content with output prefix"

        # Write with output/ prefix - should go to workspace/output/banner.html
        # NOT workspace/output/output/banner.html
        write_result = tools.write_file("output/banner.html", test_content)
        assert write_result is True

        # Verify file is in workspace/output/banner.html
        expected_file = workspace.output_dir / "banner.html"
        assert expected_file.exists(), f"File should exist at {expected_file}"

        # Verify duplicate directory was NOT created
        duplicate_file = workspace.output_dir / "output" / "banner.html"
        assert not duplicate_file.exists(), (
            "Duplicate output/output directory should not exist"
        )

        # Verify content is correct
        assert expected_file.read_text() == test_content

    def test_write_with_input_prefix(self, tmp_path):
        """Test that writing with 'input/' prefix works correctly."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Test content with input prefix"

        # Write with input/ prefix
        write_result = tools.write_file("input/data.txt", test_content)
        assert write_result is True

        # Verify file is in workspace/input/data.txt
        expected_file = workspace.input_dir / "data.txt"
        assert expected_file.exists()
        assert expected_file.read_text() == test_content

    def test_write_with_temp_prefix(self, tmp_path):
        """Test that writing with 'temp/' prefix works correctly."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Test content with temp prefix"

        # Write with temp/ prefix
        write_result = tools.write_file("temp/cache.txt", test_content)
        assert write_result is True

        # Verify file is in workspace/temp/cache.txt
        expected_file = workspace.temp_dir / "cache.txt"
        assert expected_file.exists()
        assert expected_file.read_text() == test_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
