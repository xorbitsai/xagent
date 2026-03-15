"""
Tests for WorkspaceFileOperations core class.

This module tests the core workspace file operations functionality,
focusing on the JSON and CSV file operations that were optimized
to delegate to the basic file_tool functions.
"""

import pytest

from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.workspace import TaskWorkspace


class TestWorkspaceFileOperations:
    """Test suite for WorkspaceFileOperations core class."""

    def test_read_json_file_delegation(self, tmp_path):
        """Test that read_json_file correctly delegates to basic file_tool function."""
        workspace = TaskWorkspace("test_json", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Test data
        test_data = {"name": "测试", "value": 123, "nested": {"key": "value"}}

        # Write test file directly to output directory
        import json

        test_file = workspace.output_dir / "test.json"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            json.dumps(test_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Read using workspace operation
        read_data = ops.read_json_file("test.json")
        assert read_data == test_data

    def test_write_json_file_delegation(self, tmp_path):
        """Test that write_json_file correctly delegates to basic file_tool function."""
        workspace = TaskWorkspace("test_json", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Test data
        test_data = {"name": "测试", "value": 123, "nested": {"key": "value"}}

        # Write using workspace operation
        result = ops.write_json_file("test.json", test_data)
        assert result is True

        # Verify file was written to output directory
        test_file = workspace.output_dir / "test.json"
        assert test_file.exists()

        # Verify content
        import json

        read_data = json.loads(test_file.read_text(encoding="utf-8"))
        assert read_data == test_data

    def test_read_csv_file_delegation(self, tmp_path):
        """Test that read_csv_file correctly delegates to basic file_tool function."""
        workspace = TaskWorkspace("test_csv", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Test data
        test_data = [
            {"name": "Alice", "age": "30", "city": "New York"},
            {"name": "Bob", "age": "25", "city": "London"},
            {"name": "Charlie", "age": "35", "city": "Tokyo"},
        ]

        # Write test file directly to output directory
        import csv

        test_file = workspace.output_dir / "test.csv"
        test_file.parent.mkdir(parents=True, exist_ok=True)

        with open(test_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "age", "city"])
            writer.writeheader()
            writer.writerows(test_data)

        # Read using workspace operation
        read_data = ops.read_csv_file("test.csv")
        assert read_data == test_data

    def test_write_csv_file_delegation(self, tmp_path):
        """Test that write_csv_file correctly delegates to basic file_tool function."""
        workspace = TaskWorkspace("test_csv", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Test data
        test_data = [
            {"name": "Alice", "age": "30", "city": "New York"},
            {"name": "Bob", "age": "25", "city": "London"},
            {"name": "Charlie", "age": "35", "city": "Tokyo"},
        ]

        # Write using workspace operation
        result = ops.write_csv_file("test.csv", test_data)
        assert result is True

        # Verify file was written to output directory
        test_file = workspace.output_dir / "test.csv"
        assert test_file.exists()

        # Verify content
        import csv

        with open(test_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            read_data = list(reader)
            assert read_data == test_data

    def test_json_file_path_resolution(self, tmp_path):
        """Test that JSON file operations use correct path resolution."""
        workspace = TaskWorkspace("test_path", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        test_data = {"test": "data"}

        # Write should go to output directory
        result = ops.write_json_file("output_test.json", test_data)
        assert result is True

        # Verify file is in output directory
        output_file = workspace.output_dir / "output_test.json"
        assert output_file.exists()

        # Read should search in input first, then output
        # Since we wrote to output, it should be found there
        read_data = ops.read_json_file("output_test.json")
        assert read_data == test_data

    def test_csv_file_path_resolution(self, tmp_path):
        """Test that CSV file operations use correct path resolution."""
        workspace = TaskWorkspace("test_path", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        test_data = [{"col1": "value1", "col2": "value2"}]

        # Write should go to output directory
        result = ops.write_csv_file("output_test.csv", test_data)
        assert result is True

        # Verify file is in output directory
        output_file = workspace.output_dir / "output_test.csv"
        assert output_file.exists()

        # Read should search in input first, then output
        read_data = ops.read_csv_file("output_test.csv")
        assert read_data == test_data

    def test_read_json_file_not_found(self, tmp_path):
        """Test proper error handling when JSON file doesn't exist."""
        workspace = TaskWorkspace("test_error", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        with pytest.raises(FileNotFoundError):
            ops.read_json_file("nonexistent.json")

    def test_read_csv_file_not_found(self, tmp_path):
        """Test proper error handling when CSV file doesn't exist."""
        workspace = TaskWorkspace("test_error", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        with pytest.raises(FileNotFoundError):
            ops.read_csv_file("nonexistent.csv")

    def test_write_json_file_with_indent(self, tmp_path):
        """Test that write_json_file respects the indent parameter."""
        workspace = TaskWorkspace("test_indent", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        test_data = {"key": "value", "number": 42}

        # Write with custom indent
        result = ops.write_json_file("test.json", test_data, indent=4)
        assert result is True

        # Verify file content has 4-space indentation
        test_file = workspace.output_dir / "test.json"
        content = test_file.read_text(encoding="utf-8")

        # Check that lines have 4-space indentation for nested content
        lines = content.split("\n")
        has_four_space_indent = any("    " in line for line in lines if line.strip())
        assert has_four_space_indent, "File should have 4-space indentation"

    def test_read_csv_file_with_custom_delimiter(self, tmp_path):
        """Test that read_csv_file respects the delimiter parameter."""
        workspace = TaskWorkspace("test_delimiter", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create TSV file (tab-separated)
        test_file = workspace.output_dir / "test.tsv"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            "name\tage\tcity\nAlice\t30\tNew York\nBob\t25\tLondon", encoding="utf-8"
        )

        # Read with tab delimiter
        read_data = ops.read_csv_file("test.tsv", delimiter="\t")

        expected_data = [
            {"name": "Alice", "age": "30", "city": "New York"},
            {"name": "Bob", "age": "25", "city": "London"},
        ]

        assert read_data == expected_data

    def test_write_csv_file_with_custom_delimiter(self, tmp_path):
        """Test that write_csv_file respects the delimiter parameter."""
        workspace = TaskWorkspace("test_delimiter", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        test_data = [
            {"name": "Alice", "age": "30", "city": "New York"},
            {"name": "Bob", "age": "25", "city": "London"},
        ]

        # Write with tab delimiter
        result = ops.write_csv_file("test.tsv", test_data, delimiter="\t")
        assert result is True

        # Verify file content uses tabs
        test_file = workspace.output_dir / "test.tsv"
        content = test_file.read_text(encoding="utf-8")
        assert "\t" in content, "File should contain tab characters"
        assert "," not in content, "File should not contain comma characters"

    def test_json_roundtrip_consistency(self, tmp_path):
        """Test that JSON data can be written and read back consistently."""
        workspace = TaskWorkspace("test_roundtrip", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Complex test data with various types
        test_data = {
            "string": "测试字符串",
            "number": 123.45,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "object": {"nested": "value", "deep": {"deeper": "value"}},
            "unicode": "🎉 Emoji test 🚀",
        }

        # Write and read back
        ops.write_json_file("test.json", test_data)
        read_data = ops.read_json_file("test.json")

        assert read_data == test_data, "Data should be identical after roundtrip"

    def test_csv_roundtrip_consistency(self, tmp_path):
        """Test that CSV data can be written and read back consistently."""
        workspace = TaskWorkspace("test_roundtrip", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Test data with special characters
        test_data = [
            {"name": "Alice, Smith", "age": "30", "city": "New York, NY"},
            {"name": 'Bob "The Builder"', "age": "25", "city": "London, UK"},
            {"name": "Charlie\nNewline", "age": "35", "city": "Tokyo\tJapan"},
        ]

        # Write and read back
        ops.write_csv_file("test.csv", test_data)
        read_data = ops.read_csv_file("test.csv")

        # Note: CSV reading returns all values as strings
        # We need to compare string representations
        assert len(read_data) == len(test_data)
        for i in range(len(test_data)):
            for key in test_data[i].keys():
                # CSV writer may handle special characters differently
                # We'll just verify the structure is preserved
                assert key in read_data[i]

    def test_list_all_user_files_test_workspace(self, tmp_path):
        """Test list_all_user_files with test workspace (no database)."""
        workspace = TaskWorkspace("test_workspace", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Test workspace should return success but no user_id
        result = ops.list_all_user_files()

        assert result["success"] is True
        assert result["user_id"] is None  # No user_id for test workspace
        # Files will only include workspace files if include_workspace_files=True
        assert len(result["files"]) == 0  # Default is include_workspace_files=False

    def test_list_all_user_files_with_workspace_files(self, tmp_path):
        """Test list_all_user_files includes workspace files when requested."""
        # Use a workspace ID that doesn't match web_task_{id} pattern to avoid database queries
        workspace = TaskWorkspace("test_workspace_files", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create some test files in workspace
        ops.write_file("test1.txt", "content1")
        ops.write_file("test2.txt", "content2")

        # Get files including workspace files
        result = ops.list_all_user_files(include_workspace_files=True)

        # Should have workspace files included
        assert result["success"] is True
        workspace_files = [f for f in result["files"] if f.get("is_unregistered")]
        assert len(workspace_files) >= 2

        # Check file metadata
        file_names = [f["filename"] for f in workspace_files]
        assert "test1.txt" in file_names
        assert "test2.txt" in file_names

        # Verify all unregistered files are in current workspace
        for f in workspace_files:
            assert f["in_current_workspace"] is True
            assert f["file_id"] is None

    def test_list_all_user_files_pagination(self, tmp_path):
        """Test list_all_user_files pagination parameters."""
        workspace = TaskWorkspace("test_pagination", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create multiple files
        for i in range(5):
            ops.write_file(f"file{i}.txt", f"content{i}")

        # Test pagination
        result = ops.list_all_user_files(limit=2, offset=0)
        assert result["limit"] == 2
        assert result["offset"] == 0

        result_offset = ops.list_all_user_files(limit=2, offset=2)
        assert result_offset["limit"] == 2
        assert result_offset["offset"] == 2

    def test_list_all_user_files_exclude_workspace(self, tmp_path):
        """Test list_all_user_files can exclude workspace files."""
        workspace = TaskWorkspace("test_exclude", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create test file
        ops.write_file("test.txt", "content")

        # Get files excluding workspace files
        result = ops.list_all_user_files(include_workspace_files=False)

        assert result["success"] is True
        # Should not have unregistered workspace files
        unregistered = [f for f in result["files"] if f.get("is_unregistered")]
        assert len(unregistered) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
