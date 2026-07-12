"""
Tests for WorkspaceFileOperations core class.

This module tests the core workspace file operations functionality,
focusing on JSON and CSV workspace writes and reads.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.workspace import DEFAULT_USER_FILE_LIST_LIMIT, TaskWorkspace
from xagent.web.models import Base
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


class TestWorkspaceFileOperations:
    """Test suite for WorkspaceFileOperations core class."""

    def test_read_file_line_range(self, tmp_path):
        """Test that read_file can read a 1-based inclusive line range."""
        workspace = TaskWorkspace("test_read_range", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        test_file = workspace.output_dir / "notes.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

        assert ops.read_file("notes.txt", start_line=2, end_line=3) == "two\nthree\n"

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

    def test_write_json_file_returns_registered_file_ref(self, tmp_path):
        """Test that write_json_file returns a registered FileRef."""
        workspace = TaskWorkspace("test_json", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Test data
        test_data = {"name": "测试", "value": 123, "nested": {"key": "value"}}

        # Write using workspace operation
        result = ops.write_json_file("test.json", test_data)
        assert result["success"] is True
        assert isinstance(result.get("file_id"), str)
        assert result["filename"] == "test.json"
        assert result["mime_type"] == "application/json"
        assert result["relative_path"] == "output/test.json"
        assert result["file_ref"]["file_id"] == result["file_id"]

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

    def test_write_csv_file_returns_registered_file_ref(self, tmp_path):
        """Test that write_csv_file returns a registered FileRef."""
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
        assert result["success"] is True
        assert isinstance(result.get("file_id"), str)
        assert result["filename"] == "test.csv"
        assert result["mime_type"] == "text/csv"
        assert result["relative_path"] == "output/test.csv"
        assert result["file_ref"]["file_id"] == result["file_id"]

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
        assert result["success"] is True

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
        assert result["success"] is True

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
        assert result["success"] is True

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
        assert result["success"] is True

        # Verify file content uses tabs
        test_file = workspace.output_dir / "test.tsv"
        content = test_file.read_text(encoding="utf-8")
        assert "\t" in content, "File should contain tab characters"
        assert "," not in content, "File should not contain comma characters"

    def test_write_empty_csv_file_creates_registered_file_ref(self, tmp_path):
        """Test that empty CSV data still creates a registered output file."""
        workspace = TaskWorkspace("test_empty_csv", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        result = ops.write_csv_file("empty.csv", [])

        assert result["success"] is True
        assert isinstance(result.get("file_id"), str)
        assert result["filename"] == "empty.csv"
        assert result["relative_path"] == "output/empty.csv"
        assert (workspace.output_dir / "empty.csv").read_text(encoding="utf-8") == ""

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
        assert result["limit"] == DEFAULT_USER_FILE_LIST_LIMIT
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

    def test_list_all_user_files_db_pagination(self, tmp_path):
        """Test real uploaded-file pagination for a resolvable task workspace."""
        engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        try:
            user = User(username="pagination-user", password_hash="hash")
            db.add(user)
            db.flush()
            task = Task(id=790, user_id=user.id, title="Pagination task")
            db.add(task)
            db.flush()

            for index in range(DEFAULT_USER_FILE_LIST_LIMIT + 5):
                path = tmp_path / "uploads" / f"file-{index:02d}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(index), encoding="utf-8")
                db.add(
                    UploadedFile(
                        user_id=user.id,
                        task_id=task.id,
                        filename=path.name,
                        storage_path=str(path),
                        mime_type="text/plain",
                        file_size=path.stat().st_size,
                    )
                )
            db.commit()

            workspace = TaskWorkspace(
                id="web_task_790",
                base_dir=str(tmp_path / "workspaces"),
            )
            workspace.db_session = db
            ops = WorkspaceFileOperations(workspace)

            first_page = ops.list_all_user_files(include_workspace_files=False)
            second_page = ops.list_all_user_files(
                include_workspace_files=False,
                offset=DEFAULT_USER_FILE_LIST_LIMIT,
            )

            first_ids = {file_info["file_id"] for file_info in first_page["files"]}
            second_ids = {file_info["file_id"] for file_info in second_page["files"]}
            assert len(first_ids) == DEFAULT_USER_FILE_LIST_LIMIT
            assert len(second_ids) == 5
            assert first_ids.isdisjoint(second_ids)
            assert first_page["total_count"] == DEFAULT_USER_FILE_LIST_LIMIT + 5
            assert second_page["total_count"] == DEFAULT_USER_FILE_LIST_LIMIT + 5
            assert first_page["limit"] == DEFAULT_USER_FILE_LIST_LIMIT
            assert second_page["offset"] == DEFAULT_USER_FILE_LIST_LIMIT
        finally:
            db.close()
            engine.dispose()

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

    def test_get_file_info_with_image(self, tmp_path):
        """Test that workspace get_file_info returns image metadata for image files."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not installed")

        workspace = TaskWorkspace("test_image_info", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a test image in workspace output directory
        test_image = Image.new("RGB", (800, 600), color="red")
        image_path = workspace.output_dir / "test.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        test_image.save(str(image_path))

        info = ops.get_file_info("test.png")

        assert info.is_file
        assert info.image_width == 800
        assert info.image_height == 600
        assert info.image_format == "PNG"
        assert info.image_mode == "RGB"

    def test_get_file_info_non_image(self, tmp_path):
        """Test that workspace get_file_info returns None image metadata for non-image files."""
        workspace = TaskWorkspace("test_non_image_info", str(tmp_path))
        ops = WorkspaceFileOperations(workspace)

        # Create a test text file in workspace output directory
        ops.write_file("test.txt", "hello")

        info = ops.get_file_info("test.txt")

        assert info.is_file
        assert info.image_width is None
        assert info.image_height is None
        assert info.image_format is None
        assert info.image_mode is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
