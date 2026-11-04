"""Test agent file operations functionality"""

import tempfile
from pathlib import Path

import pytest

from src.xagent.core.tools.adapters.vibe.file_tool import FILE_TOOLS


@pytest.fixture
def temp_uploads_dir():
    """Create temporary uploads directory for testing"""
    import shutil

    # Create temporary uploads directory
    temp_uploads = tempfile.mkdtemp(prefix="test_uploads_")
    uploads_path = Path(temp_uploads)

    yield uploads_path

    # Cleanup temporary directory
    if uploads_path.exists():
        shutil.rmtree(uploads_path)


class TestAgentFileOperations:
    """Test agent file operations"""

    def test_file_tools_available(self):
        """Test that file tools are properly defined"""
        # Check that we have file tools
        assert len(FILE_TOOLS) > 0

        # Check specific tools exist
        file_tool_names = [tool.metadata.name for tool in FILE_TOOLS]

        # Basic file tools
        assert "read_file" in file_tool_names
        assert "write_file" in file_tool_names
        assert "list_files" in file_tool_names

    def test_file_tool_execution(self, temp_uploads_dir):
        """Test executing file tools directly"""
        # Create a test file
        test_file = temp_uploads_dir / "tool_test.txt"
        test_content = "Hello, this is a test for file tools!"
        test_file.write_text(test_content)

        # Test read_file tool
        read_tool = next(
            tool for tool in FILE_TOOLS if tool.metadata.name == "read_file"
        )
        result = read_tool.run_json_sync(
            {"file_path": str(test_file), "encoding": "utf-8"}
        )
        assert result == test_content

        # Test file_exists tool
        exists_tool = next(
            tool for tool in FILE_TOOLS if tool.metadata.name == "file_exists"
        )
        result = exists_tool.run_json_sync({"file_path": str(test_file)})
        assert result is True
