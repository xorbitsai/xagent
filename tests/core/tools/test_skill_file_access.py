"""
Tests for skill file access tools.

This module tests the skill file access functionality, ensuring
that agents can properly read and list files in skill directories
while maintaining proper sandbox boundaries.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from xagent.core.tools.adapters.vibe.workspace_file_tool import (
    WorkspaceFileTools,
    _validate_file_path,
    _validate_skill_name,
    create_workspace_file_tools,
)


class TestSkillFileAccess:
    """Test suite for skill file access functionality."""

    @pytest.fixture
    def temp_skills_dir(self):
        """Create a temporary skills directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Create test skill structure
            test_skill = skills_dir / "test_skill"
            test_skill.mkdir()

            # Create test files
            (test_skill / "SKILL.md").write_text(
                "# Test Skill\n\nThis is a test skill."
            )
            (test_skill / "schema.json").write_text('{"type": "test"}')

            references_dir = test_skill / "references"
            references_dir.mkdir()
            (references_dir / "guide.md").write_text("# Guide\n\nReference guide.")

            # Create hidden file
            (test_skill / ".hidden").write_text("hidden content")

            yield skills_dir

    @pytest.fixture
    def mock_workspace(self):
        """Create a mock workspace for testing."""
        from xagent.core.workspace import TaskWorkspace

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = TaskWorkspace(
                id="test_task",
                base_dir=tmpdir,
            )
            yield workspace

    def test_read_skill_file_success(self, temp_skills_dir, mock_workspace):
        """Test successful file reading from skill directory."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        content = skill_tools.read_skill_file("test_skill", "SKILL.md")
        assert content == "# Test Skill\n\nThis is a test skill."

    def test_read_skill_file_not_found(self, temp_skills_dir, mock_workspace):
        """Test FileNotFoundError when file doesn't exist."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        with pytest.raises(FileNotFoundError) as exc_info:
            skill_tools.read_skill_file("test_skill", "nonexistent.md")
        assert "File not found" in str(exc_info.value)

    def test_read_skill_file_skill_not_found(self, temp_skills_dir, mock_workspace):
        """Test FileNotFoundError when skill doesn't exist."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        with pytest.raises(FileNotFoundError) as exc_info:
            skill_tools.read_skill_file("nonexistent_skill", "file.md")
        assert "Skill not found" in str(exc_info.value)

    def test_list_skill_files_all(self, temp_skills_dir, mock_workspace):
        """Test listing all files in skill directory."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        result = skill_tools.list_skill_files("test_skill")

        assert result["directory"] == "test_skill"
        # Default recursive=True includes files in subdirectories
        assert (
            result["total_count"] == 4
        )  # SKILL.md, schema.json, references/, guide.md
        assert len(result["files"]) == 4
        assert result["current_path"] == "."

        # Check file names
        file_names = {f["name"] for f in result["files"]}
        assert "SKILL.md" in file_names
        assert "schema.json" in file_names
        assert "references" in file_names
        assert "guide.md" in file_names

    def test_list_skill_files_no_hidden(self, temp_skills_dir, mock_workspace):
        """Test that hidden files are not listed by default."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        result = skill_tools.list_skill_files("test_skill")

        file_names = {f["name"] for f in result["files"]}
        assert ".hidden" not in file_names

    def test_list_skill_files_show_hidden(self, temp_skills_dir, mock_workspace):
        """Test showing hidden files when requested."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        result = skill_tools.list_skill_files("test_skill", show_hidden=True)

        file_names = {f["name"] for f in result["files"]}
        assert ".hidden" in file_names

    def test_list_skill_files_subdirectory(self, temp_skills_dir, mock_workspace):
        """Test listing files in a subdirectory."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        result = skill_tools.list_skill_files("test_skill", "references")

        assert result["total_count"] == 1
        assert result["files"][0]["name"] == "guide.md"
        assert result["current_path"] == "references"

    def test_list_skill_files_recursive(self, temp_skills_dir, mock_workspace):
        """Test recursive file listing."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        result = skill_tools.list_skill_files("test_skill", recursive=True)

        # Should include files in references/ subdirectory
        file_names = {f["name"] for f in result["files"]}
        assert "guide.md" in file_names

    def test_list_skill_files_non_recursive(self, temp_skills_dir, mock_workspace):
        """Test non-recursive file listing."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        result = skill_tools.list_skill_files("test_skill", recursive=False)

        # Should not include files in references/ subdirectory
        file_names = {f["name"] for f in result["files"]}
        assert "guide.md" not in file_names

    def test_read_skill_file_subdirectory(self, temp_skills_dir, mock_workspace):
        """Test reading file from subdirectory."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        content = skill_tools.read_skill_file("test_skill", "references/guide.md")
        assert content == "# Guide\n\nReference guide."

    def test_get_tools_includes_skill_tools(self, temp_skills_dir, mock_workspace):
        """Test that get_tools returns skill file access tools."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )
        tools = skill_tools.get_tools()

        tool_names = {tool.name for tool in tools}
        assert "read_skill_file" in tool_names
        assert "list_skill_files" in tool_names

        # Check that skill tools have the right category
        for tool in tools:
            if tool.name in ["read_skill_file", "list_skill_files"]:
                assert tool.metadata.category.value == "skill"

    def test_create_workspace_file_tools_with_skills_roots(
        self, temp_skills_dir, mock_workspace
    ):
        """Test the factory function with custom skills_roots."""
        tools = create_workspace_file_tools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )

        tool_names = {tool.name for tool in tools}
        assert "read_skill_file" in tool_names
        assert "list_skill_files" in tool_names

    def test_default_skills_roots(self, mock_workspace):
        """Test that default skills_roots includes builtin and user directories."""
        tools = WorkspaceFileTools(mock_workspace)
        # Should have at least builtin and user directories
        assert len(tools.skills_roots) >= 2
        # Last directory name should contain "skills" (case insensitive)
        assert "skills" in tools.skills_roots[-1].name.lower()

    def test_multiple_skill_roots_search_order(self, mock_workspace):
        """Test that skills are searched in root order (first match wins)."""
        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            root1 = Path(tmpdir1) / "root1"
            root1.mkdir()
            skill1 = root1 / "my_skill"
            skill1.mkdir()
            (skill1 / "file.txt").write_text("from root1")

            root2 = Path(tmpdir2) / "root2"
            root2.mkdir()
            skill2 = root2 / "my_skill"
            skill2.mkdir()
            (skill2 / "file.txt").write_text("from root2")

            # Search root1 first, then root2
            tools = WorkspaceFileTools(
                mock_workspace, skills_roots=[str(root1), str(root2)]
            )
            content = tools.read_skill_file("my_skill", "file.txt")
            assert content == "from root1"  # First match wins

    def test_external_skill_dirs_from_env(self, mock_workspace):
        """Test that XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            external_dir = Path(tmpdir) / "external_skills"
            external_dir.mkdir()
            skill = external_dir / "env_test_skill"
            skill.mkdir()
            (skill / "test.txt").write_text("from env var")

            with patch.dict(
                os.environ, {"XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS": str(external_dir)}
            ):
                tools = WorkspaceFileTools(mock_workspace)
                # Check that external directory is included
                external_paths = [str(p) for p in tools.skills_roots]
                assert any(str(external_dir) in path for path in external_paths)


class TestSkillPathValidation:
    """Test suite for path traversal attack prevention."""

    @pytest.fixture
    def temp_skills_dir(self):
        """Create a temporary skills directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Create test skill structure
            test_skill = skills_dir / "test_skill"
            test_skill.mkdir()

            # Create test files
            (test_skill / "SKILL.md").write_text("# Test Skill")
            (test_skill / "schema.json").write_text('{"type": "test"}')

            yield skills_dir

    @pytest.fixture
    def mock_workspace(self):
        """Create a mock workspace for testing."""
        from xagent.core.workspace import TaskWorkspace

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = TaskWorkspace(
                id="test_task",
                base_dir=tmpdir,
            )
            yield workspace

    def test_validate_skill_name_valid(self):
        """Test that valid skill names pass validation."""
        valid_names = ["my_skill", "MySkill-123", "test_skill", "skill123"]
        for name in valid_names:
            _validate_skill_name(name)  # Should not raise

    def test_validate_skill_name_invalid(self):
        """Test that invalid skill names are rejected."""
        invalid_names = [
            "../etc/passwd",
            "../../etc/passwd",
            "skill/../../../etc",
            "skill/../test",
            "skill..name",
            "skill/../../",
            "",
            "skill name",  # contains space
            "skill/name",  # contains slash
        ]
        for name in invalid_names:
            with pytest.raises(ValueError):
                _validate_skill_name(name)

    def test_validate_file_path_valid(self):
        """Test that valid file paths pass validation."""
        valid_paths = ["file.txt", "dir/file.txt", "SKILL.md", "schema.json"]
        for path in valid_paths:
            _validate_file_path(path)  # Should not raise

    def test_validate_file_path_invalid(self):
        """Test that invalid file paths are rejected."""
        invalid_paths = [
            "../secret.txt",
            "../../etc/passwd",
            "/etc/passwd",
            "\\windows\\system32",
            "dir/../../file.txt",
            "../",
            "..",
        ]
        for path in invalid_paths:
            with pytest.raises(ValueError):
                _validate_file_path(path)

    def test_read_skill_file_path_traversal_blocked(
        self, temp_skills_dir, mock_workspace
    ):
        """Test that path traversal attacks are blocked in read_skill_file."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )

        with pytest.raises(ValueError, match="Invalid skill name"):
            skill_tools.read_skill_file("../etc/passwd", "file.txt")

        with pytest.raises(ValueError, match="Invalid file path"):
            skill_tools.read_skill_file("test_skill", "../../etc/passwd")

    def test_list_skill_files_path_traversal_blocked(
        self, temp_skills_dir, mock_workspace
    ):
        """Test that path traversal attacks are blocked in list_skill_files."""
        skill_tools = WorkspaceFileTools(
            mock_workspace, skills_roots=[str(temp_skills_dir)]
        )

        with pytest.raises(ValueError, match="Invalid skill name"):
            skill_tools.list_skill_files("../etc")

        with pytest.raises(ValueError, match="Invalid file path"):
            skill_tools.list_skill_files("test_skill", "../../etc")
