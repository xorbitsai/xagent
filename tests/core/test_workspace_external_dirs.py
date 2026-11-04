"""
Tests for TaskWorkspace external directory whitelist functionality.
"""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from xagent.core.workspace import TaskWorkspace


@contextmanager
def changed_cwd(path: Path):
    import os

    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class TestWorkspaceExternalDirs(unittest.TestCase):
    """Test external directory whitelist functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary workspace and external directories
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = Path(self.temp_dir) / "test_workspace"
        self.external_dir = Path(self.temp_dir) / "external_uploads"
        self.external_dir.mkdir(parents=True, exist_ok=True)

        # Create test files in external directory
        self.test_file = self.external_dir / "test_kb_file.xlsx"
        self.test_file.write_text("Test knowledge base content")

        # Create workspace with external directory whitelist
        self.workspace = TaskWorkspace(
            id="test_task",
            base_dir=str(Path(self.temp_dir) / "workspaces"),
            allowed_external_dirs=[str(self.external_dir)],
        )

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_workspace_allows_external_directory_files(self):
        """Test that workspace can resolve paths in allowed external directories"""
        # Test absolute path in external directory
        absolute_path = str(self.test_file)
        resolved = self.workspace.resolve_path(absolute_path)

        self.assertEqual(resolved, self.test_file.resolve())
        print("✓ Resolved external file successfully")

    def test_workspace_rejects_paths_outside_whitelist(self):
        """Test that workspace rejects paths outside whitelist"""
        # Create a path outside both workspace and whitelist
        outside_dir = Path(self.temp_dir) / "outside_dir"
        outside_dir.mkdir(parents=True, exist_ok=True)
        outside_file = outside_dir / "outside_file.txt"
        outside_file.write_text("Outside content")

        # Should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            self.workspace.resolve_path(str(outside_file))

        self.assertIn("outside allowed directories", str(ctx.exception))
        print("✓ Correctly rejected path outside whitelist")

    def test_workspace_allows_user_upload_directory(self):
        """Test that user upload directory is automatically whitelisted"""
        # Create a user-specific upload directory
        user_upload_dir = Path(self.temp_dir) / "uploads" / "user_123"
        user_upload_dir.mkdir(parents=True, exist_ok=True)

        # Create workspace with user upload directory as whitelist
        workspace_with_user_dir = TaskWorkspace(
            id="test_task",
            base_dir=str(Path(self.temp_dir) / "workspaces"),
            allowed_external_dirs=[str(user_upload_dir)],
        )

        # Create test file in user upload directory
        user_file = user_upload_dir / "user_document.pdf"
        user_file.write_text("User document content")

        # Should be able to resolve
        resolved = workspace_with_user_dir.resolve_path(str(user_file))
        self.assertEqual(resolved, user_file.resolve())
        print("✓ User upload directory whitelisted successfully")

    def test_workspace_allows_cwd_relative_whitelisted_file(self):
        """Test CWD-relative files are allowed only through explicit whitelists"""
        user_upload_dir = Path(self.temp_dir) / "uploads" / "user_123"
        user_upload_dir.mkdir(parents=True, exist_ok=True)
        user_file = user_upload_dir / "user_document.pdf"
        user_file.write_text("User document content")

        workspace_with_user_dir = TaskWorkspace(
            id="test_task",
            base_dir=str(Path(self.temp_dir) / "workspaces"),
            allowed_external_dirs=[str(user_upload_dir)],
        )

        with changed_cwd(Path(self.temp_dir)):
            resolved = workspace_with_user_dir.resolve_path(
                "uploads/user_123/user_document.pdf"
            )

        self.assertEqual(resolved, user_file.resolve())

    def test_disallowed_cwd_relative_candidate_falls_back_to_workspace(self):
        """Disallowed CWD files should not block normal workspace-relative paths."""
        cwd_file = Path(self.temp_dir) / "README.md"
        cwd_file.write_text("CWD README")

        with changed_cwd(Path(self.temp_dir)):
            resolved = self.workspace.resolve_path("README.md")

        self.assertEqual(resolved, (self.workspace.output_dir / "README.md").resolve())

    def test_workspace_rejects_other_user_file_inside_base_dir(self):
        """Test uploads root access does not bypass the explicit whitelist"""
        uploads_dir = Path(self.temp_dir) / "uploads"
        user_upload_dir = uploads_dir / "user_123"
        other_user_dir = uploads_dir / "user_456"
        user_upload_dir.mkdir(parents=True, exist_ok=True)
        other_user_dir.mkdir(parents=True, exist_ok=True)
        other_user_file = other_user_dir / "secret.pdf"
        other_user_file.write_text("Other user document")

        workspace_with_user_dir = TaskWorkspace(
            id="task_123",
            base_dir=str(uploads_dir),
            allowed_external_dirs=[str(user_upload_dir)],
        )

        with self.assertRaises(ValueError) as ctx:
            workspace_with_user_dir.resolve_path(str(other_user_file))

        self.assertIn("outside allowed directories", str(ctx.exception))

    def test_workspace_resolve_path_with_search_in_external_dirs(self):
        """Test resolve_path_with_search finds files in external directories"""
        # Note: resolve_path_with_search expects relative paths for workspace search
        # For external files, we should use absolute path
        resolved = self.workspace.resolve_path(str(self.test_file))
        self.assertEqual(resolved, self.test_file.resolve())
        print("✓ resolve_path_with_search works with external files")

    def test_workspace_get_allowed_dirs_includes_external(self):
        """Test that get_allowed_dirs includes external directories"""
        allowed_dirs = self.workspace.get_allowed_dirs()

        # Should include workspace directories
        self.assertIn(str(self.workspace.workspace_dir), allowed_dirs)
        self.assertIn(str(self.workspace.input_dir), allowed_dirs)
        self.assertIn(str(self.workspace.output_dir), allowed_dirs)
        self.assertIn(str(self.workspace.temp_dir), allowed_dirs)

        # Should include external directories (normalize for macOS path differences)
        external_dir_resolved = str(self.external_dir.resolve())
        self.assertIn(
            external_dir_resolved,
            allowed_dirs,
            f"External dir {external_dir_resolved} not in {allowed_dirs}",
        )

        print(f"✓ get_allowed_dirs returns {len(allowed_dirs)} directories")

    def test_workspace_relative_paths_still_work(self):
        """Test that relative paths still work (backward compatibility)"""
        # Test relative path in default output directory
        relative_path = "test_output.txt"
        resolved = self.workspace.resolve_path(relative_path, default_dir="output")

        expected = (self.workspace.output_dir / "test_output.txt").resolve()
        self.assertEqual(resolved, expected)
        print("✓ Relative paths still work correctly")

    def test_workspace_with_nonexistent_external_dir(self):
        """Test that nonexistent external directories are skipped with warning"""
        nonexistent_dir = Path(self.temp_dir) / "does_not_exist"

        # Should not raise error, just skip the directory
        workspace = TaskWorkspace(
            id="test_task",
            base_dir=str(Path(self.temp_dir) / "workspaces"),
            allowed_external_dirs=[str(nonexistent_dir)],
        )

        # Should have empty external dirs list (directory doesn't exist)
        self.assertEqual(len(workspace.allowed_external_dirs), 0)
        print("✓ Nonexistent external directories are skipped")

    def test_workspace_allows_new_base_dir_as_external_root(self):
        """An eagerly created workspace keeps its future upload root allowed."""
        user_upload_dir = Path(self.temp_dir) / "uploads" / "user_123"
        self.assertFalse(user_upload_dir.exists())

        workspace = TaskWorkspace(
            id="web_task_1",
            base_dir=str(user_upload_dir),
            allowed_external_dirs=[str(user_upload_dir)],
        )

        source_file = user_upload_dir / "file_uuid" / "source.txt"
        source_file.parent.mkdir()
        source_file.write_text("channel upload")

        self.assertIn(user_upload_dir.resolve(), workspace.allowed_external_dirs)
        self.assertEqual(
            workspace.resolve_path(str(source_file)), source_file.resolve()
        )

    def test_workspace_security_path_traversal_protection(self):
        """Test that path traversal attacks are blocked even with external dirs"""
        # Try to access parent of external directory
        parent_dir = self.external_dir.parent
        parent_file = parent_dir / "parent_file.txt"
        parent_file.write_text("Parent content")

        # Should be rejected (not within external dir itself)
        with self.assertRaises(ValueError) as ctx:
            self.workspace.resolve_path(str(parent_file))

        self.assertIn("outside allowed directories", str(ctx.exception))
        print("✓ Path traversal protection works with external dirs")


if __name__ == "__main__":
    unittest.main()
