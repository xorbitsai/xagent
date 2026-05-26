"""
Tests for workspace file tool consistency between write and read operations.
"""

from types import SimpleNamespace

import pytest

from xagent.core.tools.adapters.vibe.workspace_file_tool import WorkspaceFileTools
from xagent.core.workspace import TaskWorkspace


@pytest.fixture
def mock_workspace_db(mocker):
    """Mock database operations for workspace to avoid DB access in tests."""

    # Mock _create_file_record to do nothing (avoid DB access)
    def mock_create_record(self, file_id, file_path, db_session=None):
        # Store file_id in cache for retrieval
        path_str = str(file_path)
        resolved_str = str(file_path.resolve())
        self._recently_registered_files[path_str] = file_id
        self._recently_registered_files[resolved_str] = file_id
        self._file_id_to_path[file_id] = file_path

    mocker.patch(
        "xagent.core.workspace.TaskWorkspace._create_file_record", mock_create_record
    )
    return mocker


class TestWorkspaceFileToolConsistency:
    """Test that write and read operations work consistently."""

    @pytest.mark.usefixtures("mock_workspace_db")
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
        assert write_result["success"] is True
        assert isinstance(write_result.get("file_id"), str)
        assert write_result["filename"] == test_filename
        assert write_result["mime_type"] == "text/plain"
        assert write_result["size"] == len(test_content)
        assert write_result["preview_url"].endswith(write_result["file_id"])
        assert write_result["download_url"].endswith(write_result["file_id"])
        assert write_result["markdown_link"] == (
            f"[{test_filename}](file:{write_result['file_id']})"
        )
        assert write_result["file_ref"]["file_id"] == write_result["file_id"]
        assert write_result["file_ref"]["relative_path"] == "output/test_file.txt"

        # Verify file exists in output directory
        output_file = workspace.output_dir / test_filename
        assert output_file.exists()
        assert output_file.read_text() == test_content

        # Read file back
        read_content = tools.read_file(test_filename)
        assert read_content == test_content

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_write_json_returns_file_ref(self, tmp_path):
        """Test that JSON writes expose FileRef metadata to the model."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        result = tools.write_json_file("data/report.json", {"answer": 42})

        assert result["success"] is True
        assert isinstance(result.get("file_id"), str)
        assert result["filename"] == "report.json"
        assert result["mime_type"] == "application/json"
        assert result["relative_path"] == "output/data/report.json"
        assert result["preview_url"].endswith(result["file_id"])
        assert result["download_url"].endswith(result["file_id"])
        assert result["markdown_link"] == f"[report.json](file:{result['file_id']})"
        assert result["file_ref"]["file_id"] == result["file_id"]
        assert (workspace.output_dir / "data" / "report.json").exists()

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_write_csv_returns_file_ref(self, tmp_path):
        """Test that CSV writes expose FileRef metadata to the model."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        result = tools.write_csv_file(
            "data/report.csv",
            [{"name": "Alice", "score": "10"}, {"name": "Bob", "score": "11"}],
        )

        assert result["success"] is True
        assert isinstance(result.get("file_id"), str)
        assert result["filename"] == "report.csv"
        assert result["mime_type"] == "text/csv"
        assert result["relative_path"] == "output/data/report.csv"
        assert result["preview_url"].endswith(result["file_id"])
        assert result["download_url"].endswith(result["file_id"])
        assert result["markdown_link"] == f"[report.csv](file:{result['file_id']})"
        assert result["file_ref"]["file_id"] == result["file_id"]
        assert (workspace.output_dir / "data" / "report.csv").exists()

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_write_then_read_with_relative_path(self, tmp_path):
        """Test that relative paths work consistently."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        # Manually set up the cache after workspace creation (mock runs after __init__)
        tools = WorkspaceFileTools(workspace)

        test_content = "Relative path test"
        test_filename = "subdir/test_file.txt"

        # Write file with relative path
        write_result = tools.write_file(test_filename, test_content)
        assert write_result["success"] is True
        assert isinstance(write_result.get("file_id"), str)

        # Verify file exists
        output_file = workspace.output_dir / "subdir" / "test_file.txt"
        assert output_file.exists()
        assert output_file.read_text() == test_content

        # Read file back with same relative path
        read_content = tools.read_file(test_filename)
        assert read_content == test_content

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_write_then_read_with_different_default_dirs(self, tmp_path):
        """Test that write and read use consistent default directories."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Default dir test"
        test_filename = "test_default.txt"

        # Write to output directory (default for write_file)
        write_result = tools.write_file(test_filename, test_content)
        assert write_result["success"] is True
        assert isinstance(write_result.get("file_id"), str)

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

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_write_with_output_prefix(self, tmp_path):
        """Test that writing with 'output/' prefix doesn't create duplicate directories."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Test content with output prefix"

        # Write with output/ prefix - should go to workspace/output/banner.html
        # NOT workspace/output/output/banner.html
        write_result = tools.write_file("output/banner.html", test_content)
        assert write_result["success"] is True
        assert isinstance(write_result.get("file_id"), str)

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

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_write_with_input_prefix(self, tmp_path):
        """Test that writing with 'input/' prefix works correctly."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Test content with input prefix"

        # Write with input/ prefix
        write_result = tools.write_file("input/data.txt", test_content)
        assert write_result["success"] is True
        assert isinstance(write_result.get("file_id"), str)

        # Verify file is in workspace/input/data.txt
        expected_file = workspace.input_dir / "data.txt"
        assert expected_file.exists()
        assert expected_file.read_text() == test_content

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_write_with_temp_prefix(self, tmp_path):
        """Test that writing with 'temp/' prefix works correctly."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Test content with temp prefix"

        # Write with temp/ prefix
        write_result = tools.write_file("temp/cache.txt", test_content)
        assert write_result["success"] is True
        assert isinstance(write_result.get("file_id"), str)

        expected_file = workspace.temp_dir / "cache.txt"
        assert expected_file.exists()
        assert expected_file.read_text() == test_content

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_by_file_id(self, tmp_path):
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Read by file_id"
        result = tools.write_file("output/read_by_id.txt", test_content)
        file_id = result["file_id"]

        read_content = tools.read_file(file_id)
        assert read_content == test_content

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_by_file_link_prefix(self, tmp_path):
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Read by file:file_id"
        result = tools.write_file("output/read_by_link_id.txt", test_content)
        file_id = result["file_id"]

        read_content = tools.read_file(f"file:{file_id}")
        assert read_content == test_content

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_prepare_html_asset_copies_file_id_to_output_assets(self, tmp_path):
        """Test that file_id assets get copied into the output HTML bundle."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        source = tools.write_file("input/logo.png", "fake image")
        result = tools.prepare_html_asset(source["file_id"], "index.html")

        assert result["success"] is True
        assert result["source_file_id"] == source["file_id"]
        assert isinstance(result["asset_file_id"], str)
        assert result["html_src"] == "assets/logo.png"
        assert result["filename"] == "logo.png"
        assert result["mime_type"] == "image/png"
        assert result["relative_path"] == "output/assets/logo.png"
        assert (workspace.output_dir / "assets" / "logo.png").read_text() == (
            "fake image"
        )
        assert tools.read_file(f"file:{result['asset_file_id']}") == "fake image"

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_prepare_html_asset_accepts_file_link_prefix(self, tmp_path):
        """Test that file:file_id references are accepted for HTML assets."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        source = tools.write_file("input/photo.jpg", "fake jpg")
        result = tools.prepare_html_asset(f"file:{source['file_id']}", "index.html")

        assert result["success"] is True
        assert result["html_src"] == "assets/photo.jpg"
        assert (workspace.output_dir / "assets" / "photo.jpg").exists()

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_prepare_html_asset_sanitizes_alias(self, tmp_path):
        """Test that aliases cannot escape the assets directory."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        source = tools.write_file("input/logo.png", "fake image")
        result = tools.prepare_html_asset(
            source["file_id"], "index.html", alias="../../safe.png"
        )

        assert result["html_src"] == "assets/safe.png"
        assert result["relative_path"] == "output/assets/safe.png"
        assert (workspace.output_dir / "assets" / "safe.png").exists()
        assert not (workspace.output_dir.parent / "safe.png").exists()

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize(
        "assets_subdir",
        ["/assets", "../assets", "assets/../../x"],
    )
    def test_prepare_html_asset_rejects_unsafe_assets_subdir(
        self, tmp_path, assets_subdir
    ):
        """Test that the assets directory must stay inside output."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        source = tools.write_file("input/logo.png", "fake image")

        with pytest.raises(ValueError, match="assets_subdir must"):
            tools.prepare_html_asset(
                source["file_id"], "index.html", assets_subdir=assets_subdir
            )

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_prepare_html_asset_uses_html_path_for_relative_src(self, tmp_path):
        """Test that nested HTML outputs get local relative asset paths."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        source = tools.write_file("input/logo.png", "fake image")
        result = tools.prepare_html_asset(
            source["file_id"], "reports/index.html", alias="logo.png"
        )

        assert result["html_src"] == "assets/logo.png"
        assert result["relative_path"] == "output/reports/assets/logo.png"
        assert (workspace.output_dir / "reports" / "assets" / "logo.png").exists()

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize(
        "html_path", ["/index.html", "../index.html", "input/x.html"]
    )
    def test_prepare_html_asset_rejects_unsafe_html_path(self, tmp_path, html_path):
        """Test that HTML target paths must stay inside output."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        source = tools.write_file("input/logo.png", "fake image")

        with pytest.raises(ValueError, match="html_path must"):
            tools.prepare_html_asset(source["file_id"], html_path)

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_prepare_html_asset_rejects_missing_file_ref(self, tmp_path):
        """Test that None file refs are not coerced into a filename."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        with pytest.raises(FileNotFoundError, match="File not found"):
            tools.prepare_html_asset(None, "index.html")  # type: ignore[arg-type]

    def test_resolve_file_id_rejects_other_user_records(self, tmp_path, mocker):
        """Test that DB file_id lookup is scoped to the workspace owner."""
        external_file = tmp_path / "other-user.txt"
        external_file.write_text("private")
        workspace = TaskWorkspace("web_task_10", str(tmp_path))
        workspace.owner_user_id = 1

        class FakeQuery:
            def filter(self, *_args):
                return self

            def first(self):
                return SimpleNamespace(
                    file_id="foreign-file",
                    user_id=2,
                    task_id=None,
                    storage_path=str(external_file),
                )

        class FakeSession:
            def query(self, *_args):
                return FakeQuery()

            def close(self):
                pass

        mocker.patch(
            "xagent.core.storage.manager.create_db_session",
            return_value=FakeSession(),
        )

        assert workspace.resolve_file_id("foreign-file") is None

    def test_resolve_file_id_rejects_durable_only_other_user_records(
        self, tmp_path, mocker
    ):
        """Test durable-only DB file_id lookup is scoped to the workspace owner."""
        missing_local = tmp_path / "missing-other-user.txt"
        workspace = TaskWorkspace("web_task_10", str(tmp_path))
        workspace.owner_user_id = 1

        class FakeQuery:
            def filter(self, *_args):
                return self

            def first(self):
                return SimpleNamespace(
                    file_id="foreign-file",
                    user_id=2,
                    task_id=None,
                    storage_path=str(missing_local),
                    storage_key="users/2/uploads/foreign-file/private.txt",
                    storage_status="available",
                )

        class FakeSession:
            def query(self, *_args):
                return FakeQuery()

            def close(self):
                pass

        materialize_calls = []

        class SpyManagedFileRef:
            def __init__(self, *_args, **_kwargs):
                pass

            def materialize(self):
                materialize_calls.append(None)
                return missing_local

        mocker.patch(
            "xagent.core.storage.manager.create_db_session",
            return_value=FakeSession(),
        )
        mocker.patch(
            "xagent.web.services.managed_file_ref.ManagedFileRef",
            SpyManagedFileRef,
        )

        assert workspace.resolve_file_id("foreign-file") is None
        assert materialize_calls == []

    def test_resolve_file_id_rejects_durable_only_other_user_inside_workspace_path(
        self, tmp_path, mocker
    ):
        """Test durable-only authorization does not trust stale workspace paths."""
        workspace = TaskWorkspace("web_task_10", str(tmp_path))
        workspace.owner_user_id = 1
        missing_local = workspace.output_dir / "private.txt"
        assert not missing_local.exists()

        class FakeQuery:
            def filter(self, *_args):
                return self

            def first(self):
                return SimpleNamespace(
                    file_id="foreign-file",
                    user_id=2,
                    task_id=None,
                    storage_path=str(missing_local),
                    storage_key="users/2/uploads/foreign-file/private.txt",
                    storage_status="available",
                )

        class FakeSession:
            def query(self, *_args):
                return FakeQuery()

            def close(self):
                pass

        materialize_calls = []

        class SpyManagedFileRef:
            def __init__(self, *_args, **_kwargs):
                pass

            def materialize(self):
                materialize_calls.append(None)
                return missing_local

        mocker.patch(
            "xagent.core.storage.manager.create_db_session",
            return_value=FakeSession(),
        )
        mocker.patch(
            "xagent.web.services.managed_file_ref.ManagedFileRef",
            SpyManagedFileRef,
        )

        assert workspace.resolve_file_id("foreign-file") is None
        assert materialize_calls == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
