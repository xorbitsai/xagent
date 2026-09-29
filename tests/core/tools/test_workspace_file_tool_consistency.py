"""
Tests for workspace file tool consistency between write and read operations.
"""

import contextlib
import hashlib
import json
import os
import threading
from types import SimpleNamespace

import pytest

from xagent.core.tools import tool_result_spill as spill_module
from xagent.core.tools.adapters.vibe.workspace_file_tool import WorkspaceFileTools
from xagent.core.tools.tool_result_spill import (
    SPILL_MAX_FILE_BYTES,
    SPILL_MAX_FILES_PER_RUN,
    SPILL_READ_MAX_CHARS,
    SPILL_READ_TOOL_NAME,
    SPILL_READ_TRUNCATED_INSTRUCTION,
    _write_spill_file,
    spill_dir_for_workspace,
    spill_read_unavailable,
)
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

    @pytest.mark.parametrize("state", ["available", "detached", "compensating"])
    def test_resolve_file_id_detached_uses_worker_owned_session(
        self, tmp_path, mocker, state
    ):
        """Detached resolution must not reuse the caller's SQLAlchemy session."""
        registered_file = tmp_path / "registered.txt"
        registered_file.write_text("content")
        workspace = TaskWorkspace("web_task_10", str(tmp_path))

        class BoundSession:
            def query(self, *_args):
                raise AssertionError("caller-owned session must not be reused")

        class FakeQuery:
            def filter(self, *_args):
                return self

            def first(self):
                return SimpleNamespace(
                    file_id="registered-file",
                    user_id=1,
                    task_id=None,
                    detached_reason="task_deleted" if state == "detached" else None,
                    storage_status="compensating"
                    if state == "compensating"
                    else "available",
                    storage_path=str(registered_file),
                )

        class WorkerSession:
            closed = False

            def query(self, *_args):
                return FakeQuery()

            def close(self):
                self.closed = True

        worker_session = WorkerSession()
        workspace.db_session = BoundSession()
        mocker.patch(
            "xagent.core.storage.manager.create_db_session",
            return_value=worker_session,
        )

        assert workspace.resolve_file_id_detached("registered-file") == (
            registered_file if state == "available" else None
        )
        assert worker_session.closed is True

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


class TestReadToolResult:
    """Tests for the read_tool_result tool."""

    @staticmethod
    def _spill_file(workspace, content, kind="text", tool_name="acme"):
        """Store content the way the spill writer does; return its path."""
        return _write_spill_file(
            spill_dir_for_workspace(workspace.workspace_dir), tool_name, content, kind
        )

    @staticmethod
    def _plant(workspace, name, raw):
        """Put raw bytes under the spill directory by hand, name unchecked."""
        spill_dir = workspace.output_dir / "tool-results"
        spill_dir.mkdir(parents=True, exist_ok=True)
        (spill_dir / name).write_bytes(raw)
        return f"tool-results/{name}"

    @staticmethod
    def _digest(raw):
        return hashlib.sha256(raw).hexdigest()[:32]

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_returns_whole_file_without_a_range(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, "[1, 2, 3]", kind="array")

        result = tools.read_tool_result(rel)
        assert result == {"relative_path": rel, "output": "[1, 2, 3]"}

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_slices_array_by_item(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps(list(range(1, 11))), "array")

        result = tools.read_tool_result(rel, start=3, end=5)
        assert json.loads(result["output"]) == [3, 4, 5]

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_slices_object_preserving_document_order(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        value = {f"k{i}": i for i in range(10)}
        rel = self._spill_file(workspace, json.dumps(value), "object")

        result = tools.read_tool_result(rel, start=2, end=3)
        assert list(json.loads(result["output"]).keys()) == ["k1", "k2"]

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_slices_text_by_physical_line(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, "line1\nline2\nline3\n")

        result = tools.read_tool_result(rel, start=2, end=2)
        assert result["output"] == "line2\n"

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_kind_is_decided_by_content_not_extension(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        # .json extension, but the content is not JSON.
        rel = self._spill_file(workspace, "not json\nsecond line\n", kind="array")
        assert rel.endswith(".json")

        result = tools.read_tool_result(rel, start=1, end=1)
        assert result["output"] == "not json\n"

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_crlf_text_line_count_matches_narrow_split(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, "x\r\ny")

        whole = tools.read_tool_result(rel)
        assert whole["output"] == "x\r\ny"
        first_line = tools.read_tool_result(rel, start=1, end=1)
        assert first_line["output"] == "x\r\n"
        second_line = tools.read_tool_result(rel, start=2, end=2)
        assert second_line["output"] == "y"

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize("separator", ["\r", "\x0b", "\u2028"])
    def test_read_tool_result_counts_lines_by_newline_only(self, tmp_path, separator):
        """Only \\n ends a line, as in the notice's item_count; str.splitlines
        would also break on each of these and count two."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        content = f"a{separator}b"
        assert len(content.splitlines()) == 2
        rel = self._spill_file(workspace, content)

        assert tools.read_tool_result(rel, start=1, end=1)["output"] == content
        assert tools.read_tool_result(rel, start=2)["output"] == (
            "start exceeds the item count (1)."
        )

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_end_beyond_item_count_is_clamped(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps([1, 2, 3]), "array")

        result = tools.read_tool_result(rel, start=2, end=100)
        assert json.loads(result["output"]) == [2, 3]

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_open_ended_ranges_default_to_the_ends(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps([1, 2, 3, 4]), "array")

        assert json.loads(tools.read_tool_result(rel, end=2)["output"]) == [1, 2]
        assert json.loads(tools.read_tool_result(rel, start=3)["output"]) == [3, 4]

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_start_beyond_item_count_is_invalid_range(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps([1, 2, 3]), "array")

        result = tools.read_tool_result(rel, start=10)
        assert result == spill_read_unavailable("invalid_range", item_count=3)
        assert result["output"] == "start exceeds the item count (3)."

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_empty_file_has_no_first_item(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, "")

        assert tools.read_tool_result(rel) == {"relative_path": rel, "output": ""}
        result = tools.read_tool_result(rel, start=1)
        assert result["output"] == "start exceeds the item count (0)."

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize(
        "start, end, offset",
        [
            (0, None, 0),
            (None, 0, 0),
            (3, 1, 0),
            (-1, 2, 0),
            (None, None, -1),
            (1, 2, -1),
        ],
    )
    def test_read_tool_result_rejects_invalid_ranges_without_raising(
        self, tmp_path, start, end, offset
    ):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps([1, 2, 3]), "array")

        result = tools.read_tool_result(rel, start=start, end=end, offset=offset)
        assert result == spill_read_unavailable("invalid_range")

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_without_a_range_does_not_parse_the_file(
        self, tmp_path, mocker
    ):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps([1, 2, 3]), "array")
        kind_of = mocker.patch.object(
            spill_module, "_spill_kind_of", side_effect=AssertionError
        )

        assert tools.read_tool_result(rel)["output"] == "[1, 2, 3]"
        kind_of.assert_not_called()

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_does_not_fold_a_slice_error_into_a_rejection(
        self, tmp_path, mocker
    ):
        """A ValueError out of _spill_slice is a caller bug, not a model
        mistake, so it propagates instead of reading as invalid_range."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps([1, 2, 3]), "array")
        mocker.patch.object(
            spill_module,
            "_spill_slice",
            side_effect=ValueError("caller bug"),
        )

        with pytest.raises(ValueError, match="caller bug"):
            tools.read_tool_result(rel, start=1, end=2)

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_over_limit_output_self_truncates_without_output_key(
        self, tmp_path
    ):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, "y" * 20_000)

        result = tools.read_tool_result(rel)
        assert result == {
            "relative_path": rel,
            "content_preview": "y" * SPILL_READ_MAX_CHARS,
            "content_truncated": True,
            "original_chars": 20_000,
            "item_count": 1,
            "offset": 0,
            "instruction": SPILL_READ_TRUNCATED_INSTRUCTION,
        }

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize("length", [12_000, 12_001, 60_000])
    def test_read_tool_result_cap_applies_at_exactly_the_limit(self, tmp_path, length):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, "y" * length)

        result = tools.read_tool_result(rel)
        if length <= SPILL_READ_MAX_CHARS:
            assert result == {"relative_path": rel, "output": "y" * length}
        else:
            assert "output" not in result
            assert len(result["content_preview"]) == SPILL_READ_MAX_CHARS
            assert result["original_chars"] == length

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_caps_an_over_limit_slice_too(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps(["z" * 5_000] * 4), "array")

        small = tools.read_tool_result(rel, start=1, end=1)
        large = tools.read_tool_result(rel, start=1, end=3)
        assert json.loads(small["output"]) == ["z" * 5_000]
        assert "output" not in large
        assert large["content_truncated"] is True
        assert large["original_chars"] == len(json.dumps(["z" * 5_000] * 3))

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_offset_reads_one_long_line_to_its_end(self, tmp_path):
        """One line of 30,000 characters is a single item, so no start/end
        range is narrower than it; offset reads it in three calls with the
        same range, and the three pieces are the line."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        line = "".join(chr(ord("a") + index % 26) for index in range(30_000))
        rel = self._spill_file(workspace, line)

        pieces = [
            tools.read_tool_result(rel, start=1, end=1, offset=offset)
            for offset in (0, 12_000, 24_000)
        ]

        assert [piece["offset"] for piece in pieces] == [0, 12_000, 24_000]
        assert [len(piece["content_preview"]) for piece in pieces] == [
            12_000,
            12_000,
            6_000,
        ]
        assert [piece["content_truncated"] for piece in pieces] == [
            True,
            True,
            False,
        ]
        assert [piece.get("instruction") for piece in pieces] == [
            SPILL_READ_TRUNCATED_INSTRUCTION,
            SPILL_READ_TRUNCATED_INSTRUCTION,
            None,
        ]
        assert all(piece["item_count"] == 1 for piece in pieces)
        assert all(piece["original_chars"] == 30_000 for piece in pieces)
        assert all("output" not in piece for piece in pieces)
        assert "".join(piece["content_preview"] for piece in pieces) == line

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_offset_continues_inside_one_large_array_element(
        self, tmp_path
    ):
        """The same holds for one oversized JSON element: offset moves
        through the rendered text of the selected items, and item_count is
        the stored result's own item count."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps(["z" * 20_000, 1, 2]), "array")
        rendered = json.dumps(["z" * 20_000])

        first = tools.read_tool_result(rel, start=1, end=1)
        rest = tools.read_tool_result(rel, start=1, end=1, offset=SPILL_READ_MAX_CHARS)

        assert first["item_count"] == rest["item_count"] == 3
        assert first["original_chars"] == rest["original_chars"] == len(rendered)
        assert first["content_truncated"] is True
        assert rest["content_truncated"] is False
        assert first["content_preview"] + rest["content_preview"] == rendered

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_offset_must_name_a_character_that_exists(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, "abc")
        empty = self._spill_file(workspace, "", tool_name="empty")

        assert tools.read_tool_result(rel, offset=2) == {
            "relative_path": rel,
            "content_preview": "c",
            "content_truncated": False,
            "original_chars": 3,
            "item_count": 1,
            "offset": 2,
        }
        assert tools.read_tool_result(rel, offset=3) == spill_read_unavailable(
            "invalid_range"
        )
        assert tools.read_tool_result(empty, offset=0) == {
            "relative_path": empty,
            "output": "",
        }
        assert tools.read_tool_result(empty, offset=1) == spill_read_unavailable(
            "invalid_range"
        )

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize("path", [None, "  "])
    def test_read_tool_result_listing_rejects_an_offset(self, tmp_path, path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        self._spill_file(workspace, "a" * 10)

        assert tools.read_tool_result(path, offset=5) == spill_read_unavailable(
            "invalid_range"
        )
        assert tools.read_tool_result(path, offset=0)["count"] == 1

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_truncated_instruction_reaches_model_visible_content(
        self, tmp_path
    ):
        from xagent.core.agent.context.execution import ExecutionContext

        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, "y" * 20_000)

        raw = tools.read_tool_result(rel)
        ctx = ExecutionContext()
        message = ctx.add_tool_result(SPILL_READ_TOOL_NAME, raw)
        assert SPILL_READ_TRUNCATED_INSTRUCTION in message.content

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_never_falls_back_to_input_dir(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        # A same-stem file sits in input/ -- read_file's fuzzy matching would
        # find it; read_tool_result must not.
        input_file = workspace.input_dir / "data.json"
        input_file.write_text('{"secret": "user upload"}', encoding="utf-8")

        result = tools.read_tool_result(
            "tool-results/data-00000000000000000000000000000000.json"
        )
        assert result == spill_read_unavailable("not_found")

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize(
        "path",
        ["input/data.json", "data.json", "tool-results/../input/data.json", 7],
    )
    def test_read_tool_result_rejects_a_path_outside_the_spill_directory(
        self, tmp_path, path
    ):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        assert tools.read_tool_result(path) == spill_read_unavailable("invalid_path")

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_missing_file_is_not_found(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        (workspace.output_dir / "tool-results").mkdir(parents=True, exist_ok=True)

        result = tools.read_tool_result(
            "tool-results/missing-00000000000000000000000000000000.json"
        )
        assert result == spill_read_unavailable("not_found")

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_with_no_spill_directory_is_not_found(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        assert not (workspace.output_dir / "tool-results").exists()

        result = tools.read_tool_result(
            "tool-results/acme-00000000000000000000000000000000.json"
        )
        assert result == spill_read_unavailable("not_found")

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_broken_symlink_is_not_found(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        spill_dir = workspace.output_dir / "tool-results"
        spill_dir.mkdir(parents=True, exist_ok=True)
        name = "gone-00000000000000000000000000000000.json"
        (spill_dir / name).symlink_to(spill_dir / "missing-target.json")

        result = tools.read_tool_result(f"tool-results/{name}")
        assert result == spill_read_unavailable("not_found")

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_directory_is_not_a_file(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        (workspace.output_dir / "tool-results" / "dir.json").mkdir(parents=True)

        result = tools.read_tool_result("tool-results/dir.json")
        assert result == spill_read_unavailable("not_found")

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_symlink_escape_is_rejected(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        spill_dir = workspace.output_dir / "tool-results"
        spill_dir.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / "outside.json"
        outside.write_text("[1]", encoding="utf-8")
        name = f"link-{self._digest(b'[1]')}.json"
        (spill_dir / name).symlink_to(outside)

        result = tools.read_tool_result(f"tool-results/{name}")
        assert result == spill_read_unavailable("not_found")

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_output_prefix_normalizes_to_a_hit(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, "[1]", "array")

        result = tools.read_tool_result(f"output/{rel}")
        assert result == {"relative_path": rel, "output": "[1]"}

    # --- the digest in the file name is checked against the bytes -----------

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_reads_a_file_whose_bytes_match_its_name(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps([1, 2, 3]), "array")
        stored = workspace.output_dir / rel

        assert rel.split("-")[-1].split(".")[0] == self._digest(stored.read_bytes())
        assert tools.read_tool_result(rel) == {
            "relative_path": rel,
            "output": "[1, 2, 3]",
        }

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_rejects_a_file_changed_after_it_was_written(
        self, tmp_path
    ):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps([1, 2, 3]), "array")
        stored = workspace.output_dir / rel
        raw = bytearray(stored.read_bytes())
        raw[1] = ord("9")  # one byte: "[1, ..." becomes "[9, ..."
        stored.write_bytes(bytes(raw))

        assert tools.read_tool_result(rel) == spill_read_unavailable("not_found")
        assert tools.read_tool_result(rel, start=1, end=1) == (
            spill_read_unavailable("not_found")
        )

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_rejects_a_name_whose_digest_does_not_match(
        self, tmp_path
    ):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        rel = self._spill_file(workspace, json.dumps([1, 2, 3]), "array")
        stored = workspace.output_dir / rel
        stem, extension = stored.name.rsplit(".", 1)
        flipped = "0" if stem[-1] != "0" else "1"
        renamed = stored.with_name(f"{stem[:-1]}{flipped}.{extension}")
        stored.rename(renamed)

        result = tools.read_tool_result(f"tool-results/{renamed.name}")
        assert result == spill_read_unavailable("not_found")

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_rejects_a_name_with_no_digest_segment(self, tmp_path):
        """A name the writer never produces -- no "-<digest>" -- is not read
        even when the whole stem happens to be the digest of the bytes."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        raw = b"[1, 2, 3]"
        rel = self._plant(workspace, f"{self._digest(raw)}.json", raw)

        assert tools.read_tool_result(rel) == spill_read_unavailable("not_found")

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_hashes_the_raw_bytes_before_decoding_them(self, tmp_path):
        """Bytes that are not valid UTF-8 still match a name derived from
        them; decoding first would replace them and hash something else."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        raw = b"ok\xff\n"
        rel = self._plant(workspace, f"acme-{self._digest(raw)}.txt", raw)

        assert tools.read_tool_result(rel) == {
            "relative_path": rel,
            "output": "ok\ufffd\n",
        }

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_size_gate_sits_at_the_writer_file_cap(self, tmp_path):
        """A file at exactly the writer's cap is read; one byte more cannot be
        a stored result, so it is reported unavailable even though its name
        carries the right digest."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        at_cap = b"a" * SPILL_MAX_FILE_BYTES
        over_cap = at_cap + b"a"
        at_cap_path = self._plant(workspace, f"cap-{self._digest(at_cap)}.txt", at_cap)
        over_cap_path = self._plant(
            workspace, f"over-{self._digest(over_cap)}.txt", over_cap
        )

        at_cap_result = tools.read_tool_result(at_cap_path)
        assert at_cap_result["content_truncated"] is True
        assert at_cap_result["original_chars"] == SPILL_MAX_FILE_BYTES
        assert tools.read_tool_result(over_cap_path) == spill_read_unavailable(
            "not_found"
        )

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_bounds_the_read_to_the_size_fstat_reported(
        self, tmp_path, monkeypatch, mocker
    ):
        """The size comes from the open descriptor, and it bounds the read.

        Right after the open, the path is pointed at a far larger file; right
        after fstat, the opened file itself grows by 1 MiB. The size gate has
        to see the small opened file (a size looked up by path would see the
        large one and refuse without reading), the read has to stop one byte
        past that size (an unbounded read would take the whole 1 MiB), and
        that extra byte has to be enough to report the file unavailable
        before any digest is computed.
        """
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        raw = b"[1, 2, 3]"
        rel = self._plant(workspace, f"acme-{self._digest(raw)}.json", raw)
        entry = workspace.output_dir / rel
        opened_inode = entry.with_name("opened-inode")
        growth = 1024 * 1024
        opened = []
        read_lengths = []
        real_open, real_fstat, real_fdopen = os.open, os.fstat, os.fdopen

        def open_then_repoint_path(path, flags, *args, **kwargs):
            descriptor = real_open(path, flags, *args, **kwargs)
            if os.fspath(path) == os.fspath(entry) and not opened:
                opened.append(descriptor)
                os.rename(entry, opened_inode)
                with entry.open("wb") as large:
                    large.truncate(8 * SPILL_MAX_FILE_BYTES)
            return descriptor

        def fstat_then_grow(descriptor):
            result = real_fstat(descriptor)
            if opened and descriptor == opened[0]:
                with opened_inode.open("ab") as handle:
                    handle.write(b"x" * growth)
            return result

        class _RecordingReader:
            def __init__(self, handle):
                self._handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                self._handle.close()

            def read(self, *args):
                data = self._handle.read(*args)
                read_lengths.append(len(data))
                return data

        def fdopen_recording(descriptor, *args, **kwargs):
            handle = real_fdopen(descriptor, *args, **kwargs)
            if opened and descriptor == opened[0]:
                return _RecordingReader(handle)
            return handle

        monkeypatch.setattr(spill_module.os, "open", open_then_repoint_path)
        monkeypatch.setattr(spill_module.os, "fstat", fstat_then_grow)
        monkeypatch.setattr(spill_module.os, "fdopen", fdopen_recording)
        sha256 = mocker.patch.object(
            spill_module.hashlib, "sha256", wraps=hashlib.sha256
        )

        result = tools.read_tool_result(rel)

        assert result == spill_read_unavailable("not_found")
        assert read_lengths == [len(raw) + 1]
        sha256.assert_not_called()

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")
    def test_read_tool_result_rejects_an_entry_swapped_for_a_fifo(
        self, tmp_path, monkeypatch
    ):
        """An entry replaced by a FIFO after the lookup is reported
        unavailable without the open waiting for a writer."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        raw = b"[1, 2, 3]"
        rel = self._plant(workspace, f"acme-{self._digest(raw)}.json", raw)
        real_resolve = spill_module.resolve_spilled_under
        swapped = []

        def resolve_then_swap(spill_dir, name):
            resolved = real_resolve(spill_dir, name)
            resolved.unlink()
            os.mkfifo(resolved)
            swapped.append(resolved)
            return resolved

        monkeypatch.setattr(spill_module, "resolve_spilled_under", resolve_then_swap)
        outcome = []
        reader = threading.Thread(
            target=lambda: outcome.append(tools.read_tool_result(rel)), daemon=True
        )
        reader.start()
        reader.join(timeout=5)
        if reader.is_alive():
            # Release the blocked open so the thread does not outlive the
            # test, then fail on what the test is about.
            os.close(os.open(swapped[0], os.O_WRONLY | os.O_NONBLOCK))
            reader.join(timeout=5)
            pytest.fail("read_tool_result blocked opening a FIFO")

        assert outcome == [spill_read_unavailable("not_found")]

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_does_not_follow_an_entry_swapped_for_a_symlink(
        self, tmp_path, monkeypatch
    ):
        """An entry replaced by a symlink after the lookup is not followed,
        even when the link target holds exactly the bytes the name's digest
        describes: the open refuses the link itself."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        raw = b"[1, 2, 3]"
        rel = self._plant(workspace, f"acme-{self._digest(raw)}.json", raw)
        same_bytes = tmp_path / "same-bytes.json"
        same_bytes.write_bytes(raw)
        real_resolve = spill_module.resolve_spilled_under

        def resolve_then_swap(spill_dir, name):
            resolved = real_resolve(spill_dir, name)
            resolved.unlink()
            resolved.symlink_to(same_bytes)
            return resolved

        monkeypatch.setattr(spill_module, "resolve_spilled_under", resolve_then_swap)

        assert tools.read_tool_result(rel) == spill_read_unavailable("not_found")

    # --- no path: list the stored results -----------------------------------

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_lists_nothing_when_nothing_was_stored(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        empty = {
            "stored_results": [],
            "count": 0,
            "start": None,
            "end": None,
            "omitted": 0,
        }

        assert tools.read_tool_result() == empty
        (workspace.output_dir / "tool-results").mkdir(parents=True)
        assert tools.read_tool_result() == empty

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_lists_nothing_when_the_spill_path_is_a_file(
        self, tmp_path
    ):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        workspace.output_dir.mkdir(parents=True, exist_ok=True)
        (workspace.output_dir / "tool-results").write_text("x", encoding="utf-8")

        assert tools.read_tool_result() == {
            "stored_results": [],
            "count": 0,
            "start": None,
            "end": None,
            "omitted": 0,
        }

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_lists_only_pattern_matching_regular_files(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        second = self._spill_file(workspace, "b" * 50, tool_name="zeta")
        first = self._spill_file(workspace, json.dumps([1, 2]), "array", "alpha")
        spill_dir = workspace.output_dir / "tool-results"
        (spill_dir / "not a spill name.json").write_text("x", encoding="utf-8")
        (spill_dir / "notes.md").write_text("x", encoding="utf-8")
        (spill_dir / "trailing-space.json ").write_text("x", encoding="utf-8")
        (spill_dir / "subdir-00000000000000000000000000000000.json").mkdir()
        outside = tmp_path / "outside.json"
        outside.write_text("[1]", encoding="utf-8")
        (spill_dir / "link-00000000000000000000000000000000.json").symlink_to(outside)

        listing = tools.read_tool_result(None)

        assert listing == {
            "stored_results": [
                {
                    "relative_path": first,
                    "bytes": (workspace.output_dir / first).stat().st_size,
                },
                {
                    "relative_path": second,
                    "bytes": (workspace.output_dir / second).stat().st_size,
                },
            ],
            "count": 2,
            "start": 1,
            "end": 2,
            "omitted": 0,
        }
        # Every listed path reads back through the same tool.
        for entry in listing["stored_results"]:
            assert "output" in tools.read_tool_result(entry["relative_path"])

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_read_tool_result_blank_path_lists_like_no_path(self, tmp_path, blank):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        self._spill_file(workspace, "a" * 10)

        assert tools.read_tool_result(blank) == tools.read_tool_result()
        assert tools.read_tool_result(blank)["count"] == 1

    @staticmethod
    def _plant_listing(workspace, how_many):
        spill_dir = workspace.output_dir / "tool-results"
        spill_dir.mkdir(parents=True, exist_ok=True)
        for index in range(how_many):
            (spill_dir / f"t{index:03d}-{index:032d}.json").write_bytes(b"")
        return [
            f"tool-results/t{index:03d}-{index:032d}.json" for index in range(how_many)
        ]

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_listing_is_capped_and_counts_the_rest(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        paths = self._plant_listing(workspace, SPILL_MAX_FILES_PER_RUN + 1)

        listing = tools.read_tool_result()

        assert SPILL_MAX_FILES_PER_RUN == 64
        assert listing["count"] == 65
        assert (listing["start"], listing["end"], listing["omitted"]) == (1, 64, 1)
        listed = [entry["relative_path"] for entry in listing["stored_results"]]
        assert listed == paths[:64]

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_listing_pages_past_the_cap(self, tmp_path):
        """Every stored file is reachable: the entries past the first page
        are read by asking for the next page."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        paths = self._plant_listing(workspace, 150)

        first = tools.read_tool_result()
        second = tools.read_tool_result(start=first["end"] + 1)
        third = tools.read_tool_result(start=second["end"] + 1)

        assert [(page["start"], page["end"]) for page in (first, second, third)] == [
            (1, 64),
            (65, 128),
            (129, 150),
        ]
        assert [page["count"] for page in (first, second, third)] == [150] * 3
        assert [page["omitted"] for page in (first, second, third)] == [86, 86, 128]
        listed = [
            entry["relative_path"]
            for page in (first, second, third)
            for entry in page["stored_results"]
        ]
        assert listed == paths

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize(
        "start, end, expected_bounds",
        [
            (3, 5, (3, 5)),
            (None, 2, (1, 2)),
            (8, None, (8, 10)),
            (9, 100, (9, 10)),
            (10, 10, (10, 10)),
        ],
    )
    def test_read_tool_result_listing_returns_the_requested_entries(
        self, tmp_path, start, end, expected_bounds
    ):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        paths = self._plant_listing(workspace, 10)

        listing = tools.read_tool_result(start=start, end=end)

        first, last = expected_bounds
        assert (listing["start"], listing["end"]) == expected_bounds
        assert listing["count"] == 10
        assert listing["omitted"] == 10 - (last - first + 1)
        assert [entry["relative_path"] for entry in listing["stored_results"]] == paths[
            first - 1 : last
        ]

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_listing_page_is_cut_to_the_cap(self, tmp_path):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        self._plant_listing(workspace, 100)

        listing = tools.read_tool_result(start=2, end=100)

        assert (listing["start"], listing["end"]) == (2, 65)
        assert len(listing["stored_results"]) == SPILL_MAX_FILES_PER_RUN
        assert listing["omitted"] == 36

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize(
        "path, start, end",
        [(None, 0, None), (None, None, 0), ("  ", 3, 2), (None, -1, 2)],
    )
    def test_read_tool_result_listing_rejects_an_invalid_range(
        self, tmp_path, path, start, end
    ):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        self._plant_listing(workspace, 3)

        result = tools.read_tool_result(path, start=start, end=end)
        assert result == spill_read_unavailable("invalid_range")

    @pytest.mark.usefixtures("mock_workspace_db")
    @pytest.mark.parametrize("stored", [0, 3])
    def test_read_tool_result_listing_start_past_the_last_entry_names_the_count(
        self, tmp_path, stored
    ):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        self._plant_listing(workspace, stored)

        result = tools.read_tool_result(start=stored + 1)
        assert result == spill_read_unavailable("invalid_range", item_count=stored)
        assert result["output"] == f"start exceeds the item count ({stored})."

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_read_tool_result_listing_skips_an_entry_whose_stat_fails(
        self, tmp_path, mocker
    ):
        """One entry that cannot be stat'ed is left out; the rest are still
        listed rather than the whole listing coming back empty."""
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        kept_first = self._spill_file(workspace, "a" * 10, tool_name="alpha")
        failing = self._spill_file(workspace, "b" * 10, tool_name="beta")
        kept_second = self._spill_file(workspace, "c" * 10, tool_name="gamma")
        failing_name = failing.split("/", 1)[1]
        real_scandir = os.scandir

        class _StatFailingEntry:
            def __init__(self, entry):
                self._entry = entry
                self.name = entry.name

            def is_file(self, *, follow_symlinks=True):
                return self._entry.is_file(follow_symlinks=follow_symlinks)

            def stat(self, *, follow_symlinks=True):
                if self.name == failing_name:
                    raise PermissionError(13, "Permission denied", self.name)
                return self._entry.stat(follow_symlinks=follow_symlinks)

        @contextlib.contextmanager
        def scandir_with_one_failing_stat(path):
            with real_scandir(path) as entries:
                yield (_StatFailingEntry(entry) for entry in entries)

        mocker.patch.object(spill_module.os, "scandir", scandir_with_one_failing_stat)

        listing = tools.read_tool_result()

        assert [entry["relative_path"] for entry in listing["stored_results"]] == [
            kept_first,
            kept_second,
        ]
        assert listing["count"] == 2
        assert (listing["start"], listing["end"], listing["omitted"]) == (1, 2, 0)

    def test_read_tool_result_listing_authority_failure_raises(self, tmp_path, mocker):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        mocker.patch.object(
            workspace,
            "requires_exact_file_operation_scope",
            side_effect=RuntimeError("boom"),
        )

        with pytest.raises(ValueError, match="File Operation unavailable"):
            tools.read_tool_result()

    def test_read_tool_result_calls_workspace_authority_check(self, tmp_path, mocker):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        spy = mocker.spy(tools.inner, "_require_workspace_authority")
        (workspace.output_dir / "tool-results").mkdir(parents=True, exist_ok=True)

        tools.read_tool_result(
            "tool-results/missing-00000000000000000000000000000000.json"
        )
        spy.assert_called_once()

    def test_read_tool_result_authority_failure_raises(self, tmp_path, mocker):
        workspace = TaskWorkspace("task-1", str(tmp_path))
        tools = WorkspaceFileTools(workspace)
        mocker.patch.object(
            workspace,
            "requires_exact_file_operation_scope",
            side_effect=RuntimeError("boom"),
        )

        with pytest.raises(ValueError, match="File Operation unavailable"):
            tools.read_tool_result(
                "tool-results/anything-00000000000000000000000000000000.json"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
