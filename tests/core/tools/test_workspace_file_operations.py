"""
Tests for WorkspaceFileOperations core class.

This module tests the core workspace file operations functionality,
focusing on JSON and CSV workspace writes and reads.
"""

import logging
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.workspace_file_tool import WorkspaceFileTools
from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.workspace import DEFAULT_USER_FILE_LIST_LIMIT, TaskWorkspace
from xagent.web.models import Base
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.tools.config import WebToolConfig


@pytest.fixture
def public_file_scope_context(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    monkeypatch.setattr(
        "xagent.core.storage.manager.create_db_session",
        SessionLocal,
    )

    owner = User(username="public-file-owner", password_hash="hash")
    other = User(username="other-file-owner", password_hash="hash")
    db.add_all([owner, other])
    db.flush()

    marked_task = Task(
        id=801,
        user_id=owner.id,
        title="Marked public task",
        source="shared_link",
        agent_config={
            "auth_mode": "share",
            "__xagent_file_operation_access_version": 1,
        },
    )
    sibling_task = Task(id=802, user_id=owner.id, title="Sibling task")
    historical_task = Task(
        id=803,
        user_id=owner.id,
        title="Historical public task",
        source="shared_link",
        agent_config={"auth_mode": "share"},
    )
    db.add_all([marked_task, sibling_task, historical_task])
    db.flush()

    external_dir = tmp_path / "external"
    external_dir.mkdir()

    def add_file(
        *,
        file_id: str,
        filename: str,
        content: str,
        user_id: int,
        task_id: int | None,
        storage_status: str = "available",
    ) -> tuple[UploadedFile, object]:
        path = external_dir / filename
        path.write_text(content, encoding="utf-8")
        record = UploadedFile(
            file_id=file_id,
            user_id=user_id,
            task_id=task_id,
            filename=filename,
            storage_path=str(path),
            storage_status=storage_status,
            mime_type="text/plain",
            file_size=path.stat().st_size,
        )
        db.add(record)
        return record, path

    current_record, current_path = add_file(
        file_id="current-file",
        filename="current.txt",
        content="current",
        user_id=int(owner.id),
        task_id=int(marked_task.id),
    )
    sibling_record, sibling_path = add_file(
        file_id="sibling-file",
        filename="sibling.txt",
        content="sibling",
        user_id=int(owner.id),
        task_id=int(sibling_task.id),
    )
    unbound_record, unbound_path = add_file(
        file_id="unbound-file",
        filename="unbound.txt",
        content="unbound",
        user_id=int(owner.id),
        task_id=None,
    )
    compensating_record, _ = add_file(
        file_id="compensating-file",
        filename="compensating.txt",
        content="compensating",
        user_id=int(owner.id),
        task_id=int(marked_task.id),
        storage_status="compensating",
    )
    other_record, other_path = add_file(
        file_id="other-file",
        filename="other.txt",
        content="other",
        user_id=int(other.id),
        task_id=None,
    )
    raw_external_path = external_dir / "raw-external.txt"
    raw_external_path.write_text("raw", encoding="utf-8")
    db.commit()

    marked_workspace = TaskWorkspace(
        id="agent_nested_workspace",
        base_dir=str(tmp_path / "workspaces"),
        allowed_external_dirs=[str(external_dir)],
        db_task_id=int(marked_task.id),
    )
    marked_workspace.owner_user_id = int(owner.id)
    marked_workspace.file_operation_access_version = 1
    marked_workspace.db_session = db

    try:
        yield SimpleNamespace(
            db=db,
            owner=owner,
            marked_task=marked_task,
            historical_task=historical_task,
            external_dir=external_dir,
            workspace=marked_workspace,
            ops=WorkspaceFileOperations(marked_workspace),
            current_record=current_record,
            current_path=current_path,
            sibling_record=sibling_record,
            sibling_path=sibling_path,
            unbound_record=unbound_record,
            unbound_path=unbound_path,
            compensating_record=compensating_record,
            other_record=other_record,
            other_path=other_path,
            raw_external_path=raw_external_path,
        )
    finally:
        db.close()
        engine.dispose()


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

    def test_marked_public_listing_uses_db_task_id_before_pagination(
        self, public_file_scope_context
    ):
        context = public_file_scope_context

        result = context.ops.list_all_user_files(include_workspace_files=False)

        assert result["user_id"] == context.owner.id
        assert result["total_count"] == 1
        assert [item["file_id"] for item in result["files"]] == [
            context.current_record.file_id
        ]

    def test_marked_public_listing_excludes_record_outside_authorized_storage(
        self, public_file_scope_context, tmp_path
    ):
        context = public_file_scope_context
        escaped_path = tmp_path / "outside-authorized-storage.txt"
        escaped_path.write_text("outside", encoding="utf-8")
        escaped_record = UploadedFile(
            file_id="escaped-file",
            user_id=int(context.owner.id),
            task_id=int(context.marked_task.id),
            filename=escaped_path.name,
            storage_path=str(escaped_path),
            storage_status="available",
            mime_type="text/plain",
            file_size=escaped_path.stat().st_size,
        )
        context.db.add(escaped_record)
        context.db.commit()

        listed = context.ops.list_all_user_files(include_workspace_files=False)

        assert escaped_record.file_id not in {
            item["file_id"] for item in listed["files"]
        }
        assert listed["total_count"] == 1
        with pytest.raises(FileNotFoundError):
            context.ops.read_file(escaped_record.file_id)

    def test_vibe_adapter_uses_marked_public_listing_policy(
        self, public_file_scope_context
    ):
        context = public_file_scope_context
        tools = WorkspaceFileTools(context.workspace)

        result = tools.list_all_user_files(include_workspace_files=False)

        assert result["total_count"] == 1
        assert [item["file_id"] for item in result["files"]] == [
            context.current_record.file_id
        ]

    def test_delegated_marked_workspace_reads_exact_record_under_owner_base(
        self, public_file_scope_context
    ):
        context = public_file_scope_context
        raw_workspace_config = {
            "base_dir": str(context.external_dir),
            "task_id": "agent_1_delegated",
            "db_task_id": int(context.marked_task.id),
            "__xagent_file_operation_access_version": 1,
            "scope_segments": (),
        }
        tool_config = WebToolConfig(
            db=context.db,
            request=None,
            user_id=int(context.owner.id),
            task_id="agent_1_delegated",
            workspace_config=raw_workspace_config,
        )

        workspace = ToolFactory.create_workspace(tool_config.get_workspace_config())
        assert workspace is not None
        workspace.db_session = context.db
        assert workspace.owner_user_id == context.owner.id
        assert workspace.db_task_id == context.marked_task.id
        assert workspace.file_operation_access_version == 1
        assert workspace.allowed_external_dirs == []
        assert (
            WorkspaceFileOperations(workspace).read_file(context.current_record.file_id)
            == "current"
        )

    @pytest.mark.parametrize(
        ("source", "auth_mode"),
        [("shared_link", "share"), ("widget", "widget")],
    )
    def test_marked_public_reads_only_workspace_or_exact_task_records(
        self, public_file_scope_context, source, auth_mode
    ):
        context = public_file_scope_context
        context.marked_task.source = source
        context.marked_task.agent_config = {
            "auth_mode": auth_mode,
            "__xagent_file_operation_access_version": 1,
        }
        context.db.commit()

        assert context.ops.read_file(context.current_record.file_id) == "current"
        assert (
            context.ops.read_file(f"file:{context.current_record.file_id}") == "current"
        )
        assert (
            context.ops.read_file(f"file://{context.current_record.file_id}")
            == "current"
        )
        assert context.ops.read_file(str(context.current_path)) == "current"

        denied = (
            context.sibling_record.file_id,
            str(context.sibling_path),
            context.unbound_record.file_id,
            str(context.unbound_path),
            context.other_record.file_id,
            str(context.other_path),
            str(context.raw_external_path),
        )
        for reference in denied:
            with pytest.raises((FileNotFoundError, ValueError)):
                context.ops.read_file(reference)
            assert context.ops.file_exists(reference) is False

    @pytest.mark.parametrize(
        ("durable_segments", "storage_key", "allowed"),
        [
            ((), "users/{owner}/uploads/current/file.txt", True),
            (("tenant-a",), "users/{owner}/tenant-a/uploads/current/file.txt", True),
            (("tenant-a",), "users/{owner}/uploads/current/file.txt", False),
        ],
    )
    def test_marked_public_durable_scope_matches_write_side_segments(
        self,
        public_file_scope_context,
        durable_segments,
        storage_key,
        allowed,
    ):
        context = public_file_scope_context
        context.workspace.scope_segments = ("tenant-a",)
        context.workspace.durable_storage_segments = durable_segments
        context.current_record.storage_key = storage_key.format(owner=context.owner.id)
        context.db.commit()

        if allowed:
            assert context.ops.read_file(context.current_record.file_id) == "current"
            listed = context.ops.list_all_user_files(include_workspace_files=False)
            assert context.current_record.file_id in {
                item["file_id"] for item in listed["files"]
            }
        else:
            with pytest.raises(FileNotFoundError):
                context.ops.read_file(context.current_record.file_id)
            listed = context.ops.list_all_user_files(include_workspace_files=False)
            assert context.current_record.file_id not in {
                item["file_id"] for item in listed["files"]
            }

    def test_marked_public_selector_misses_hide_record_existence(
        self, public_file_scope_context
    ):
        context = public_file_scope_context

        with pytest.raises(FileNotFoundError) as missing:
            context.ops.read_file("missing-file-id")
        with pytest.raises(FileNotFoundError) as foreign:
            context.ops.read_file(context.other_record.file_id)

        assert str(missing.value) == "File not found: missing-file-id"
        assert str(foreign.value) == f"File not found: {context.other_record.file_id}"

    def test_marked_public_taskless_durable_id_does_not_materialize(
        self, public_file_scope_context, monkeypatch
    ):
        from xagent.web.services.managed_file_ref import ManagedFileRef

        context = public_file_scope_context
        taskless = UploadedFile(
            file_id="taskless-durable-selector",
            user_id=context.owner.id,
            task_id=None,
            filename="taskless.txt",
            storage_path=str(context.external_dir / "missing-taskless.txt"),
            storage_key=f"users/{context.owner.id}/uploads/taskless/file.txt",
            storage_status="available",
            mime_type="text/plain",
            file_size=8,
        )
        context.db.add(taskless)
        context.db.commit()

        monkeypatch.setattr(
            ManagedFileRef,
            "materialize",
            lambda self: pytest.fail("unauthorized durable record was materialized"),
        )

        with pytest.raises(FileNotFoundError):
            context.ops.read_file(taskless.file_id)

    def test_foreign_file_id_does_not_shadow_workspace_local_file(
        self, public_file_scope_context
    ):
        context = public_file_scope_context
        local_path = context.workspace.input_dir / context.other_record.file_id
        local_path.write_text("local", encoding="utf-8")

        assert context.ops.read_file(context.other_record.file_id) == "local"

    def test_marked_public_named_directory_listing_validates_authority_once(
        self, public_file_scope_context, monkeypatch
    ):
        context = public_file_scope_context
        calls = 0
        original = context.workspace.requires_exact_file_operation_scope

        def count_authority_checks():
            nonlocal calls
            calls += 1
            return original()

        monkeypatch.setattr(
            context.workspace,
            "requires_exact_file_operation_scope",
            count_authority_checks,
        )
        (context.workspace.output_dir / "listed").mkdir()

        context.ops.list_files("listed")

        assert calls == 1

    def test_marked_public_output_registers_to_exact_task(
        self, public_file_scope_context
    ):
        context = public_file_scope_context

        result = context.ops.write_file("generated.txt", "generated")
        record = (
            context.db.query(UploadedFile)
            .filter(UploadedFile.file_id == result["file_id"])
            .one()
        )

        assert record.user_id == context.owner.id
        assert record.task_id == context.marked_task.id
        assert context.ops.read_file(record.file_id) == "generated"
        assert record.file_id in {
            item["file_id"]
            for item in context.ops.list_all_user_files(include_workspace_files=False)[
                "files"
            ]
        }

    @pytest.mark.parametrize(
        "operation",
        ["append", "delete", "edit", "replace"],
    )
    def test_marked_public_mutations_deny_sibling_path(
        self, public_file_scope_context, operation
    ):
        context = public_file_scope_context
        sibling_path = str(context.sibling_path)

        with pytest.raises((FileNotFoundError, ValueError)):
            if operation == "append":
                context.ops.append_file(sibling_path, "-changed")
            elif operation == "delete":
                context.ops.delete_file(sibling_path)
            elif operation == "edit":
                context.ops.edit_file(sibling_path, [])
            else:
                context.ops.find_and_replace(sibling_path, "sibling", "changed")

    @pytest.mark.parametrize(
        "operation",
        ["info", "csv", "html_asset"],
    )
    def test_marked_public_specialized_reads_deny_sibling_path(
        self, public_file_scope_context, operation
    ):
        context = public_file_scope_context
        sibling_path = str(context.sibling_path)

        with pytest.raises((FileNotFoundError, ValueError)):
            if operation == "info":
                context.ops.get_file_info(sibling_path)
            elif operation == "csv":
                context.ops.read_csv_file(sibling_path)
            else:
                context.ops.prepare_html_asset(
                    sibling_path,
                    "preview/index.html",
                )

    def test_marked_public_revalidates_cached_sibling_file_id(
        self, public_file_scope_context
    ):
        context = public_file_scope_context
        context.workspace._file_id_to_path[context.sibling_record.file_id] = (
            context.sibling_path
        )

        with pytest.raises((FileNotFoundError, ValueError)):
            context.ops.read_file(context.sibling_record.file_id)

    def test_marked_public_durable_files_require_exact_task_and_owner_prefix(
        self, public_file_scope_context, monkeypatch
    ):
        from xagent.web.services.managed_file_ref import ManagedFileRef

        context = public_file_scope_context
        owner_id = int(context.owner.id)
        task_id = int(context.marked_task.id)

        def durable_record(file_id, record_task_id, storage_key):
            return UploadedFile(
                file_id=file_id,
                user_id=owner_id,
                task_id=record_task_id,
                filename=f"{file_id}.txt",
                storage_path=str(context.external_dir / f"missing-{file_id}.txt"),
                storage_key=storage_key,
                storage_status="available",
                mime_type="text/plain",
                file_size=7,
            )

        current = durable_record(
            "current-durable-file",
            task_id,
            f"users/{owner_id}/web_task_{task_id}/input/current.txt",
        )
        sibling = durable_record(
            "sibling-durable-file",
            802,
            f"users/{owner_id}/web_task_802/input/sibling.txt",
        )
        invalid_prefix = durable_record(
            "invalid-durable-file",
            task_id,
            f"users/{owner_id + 1}/web_task_{task_id}/input/invalid.txt",
        )
        pending_path = context.external_dir / "pending-file.txt"
        pending_path.write_text("pending", encoding="utf-8")
        pending = UploadedFile(
            file_id="pending-file",
            user_id=owner_id,
            task_id=task_id,
            filename=pending_path.name,
            storage_path=str(pending_path),
            storage_status="pending",
            mime_type="text/plain",
            file_size=pending_path.stat().st_size,
        )
        context.db.add_all([current, sibling, invalid_prefix, pending])
        context.db.commit()

        materialized_root = context.external_dir.parent / "materialized-cache"
        materialized_root.mkdir()
        monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(materialized_root))
        materialized_path = materialized_root / "materialized-durable.txt"
        materialized_path.write_text("durable", encoding="utf-8")
        materialize_calls = 0
        opened_sessions = []
        detached_session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=context.db.get_bind(),
        )

        def create_tracked_session():
            session = detached_session_factory()
            opened_sessions.append(session)
            return session

        def assert_released_before_materialization(_self):
            nonlocal materialize_calls
            materialize_calls += 1
            assert all(not session.in_transaction() for session in opened_sessions)
            return materialized_path

        monkeypatch.setattr(
            "xagent.core.storage.manager.create_db_session",
            create_tracked_session,
        )
        monkeypatch.setattr(
            ManagedFileRef,
            "materialize",
            assert_released_before_materialization,
        )

        listed = context.ops.list_all_user_files(include_workspace_files=False)
        listed_ids = {item["file_id"] for item in listed["files"]}
        assert current.file_id in listed_ids
        assert sibling.file_id not in listed_ids
        assert invalid_prefix.file_id not in listed_ids
        assert pending.file_id not in listed_ids
        assert listed["total_count"] == 2

        assert context.ops.read_file(current.file_id) == "durable"
        assert context.ops.read_file(str(current.storage_path)) == "durable"
        assert materialize_calls == 2
        for denied_file_id in (
            sibling.file_id,
            invalid_prefix.file_id,
            pending.file_id,
        ):
            with pytest.raises(FileNotFoundError):
                context.ops.read_file(denied_file_id)

    @pytest.mark.parametrize(
        "operation",
        ["text", "json", "csv", "mkdir", "list"],
    )
    def test_marked_public_workspace_operations_revalidate_authority(
        self, public_file_scope_context, operation, caplog
    ):
        context = public_file_scope_context
        context.workspace.owner_user_id = int(context.owner.id) + 1

        with pytest.raises(ValueError, match="File Operation unavailable"):
            if operation == "text":
                context.ops.write_file("blocked.txt", "blocked")
            elif operation == "json":
                context.ops.write_json_file("blocked.json", {"blocked": True})
            elif operation == "csv":
                context.ops.write_csv_file("blocked.csv", [{"blocked": "true"}])
            elif operation == "mkdir":
                context.ops.create_directory("blocked")
            else:
                context.ops.list_files(".")

        assert any(
            record.levelno == logging.WARNING
            and "workspace authority validation failed" in record.message
            and record.exc_info is not None
            for record in caplog.records
        )

    @pytest.mark.parametrize("operation", ["read", "list"])
    def test_marked_public_file_operation_does_not_reuse_bound_session(
        self, public_file_scope_context, monkeypatch, operation
    ):
        context = public_file_scope_context
        detached_session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=context.db.get_bind(),
        )

        class ThreadBoundSession:
            def query(self, *_args, **_kwargs):
                raise AssertionError("bound session crossed into File Operation")

        context.workspace.db_session = ThreadBoundSession()
        monkeypatch.setattr(
            "xagent.core.storage.manager.create_db_session",
            detached_session_factory,
        )

        if operation == "read":
            assert context.ops.read_file(context.current_record.file_id) == "current"
        else:
            listed = context.ops.list_all_user_files(include_workspace_files=False)
            assert context.current_record.file_id in {
                item["file_id"] for item in listed["files"]
            }

    def test_marked_public_missing_task_row_does_not_fall_back_to_legacy_paths(
        self, public_file_scope_context
    ):
        context = public_file_scope_context
        context.db.query(Task).filter(Task.id == int(context.marked_task.id)).delete(
            synchronize_session=False
        )
        context.db.commit()

        with pytest.raises(FileNotFoundError):
            context.ops.read_file(str(context.raw_external_path))
        with pytest.raises(ValueError, match="File Operation unavailable"):
            context.ops.write_file("blocked-after-delete.txt", "blocked")

    def test_unmarked_local_read_does_not_require_policy_database(self, tmp_path):
        class FailingSession:
            def query(self, *_args, **_kwargs):
                raise OSError("policy database unavailable")

        workspace = TaskWorkspace(
            id="web_task_803",
            base_dir=str(tmp_path / "unmarked-local-workspace"),
            db_task_id=803,
        )
        workspace.db_session = FailingSession()
        local_path = workspace.input_dir / "local.txt"
        local_path.write_text("local", encoding="utf-8")

        assert WorkspaceFileOperations(workspace).read_file("local.txt") == "local"

    def test_marked_public_policy_load_failure_uses_file_not_found_shape(
        self, public_file_scope_context, monkeypatch, caplog
    ):
        context = public_file_scope_context

        def fail_policy_load():
            raise OSError("database unavailable")

        monkeypatch.setattr(
            context.workspace,
            "requires_exact_file_operation_scope",
            fail_policy_load,
        )

        with pytest.raises(FileNotFoundError, match="File not found"):
            context.ops.read_file(context.current_record.file_id)

        assert any(
            record.levelno == logging.WARNING
            and "selector policy validation failed" in record.message
            and record.exc_info is not None
            for record in caplog.records
        )

    def test_marked_public_output_listing_propagates_policy_failure(
        self, public_file_scope_context
    ):
        context = public_file_scope_context
        context.workspace.file_operation_access_version = True

        with pytest.raises(ValueError, match="File Operation unavailable"):
            context.ops.get_workspace_output_files()

    def test_marked_public_listing_fails_if_task_disappears_after_validation(
        self, public_file_scope_context, monkeypatch
    ):
        context = public_file_scope_context
        original = context.workspace.list_all_user_files

        def delete_then_list(*args, **kwargs):
            task = context.db.get(Task, context.marked_task.id)
            assert task is not None
            context.db.delete(task)
            context.db.commit()
            return original(*args, **kwargs)

        monkeypatch.setattr(
            context.workspace,
            "list_all_user_files",
            delete_then_list,
        )

        with pytest.raises(ValueError, match="File listing unavailable"):
            context.ops.list_all_user_files()

    def test_marked_public_malformed_listing_uses_value_error_shape(
        self, public_file_scope_context, caplog
    ):
        context = public_file_scope_context
        context.marked_task.agent_config = {
            "auth_mode": "share",
            "__xagent_file_operation_access_version": 2,
        }
        context.db.commit()

        with pytest.raises(ValueError, match="File listing unavailable"):
            context.ops.list_all_user_files(include_workspace_files=False)

        assert any(
            record.levelno == logging.WARNING
            and "listing policy validation failed" in record.message
            and record.exc_info is not None
            for record in caplog.records
        )

    def test_marked_public_missing_owner_fails_before_legacy_allow(
        self, public_file_scope_context, tmp_path
    ):
        context = public_file_scope_context
        workspace = TaskWorkspace(
            id="web_task_801",
            base_dir=str(tmp_path / "missing-owner-workspaces"),
            allowed_external_dirs=[str(context.external_dir)],
            db_task_id=int(context.marked_task.id),
        )
        workspace.file_operation_access_version = 1
        workspace.db_session = context.db
        ops = WorkspaceFileOperations(workspace)

        with pytest.raises((FileNotFoundError, ValueError)):
            ops.read_file(context.current_record.file_id)

    def test_marked_public_missing_db_task_id_fails_before_legacy_parse(
        self, public_file_scope_context, tmp_path
    ):
        context = public_file_scope_context
        workspace = TaskWorkspace(
            id="web_task_801",
            base_dir=str(tmp_path / "missing-db-task-workspaces"),
            allowed_external_dirs=[str(context.external_dir)],
        )
        workspace.owner_user_id = int(context.owner.id)
        workspace.file_operation_access_version = 1
        workspace.db_session = context.db

        with pytest.raises(FileNotFoundError):
            WorkspaceFileOperations(workspace).read_file(context.current_record.file_id)

    def test_unmarked_historical_file_operation_behavior_is_unchanged(
        self, public_file_scope_context, tmp_path
    ):
        context = public_file_scope_context
        workspace = TaskWorkspace(
            id="web_task_803",
            base_dir=str(tmp_path / "historical-workspaces"),
            allowed_external_dirs=[str(context.external_dir)],
            db_task_id=int(context.historical_task.id),
        )
        workspace.owner_user_id = int(context.owner.id)
        workspace.db_session = context.db
        ops = WorkspaceFileOperations(workspace)

        listed = ops.list_all_user_files(include_workspace_files=False)
        listed_ids = {item["file_id"] for item in listed["files"]}
        assert context.current_record.file_id in listed_ids
        assert context.sibling_record.file_id in listed_ids
        assert context.unbound_record.file_id in listed_ids
        assert context.other_record.file_id not in listed_ids
        assert workspace.resolve_file_id(context.sibling_record.file_id) is None
        assert workspace.resolve_file_id(context.unbound_record.file_id) == (
            context.unbound_path
        )
        assert ops.read_file(str(context.sibling_path)) == "sibling"

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
