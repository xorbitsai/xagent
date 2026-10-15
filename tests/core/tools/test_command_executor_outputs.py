"""Files a shell command writes must leave the call with a usable file_id.

Without one the agent cannot hand a finished deliverable to the user, and the
deliverable is lost when the task workspace is removed (#2953).
"""

import asyncio
import json
import os
import re
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import xagent.config as xagent_config
from xagent.core.agent.context.execution import ExecutionContext
from xagent.core.file_storage.factory import (
    get_unscoped_file_storage,
    get_user_file_storage,
)
from xagent.core.tools.adapters.vibe.command_executor import (
    MAX_REGISTERED_FILES_PER_COMMAND,
    CommandExecutorTool,
    CommandExecutorToolForBasic,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
)
from xagent.core.workspace import SANDBOX_FILE_ID_PREFIX, TaskWorkspace
from xagent.web.models import Base
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


def _run(tool: CommandExecutorTool, command: str) -> dict:
    return tool.run_json_sync({"command": command})


@pytest.fixture
def workspace(tmp_path):
    return TaskWorkspace("test_command_outputs", str(tmp_path))


@pytest.fixture
def registrations(monkeypatch):
    registered: list[str] = []
    original_register_file = TaskWorkspace.register_file

    def _counting_register_file(self, file_path, *args, **kwargs):
        registered.append(str(file_path))
        return original_register_file(self, file_path, *args, **kwargs)

    monkeypatch.setattr(TaskWorkspace, "register_file", _counting_register_file)
    return registered


def test_written_deliverable_gets_a_file_id(workspace):
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(tool, "printf 'PK\\003\\004 deck' > deck.pptx")

    assert result["success"] is True
    assert result["generated_files"] == ["deck.pptx"]
    file_ref = result["file_refs"][0]
    assert file_ref["file_id"]
    assert result["artifacts"][0]["file_id"] == file_ref["file_id"]
    listed = {f["filename"]: f["file_id"] for f in workspace.get_output_files()}
    assert listed["deck.pptx"] == file_ref["file_id"]


def test_failing_command_still_registers_what_it_wrote(workspace):
    """A shell's exit code describes only its last command, not the deliverable."""
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(tool, "printf 'PK\\003\\004 deck' > deck.pptx; exit 3")

    assert result["success"] is False
    assert result["return_code"] == 3
    assert result["file_refs"][0]["filename"] == "deck.pptx"
    assert result["file_refs"][0]["file_id"]
    assert "validation" in result["file_refs"][0]


def test_unchanged_files_are_not_registered(workspace, registrations):
    (workspace.output_dir / "existing.pptx").write_bytes(b"PK\x03\x04 old")
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(tool, "ls")

    assert registrations == []
    assert "file_refs" not in result


def test_plain_command_keeps_its_stdout_observation(workspace):
    """Without registrations the result shape, and so the observation, is unchanged."""
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(tool, "printf 'a\\nb\\n'")

    assert set(result) == {"success", "output", "error", "return_code"}
    observation = ExecutionContext()._format_tool_result("execute_command", result)
    assert observation == "Tool execute_command returned: a\nb\n"


def test_touch_registers_a_file_left_over_from_an_earlier_call(workspace):
    """The over-cap note's remedy: a touch changes mtime, so the file registers."""
    (workspace.output_dir / "deck.pptx").write_bytes(b"PK\x03\x04 deck")
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(tool, "touch deck.pptx")

    assert result["generated_files"] == ["deck.pptx"]


@pytest.mark.parametrize(
    "relative_path",
    [
        "node_modules/pkg/logo.png",
        "__pycache__/chart.png",
        ".cache/chart.png",
        "venv/lib/chart.png",
        "site-packages/pkg/chart.png",
        ".hidden.png",
    ],
)
def test_dependency_cache_and_hidden_paths_are_skipped(
    workspace, registrations, relative_path
):
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(
        tool,
        f"mkdir -p \"$(dirname {relative_path})\" && printf 'x' > {relative_path}",
    )

    assert (workspace.output_dir / relative_path).exists()
    assert registrations == []
    assert "file_refs" not in result


def test_file_in_a_nested_directory_is_registered(workspace):
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(
        tool,
        "mkdir -p reports/q3 && printf 'PK\\003\\004 deck' > reports/q3/deck.pptx",
    )

    assert result["generated_files"] == ["deck.pptx"]
    file_ref = result["file_refs"][0]
    assert file_ref["file_id"]
    assert file_ref["relative_path"].endswith("reports/q3/deck.pptx")


def test_engine_owned_spill_directory_is_skipped(workspace, registrations):
    spill = workspace.engine_owned_output_dir
    spill.mkdir(parents=True, exist_ok=True)
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(tool, f"printf 'a,b' > {spill.name}/x.csv")

    assert (spill / "x.csv").exists()
    assert registrations == []
    assert "file_refs" not in result


def test_over_cap_registers_none_and_says_so(workspace, registrations):
    tool = CommandExecutorTool(workspace=workspace)
    cap = MAX_REGISTERED_FILES_PER_COMMAND

    result = _run(
        tool,
        f"mkdir -p parts && for i in $(seq 1 {cap + 1}); do "
        "printf 'x' > parts/image$i.png; done && "
        "printf 'PK\\003\\004 deck' > deck.pptx",
    )

    assert registrations == []
    assert result["file_refs"] == []
    assert result["artifacts"] == []
    observation = ExecutionContext()._format_tool_result("execute_command", result)
    assert "registration_note" in observation
    note = result["registration_note"]
    assert str(cap + 2) in note
    assert str(cap) in note
    assert "deck.pptx" in note
    assert not re.search(r"image\d+\.png", note)
    assert str(workspace.output_dir) not in note


def test_over_cap_note_truncates_the_non_image_list(workspace, registrations):
    tool = CommandExecutorTool(workspace=workspace)
    cap = MAX_REGISTERED_FILES_PER_COMMAND

    result = _run(
        tool,
        f"mkdir -p parts && for i in $(seq 1 {cap + 5}); do "
        "printf 'x' > parts/part$i.csv; done",
    )

    note = result["registration_note"]
    assert registrations == []
    assert "and 5 more" in note
    assert len(re.findall(r"parts/part\d+\.csv", note)) == cap


def test_exactly_the_cap_is_registered(workspace):
    tool = CommandExecutorTool(workspace=workspace)
    count = MAX_REGISTERED_FILES_PER_COMMAND

    result = _run(
        tool,
        f"for i in $(seq 1 {count}); do printf 'x' > image$i.png; done",
    )

    assert len(result["file_refs"]) == count
    assert result["registration_note"] == ""


def test_failed_registration_is_reported(workspace, monkeypatch):
    def _fail(self, file_path, *args, **kwargs):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(TaskWorkspace, "register_file", _fail)
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(tool, "printf 'PK\\003\\004 deck' > deck.pptx")

    assert result["success"] is True
    assert result["file_refs"] == []
    assert "deck.pptx" in result["registration_note"]


def test_failed_registrations_are_listed_by_relative_path(workspace, monkeypatch):
    def _fail(self, file_path, *args, **kwargs):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(TaskWorkspace, "register_file", _fail)
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(
        tool,
        "mkdir -p a b && printf 'x' > a/chart.png && printf 'x' > b/chart.png",
    )

    note = result["registration_note"]
    assert "a/chart.png" in note
    assert "b/chart.png" in note
    assert str(workspace.output_dir) not in note


def test_ref_dropped_after_registration_is_reported(workspace, monkeypatch):
    def _no_ref(**kwargs):
        raise RuntimeError("validator crashed")

    monkeypatch.setattr("xagent.core.tools.artifacts.build_workspace_file_ref", _no_ref)
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(tool, "printf 'PK\\003\\004 deck' > deck.pptx")

    assert result["file_refs"] == []
    note = result["registration_note"]
    assert "Registered, but no file reference could be built" in note
    assert "deck.pptx" in note
    assert "Written but not registered" not in note


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unreadable_directory_keeps_the_command_result(workspace):
    tool = CommandExecutorTool(workspace=workspace)
    locked = workspace.output_dir / "locked"
    try:
        result = _run(
            tool,
            "mkdir locked && printf 'x' > locked/chart.png && "
            "chmod a-x locked && echo done",
        )
    finally:
        locked.chmod(0o755)

    assert result["success"] is True
    assert result["output"].strip() == "done"


def test_overwritten_registered_file_is_re_registered(workspace, registrations):
    tool = CommandExecutorTool(workspace=workspace)

    _run(tool, "printf 'PK\\003\\004 first' > deck.pptx")
    second = _run(tool, "printf 'PK\\003\\004 revised and longer' > deck.pptx")

    # id stability needs a persisted task; see test_overwrite_serves_the_revised_bytes
    deck = str((workspace.output_dir / "deck.pptx").resolve())
    assert registrations == [deck, deck]
    assert second["file_refs"][0]["size"] == len(b"PK\x03\x04 revised and longer")


def test_no_workspace_leaves_result_without_refs():
    result = _run(CommandExecutorTool(), "echo hi")

    assert result["success"] is True
    assert set(result) == {"success", "output", "error", "return_code"}


@pytest.fixture
def durable_workspace(monkeypatch, tmp_path):
    """A workspace whose register_file creates real rows and storage objects."""
    # StaticPool: registration may run on another thread, and a per-thread
    # connection would open a second, empty in-memory database.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    monkeypatch.setattr("xagent.core.storage.manager.create_db_session", SessionLocal)
    # _create_registration_session prefers the web session factory and only falls
    # back to create_db_session on RuntimeError, so patching one is not enough:
    # an earlier test that ran configure_db would leave _SessionLocal pointing at
    # another database.
    monkeypatch.setattr("xagent.web.models.database._SessionLocal", SessionLocal)
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()

    user = User(username="command-durable-user", password_hash="hash")
    db.add(user)
    db.flush()
    db.add(Task(id=9201, user_id=user.id, title="Command durable task"))
    db.commit()

    workspace = TaskWorkspace(id="web_task_9201", base_dir=str(tmp_path / "workspaces"))
    try:
        yield workspace, db, int(user.id)
    finally:
        db.close()
        engine.dispose()
        get_unscoped_file_storage.cache_clear()


def _served_bytes(user_id: int, db, file_id: str) -> bytes:
    record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
    with get_user_file_storage(user_id).open_read(str(record.storage_key)) as handle:
        return handle.read()


def test_overwrite_serves_the_revised_bytes(durable_workspace):
    """The staged copy must follow the file, or cleanup leaves the old deck."""
    workspace, db, user_id = durable_workspace
    tool = CommandExecutorTool(workspace=workspace)

    first = _run(tool, "printf 'PK\\003\\004 first' > deck.pptx")
    assert _served_bytes(user_id, db, first["file_refs"][0]["file_id"]) == (
        b"PK\x03\x04 first"
    )

    second = _run(tool, "printf 'PK\\003\\004 revised' > deck.pptx")
    file_id = second["file_refs"][0]["file_id"]

    assert file_id == first["file_refs"][0]["file_id"]
    assert _served_bytes(user_id, db, file_id) == b"PK\x03\x04 revised"


def _fake_sandbox(payload: dict) -> MagicMock:
    def _exec(*args, **kwargs):
        result = MagicMock()
        result.exit_code = 0
        result.stdout = json.dumps(payload) if args[0] == "cat" else ""
        result.stderr = ""
        return result

    sandbox = MagicMock()
    sandbox.name = "sandbox-test"
    sandbox.exec = AsyncMock(side_effect=_exec)
    sandbox.write_file = AsyncMock()
    return sandbox


def test_sandbox_refs_are_re_registered_on_the_host(workspace, monkeypatch):
    monkeypatch.setattr(xagent_config, "_IN_SANDBOX_TOOL_RUNNER", True)
    guest_result = _run(
        CommandExecutorToolForBasic(workspace=workspace),
        "printf 'PK\\003\\004 deck' > deck.pptx",
    )
    guest_id = guest_result["file_refs"][0]["file_id"]
    assert guest_id.startswith(SANDBOX_FILE_ID_PREFIX)

    monkeypatch.setattr(xagent_config, "_IN_SANDBOX_TOOL_RUNNER", False)
    wrapper = SandboxedToolWrapper(
        CommandExecutorToolForBasic(workspace=workspace),
        _fake_sandbox(guest_result),
    )
    result = asyncio.run(wrapper.run_json_async({"command": "ignored"}))

    host_id = result["file_refs"][0]["file_id"]
    assert not host_id.startswith(SANDBOX_FILE_ID_PREFIX)
    assert result["artifacts"][0]["file_id"] == host_id
    assert result["generated_files"] == ["deck.pptx"]


def test_symlinks_are_not_registered(workspace, registrations, tmp_path):
    outside = tmp_path / "secret.csv"
    outside.write_text("a,b\n")
    tool = CommandExecutorTool(workspace=workspace)

    result = _run(tool, f"ln -s {outside} leaked.csv")

    assert registrations == []
    assert "file_refs" not in result
