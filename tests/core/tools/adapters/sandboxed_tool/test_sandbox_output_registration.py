"""Sandbox-produced files must get their file_id from the host process.

The sandbox runner has no database credentials, so a file_id minted in there
names no real record -- which is what made an agent overwrite a generated
.docx with a placeholder to obtain a "real" one.
"""

import asyncio
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import xagent.config as xagent_config
from tests.core.tools.adapters.sandboxed_tool.conftest import FakeBaseTool
from xagent.core.file_storage.factory import (
    get_unscoped_file_storage,
    get_user_file_storage,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import sandbox_config
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
)
from xagent.core.workspace import SANDBOX_FILE_ID_PREFIX, TaskWorkspace
from xagent.web.models import Base
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User

SANDBOX_MINTED_FILE_ID = "sandbox-only-file-id"

# A workspace name without a task id parses to task_id=None, so should_persist is
# False: every test here except the durable_workspace ones asserts that ids and
# metadata changed, not that anything reached the database or object storage.


@sandbox_config()
class _FakeGeneratingTool(FakeBaseTool):
    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "fake_generating_tool"


def _make_sandbox(payload: dict) -> MagicMock:
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


def test_host_process_reregisters_sandbox_generated_files(tmp_path):
    workspace = TaskWorkspace("test_sandbox_outputs", str(tmp_path))
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"PK\x03\x04 not really a docx")

    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["report.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "report.docx",
                        "file_path": str(generated),
                    }
                ],
                "artifacts": [],
            }
        ),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert result["file_refs"], "host registration must not drop the generated file"
    file_ref = result["file_refs"][0]
    assert file_ref["file_id"] != SANDBOX_MINTED_FILE_ID
    assert file_ref["filename"] == "report.docx"
    assert file_ref["size"] == generated.stat().st_size
    assert result["generated_files"] == ["report.docx"]


def test_unreachable_sandbox_paths_are_left_untouched(tmp_path):
    """No host-visible path at all: the early return keeps the sandbox refs."""
    workspace = TaskWorkspace("test_sandbox_guest_paths", str(tmp_path))
    payload = {
        "success": True,
        "generated_files": ["report.docx"],
        "file_refs": [
            {
                "file_id": SANDBOX_MINTED_FILE_ID,
                "filename": "report.docx",
                "file_path": "/guest/only/report.docx",
            }
        ],
        "artifacts": [],
    }
    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(payload),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert result["file_refs"][0]["file_id"] == SANDBOX_MINTED_FILE_ID


def _enter_sandbox_runner_mode(monkeypatch) -> None:
    """Enter sandbox-runner mode the way the process itself does.

    The marker is snapshotted at import, so setting the env var here would be
    ignored — which is exactly the property that keeps agent code from
    flipping host registration.
    """
    monkeypatch.setattr(xagent_config, "_IN_SANDBOX_TOOL_RUNNER", True)


def test_register_files_inside_sandbox_runner_never_touches_the_database(
    tmp_path, monkeypatch
):
    workspace = TaskWorkspace("test_sandbox_runner", str(tmp_path))
    target = workspace.output_dir / "note.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello")

    def _fail(*args, **kwargs):
        raise AssertionError("sandbox runner must not reach the metadata store")

    monkeypatch.setattr(TaskWorkspace, "_register_files_locked", _fail)
    _enter_sandbox_runner_mode(monkeypatch)

    assert (
        workspace.register_file(str(target), file_id="requested-id") == "requested-id"
    )
    assert workspace.register_file(str(target)) == "requested-id"

    monkeypatch.setattr(xagent_config, "_IN_SANDBOX_TOOL_RUNNER", False)
    with pytest.raises(AssertionError):
        workspace.register_file(str(target))


def test_partially_visible_refs_keep_the_sandbox_entry(tmp_path):
    workspace = TaskWorkspace("test_sandbox_mixed", str(tmp_path))
    visible = workspace.output_dir / "visible.docx"
    visible.parent.mkdir(parents=True, exist_ok=True)
    visible.write_bytes(b"PK\x03\x04 visible")

    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["visible.docx", "guest.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "visible.docx",
                        "file_path": str(visible),
                    },
                    {
                        "file_id": "guest-only-id",
                        "filename": "guest.docx",
                        "file_path": "/guest/only/guest.docx",
                    },
                ],
                "artifacts": [],
            }
        ),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert [ref["filename"] for ref in result["file_refs"]] == [
        "visible.docx",
        "guest.docx",
    ]
    assert result["file_refs"][0]["file_id"] != SANDBOX_MINTED_FILE_ID
    assert result["file_refs"][1]["file_id"] == "guest-only-id"
    assert result["generated_files"] == ["visible.docx", "guest.docx"]
    # The unregistered ref stays in artifacts on purpose; its id is recognizably
    # not database-backed rather than silently dropped.
    artifact_ids = [artifact.get("file_id") for artifact in result["artifacts"]]
    assert artifact_ids[0] == result["file_refs"][0]["file_id"]
    assert artifact_ids[1] == "guest-only-id"


def test_a_surviving_sandbox_id_is_never_offered_to_the_model_as_a_link(tmp_path):
    """A sandbox id means registration failed; any link built from it 404s."""
    from xagent.core.tools.artifacts import (
        build_inline_artifact,
        format_tool_result_for_observation,
        markdown_reference_for_artifact,
    )

    # The production shape: _register_sandbox_outputs always emits file_refs
    # beside artifacts, and the observation renders both.
    dead_ref = {
        "file_id": SANDBOX_MINTED_FILE_ID,
        "filename": "report.docx",
        "mime_type": "application/vnd.openxmlformats-officedocument",
        "markdown_link": f"[report.docx](file:{SANDBOX_MINTED_FILE_ID})",
        "download_url": f"/api/files/download/{SANDBOX_MINTED_FILE_ID}",
        "file_path": "/w/output/report.docx",
        "size": 99,
    }
    assert markdown_reference_for_artifact(dead_ref) is None

    observation = format_tool_result_for_observation(
        "execute_python_code",
        {
            "success": True,
            "file_refs": [dead_ref],
            "artifacts": [build_inline_artifact(dead_ref)],
            "generated_files": ["report.docx"],
        },
    )
    assert SANDBOX_MINTED_FILE_ID not in observation
    assert "download_url" not in observation
    assert "markdown_link" not in observation
    # Still announced: silence is what made an agent rewrite a real document.
    assert "report.docx" in observation
    assert "registration did not complete" in observation

    live_id = "6f1c9e6c-0000-4000-8000-000000000000"
    live_ref = dict(
        dead_ref, file_id=live_id, markdown_link=f"[report.docx](file:{live_id})"
    )
    assert markdown_reference_for_artifact(live_ref) == live_ref["markdown_link"]
    live_observation = format_tool_result_for_observation(
        "execute_python_code",
        {"success": True, "file_refs": [live_ref], "artifacts": [live_ref]},
    )
    assert live_id in live_observation
    assert "markdown_link" in live_observation


def test_resolving_a_sandbox_id_stops_before_the_database(tmp_path, monkeypatch):
    workspace = TaskWorkspace("test_sandbox_resolve", str(tmp_path))
    calls = []

    # A spy, not a raise: resolve_file_id swallows exceptions from this block,
    # so a raising fake would pass with the short-circuit deleted.
    def _spy(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("no database in this test")

    monkeypatch.setattr("xagent.core.storage.manager.create_db_session", _spy)

    assert workspace.resolve_file_id(SANDBOX_MINTED_FILE_ID) is None
    assert calls == []


def test_failed_host_registration_keeps_the_sandbox_metadata(tmp_path, monkeypatch):
    workspace = TaskWorkspace("test_sandbox_failed_host", str(tmp_path))
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"PK\x03\x04 report")

    def _boom(*args, **kwargs):
        raise RuntimeError("metadata store unreachable")

    monkeypatch.setattr(TaskWorkspace, "register_file", _boom)
    monkeypatch.setattr(TaskWorkspace, "get_file_id_from_path", lambda self, path: None)

    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["report.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "report.docx",
                        "file_path": str(generated),
                    }
                ],
                "artifacts": [],
            }
        ),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert result["generated_files"] == ["report.docx"]
    assert result["file_refs"][0]["file_id"] == SANDBOX_MINTED_FILE_ID
    assert result["artifacts"] == []


def test_sandbox_runner_reuses_one_prefixed_id_per_path(tmp_path, monkeypatch):
    workspace = TaskWorkspace("test_sandbox_stable_ids", str(tmp_path))
    target = workspace.output_dir / "note.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello")
    _enter_sandbox_runner_mode(monkeypatch)

    first = workspace.register_file(str(target))
    assert first.startswith(SANDBOX_FILE_ID_PREFIX)
    assert workspace.register_file(str(target)) == first
    assert workspace.get_file_id_from_path(str(target)) == first


def test_symlinked_guest_spelling_still_merges(tmp_path):
    """A guest mount keeps the unresolved spelling; the merge must still match."""
    real_base = tmp_path / "real"
    real_base.mkdir()
    linked_base = tmp_path / "linked"
    linked_base.symlink_to(real_base, target_is_directory=True)

    workspace = TaskWorkspace("test_sandbox_symlink", str(real_base))
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"PK\x03\x04 report")

    guest_path = str(linked_base / generated.relative_to(real_base.resolve()))
    assert guest_path != str(generated)

    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["report.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "report.docx",
                        "file_path": guest_path,
                    }
                ],
                "artifacts": [],
            }
        ),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert result["file_refs"][0]["file_id"] != SANDBOX_MINTED_FILE_ID


def test_regenerated_output_is_re_registered(tmp_path, monkeypatch):
    """A second run over the same path must re-stage, not reuse the old bytes."""
    workspace = TaskWorkspace("test_sandbox_regenerate", str(tmp_path))
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)

    registered: list[str] = []
    original_register_file = TaskWorkspace.register_file

    def _counting_register_file(self, file_path, *args, **kwargs):
        registered.append(str(file_path))
        return original_register_file(self, file_path, *args, **kwargs)

    monkeypatch.setattr(TaskWorkspace, "register_file", _counting_register_file)

    def _run(payload_bytes: bytes):
        generated.write_bytes(payload_bytes)
        wrapper = SandboxedToolWrapper(
            _FakeGeneratingTool(workspace=workspace),
            _make_sandbox(
                {
                    "success": True,
                    "generated_files": ["report.docx"],
                    "file_refs": [
                        {
                            "file_id": SANDBOX_MINTED_FILE_ID,
                            "filename": "report.docx",
                            "file_path": str(generated),
                        }
                    ],
                    "artifacts": [],
                }
            ),
        )
        return asyncio.run(wrapper.run_json_async({}))

    _run(b"PK\x03\x04 draft")
    second = _run(b"PK\x03\x04 revised and longer")

    assert len(registered) == 2
    assert second["file_refs"][0]["size"] == generated.stat().st_size


def test_failed_registration_never_falls_back_to_a_stale_id(tmp_path, monkeypatch):
    """A path that failed to re-stage must not be described from its old id."""
    workspace = TaskWorkspace("test_sandbox_stale", str(tmp_path))
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"PK\x03\x04 revised")

    def _boom(self, file_path, *args, **kwargs):
        raise RuntimeError("staging failed")

    monkeypatch.setattr(TaskWorkspace, "register_file", _boom)
    # The previous generation is still cached under this path.
    monkeypatch.setattr(
        TaskWorkspace, "get_file_id_from_path", lambda self, path: "stale-generation-id"
    )

    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["report.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "report.docx",
                        "file_path": str(generated),
                    }
                ],
                "artifacts": [],
            }
        ),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert result["file_refs"][0]["file_id"] == SANDBOX_MINTED_FILE_ID
    assert result["file_refs"][0]["file_id"] != "stale-generation-id"


def test_symlinked_output_cannot_register_a_file_outside_output(tmp_path):
    """A guest path is not authorization: sandboxed code can plant a symlink."""
    workspace = TaskWorkspace("test_sandbox_symlink_escape", str(tmp_path))
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    secret = workspace.input_dir / "uploaded.docx"
    secret.write_bytes(b"PK\x03\x04 someone elses upload")
    planted = workspace.output_dir / "report.docx"
    planted.symlink_to(secret)

    registered: list[str] = []
    original_register_file = TaskWorkspace.register_file

    def _counting_register_file(self, file_path, *args, **kwargs):
        registered.append(str(file_path))
        return original_register_file(self, file_path, *args, **kwargs)

    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["report.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "report.docx",
                        "file_path": str(planted),
                    }
                ],
                "artifacts": [],
            }
        ),
    )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(TaskWorkspace, "register_file", _counting_register_file)
        result = asyncio.run(wrapper.run_json_async({}))

    assert registered == []
    assert result["file_refs"][0]["file_id"] == SANDBOX_MINTED_FILE_ID


def _guest_ref_wrapper(workspace, file_path):
    return SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["report.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "report.docx",
                        "file_path": str(file_path),
                    }
                ],
                "artifacts": [],
            }
        ),
    )


def _registrations_during(wrapper, monkeypatch):
    registered: list[str] = []
    original = TaskWorkspace.register_file

    def _counting(self, file_path, *args, **kwargs):
        registered.append(str(file_path))
        return original(self, file_path, *args, **kwargs)

    monkeypatch.setattr(TaskWorkspace, "register_file", _counting)
    return registered, asyncio.run(wrapper.run_json_async({}))


def test_a_guest_ref_outside_output_is_not_registered(tmp_path, monkeypatch):
    """No symlink needed: the ref itself can just name a file outside output."""
    workspace = TaskWorkspace("test_sandbox_outside_output", str(tmp_path))
    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    uploaded = workspace.input_dir / "uploaded.docx"
    uploaded.write_bytes(b"PK\x03\x04 someone elses upload")

    registered, result = _registrations_during(
        _guest_ref_wrapper(workspace, uploaded), monkeypatch
    )

    assert registered == []
    assert result["file_refs"][0]["file_id"] == SANDBOX_MINTED_FILE_ID


def test_a_symlink_inside_output_is_not_registered(tmp_path, monkeypatch):
    """Registering the link target would file the bytes under the wrong name."""
    workspace = TaskWorkspace("test_sandbox_inner_symlink", str(tmp_path))
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    real = workspace.output_dir / "actual.docx"
    real.write_bytes(b"PK\x03\x04 real")
    link = workspace.output_dir / "report.docx"
    link.symlink_to(real)

    registered, result = _registrations_during(
        _guest_ref_wrapper(workspace, link), monkeypatch
    )

    assert registered == []
    assert result["file_refs"][0]["file_id"] == SANDBOX_MINTED_FILE_ID


@pytest.fixture
def durable_workspace(monkeypatch, tmp_path):
    """A workspace whose register_file creates real rows and storage objects."""
    # StaticPool: host registration runs in a worker thread, and a per-thread
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
    # any earlier test in the process that ran configure_db leaves _SessionLocal
    # set, and registration would silently use that database instead of this one.
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local", lambda: SessionLocal
    )
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()

    user = User(username="sandbox-durable-user", password_hash="hash")
    db.add(user)
    db.flush()
    db.add(Task(id=9101, user_id=user.id, title="Sandbox durable task"))
    db.commit()

    workspace = TaskWorkspace(id="web_task_9101", base_dir=str(tmp_path / "workspaces"))
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


def test_host_registration_creates_a_durable_record(durable_workspace):
    """The returned file_id must name a real row whose object serves the bytes."""
    workspace, db, user_id = durable_workspace
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"PK\x03\x04 first generation")

    wrapper = _guest_ref_wrapper(workspace, generated)
    result = asyncio.run(wrapper.run_json_async({}))

    file_id = result["file_refs"][0]["file_id"]
    assert file_id != SANDBOX_MINTED_FILE_ID
    assert not file_id.startswith(SANDBOX_FILE_ID_PREFIX)
    record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
    assert record.user_id == user_id
    assert record.task_id == 9101
    assert _served_bytes(user_id, db, file_id) == b"PK\x03\x04 first generation"


def test_regeneration_serves_the_revised_bytes(durable_workspace):
    """The second run must re-stage, not leave the first draft as the served object."""
    workspace, db, user_id = durable_workspace
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)

    generated.write_bytes(b"PK\x03\x04 first draft")
    first = asyncio.run(_guest_ref_wrapper(workspace, generated).run_json_async({}))
    first_id = first["file_refs"][0]["file_id"]

    generated.write_bytes(b"PK\x03\x04 revised and longer")
    second = asyncio.run(_guest_ref_wrapper(workspace, generated).run_json_async({}))
    second_id = second["file_refs"][0]["file_id"]

    assert second_id == first_id
    assert _served_bytes(user_id, db, second_id) == b"PK\x03\x04 revised and longer"
