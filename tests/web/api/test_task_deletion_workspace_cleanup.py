"""Deleting a task removes its workspace directory, not somebody else's.

The workspace path is derived from the task's owner and its execution scope.
Both of those live on the task row, so a deletion that cleans up afterwards --
with the requester's id, or with the scope of a row it has already deleted --
looks successful while leaving the directory on disk forever.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Iterator

import pytest
from fastapi import HTTPException

from tests.shared.execution_scope import register_scope_resolver
from xagent.core.execution_scope import (
    EXECUTION_SCOPE_AGENT_CONFIG_KEY,
    ExecutionScope,
    set_execution_scope_snapshot_loader,
)
from xagent.core.task_runtime import TaskRuntimeContribution
from xagent.web.api import admin_users
from xagent.web.api import websocket as websocket_module
from xagent.web.api.admin_users import delete_user
from xagent.web.api.chat import delete_task
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.services import agent_service_manager
from xagent.web.services import task_execution as task_execution_module
from xagent.web.services.agent_service_manager import get_agent_manager
from xagent.web.services.execution_scope_snapshot import (
    load_task_execution_scope_snapshot,
)
from xagent.web.services.task_runtime import (
    agent_config_with_task_extension_bindings,
    register_task_extension,
    unregister_task_extension,
)

from .conftest import _admin_headers, _direct_db_session, _register_second_user

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _workspace_root(tmp_path, monkeypatch) -> Iterator[Path]:
    """Point the uploads root at a temp dir and use the real snapshot loader.

    The loader is the production one on purpose: the scoped-workspace case
    below turns on it returning ``None`` once the task row is gone, which a
    stubbed loader would paper over.
    """
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads))
    monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", "")
    register_scope_resolver(None)
    set_execution_scope_snapshot_loader(load_task_execution_scope_snapshot)

    manager = get_agent_manager()
    manager._agents.clear()
    manager._agent_owner_ids.clear()

    yield uploads

    set_execution_scope_snapshot_loader(None)


def _make_workspace(base: Path, task_id: int) -> Path:
    workspace = base / f"web_task_{task_id}"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "result.txt").write_text("payload")
    return workspace


@pytest.mark.asyncio
async def test_admin_delete_cleans_the_task_owners_workspace(
    _workspace_root: Path,
) -> None:
    """An admin's own id names a directory that was never this task's.

    With no agent cached there is nothing in memory that knows the real path,
    so cleanup has to be pointed at the *owner*. Sending it at the requester
    probes ``user_<admin>/``, finds nothing, and reports success while the
    owner's tree stays on disk.
    """
    _admin_headers()
    _register_second_user("ws-owner", "wsownerpass1")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        owner = db.query(User).filter(User.username == "ws-owner").one()
        task = Task(user_id=int(owner.id), title="owned", description="")
        db.add(task)
        db.commit()
        task_id = int(task.id)

        workspace = _make_workspace(_workspace_root / f"user_{int(owner.id)}", task_id)

        result = await delete_task(task_id, db=db, user=admin)

        assert result["success"] is True
        assert result["workspace_cleanup_pending"] is False
        assert not workspace.exists()
        assert db.query(Task).filter(Task.id == task_id).count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_delete_cleans_a_scoped_workspace_only_the_row_can_locate(
    _workspace_root: Path,
) -> None:
    """The scope has to be read before the row that carries it is deleted.

    The workspace lives under the scope's segment. Once the task row is gone
    the snapshot loader has nothing to read, the segment list comes back
    empty, and no probed candidate names this directory any more.
    """
    _admin_headers()
    _register_second_user("scoped-owner", "scopedpass1")
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "scoped-owner").one()
        scope = ExecutionScope(
            sandbox_key_suffix="team-9", workspace_segments=("team-9",)
        )
        task = Task(
            user_id=int(owner.id),
            title="scoped",
            description="",
            agent_config={EXECUTION_SCOPE_AGENT_CONFIG_KEY: scope.to_dict()},
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)

        workspace = _make_workspace(
            _workspace_root / f"user_{int(owner.id)}" / "team-9", task_id
        )

        result = await delete_task(task_id, db=db, user=owner)

        assert result["success"] is True
        assert result["workspace_cleanup_pending"] is False
        assert not workspace.exists()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_workspace_cleanup_failure_reports_pending_not_a_failed_delete(
    _workspace_root: Path, monkeypatch
) -> None:
    """The rows are already committed, so 500 would be a lie.

    A client told the deletion failed retries it and gets 404. Report the
    deletion that did happen, and the cleanup that did not, separately.
    """
    _admin_headers()
    _register_second_user("pending-owner", "pendingpass1")
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "pending-owner").one()
        task = Task(user_id=int(owner.id), title="pending", description="")
        db.add(task)
        db.commit()
        task_id = int(task.id)

        def _explode(*args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(
            type(get_agent_manager()), "_cleanup_workspace_directory", _explode
        )

        result = await delete_task(task_id, db=db, user=owner)

        assert result["success"] is True
        assert result["workspace_cleanup_pending"] is True
        assert db.query(Task).filter(Task.id == task_id).count() == 0
    finally:
        db.close()


# ===== account deletion =====


class _FailingProvider:
    """Enough of a provider to make account deletion abort at the 503."""

    async def on_task_created(self, context, configuration) -> None:
        return None

    async def build_runtime(self, context) -> TaskRuntimeContribution:
        return TaskRuntimeContribution()

    async def public_metadata(self, context) -> dict:
        return {}

    async def on_task_deleted(self, context) -> None:
        raise RuntimeError("provider is down")


@pytest.fixture
def _registered_extension() -> Iterator[str]:
    name = "workspace_cleanup_test_ext"
    register_task_extension(name, _FailingProvider())
    yield name
    unregister_task_extension(name)


@pytest.mark.asyncio
async def test_user_delete_cleans_every_task_workspace(
    _workspace_root: Path,
) -> None:
    """Account deletion removes task rows in bulk and owns their directories.

    Nothing else will: the per-task delete endpoint is never called on this
    path, so a workspace it leaves behind has no later owner at all.
    """
    _admin_headers()
    _register_second_user("bulk-owner", "bulkpass1")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "bulk-owner").one()
        tasks = [
            Task(user_id=int(target.id), title=f"bulk {index}", description="")
            for index in range(3)
        ]
        db.add_all(tasks)
        db.commit()
        owner_root = _workspace_root / f"user_{int(target.id)}"
        workspaces = [_make_workspace(owner_root, int(task.id)) for task in tasks]
        target_id = int(target.id)

        response = await delete_user(target_id, admin, db)

        assert response == {
            "message": "User deleted successfully",
            "workspace_cleanup_pending": False,
        }
        assert [workspace.exists() for workspace in workspaces] == [False] * 3
        assert db.query(User).filter(User.id == target_id).count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_user_delete_that_aborts_keeps_the_live_users_workspaces(
    _workspace_root: Path, _registered_extension: str
) -> None:
    """The account survives a 503, so its files have to survive it too.

    What this pins is that removal stays *after* the provider gate: a cleanup
    folded into the task-page walk, which is where it would naturally go, runs
    before the gate can still abort and would destroy a live user's files.
    """
    _admin_headers()
    _register_second_user("aborting-owner", "abortpass1")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "aborting-owner").one()
        task = Task(
            user_id=int(target.id),
            title="bound",
            description="",
            agent_config=agent_config_with_task_extension_bindings(
                {}, [_registered_extension]
            ),
        )
        db.add(task)
        db.commit()
        workspace = _make_workspace(
            _workspace_root / f"user_{int(target.id)}", int(task.id)
        )
        target_id = int(target.id)

        with pytest.raises(HTTPException) as exc_info:
            await delete_user(target_id, admin, db)

        assert exc_info.value.status_code == 503
        assert workspace.exists()
        assert db.query(User).filter(User.id == target_id).count() == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_failed_capture_is_reported_as_pending_not_as_cleaned(
    _workspace_root: Path, monkeypatch
) -> None:
    """An unresolvable scope predicts a leak, so it cannot report success.

    After the row is gone nothing can resolve the scope again, so the
    post-deletion fallback looks only under the unscoped candidates and a
    scoped workspace stays on disk. Reporting that as cleaned would make the
    field silent in exactly the case it exists to surface.
    """
    _admin_headers()
    _register_second_user("capture-fail-owner", "capturepass1")
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "capture-fail-owner").one()
        task = Task(user_id=int(owner.id), title="capture", description="")
        db.add(task)
        db.commit()
        task_id = int(task.id)
        workspace = _make_workspace(_workspace_root / f"user_{int(owner.id)}", task_id)

        def _unresolvable(*args, **kwargs):
            raise RuntimeError("scope resolver is down")

        monkeypatch.setattr(
            "xagent.web.services.task_workspace_cleanup."
            "capture_workspace_cleanup_target",
            _unresolvable,
        )

        result = await delete_task(task_id, db=db, user=owner)

        assert result["success"] is True
        assert result["workspace_cleanup_pending"] is True
        # Pending is the honest answer for a scope nobody could resolve, but
        # the unscoped candidates are still probed: an unscoped workspace is
        # reclaimed rather than abandoned alongside the warning.
        assert not workspace.exists()
        assert db.query(Task).filter(Task.id == task_id).count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_workspace_removal_runs_off_the_event_loop(
    _workspace_root: Path, monkeypatch
) -> None:
    """Removal is a recursive rmtree, so it must not run on the loop thread.

    A task workspace can hold a large tree; deleting it inline would stall
    every other request and WebSocket heartbeat on this worker.
    """
    _admin_headers()
    _register_second_user("offloop-owner", "offlooppass1")
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "offloop-owner").one()
        task = Task(user_id=int(owner.id), title="offloop", description="")
        db.add(task)
        db.commit()
        task_id = int(task.id)
        _make_workspace(_workspace_root / f"user_{int(owner.id)}", task_id)

        loop_thread_ident = threading.get_ident()
        removal_threads: list[int] = []
        real_remove = agent_service_manager.remove_task_workspace

        def _record(target):
            removal_threads.append(threading.get_ident())
            return real_remove(target)

        monkeypatch.setattr(agent_service_manager, "remove_task_workspace", _record)

        await delete_task(task_id, db=db, user=owner)

        assert removal_threads, "workspace removal never ran"
        on_loop = [ident for ident in removal_threads if ident == loop_thread_ident]
        assert not on_loop, (
            f"{len(on_loop)} of {len(removal_threads)} workspace removals ran on "
            "the event loop thread"
        )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_user_delete_reports_pending_when_a_workspace_cannot_be_removed(
    _workspace_root: Path, monkeypatch
) -> None:
    """An admin told only "deleted" cannot learn that directories survived."""
    _admin_headers()
    _register_second_user("admin-pending-owner", "adminpendingpass1")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "admin-pending-owner").one()
        task = Task(user_id=int(target.id), title="stuck", description="")
        db.add(task)
        db.commit()
        _make_workspace(_workspace_root / f"user_{int(target.id)}", int(task.id))
        target_id = int(target.id)

        def _explode(_target):
            raise OSError("directory busy")

        monkeypatch.setattr(admin_users, "remove_task_workspace", _explode)

        response = await delete_user(target_id, admin, db)

        assert response == {
            "message": "User deleted successfully",
            "workspace_cleanup_pending": True,
        }
        assert db.query(User).filter(User.id == target_id).count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_user_delete_removes_workspaces_only_after_the_rows_are_gone(
    _workspace_root: Path, monkeypatch
) -> None:
    """Removal must not overtake the deletion it belongs to.

    The abort guard above pins that cleanup stays out of the task-page walk.
    This pins the other half, which the walk guard cannot reach: by the time a
    directory is removed, the rows it belonged to are already committed as
    deleted. Anything that moved removal earlier would be destroying files
    while the account could still survive the request.
    """
    _admin_headers()
    _register_second_user("ordering-owner", "orderingpass1")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "ordering-owner").one()
        task = Task(user_id=int(target.id), title="ordered", description="")
        db.add(task)
        db.commit()
        _make_workspace(_workspace_root / f"user_{int(target.id)}", int(task.id))
        target_id = int(target.id)
        task_id = int(task.id)

        rows_at_removal: list[int] = []
        real_remove = admin_users.remove_task_workspace

        def _observe(target_):
            probe = _direct_db_session()
            try:
                rows_at_removal.append(
                    probe.query(Task).filter(Task.id == task_id).count()
                )
            finally:
                probe.close()
            return real_remove(target_)

        monkeypatch.setattr(admin_users, "remove_task_workspace", _observe)

        await delete_user(target_id, admin, db)

        assert rows_at_removal == [0], (
            "a workspace was removed while its task row still existed: "
            f"{rows_at_removal}"
        )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_user_delete_cleans_a_scoped_workspace(_workspace_root: Path) -> None:
    """The account path resolves each task's own scope, not one shared answer.

    It captures for many tasks at once, so it cannot use an activated scope;
    every task's segments have to come from its own row.
    """
    _admin_headers()
    _register_second_user("bulk-scoped-owner", "bulkscopedpass1")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "bulk-scoped-owner").one()
        scope = ExecutionScope(
            sandbox_key_suffix="team-4", workspace_segments=("team-4",)
        )
        task = Task(
            user_id=int(target.id),
            title="bulk scoped",
            description="",
            agent_config={EXECUTION_SCOPE_AGENT_CONFIG_KEY: scope.to_dict()},
        )
        db.add(task)
        db.commit()
        workspace = _make_workspace(
            _workspace_root / f"user_{int(target.id)}" / "team-4", int(task.id)
        )
        target_id = int(target.id)

        response = await delete_user(target_id, admin, db)

        assert response == {
            "message": "User deleted successfully",
            "workspace_cleanup_pending": False,
        }
        assert not workspace.exists()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_user_delete_reports_pending_when_a_capture_fails(
    _workspace_root: Path, monkeypatch
) -> None:
    """A scope nobody can resolve is a pending cleanup, not a clean account.

    The unscoped candidates are still probed, so an unscoped directory is
    reclaimed -- but a workspace under a scope segment cannot be named by
    them, and that is what the flag is reporting.
    """
    _admin_headers()
    _register_second_user("bulk-capture-owner", "bulkcapturepass1")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "bulk-capture-owner").one()
        task = Task(user_id=int(target.id), title="bulk capture", description="")
        db.add(task)
        db.commit()
        workspace = _make_workspace(
            _workspace_root / f"user_{int(target.id)}", int(task.id)
        )
        target_id = int(target.id)

        def _unresolvable(*args, **kwargs):
            raise RuntimeError("scope resolver is down")

        monkeypatch.setattr(
            "xagent.web.services.task_workspace_cleanup."
            "capture_workspace_cleanup_target",
            _unresolvable,
        )

        response = await delete_user(target_id, admin, db)

        assert response == {
            "message": "User deleted successfully",
            "workspace_cleanup_pending": True,
        }
        assert not workspace.exists()
        assert db.query(User).filter(User.id == target_id).count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_cancelled_delete_still_detaches_the_deleted_tasks_runtime(
    _workspace_root: Path, monkeypatch
) -> None:
    """Cancellation must not strand a deleted task's connections.

    Everything after the row delete suspends, and ``CancelledError`` is a
    ``BaseException``, so no handler in the endpoint catches it. If the detach
    and the runtime cleanup sat below that suspension point, a cancelled
    request would leave the WebSocket connections of a task that no longer
    exists attached, and its background execution still running.
    """
    _admin_headers()
    _register_second_user("cancelled-owner", "cancelledpass1")
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "cancelled-owner").one()
        task = Task(user_id=int(owner.id), title="cancelled", description="")
        db.add(task)
        db.commit()
        task_id = int(task.id)

        detached: list[int] = []
        real_detach = websocket_module.manager.detach_task_connections

        def _record_detach(observed_task_id):
            detached.append(observed_task_id)
            return real_detach(observed_task_id)

        monkeypatch.setattr(
            websocket_module.manager, "detach_task_connections", _record_detach
        )

        def _cancelled(*args, **kwargs):
            raise asyncio.CancelledError()

        monkeypatch.setattr(
            type(get_agent_manager()), "_cleanup_workspace_directory", _cancelled
        )

        cancelled: list[int] = []

        async def _record_cancel(observed_task_id, timeout_seconds=None):
            cancelled.append(observed_task_id)

        monkeypatch.setattr(
            task_execution_module.background_task_manager,
            "cancel_task",
            _record_cancel,
        )

        with pytest.raises(asyncio.CancelledError):
            await delete_task(task_id, db=db, user=owner)

        # Scheduled before the suspension point, so it is already owned by the
        # loop and runs even though the handler never returned.
        await asyncio.sleep(0)

        assert detached == [task_id], (
            "the deleted task's connections were never detached"
        )
        assert cancelled == [task_id], (
            "the deleted task's background execution was never cancelled"
        )
        assert db.query(Task).filter(Task.id == task_id).count() == 0
    finally:
        db.close()
