"""What task and account deletion owe after their rows are gone (#2587).

Both endpoints commit the row deletion before releasing the workspace and any
runtime-extension state, so a release that fails -- or a process that dies in
between -- has to leave something durable behind. These tests pin that the
obligation is written with the deletion and discharged only by a release that
actually happened, observed through ``list_cleanup_obligations`` and the
directory on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi import HTTPException

from tests.shared.execution_scope import register_scope_resolver
from xagent.core.execution_scope import set_execution_scope_snapshot_loader
from xagent.web.api import admin_users
from xagent.web.api.admin_users import delete_user
from xagent.web.api.chat import delete_task
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.services.agent_service_manager import get_agent_manager
from xagent.web.services.execution_scope_snapshot import (
    load_task_execution_scope_snapshot,
)
from xagent.web.services.task_cleanup_obligations import (
    CleanupObligationStatus,
    CleanupResourceKind,
    list_cleanup_obligations,
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


class _Provider:
    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete

    async def on_task_created(self, context, configuration) -> None:
        return None

    async def build_runtime(self, context):
        return None

    async def public_metadata(self, context):
        return None

    async def on_task_deleted(self, context) -> None:
        if self.fail_delete:
            raise RuntimeError("provider is down")


@pytest.fixture
def failing_provider() -> Iterator[str]:
    name = "flaky_sandbox"
    register_task_extension(name, _Provider(fail_delete=True))
    yield name
    unregister_task_extension(name)


def _make_workspace(base: Path, task_id: int) -> Path:
    workspace = base / f"web_task_{task_id}"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "result.txt").write_text("payload")
    return workspace


def _owned_task(db, username: str, *, agent_config: dict | None = None) -> tuple:
    _register_second_user(username, f"{username}-pass1")
    owner = db.query(User).filter(User.username == username).one()
    task = Task(
        user_id=int(owner.id),
        title=username,
        description="",
        agent_config=agent_config,
    )
    db.add(task)
    db.commit()
    return owner, int(task.id)


def _break_workspace_removal(monkeypatch) -> None:
    def _explode(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(
        type(get_agent_manager()), "_cleanup_workspace_directory", _explode
    )


# ===== task deletion =====


@pytest.mark.asyncio
async def test_a_failed_workspace_removal_leaves_a_pending_obligation(
    _workspace_root: Path, monkeypatch
) -> None:
    _admin_headers()
    db = _direct_db_session()
    try:
        owner, task_id = _owned_task(db, "obligation-owner")
        _make_workspace(_workspace_root / f"user_{int(owner.id)}", task_id)
        _break_workspace_removal(monkeypatch)

        result = await delete_task(task_id, db=db, user=owner)

        assert result["success"] is True
        assert result["workspace_cleanup_pending"] is True
        assert db.query(Task).filter(Task.id == task_id).count() == 0
        [owed] = list_cleanup_obligations(db)
        assert (owed.task_id, owed.owner_id) == (task_id, int(owner.id))
        assert owed.kind is CleanupResourceKind.WORKSPACE
        assert owed.status is CleanupObligationStatus.PENDING
        assert owed.attempts == 1
        assert "permission denied" in (owed.last_error or "")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_clean_deletion_leaves_no_obligation(_workspace_root: Path) -> None:
    _admin_headers()
    db = _direct_db_session()
    try:
        owner, task_id = _owned_task(db, "clean-owner")
        workspace = _make_workspace(_workspace_root / f"user_{int(owner.id)}", task_id)

        result = await delete_task(task_id, db=db, user=owner)

        assert result["workspace_cleanup_pending"] is False
        assert result["external_cleanup_pending"] is False
        assert not workspace.exists()
        assert list_cleanup_obligations(db) == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_an_unresolvable_scope_ends_on_the_reconciliation_list(
    _workspace_root: Path, monkeypatch
) -> None:
    _admin_headers()
    db = _direct_db_session()
    try:
        owner, task_id = _owned_task(db, "unresolved-owner")

        def _unresolvable(*args, **kwargs):
            raise RuntimeError("scope resolver is down")

        monkeypatch.setattr(
            "xagent.web.services.task_workspace_cleanup."
            "capture_workspace_cleanup_target",
            _unresolvable,
        )

        result = await delete_task(task_id, db=db, user=owner)

        assert result["workspace_cleanup_pending"] is True
        [owed] = list_cleanup_obligations(db)
        assert owed.status is CleanupObligationStatus.ABANDONED
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_force_delete_records_the_extension_it_could_not_release(
    _workspace_root: Path, failing_provider: str
) -> None:
    """The admin accepted the leak; the leak still goes on the list."""
    _admin_headers()
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        owner, task_id = _owned_task(
            db,
            "force-owner",
            agent_config=agent_config_with_task_extension_bindings(
                {}, [failing_provider]
            ),
        )

        result = await delete_task(task_id, db=db, user=admin, force=True)

        assert result["success"] is True
        [owed] = list_cleanup_obligations(db)
        assert owed.kind is CleanupResourceKind.RUNTIME_EXTENSION
        assert owed.key == failing_provider
        assert owed.owner_id == int(owner.id)
        assert owed.status is CleanupObligationStatus.ABANDONED
        assert "force delete" in (owed.last_error or "")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_refused_delete_owes_nothing(
    _workspace_root: Path, failing_provider: str
) -> None:
    """Without force the task survives for a retry, so nothing is owed yet."""
    _admin_headers()
    db = _direct_db_session()
    try:
        owner, task_id = _owned_task(
            db,
            "refused-owner",
            agent_config=agent_config_with_task_extension_bindings(
                {}, [failing_provider]
            ),
        )

        with pytest.raises(HTTPException) as refused:
            await delete_task(task_id, db=db, user=owner)

        assert refused.value.status_code == 503
        assert db.query(Task).filter(Task.id == task_id).count() == 1
        assert list_cleanup_obligations(db) == []
    finally:
        db.close()


# ===== account deletion =====


@pytest.mark.asyncio
async def test_account_deletion_records_the_workspaces_it_could_not_remove(
    _workspace_root: Path, monkeypatch
) -> None:
    _admin_headers()
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        owner, stuck_task = _owned_task(db, "account-owner")
        owner_id = int(owner.id)
        second = Task(user_id=owner_id, title="removable", description="")
        db.add(second)
        db.commit()
        removable_task = int(second.id)
        base = _workspace_root / f"user_{owner_id}"
        _make_workspace(base, stuck_task)
        removable = _make_workspace(base, removable_task)

        real_remove = admin_users.remove_task_workspace

        def _remove(target):
            if target.task_id == stuck_task:
                raise OSError("directory busy")
            real_remove(target)

        monkeypatch.setattr(admin_users, "remove_task_workspace", _remove)

        response = await delete_user(owner_id, admin, db)

        assert response["workspace_cleanup_pending"] is True
        assert not removable.exists()
        [owed] = list_cleanup_obligations(db)
        assert (owed.task_id, owed.owner_id) == (stuck_task, owner_id)
        assert owed.status is CleanupObligationStatus.PENDING
        assert "directory busy" in (owed.last_error or "")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_clean_account_deletion_leaves_no_obligation(
    _workspace_root: Path,
) -> None:
    _admin_headers()
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        owner, task_id = _owned_task(db, "clean-account-owner")
        workspace = _make_workspace(_workspace_root / f"user_{int(owner.id)}", task_id)

        response = await delete_user(int(owner.id), admin, db)

        assert response["workspace_cleanup_pending"] is False
        assert response["external_cleanup_pending"] is False
        assert not workspace.exists()
        assert list_cleanup_obligations(db) == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_account_deletion_keeps_bindings_no_process_can_release_owed(
    _workspace_root: Path,
) -> None:
    """With no provider registered at all, a bound task's provider state still
    exists somewhere; deleting the account must not drop the only record."""
    _admin_headers()
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        owner, task_id = _owned_task(
            db,
            "unregistered-owner",
            agent_config=agent_config_with_task_extension_bindings(
                {}, ["gone_provider"]
            ),
        )
        owner_id = int(owner.id)

        response = await delete_user(owner_id, admin, db)

        assert response["message"] == "User deleted successfully"
        assert response["workspace_cleanup_pending"] is False
        assert response["external_cleanup_pending"] is True
        [owed] = list_cleanup_obligations(db)
        assert owed.kind is CleanupResourceKind.RUNTIME_EXTENSION
        assert (owed.task_id, owed.key, owed.owner_id) == (
            task_id,
            "gone_provider",
            owner_id,
        )
        assert owed.status is CleanupObligationStatus.PENDING
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_stale_obligation_for_a_reused_id_does_not_swallow_the_next_one(
    _workspace_root: Path, monkeypatch
) -> None:
    """SQLite gave this task the id of an earlier, deleted one whose cleanup
    ended on the reconciliation list. Deleting this task must still record,
    and report, its own outstanding cleanup."""
    from xagent.web.services.task_cleanup_obligations import (
        record_cleanup_obligations_no_commit,
        workspace_obligation,
    )
    from xagent.web.services.task_workspace_cleanup import WorkspaceCleanupTarget

    _admin_headers()
    db = _direct_db_session()
    try:
        owner, task_id = _owned_task(db, "reused-id-owner")
        record_cleanup_obligations_no_commit(
            db,
            [
                workspace_obligation(
                    WorkspaceCleanupTarget(
                        task_id=task_id, owner_id=999, base_dirs=("/earlier",)
                    ),
                    scope_resolved=False,
                )
            ],
        )
        db.commit()
        [earlier] = list_cleanup_obligations(db)
        _break_workspace_removal(monkeypatch)

        result = await delete_task(task_id, db=db, user=owner)

        assert result["workspace_cleanup_pending"] is True
        assert result["external_cleanup_pending"] is True
        owed = {o.id: o for o in list_cleanup_obligations(db)}
        assert owed.pop(earlier.id).owner_id == 999
        [mine] = owed.values()
        assert (mine.task_id, mine.owner_id) == (task_id, int(owner.id))
        assert mine.status is CleanupObligationStatus.PENDING
        assert mine.attempts == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_force_delete_keeps_an_unregistered_provider_owed(
    _workspace_root: Path, failing_provider: str
) -> None:
    """``force`` accepts the leak of a provider that failed. A provider that is
    simply not registered here was never asked, so it stays on the retry
    queue, exactly as it would without ``force``."""
    _admin_headers()
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        _owner, task_id = _owned_task(
            db,
            "force-unregistered-owner",
            agent_config=agent_config_with_task_extension_bindings(
                {}, [failing_provider, "not_loaded_here"]
            ),
        )

        result = await delete_task(task_id, db=db, user=admin, force=True)

        assert result["external_cleanup_pending"] is True
        statuses = {o.key: o.status for o in list_cleanup_obligations(db)}
        assert statuses == {
            failing_provider: CleanupObligationStatus.ABANDONED,
            "not_loaded_here": CleanupObligationStatus.PENDING,
        }
    finally:
        db.close()
