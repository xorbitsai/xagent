"""What the retention purge leaves owed when it expires a conversation (#2587).

The purge deletes rows under a row lock and makes no external call, so the
workspace directory and any runtime-extension state of an expired task are
not released by the purge itself. They are recorded, in the purge's own
transaction, for the cleanup retry driver. These tests pin that split: what
the purge records, that a dry run records nothing, that a failed purge records
nothing, and that the driver then actually releases what was recorded.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import task_retention_purge as purge_module
from xagent.web.services.task_cleanup_obligations import (
    CleanupObligationStatus,
    CleanupResourceKind,
    list_cleanup_obligations,
    run_cleanup_obligation_batch,
)
from xagent.web.services.task_retention_purge import (
    RetentionPurgeAction,
    purge_task,
    run_retention_purge_batch,
)
from xagent.web.services.task_runtime import (
    agent_config_with_task_extension_bindings,
    register_task_extension,
    unregister_task_extension,
)

engine = engine_fixture

NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
CONVERSATION_DAYS = 365
TRACE_DAYS = 90


@pytest.fixture(autouse=True)
def _environment(tmp_path, monkeypatch) -> Path:
    from xagent import config as _config

    for name in _config.RETENTION_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads))
    monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", "")
    return uploads


@pytest.fixture
def sessions(engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    return sa.orm.sessionmaker(bind=engine, autoflush=False)


class _Provider:
    def __init__(self) -> None:
        self.deleted: list[int] = []

    async def on_task_created(self, context, configuration) -> None:
        return None

    async def build_runtime(self, context):
        return None

    async def public_metadata(self, context):
        return None

    async def on_task_deleted(self, context) -> None:
        self.deleted.append(context.task_id)


@pytest.fixture
def provider():
    instance = _Provider()
    register_task_extension("purge_sandbox", instance)
    yield instance
    unregister_task_extension("purge_sandbox")


def _expired_task(
    sessions,
    username: str,
    *,
    age_days: int = 400,
    agent_config: dict | None = None,
) -> tuple[int, int]:
    with sessions() as db:
        user = User(username=username, password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(
            user_id=int(user.id),
            title="retention cleanup fixture",
            status=TaskStatus.COMPLETED,
            last_activity_at=NOW - timedelta(days=age_days),
            agent_config=agent_config,
        )
        db.add(task)
        db.commit()
        return int(user.id), int(task.id)


def _make_workspace(uploads: Path, owner_id: int, task_id: int) -> Path:
    workspace = uploads / f"user_{owner_id}" / f"web_task_{task_id}"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "result.txt").write_text("payload")
    return workspace


def _purge(sessions, task_id: int, *, dry_run: bool = False) -> RetentionPurgeAction:
    with sessions() as db:
        return purge_task(
            db,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
            dry_run=dry_run,
        )


def _owed(sessions) -> list:
    with sessions() as db:
        return list_cleanup_obligations(db)


def test_conversation_expiry_records_the_workspace_and_bound_extensions(
    sessions, _environment: Path, provider
) -> None:
    owner_id, task_id = _expired_task(
        sessions,
        "expired-owner",
        agent_config=agent_config_with_task_extension_bindings({}, ["purge_sandbox"]),
    )
    workspace = _make_workspace(_environment, owner_id, task_id)

    assert _purge(sessions, task_id) is RetentionPurgeAction.PURGED_CONVERSATION

    # The purge itself released nothing: it holds a row lock while it runs.
    assert workspace.exists()
    assert provider.deleted == []
    owed = {(o.kind, o.key): o for o in _owed(sessions)}
    assert set(owed) == {
        (CleanupResourceKind.WORKSPACE, ""),
        (CleanupResourceKind.RUNTIME_EXTENSION, "purge_sandbox"),
    }
    assert all(o.status is CleanupObligationStatus.PENDING for o in owed.values())
    assert all(o.owner_id == owner_id for o in owed.values())


@pytest.mark.asyncio
async def test_the_retry_driver_releases_what_the_purge_recorded(
    sessions, _environment: Path, provider
) -> None:
    owner_id, task_id = _expired_task(
        sessions,
        "driven-owner",
        agent_config=agent_config_with_task_extension_bindings({}, ["purge_sandbox"]),
    )
    workspace = _make_workspace(_environment, owner_id, task_id)
    _purge(sessions, task_id)

    report = await run_cleanup_obligation_batch(sessions, now=NOW)

    assert report.completed == 2
    assert not workspace.exists()
    assert provider.deleted == [task_id]
    assert _owed(sessions) == []


def test_a_dry_run_records_nothing_and_touches_no_filesystem(
    sessions, _environment: Path
) -> None:
    owner_id, task_id = _expired_task(sessions, "dry-owner")
    workspace = _make_workspace(_environment, owner_id, task_id)

    assert (
        _purge(sessions, task_id, dry_run=True)
        is RetentionPurgeAction.PURGED_CONVERSATION
    )

    assert workspace.exists()
    assert _owed(sessions) == []


def test_trace_expiry_owes_nothing(sessions, _environment: Path) -> None:
    """The conversation stays, and so does the workspace it owns."""
    owner_id, task_id = _expired_task(sessions, "trace-owner", age_days=100)
    with sessions() as db:
        db.add(
            purge_module.TraceEvent(
                task_id=task_id,
                event_id=f"evt-{task_id}",
                event_type="agent_execution_checkpoint",
                timestamp=NOW,
                data={},
            )
        )
        db.commit()

    assert _purge(sessions, task_id) is RetentionPurgeAction.PURGED_TRACES
    assert _owed(sessions) == []


def test_a_purge_that_fails_records_nothing(
    sessions, _environment: Path, monkeypatch
) -> None:
    """Same transaction as the row deletion: rolled back together."""
    _owner_id, task_id = _expired_task(sessions, "failing-owner")

    def _explode(db, *, task_id, detached_reason):
        raise RuntimeError("a foreign key this census missed")

    monkeypatch.setattr(purge_module, "purge_task_rows", _explode)

    with pytest.raises(RuntimeError):
        _purge(sessions, task_id)

    assert _owed(sessions) == []
    with sessions() as db:
        assert db.get(Task, task_id) is not None


def test_the_batch_report_separates_rows_deleted_from_cleanup_owed(
    sessions, _environment: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        purge_module, "ensure_retention_purge_supported", lambda db: None
    )
    _expired_task(sessions, "report-owner-1")
    _expired_task(sessions, "report-owner-2")

    report = run_retention_purge_batch(
        sessions, now=NOW, periods=(CONVERSATION_DAYS, TRACE_DAYS), dry_run=False
    )

    assert report.purged_conversations == 2
    assert report.cleanup_owed == 2
    assert "cleanup_owed=2" in report.audit_line()
