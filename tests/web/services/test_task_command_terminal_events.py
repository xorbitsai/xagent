from __future__ import annotations

import asyncio
import os
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier
from threading import Event as ThreadEvent
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy import event as sa_event
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateSchema, DropSchema

from xagent.web.api.admin_users import delete_user
from xagent.web.models.database import (
    Base,
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent
from xagent.web.models.user import User
from xagent.web.services import task_command_transport as task_command_transport_module
from xagent.web.services.task_command_terminal_events import (
    TerminalTaskEventDraft,
    stage_terminal_event,
)
from xagent.web.services.task_command_transport import (
    TaskCommandKind,
    claim_task_command,
    enqueue_task_command,
    fail_task_command,
    finish_task_command,
    retry_failed_task_command,
)


@pytest.fixture(
    params=(
        "sqlite",
        pytest.param("postgresql", marks=pytest.mark.postgresql),
    )
)
def db_session(request: pytest.FixtureRequest, tmp_path):
    admin_engine = None
    isolated_schema = None
    if request.param == "postgresql":
        raw_url = os.getenv("XAGENT_TEST_POSTGRES_URL") or os.getenv(
            "POSTGRES_TEST_DATABASE_URL"
        )
        if not raw_url:
            pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
        isolated_schema = f"xagent_terminal_events_{uuid.uuid4().hex}"
        admin_engine = create_engine(raw_url)
        with admin_engine.begin() as connection:
            connection.execute(CreateSchema(isolated_schema))
        db_url = str(
            make_url(raw_url).update_query_dict(
                {"options": f"-csearch_path={isolated_schema}"}
            )
        )
    else:
        db_url = f"sqlite:///{tmp_path / 'terminal-task-events.db'}"

    init_db(db_url=db_url)
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        if admin_engine is None:
            Base.metadata.drop_all(bind=get_engine())
        else:
            get_engine().dispose()
            assert isolated_schema is not None
            with admin_engine.begin() as connection:
                connection.execute(DropSchema(isolated_schema, cascade=True))
            admin_engine.dispose()


def _create_running_task(db) -> tuple[User, Task]:
    user = User(username="terminal-event-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    task = Task(
        user_id=user.id,
        title="Terminal task events",
        description="Terminal task events",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        run_id="run-1",
        state_version=3,
        runner_id="worker-a",
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return user, task


def _claim_command(
    db,
    user: User,
    task: Task,
    command_id: str,
    *,
    kind: TaskCommandKind = TaskCommandKind.PAUSE,
):
    enqueued = enqueue_task_command(
        db,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id=command_id,
        kind=kind,
        payload={"type": f"{kind.value}_task"},
    )
    claimed = claim_task_command(
        db,
        runner_id="worker-a",
        command_db_id=enqueued.command_id,
    )
    assert claimed is not None
    return claimed


def test_terminal_event_uses_the_command_acceptance_snapshot(db_session) -> None:
    accepted_owner, task = _create_running_task(db_session)
    accepted_owner_subject = accepted_owner.actor_subject
    assert accepted_owner_subject is not None
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(accepted_owner.id),
        command_id="stale-run-outcome",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )

    current_owner = User(
        username="terminal-event-reassigned-owner",
        password_hash="hash",
        is_admin=False,
    )
    db_session.add(current_owner)
    db_session.flush()
    task.user_id = int(current_owner.id)
    task.run_id = "run-2"
    task.state_version = 9
    db_session.commit()
    claimed = claim_task_command(
        db_session,
        runner_id="worker-a",
        command_db_id=enqueued.command_id,
    )
    assert claimed is not None
    assert fail_task_command(
        claimed.id,
        "worker-a",
        "internal detail",
        force_terminal=True,
        expected_attempt_count=claimed.attempt_count,
    )

    db_session.expire_all()
    command = db_session.get(TaskExecutionCommand, claimed.id)
    event = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
        .one()
    )
    assert command is not None
    assert command.target_run_id == "run-1"
    assert command.target_state_version == 3
    assert command.task_owner_user_id == int(accepted_owner.id)
    assert command.task_owner_subject == accepted_owner_subject
    assert event.task_run_id == "run-1"
    assert event.task_state_version == 3
    assert event.task_owner_user_id == int(accepted_owner.id)
    assert event.task_owner_subject == accepted_owner_subject


def test_terminal_event_refreshes_identity_mapped_command_after_bulk_update(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    claimed = _claim_command(db_session, user, task, "identity-mapped-command")
    command = db_session.get(TaskExecutionCommand, claimed.id)
    assert command is not None
    assert command.status == "processing"

    updated = (
        db_session.query(TaskExecutionCommand)
        .filter(TaskExecutionCommand.id == claimed.id)
        .update(
            {TaskExecutionCommand.status: "completed"},
            synchronize_session=False,
        )
    )
    assert updated == 1
    assert command.status == "processing"

    event = stage_terminal_event(db_session, command_db_id=claimed.id)

    assert event.outcome == "completed"


@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_external_cancel_suppresses_command_identity_projection(
    db_session,
    outcome: str,
) -> None:
    user, task = _create_running_task(db_session)
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id=f"external-{outcome}",
        kind=TaskCommandKind.CANCEL,
        payload={"type": "cancel_task", "scope": "external"},
    )
    claimed = claim_task_command(
        db_session,
        runner_id="worker-a",
        command_db_id=enqueued.command_id,
    )
    assert claimed is not None
    if outcome == "completed":
        assert finish_task_command(
            claimed.id,
            "worker-a",
            expected_attempt_count=claimed.attempt_count,
        )
    else:
        assert fail_task_command(
            claimed.id,
            "worker-a",
            "rejected",
            force_terminal=True,
            expected_attempt_count=claimed.attempt_count,
        )

    db_session.expire_all()
    event = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
        .one()
    )
    assert event.include_command_identity is False
    assert event.command_id == f"external-{outcome}"
    assert event.command_kind == TaskCommandKind.CANCEL.value
    assert event.actor_user_id == user.id
    assert event.task_owner_user_id == user.id
    assert event.task_run_id == "run-1"
    assert event.task_state_version == 3
    assert event.outcome_version == claimed.attempt_count


@pytest.mark.parametrize(
    ("kind", "payload", "with_actor", "expected_identity"),
    [
        pytest.param(
            TaskCommandKind.PAUSE,
            {"scope": "external"},
            True,
            # The disclosure rule is scope-based, not kind-based: any
            # external-scope command withholds durable command identity,
            # because the audience is anonymous whatever the kind
            # (#1979 review NEW-1 widened this from cancel-only).
            False,
            id="external-scope-on-non-cancel",
        ),
        pytest.param(
            TaskCommandKind.CANCEL,
            {"type": "cancel_task"},
            True,
            True,
            id="cancel-without-scope",
        ),
        pytest.param(
            TaskCommandKind.CANCEL,
            {"scope": ["external"]},
            True,
            True,
            id="cancel-with-non-string-scope",
        ),
        pytest.param(
            TaskCommandKind.PAUSE,
            {"type": "pause_task"},
            False,
            True,
            id="actorless-command",
        ),
    ],
)
def test_terminal_event_identity_matches_kind_scope_and_actor(
    db_session,
    kind: TaskCommandKind,
    payload: dict,
    with_actor: bool,
    expected_identity: bool,
) -> None:
    user, task = _create_running_task(db_session)
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id) if with_actor else None,
        command_id=f"identity-axis-{uuid.uuid4().hex}",
        kind=kind,
        payload=payload,
    )
    claimed = claim_task_command(
        db_session,
        runner_id="worker-a",
        command_db_id=enqueued.command_id,
    )
    assert claimed is not None
    assert finish_task_command(
        claimed.id,
        "worker-a",
        expected_attempt_count=claimed.attempt_count,
    )

    db_session.expire_all()
    event = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
        .one()
    )
    assert event.include_command_identity is expected_identity
    assert event.actor_user_id == (int(user.id) if with_actor else None)
    assert (event.actor_subject is not None) is with_actor


def test_terminal_event_preserves_explicit_identity_suppression(db_session) -> None:
    user, task = _create_running_task(db_session)
    claimed = _claim_command(db_session, user, task, "suppressed-command-identity")
    assert fail_task_command(
        claimed.id,
        "worker-a",
        "rejected",
        force_terminal=True,
        expected_attempt_count=claimed.attempt_count,
        terminal_event=TerminalTaskEventDraft(
            message_code=None,
            resend_safe=False,
            include_command_identity=False,
        ),
    )

    db_session.expire_all()
    event = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
        .one()
    )
    assert event.include_command_identity is False
    assert event.outcome == "failed"


def test_one_concurrent_claim_winner_stages_one_terminal_event(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="concurrent-terminal-event",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
    )
    barrier = Barrier(2)

    def claim(runner_id: str):
        SessionLocal = get_session_local()
        with SessionLocal() as db:
            barrier.wait()
            return claim_task_command(
                db,
                runner_id=runner_id,
                command_db_id=enqueued.command_id,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = [
            executor.submit(claim, "worker-a"),
            executor.submit(claim, "worker-b"),
        ]
        results = [attempt.result() for attempt in attempts]
    winners = [(index, result) for index, result in enumerate(results) if result]
    assert len(winners) == 1
    winner_index, winner = winners[0]
    assert winner is not None
    assert fail_task_command(
        winner.id,
        "worker-a" if winner_index == 0 else "worker-b",
        "internal detail",
        force_terminal=True,
        expected_attempt_count=winner.attempt_count,
    )

    db_session.expire_all()
    events = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == winner.id)
        .all()
    )
    assert len(events) == 1


def test_terminal_disposition_rolls_back_when_event_staging_fails(db_session) -> None:
    user, task = _create_running_task(db_session)
    claimed = _claim_command(
        db_session,
        user,
        task,
        "cancel-1",
        kind=TaskCommandKind.CANCEL,
    )

    with patch.object(
        task_command_transport_module,
        "stage_terminal_event",
        side_effect=RuntimeError("event write failed"),
    ):
        with pytest.raises(RuntimeError, match="event write failed"):
            fail_task_command(
                claimed.id,
                "worker-a",
                "internal detail",
                force_terminal=True,
                expected_attempt_count=claimed.attempt_count,
            )

    db_session.expire_all()
    command = db_session.get(TaskExecutionCommand, claimed.id)
    assert command is not None
    assert command.status == "processing"
    assert command.claimed_by == "worker-a"
    event_count = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
        .count()
    )
    assert event_count == 0

    db_session.rollback()
    assert fail_task_command(
        claimed.id,
        "worker-a",
        "retry after event write recovery",
        force_terminal=True,
        expected_attempt_count=claimed.attempt_count,
    )

    db_session.expire_all()
    recovered = db_session.get(TaskExecutionCommand, claimed.id)
    assert recovered is not None
    assert recovered.status == "failed"
    events = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
        .all()
    )
    assert len(events) == 1


def test_live_legacy_actor_replays_command_after_upgrade(db_session) -> None:
    if get_engine().dialect.name != "sqlite":
        pytest.skip("full parent-schema replay setup is SQLite-specific")

    actor, task = _create_running_task(db_session)
    actor_id = int(actor.id)
    task_id = int(task.id)
    first = enqueue_task_command(
        db_session,
        task_id=task_id,
        actor_user_id=actor_id,
        command_id="live-legacy-actor-replay",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )

    database_path = get_engine().url.database
    assert database_path is not None
    db_session.close()
    get_engine().dispose()
    legacy = sqlite3.connect(database_path)
    try:
        legacy.execute("DROP TABLE task_command_terminal_events")
        legacy.execute("ALTER TABLE task_execution_commands DROP COLUMN actor_subject")
        legacy.execute("DROP INDEX ix_users_actor_subject")
        legacy.execute("ALTER TABLE users DROP COLUMN actor_subject")
        legacy.execute(
            "ALTER TABLE task_execution_commands DROP COLUMN target_state_version"
        )
        legacy.execute(
            "UPDATE alembic_version SET version_num = ?",
            ("20260821_actor_oauth_flow_states",),
        )
        legacy.commit()
    finally:
        legacy.close()

    init_db(db_url=f"sqlite:///{database_path}")

    SessionLocal = get_session_local()
    with SessionLocal() as check:
        replay = enqueue_task_command(
            check,
            task_id=task_id,
            actor_user_id=actor_id,
            command_id="live-legacy-actor-replay",
            kind=TaskCommandKind.PAUSE,
            payload={"type": "pause_task"},
        )
        assert replay.command_id == first.command_id
        assert replay.created is False
        assert replay.payload_matches is True


@pytest.mark.parametrize(
    "replacement_before_upgrade",
    [False, True],
    ids=["orphan-remains", "actor-id-reused"],
)
def test_legacy_orphan_actor_does_not_block_terminal_disposition(
    db_session,
    replacement_before_upgrade: bool,
) -> None:
    if get_engine().dialect.name != "sqlite":
        pytest.skip("legacy orphan preservation is SQLite-specific")

    owner, task = _create_running_task(db_session)
    task_id = int(task.id)
    actor = User(
        username="terminal-event-admin-actor",
        password_hash="hash",
        is_admin=True,
    )
    db_session.add(actor)
    db_session.commit()
    actor_id = int(actor.id)
    first = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=actor_id,
        command_id="legacy-orphan-actor",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    second = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id="after-legacy-orphan-actor",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
    )
    claimed = claim_task_command(
        db_session,
        runner_id="worker-a",
        command_db_id=first.command_id,
    )
    assert claimed is not None

    database_path = get_engine().url.database
    assert database_path is not None
    db_session.close()
    get_engine().dispose()
    legacy = sqlite3.connect(database_path)
    try:
        assert legacy.execute("PRAGMA foreign_keys").fetchone() == (0,)
        legacy.execute("DROP TABLE task_command_terminal_events")
        legacy.execute("ALTER TABLE task_execution_commands DROP COLUMN actor_subject")
        legacy.execute("DROP INDEX ix_users_actor_subject")
        legacy.execute("ALTER TABLE users DROP COLUMN actor_subject")
        legacy.execute(
            "ALTER TABLE task_execution_commands DROP COLUMN target_state_version"
        )
        legacy.execute("DELETE FROM users WHERE id = ?", (actor_id,))
        if replacement_before_upgrade:
            legacy.execute(
                "INSERT INTO users "
                "(id, username, password_hash, is_admin, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    actor_id,
                    "terminal-event-replacement",
                    "hash",
                    1,
                    "9999-12-31 23:59:59",
                ),
            )
        legacy.execute(
            "UPDATE alembic_version SET version_num = ?",
            ("20260821_actor_oauth_flow_states",),
        )
        legacy.commit()
    finally:
        legacy.close()

    init_db(db_url=f"sqlite:///{database_path}")
    assert fail_task_command(
        claimed.id,
        "worker-a",
        "executor failed",
        force_terminal=True,
        expected_attempt_count=claimed.attempt_count,
    )

    SessionLocal = get_session_local()
    with SessionLocal() as check:
        command = check.get(TaskExecutionCommand, claimed.id)
        assert command is not None
        assert command.status == "failed"
        assert command.failure_count == 1
        event = (
            check.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
            .one()
        )
        assert event.actor_user_id == actor_id
        assert event.actor_subject == f"legacy-user-id:{actor_id}"
        replacement = check.get(User, actor_id)
        if replacement_before_upgrade:
            assert replacement is not None
            assert replacement.actor_subject is not None
            assert replacement.actor_subject != event.actor_subject
            replay = enqueue_task_command(
                check,
                task_id=task_id,
                actor_user_id=actor_id,
                command_id="legacy-orphan-actor",
                kind=TaskCommandKind.PAUSE,
                payload={"type": "pause_task"},
            )
            assert replay.created is False
            assert replay.payload_matches is False
            replacement_command = enqueue_task_command(
                check,
                task_id=task_id,
                actor_user_id=actor_id,
                command_id="replacement-actor-command",
                kind=TaskCommandKind.PAUSE,
                payload={"type": "pause_task"},
            )
            replacement_row = check.get(
                TaskExecutionCommand,
                replacement_command.command_id,
            )
            assert replacement_row is not None
            assert replacement_row.actor_subject == replacement.actor_subject
        else:
            assert replacement is None
        assert (
            claim_task_command(
                check,
                runner_id="worker-a",
                command_db_id=second.command_id,
            )
            is not None
        )


def test_legacy_orphan_task_owner_does_not_block_terminal_disposition(
    db_session,
) -> None:
    if get_engine().dialect.name != "sqlite":
        pytest.skip("legacy orphan preservation is SQLite-specific")

    owner, task = _create_running_task(db_session)
    actor = User(
        username="terminal-event-owner-orphan-actor",
        password_hash="hash",
        is_admin=True,
    )
    db_session.add(actor)
    db_session.commit()
    owner_id = int(owner.id)
    owner_subject = owner.actor_subject
    assert owner_subject is not None
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(actor.id),
        command_id="legacy-orphan-task-owner",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    claimed = claim_task_command(
        db_session,
        runner_id="worker-a",
        command_db_id=enqueued.command_id,
    )
    assert claimed is not None

    database_path = get_engine().url.database
    assert database_path is not None
    db_session.close()
    get_engine().dispose()
    legacy = sqlite3.connect(database_path)
    try:
        assert legacy.execute("PRAGMA foreign_keys").fetchone() == (0,)
        legacy.execute("DELETE FROM users WHERE id = ?", (owner_id,))
        legacy.commit()
    finally:
        legacy.close()

    assert fail_task_command(
        claimed.id,
        "worker-a",
        "executor failed",
        force_terminal=True,
        expected_attempt_count=claimed.attempt_count,
    )
    SessionLocal = get_session_local()
    with SessionLocal() as check:
        command = check.get(TaskExecutionCommand, claimed.id)
        event = (
            check.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
            .one()
        )
        assert command is not None
        assert command.status == "failed"
        assert event.task_owner_user_id == owner_id
        assert event.task_owner_subject == owner_subject


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delete_before_staging",
    [True, False],
    ids=["actor-deleted-before-staging", "actor-deleted-after-staging"],
)
async def test_actor_snapshot_survives_supported_user_deletion(
    db_session,
    delete_before_staging: bool,
) -> None:
    owner, task = _create_running_task(db_session)
    deleting_admin = User(
        username="terminal-event-deleting-admin",
        password_hash="hash",
        is_admin=True,
    )
    db_session.add(deleting_admin)
    db_session.flush()
    actor = User(
        username="terminal-event-cross-user-actor",
        password_hash="hash",
        is_admin=True,
    )
    db_session.add(actor)
    db_session.commit()
    actor_id = int(actor.id)
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=actor_id,
        command_id=f"actor-snapshot-{delete_before_staging}",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    accepted = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert accepted is not None
    actor_subject = str(accepted.actor_subject)
    assert actor_subject
    same_actor_command = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=actor_id,
        command_id=f"actor-snapshot-same-subject-{delete_before_staging}",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
    )
    same_actor = db_session.get(TaskExecutionCommand, same_actor_command.command_id)
    assert same_actor is not None
    assert same_actor.actor_subject == actor_subject

    if delete_before_staging:
        assert await delete_user(actor_id, deleting_admin, db_session) == {
            "message": "User deleted successfully"
        }
    claimed = claim_task_command(
        db_session,
        runner_id="worker-a",
        command_db_id=enqueued.command_id,
    )
    assert claimed is not None
    assert finish_task_command(
        claimed.id,
        "worker-a",
        expected_attempt_count=claimed.attempt_count,
    )
    if not delete_before_staging:
        assert await delete_user(actor_id, deleting_admin, db_session) == {
            "message": "User deleted successfully"
        }
    if actor in db_session:
        db_session.expunge(actor)
    replacement = User(
        username="terminal-event-replacement-actor",
        password_hash="hash",
        is_admin=True,
    )
    db_session.add(replacement)
    db_session.commit()
    if get_engine().dialect.name == "sqlite":
        assert int(replacement.id) == actor_id
    assert replacement.actor_subject != actor_subject

    db_session.expire_all()
    command = db_session.get(TaskExecutionCommand, claimed.id)
    event = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
        .one()
    )
    assert command is not None
    assert command.actor_user_id is None
    assert command.actor_subject == actor_subject
    assert event.actor_user_id == (None if delete_before_staging else actor_id)
    assert event.actor_subject == actor_subject
    assert db_session.get(Task, task.id) is not None
    assert db_session.get(User, owner.id) is not None


def test_concurrent_first_commands_share_one_actor_subject(db_session) -> None:
    actor, first_task = _create_running_task(db_session)
    actor_id = int(actor.id)
    second_task = Task(
        user_id=actor.id,
        title="Concurrent terminal task events",
        description="Concurrent terminal task events",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        run_id="run-2",
        state_version=3,
        runner_id="worker-b",
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    db_session.add(second_task)
    db_session.commit()
    db_session.query(User).filter(User.id == actor_id).update(
        {User.actor_subject: None},
        synchronize_session=False,
    )
    db_session.commit()

    barrier = Barrier(2)
    SessionLocal = get_session_local()

    def accept(task_id: int, command_id: str) -> str:
        with SessionLocal() as session:
            barrier.wait()
            enqueued = enqueue_task_command(
                session,
                task_id=task_id,
                actor_user_id=actor_id,
                command_id=command_id,
                kind=TaskCommandKind.PAUSE,
                payload={"type": "pause_task"},
            )
            command = session.get(TaskExecutionCommand, enqueued.command_id)
            assert command is not None
            assert command.actor_subject is not None
            assert command.task_owner_subject == command.actor_subject
            return str(command.actor_subject)

    with ThreadPoolExecutor(max_workers=2) as pool:
        subjects = list(
            pool.map(
                lambda item: accept(*item),
                [
                    (int(first_task.id), "concurrent-actor-subject-a"),
                    (int(second_task.id), "concurrent-actor-subject-b"),
                ],
            )
        )

    assert len(set(subjects)) == 1


@pytest.mark.asyncio
async def test_cancellation_after_commit_survives_a_fresh_database_session(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    claimed = _claim_command(db_session, user, task, "cancel-after-commit")
    committed = ThreadEvent()
    release_worker = ThreadEvent()

    def persist_terminal_disposition() -> bool:
        persisted = fail_task_command(
            claimed.id,
            "worker-a",
            "internal detail",
            force_terminal=True,
            expected_attempt_count=claimed.attempt_count,
        )
        committed.set()
        assert release_worker.wait(timeout=5)
        return persisted

    persistence = asyncio.create_task(
        task_command_transport_module._persist_task_command_disposition(
            claimed,
            disposition="fail_task_command",
            operation=persist_terminal_disposition,
        )
    )
    assert await asyncio.to_thread(committed.wait, 2)
    persistence.cancel()
    release_worker.set()
    with pytest.raises(asyncio.CancelledError):
        await persistence

    db_url = str(get_engine().url)
    db_session.close()
    get_engine().dispose()
    init_db(db_url=db_url)
    fresh_db = next(get_db())
    try:
        events = (
            fresh_db.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
            .all()
        )
        assert len(events) == 1
    finally:
        fresh_db.close()


def test_retry_appends_one_new_outcome_version_and_restaging_is_idempotent(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    first_claim = _claim_command(db_session, user, task, "retry-terminal-outcome")
    assert fail_task_command(
        first_claim.id,
        "worker-a",
        "first terminal failure",
        force_terminal=True,
        expected_attempt_count=first_claim.attempt_count,
    )
    assert retry_failed_task_command(db_session, first_claim.id)
    second_claim = claim_task_command(
        db_session,
        runner_id="worker-a",
        command_db_id=first_claim.id,
    )
    assert second_claim is not None
    assert fail_task_command(
        second_claim.id,
        "worker-a",
        "second terminal failure",
        force_terminal=True,
        expected_attempt_count=second_claim.attempt_count,
    )

    marker_username = "caller-write-survives-terminal-event-conflict"
    db_session.add(User(username=marker_username, password_hash="hash", is_admin=False))
    db_session.flush()
    statements: list[str] = []

    def record_statement(_conn, _cursor, statement, *_args) -> None:
        statements.append(statement)

    engine = get_engine()
    sa_event.listen(engine, "before_cursor_execute", record_statement)
    try:
        restaged = stage_terminal_event(db_session, command_db_id=second_claim.id)
    finally:
        sa_event.remove(engine, "before_cursor_execute", record_statement)
    db_session.commit()
    db_session.expire_all()
    events = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == second_claim.id)
        .order_by(TaskCommandTerminalEvent.outcome_version)
        .all()
    )
    assert any("SAVEPOINT" in statement.upper() for statement in statements)
    assert any("ROLLBACK TO SAVEPOINT" in statement.upper() for statement in statements)
    assert [event.outcome_version for event in events] == [1, 2]
    assert restaged.id == events[1].id
    assert db_session.query(User).filter(User.username == marker_username).one()


def test_terminal_event_draft_uses_the_command_disposition_outcome(db_session) -> None:
    user, task = _create_running_task(db_session)
    claimed = _claim_command(
        db_session,
        user,
        task,
        "mismatched-outcome",
        kind=TaskCommandKind.CANCEL,
    )
    assert fail_task_command(
        claimed.id,
        "worker-a",
        "internal detail",
        force_terminal=True,
        expected_attempt_count=claimed.attempt_count,
        terminal_event=TerminalTaskEventDraft(
            message_code=None,
            resend_safe=False,
        ),
    )

    db_session.expire_all()
    event = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == claimed.id)
        .one()
    )
    assert event.outcome == "failed"
