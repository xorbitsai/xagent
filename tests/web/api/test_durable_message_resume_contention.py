"""Durable form responses defer while the live-control resume slot is busy."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    ResumeReservationOutcome,
    execute_durable_task_command,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent
from xagent.web.models.user import User
from xagent.web.services.chat_history_service import DELIVERY_PENDING
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    COMMAND_PENDING,
    COMMAND_PROCESSING,
    MAX_COMMAND_DEFERS,
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    dispatch_one_task_command,
    enqueue_task_command,
    get_runner_id,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'resume_contention.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _user(db, username: str) -> User:
    user = User(username=username, password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _live_task(db, owner_id: int) -> Task:
    task = Task(
        user_id=owner_id,
        title="t",
        description="d",
        status=TaskStatus.RUNNING,
        execution_mode="balanced",
        source="sdk",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    task.runner_id = "live-runner"
    task.run_id = "live-run"
    db.commit()
    return task


def _message_command(
    task: Task,
    owner: User,
    command_id: str,
    *,
    attempt_count: int = 1,
) -> ClaimedTaskCommand:
    return ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id=command_id,
        kind=TaskCommandKind.MESSAGE,
        payload={
            "type": "chat_message",
            "message": "apply this form response once",
            "client_message_id": command_id,
            "files": [],
        },
        target_run_id="live-run",
        attempt_count=attempt_count,
    )


@contextmanager
def _live_control_environment(
    *,
    outcome: ResumeReservationOutcome | None = None,
    background_manager=None,
) -> Iterator[tuple[MagicMock, object]]:
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(return_value=True)
    if background_manager is None:
        background_manager = MagicMock()
        background_manager.try_reserve_resume.return_value = outcome
        background_manager.resume_holder_age_seconds.return_value = None
        background_manager.running_tasks.get.return_value = None

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch(
            "xagent.web.api.websocket.manager",
            MagicMock(
                broadcast_to_task=AsyncMock(),
                send_personal_message=AsyncMock(),
            ),
        ),
        patch("xagent.web.api.websocket.execute_resume_background", AsyncMock()),
        patch(
            "xagent.web.api.websocket.background_task_manager",
            background_manager,
        ),
    ):
        yield agent, background_manager


@pytest.mark.asyncio
async def test_cancel_clears_pre_registration_resume_reservation() -> None:
    background_manager = websocket_api.BackgroundTaskManager()
    assert (
        background_manager.try_reserve_resume(7, expected_run_id="live-run")
        is ResumeReservationOutcome.RESERVED
    )
    assert background_manager.resume_holder_age_seconds(7) is not None

    outcome = await background_manager.cancel_task(7)

    assert outcome.requested is False
    assert background_manager.resume_holder_age_seconds(7) is None
    assert (
        background_manager.try_reserve_resume(7, expected_run_id="live-run")
        is ResumeReservationOutcome.RESERVED
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        ResumeReservationOutcome.RESERVATION_HELD,
        ResumeReservationOutcome.COORDINATOR_RUNNING,
        ResumeReservationOutcome.SHUTTING_DOWN,
    ],
)
async def test_contended_durable_message_defers_without_injection(
    db_session,
    outcome: ResumeReservationOutcome,
) -> None:
    owner = _user(db_session, f"contention-owner-{outcome.value}")
    task = _live_task(db_session, int(owner.id))
    command = _message_command(task, owner, "form-turn")

    with (
        _live_control_environment(outcome=outcome) as (agent, _),
        pytest.raises(TaskCommandDeferred) as exc_info,
    ):
        await execute_durable_task_command(command)

    assert "resume slot" in str(exc_info.value)
    assert exc_info.value.resend_safe is True
    agent.post_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_delivery_makes_contention_unsafe_to_resend(
    db_session,
) -> None:
    owner = _user(db_session, "recovered-delivery-owner")
    task = _live_task(db_session, int(owner.id))
    command_id = "recovered-form-turn"
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            message_type="user_message",
            content="apply this form response once",
            turn_id=command_id,
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    command = _message_command(
        task,
        owner,
        command_id,
        attempt_count=2,
    )

    with (
        _live_control_environment(
            outcome=ResumeReservationOutcome.RESERVATION_HELD
        ) as (agent, _),
        pytest.raises(TaskCommandDeferred) as exc_info,
    ):
        await execute_durable_task_command(command)

    assert exc_info.value.resend_safe is False
    agent.post_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatcher_reclaims_and_applies_message_after_contention_clears(
    db_session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger=websocket_api.__name__)
    owner = _user(db_session, "eventual-delivery-owner")
    task = _live_task(db_session, int(owner.id))
    task.runner_id = get_runner_id()
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    db_session.commit()

    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id="eventual-form-turn",
        kind=TaskCommandKind.MESSAGE,
        payload={
            "type": "chat_message",
            "message": "apply this form response once",
            "client_message_id": "eventual-form-turn",
            "files": [],
        },
    )
    db_session.commit()

    background_manager = websocket_api.BackgroundTaskManager()
    assert (
        background_manager.try_reserve_resume(
            int(task.id),
            expected_run_id="live-run",
        )
        is ResumeReservationOutcome.RESERVED
    )
    with _live_control_environment(background_manager=background_manager) as (
        agent,
        _,
    ):
        assert await dispatch_one_task_command(
            execute_durable_task_command,
            command_db_id=enqueued.command_id,
        )

        db_session.expire_all()
        stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
        assert stored is not None
        assert stored.status == COMMAND_PENDING
        assert stored.failure_count == 0
        assert stored.defer_count == 1
        # A defer that still has budget must not stage a terminal event.
        assert (
            db_session.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == enqueued.command_id)
            .count()
            == 0
        )
        assert "holder_age_seconds=" in caplog.text
        assert "holder_age_seconds=unknown" not in caplog.text

        background_manager.release_resume_reservation(int(task.id))
        stored.claim_expires_at = None
        db_session.commit()
        assert await dispatch_one_task_command(
            execute_durable_task_command,
            command_db_id=enqueued.command_id,
        )
        await asyncio.sleep(0)

    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_COMPLETED
    assert stored.failure_count == 0
    assert stored.defer_count == 1
    agent.post_user_message.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unsettled_prior_claim", "recovered_delivery", "expected_resend_safe"),
    [
        (False, False, True),
        # Models worker B reclaiming an expired attempt from worker A before
        # worker A notices claim loss. The missing delivery row is not durable
        # proof that worker A cannot resume and inject after this read.
        (True, False, False),
        # A recovered delivery row alone must force unsafe even when the
        # claim arithmetic is clean: this pins the marker term in isolation.
        (False, True, False),
        (True, True, False),
    ],
)
async def test_exhausted_contention_persists_evidence_based_resend_safety(
    db_session,
    unsettled_prior_claim: bool,
    recovered_delivery: bool,
    expected_resend_safe: bool,
) -> None:
    owner = _user(
        db_session,
        f"terminal-owner-{unsettled_prior_claim}-{recovered_delivery}",
    )
    task = _live_task(db_session, int(owner.id))
    task.runner_id = get_runner_id()
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    db_session.commit()
    command_id = f"terminal-form-turn-{unsettled_prior_claim}-{recovered_delivery}"
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id=command_id,
        kind=TaskCommandKind.MESSAGE,
        payload={
            "type": "chat_message",
            "message": "apply this form response once",
            "client_message_id": command_id,
            "files": [],
        },
    )
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    stored.defer_count = MAX_COMMAND_DEFERS - 1
    # Each prior clean contention both claimed and deferred once. A stale
    # worker adds one unmatched claim that the reclaiming worker must treat as
    # possible in-flight injection evidence even when no delivery row exists.
    stored.attempt_count = MAX_COMMAND_DEFERS - 1
    if unsettled_prior_claim:
        stored.status = COMMAND_PROCESSING
        stored.claimed_by = "stale-worker"
        stored.claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        stored.attempt_count += 1
    if recovered_delivery:
        db_session.add(
            TaskChatMessage(
                task_id=int(task.id),
                user_id=int(owner.id),
                role="user",
                message_type="user_message",
                content="apply this form response once",
                turn_id=command_id,
                delivery_status=DELIVERY_PENDING,
            )
        )
    db_session.commit()

    with _live_control_environment(
        outcome=ResumeReservationOutcome.RESERVATION_HELD
    ) as (agent, _):
        assert await dispatch_one_task_command(
            execute_durable_task_command,
            command_db_id=enqueued.command_id,
        )

    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_FAILED
    assert stored.failure_count == 0
    assert stored.defer_count == MAX_COMMAND_DEFERS
    event = (
        db_session.query(TaskCommandTerminalEvent)
        .filter(TaskCommandTerminalEvent.task_command_id == enqueued.command_id)
        .one()
    )
    assert event.resend_safe is expected_resend_safe
    agent.post_user_message.assert_not_awaited()
