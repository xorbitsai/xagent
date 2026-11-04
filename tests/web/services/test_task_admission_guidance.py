import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from tests.web.api.test_durable_message_resume_contention import (
    _live_control_environment,
)
from tests.web.pool_contention_shared import GUARD_TIMEOUT
from tests.web.services.test_task_execution_admission import (
    Execution,
)
from tests.web.services.test_task_execution_admission import engine as engine_fixture
from tests.web.services.test_task_execution_admission import (
    enqueue,
)
from tests.web.services.test_task_execution_admission import host as host_fixture
from xagent.core.agent.runner import UserMessageInjectionOutcome
from xagent.web.models.task import Task
from xagent.web.models.task_admission import TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import (
    task_admission_execution,
    task_command_execution,
)
from xagent.web.services import task_command_transport as transport
from xagent.web.services import (
    task_execution,
    task_interaction_close,
    task_setup_snapshot,
)
from xagent.web.services.task_execution import ResumeReservationOutcome

engine = engine_fixture
host = host_fixture


async def test_failed_live_guidance_uses_capped_backoff_then_posts_to_original_run(
    host, monkeypatch
):
    fixed_now = datetime(2030, 1, 1, tzinfo=timezone.utc)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(task_admission_execution, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        task_command_execution, "get_session_local", lambda: host.sessions
    )
    monkeypatch.setattr(task_execution, "get_session_local", lambda: host.sessions)
    monkeypatch.setattr(
        task_interaction_close, "get_session_local", lambda: host.sessions
    )
    monkeypatch.setattr(task_setup_snapshot, "get_session_local", lambda: host.sessions)

    original = enqueue(host)
    running = Execution(host)
    assert await transport.dispatch_one_task_command(running)
    with host.sessions() as db:
        task = db.get(TaskExecutionCommand, original.command_id)
        task_id = task.task_id
        original_run_id = db.get(Task, task_id).run_id
        guidance = transport.enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=host.user,
            command_id="checkpoint-guidance",
            kind=transport.TaskCommandKind.MESSAGE,
            payload={
                "message": "keep going",
                "client_message_id": "checkpoint-guidance",
            },
        )

    real_resume = task_execution.execute_resume_background
    resume_started = asyncio.Event()

    async def resume_original(*_args, **_kwargs):
        resume_started.set()
        await asyncio.Event().wait()

    background = None
    resume_waiter = None
    try:
        with _live_control_environment(outcome=ResumeReservationOutcome.RESERVED) as (
            agent,
            background_manager,
        ):
            task_execution.execute_resume_background.side_effect = real_resume
            agent.resume_execution_by_id = AsyncMock(side_effect=resume_original)
            agent.post_user_message.return_value = (
                UserMessageInjectionOutcome.NOT_POSTED
            )
            observed_delays = []
            for _ in range(8):
                assert await transport.dispatch_one_task_command(
                    task_command_execution.execute_durable_task_command,
                    command_db_id=guidance.command_id,
                )
                with host.sessions() as db, db.begin():
                    row = db.get(TaskExecutionCommand, guidance.command_id)
                    retry_at = row.retry_available_at
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    observed_delays.append((retry_at - fixed_now).total_seconds())
                    assert (row.status, row.failure_count, row.defer_count) == (
                        "pending",
                        0,
                        0,
                    )
                    row.retry_available_at = datetime.now(timezone.utc) - timedelta(
                        seconds=1
                    )

            assert observed_delays == [1, 2, 4, 8, 16, 32, 60, 60]
            agent.post_user_message.return_value = (
                UserMessageInjectionOutcome.POSTED_FRESH
            )
            assert await transport.dispatch_one_task_command(
                task_command_execution.execute_durable_task_command,
                command_db_id=guidance.command_id,
            )
            background = background_manager.register_reserved_resume.call_args.args[1]
            resume_waiter = asyncio.create_task(resume_started.wait())
            done, _ = await asyncio.wait(
                {background, resume_waiter},
                timeout=GUARD_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if background in done:
                await background
            assert resume_waiter in done

            with host.sessions() as db:
                row = db.get(TaskExecutionCommand, guidance.command_id)
                assert (row.status, row.failure_count, row.defer_count) == (
                    "completed",
                    0,
                    0,
                )
                assert db.get(TaskAdmissionTicket, guidance.command_id) is None
                assert db.get(Task, task_id).run_id == original_run_id
            assert running.started == [original.command_id]
            assert agent.post_user_message.await_count == 9
    finally:
        pending = [task for task in (resume_waiter, background) if task is not None]
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        running.finish.set()
        running.cleanup.set()


@pytest.mark.parametrize("stale_fence", ["attempt", "owner"])
async def test_stale_guidance_disposition_cannot_requeue_a_new_claim_owner(
    host, stale_fence
):
    command = enqueue(host)
    successor_retry = datetime(2040, 1, 1, tzinfo=timezone.utc)

    async def lose_claim(claimed):
        with host.sessions() as db, db.begin():
            row = db.get(TaskExecutionCommand, claimed.id)
            if stale_fence == "attempt":
                row.attempt_count = claimed.attempt_count + 1
            else:
                task = db.get(Task, claimed.task_id)
                task.runner_id = "worker-2"
                task.lease_attempt_id = "successor-attempt"
                task.lease_expires_at = datetime.now(timezone.utc) + timedelta(
                    minutes=1
                )
            row.retry_available_at = successor_retry
            row.error = "successor owns this attempt"
        raise task_admission_execution.AdmissionWaiting("retry admission")

    (dispatch_result,) = await asyncio.gather(
        transport.dispatch_one_task_command(
            lose_claim, command_db_id=command.command_id
        ),
        return_exceptions=True,
    )
    if stale_fence == "owner":
        # Owner loss may cancel the dispatcher after its cancellation-safe
        # disposition has completed; both outcomes preserve the successor.
        assert dispatch_result is True or isinstance(
            dispatch_result, asyncio.CancelledError
        )
    else:
        assert dispatch_result is True

    with host.sessions() as db:
        row = db.get(TaskExecutionCommand, command.command_id)
        retry_at = row.retry_available_at
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        assert (row.status, row.claimed_by, row.attempt_count) == (
            "processing",
            None,
            2 if stale_fence == "attempt" else 1,
        )
        assert retry_at == successor_retry
        assert row.error == "successor owns this attempt"
        if stale_fence == "owner":
            task = db.get(Task, row.task_id)
            assert (task.runner_id, task.lease_attempt_id) == (
                "worker-2",
                "successor-attempt",
            )
