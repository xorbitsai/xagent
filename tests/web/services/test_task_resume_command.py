"""Reply acceptance, exact handoff, and cross-process preparation results."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from tests.web.services.admission_capacity_shared import (
    release_bucket,
    saturate_bucket,
)
from tests.web.services.coordinator_command_shared import (
    claim_task_command,
    settle_command,
)
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import (
    task_coordinator_runtime,
    task_coordinator_service,
    task_resume,
)
from xagent.web.services import task_resume_command as module
from xagent.web.services.task_command_execution import execute_durable_task_command
from xagent.web.services.task_command_transport import TaskCommandRejected
from xagent.web.services.task_resume import TaskReplyInput, TaskResumeBusyError


def owner_lease(task_id):
    with get_session_local()() as db, db.begin():
        task_coordinator_service.recover_expired_idle_task_lease_no_commit(db, task_id)
        lease = task_coordinator_service.acquire_task_lease_no_commit(
            db, task_id, runner_id="worker-1"
        )
        if lease is None:
            task = db.get(Task, task_id)
            assert task.runner_id == "worker-1"
            lease = task_coordinator_service.TaskLease(
                task_id, task.runner_id, task.lease_attempt_id
            )
        return lease


def commit_handoff(command):
    return module._handoff(command, owner_lease(command.task_id))


@pytest.fixture
async def reply(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_coordinator_runtime, "get_runner_id", lambda: "worker-1")
    init_db(db_url=f"sqlite:///{tmp_path / 'reply.db'}")
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        agent = Agent(user_id=user.id, name="reply")
        db.add(agent)
        db.flush()
        task = Task(
            user_id=user.id,
            agent_id=agent.id,
            title="Reply",
            source="sdk",
            status=TaskStatus.WAITING_FOR_USER,
            run_id="run-1",
            state_version=1,
            control_state="waiting_for_user",
        )
        db.add(task)
        db.commit()
        ctx = TaskReplyInput(
            task_id=task.id,
            agent_id=agent.id,
            task_owner_user_id=user.id,
            actor_user_id=user.id,
            run_id="run-1",
            status=TaskStatus.WAITING_FOR_USER,
            text="answer",
        )
    yield ctx
    await task_coordinator_runtime.close_task_coordinators()
    Base.metadata.drop_all(bind=get_engine())


async def claim(ctx):
    command_id = module._admit_reply(ctx, "sdk", "", "reply-1")
    with get_session_local()() as db:
        return await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )


def test_reply_is_durable_without_acquiring_a_request_lease(reply):
    command_id = module._admit_reply(reply, "sdk", "", "reply-1")
    with get_session_local()() as db:
        task = db.get(Task, reply.task_id)
        command = db.get(TaskExecutionCommand, command_id)
        assert task.runner_id is None
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.control_state == "waiting_for_user"
        assert command.payload["text"] == "answer"
    with pytest.raises(TaskResumeBusyError):
        module._admit_reply(reply, "sdk", "", "reply-2")


async def test_reply_handoff_and_command_completion_are_one_transaction(
    reply, monkeypatch
):
    command = await claim(reply)
    monkeypatch.setattr(
        module, "finish_task_command_no_commit", lambda *args, **kwargs: False
    )
    with pytest.raises(TaskCommandRejected):
        commit_handoff(command)
    with get_session_local()() as db:
        task = db.get(Task, reply.task_id)
        assert task.runner_id == "worker-1"
        assert task.control_state == "waiting_for_user"
        assert db.get(TaskExecutionCommand, command.id).status == "processing"


async def test_old_reply_claim_cannot_acquire_or_restore_a_new_attempt(reply):
    command = await claim(reply)
    old = replace(command, attempt_count=command.attempt_count - 1)
    with pytest.raises(TaskCommandRejected):
        commit_handoff(old)
    from xagent.web.services.task_command_transport import fail_task_command

    assert not await settle_command(
        old,
        lambda: fail_task_command(
            old.id,
            "worker-1",
            "stale",
            force_terminal=True,
            expected_attempt_count=old.attempt_count,
        ),
    )
    with get_session_local()() as db:
        assert db.get(Task, reply.task_id).control_state == "waiting_for_user"
    payload, lease, owner, state = commit_handoff(command)
    assert lease.run_id == reply.run_id
    assert state["lease_attempt_id"] == lease.attempt_id
    assert module._read_reply_outcome(command.id) is None
    module._record_outcome(command, state, "accepted", owner_lease(command.task_id))
    assert module._read_reply_outcome(command.id)["outcome"] == "accepted"


@pytest.mark.asyncio
async def test_worker_uses_preacquired_lease_and_stable_turn_identity(
    reply, monkeypatch
):
    command = await claim(reply)
    prepare = AsyncMock()
    monkeypatch.setattr(task_resume, "resume_task_reply", prepare)
    await execute_durable_task_command(command)
    assert prepare.await_count == 1
    assert prepare.call_args.kwargs["turn_id"] == command.command_id
    assert prepare.call_args.kwargs["preacquired_lease"].run_id == "run-1"
    assert module._read_reply_outcome(command.id)["outcome"] == "accepted"


@pytest.mark.parametrize("replaced", [False, True])
async def test_terminal_reply_failure_restores_only_its_admission(reply, replaced):
    from xagent.web.services.task_command_transport import (
        MAX_COMMAND_FAILURES,
        fail_task_command,
    )

    command = await claim(reply)
    with get_session_local()() as db:
        db.get(TaskExecutionCommand, command.id).failure_count = (
            MAX_COMMAND_FAILURES - 1
        )
        if replaced:
            db.get(Task, reply.task_id).state_version += 1
        db.commit()
    assert await settle_command(
        command,
        lambda: fail_task_command(
            command.id,
            "worker-1",
            "unavailable",
            expected_attempt_count=command.attempt_count,
        ),
    )
    await task_coordinator_runtime.close_task_coordinators()
    with get_session_local()() as db:
        task = db.get(Task, reply.task_id)
        assert task.runner_id is None
        assert task.control_state == "waiting_for_user"
    assert module._read_reply_outcome(command.id)["outcome"] == "unavailable"


@pytest.mark.parametrize(
    "invalid",
    [{"version": True}, {"agent_id": 0}, {"run_id": ""}, {"prior_status": "paused"}],
)
async def test_invalid_reply_payload_cannot_acquire_lease(reply, invalid):
    command = await claim(reply)
    command = replace(command, payload={**command.payload, **invalid})
    with pytest.raises(TaskCommandRejected):
        commit_handoff(command)
    with get_session_local()() as db:
        assert db.get(Task, reply.task_id).status == TaskStatus.WAITING_FOR_USER


@pytest.mark.parametrize(
    "source,status",
    [
        ("sdk", TaskStatus.WAITING_FOR_USER),
        ("a2a", TaskStatus.WAITING_FOR_USER),
        ("a2a", TaskStatus.PAUSED),
    ],
)
async def test_reply_waits_for_previous_execution_to_release_lease(
    reply, source, status
):
    from datetime import datetime, timedelta, timezone

    from xagent.web.services.task_lease_service import TaskLease
    from xagent.web.services.task_orchestrator import settle_task_lease_isolated

    ctx = replace(reply, status=status)
    with get_session_local()() as db:
        task = db.get(Task, ctx.task_id)
        task.source = source
        task.status = status
        task.control_state = status.value
        task.runner_id = "old-worker"
        task.lease_attempt_id = "old-attempt"
        task.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=60)
        task.last_heartbeat_at = datetime.now(timezone.utc)
        db.commit()
    command_id = module._admit_reply(ctx, source, "message", "reply")
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.get(Task, ctx.task_id).lease_attempt_id == "old-attempt"
        assert (
            await claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
            is None
        )
        assert db.get(Task, ctx.task_id).control_state == status.value
    assert settle_task_lease_isolated(
        TaskLease(
            task_id=ctx.task_id,
            runner_id="old-worker",
            run_id=ctx.run_id,
            attempt_id="old-attempt",
        )
    )
    command_id = module._admit_reply(ctx, source, "message", "reply")
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    _, lease, _, _ = commit_handoff(command)
    assert lease.run_id == ctx.run_id


@pytest.mark.asyncio
async def test_a2a_retries_checkpoint_read_failure_with_same_message_id(
    reply, monkeypatch
):
    from types import SimpleNamespace

    from xagent.core.agent.checkpoint import CheckpointUnavailableError
    from xagent.core.agent.runner import UserMessageInjectionOutcome
    from xagent.web.services.task_lease_service import stop_task_lease_heartbeat

    with get_session_local()() as db:
        db.get(Task, reply.task_id).source = "a2a"
        db.commit()
    post = AsyncMock(
        side_effect=[
            CheckpointUnavailableError("temporary read outage"),
            UserMessageInjectionOutcome.POSTED_FRESH,
        ]
    )
    manager = SimpleNamespace(
        get_agent_for_task=AsyncMock(
            return_value=SimpleNamespace(post_user_message=post)
        )
    )
    monkeypatch.setattr(
        task_resume.agent_runtime_service, "get_agent_manager", lambda: manager
    )

    async def schedule(**kwargs):
        await stop_task_lease_heartbeat(
            kwargs["heartbeat_task"], kwargs["heartbeat_stop"]
        )

    scheduled = AsyncMock(side_effect=schedule)
    monkeypatch.setattr(task_resume, "_schedule_waiting_a2a_resume", scheduled)
    original_id = module._admit_reply(reply, "a2a", "message-1", "original-command")
    with get_session_local()() as db:
        first = await claim_task_command(
            db, runner_id="worker-1", command_db_id=original_id
        )
        original_version = db.get(
            TaskExecutionCommand, original_id
        ).target_state_version
    await execute_durable_task_command(first)
    assert module._read_reply_outcome(original_id)["outcome"] == "retryable_unavailable"
    with get_session_local()() as db:
        task = db.get(Task, reply.task_id)
        assert task.runner_id == "worker-1"
        assert task.control_state == "waiting_for_user"
    retry_id = module._admit_reply(reply, "a2a", "message-1", "original-command")
    assert retry_id != original_id
    assert (
        module._admit_reply(reply, "a2a", "message-1", "original-command") == retry_id
    )
    with get_session_local()() as db:
        second = await claim_task_command(
            db, runner_id="worker-1", command_db_id=retry_id
        )
    await execute_durable_task_command(second)
    assert module._read_reply_outcome(retry_id)["outcome"] == "accepted"
    assert (
        module._admit_reply(reply, "a2a", "message-1", "original-command") == retry_id
    )
    assert post.await_count == 2
    assert {call.kwargs["turn_id"] for call in post.await_args_list} == {
        f"a2a:{reply.task_id}:message-1"
    }
    scheduled.assert_awaited_once()
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 2
        assert (
            db.get(TaskExecutionCommand, original_id).target_state_version
            == original_version
        )


@pytest.mark.parametrize("outcome", ["unavailable", "accepted", "not_resumable"])
async def test_a2a_does_not_retry_unknown_or_terminal_outcome(reply, outcome):
    with get_session_local()() as db:
        db.get(Task, reply.task_id).source = "a2a"
        db.commit()
    command_id = module._admit_reply(reply, "a2a", "message", "original")
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    _, lease, _, state = commit_handoff(command)
    assert task_resume._restore_a2a_resume_prelease_sync(
        lease, status=TaskStatus.WAITING_FOR_USER
    )
    module._record_outcome(command, state, outcome, owner_lease(command.task_id))
    assert module._admit_reply(reply, "a2a", "message", "original") == command_id
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 1
        assert (
            await claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
            is None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_stage", ["after_injection", "lease_restore", "unknown_result"]
)
async def test_a2a_unsafe_checkpoint_failure_cannot_create_retry(
    reply, monkeypatch, failure_stage
):
    from types import SimpleNamespace

    from xagent.core.agent.checkpoint import CheckpointUnavailableError
    from xagent.core.agent.runner import UserMessageInjectionOutcome

    with get_session_local()() as db:
        db.get(Task, reply.task_id).source = "a2a"
        db.commit()
    post = AsyncMock(return_value=UserMessageInjectionOutcome.POSTED_FRESH)
    manager = SimpleNamespace(
        get_agent_for_task=AsyncMock(
            return_value=SimpleNamespace(post_user_message=post)
        )
    )
    monkeypatch.setattr(
        task_resume.agent_runtime_service, "get_agent_manager", lambda: manager
    )
    failure = CheckpointUnavailableError("temporary outage")
    if failure_stage == "unknown_result":
        post.return_value = UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    elif failure_stage == "after_injection":
        monkeypatch.setattr(
            task_resume, "_schedule_waiting_a2a_resume", AsyncMock(side_effect=failure)
        )
    else:
        post.side_effect = failure
        monkeypatch.setattr(
            task_resume,
            "_restore_a2a_resume_prelease_isolated",
            AsyncMock(return_value=False),
        )
    command_id = module._admit_reply(reply, "a2a", "message", "original")
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    await execute_durable_task_command(command)
    assert module._read_reply_outcome(command_id)["outcome"] == (
        "unavailable" if failure_stage == "lease_restore" else "unknown"
    )
    assert module._admit_reply(reply, "a2a", "message", "original") == command_id
    post.assert_awaited_once()
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 1


@pytest.mark.parametrize(
    "source,status",
    [
        ("sdk", TaskStatus.WAITING_FOR_USER),
        ("a2a", TaskStatus.WAITING_FOR_USER),
        ("a2a", TaskStatus.PAUSED),
    ],
)
async def test_reply_reclaims_expired_resting_lease_atomically(
    reply, monkeypatch, source, status
):
    from datetime import datetime, timedelta, timezone

    from xagent.web.services.task_lease_service import (
        TaskLease,
        TaskLeaseRefreshState,
        refresh_task_lease,
    )
    from xagent.web.services.task_orchestrator import settle_task_lease_isolated

    ctx = replace(reply, status=status)
    # Use the new worker's runner ID too: the attempt must fence an old
    # execution even when the process identity is reused.
    old_lease = TaskLease(
        task_id=ctx.task_id,
        runner_id="worker-1",
        run_id=ctx.run_id,
        attempt_id="old-attempt",
    )
    expired_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    with get_session_local()() as db:
        task = db.get(Task, ctx.task_id)
        task.source = source
        task.status = status
        task.control_state = status.value
        task.runner_id = old_lease.runner_id
        task.lease_attempt_id = old_lease.attempt_id
        task.lease_expires_at = expired_at
        original_version = task.state_version
        db.commit()

    def fail_staging(*args, **kwargs):
        raise RuntimeError("command staging failed")

    with monkeypatch.context() as patch:
        patch.setattr(module, "stage_task_command", fail_staging)
        with pytest.raises(RuntimeError, match="command staging failed"):
            module._admit_reply(ctx, source, "message", "reply")
    with get_session_local()() as db:
        task = db.get(Task, ctx.task_id)
        assert task.runner_id == old_lease.runner_id
        assert task.lease_attempt_id == old_lease.attempt_id
        assert task.lease_expires_at.replace(tzinfo=timezone.utc) == expired_at
        assert task.control_state == status.value
        assert task.state_version == original_version
        assert db.query(TaskExecutionCommand).count() == 0

    command_id = module._admit_reply(ctx, source, "message", "reply")
    with get_session_local()() as db:
        task = db.get(Task, ctx.task_id)
        assert task.runner_id == old_lease.runner_id
        assert task.lease_attempt_id == old_lease.attempt_id
        assert task.control_state == status.value
        assert task.state_version == original_version
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    _, lease, _, state = commit_handoff(command)
    assert lease.run_id == ctx.run_id
    assert lease.attempt_id != old_lease.attempt_id
    assert not settle_task_lease_isolated(old_lease)
    with get_session_local()() as db:
        assert refresh_task_lease(db, old_lease) == TaskLeaseRefreshState.LOST
        task = db.get(Task, ctx.task_id)
        assert task.lease_attempt_id == lease.attempt_id
        assert task.state_version == state["state_version"]
        assert task.control_state == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["sdk", "a2a"])
async def test_following_command_waits_for_reply_registration(
    reply, monkeypatch, source
):
    import asyncio
    import threading

    from xagent.web.services import task_command_execution
    from xagent.web.services.task_command_transport import (
        TaskCommandKind,
        stage_task_command,
    )

    with get_session_local()() as db:
        db.get(Task, reply.task_id).source = source
        db.commit()
    command_id = module._admit_reply(reply, source, "message", "reply")
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    committed, release = threading.Event(), threading.Event()
    original = module._handoff
    registered = False

    def slow_handoff(command, lease):
        handoff = original(command, lease)
        committed.set()
        assert release.wait(5)
        return handoff

    async def prepare(*args, **kwargs):
        nonlocal registered
        registered = True

    async def followup(command):
        assert registered
        return {}

    monkeypatch.setattr(module, "_handoff", slow_handoff)
    monkeypatch.setattr(
        task_resume,
        "resume_task_reply" if source == "sdk" else "resume_a2a_task",
        prepare,
    )
    effects = AsyncMock(side_effect=followup)
    monkeypatch.setattr(
        task_command_execution, "_execute_and_report_task_command", effects
    )
    resume = asyncio.create_task(
        task_command_execution.execute_durable_task_command(command)
    )
    next_task = None
    try:
        assert await asyncio.to_thread(committed.wait, 5)
        with get_session_local()() as db:
            staged = stage_task_command(
                db,
                task_id=reply.task_id,
                actor_user_id=reply.task_owner_user_id,
                command_id="followup",
                kind=TaskCommandKind.MESSAGE,
                payload={"message": "next"},
            )
            db.commit()
            following = await claim_task_command(
                db, runner_id="worker-1", command_db_id=staged.staged_db_id
            )
            assert following is not None
        next_task = asyncio.create_task(
            task_command_execution.execute_durable_task_command(following)
        )
        await asyncio.sleep(0.05)
        effects.assert_not_awaited()
        release.set()
        await asyncio.wait_for(asyncio.gather(resume, next_task), 5)
        effects.assert_awaited_once()
    finally:
        release.set()
        await asyncio.gather(
            resume, *([next_task] if next_task else []), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["sdk", "a2a"])
async def test_reply_timeout_retains_accepted_command_for_later_execution(
    reply, monkeypatch, source
):
    from unittest.mock import Mock

    from xagent.web.services import task_event_bridge

    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    monkeypatch.setattr(module, "get_task_reply_wait_timeout_seconds", lambda: 0.01)
    with get_session_local()() as db:
        db.get(Task, reply.task_id).source = source
        db.commit()
    with pytest.raises(task_resume.TaskResumeOutcomeUnknownError):
        await module.enqueue_resume_input(
            reply, source=source, message_id="message", command_id="accepted"
        )
    with get_session_local()() as db:
        row = db.query(TaskExecutionCommand).one()
        assert row.status == "pending"
        assert db.get(Task, reply.task_id).control_state == "waiting_for_user"
        command_id = row.id
    assert module._admit_reply(reply, source, "message", "accepted") == command_id
    with pytest.raises(TaskResumeBusyError):
        module._admit_reply(reply, source, "message", "duplicate")
    prepare = AsyncMock()
    monkeypatch.setattr(
        task_resume,
        "resume_task_reply" if source == "sdk" else "resume_a2a_task",
        prepare,
    )
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    await execute_durable_task_command(command)
    prepare.assert_awaited_once()
    assert module._read_reply_outcome(command_id)["outcome"] == "accepted"
    with get_session_local()() as db:
        assert (
            await claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
            is None
        )


@pytest.mark.parametrize("source", ["sdk", "a2a"])
async def test_unknown_reply_is_settled_paused_and_replayed(reply, monkeypatch, source):
    from types import SimpleNamespace

    from xagent.core.agent.runner import UserMessageInjectionOutcome

    with get_session_local()() as db:
        db.get(Task, reply.task_id).source = source
        db.commit()
    post = AsyncMock(return_value=UserMessageInjectionOutcome.OUTCOME_UNKNOWN)
    manager = SimpleNamespace(
        get_agent_for_task=AsyncMock(
            return_value=SimpleNamespace(post_user_message=post)
        )
    )
    monkeypatch.setattr(
        task_resume.agent_runtime_service, "get_agent_manager", lambda: manager
    )
    schedule = AsyncMock()
    monkeypatch.setattr(task_resume, "_schedule_waiting_reply_resume", schedule)
    monkeypatch.setattr(task_resume, "_schedule_waiting_a2a_resume", schedule)
    command_id = module._admit_reply(reply, source, "message", "unknown-reply")
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    await execute_durable_task_command(command)
    assert module._read_reply_outcome(command_id)["outcome"] == "unknown"
    assert module._admit_reply(reply, source, "message", "unknown-reply") == command_id
    with get_session_local()() as db:
        assert db.get(Task, reply.task_id).status == TaskStatus.PAUSED
    schedule.assert_not_awaited()
    post.assert_awaited_once()


@pytest.mark.parametrize("source", ["sdk", "a2a"])
async def test_fenced_reply_rejection_settles_busy_without_resuming(
    reply, monkeypatch, source
):
    from types import SimpleNamespace

    from xagent.core.agent.runner import UserMessageInjectionOutcome

    with get_session_local()() as db:
        db.get(Task, reply.task_id).source = source
        db.commit()
    post = AsyncMock(return_value=UserMessageInjectionOutcome.REJECTED_RETRYABLE)
    manager = SimpleNamespace(
        get_agent_for_task=AsyncMock(
            return_value=SimpleNamespace(post_user_message=post)
        )
    )
    monkeypatch.setattr(
        task_resume.agent_runtime_service, "get_agent_manager", lambda: manager
    )
    schedule = AsyncMock()
    monkeypatch.setattr(task_resume, "_schedule_waiting_reply_resume", schedule)
    monkeypatch.setattr(task_resume, "_schedule_waiting_a2a_resume", schedule)
    command_id = module._admit_reply(reply, source, "message", "fenced-reply")
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    await execute_durable_task_command(command)
    # Nothing was written, so the outcome is a known rejection, not "unknown".
    assert module._read_reply_outcome(command_id)["outcome"] == "busy"
    with get_session_local()() as db:
        task = db.get(Task, reply.task_id)
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.control_state == "waiting_for_user"
    schedule.assert_not_awaited()
    post.assert_awaited_once()


@pytest.mark.asyncio
async def test_capacity_blocked_reply_is_acknowledged_as_queued_without_waiting(
    reply, monkeypatch, request
):
    source = "sdk"
    from unittest.mock import Mock

    from xagent.web.services import task_event_bridge
    from xagent.web.services import task_execution_admission as admission

    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("tenant:batch", 1, 20)
    )
    request.addfinalizer(lambda: admission.set_task_admission_hook(None))
    holding = saturate_bucket(reply.task_owner_user_id, reply.agent_id)

    async def submit():
        return await asyncio.wait_for(
            module.enqueue_resume_input(
                reply, source=source, message_id="message", command_id="queued"
            ),
            10,
        )

    queued = await submit()
    assert queued.queued is True
    assert (queued.command_id, queued.run_id) == ("queued", "run-1")
    assert queued.control_state == "waiting_for_user"
    with get_session_local()() as db:
        row = db.query(TaskExecutionCommand).filter_by(command_id="queued").one()
        assert (row.status, row.attempt_count, row.defer_count) == ("pending", 0, 0)
        # The snapshot reports the acceptance version, which is the task's.
        assert queued.state_version == row.target_state_version == 1
        assert db.get(Task, reply.task_id).control_state == "waiting_for_user"
        command_id = row.id
    # Same-ID replay converges on the same durable command and stays queued.
    assert (await submit()).queued is True
    with get_session_local()() as db:
        assert (
            db.query(TaskExecutionCommand).filter_by(kind="resume_input").count() == 1
        )
        assert (
            await claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
            is None
        )
    with get_session_local()() as db:
        row = db.get(TaskExecutionCommand, command_id)
        assert (row.status, row.attempt_count, row.defer_count) == ("pending", 0, 0)

    release_bucket(holding)
    prepare = AsyncMock()
    monkeypatch.setattr(
        task_resume,
        "resume_task_reply" if source == "sdk" else "resume_a2a_task",
        prepare,
    )
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    assert command is not None
    await execute_durable_task_command(command)
    prepare.assert_awaited_once()
    # Once executed, the same ID replays the stored outcome instead of queueing.
    accepted = await submit()
    assert accepted.queued is False
    assert (accepted.command_id, accepted.run_id) == ("queued", "run-1")


@pytest.mark.asyncio
async def test_ungoverned_reply_keeps_waiting_for_its_outcome(reply, monkeypatch):
    from unittest.mock import Mock

    from xagent.web.services import task_event_bridge

    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    monkeypatch.setattr(module, "get_task_reply_wait_timeout_seconds", lambda: 0.05)
    with pytest.raises(task_resume.TaskResumeOutcomeUnknownError):
        await module.enqueue_resume_input(
            reply, source="sdk", message_id="", command_id="slow-worker"
        )


@pytest.mark.asyncio
async def test_reply_stopped_by_a_control_while_queued_replays_as_not_accepted(
    reply, monkeypatch, request
):
    from unittest.mock import Mock

    from xagent.web.services import task_event_bridge
    from xagent.web.services import task_execution_admission as admission
    from xagent.web.services.task_command_transport import (
        TaskCommandKind,
        stage_task_command,
    )

    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("tenant:batch", 1, 20)
    )
    request.addfinalizer(lambda: admission.set_task_admission_hook(None))
    saturate_bucket(reply.task_owner_user_id, reply.agent_id)
    queued = await asyncio.wait_for(
        module.enqueue_resume_input(
            reply, source="sdk", message_id="", command_id="stopped"
        ),
        10,
    )
    assert queued.queued is True
    # A cancel control invalidates the waiting reply before it ever ran.
    with get_session_local()() as db:
        cancel = stage_task_command(
            db,
            task_id=reply.task_id,
            actor_user_id=reply.task_owner_user_id,
            command_id="cancel-1",
            kind=TaskCommandKind.CANCEL,
            payload={},
        )
        db.get(Task, reply.task_id).state_version += 1
        db.flush()
        admission.settle_cancelled_admissions(db, cancel.staged_db_id)
        db.commit()
        row = db.query(TaskExecutionCommand).filter_by(command_id="stopped").one()
        assert row.status == "failed"
        assert row.result == {"rejection_reason": "cancelled_before_admission"}
    # Nothing was injected: the stored answer is definite, not unknown.
    with pytest.raises(task_resume.TaskResumeNotAcceptedError):
        await asyncio.wait_for(
            module.enqueue_resume_input(
                reply, source="sdk", message_id="", command_id="stopped"
            ),
            10,
        )


def governed(request, policy=None):
    from xagent.web.services import task_execution_admission as admission

    admission.set_task_admission_hook(
        lambda db, command: policy or admission.AdmissionPolicy("tenant:batch", 1, 20)
    )
    request.addfinalizer(lambda: admission.set_task_admission_hook(None))


@pytest.mark.asyncio
async def test_governed_reply_with_free_capacity_waits_for_its_outcome(
    reply, monkeypatch, request
):
    """An installed policy alone is not a reason to answer queued."""
    from unittest.mock import Mock

    from xagent.web.services import task_event_bridge

    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    monkeypatch.setattr(module, "get_task_reply_wait_timeout_seconds", lambda: 0.05)
    governed(request)
    with pytest.raises(task_resume.TaskResumeOutcomeUnknownError):
        await module.enqueue_resume_input(
            reply, source="sdk", message_id="", command_id="free"
        )
    with get_session_local()() as db:
        row = db.query(TaskExecutionCommand).filter_by(command_id="free").one()
        assert row.status == "pending"


@pytest.mark.asyncio
async def test_governed_a2a_reply_keeps_waiting_while_capacity_is_occupied(
    reply, monkeypatch, request
):
    """A2A has no queued projection, so its reply keeps the outcome wait."""
    from unittest.mock import Mock

    from xagent.web.services import task_event_bridge

    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    monkeypatch.setattr(module, "get_task_reply_wait_timeout_seconds", lambda: 0.05)
    governed(request)
    with get_session_local()() as db:
        db.get(Task, reply.task_id).source = "a2a"
        db.commit()
    saturate_bucket(reply.task_owner_user_id, reply.agent_id)
    with pytest.raises(task_resume.TaskResumeOutcomeUnknownError):
        await module.enqueue_resume_input(
            reply, source="a2a", message_id="message", command_id="a2a-queued"
        )
    with get_session_local()() as db:
        row = db.query(TaskExecutionCommand).filter_by(command_id="a2a-queued").one()
        assert (row.status, row.attempt_count, row.defer_count) == ("pending", 0, 0)


def test_older_unheld_waiter_blocks_a_newer_reply(reply, request):
    from xagent.web.services import task_execution_admission as admission

    governed(request)
    saturate_bucket(reply.task_owner_user_id, reply.agent_id, held=False)
    command_id = module._admit_reply(reply, "sdk", "", "behind")
    with get_session_local()() as db:
        # No slot is occupied; FIFO order alone keeps the reply waiting.
        assert admission.waiting_for_capacity(db, command_id) is True


def test_startup_pacing_blocks_a_reply_until_its_next_start(reply, request):
    from xagent.web.services import task_execution_admission as admission
    from xagent.web.services.task_admission_pacing import (
        StartupPacing,
        reserve_startup,
    )

    governed(
        request,
        admission.AdmissionPolicy("tenant:batch", 1, 20, pacing=StartupPacing(3600, 1)),
    )
    command_id = module._admit_reply(reply, "sdk", "", "paced")
    with get_session_local()() as db:
        assert admission.waiting_for_capacity(db, command_id) is False
        # Another start consumed the only burst; the next one is an hour away.
        assert reserve_startup(db, "tenant:batch")
        db.commit()
        assert admission.waiting_for_capacity(db, command_id) is True


@pytest.mark.asyncio
async def test_unknown_tool_effect_is_terminal_reply_outcome(reply, monkeypatch):
    from types import SimpleNamespace

    from xagent.core.agent.checkpoint import UnknownToolEffectError

    with get_session_local()() as db:
        db.get(Task, reply.task_id).source = "a2a"
        db.commit()
    post = AsyncMock(side_effect=UnknownToolEffectError("unknown external effect"))
    manager = SimpleNamespace(
        get_agent_for_task=AsyncMock(
            return_value=SimpleNamespace(post_user_message=post)
        )
    )
    monkeypatch.setattr(
        task_resume.agent_runtime_service, "get_agent_manager", lambda: manager
    )
    command_id = module._admit_reply(reply, "a2a", "message", "original")
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    await execute_durable_task_command(command)
    assert module._read_reply_outcome(command_id)["outcome"] == "not_resumable"
    assert module._admit_reply(reply, "a2a", "message", "original") == command_id
    post.assert_awaited_once()
