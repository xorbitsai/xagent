"""Reply acceptance, exact handoff, and cross-process preparation results."""

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from xagent.web.models.agent import Agent
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_resume
from xagent.web.services import task_resume_command as module
from xagent.web.services.task_command_transport import (
    TaskCommandRejected,
    claim_task_command,
)
from xagent.web.services.task_resume import TaskReplyInput, TaskResumeBusyError


@pytest.fixture
def reply(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(module, "get_runner_id", lambda: "worker-1")
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
            run_id="run-1",
            status=TaskStatus.WAITING_FOR_USER,
            text="answer",
        )
    yield ctx
    Base.metadata.drop_all(bind=get_engine())


def claim(ctx):
    command_id = module._admit_reply(ctx, "sdk", "", "reply-1")
    with get_session_local()() as db:
        return claim_task_command(db, runner_id="worker-1", command_db_id=command_id)


def test_reply_is_durable_without_acquiring_a_request_lease(reply):
    command_id = module._admit_reply(reply, "sdk", "", "reply-1")
    with get_session_local()() as db:
        task = db.get(Task, reply.task_id)
        command = db.get(TaskExecutionCommand, command_id)
        assert task.runner_id is None
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.control_state == "resume_requested"
        assert command.payload["text"] == "answer"
    with pytest.raises(TaskResumeBusyError):
        module._admit_reply(reply, "sdk", "", "reply-2")


def test_reply_handoff_and_command_completion_are_one_transaction(reply, monkeypatch):
    command = claim(reply)
    monkeypatch.setattr(
        module, "finish_task_command_no_commit", lambda *args, **kwargs: False
    )
    with pytest.raises(TaskCommandRejected):
        module._handoff(command)
    with get_session_local()() as db:
        task = db.get(Task, reply.task_id)
        assert task.runner_id is None
        assert task.control_state == "resume_requested"
        assert db.get(TaskExecutionCommand, command.id).status == "processing"


def test_old_reply_claim_cannot_acquire_or_restore_a_new_attempt(reply):
    command = claim(reply)
    old = replace(command, attempt_count=command.attempt_count - 1)
    with pytest.raises(TaskCommandRejected):
        module._handoff(old)
    from xagent.web.services.task_command_transport import fail_task_command

    assert not fail_task_command(
        old.id,
        "worker-1",
        "stale",
        force_terminal=True,
        expected_attempt_count=old.attempt_count,
    )
    with get_session_local()() as db:
        assert db.get(Task, reply.task_id).control_state == "resume_requested"
    payload, lease, owner, state = module._handoff(command)
    assert lease.run_id == reply.run_id
    assert state["lease_attempt_id"] == lease.attempt_id
    assert module._read_reply_outcome(command.id) is None
    module._record_outcome(command, state, "accepted")
    assert module._read_reply_outcome(command.id)["outcome"] == "accepted"


@pytest.mark.asyncio
async def test_worker_uses_preacquired_lease_and_stable_turn_identity(
    reply, monkeypatch
):
    command = claim(reply)
    prepare = AsyncMock()
    monkeypatch.setattr(task_resume, "resume_task_reply", prepare)
    await module.execute_resume_input(command)
    assert prepare.await_count == 1
    assert prepare.call_args.kwargs["turn_id"] == command.command_id
    assert prepare.call_args.kwargs["preacquired_lease"].run_id == "run-1"
    assert module._read_reply_outcome(command.id)["outcome"] == "accepted"


@pytest.mark.parametrize("replaced", [False, True])
def test_terminal_reply_failure_restores_only_its_admission(reply, replaced):
    from xagent.web.services.task_command_transport import (
        MAX_COMMAND_FAILURES,
        fail_task_command,
    )

    command = claim(reply)
    with get_session_local()() as db:
        db.get(TaskExecutionCommand, command.id).failure_count = (
            MAX_COMMAND_FAILURES - 1
        )
        if replaced:
            db.get(Task, reply.task_id).state_version += 1
        db.commit()
    assert fail_task_command(
        command.id,
        "worker-1",
        "unavailable",
        expected_attempt_count=command.attempt_count,
    )
    with get_session_local()() as db:
        task = db.get(Task, reply.task_id)
        assert task.runner_id is None
        assert task.control_state == (
            "resume_requested" if replaced else "waiting_for_user"
        )
    assert module._read_reply_outcome(command.id)["outcome"] == "unavailable"


@pytest.mark.parametrize(
    "invalid",
    [{"version": True}, {"agent_id": 0}, {"run_id": ""}, {"prior_status": "paused"}],
)
def test_invalid_reply_payload_cannot_acquire_lease(reply, invalid):
    command = claim(reply)
    command = replace(command, payload={**command.payload, **invalid})
    with pytest.raises(TaskCommandRejected):
        module._handoff(command)
    with get_session_local()() as db:
        assert db.get(Task, reply.task_id).runner_id is None


@pytest.mark.parametrize(
    "source,status",
    [
        ("sdk", TaskStatus.WAITING_FOR_USER),
        ("a2a", TaskStatus.WAITING_FOR_USER),
        ("a2a", TaskStatus.PAUSED),
    ],
)
def test_reply_waits_for_previous_execution_to_release_lease(reply, source, status):
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
    with pytest.raises(TaskResumeBusyError):
        module._admit_reply(ctx, source, "message", "reply")
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 0
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
        command = claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
    _, lease, _, _ = module._handoff(command)
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
        first = claim_task_command(db, runner_id="worker-1", command_db_id=original_id)
        original_version = db.get(
            TaskExecutionCommand, original_id
        ).target_state_version
    await module.execute_resume_input(first)
    assert module._read_reply_outcome(original_id)["outcome"] == "retryable_unavailable"
    with get_session_local()() as db:
        task = db.get(Task, reply.task_id)
        assert task.runner_id is None
        assert task.control_state == "waiting_for_user"
    retry_id = module._admit_reply(reply, "a2a", "message-1", "original-command")
    assert retry_id != original_id
    assert (
        module._admit_reply(reply, "a2a", "message-1", "original-command") == retry_id
    )
    with get_session_local()() as db:
        second = claim_task_command(db, runner_id="worker-1", command_db_id=retry_id)
    await module.execute_resume_input(second)
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
def test_a2a_does_not_retry_unknown_or_terminal_outcome(reply, outcome):
    with get_session_local()() as db:
        db.get(Task, reply.task_id).source = "a2a"
        db.commit()
    command_id = module._admit_reply(reply, "a2a", "message", "original")
    with get_session_local()() as db:
        command = claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
    _, lease, _, state = module._handoff(command)
    assert task_resume._restore_a2a_resume_prelease_sync(
        lease, status=TaskStatus.WAITING_FOR_USER
    )
    module._record_outcome(command, state, outcome)
    assert module._admit_reply(reply, "a2a", "message", "original") == command_id
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 1
        assert (
            claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
            is None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["after_injection", "lease_restore"])
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
    if failure_stage == "after_injection":
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
        command = claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
    await module.execute_resume_input(command)
    assert module._read_reply_outcome(command_id)["outcome"] == "unavailable"
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
def test_reply_reclaims_expired_resting_lease_atomically(
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
    assert not settle_task_lease_isolated(old_lease)
    with get_session_local()() as db:
        assert refresh_task_lease(db, old_lease) == TaskLeaseRefreshState.LOST
        task = db.get(Task, ctx.task_id)
        assert task.runner_id is None
        assert task.lease_attempt_id is None
        assert task.lease_expires_at is None
        assert task.control_state == "resume_requested"
        assert task.state_version == original_version + 1
        command = claim_task_command(db, runner_id="worker-1", command_db_id=command_id)
    _, lease, _, state = module._handoff(command)
    assert lease.run_id == ctx.run_id
    assert lease.attempt_id != old_lease.attempt_id
    assert not settle_task_lease_isolated(old_lease)
    with get_session_local()() as db:
        assert refresh_task_lease(db, old_lease) == TaskLeaseRefreshState.LOST
        task = db.get(Task, ctx.task_id)
        assert task.lease_attempt_id == lease.attempt_id
        assert task.state_version == state["state_version"]
        assert task.control_state == "running"
