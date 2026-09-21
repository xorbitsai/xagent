"""Shared START transaction and execution-attempt fences."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tests.web.services.coordinator_command_shared import (
    claim_task_command,
    settle_command,
)
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_coordinator_runtime, task_coordinator_service
from xagent.web.services import task_start_consumer as consumer
from xagent.web.services.task_command_execution import execute_durable_task_command
from xagent.web.services.task_command_transport import (
    SettledTaskCommand,
    TaskCommandRejected,
)
from xagent.web.services.task_start_protocol import (
    TaskStartPayload,
    stage_task_start_command,
)


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
    return consumer._commit_handoff(command, owner_lease(command.task_id))


@pytest.fixture
async def accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_coordinator_runtime, "get_runner_id", lambda: "worker-1")
    init_db(db_url=f"sqlite:///{tmp_path / 'consumer.db'}")
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(
            user_id=user.id,
            title="Shared",
            source="sdk",
            status=TaskStatus.PENDING,
            run_id=None,
            state_version=1,
            control_state="idle",
        )
        db.add(task)
        db.flush()
        start = TaskStartPayload(
            version=1,
            run_id="run-1",
            state_version=1,
            turn_id="turn-1",
            kind="create",
            message="Hello",
            execution_message="Hello",
            file_ids=[],
        )
        staged = stage_task_start_command(
            db, task_id=task.id, actor_user_id=user.id, start=start
        )
        db.commit()
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=staged.staged_db_id
        )
        assert command is not None
    yield command
    await task_coordinator_runtime.close_task_coordinators()
    Base.metadata.drop_all(bind=get_engine())


def test_handoff_commits_lease_and_completion_together(accepted):
    handoff = commit_handoff(accepted)
    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        command = db.get(TaskExecutionCommand, accepted.id)
        assert task.run_id == "run-1"
        assert task.state_version == 2
        assert task.runner_id == "worker-1"
        assert task.lease_attempt_id == handoff.claimed.task_lease.attempt_id
        assert command.status == "completed"
        assert command.result["lease_attempt_id"] == task.lease_attempt_id
    with pytest.raises(TaskCommandRejected):
        commit_handoff(accepted)


def test_completion_fence_failure_rolls_back_lease(accepted, monkeypatch):
    monkeypatch.setattr(
        consumer, "finish_task_command_no_commit", lambda *a, **kw: False
    )
    with pytest.raises(TaskCommandRejected):
        commit_handoff(accepted)
    with get_session_local()() as db:
        assert db.get(Task, accepted.task_id).status == TaskStatus.PENDING
        assert db.get(TaskExecutionCommand, accepted.id).status == "processing"


def test_old_command_attempt_cannot_acquire_execution(accepted):
    with get_session_local()() as db:
        db.get(TaskExecutionCommand, accepted.id).attempt_count += 1
        db.commit()
    with pytest.raises(TaskCommandRejected):
        commit_handoff(accepted)
    with get_session_local()() as db:
        assert db.get(Task, accepted.task_id).status == TaskStatus.PENDING
        assert db.get(TaskExecutionCommand, accepted.id).status == "processing"


def test_obsolete_command_expiry_does_not_override_task_owner(accepted):
    with get_session_local()() as db:
        db.get(TaskExecutionCommand, accepted.id).claim_expires_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
        db.commit()
    commit_handoff(accepted)
    with get_session_local()() as db:
        assert db.get(TaskExecutionCommand, accepted.id).status == "completed"


def test_old_run_rejection_does_not_fail_new_run(accepted):
    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        task.run_id = "run-2"
        task.status = TaskStatus.RUNNING
        task.state_version = 2
        db.commit()
    assert isinstance(commit_handoff(accepted), SettledTaskCommand)
    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        assert task.run_id == "run-2"
        assert task.status == TaskStatus.RUNNING
        assert task.state_version == 2
        assert db.get(TaskExecutionCommand, accepted.id).status == "failed"


def test_invalid_current_start_settles_accepted_task(accepted):
    broken = replace(accepted, payload={**accepted.payload, "version": 999})
    assert isinstance(commit_handoff(broken), SettledTaskCommand)
    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        assert task.status == TaskStatus.FAILED
        assert task.control_state == "failed"
        assert task.lease_attempt_id is not None
        assert db.get(TaskExecutionCommand, accepted.id).status == "failed"


def test_stream_reconciliation_reads_current_run_and_output_together(accepted):
    from xagent.web.services.task_stream_snapshot import load_task_stream_snapshots

    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        task.status = TaskStatus.COMPLETED
        task.control_state = "completed"
        task.output = "durable complete answer"
        task.run_id = "run-1"
        db.commit()
    snapshots = load_task_stream_snapshots([accepted.task_id])
    assert len(snapshots) == 1
    assert snapshots[0]["run_id"] == "run-1"
    assert snapshots[0]["output"] == "durable complete answer"
    assert snapshots[0]["status"] == "completed"
    assert load_task_stream_snapshots([]) == []


def test_existing_execution_accepts_without_another_transcript_row(
    accepted, monkeypatch
):
    from unittest.mock import Mock

    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.services import task_event_bridge
    from xagent.web.services.task_existing_command import enqueue_existing_execution

    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        task.status = TaskStatus.COMPLETED
        task.control_state = "completed"
        task.output = "previous answer"
        db.get(TaskExecutionCommand, accepted.id).status = "completed"
        db.commit()
        previous_owner = (task.runner_id, task.lease_attempt_id)
        owner_id = task.user_id
        before = db.query(TaskChatMessage).filter_by(task_id=task.id).count()
    run_id = enqueue_existing_execution(
        task_id=accepted.task_id,
        task_owner_user_id=owner_id,
        actor_user_id=owner_id,
        task_description="description",
        context={"execution_mode": "react"},
    )
    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        assert task.run_id != run_id
        assert task.output == "previous answer"
        assert (task.runner_id, task.lease_attempt_id) == previous_owner
        assert db.query(TaskChatMessage).filter_by(task_id=task.id).count() == before
        command = (
            db.query(TaskExecutionCommand)
            .order_by(TaskExecutionCommand.id.desc())
            .first()
        )
        assert command.payload["kind"] == "existing"
        assert command.payload["existing_context"]["execution_mode"] == "react"


@pytest.mark.asyncio
async def test_cancellation_after_handoff_still_registers_execution(
    accepted, monkeypatch
):
    import asyncio
    import threading
    from unittest.mock import AsyncMock

    committed, release = threading.Event(), threading.Event()
    original = consumer._commit_handoff

    def slow_handoff(command, lease):
        handoff = original(command, lease)
        committed.set()
        assert release.wait(5)
        return handoff

    monkeypatch.setattr(consumer, "_commit_handoff", slow_handoff)
    schedule = AsyncMock()
    monkeypatch.setattr(consumer, "_schedule_committed_turn", schedule)
    pending = asyncio.create_task(execute_durable_task_command(accepted))
    try:
        assert await asyncio.to_thread(committed.wait, 5)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    schedule.assert_awaited_once()
    with get_session_local()() as db:
        assert db.get(TaskExecutionCommand, accepted.id).status == "completed"


def test_shared_acceptance_binds_inputs_and_start_in_one_transaction(
    accepted, monkeypatch
):
    from unittest.mock import Mock

    from cryptography.fernet import Fernet

    from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRef
    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.models.task_runtime_secret import TaskRuntimeSecret
    from xagent.web.services import task_event_bridge
    from xagent.web.services.task_orchestrator import (
        TaskTurnOrchestrator,
        TaskTurnPayload,
    )
    from xagent.web.services.task_runtime_secrets import stage_runtime_values

    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: Mock())
    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        previous_owner = (task.runner_id, task.lease_attempt_id)
        task.status = TaskStatus.COMPLETED
        db.get(TaskExecutionCommand, accepted.id).status = "completed"
        db.commit()
        stage_runtime_values(
            db,
            task_id=task.id,
            turn_id="next-turn",
            values_by_ref={ConnectorRef("mcp", 1): {"secrets": {"key": "synthetic"}}},
        )
        prepared = TaskTurnOrchestrator.claim_append_turn_no_commit(
            db,
            task_id=task.id,
            task_owner_user_id=task.user_id,
            payload=TaskTurnPayload("next", turn_id="next-turn"),
        )
        row = db.get(TaskExecutionCommand, prepared.command_db_id)
        assert row.payload["runtime_values_ref"] == "next-turn"
        assert "synthetic" not in str(row.payload)
        assert db.query(TaskRuntimeSecret).one().run_id == prepared.run_id
        assert (task.runner_id, task.lease_attempt_id) == previous_owner
        db.rollback()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 0
        assert db.query(TaskChatMessage).filter_by(turn_id="next-turn").count() == 0
        assert db.get(TaskExecutionCommand, prepared.command_db_id) is None


@pytest.mark.parametrize("replaced", [False, True])
async def test_terminal_transport_failure_settles_only_its_owned_run(
    accepted, replaced
):
    from xagent.web.services.task_command_transport import (
        MAX_COMMAND_FAILURES,
        fail_task_command,
    )

    with get_session_local()() as db:
        db.get(TaskExecutionCommand, accepted.id).failure_count = (
            MAX_COMMAND_FAILURES - 1
        )
        if replaced:
            db.get(Task, accepted.task_id).run_id = "replacement"
        db.commit()
    assert await settle_command(
        accepted,
        lambda: fail_task_command(
            accepted.id,
            "worker-1",
            "preparation failed",
            expected_attempt_count=accepted.attempt_count,
        ),
    )
    with get_session_local()() as db:
        assert db.get(TaskExecutionCommand, accepted.id).status == "failed"
        assert db.get(Task, accepted.task_id).status == (
            TaskStatus.PENDING if replaced else TaskStatus.FAILED
        )


async def test_replaced_owner_cannot_fail_accepted_run(accepted):
    from xagent.web.services.task_command_transport import fail_task_command

    with get_session_local()() as db:
        db.get(Task, accepted.task_id).lease_attempt_id = "replacement-owner"
        db.commit()
    assert not await settle_command(
        accepted,
        lambda: fail_task_command(
            accepted.id,
            "worker-1",
            "stale owner",
            force_terminal=True,
            expected_attempt_count=accepted.attempt_count,
        ),
    )
    with get_session_local()() as db:
        assert db.get(Task, accepted.task_id).status == TaskStatus.PENDING


async def test_live_owner_routes_controls_even_without_immutable_target(accepted):
    from xagent.web.services.task_command_transport import (
        TaskCommandKind,
        stage_task_command,
    )

    commit_handoff(accepted)
    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        staged = stage_task_command(
            db,
            task_id=task.id,
            actor_user_id=task.user_id,
            command_id="pause",
            kind=TaskCommandKind.PAUSE,
            payload={},
        )
        db.get(TaskExecutionCommand, staged.staged_db_id).target_runner_id = None
        db.commit()
        assert (
            await claim_task_command(
                db, runner_id="worker-2", command_db_id=staged.staged_db_id
            )
            is None
        )
        row = db.get(TaskExecutionCommand, staged.staged_db_id)
        assert row.attempt_count == 0
        assert (
            await claim_task_command(
                db, runner_id="worker-1", command_db_id=staged.staged_db_id
            )
            is not None
        )


async def test_worker_death_after_handoff_recovers_without_replaying_start(accepted):
    from xagent.web.services.task_lease_recovery import (
        recover_expired_task_leases_batch_isolated,
    )

    commit_handoff(accepted)
    with get_session_local()() as db:
        db.get(Task, accepted.task_id).lease_expires_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
        db.commit()
    recovered = recover_expired_task_leases_batch_isolated(
        cutoff=datetime.now(timezone.utc), batch_size=10, after=None
    )
    assert recovered.recovered == 1
    with get_session_local()() as db:
        assert db.get(Task, accepted.task_id).status == TaskStatus.FAILED
        assert db.get(Task, accepted.task_id).runner_id is None
        assert db.get(TaskExecutionCommand, accepted.id).status == "completed"
        assert (
            await claim_task_command(
                db, runner_id="worker-2", command_db_id=accepted.id
            )
            is None
        )


def test_terminal_settlement_deletes_only_accepted_run_values(accepted, monkeypatch):
    from cryptography.fernet import Fernet

    from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRef
    from xagent.web.models.task_runtime_secret import TaskRuntimeSecret
    from xagent.web.services.task_orchestrator import settle_task_lease_isolated
    from xagent.web.services.task_runtime_secrets import (
        bind_runtime_values_to_run,
        stage_runtime_values,
    )

    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    with get_session_local()() as db:
        stage_runtime_values(
            db,
            task_id=accepted.task_id,
            turn_id="turn-1",
            values_by_ref={ConnectorRef("mcp", 1): {"secrets": {"key": "synthetic"}}},
        )
        bind_runtime_values_to_run(
            db, task_id=accepted.task_id, turn_id="turn-1", run_id="run-1"
        )
        db.commit()
    handoff = commit_handoff(accepted)
    assert settle_task_lease_isolated(
        handoff.claimed.task_lease, error_message="execution failed"
    )
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 0


@pytest.mark.asyncio
async def test_following_command_waits_for_start_registration(accepted, monkeypatch):
    import asyncio
    import threading
    from unittest.mock import AsyncMock

    from xagent.web.services import task_command_execution
    from xagent.web.services.task_command_transport import (
        TaskCommandKind,
        stage_task_command,
    )

    committed, release = threading.Event(), threading.Event()
    original = consumer._commit_handoff
    registered = False

    def slow_handoff(command, lease):
        handoff = original(command, lease)
        committed.set()
        assert release.wait(5)
        return handoff

    async def schedule(**kwargs):
        nonlocal registered
        registered = True

    async def followup(command):
        assert registered
        with get_session_local()() as db:
            task = db.get(Task, accepted.task_id)
            assert task.run_id == accepted.target_run_id
            assert task.runner_id == "worker-1"
        return {}

    monkeypatch.setattr(consumer, "_commit_handoff", slow_handoff)
    monkeypatch.setattr(consumer, "_schedule_committed_turn", schedule)
    effects = AsyncMock(side_effect=followup)
    monkeypatch.setattr(
        task_command_execution, "_execute_and_report_task_command", effects
    )
    start = asyncio.create_task(
        task_command_execution.execute_durable_task_command(accepted)
    )
    next_task = None
    try:
        assert await asyncio.to_thread(committed.wait, 5)
        with get_session_local()() as db:
            task = db.get(Task, accepted.task_id)
            staged = stage_task_command(
                db,
                task_id=task.id,
                actor_user_id=task.user_id,
                command_id="followup",
                kind=TaskCommandKind.MESSAGE,
                payload={"message": "next"},
            )
            db.commit()
            command = await claim_task_command(
                db, runner_id="worker-1", command_db_id=staged.staged_db_id
            )
            assert command is not None
        next_task = asyncio.create_task(
            task_command_execution.execute_durable_task_command(command)
        )
        await asyncio.sleep(0.05)
        effects.assert_not_awaited()
        release.set()
        await asyncio.wait_for(asyncio.gather(start, next_task), 5)
        effects.assert_awaited_once()
    finally:
        release.set()
        await asyncio.gather(
            start, *([next_task] if next_task else []), return_exceptions=True
        )


def test_completion_wait_recovers_owner_lost_after_real_result_finalization(accepted):
    from xagent.web.services.task_completion import _is_run_finished
    from xagent.web.services.task_execution import (
        _finalize_task_execution_result_isolated,
        _PreparedTaskFileOutputs,
    )

    handoff = commit_handoff(accepted)
    result = _finalize_task_execution_result_isolated(
        task_id=accepted.task_id,
        task_user_id=handoff.task_owner_user_id,
        pre_run_status=TaskStatus.RUNNING,
        result={"success": True, "output": "finished"},
        expected_run_id="run-1",
        task_lease=handoff.claimed.task_lease,
        resolved_scope_segments=(),
        prepared_outputs=_PreparedTaskFileOutputs((), (), ()),
    )
    assert result.terminal_state_committed
    assert not _is_run_finished(accepted.task_id, "run-1")
    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.runner_id == "worker-1"
        task.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        db.commit()
    assert not _is_run_finished(accepted.task_id, "run-1")
    from xagent.web.services.task_lease_recovery import recover_expired_idle_task_leases

    assert recover_expired_idle_task_leases(batch_size=10) == 1
    assert _is_run_finished(accepted.task_id, "run-1")
    with get_session_local()() as db:
        task = db.get(Task, accepted.task_id)
        assert task.runner_id is None
        assert task.output == "finished"
        assert task.status == TaskStatus.COMPLETED
