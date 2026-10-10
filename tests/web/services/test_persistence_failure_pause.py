"""A run interrupted by a persistence failure rests PAUSED instead of FAILED.

Covers every settlement point that acts on the interruption: the exception
path of a new run (``_schedule_bg`` -> ``settle_task_lease_isolated``), its
result path (``_finalize_task_execution_result_isolated``), and both paths of
a resumed run (``_finalize_resumed_task`` and the resume exception branch).
Runs on SQLite and, when ``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL
through the shared ``canonical`` fixture.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError

from tests.core.agent.test_react import FakeLLM, FakeTool
from tests.core.agent.test_runner import FakeWorkspaceManager
from tests.web.services.test_execution_event_persistence_settlement import (
    _clear_bg_manager as _clear_bg_manager_fixture,
)
from tests.web.services.test_execution_event_persistence_settlement import (
    _fail_once,
    _react_execution,
    _run_scheduled_turn,
    _tool_turn_llm,
)
from tests.web.services.test_execution_event_recovery import tracer_for
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    facts,
)
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from tests.web.services.test_task_lease_expiry_interruption import (
    _fail_recording,
    _snapshot,
)
from xagent.core.agent import (
    Agent,
    AgentRunner,
    ExecutionContext,
    PatternRuntime,
    ReActPattern,
)
from xagent.core.agent.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_TYPE,
    CheckpointUnavailableError,
    ExecutionEventPersistenceError,
    TraceCheckpointStore,
)
from xagent.core.agent.interruption import InterruptionReason
from xagent.web.models.agent import Agent as AgentModel
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services import task_auto_recovery
from xagent.web.services import task_execution_event_recovery as event_recovery
from xagent.web.services import task_orchestrator as orchestrator
from xagent.web.services.chat_history_service import (
    DELIVERY_PENDING,
    persist_user_message_no_commit,
)
from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE
from xagent.web.services.task_auto_recovery import (
    TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR,
    InterruptionSettlementDeferred,
    settlement_interruption_for_failure,
    settlement_interruption_for_result,
)
from xagent.web.services.task_execution import (
    _acquire_resume_task_lease,
    _finalize_resumed_task,
    _finalize_task_execution_result_isolated,
    _PreparedTaskFileOutputs,
    execute_resume_background,
)
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.services.task_lease_recovery import (
    TASK_LEASE_PAUSED_TRIGGER_ERROR,
    recover_task_lease_candidate_no_commit,
)
from xagent.web.services.task_lease_service import (
    TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR,
    CheckpointRecoveryResolution,
    CheckpointRecoveryVerdict,
    TaskLease,
    acquire_task_lease,
    bind_task_lease_context,
    get_expired_task_lease_candidates,
    utc_now,
)
from xagent.web.services.task_orchestrator import (
    TASK_INTERRUPTION_PAUSED_MESSAGE,
    TASK_USER_PAUSED_MESSAGE,
    TaskTurnPayload,
    _schedule_bg,
    publish_settlement_pause,
    settle_task_lease_isolated,
)
from xagent.web.services.task_settlement_report import SettlementReport

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
_clear_bg_manager = _clear_bg_manager_fixture

PAUSE_SWITCH = "XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED"
RUN_ERROR = "setup/run error: ExecutionEventPersistenceError: event write failed"
NO_OUTPUTS = _PreparedTaskFileOutputs((), (), ())


@pytest.fixture(autouse=True)
def _pause_switch_on(monkeypatch):
    # Off by default until automatic resume ships; these tests exercise it on.
    monkeypatch.setenv(PAUSE_SWITCH, "true")


@pytest.fixture(autouse=True)
def _late_bound_sessions(monkeypatch):
    """``task_execution`` binds the session factory at import; follow the
    one the ``canonical`` fixture installs instead."""
    from xagent.web.models import database

    monkeypatch.setattr(
        "xagent.web.services.task_execution.get_session_local",
        lambda: database.get_session_local(),
    )


def _start_run(factory, tid: int, **fields: Any) -> TaskLease:
    with factory() as db:
        task = db.get(Task, tid)
        for name, value in fields.items():
            setattr(task, name, value)
        db.commit()
        lease = acquire_task_lease(db, tid, new_run=True)
    assert lease is not None
    return lease


def _checkpoint(factory, tid: int, run_id: str, *, messages: int = 2) -> None:
    """A recoverable V2 checkpoint of ``run_id``."""
    with factory() as db:
        append_fact_no_commit(
            db,
            task_id=tid,
            kind="recovery_state",
            key=f"runtime:{uuid.uuid4()}",
            payload={
                "data": {
                    "checkpoint_type": CHECKPOINT_TYPE,
                    "snapshot_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "execution_id": str(tid),
                    "snapshot": _snapshot(messages=messages, iteration=2),
                }
            },
            run_id=run_id,
        )
        db.commit()


def _state(factory, tid: int) -> tuple[Task, Any, list[Any]]:
    with factory() as db:
        task = db.get(Task, tid)
        row = db.get(TaskAutoRecovery, tid)
        events = list(
            db.scalars(
                sa.select(TaskRecoveryEvent)
                .where(TaskRecoveryEvent.task_id == tid)
                .order_by(TaskRecoveryEvent.id)
            )
        )
        db.expunge_all()
    return task, row, events


def _assistant_lines(factory, tid: int) -> list[str]:
    with factory() as db:
        return [
            row.content
            for row in db.query(TaskChatMessage)
            .filter(TaskChatMessage.task_id == tid, TaskChatMessage.role == "assistant")
            .order_by(TaskChatMessage.id)
        ]


def _settled_facts(factory, tid: int) -> list[dict[str, Any]]:
    with factory() as db:
        return [
            row.payload for row in facts(db, tid) if row.kind == "execution_settled"
        ]


def _assert_no_failure_in_model_context(factory, tid: int) -> None:
    """The paused run must not read as a failed execution to the next turn."""
    from xagent.web.services.task_event_context_service import (
        load_task_event_context,
    )

    with factory() as db:
        context = load_task_event_context(db, tid)
    # summarize_execution_failure_event's line for a failed settlement.
    assert "previous execution failed" not in str(context).lower()


def _assert_paused(factory, tid: int, lease: TaskLease, *, reason: str) -> Any:
    task, row, events = _state(factory, tid)
    assert task.status == TaskStatus.PAUSED
    assert task.control_state == "paused"
    assert task.run_id == lease.run_id
    assert task.runner_id is None
    assert task.error_message is None
    assert row.run_id == lease.run_id
    assert (row.reason, row.state) == (reason, "manual")
    assert row.paused_state_version == task.state_version
    assert row.progress_marker is not None
    assert row.next_attempt_at is None
    assert [(e.event, e.reason, e.run_id) for e in events] == [
        ("interrupted", reason, lease.run_id)
    ]
    assert events[0].detail["task_status"] == "paused"
    return row


def _settle(lease: TaskLease, **kwargs: Any) -> tuple[bool, SettlementReport]:
    report = SettlementReport()
    settled = settle_task_lease_isolated(
        lease,
        error_message=RUN_ERROR,
        classify_unknown_tool_effect=True,
        report=report,
        **kwargs,
    )
    return settled, report


PAUSED_FACT = {"status": "paused", "result": {"error": None}}


def _persistence_interruption() -> InterruptionReason:
    reason = settlement_interruption_for_failure(
        ExecutionEventPersistenceError("event write failed")
    )
    assert reason is InterruptionReason.PERSISTENCE_FAILURE
    return reason


# ---------------------------------------------------------------- classifiers


def test_settlement_acts_on_persistence_failures():
    assert settlement_interruption_for_failure(
        OperationalError("SELECT 1", {}, Exception("connection reset"))
    ) is (InterruptionReason.PERSISTENCE_FAILURE)
    assert settlement_interruption_for_failure(RuntimeError("boom")) is None
    persistence = {"success": False, "interruption_reason": "persistence_failure"}
    assert (
        settlement_interruption_for_result(persistence)
        is InterruptionReason.PERSISTENCE_FAILURE
    )
    # Provider and model-output reasons: test_provider_failure_pause.py.
    assert (
        settlement_interruption_for_result(
            {"success": False, "interruption_reason": "lease_expired"}
        )
        is None
    )
    # A quota stop keeps the runner's reason but wins over it.
    assert (
        settlement_interruption_for_result({**persistence, "status": "quota_exceeded"})
        is None
    )
    # Only the top-level field counts, never the raw pattern result.
    assert (
        settlement_interruption_for_result(
            {"success": False, "agent_result": persistence}
        )
        is None
    )
    assert settlement_interruption_for_result({**persistence, "success": True}) is None


# ------------------------------------------------- S2: new run, exception path


@pytest.mark.asyncio
async def test_checkpoint_write_failure_pauses_and_the_run_resumes(
    canonical, monkeypatch, tmp_path
):
    """The checkpoint after a tool fails; settlement (DB fine again) pauses.

    The paused task is resumable: a resume lease claims the same run and the
    run continues from its checkpoint without re-running the tool.
    """

    factory, tid = canonical
    lease = _start_run(factory, tid)
    tool = FakeTool()

    async def execute(**kwargs):
        with bind_task_lease_context(kwargs["task_lease"]):
            runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer_for(tid)))
            original = runtime.checkpoint

            async def fail_after_tool(label, **checkpoint_kwargs):
                if label.endswith(("after_tool", "after_tool_batch")):
                    raise ExecutionEventPersistenceError("checkpoint write failed")
                return await original(label, **checkpoint_kwargs)

            monkeypatch.setattr(runtime, "checkpoint", fail_after_tool)
            context = ExecutionContext(execution_id=str(tid), system_prompt="Calc")
            context.add_user_message("2+2", metadata={"turn_id": "turn"})
            await ReActPattern(max_iterations=3).run(
                context=context, tools=[tool], runtime=runtime, llm=_tool_turn_llm()
            )

    events: list[dict[str, Any]] = []

    async def publish(event, _task_id):
        events.append(event)

    with patch("xagent.web.services.task_events.publish_task_event", new=publish):
        results = await _run_scheduled_turn(tid, lease, execute)

    assert len(tool.calls) == 1
    assert results == []
    _assert_paused(factory, tid, lease, reason="persistence_failure")
    _task_after, row, _events = _state(factory, tid)
    assert "checkpoint write failed" in row.last_error
    assert [event["type"] for event in events] == ["task_paused"]
    assert events[0]["message"] == TASK_INTERRUPTION_PAUSED_MESSAGE
    assert (events[0]["status"], events[0]["control_state"]) == ("paused", "paused")
    assert events[0]["run_id"] == lease.run_id
    assert "checkpoint write failed" not in str(events)
    assert CLIENT_SAFE_TASK_FAILURE not in _assistant_lines(factory, tid)
    assert _settled_facts(factory, tid) == [
        {"status": "paused", "result": {"error": None}}
    ]

    # Manual resume: same run, continues from the checkpoint.
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)
    resume_lease = _acquire_resume_task_lease(
        tid, user_id, lease.run_id, refuse_terminal_status=True
    )
    assert resume_lease is not None and resume_lease.run_id == lease.run_id
    resume_llm = FakeLLM(responses=[{"content": "4", "done": True}])
    with bind_task_lease_context(resume_lease):
        runner = AgentRunner(
            agent=Agent(
                name="resumed",
                patterns=[ReActPattern(max_iterations=3)],
                tools=[tool],
                llm=resume_llm,
            ),
            tracer=TraceCheckpointStore(tracer_for(tid)),
            workspace_manager=FakeWorkspaceManager(tmp_path),
        )
        result = await runner.resume(str(tid), task="2+2")
    assert result["success"]
    assert result["output"] == "4"
    assert len(tool.calls) == 1


@pytest.mark.asyncio
async def test_unknown_tool_effect_still_fails_and_records_why(canonical, monkeypatch):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _fail_once(monkeypatch, "tool_execution_end")
    tool = FakeTool()

    published = await _run_scheduled_turn(tid, lease, _react_execution(tid, tool))

    assert len(tool.calls) == 1
    task, row, events = _state(factory, tid)
    assert task.status == TaskStatus.FAILED
    assert task.error_message == TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
    assert [event["type"] for event in published] == ["task_error"]
    assert (row.reason, row.state) == ("unknown_tool_effect", "manual")
    assert [(e.event, e.reason) for e in events] == [
        ("interrupted", "unknown_tool_effect")
    ]
    assert events[0].detail["task_status"] == "failed"


def test_run_without_checkpoint_fails_as_before(canonical):
    factory, tid = canonical
    lease = _start_run(factory, tid)

    settled, report = _settle(lease, interruption=_persistence_interruption())

    assert settled and not report.paused
    task, row, _events = _state(factory, tid)
    assert task.status == TaskStatus.FAILED
    # The fail branch reports the committed FAILED control identity.
    assert report.control_state["status"] == "failed"
    assert report.control_state["run_id"] == lease.run_id
    assert report.control_state["state_version"] == task.state_version
    assert task.error_message == RUN_ERROR
    assert (row.reason, row.state) == ("not_recoverable", "manual")


def test_report_stays_empty_when_the_lease_no_longer_owns_the_run(canonical):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    with factory() as db:
        task = db.get(Task, tid)
        task.control_state = "pause_requested"
        task.run_id = "replacement-run"
        db.commit()

    settled, report = _settle(lease, interruption=_persistence_interruption())

    assert not settled
    assert report == SettlementReport()


def test_report_stays_empty_when_the_fenced_pause_write_misses(canonical, monkeypatch):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    with factory() as db:
        db.get(Task, tid).control_state = "pause_requested"
        db.commit()
    # The decision is to pause, but the fenced PAUSED write matches no row.
    monkeypatch.setattr(
        orchestrator,
        "pause_and_release_task_lease_no_commit",
        lambda _db, _lease: False,
    )

    _settled, report = _settle(lease, interruption=_persistence_interruption())

    assert report == SettlementReport()


def test_report_stays_empty_when_the_pause_commit_fails(canonical, monkeypatch):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    with factory() as db:
        db.get(Task, tid).control_state = "pause_requested"
        db.commit()
    report = SettlementReport()

    def failing_commit(self):
        raise RuntimeError("commit failed")

    monkeypatch.setattr("sqlalchemy.orm.Session.commit", failing_commit)
    with pytest.raises(RuntimeError, match="commit failed"):
        settle_task_lease_isolated(
            lease,
            error_message=RUN_ERROR,
            classify_unknown_tool_effect=True,
            interruption=_persistence_interruption(),
            report=report,
        )

    assert report == SettlementReport()


def test_pause_requested_run_pauses_for_the_user(canonical):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    with factory() as db:
        db.get(Task, tid).control_state = "pause_requested"
        db.commit()

    settled, report = _settle(lease, interruption=_persistence_interruption())

    assert settled and report.paused_for is InterruptionReason.USER_PAUSE
    _assert_paused(factory, tid, lease, reason="user_pause")


def test_failed_settlement_keeps_the_lease_for_ttl_recovery(canonical, monkeypatch):
    """Settlement itself fails (the database is still down): unchanged path.

    The lease is retained, TTL recovery later pauses the run as
    ``lease_expired`` -- the reason it can still persist.
    """

    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)

    def database_down(*_args, **_kwargs):
        raise OperationalError("UPDATE tasks", {}, Exception("connection refused"))

    monkeypatch.setattr(
        orchestrator, "pause_and_release_task_lease_no_commit", database_down
    )
    with pytest.raises(OperationalError):
        _settle(lease, interruption=_persistence_interruption())

    task, row, events = _state(factory, tid)
    assert task.status == TaskStatus.RUNNING
    assert (task.runner_id, task.run_id) == (lease.runner_id, lease.run_id)
    assert row is None and events == []

    with factory() as db:
        db.get(Task, tid).lease_expires_at = utc_now() - timedelta(seconds=5)
        db.commit()
    with factory() as db:
        candidate = get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=1)[0]
        assert (
            recover_task_lease_candidate_no_commit(
                db, candidate, recovered_at=utc_now()
            )
            == TaskStatus.PAUSED
        )
        db.commit()
    _assert_paused(factory, tid, lease, reason="lease_expired")


def test_unresolvable_checkpoint_keeps_the_lease(canonical, monkeypatch):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)

    def unavailable(*_args, **_kwargs):
        raise CheckpointUnavailableError("recovery read failed")

    monkeypatch.setattr(event_recovery, "read_event_checkpoint", unavailable)
    with pytest.raises(InterruptionSettlementDeferred):
        _settle(lease, interruption=_persistence_interruption())

    task, row, _events = _state(factory, tid)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == lease.runner_id
    assert row is None


@pytest.mark.asyncio
async def test_deferred_result_settlement_retains_the_lease(canonical):
    """A result path that cannot resolve the checkpoint leaves it to TTL."""

    factory, tid = canonical
    lease = _start_run(factory, tid)

    async def execute(**_kwargs):
        raise InterruptionSettlementDeferred("checkpoint not resolvable")

    published = await _run_scheduled_turn(tid, lease, execute)

    assert published == []
    task, row, _events = _state(factory, tid)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == lease.runner_id
    assert row is None


# --------------------------------------------------- S1: new run, result path


def _finalize(factory, tid: int, lease: TaskLease, result: dict[str, Any]):
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)
    return _finalize_task_execution_result_isolated(
        task_id=tid,
        task_user_id=user_id,
        pre_run_status=TaskStatus.RUNNING,
        result=result,
        expected_run_id=lease.run_id,
        task_lease=lease,
        resolved_scope_segments=(),
        prepared_outputs=NO_OUTPUTS,
    )


PERSISTENCE_RESULT = {
    "success": False,
    "status": "failed",
    "output": "All patterns failed",
    "error": "All 1 patterns failed: OperationalError: connection reset",
    "interruption_reason": "persistence_failure",
}


def test_result_persistence_failure_pauses(canonical):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)

    finalized = _finalize(factory, tid, lease, PERSISTENCE_RESULT)

    assert finalized.report.paused_for is InterruptionReason.PERSISTENCE_FAILURE
    assert finalized.waiting_for_control and finalized.terminal_state_committed
    assert finalized.final_task_status == "paused"
    assert finalized.report.control_state["control_state"] == "paused"
    # The scheduler's settlement releases the paused lease.
    assert settle_task_lease_isolated(lease)
    row = _assert_paused(factory, tid, lease, reason="persistence_failure")
    assert row.last_error == PERSISTENCE_RESULT["error"]
    # Settled like lease recovery, not as the run's unsuccessful result.
    assert _settled_facts(factory, tid) == [PAUSED_FACT]
    _assert_no_failure_in_model_context(factory, tid)
    assert _assistant_lines(factory, tid) == []


@pytest.mark.parametrize(
    "result",
    [
        {**PERSISTENCE_RESULT, "status": "quota_exceeded", "error_code": "quota"},
        {**PERSISTENCE_RESULT, "interruption_reason": "lease_expired"},
        {k: v for k, v in PERSISTENCE_RESULT.items() if k != "interruption_reason"},
    ],
    ids=["quota_exceeded", "lease_expired", "no_reason"],
)
def test_other_results_fail_as_before(canonical, result):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)

    finalized = _finalize(factory, tid, lease, result)

    assert finalized.report.paused_for is None
    assert finalized.final_task_status == "failed"
    task, row, events = _state(factory, tid)
    assert task.status == TaskStatus.FAILED
    assert task.error_message == result["error"]
    assert row is None and events == []


def test_result_unresolvable_checkpoint_raises_without_writing(canonical, monkeypatch):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)

    def unavailable(*_args, **_kwargs):
        raise CheckpointUnavailableError("recovery read failed")

    monkeypatch.setattr(event_recovery, "read_event_checkpoint", unavailable)
    with pytest.raises(InterruptionSettlementDeferred):
        _finalize(factory, tid, lease, PERSISTENCE_RESULT)

    task, row, _events = _state(factory, tid)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == lease.runner_id
    assert row is None


# ------------------------------------------------------- S3: resumed run


def test_resumed_result_persistence_failure_pauses(canonical):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)

    finalized = _finalize_resumed_task(
        tid,
        status="failed",
        success=False,
        output=PERSISTENCE_RESULT["output"],
        task_owner_user_id=user_id,
        result=dict(PERSISTENCE_RESULT),
        task_lease=lease,
        prepared_outputs=NO_OUTPUTS,
    )

    assert finalized["final_status"] == "paused"
    assert finalized["lease_released"]
    assert finalized["report"].paused_for is InterruptionReason.PERSISTENCE_FAILURE
    _assert_paused(factory, tid, lease, reason="persistence_failure")
    assert _assistant_lines(factory, tid) == []
    assert _settled_facts(factory, tid) == [PAUSED_FACT]
    _assert_no_failure_in_model_context(factory, tid)


class _FakeTracker:
    quota_interrupt_reason = None

    def __init__(self, *, task_id: int, **_kwargs: Any) -> None:
        self.complete_tracking = AsyncMock()
        self.stop_periodic_updates = AsyncMock()

    async def start_tracking(self) -> None:
        return None

    async def interrupt_reason_for_quota(self) -> None:
        return None


async def _resume_raising(
    error: BaseException, settle: Any, *, explicit_cancel: bool = False
) -> tuple[AsyncMock, list[str]]:
    """Drive the resume entry point into its exception branch.

    Returns the interruption-pause publisher and the broadcast frame types.
    """

    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    heartbeat_task: asyncio.Task[Any] = asyncio.create_task(asyncio.sleep(0))
    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(side_effect=error)
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())
    publish_pause = AsyncMock()
    patches = [
        patch("xagent.web.services.task_execution._settle_resumed_task_lease", settle),
        patch(
            "xagent.web.services.task_execution.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_execution.background_task_manager.promote_resume_task"
        ),
        patch("xagent.web.tracking.task_tracker.TaskTracker", _FakeTracker),
        patch(
            "xagent.web.services.task_orchestrator.publish_settlement_pause",
            publish_pause,
        ),
        patch(
            "xagent.web.services.task_execution.background_task_manager"
            ".cancel_was_requested",
            return_value=explicit_cancel,
        ),
    ]
    for item in patches:
        item.start()
    try:
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            expected_run_id="run-a",
            resolved_execution_scope=None,
            preacquired_lease=lease,
            preacquired_heartbeat_stop=asyncio.Event(),
            preacquired_heartbeat_task=heartbeat_task,
            preacquired_prior_status=TaskStatus.PAUSED,
        )
    finally:
        for item in reversed(patches):
            item.stop()
    return publish_pause, [
        call.args[0].get("type") for call in ws_manager.broadcast_to_task.call_args_list
    ]


@pytest.mark.asyncio
async def test_resume_exception_passes_the_interruption_and_announces_the_pause():
    def settle(lease, **kwargs):
        assert kwargs["interruption"] is InterruptionReason.PERSISTENCE_FAILURE
        report = kwargs["report"]
        report.paused_for = InterruptionReason.PERSISTENCE_FAILURE
        report.control_state.update(status="paused", run_id=lease.run_id)
        return True

    settle_mock = MagicMock(side_effect=settle)
    publish_pause, frames = await _resume_raising(
        ExecutionEventPersistenceError("event write failed"), settle_mock
    )

    settle_mock.assert_called_once()
    publish_pause.assert_awaited_once_with(
        42,
        SettlementReport(
            control_state={"status": "paused", "run_id": "run-a"},
            paused_for=InterruptionReason.PERSISTENCE_FAILURE,
        ),
    )
    assert "task_error" not in frames


@pytest.mark.asyncio
async def test_resume_exception_without_interruption_settles_unchanged():
    settle_mock = MagicMock(return_value=True)
    publish_pause, frames = await _resume_raising(
        RuntimeError("tool crashed"), settle_mock
    )

    settle_mock.assert_called_once()
    assert settle_mock.call_args.kwargs["interruption"] is None
    publish_pause.assert_not_awaited()
    assert "task_error" in frames


@pytest.mark.asyncio
async def test_resume_deferred_settlement_retains_the_lease():
    settle_mock = MagicMock(return_value=True)
    publish_pause, frames = await _resume_raising(
        InterruptionSettlementDeferred("checkpoint not resolvable"), settle_mock
    )

    settle_mock.assert_not_called()
    publish_pause.assert_not_awaited()
    assert "task_error" not in frames


# ------------------------------------------- projections, compared differentially
#
# Each case settles two equivalent tasks in one database and compares every
# lifecycle projection: the task row, its lease fields, the settled fact and
# transcript, the TriggerRun, the WorkforceRun and the user's delivery row.


def _workforce_run(db, user: User) -> int:
    """A running, non-preview workforce run (names unique per call)."""
    suffix = uuid.uuid4().hex[:8]
    manager = AgentModel(user_id=user.id, name=f"pause manager {suffix}")
    db.add(manager)
    db.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name=f"Pause workforce {suffix}",
        manager_agent_id=manager.id,
        status="published",
    )
    db.add(workforce)
    db.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        user_id=user.id,
        status="running",
        snapshot={},
        is_preview=False,
    )
    db.add(run)
    db.flush()
    return int(run.id)


def _projected_task(factory, *, source: str = "trigger") -> dict[str, Any]:
    """A V2 workforce task with a RUNNING TriggerRun, ready to start a run."""
    with factory() as db:
        user = db.query(User).order_by(User.id).first()
        task = Task(
            user_id=user.id,
            title="Projected task",
            description="persistence failure projections",
            source=source,
            conversation_storage_version=2,
        )
        db.add(task)
        db.flush()
        workforce_run_id = _workforce_run(db, user)
        workforce_run = db.get(WorkforceRun, workforce_run_id)
        workforce_run.task_id = task.id
        task.agent_config = {"workforce_run_id": workforce_run_id}
        trigger = AgentTrigger(
            user_id=user.id,
            workforce_id=workforce_run.workforce_id,
            type=TriggerType.SCHEDULED.value,
            name=f"Projection trigger {task.id}",
            config={},
        )
        db.add(trigger)
        db.flush()
        trigger_run = TriggerRun(
            trigger_id=trigger.id,
            task_id=task.id,
            status=TriggerRunStatus.RUNNING.value,
            idempotency_key=f"projection-{task.id}",
        )
        db.add(trigger_run)
        db.commit()
        return {
            "task": int(task.id),
            "user": int(user.id),
            "workforce_run": workforce_run_id,
            "trigger_run": int(trigger_run.id),
            "turn_id": f"turn-{uuid.uuid4().hex[:8]}",
        }


def _prepare_run(
    factory, ids: dict[str, Any], *, checkpoint: bool = True, **fields: Any
) -> TaskLease:
    """Start a run with prior output, a pending user turn and a checkpoint."""
    lease = _start_run(factory, ids["task"], **fields)
    with factory() as db:
        task = db.get(Task, ids["task"])
        task.output = "previous answer"
        persist_user_message_no_commit(
            db,
            ids["task"],
            ids["user"],
            "2+2",
            turn_id=ids["turn_id"],
            delivery_status=DELIVERY_PENDING,
        )
        db.commit()
    if checkpoint:
        _checkpoint(factory, ids["task"], lease.run_id)
    with factory() as db:
        ids["version"] = int(db.get(Task, ids["task"]).state_version)
    return lease


def _projection(factory, ids: dict[str, Any], lease: TaskLease) -> dict[str, Any]:
    tid = ids["task"]
    with factory() as db:
        task = db.get(Task, tid)
        trigger_run = db.get(TriggerRun, ids["trigger_run"])
        workforce_run = db.get(WorkforceRun, ids["workforce_run"])
        messages = (
            db.query(TaskChatMessage)
            .filter(TaskChatMessage.task_id == tid)
            .order_by(TaskChatMessage.id)
            .all()
        )
        return {
            "status": task.status,
            "control_state": task.control_state,
            "error_message": task.error_message,
            "output": task.output,
            "completion_outcome": task.completion_outcome,
            "runner_id": task.runner_id,
            "lease_expires_at": task.lease_expires_at,
            "lease_attempt_id": task.lease_attempt_id,
            "same_run": task.run_id == lease.run_id,
            "version_delta": int(task.state_version) - ids["version"],
            "settled": [
                row.payload for row in facts(db, tid) if row.kind == "execution_settled"
            ],
            "assistant": [m.content for m in messages if m.role == "assistant"],
            "delivery": [m.delivery_status for m in messages if m.role == "user"],
            "trigger_run": (
                trigger_run.status,
                trigger_run.error_message,
                trigger_run.finished_at is not None,
            ),
            "workforce_run": (workforce_run.status, workforce_run.completed_at is None),
        }


async def _schedule_failing_turn(
    factory, ids: dict[str, Any], lease: TaskLease, error: BaseException
) -> list[str]:
    """Run one scheduled turn whose execution raises ``error``."""
    events: list[dict[str, Any]] = []

    async def publish(event, _task_id):
        events.append(event)

    async def execute(**_kwargs):
        raise error

    with (
        patch.object(orchestrator, "run_task_lease_heartbeat", new=AsyncMock()),
        patch.object(
            orchestrator, "load_task_setup_snapshot_sync", return_value=MagicMock()
        ),
        patch.object(
            orchestrator, "resolve_execution_scope", return_value=None, create=True
        ),
        patch("xagent.web.services.task_execution.execute_task_background", execute),
        patch("xagent.web.services.task_event_display.publish_task_result", publish),
        patch("xagent.web.services.task_events.publish_task_event", publish),
        patch.object(orchestrator, "_get_agent_manager", return_value=MagicMock()),
    ):
        await _schedule_bg(
            task_id=ids["task"],
            task_owner_user_id=ids["user"],
            task_source="trigger",
            task_lease=lease,
            payload=TaskTurnPayload("2+2", turn_id=ids["turn_id"]),
            force_fresh=False,
            context=None,
        )
    return [str(event.get("type")) for event in events]


def _without_interruption(monkeypatch) -> None:
    """Settle as before this change: no settling path sees an interruption."""
    monkeypatch.setattr(
        orchestrator, "settlement_interruption_for_failure", lambda _exc: None
    )


def _persistence_error() -> ExecutionEventPersistenceError:
    return ExecutionEventPersistenceError("event write failed")


def _expire_and_recover(factory, ids: dict[str, Any]) -> TaskStatus | None:
    with factory() as db:
        db.get(Task, ids["task"]).lease_expires_at = utc_now() - timedelta(seconds=5)
        db.commit()
    with factory() as db:
        (candidate,) = [
            c
            for c in get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=10)
            if c.task_id == ids["task"]
        ]
        status = recover_task_lease_candidate_no_commit(
            db, candidate, recovered_at=utc_now()
        )
        db.commit()
    return status


@pytest.mark.asyncio
async def test_pause_projects_like_lease_recovery(canonical):
    """S2's pause leaves every projection as a lease-expiry pause does,
    except the TriggerRun message, which names an interruption."""

    factory, _tid = canonical
    paused, recovered = _projected_task(factory), _projected_task(factory)
    paused_lease = _prepare_run(factory, paused)
    recovered_lease = _prepare_run(factory, recovered)

    frames = await _schedule_failing_turn(
        factory, paused, paused_lease, _persistence_error()
    )
    assert _expire_and_recover(factory, recovered) == TaskStatus.PAUSED

    left = _projection(factory, paused, paused_lease)
    right = _projection(factory, recovered, recovered_lease)
    assert left.pop("trigger_run") == (
        TriggerRunStatus.FAILED.value,
        TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR,
        True,
    )
    assert right.pop("trigger_run") == (
        TriggerRunStatus.FAILED.value,
        TASK_LEASE_PAUSED_TRIGGER_ERROR,
        True,
    )
    assert TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR != TASK_LEASE_PAUSED_TRIGGER_ERROR
    assert left == right
    assert left["status"] == TaskStatus.PAUSED
    assert left["output"] == "previous answer"
    assert left["workforce_run"] == ("paused", True)
    # The run may have consumed the message: "do not resend", never failed.
    assert left["delivery"] == ["dispatched"]
    assert frames == ["task_paused"]
    _task, row, _events = _state(factory, paused["task"])
    assert (row.reason, row.state) == ("persistence_failure", "manual")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "state", "detail", "event"),
    [
        ("trigger", "disabled", None, "interrupted"),
        # Ineligibility wins over the switch.
        ("sdk", "ineligible", "ineligible:source_sdk", "ineligible"),
    ],
)
async def test_switch_off_settles_exactly_as_before(
    canonical, monkeypatch, source, state, detail, event
):
    """Switch off: the same FAILED settlement as without any interruption,
    plus a ``disabled`` row (``ineligible`` for an ineligible task)."""

    monkeypatch.setenv(PAUSE_SWITCH, "false")
    factory, _tid = canonical
    gated = _projected_task(factory, source=source)
    baseline = _projected_task(factory, source=source)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    gated_frames = await _schedule_failing_turn(
        factory, gated, gated_lease, _persistence_error()
    )
    with monkeypatch.context() as m:
        _without_interruption(m)
        baseline_frames = await _schedule_failing_turn(
            factory, baseline, baseline_lease, _persistence_error()
        )

    gated_projection = _projection(factory, gated, gated_lease)
    assert gated_projection == _projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    assert gated_frames == baseline_frames == ["task_error"]
    _task, row, events = _state(factory, gated["task"])
    assert (row.reason, row.state, row.state_detail) == (
        "persistence_failure",
        state,
        detail,
    )
    assert [(e.event, e.detail["task_status"]) for e in events] == [(event, "failed")]
    assert _state(factory, baseline["task"])[1] is None


@pytest.mark.asyncio
async def test_ineligible_task_settles_exactly_as_before(canonical, monkeypatch):
    factory, _tid = canonical
    gated = _projected_task(factory, source="sdk")
    baseline = _projected_task(factory, source="sdk")
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    await _schedule_failing_turn(factory, gated, gated_lease, _persistence_error())
    with monkeypatch.context() as m:
        _without_interruption(m)
        await _schedule_failing_turn(
            factory, baseline, baseline_lease, _persistence_error()
        )

    assert _projection(factory, gated, gated_lease) == _projection(
        factory, baseline, baseline_lease
    )
    _task, row, events = _state(factory, gated["task"])
    assert (row.state, row.state_detail) == ("ineligible", "ineligible:source_sdk")
    assert [e.event for e in events] == ["ineligible"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["python", "database", "marker"])
async def test_recording_failure_does_not_change_the_pause(
    canonical, monkeypatch, failure
):
    factory, _tid = canonical
    recorded, unrecorded = _projected_task(factory), _projected_task(factory)
    recorded_lease = _prepare_run(factory, recorded)
    unrecorded_lease = _prepare_run(factory, unrecorded)

    recorded_frames = await _schedule_failing_turn(
        factory, recorded, recorded_lease, _persistence_error()
    )
    with monkeypatch.context() as m:
        _fail_recording(m, failure)
        unrecorded_frames = await _schedule_failing_turn(
            factory, unrecorded, unrecorded_lease, _persistence_error()
        )

    assert _projection(factory, recorded, recorded_lease) == _projection(
        factory, unrecorded, unrecorded_lease
    )
    assert recorded_frames == unrecorded_frames == ["task_paused"]
    assert _state(factory, unrecorded["task"])[1:] == (None, [])


def test_resume_requested_run_pauses_like_lease_recovery(canonical):
    """A RUNNING row already RESUME_REQUESTED is paused (control ``paused``)
    by an interruption, exactly as lease-expiry recovery pauses it."""

    factory, _tid = canonical
    settled, recovered = _projected_task(factory), _projected_task(factory)
    settled_lease = _prepare_run(factory, settled)
    recovered_lease = _prepare_run(factory, recovered)
    with factory() as db:
        for ids in (settled, recovered):
            db.get(Task, ids["task"]).control_state = "resume_requested"
        db.commit()
    with factory() as db:
        for ids in (settled, recovered):
            ids["version"] = int(db.get(Task, ids["task"]).state_version)

    assert _settle(settled_lease, interruption=_persistence_interruption())[0]
    assert _expire_and_recover(factory, recovered) == TaskStatus.PAUSED

    left = _projection(factory, settled, settled_lease)
    right = _projection(factory, recovered, recovered_lease)
    for side in (left, right):
        side.pop("trigger_run")  # Messages differ by design.
        side.pop("delivery")  # Reconciled by the scheduler, not by settle.
    assert left == right
    assert (left["status"], left["control_state"]) == (TaskStatus.PAUSED, "paused")
    _task, row, _events = _state(factory, settled["task"])
    assert row.reason == "persistence_failure"


def _finalize_and_release(factory, ids, lease, result) -> Any:
    finalized = _finalize(factory, ids["task"], lease, result)
    # The scheduler's settlement releases the lease after the result path.
    assert settle_task_lease_isolated(lease)
    return finalized


def _drop_reason(projection: dict[str, Any]) -> dict[str, Any]:
    """The settled fact carries the result itself, reason included."""
    for fact in projection["settled"]:
        assert fact["result"].pop("interruption_reason") == "persistence_failure"
    return projection


@pytest.mark.parametrize("switch", ["on", "off"])
def test_quota_stop_wins_over_its_interruption_reason(canonical, monkeypatch, switch):
    """``quota_exceeded`` keeps the runner's reason but settles as a quota
    stop: the same FAILED projections as a quota result without one."""

    if switch == "off":
        monkeypatch.setenv(PAUSE_SWITCH, "false")
    factory, _tid = canonical
    quota = {**PERSISTENCE_RESULT, "status": "quota_exceeded", "error_code": "q"}
    plain_quota = {k: v for k, v in quota.items() if k != "interruption_reason"}
    gated, baseline = _projected_task(factory), _projected_task(factory)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    left = _finalize_and_release(factory, gated, gated_lease, quota)
    right = _finalize_and_release(factory, baseline, baseline_lease, plain_quota)

    assert left.report.paused_for is None
    assert right.report.paused_for is None
    gated_projection = _drop_reason(_projection(factory, gated, gated_lease))
    assert gated_projection == _projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    assert _state(factory, gated["task"])[1] is None


def test_result_switch_off_settles_exactly_as_before(canonical, monkeypatch):
    monkeypatch.setenv(PAUSE_SWITCH, "0")
    factory, _tid = canonical
    plain = {k: v for k, v in PERSISTENCE_RESULT.items() if k != "interruption_reason"}
    gated, baseline = _projected_task(factory), _projected_task(factory)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    _finalize_and_release(factory, gated, gated_lease, dict(PERSISTENCE_RESULT))
    _finalize_and_release(factory, baseline, baseline_lease, plain)

    gated_projection = _drop_reason(_projection(factory, gated, gated_lease))
    assert gated_projection == _projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    _task, row, _events = _state(factory, gated["task"])
    assert (row.reason, row.state) == ("persistence_failure", "disabled")


@pytest.mark.parametrize("failure", ["python", "database", "marker"])
def test_result_recording_failure_does_not_change_the_pause(
    canonical, monkeypatch, failure
):
    factory, _tid = canonical
    recorded, unrecorded = _projected_task(factory), _projected_task(factory)
    recorded_lease = _prepare_run(factory, recorded)
    unrecorded_lease = _prepare_run(factory, unrecorded)

    left = _finalize_and_release(
        factory, recorded, recorded_lease, dict(PERSISTENCE_RESULT)
    )
    with monkeypatch.context() as m:
        _fail_recording(m, failure)
        right = _finalize_and_release(
            factory, unrecorded, unrecorded_lease, dict(PERSISTENCE_RESULT)
        )

    assert left.report.paused_for is not None
    assert right.report.paused_for is not None
    left_projection = _projection(factory, recorded, recorded_lease)
    assert left_projection == _projection(factory, unrecorded, unrecorded_lease)
    assert left_projection["status"] == TaskStatus.PAUSED
    assert left_projection["trigger_run"][1] == TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR
    assert _state(factory, unrecorded["task"])[1:] == (None, [])


def test_resumed_quota_result_settles_exactly_as_before(canonical):
    factory, _tid = canonical
    quota = {**PERSISTENCE_RESULT, "status": "quota_exceeded"}
    plain_quota = {k: v for k, v in quota.items() if k != "interruption_reason"}
    gated, baseline = _projected_task(factory), _projected_task(factory)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    for ids, lease, result in (
        (gated, gated_lease, quota),
        (baseline, baseline_lease, plain_quota),
    ):
        finalized = _finalize_resumed_task(
            ids["task"],
            status="quota_exceeded",
            success=False,
            output=result["output"],
            task_owner_user_id=ids["user"],
            result=dict(result),
            task_lease=lease,
            prepared_outputs=NO_OUTPUTS,
        )
        assert finalized["final_status"] == "failed"
        assert finalized["report"].paused_for is None

    gated_projection = _drop_reason(_projection(factory, gated, gated_lease))
    assert gated_projection == _projection(factory, baseline, baseline_lease)
    assert _state(factory, gated["task"])[1] is None


# --------------------------------------------- result paths, real database


def _finalize_resumed(ids: dict[str, Any], lease: TaskLease, result: dict[str, Any]):
    return _finalize_resumed_task(
        ids["task"],
        status=str(result.get("status") or "failed"),
        success=False,
        output=result["output"],
        task_owner_user_id=ids["user"],
        result=dict(result),
        task_lease=lease,
        prepared_outputs=NO_OUTPUTS,
    )


def _settle_result(path: str, factory, ids, lease, result) -> None:
    """Settle ``result`` through S1 (and the scheduler's release) or S3."""
    if path == "result":
        _finalize_and_release(factory, ids, lease, dict(result))
    else:
        assert _finalize_resumed(ids, lease, result)["lease_released"]


RESULT_PATHS = pytest.mark.parametrize("path", ["result", "resumed_result"])


@RESULT_PATHS
def test_result_pause_projects_like_lease_recovery(canonical, path):
    factory, _tid = canonical
    paused, recovered = _projected_task(factory), _projected_task(factory)
    paused_lease = _prepare_run(factory, paused)
    recovered_lease = _prepare_run(factory, recovered)

    _settle_result(path, factory, paused, paused_lease, PERSISTENCE_RESULT)
    assert _expire_and_recover(factory, recovered) == TaskStatus.PAUSED

    left = _projection(factory, paused, paused_lease)
    right = _projection(factory, recovered, recovered_lease)
    assert left.pop("trigger_run")[1] == TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR
    assert right.pop("trigger_run")[1] == TASK_LEASE_PAUSED_TRIGGER_ERROR
    # Only lease recovery reconciles orphaned delivery rows.
    left.pop("delivery")
    right.pop("delivery")
    assert left == right
    assert left["settled"] == [PAUSED_FACT]
    _task, row, _events = _state(factory, paused["task"])
    assert (row.reason, row.state) == ("persistence_failure", "manual")


@RESULT_PATHS
@pytest.mark.parametrize(
    "case",
    ["unknown_tool_effect", "not_recoverable", "pause_requested", "resume_requested"],
)
def test_result_paths_decide_every_verdict(canonical, monkeypatch, path, case):
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids, checkpoint=case != "not_recoverable")
    if case == "unknown_tool_effect":
        monkeypatch.setattr(
            task_auto_recovery,
            "resolve_checkpoint_recovery_with_data",
            lambda _db, _candidate: CheckpointRecoveryResolution(
                CheckpointRecoveryVerdict.UNKNOWN_TOOL_EFFECT
            ),
        )
    if case in {"pause_requested", "resume_requested"}:
        with factory() as db:
            db.get(Task, ids["task"]).control_state = case
            db.commit()

    _settle_result(path, factory, ids, lease, PERSISTENCE_RESULT)

    task, row, _events = _state(factory, ids["task"])
    settled = _projection(factory, ids, lease)["settled"]
    if case in {"unknown_tool_effect", "not_recoverable"}:
        assert task.status == TaskStatus.FAILED
        assert task.error_message == (
            TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
            if case == "unknown_tool_effect"
            else PERSISTENCE_RESULT["error"]
        )
        assert (row.reason, row.state) == (case, "manual")
        assert [fact["status"] for fact in settled] == ["failed"]
    else:
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        assert task.runner_id is None
        assert (row.reason, row.state) == (
            "user_pause" if case == "pause_requested" else "persistence_failure",
            "manual",
        )
        assert settled == [PAUSED_FACT]


@RESULT_PATHS
@pytest.mark.parametrize("case", ["switch_off", "ineligible"])
def test_result_paths_settle_exactly_as_before(canonical, monkeypatch, path, case):
    if case == "switch_off":
        monkeypatch.setenv(PAUSE_SWITCH, "false")
    source = "sdk" if case == "ineligible" else "trigger"
    factory, _tid = canonical
    plain = {k: v for k, v in PERSISTENCE_RESULT.items() if k != "interruption_reason"}
    gated = _projected_task(factory, source=source)
    baseline = _projected_task(factory, source=source)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    _settle_result(path, factory, gated, gated_lease, PERSISTENCE_RESULT)
    _settle_result(path, factory, baseline, baseline_lease, plain)

    gated_projection = _drop_reason(_projection(factory, gated, gated_lease))
    assert gated_projection == _projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    _task, row, _events = _state(factory, gated["task"])
    assert (row.reason, row.state) == (
        "persistence_failure",
        "disabled" if case == "switch_off" else "ineligible",
    )
    assert _state(factory, baseline["task"])[1] is None


def test_resume_settlement_pauses_on_a_real_database(canonical):
    """The resume exception path's settlement against the real database."""
    from xagent.web.services.task_execution import _settle_resumed_task_lease

    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)
    report = SettlementReport()

    assert _settle_resumed_task_lease(
        lease,
        error_message=RUN_ERROR,
        interruption=_persistence_interruption(),
        report=report,
    )

    assert report.paused_for is InterruptionReason.PERSISTENCE_FAILURE
    projection = _projection(factory, ids, lease)
    assert (projection["status"], projection["runner_id"]) == (TaskStatus.PAUSED, None)
    assert projection["settled"] == [PAUSED_FACT]
    assert projection["trigger_run"][1] == TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR


# ----------------------------------------- undecidable interruptions defer to TTL


def _break_decision(m: pytest.MonkeyPatch, failure: str, error: Exception) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    if failure == "eligibility":
        m.setattr(task_auto_recovery, "auto_recovery_eligibility", boom)
    else:
        m.setattr(event_recovery, "read_event_checkpoint", boom)


def _connection_reset() -> OperationalError:
    return OperationalError("SELECT", {}, Exception("connection reset"))


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["exception", "result", "resumed_result"])
@pytest.mark.parametrize("failure", ["eligibility", "checkpoint_read"])
async def test_undecidable_interruption_keeps_the_lease_for_ttl(
    canonical, monkeypatch, path, failure
):
    """A connectivity failure deciding the interruption defers it on every
    settling path: the run stays RUNNING under its lease, and TTL recovery
    pauses it once the database answers again."""

    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)

    with monkeypatch.context() as m:
        _break_decision(m, failure, _connection_reset())
        if path == "exception":
            frames = await _schedule_failing_turn(
                factory, ids, lease, _persistence_error()
            )
            assert frames == []
        else:
            with pytest.raises(InterruptionSettlementDeferred) as deferred:
                if path == "result":
                    _finalize(factory, ids["task"], lease, dict(PERSISTENCE_RESULT))
                else:
                    _finalize_resumed(ids, lease, PERSISTENCE_RESULT)
            assert deferred.value.__cause__ is not None

    task, row, _events = _state(factory, ids["task"])
    assert task.status == TaskStatus.RUNNING
    assert (task.runner_id, task.run_id) == (lease.runner_id, lease.run_id)
    assert row is None
    assert _projection(factory, ids, lease)["settled"] == []

    assert _expire_and_recover(factory, ids) == TaskStatus.PAUSED
    _task, row, _events = _state(factory, ids["task"])
    assert (row.reason, row.state) == ("lease_expired", "manual")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["exception", "result", "resumed_result"])
@pytest.mark.parametrize("failure", ["eligibility", "checkpoint_read"])
async def test_reproducible_decision_fault_settles_as_before(
    canonical, monkeypatch, path, failure
):
    """A fault that would recur on every attempt must not defer: TTL recovery
    reads the checkpoint the same way, so the run would stay RUNNING forever.
    It settles FAILED, as before this change, and releases the lease."""

    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)

    with monkeypatch.context() as m:
        _break_decision(m, failure, TypeError("resolver bug"))
        if path == "exception":
            await _schedule_failing_turn(factory, ids, lease, _persistence_error())
        elif path == "result":
            _finalize_and_release(factory, ids, lease, dict(PERSISTENCE_RESULT))
        else:
            _finalize_resumed(ids, lease, PERSISTENCE_RESULT)

    task, _row, _events = _state(factory, ids["task"])
    if path == "exception" and failure == "checkpoint_read":
        # The legacy settlement itself classifies the run's tool effect from
        # the same checkpoint (run_has_unknown_tool_effect) and has always
        # kept the lease when that read raises; unchanged here.
        assert task.status == TaskStatus.RUNNING
        assert (task.runner_id, task.run_id) == (lease.runner_id, lease.run_id)
        return
    assert task.status == TaskStatus.FAILED
    assert task.runner_id is None and task.lease_expires_at is None


# ----------------------------------------------------- broadcasts and metrics


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "message"),
    [
        (InterruptionReason.USER_PAUSE, TASK_USER_PAUSED_MESSAGE),
        (InterruptionReason.PERSISTENCE_FAILURE, TASK_INTERRUPTION_PAUSED_MESSAGE),
    ],
)
async def test_pause_broadcast_says_who_paused(reason, message):
    with patch(
        "xagent.web.services.task_events.publish_task_event", new=AsyncMock()
    ) as published:
        await publish_settlement_pause(
            7,
            SettlementReport(control_state={"status": "paused"}, paused_for=reason),
        )

    event = published.await_args.args[0]
    assert (event["type"], event["message"], event["status"]) == (
        "task_paused",
        message,
        "paused",
    )
    assert event["interruption_reason"] == reason.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "report",
    [SettlementReport(), SettlementReport(control_state={"status": "failed"})],
)
async def test_pause_broadcast_skips_a_report_without_a_pause(report):
    with patch(
        "xagent.web.services.task_events.publish_task_event", new=AsyncMock()
    ) as published:
        await publish_settlement_pause(7, report)

    published.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(("switch", "failed"), [("true", False), ("false", True)])
async def test_trigger_metrics_do_not_count_a_pause_as_a_failure(
    canonical, monkeypatch, switch, failed
):
    from xagent.core.utils import setup_metrics

    monkeypatch.setenv(PAUSE_SWITCH, switch)
    finished: list[bool] = []
    monkeypatch.setattr(setup_metrics.trigger_execution, "start", lambda: 0.0)
    monkeypatch.setattr(
        setup_metrics.trigger_execution,
        "finish",
        lambda _started, *, cancelled, failed: finished.append(failed),
    )
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)

    await _schedule_failing_turn(factory, ids, lease, _persistence_error())
    await asyncio.sleep(0)

    assert finished == [failed]


@pytest.mark.asyncio
async def test_explicit_cancel_is_not_settled_as_an_interruption():
    settle_mock = MagicMock(return_value=True)
    publish_pause, _frames = await _resume_raising(
        ExecutionEventPersistenceError("event write failed"),
        settle_mock,
        explicit_cancel=True,
    )

    settle_mock.assert_called_once()
    assert settle_mock.call_args.kwargs["interruption"] is None
    publish_pause.assert_not_awaited()
