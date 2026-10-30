"""Unknown-outcome signals for execution events.

Every case where the system does not know whether something durable happened
produces one structured WARNING/INFO line that locates the task and one
bounded-attribute counter increment: a tool whose external effect is unknown,
a turn or command acceptance whose COMMIT outcome is unknown, a finalize whose
lost acknowledgement could not be resolved, a settlement deferred to lease
recovery. Each test drives the real code path; metrics are asserted as
exported.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError
from sqlalchemy.orm import Session

from tests.core.agent.test_react import FakeTool
from tests.web.api.test_websocket_owner_actor import _task as _owner_task
from tests.web.api.test_websocket_owner_actor import _user as _owner_user
from tests.web.api.test_websocket_owner_actor import db_session as db_session_fixture
from tests.web.services.test_execution_event_persistence_settlement import (
    _clear_bg_manager as _clear_bg_manager_fixture,
)
from tests.web.services.test_execution_event_persistence_settlement import (
    _fail_once,
    _react_execution,
    _run_scheduled_turn,
)
from tests.web.services.test_execution_event_recovery import tracer_for
from tests.web.services.test_execution_event_signals import (
    exported,
    fields,
    key,
)
from tests.web.services.test_execution_event_signals import (
    metric_reader as metric_reader_fixture,
)
from tests.web.services.test_persistence_failure_pause import (
    NO_OUTPUTS,
    _finalize,
)
from tests.web.services.test_persistence_failure_pause import (
    _late_bound_sessions as _late_bound_sessions_fixture,
)
from tests.web.services.test_persistence_failure_pause import (
    _resume_raising,
    _settle,
    _start_run,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent import PatternRuntime
from xagent.core.agent.checkpoint import (
    CheckpointUnavailableError,
    TraceCheckpointStore,
)
from xagent.web.models import database as database_module
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services import (
    execution_event_signals,
    task_command_execution,
    task_execution,
)
from xagent.web.services import task_execution_event_recovery as event_recovery
from xagent.web.services import (
    task_lease_recovery,
)
from xagent.web.services import task_orchestrator as orchestrator
from xagent.web.services.execution_event_signals import report_unknown_outcome
from xagent.web.services.task_auto_recovery import InterruptionSettlementDeferred
from xagent.web.services.task_execution import (
    _ChatMessageWitness,
    _commit_finalization_or_reconcile,
    _finalization_witness_committed,
    _finalize_resumed_task,
)
from xagent.web.services.task_execution_event_writer import FactWitness
from xagent.web.services.task_lease_recovery import (
    recover_expired_task_leases_batch_isolated,
    recover_task_lease_candidate_isolated,
)
from xagent.web.services.task_lease_service import (
    TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR,
    TaskLease,
    acquire_task_lease,
    bind_task_lease_context,
    get_expired_task_lease_candidates,
    utc_now,
)
from xagent.web.services.task_orchestrator import (
    TaskTurnCommitOutcomeUnknown,
    TaskTurnPayload,
    TurnKind,
    _begin_turn_atomic_sync,
    _ClaimedTurn,
    commit_claimed_turn_or_reconcile,
)

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
metric_reader = metric_reader_fixture
db_session = db_session_fixture
_clear_bg_manager = _clear_bg_manager_fixture
_late_bound_sessions = _late_bound_sessions_fixture

SENTINEL = "T3A-SENTINEL"
COUNTER = "xagent.execution_event.unknown_outcomes"
PREFIX = "execution_event_outcome_unknown "


@pytest.fixture(autouse=True)
def _capture(caplog):
    caplog.set_level(logging.INFO, logger=execution_event_signals.logger.name)


def records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith(PREFIX)]


def lines(caplog) -> list[str]:
    return [r.getMessage() for r in records(caplog)]


def counts(reader) -> dict[tuple[tuple[str, str], ...], int | float]:
    return exported(reader, COUNTER)


def assert_no_leak(caplog) -> None:
    """No line carries exception or payload text (every injected error and
    payload embeds ``SENTINEL``)."""
    for line in lines(caplog):
        assert SENTINEL not in line


def assert_one(
    caplog,
    reader,
    *,
    kind: str,
    task: int,
    run_id: str | None,
    ref: str | None,
    storage: str,
    level: int = logging.WARNING,
) -> dict[str, str]:
    """Exactly one line and one counter increment, with these fields."""
    found = records(caplog)
    assert len(found) == 1, [r.getMessage() for r in found]
    assert found[0].levelno == level
    line = found[0].getMessage()
    assert line.startswith(PREFIX + f"task_id={task} ")
    data = fields(line)
    assert data == {
        "task_id": str(task),
        "run_id": run_id or "-",
        "storage": storage,
        "component": "execution-events",
        "kind": kind,
        "ref": ref or "-",
    }
    assert counts(reader) == {key(operation=kind, storage=storage): 1}
    assert_no_leak(caplog)
    return data


def assert_none(caplog, reader) -> None:
    assert lines(caplog) == []
    assert counts(reader) == {}


# ------------------------------------------------------ report_unknown_outcome


def test_report_unknown_outcome_line_and_counter(caplog, metric_reader):
    report_unknown_outcome("turn_commit", task_id=7, run_id="run-1", ref="turn-1")
    report_unknown_outcome(
        "finalize_ack_reconciled", task_id=8, storage="v2", level=logging.INFO
    )

    first, second = records(caplog)
    assert first.levelno == logging.WARNING
    assert first.getMessage() == (
        "execution_event_outcome_unknown task_id=7 run_id=run-1 storage=unknown "
        "component=execution-events kind=turn_commit ref=turn-1"
    )
    assert second.levelno == logging.INFO
    assert second.getMessage() == (
        "execution_event_outcome_unknown task_id=8 run_id=- storage=v2 "
        "component=execution-events kind=finalize_ack_reconciled ref=-"
    )
    assert counts(metric_reader) == {
        key(operation="turn_commit", storage="unknown"): 1,
        key(operation="finalize_ack_reconciled", storage="v2"): 1,
    }
    assert first.exc_info is None and second.exc_info is None


def test_report_unknown_outcome_never_raises(monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr(execution_event_signals, "increment_counter", boom)
    # A failing counter must not suppress the log line.
    report_unknown_outcome("tool_effect", task_id=1, storage="v2")
    assert len(lines(caplog)) == 1

    monkeypatch.setattr(execution_event_signals.logger, "log", boom)
    report_unknown_outcome("tool_effect", task_id=1, storage="v2")
    assert len(lines(caplog)) == 1


# --------------------------------------------------------------- tool_effect


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_switch", ["true", "false"], ids=["decided", "legacy"])
@pytest.mark.parametrize("channel", [False, True], ids=["web", "channel"])
async def test_tool_effect_is_reported_once_per_settlement(
    canonical, monkeypatch, caplog, metric_reader, channel, pause_switch
):
    """Both settlement classifiers (the interruption decision and the legacy
    classification) report the same occurrence once, web and channel."""
    monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", pause_switch)
    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    failed = _fail_once(monkeypatch, "tool_execution_end")

    await _run_scheduled_turn(
        tid, lease, _react_execution(tid, FakeTool()), channel=channel
    )

    assert failed == ["tool_execution_end"]
    with factory() as db:
        assert (
            db.get(Task, tid).error_message == TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
        )
    (attempt,) = _started_attempt_ids(factory, tid)
    assert_one(
        caplog,
        metric_reader,
        kind="tool_effect",
        task=tid,
        run_id=lease.run_id,
        ref=attempt,
        storage="v2",
    )


def _started_attempt_ids(factory, tid: int) -> list[str]:
    with factory() as db:
        return [
            str(attempt)
            for attempt in db.scalars(
                sa.select(TaskExecutionEvent.tool_attempt_id)
                .where(
                    TaskExecutionEvent.task_id == tid,
                    TaskExecutionEvent.kind == "tool_execution_start",
                )
                .order_by(TaskExecutionEvent.sequence)
            )
        ]


@pytest.mark.asyncio
async def test_unstarted_tool_is_not_a_tool_effect(
    canonical, monkeypatch, caplog, metric_reader
):
    monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", "false")
    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    failed = _fail_once(monkeypatch, "tool_execution_start")

    await _run_scheduled_turn(tid, lease, _react_execution(tid, FakeTool()))

    assert failed == ["tool_execution_start"]
    assert_none(caplog, metric_reader)


ATTEMPT = "attempt-unknown"


async def _run_with_unknown_tool(factory, tid: int) -> TaskLease:
    """A RUNNING run whose checkpoint names a started tool attempt with no
    committed outcome."""
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    call = {
        "id": "call",
        "name": "calculator",
        "args": {},
        "assistant_message_id": "batch",
        "tool_attempt_id": ATTEMPT,
    }
    with bind_task_lease_context(lease):
        store = TraceCheckpointStore(tracer_for(tid))
        runtime = PatternRuntime(tracer=store)
        await store.save(
            {
                "execution_id": str(tid),
                "context": {"messages": []},
                "pattern_state": {"pending_tool_calls": [call]},
            }
        )
        await runtime.on_tool_start(tool_call=call)
    return lease


def _expire(factory, tid: int) -> None:
    with factory() as db:
        db.get(Task, tid).lease_expires_at = utc_now() - timedelta(seconds=5)
        db.commit()


async def _expire_run_with_unknown_tool(factory, tid: int) -> TaskLease:
    lease = await _run_with_unknown_tool(factory, tid)
    _expire(factory, tid)
    return lease


@pytest.mark.asyncio
async def test_tool_effect_is_reported_when_lease_recovery_settles_it(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    lease = await _expire_run_with_unknown_tool(factory, tid)
    with factory() as db:
        candidate = get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=1)[0]

    assert (
        recover_task_lease_candidate_isolated(candidate, recovered_at=utc_now())
        == TaskStatus.FAILED
    )
    # A second attempt on the same candidate settles nothing and reports nothing.
    assert (
        recover_task_lease_candidate_isolated(candidate, recovered_at=utc_now()) is None
    )

    assert_one(
        caplog,
        metric_reader,
        kind="tool_effect",
        task=tid,
        run_id=lease.run_id,
        ref=ATTEMPT,
        storage="v2",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["cas", "postgresql"])
async def test_tool_effect_is_reported_by_the_recovery_sweep(
    canonical, engine, monkeypatch, caplog, metric_reader, path
):
    """Both sweep paths, the CAS batch and the PostgreSQL row-lock batch."""
    from tests.web.services.test_task_lease_expiry_interruption import (
        _force_postgresql_batch,
    )

    factory, tid = canonical
    if path == "postgresql":
        _force_postgresql_batch(monkeypatch, engine)
    else:
        monkeypatch.setattr(
            task_lease_recovery, "_use_postgresql_recovery_partitioning", lambda: False
        )
    lease = await _expire_run_with_unknown_tool(factory, tid)

    batch = recover_expired_task_leases_batch_isolated(
        cutoff=utc_now(), batch_size=10, after=None
    )
    again = recover_expired_task_leases_batch_isolated(
        cutoff=utc_now(), batch_size=10, after=None
    )

    assert (batch.recovered, again.recovered) == (1, 0)
    assert_one(
        caplog,
        metric_reader,
        kind="tool_effect",
        task=tid,
        run_id=lease.run_id,
        ref=ATTEMPT,
        storage="v2",
    )


PERSISTENCE_RESULT = {
    "success": False,
    "status": "failed",
    "output": "All patterns failed",
    "error": "All 1 patterns failed: OperationalError: connection reset",
    "interruption_reason": "persistence_failure",
}


def _settle_unknown_tool_effect(path: str, factory, tid: int, lease: TaskLease) -> Any:
    """Settle the run through one settling path that decides it as an
    unknown tool effect."""
    from tests.web.services.test_persistence_failure_pause import (
        _persistence_interruption,
    )
    from xagent.web.services.managed_task_lease import (
        finalize_managed_task_lease_result,
    )

    if path == "settle_decided":
        return _settle(lease, interruption=_persistence_interruption())
    if path == "settle_classified":
        return _settle(lease)
    if path == "managed":
        with factory() as db:
            return finalize_managed_task_lease_result(
                db,
                lease,
                status=TaskStatus.FAILED,
                error_message="channel run abandoned",
            )
    if path == "finalize":
        return _finalize(factory, tid, lease, dict(PERSISTENCE_RESULT))
    assert path == "resume_finalize"
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)
    return _finalize_resumed_task(
        tid,
        status="failed",
        success=False,
        output="All patterns failed",
        task_owner_user_id=user_id,
        result=dict(PERSISTENCE_RESULT),
        task_lease=lease,
        prepared_outputs=NO_OUTPUTS,
    )


SETTLEMENT_PATHS = [
    "settle_decided",
    "settle_classified",
    "managed",
    "finalize",
    "resume_finalize",
]


@pytest.fixture
def settlement_commit(monkeypatch):
    """Refuse the COMMIT of the settlement transaction (nothing applied).

    That transaction is the one that stages the result fact; the staged row
    is recorded so a test can prove the settlement had decided the run as an
    unknown tool effect before it rolled back. The refusal is a deadlock, not
    a dropped connection, so no other unknown outcome is reported for it.
    """
    from xagent.web.services import task_execution_event_writer

    state: dict[str, Any] = {"refuse": False, "armed": False, "staged": []}
    stage = task_execution_event_writer.stage_result_fact_no_commit
    original = Session.commit

    def staged(db, task, *args, **kwargs):
        state["staged"].append(task.error_message)
        state["armed"] = True
        return stage(db, task, *args, **kwargs)

    def commit(session: Session) -> None:
        if not (state["armed"] and state["refuse"]):
            state["armed"] = False
            return original(session)
        state["armed"] = False
        raise _deadlock()

    monkeypatch.setattr(
        task_execution_event_writer, "stage_result_fact_no_commit", staged
    )
    monkeypatch.setattr(task_execution, "stage_result_fact_no_commit", staged)
    monkeypatch.setattr(Session, "commit", commit)
    return state


@pytest.fixture
def _settles_unknown_tool_effect(monkeypatch):
    """Settlement decides interruptions; nothing schedules a resume."""
    monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", SETTLEMENT_PATHS)
@pytest.mark.usefixtures("_settles_unknown_tool_effect")
async def test_a_committed_settlement_reports_the_tool_effect_once(
    canonical, settlement_commit, caplog, metric_reader, path
):
    """Every settling path reports its unknown tool effect once, after its
    commit, carrying the attempt the verdict was reached on."""
    factory, tid = canonical
    lease = await _run_with_unknown_tool(factory, tid)

    _settle_unknown_tool_effect(path, factory, tid, lease)

    assert settlement_commit["staged"] == [TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR]
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.error_message == TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
    _expire(factory, tid)
    with factory() as db:
        assert not get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=1)
    assert_one(
        caplog,
        metric_reader,
        kind="tool_effect",
        task=tid,
        run_id=lease.run_id,
        ref=ATTEMPT,
        storage="v2",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", SETTLEMENT_PATHS)
@pytest.mark.usefixtures("_settles_unknown_tool_effect")
async def test_a_rolled_back_settlement_leaves_the_one_report_to_lease_recovery(
    canonical, settlement_commit, caplog, metric_reader, path
):
    """A settlement that decided the unknown tool effect but did not commit
    reports nothing; TTL recovery settles the run and reports it once."""
    factory, tid = canonical
    lease = await _run_with_unknown_tool(factory, tid)
    settlement_commit["refuse"] = True

    with pytest.raises(OperationalError):
        _settle_unknown_tool_effect(path, factory, tid, lease)

    assert settlement_commit["staged"] == [TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR]
    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.RUNNING
    assert_none(caplog, metric_reader)

    settlement_commit["refuse"] = False
    _expire(factory, tid)
    with factory() as db:
        candidate = get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=1)[0]
    assert (
        recover_task_lease_candidate_isolated(candidate, recovered_at=utc_now())
        == TaskStatus.FAILED
    )

    assert_one(
        caplog,
        metric_reader,
        kind="tool_effect",
        task=tid,
        run_id=lease.run_id,
        ref=ATTEMPT,
        storage="v2",
    )


@pytest.mark.asyncio
async def test_recoverable_expired_run_is_not_a_tool_effect(
    canonical, monkeypatch, caplog, metric_reader
):
    from tests.web.services.test_persistence_failure_pause import _checkpoint

    monkeypatch.setattr(
        task_lease_recovery, "_use_postgresql_recovery_partitioning", lambda: False
    )
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    with factory() as db:
        db.get(Task, tid).lease_expires_at = utc_now() - timedelta(seconds=5)
        db.commit()

    batch = recover_expired_task_leases_batch_isolated(
        cutoff=utc_now(), batch_size=10, after=None
    )

    assert batch.recovered == 1
    assert_none(caplog, metric_reader)


# --------------------------------------------------------------- turn_commit


class _FailedCommitSession:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        raise self.error

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


def _claimed() -> _ClaimedTurn:
    return _ClaimedTurn(
        task_lease=MagicMock(runner_id="runner", run_id="run-turn"),
        status=TaskStatus.RUNNING,
        updated_at=None,
        before_message_id=None,
        task_source="sdk",
        run_id="run-turn",
    )


def _unreachable_reconciliation(monkeypatch, first_session=None) -> None:
    """Reconciliation cannot read: every Session after ``first_session`` fails."""
    calls = 0

    def session_factory():
        nonlocal calls
        calls += 1
        if first_session is not None and calls == 1:
            return first_session
        raise RuntimeError(f"reconciliation database unavailable {SENTINEL}")

    monkeypatch.setattr(database_module, "_SessionLocal", session_factory)
    monkeypatch.setattr(orchestrator.time, "sleep", MagicMock())


def test_turn_commit_is_reported_at_begin_turn(monkeypatch, caplog, metric_reader):
    commit_error = ConnectionError(f"commit acknowledgement lost {SENTINEL}")
    _unreachable_reconciliation(monkeypatch, _FailedCommitSession(commit_error))
    monkeypatch.setattr(
        orchestrator, "_claim_turn_no_commit", MagicMock(return_value=_claimed())
    )

    with pytest.raises(TaskTurnCommitOutcomeUnknown):
        _begin_turn_atomic_sync(
            123,
            456,
            payload=TaskTurnPayload(f"accepted {SENTINEL}", turn_id="turn-begin"),
            kind=TurnKind.CREATE,
        )

    assert_one(
        caplog,
        metric_reader,
        kind="turn_commit",
        task=123,
        run_id="run-turn",
        ref="turn-begin",
        storage="unknown",
    )


def test_turn_commit_is_reported_at_a_claimed_commit(
    monkeypatch, caplog, metric_reader
):
    commit_error = ConnectionError("commit acknowledgement lost")
    _unreachable_reconciliation(monkeypatch)

    with pytest.raises(TaskTurnCommitOutcomeUnknown):
        commit_claimed_turn_or_reconcile(
            _FailedCommitSession(commit_error),  # type: ignore[arg-type]
            task_id=124,
            task_owner_user_id=456,
            payload=TaskTurnPayload("accepted", turn_id="turn-claimed"),
            claimed=_claimed(),
        )

    assert_one(
        caplog,
        metric_reader,
        kind="turn_commit",
        task=124,
        run_id="run-turn",
        ref="turn-claimed",
        storage="unknown",
    )


def test_a_reconciled_turn_commit_is_not_reported(monkeypatch, caplog, metric_reader):
    monkeypatch.setattr(
        orchestrator,
        "_reconcile_claimed_turn_after_commit_ack_failure",
        lambda **_kwargs: True,
    )

    commit_claimed_turn_or_reconcile(
        _FailedCommitSession(ConnectionError("lost")),  # type: ignore[arg-type]
        task_id=125,
        task_owner_user_id=456,
        payload=TaskTurnPayload("accepted", turn_id="turn-reconciled"),
        claimed=_claimed(),
    )

    assert_none(caplog, metric_reader)


# ------------------------------------------------------------ command_commit


def _lose_every_commit(monkeypatch, db_session) -> None:
    original = Session.commit

    def lose(session: Session) -> None:
        if session is db_session:
            return original(session)
        raise _commit_error()

    monkeypatch.setattr(Session, "commit", lose)
    monkeypatch.setattr(task_command_execution.time, "sleep", MagicMock())


def test_command_commit_is_reported_for_a_live_delivery(
    db_session, monkeypatch, caplog, metric_reader
):
    owner = _owner_user(db_session, "unknown-live-owner")
    task = _owner_task(db_session, owner.id, status=TaskStatus.RUNNING)
    _lose_every_commit(monkeypatch, db_session)

    with pytest.raises(task_command_execution._TaskCommandCommitOutcomeUnknown):
        task_command_execution._claim_user_message_delivery_isolated(
            task_id=int(task.id),
            task_owner_user_id=int(owner.id),
            content=f"Apply {SENTINEL}",
            attachments=None,
            file_ids=[],
            turn_id="live-unknown-turn",
            expected_run_id="live-run",
            expected_status=TaskStatus.RUNNING,
        )

    assert_one(
        caplog,
        metric_reader,
        kind="command_commit",
        task=int(task.id),
        run_id="live-run",
        ref="live-unknown-turn",
        storage="unknown",
    )


def test_command_commit_is_reported_for_a_created_task_turn(
    db_session, monkeypatch, caplog, metric_reader
):
    owner = _owner_user(db_session, "unknown-created-owner")
    _lose_every_commit(monkeypatch, db_session)

    with pytest.raises(task_command_execution._TaskCommandCommitOutcomeUnknown):
        task_command_execution._prepare_task_message_sync(
            requested_task_id=987654,
            actor_user_id=int(owner.id),
            actor_is_admin=False,
            user_message=f"unknown acceptance {SENTINEL}",
            raw_context={},
            raw_files=[],
            client_message_id="created-unknown-turn",
            turn_id="created-unknown-turn",
            durable_attempt_count=1,
            durable_target_run_id=None,
            pause_accepted=False,
        )

    found = records(caplog)
    assert len(found) == 1
    data = fields(found[0].getMessage())
    assert data["kind"] == "command_commit"
    assert data["ref"] == "created-unknown-turn"
    assert data["storage"] == "unknown"
    assert data["run_id"] != "-"
    assert counts(metric_reader) == {
        key(operation="command_commit", storage="unknown"): 1
    }
    assert_no_leak(caplog)


def _fail_the_acceptance_flush(monkeypatch, db_session, arm_after: str) -> list[str]:
    """Fail the explicit flush right before the acceptance COMMIT.

    ``arm_after`` names the step that runs just before that flush; the next
    flush of a command Session (not the test's) raises. Nothing is written.
    """
    flushed: list[str] = []
    armed = {"on": False}
    original_flush = Session.flush

    def flush(session: Session, *args: Any, **kwargs: Any) -> None:
        if armed["on"] and session is not db_session:
            armed["on"] = False
            flushed.append("failed")
            raise OperationalError("INSERT", {}, Exception(f"flush failed {SENTINEL}"))
        return original_flush(session, *args, **kwargs)

    if arm_after == "claim_snapshot":
        step = task_command_execution._snapshot_user_message_delivery

        def armed_step(*args: Any, **kwargs: Any) -> Any:
            result = step(*args, **kwargs)
            armed["on"] = True
            return result

        monkeypatch.setattr(
            task_command_execution, "_snapshot_user_message_delivery", armed_step
        )
    else:
        assert arm_after == "created_turn_routing"
        orchestrator_cls = orchestrator.TaskTurnOrchestrator
        claim = orchestrator_cls.claim_created_turn_no_commit
        load = task_command_execution._load_task_command_routing_snapshot
        claimed = {"done": False}

        def armed_claim(*args: Any, **kwargs: Any) -> Any:
            result = claim(*args, **kwargs)
            claimed["done"] = True
            return result

        def armed_load(*args: Any, **kwargs: Any) -> Any:
            result = load(*args, **kwargs)
            if claimed["done"]:
                armed["on"] = True
            return result

        monkeypatch.setattr(
            orchestrator_cls, "claim_created_turn_no_commit", staticmethod(armed_claim)
        )
        monkeypatch.setattr(
            task_command_execution, "_load_task_command_routing_snapshot", armed_load
        )
    monkeypatch.setattr(Session, "flush", flush)
    monkeypatch.setattr(task_command_execution.time, "sleep", MagicMock())
    return flushed


def test_a_failed_live_delivery_flush_is_not_an_unknown_commit(
    db_session, monkeypatch, caplog, metric_reader
):
    """A flush that fails wrote nothing: the raised error is unchanged, but
    no unknown commit outcome is reported."""
    owner = _owner_user(db_session, "flush-live-owner")
    task = _owner_task(db_session, owner.id, status=TaskStatus.RUNNING)
    flushed = _fail_the_acceptance_flush(monkeypatch, db_session, "claim_snapshot")

    with pytest.raises(task_command_execution._TaskCommandCommitOutcomeUnknown):
        task_command_execution._claim_user_message_delivery_isolated(
            task_id=int(task.id),
            task_owner_user_id=int(owner.id),
            content=f"Apply {SENTINEL}",
            attachments=None,
            file_ids=[],
            turn_id="live-flush-turn",
            expected_run_id="live-run",
            expected_status=TaskStatus.RUNNING,
        )

    assert flushed == ["failed"]
    assert_none(caplog, metric_reader)


def test_a_failed_created_turn_flush_is_not_an_unknown_commit(
    db_session, monkeypatch, caplog, metric_reader
):
    owner = _owner_user(db_session, "flush-created-owner")
    flushed = _fail_the_acceptance_flush(
        monkeypatch, db_session, "created_turn_routing"
    )

    with pytest.raises(task_command_execution._TaskCommandCommitOutcomeUnknown):
        task_command_execution._prepare_task_message_sync(
            requested_task_id=987655,
            actor_user_id=int(owner.id),
            actor_is_admin=False,
            user_message=f"flush failure {SENTINEL}",
            raw_context={},
            raw_files=[],
            client_message_id="created-flush-turn",
            turn_id="created-flush-turn",
            durable_attempt_count=1,
            durable_target_run_id=None,
            pause_accepted=False,
        )

    assert flushed == ["failed"]
    assert_none(caplog, metric_reader)


# ------------------------------------------------------------------ finalize


def _commit_error() -> OperationalError:
    # psycopg2 reports a lost COMMIT acknowledgement as a dropped connection.
    return OperationalError(
        "COMMIT",
        {},
        Exception(f"commit acknowledgement lost {SENTINEL}"),
        connection_invalidated=True,
    )


def _deadlock() -> OperationalError:
    # PostgreSQL refused the COMMIT (40P01): a definite non-apply.
    return OperationalError("COMMIT", {}, Exception(f"deadlock detected {SENTINEL}"))


def _database_locked() -> OperationalError:
    # SQLite refused the COMMIT after its busy timeout: a definite non-apply.
    return OperationalError("COMMIT", {}, Exception("database is locked"))


def _integrity_error() -> IntegrityError:
    return IntegrityError("COMMIT", {}, Exception(f"deferred constraint {SENTINEL}"))


# Commit errors that cannot hide an applied COMMIT: the server refused it.
REFUSED_COMMITS = {
    "integrity": _integrity_error,
    "deadlock": _deadlock,
    "database_locked": _database_locked,
}


def _failing_db(error: Exception) -> MagicMock:
    db = MagicMock()
    db.commit.side_effect = error
    return db


FACT_WITNESS = FactWitness(
    task_id=31, scope_id="root", idempotency_key="result", event_id="event-1"
)
LEGACY_WITNESS = _ChatMessageWitness(
    task_id=31,
    message_id=5,
    role="assistant",
    message_type="final",
    content=f"answer {SENTINEL}",
)


def test_finalize_ack_reconciled_is_reported_at_info(
    monkeypatch, caplog, metric_reader
):
    monkeypatch.setattr(
        task_execution, "_finalization_witness_committed", lambda w: True
    )

    _commit_finalization_or_reconcile(
        _failing_db(_commit_error()),
        task_id=31,
        witness=FACT_WITNESS,
        run_id="run-f",
        storage="v2",
    )

    assert_one(
        caplog,
        metric_reader,
        kind="finalize_ack_reconciled",
        task=31,
        run_id="run-f",
        ref=None,
        storage="v2",
        level=logging.INFO,
    )


@pytest.mark.parametrize(
    "witness, storage",
    [(FACT_WITNESS, "v2"), (LEGACY_WITNESS, "v1")],
    ids=["fact", "legacy"],
)
def test_finalize_ack_unresolved_when_the_read_back_fails(
    monkeypatch, caplog, metric_reader, witness, storage
):
    def read_back_fails(_witness):
        raise OSError(f"read-back failed {SENTINEL}")

    monkeypatch.setattr(
        task_execution, "_finalization_witness_committed", read_back_fails
    )
    error = _commit_error()

    with pytest.raises(OperationalError) as raised:
        _commit_finalization_or_reconcile(
            _failing_db(error),
            task_id=31,
            witness=witness,
            run_id="run-f",
            storage=storage,
        )

    assert raised.value is error
    assert_one(
        caplog,
        metric_reader,
        kind="finalize_ack_unresolved",
        task=31,
        run_id="run-f",
        ref=None,
        storage=storage,
    )


def test_finalize_ack_unresolved_without_a_witness(monkeypatch, caplog, metric_reader):
    error = _commit_error()

    with pytest.raises(OperationalError) as raised:
        _commit_finalization_or_reconcile(_failing_db(error), task_id=31, witness=None)

    assert raised.value is error
    assert_one(
        caplog,
        metric_reader,
        kind="finalize_ack_unresolved",
        task=31,
        run_id=None,
        ref=None,
        storage="unknown",
    )


def test_finalize_commit_proven_absent_is_a_plain_failure(
    monkeypatch, caplog, metric_reader
):
    """The read-back completed and found nothing: the commit did not apply."""
    monkeypatch.setattr(
        task_execution, "_finalization_witness_committed", lambda w: False
    )
    error = _commit_error()

    with pytest.raises(OperationalError) as raised:
        _commit_finalization_or_reconcile(
            _failing_db(error), task_id=31, witness=FACT_WITNESS, run_id="run-f"
        )

    assert raised.value is error
    assert_none(caplog, metric_reader)


@pytest.mark.parametrize("refused", REFUSED_COMMITS)
@pytest.mark.parametrize("witness", [None, FACT_WITNESS], ids=["no-witness", "fact"])
def test_finalize_commit_error_that_cannot_hide_an_applied_commit_is_not_unknown(
    monkeypatch, caplog, metric_reader, witness, refused
):
    """Only a dropped connection can lose the acknowledgement of an applied
    COMMIT; an OperationalError the server raised (a deadlock, SQLite's
    "database is locked") is a refusal, not a lost acknowledgement."""

    def read_back_fails(_witness):
        raise OSError("read-back failed")

    monkeypatch.setattr(
        task_execution, "_finalization_witness_committed", read_back_fails
    )
    error = REFUSED_COMMITS[refused]()

    with pytest.raises(type(error)) as raised:
        _commit_finalization_or_reconcile(
            _failing_db(error), task_id=31, witness=witness, run_id="run-f"
        )

    assert raised.value is error
    assert_none(caplog, metric_reader)


def test_finalize_ack_unresolved_on_a_closed_connection(caplog, metric_reader):
    """``InterfaceError`` is the DBAPI's connection-level class too."""
    error = InterfaceError("COMMIT", {}, Exception("connection already closed"))

    with pytest.raises(InterfaceError):
        _commit_finalization_or_reconcile(
            _failing_db(error), task_id=31, witness=None, run_id="run-f", storage="v1"
        )

    assert_one(
        caplog,
        metric_reader,
        kind="finalize_ack_unresolved",
        task=31,
        run_id="run-f",
        ref=None,
        storage="v1",
    )


def test_finalize_ack_unresolved_through_a_wrapped_cause(caplog, metric_reader):
    wrapped = RuntimeError("finalize failed")
    wrapped.__cause__ = _commit_error()

    with pytest.raises(RuntimeError):
        _commit_finalization_or_reconcile(
            _failing_db(wrapped), task_id=31, witness=None, run_id="run-f"
        )

    assert [fields(line)["kind"] for line in lines(caplog)] == [
        "finalize_ack_unresolved"
    ]


def test_finalize_that_commits_normally_reports_nothing(caplog, metric_reader):
    _commit_finalization_or_reconcile(MagicMock(), task_id=31, witness=FACT_WITNESS)
    assert_none(caplog, metric_reader)


SUCCESS = {"success": True, "status": "completed", "output": "done"}


@pytest.fixture
def commit_fate(monkeypatch):
    """Lose the COMMIT of the finalize transaction, applied or not.

    The finalize transaction is the one that just staged the result fact.
    """
    state: dict[str, Any] = {"armed": False, "apply": True, "error": _commit_error}
    stage = task_execution.stage_result_fact_no_commit
    original = Session.commit

    def staged(*args, **kwargs):
        witness = stage(*args, **kwargs)
        state["armed"] = True
        return witness

    def commit(session: Session) -> None:
        if not state["armed"]:
            return original(session)
        state["armed"] = False
        if state["apply"]:
            original(session)
        raise state["error"]()

    monkeypatch.setattr(task_execution, "stage_result_fact_no_commit", staged)
    monkeypatch.setattr(Session, "commit", commit)
    return state


def test_a_lost_finalize_ack_the_read_back_confirms_is_reconciled(
    canonical, commit_fate, caplog, metric_reader
):
    factory, tid = canonical
    lease = _start_run(factory, tid)

    finalized = _finalize(factory, tid, lease, SUCCESS)

    assert finalized.terminal_state_committed
    assert_one(
        caplog,
        metric_reader,
        kind="finalize_ack_reconciled",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v2",
        level=logging.INFO,
    )


def test_a_lost_finalize_ack_with_no_read_back_is_unresolved(
    canonical, commit_fate, monkeypatch, caplog, metric_reader
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    commit_fate["apply"] = False

    def read_back_fails(_witness):
        raise CheckpointUnavailableError(f"read-back unavailable {SENTINEL}")

    monkeypatch.setattr(
        task_execution, "_finalization_witness_committed", read_back_fails
    )

    with pytest.raises(OperationalError):
        _finalize(factory, tid, lease, SUCCESS)

    assert_one(
        caplog,
        metric_reader,
        kind="finalize_ack_unresolved",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v2",
    )


def test_a_finalize_commit_the_read_back_proves_absent_is_not_unknown(
    canonical, commit_fate, caplog, metric_reader
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    commit_fate["apply"] = False
    assert (
        _finalization_witness_committed
        is task_execution._finalization_witness_committed
    )

    with pytest.raises(OperationalError):
        _finalize(factory, tid, lease, SUCCESS)

    assert_none(caplog, metric_reader)


# ------------------------------------------------- resume_finalize_unreconciled


@pytest.mark.parametrize("apply", [True, False], ids=["applied", "not_applied"])
def test_resumed_finalize_commit_failure_is_reported(
    canonical, commit_fate, caplog, metric_reader, apply
):
    """No reconciliation exists for this commit: whether or not it applied,
    a failing COMMIT is an unreconciled outcome."""
    factory, tid = canonical
    lease = _start_run(factory, tid)
    commit_fate["apply"] = apply
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)

    with pytest.raises(OperationalError):
        _finalize_resumed_task(
            tid,
            status="completed",
            success=True,
            output="done",
            task_owner_user_id=user_id,
            result=dict(SUCCESS),
            task_lease=lease,
            prepared_outputs=NO_OUTPUTS,
        )

    assert_one(
        caplog,
        metric_reader,
        kind="resume_finalize_unreconciled",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v2",
    )


@pytest.mark.parametrize("refused", REFUSED_COMMITS)
def test_resumed_finalize_commit_refusal_is_not_unknown(
    canonical, commit_fate, caplog, metric_reader, refused
):
    """A COMMIT the server refused did not apply: a plain failure."""
    factory, tid = canonical
    lease = _start_run(factory, tid)
    commit_fate["apply"] = False
    commit_fate["error"] = REFUSED_COMMITS[refused]
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)

    with pytest.raises((IntegrityError, OperationalError)):
        _finalize_resumed_task(
            tid,
            status="completed",
            success=True,
            output="done",
            task_owner_user_id=user_id,
            result=dict(SUCCESS),
            task_lease=lease,
            prepared_outputs=NO_OUTPUTS,
        )

    assert_none(caplog, metric_reader)


def test_resumed_finalize_that_commits_reports_nothing(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)

    finalized = _finalize_resumed_task(
        tid,
        status="completed",
        success=True,
        output="done",
        task_owner_user_id=user_id,
        result=dict(SUCCESS),
        task_lease=lease,
        prepared_outputs=NO_OUTPUTS,
    )

    assert finalized["lease_released"]
    assert_none(caplog, metric_reader)


# --------------------------------------------------------- settlement_deferred


@pytest.mark.asyncio
async def test_a_deferred_run_result_is_reported_once(canonical, caplog, metric_reader):
    factory, tid = canonical
    lease = _start_run(factory, tid)

    async def execute(**_kwargs):
        raise InterruptionSettlementDeferred(f"checkpoint not resolvable {SENTINEL}")

    await _run_scheduled_turn(tid, lease, execute)

    assert_one(
        caplog,
        metric_reader,
        kind="settlement_deferred",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="unknown",
    )
    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_a_deferral_raised_by_the_settlement_is_reported_once(
    canonical, monkeypatch, caplog, metric_reader
):
    """The run failed and its own settlement could not decide: the one
    occurrence is reported where that settlement's error is handled."""
    factory, tid = canonical
    lease = _start_run(factory, tid)

    async def execute(**_kwargs):
        raise RuntimeError("tool crashed")

    def deferred(*_args, **_kwargs):
        raise InterruptionSettlementDeferred(f"not decidable now {SENTINEL}")

    monkeypatch.setattr(orchestrator, "settle_task_lease_isolated", deferred)

    await _run_scheduled_turn(tid, lease, execute)

    assert_one(
        caplog,
        metric_reader,
        kind="settlement_deferred",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="unknown",
    )


def test_the_raising_settlement_itself_reports_nothing(
    canonical, monkeypatch, caplog, metric_reader
):
    """Only the handlers report, so a deferral never counts twice."""
    factory, tid = canonical
    lease = _start_run(factory, tid)

    def unavailable(*_args, **_kwargs):
        raise CheckpointUnavailableError("recovery read failed")

    from tests.web.services.test_persistence_failure_pause import (
        _checkpoint,
        _persistence_interruption,
    )

    _checkpoint(factory, tid, lease.run_id)
    monkeypatch.setattr(event_recovery, "read_event_checkpoint", unavailable)
    with pytest.raises(InterruptionSettlementDeferred):
        _settle(lease, interruption=_persistence_interruption())

    assert_none(caplog, metric_reader)


@pytest.mark.asyncio
async def test_a_deferred_resume_is_reported_once(caplog, metric_reader):
    settle = MagicMock(return_value=True)

    await _resume_raising(
        InterruptionSettlementDeferred(f"checkpoint not resolvable {SENTINEL}"),
        settle,
    )

    settle.assert_not_called()
    assert_one(
        caplog,
        metric_reader,
        kind="settlement_deferred",
        task=42,
        run_id="run-a",
        ref=None,
        storage="unknown",
    )


@pytest.mark.asyncio
async def test_a_resume_settlement_deferral_is_reported_once(caplog, metric_reader):
    settle = MagicMock(
        side_effect=InterruptionSettlementDeferred(f"not decidable now {SENTINEL}")
    )

    await _resume_raising(RuntimeError("tool crashed"), settle)

    settle.assert_called_once()
    assert_one(
        caplog,
        metric_reader,
        kind="settlement_deferred",
        task=42,
        run_id="run-a",
        ref=None,
        storage="unknown",
    )


@pytest.mark.asyncio
async def test_other_settlement_errors_are_not_deferrals(caplog, metric_reader):
    settle = MagicMock(side_effect=RuntimeError("settlement failed"))

    await _resume_raising(RuntimeError("tool crashed"), settle)

    assert_none(caplog, metric_reader)
