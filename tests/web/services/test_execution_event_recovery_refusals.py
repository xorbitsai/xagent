"""Recovery-refusal signals for V2 execution-event reads.

Every refusal point of the recovery readers reports one structured log line
and one counter increment where it is first raised. Refusals carry only bounded
names and identifiers; no payload content or exception text reaches a signal.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from tests.web.services.test_execution_event_signals import (
    SENTINEL,
    exported,
    fields,
    key,
    metric_reader,
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
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointUnavailableError,
    TraceCheckpointStore,
    UnknownToolEffectError,
)
from xagent.core.agent.trace import Tracer
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services import execution_event_signals
from xagent.web.services import task_execution_event_recovery as recovery
from xagent.web.services.execution_event_signals import report_recovery_refusal
from xagent.web.services.task_execution_event_recovery import (
    check_recovery_owner,
    read_committed_tool_outcome,
    read_event_checkpoint,
)
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.services.task_lease_service import (
    acquire_task_lease,
    bind_task_lease_context,
)
from xagent.web.tracing import ExecutionEventTraceAdapter

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
metric_reader = metric_reader

COUNTER = "xagent.execution_event.recovery.refusals"
PREFIX = "execution_event_recovery_refused "


def tracer_for(task_id, scope=None):
    adapter = ExecutionEventTraceAdapter(task_id, build_id=scope)
    tracer = Tracer()
    tracer.handlers = [adapter]
    tracer.event_writer = adapter.commit_event
    return tracer


def records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith(PREFIX)]


def refusal_counts(reader) -> dict[tuple[tuple[str, str], ...], int | float]:
    return exported(reader, COUNTER)


def call(attempt: str = "attempt", **extra) -> dict:
    return {
        "id": "call",
        "name": "calculator",
        "args": {"expression": SENTINEL},
        "assistant_message_id": "batch",
        "tool_attempt_id": attempt,
        **extra,
    }


def assert_one(
    caplog,
    metric_reader,
    *,
    reader: str,
    outcome: str,
    level: int,
    task_id: int,
    run_id: str | None = None,
    tool_attempt_id: str | None = None,
) -> None:
    found = records(caplog)
    assert len(found) == 1, [r.getMessage() for r in found]
    record = found[0]
    assert record.levelno == level
    assert record.exc_info is None
    line = record.getMessage()
    expected = (
        f"{PREFIX}task_id={task_id} run_id={run_id or '-'} storage=v2 "
        f"component=execution-events reader={reader} outcome={outcome}"
    )
    if tool_attempt_id:
        expected += f" tool_attempt_id={tool_attempt_id}"
    assert line == expected
    assert SENTINEL not in line
    assert fields(line)["reader"] == reader
    assert refusal_counts(metric_reader) == {
        key(operation=reader, outcome=outcome, storage="v2"): 1
    }


async def save_checkpoint(
    tid, *, state=None, pattern="ReActPattern", scope=None, execution_id="execution"
):
    await TraceCheckpointStore(tracer_for(tid, scope)).save(
        {
            "execution_id": execution_id,
            "context": {"messages": []},
            "pattern": pattern,
            "pattern_state": state or {},
            "label": SENTINEL,
        }
    )


# ---------------------------------------------------------------- owner_check


def test_owner_check_lease_bound_to_another_task(canonical, caplog, metric_reader):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    with bind_task_lease_context(lease), factory() as db:
        with pytest.raises(CheckpointAccessRefusedError) as raised:
            check_recovery_owner(db, tid + 1)
    assert raised.value.reason == "lease_mismatch"
    assert_one(
        caplog,
        metric_reader,
        reader="owner_check",
        outcome="lease_mismatch",
        level=logging.INFO,
        task_id=tid + 1,
        run_id=lease.run_id,
    )


def test_owner_check_lease_lost_to_another_owner(canonical, caplog, metric_reader):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
        db.get(Task, tid).runner_id = "replacement-owner"
        db.commit()
    with bind_task_lease_context(lease), factory() as db:
        with pytest.raises(CheckpointAccessRefusedError) as raised:
            check_recovery_owner(db, tid)
    assert raised.value.reason == "lease_mismatch"
    assert_one(
        caplog,
        metric_reader,
        reader="owner_check",
        outcome="lease_mismatch",
        level=logging.INFO,
        task_id=tid,
        run_id=lease.run_id,
    )


def test_owner_check_task_missing_without_a_lease(canonical, caplog, metric_reader):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    with factory() as db:
        with pytest.raises(CheckpointAccessRefusedError):
            check_recovery_owner(db, tid + 1000)
    assert_one(
        caplog,
        metric_reader,
        reader="owner_check",
        outcome="task_missing",
        level=logging.INFO,
        task_id=tid + 1000,
    )


@pytest.mark.parametrize("status", [TaskStatus.COMPLETED, TaskStatus.FAILED])
@pytest.mark.parametrize("leased", [False, True])
def test_owner_check_task_ended(canonical, caplog, metric_reader, status, leased):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    lease = None
    with factory() as db:
        if leased:
            lease = acquire_task_lease(db, tid, new_run=True)
        db.get(Task, tid).status = status
        db.commit()
    with factory() as db:
        if lease is None:
            with pytest.raises(CheckpointAccessRefusedError):
                check_recovery_owner(db, tid)
        else:
            with bind_task_lease_context(lease):
                with pytest.raises(CheckpointAccessRefusedError):
                    check_recovery_owner(db, tid)
    assert_one(
        caplog,
        metric_reader,
        reader="owner_check",
        outcome="task_ended",
        level=logging.INFO,
        task_id=tid,
        run_id=lease.run_id if lease is not None else None,
    )


def test_owner_check_success_reports_nothing(canonical, caplog, metric_reader):
    factory, tid = canonical
    caplog.set_level(logging.DEBUG)
    with factory() as db:
        check_recovery_owner(db, tid)
        lease = acquire_task_lease(db, tid, new_run=True)
    with bind_task_lease_context(lease), factory() as db:
        check_recovery_owner(db, tid)
    assert records(caplog) == []
    assert refusal_counts(metric_reader) == {}


# ----------------------------------------------------------------- checkpoint


def read(db, tid, **overrides):
    arguments = {
        "task_id": tid,
        "scope_id": "root",
        "execution_id": "execution",
        "run_id": None,
    }
    arguments.update(overrides)
    return read_event_checkpoint(db, **arguments)


def test_checkpoint_task_missing(canonical, caplog, metric_reader):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    with factory() as db:
        with pytest.raises(CheckpointUnavailableError):
            read(db, tid + 1000, run_id="run-x")
    assert_one(
        caplog,
        metric_reader,
        reader="checkpoint",
        outcome="task_missing",
        level=logging.INFO,
        task_id=tid + 1000,
        run_id="run-x",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed"])
async def test_checkpoint_settled_execution_is_task_ended(
    canonical, caplog, metric_reader, status
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    await save_checkpoint(tid)
    with factory() as db:
        append_fact_no_commit(
            db,
            task_id=tid,
            kind="execution_settled",
            key="settled",
            payload={"status": status, "result": {"note": SENTINEL}},
        )
        db.commit()
    with factory() as db:
        with pytest.raises(CheckpointAccessRefusedError):
            read(db, tid)
    assert_one(
        caplog,
        metric_reader,
        reader="checkpoint",
        outcome="task_ended",
        level=logging.INFO,
        task_id=tid,
    )


@pytest.mark.asyncio
async def test_checkpoint_unsettled_run_is_not_refused(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    caplog.set_level(logging.DEBUG)
    await save_checkpoint(tid)
    with factory() as db:
        append_fact_no_commit(
            db,
            task_id=tid,
            kind="execution_settled",
            key="settled",
            payload={"status": "paused", "result": {"note": SENTINEL}},
        )
        db.commit()
    with factory() as db:
        assert read(db, tid) is not None
        check_recovery_owner(db, tid)
    assert records(caplog) == []
    assert refusal_counts(metric_reader) == {}


@pytest.mark.asyncio
async def test_checkpoint_page_limit(canonical, caplog, metric_reader, monkeypatch):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    pages: list[int] = []

    def other_execution_page(self, statement, *args, **kwargs):
        sequence = 100_000 - len(pages)
        pages.append(sequence)
        return [
            SimpleNamespace(
                sequence=sequence,
                run_id=None,
                payload={"data": {"execution_id": "other", "note": SENTINEL}},
            )
        ]

    monkeypatch.setattr(Session, "scalars", other_execution_page)
    with factory() as db:
        with pytest.raises(CheckpointUnavailableError, match="page limit"):
            read(db, tid, run_id="run-p")
    assert len(pages) == 51
    assert_one(
        caplog,
        metric_reader,
        reader="checkpoint",
        outcome="page_limit",
        level=logging.WARNING,
        task_id=tid,
        run_id="run-p",
    )


def set_payload_version(db, tid, version=99):
    event = db.scalars(
        sa.select(TaskExecutionEvent).where(
            TaskExecutionEvent.task_id == tid,
            TaskExecutionEvent.kind == "recovery_state",
        )
    ).one()
    event.payload_version = version
    db.commit()


@pytest.mark.asyncio
async def test_checkpoint_unreadable_envelope_is_corrupt(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    await save_checkpoint(tid)
    with factory() as db:
        set_payload_version(db, tid)
    with factory() as db:
        with pytest.raises(CheckpointCorruptError):
            read(db, tid, run_id=None)
    assert_one(
        caplog,
        metric_reader,
        reader="checkpoint",
        outcome="corrupt",
        level=logging.ERROR,
        task_id=tid,
    )


@pytest.mark.asyncio
async def test_checkpoint_unreadable_settlement_is_corrupt(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    await save_checkpoint(tid)
    with factory() as db:
        append_fact_no_commit(
            db,
            task_id=tid,
            kind="execution_settled",
            key="settled",
            payload={"status": SENTINEL},
        )
        db.commit()
    with factory() as db:
        with pytest.raises(CheckpointCorruptError, match="settlement"):
            read(db, tid)
    assert_one(
        caplog,
        metric_reader,
        reader="checkpoint",
        outcome="corrupt",
        level=logging.ERROR,
        task_id=tid,
    )


BAD_CALL = {"name": "calculator", "args": {"expression": SENTINEL}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pattern", "state"),
    [
        ("ReActPattern", {"pending_tool_calls": "not-a-list"}),
        ("ReActPattern", {"pending_tool_calls": ["not-a-dict"]}),
        ("ReActPattern", {"pending_tool_calls": [{"args": SENTINEL}]}),
        ("ReActPattern", {"pending_tool_calls": [BAD_CALL]}),
        ("ReActPattern", {"active_step_pattern_states": "not-a-dict"}),
        ("AutoPattern", "not-a-dict"),
        ("AutoPattern", {"decision": "not-a-dict"}),
        ("AutoPattern", {"decision": {}, "react_state": {"note": SENTINEL}}),
    ],
    ids=[
        "pending-not-list",
        "pending-call-not-dict",
        "pending-call-no-name",
        "pending-call-no-attempt-identity",
        "step-states-not-dict",
        "auto-state-not-dict",
        "auto-decision-not-dict",
        "auto-unrecognized-decision",
    ],
)
async def test_checkpoint_unreadable_pattern_state_is_corrupt(
    canonical, caplog, metric_reader, pattern, state
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    await save_checkpoint(tid, state=state, pattern=pattern)
    with factory() as db:
        with pytest.raises(CheckpointCorruptError):
            read(db, tid)
    assert_one(
        caplog,
        metric_reader,
        reader="checkpoint",
        outcome="corrupt",
        level=logging.ERROR,
        task_id=tid,
    )


@pytest.mark.asyncio
async def test_checkpoint_success_reports_nothing(canonical, caplog, metric_reader):
    factory, tid = canonical
    caplog.set_level(logging.DEBUG)
    with factory() as db:
        assert read(db, tid) is None
    await save_checkpoint(tid, state={"pending_tool_calls": [call()]})
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer_for(tid)))
    await runtime.on_tool_start(tool_call=call())
    await runtime.on_tool_end(tool_call=call(), result={"value": SENTINEL})
    with factory() as db:
        data = read(db, tid)
        assert data is not None
    assert (
        await TraceCheckpointStore(tracer_for(tid)).load_latest_checkpoint("execution")
        is not None
    )
    assert records(caplog) == []
    assert refusal_counts(metric_reader) == {}


# --------------------------------------------------------------- tool_outcome


async def start_attempt(tid, attempt="attempt", **extra):
    runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer_for(tid)))
    await runtime.on_tool_start(tool_call=call(attempt, **extra))
    return runtime


def outcome(db, tid, tool_call, **overrides):
    return read_committed_tool_outcome(
        db, task_id=tid, scope_id="root", tool_call=tool_call, **overrides
    )


@pytest.mark.asyncio
async def test_tool_outcome_started_without_result_is_unknown_effect(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    await start_attempt(tid, "attempt-a")
    with factory() as db:
        with pytest.raises(UnknownToolEffectError, match="replay is unsafe"):
            outcome(db, tid, call("attempt-a"))
    assert_one(
        caplog,
        metric_reader,
        reader="tool_outcome",
        outcome="unknown_tool_effect",
        level=logging.WARNING,
        task_id=tid,
        tool_attempt_id="attempt-a",
    )


@pytest.mark.asyncio
async def test_tool_outcome_interrupted_is_unknown_effect(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    runtime = await start_attempt(tid, "attempt-i")
    await runtime.on_tool_end(tool_call=call("attempt-i"), result=SENTINEL)
    with factory() as db:
        end = db.scalars(
            sa.select(TaskExecutionEvent).where(
                TaskExecutionEvent.kind == "tool_execution_end"
            )
        ).one()
        payload = dict(end.payload)
        payload["data"] = {**payload["data"], "interrupted": True}
        end.payload = payload
        db.commit()
    with factory() as db:
        with pytest.raises(UnknownToolEffectError, match="Interrupted"):
            outcome(db, tid, call("attempt-i"), run_id="run-i")
    assert_one(
        caplog,
        metric_reader,
        reader="tool_outcome",
        outcome="unknown_tool_effect",
        level=logging.WARNING,
        task_id=tid,
        run_id="run-i",
        tool_attempt_id="attempt-i",
    )


@pytest.mark.asyncio
async def test_tool_outcome_mismatched_facts_are_corrupt(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    await start_attempt(tid, "attempt-m")
    with factory() as db:
        with pytest.raises(CheckpointCorruptError, match="do not match"):
            outcome(db, tid, call("attempt-m", assistant_message_id="other-batch"))
    assert_one(
        caplog,
        metric_reader,
        reader="tool_outcome",
        outcome="corrupt",
        level=logging.ERROR,
        task_id=tid,
        tool_attempt_id="attempt-m",
    )


@pytest.mark.asyncio
async def test_tool_outcome_without_reusable_result_is_corrupt(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    runtime = await start_attempt(tid, "attempt-r")
    await runtime.on_tool_end(tool_call=call("attempt-r"), result=SENTINEL)
    with factory() as db:
        end = db.scalars(
            sa.select(TaskExecutionEvent).where(
                TaskExecutionEvent.kind == "tool_execution_end"
            )
        ).one()
        payload = dict(end.payload)
        payload["data"] = {
            k: v for k, v in payload["data"].items() if k != "result"
        } | {"note": SENTINEL}
        end.payload = payload
        db.commit()
    with factory() as db:
        with pytest.raises(CheckpointCorruptError, match="no reusable result"):
            outcome(db, tid, call("attempt-r"))
    assert_one(
        caplog,
        metric_reader,
        reader="tool_outcome",
        outcome="corrupt",
        level=logging.ERROR,
        task_id=tid,
        tool_attempt_id="attempt-r",
    )


@pytest.mark.asyncio
async def test_tool_outcome_success_reports_nothing(canonical, caplog, metric_reader):
    factory, tid = canonical
    caplog.set_level(logging.DEBUG)
    with factory() as db:
        assert outcome(db, tid, call("never-started")) is None
    runtime = await start_attempt(tid, "attempt-ok")
    await runtime.on_tool_end(tool_call=call("attempt-ok"), result={"v": SENTINEL})
    with factory() as db:
        committed = outcome(db, tid, call("attempt-ok"))
    assert committed is not None and committed["error"] is None
    assert records(caplog) == []
    assert refusal_counts(metric_reader) == {}


# ------------------------------------------------- nesting and propagation


@pytest.mark.asyncio
async def test_unknown_effect_found_by_checkpoint_read_is_reported_once(
    canonical, caplog, metric_reader
):
    """The checkpoint reader's tool preflight reports as tool_outcome, once."""
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    with bind_task_lease_context(lease):
        await save_checkpoint(tid, state={"pending_tool_calls": [call("attempt-n")]})
        await PatternRuntime(
            tracer=TraceCheckpointStore(tracer_for(tid))
        ).on_tool_start(tool_call=call("attempt-n"))
    with factory() as db:
        with pytest.raises(UnknownToolEffectError):
            read(db, tid, run_id=lease.run_id)
    assert_one(
        caplog,
        metric_reader,
        reader="tool_outcome",
        outcome="unknown_tool_effect",
        level=logging.WARNING,
        task_id=tid,
        run_id=lease.run_id,
        tool_attempt_id="attempt-n",
    )


@pytest.mark.asyncio
async def test_refusal_is_reported_once_through_the_checkpoint_store(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    await save_checkpoint(tid)
    with factory() as db:
        set_payload_version(db, tid)
    with pytest.raises(CheckpointCorruptError):
        await TraceCheckpointStore(tracer_for(tid)).load_latest_checkpoint("execution")
    assert_one(
        caplog,
        metric_reader,
        reader="checkpoint",
        outcome="corrupt",
        level=logging.ERROR,
        task_id=tid,
    )


@pytest.mark.asyncio
async def test_stale_lease_is_reported_once_per_read(canonical, caplog, metric_reader):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    store = TraceCheckpointStore(tracer_for(tid))
    with bind_task_lease_context(lease):
        await store.save({"execution_id": "execution", "context": {"messages": []}})
        await PatternRuntime(tracer=store).on_tool_start(tool_call=call("attempt-s"))
        await PatternRuntime(tracer=store).on_tool_end(
            tool_call=call("attempt-s"), result=4
        )
    with factory() as db:
        db.get(Task, tid).runner_id = "replacement-owner"
        db.commit()
    with bind_task_lease_context(lease):
        with pytest.raises(CheckpointAccessRefusedError):
            await store.load_latest_checkpoint("execution")
    assert_one(
        caplog,
        metric_reader,
        reader="owner_check",
        outcome="lease_mismatch",
        level=logging.INFO,
        task_id=tid,
        run_id=lease.run_id,
    )
    caplog.clear()
    with bind_task_lease_context(lease):
        with pytest.raises(CheckpointAccessRefusedError):
            await store.load_committed_tool_outcome(call("attempt-s"))
    assert len(records(caplog)) == 1
    assert f"run_id={lease.run_id} " in records(caplog)[0].getMessage()
    assert refusal_counts(metric_reader) == {
        key(operation="owner_check", outcome="lease_mismatch", storage="v2"): 2
    }


@pytest.mark.asyncio
async def test_lease_sweeper_classification_reports_the_refusal_once(
    canonical, caplog, metric_reader
):
    from xagent.web.services.task_lease_service import (
        CheckpointRecoveryVerdict,
        _resolve_event_checkpoint_recovery_with_data,
    )

    factory, tid = canonical
    caplog.set_level(logging.INFO)
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    with bind_task_lease_context(lease):
        await save_checkpoint(
            tid,
            state={"pending_tool_calls": [call("attempt-w")]},
            execution_id=str(tid),
        )
        await PatternRuntime(
            tracer=TraceCheckpointStore(tracer_for(tid))
        ).on_tool_start(tool_call=call("attempt-w"))
    with factory() as db:
        resolution = _resolve_event_checkpoint_recovery_with_data(
            db, task_id=tid, run_id=lease.run_id
        )
    assert resolution.verdict is CheckpointRecoveryVerdict.UNKNOWN_TOOL_EFFECT
    assert_one(
        caplog,
        metric_reader,
        reader="tool_outcome",
        outcome="unknown_tool_effect",
        level=logging.WARNING,
        task_id=tid,
        run_id=lease.run_id,
        tool_attempt_id="attempt-w",
    )


# ---------------------------------------------------------------- unavailable


def failing_read(*args, **kwargs):
    raise OperationalError(
        "SELECT", {"param": SENTINEL}, RuntimeError(f"offline {SENTINEL}")
    )


@pytest.mark.asyncio
async def test_database_failure_during_checkpoint_read_is_unavailable_once(
    canonical, caplog, metric_reader, monkeypatch
):
    _, tid = canonical
    caplog.set_level(logging.INFO)
    store = TraceCheckpointStore(tracer_for(tid))
    await save_checkpoint(tid)
    monkeypatch.setattr(recovery, "read_event_checkpoint", failing_read)
    with pytest.raises(CheckpointUnavailableError):
        await store.load_latest_checkpoint("execution")
    assert_one(
        caplog,
        metric_reader,
        reader="checkpoint",
        outcome="unavailable",
        level=logging.WARNING,
        task_id=tid,
    )


@pytest.mark.asyncio
async def test_database_failure_during_tool_outcome_read_is_unavailable_once(
    canonical, caplog, metric_reader, monkeypatch
):
    _, tid = canonical
    caplog.set_level(logging.INFO)
    store = TraceCheckpointStore(tracer_for(tid))
    monkeypatch.setattr(recovery, "read_committed_tool_outcome", failing_read)
    with pytest.raises(CheckpointUnavailableError):
        await store.load_committed_tool_outcome(call("attempt-u"))
    assert_one(
        caplog,
        metric_reader,
        reader="tool_outcome",
        outcome="unavailable",
        level=logging.WARNING,
        task_id=tid,
        tool_attempt_id="attempt-u",
    )


@pytest.mark.asyncio
async def test_adapter_tool_outcome_refusal_carries_the_bound_run_id(
    canonical, caplog, metric_reader
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    store = TraceCheckpointStore(tracer_for(tid))
    with bind_task_lease_context(lease):
        await PatternRuntime(tracer=store).on_tool_start(tool_call=call("attempt-r1"))
        with pytest.raises(UnknownToolEffectError):
            await store.load_committed_tool_outcome(call("attempt-r1"))
    assert_one(
        caplog,
        metric_reader,
        reader="tool_outcome",
        outcome="unknown_tool_effect",
        level=logging.WARNING,
        task_id=tid,
        run_id=lease.run_id,
        tool_attempt_id="attempt-r1",
    )


@pytest.mark.asyncio
async def test_adapter_unavailable_reports_carry_the_bound_run_id(
    canonical, caplog, metric_reader, monkeypatch
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    store = TraceCheckpointStore(tracer_for(tid))
    monkeypatch.setattr(recovery, "read_committed_tool_outcome", failing_read)
    with bind_task_lease_context(lease):
        with pytest.raises(CheckpointUnavailableError):
            await store.load_committed_tool_outcome(call("attempt-r2"))
    assert_one(
        caplog,
        metric_reader,
        reader="tool_outcome",
        outcome="unavailable",
        level=logging.WARNING,
        task_id=tid,
        run_id=lease.run_id,
        tool_attempt_id="attempt-r2",
    )


@pytest.mark.asyncio
async def test_adapter_checkpoint_unavailable_carries_the_bound_run_id(
    canonical, caplog, metric_reader, monkeypatch
):
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    store = TraceCheckpointStore(tracer_for(tid))
    monkeypatch.setattr(recovery, "read_event_checkpoint", failing_read)
    with bind_task_lease_context(lease):
        with pytest.raises(CheckpointUnavailableError):
            await store.load_latest_checkpoint("execution")
    assert_one(
        caplog,
        metric_reader,
        reader="checkpoint",
        outcome="unavailable",
        level=logging.WARNING,
        task_id=tid,
        run_id=lease.run_id,
    )


@pytest.mark.asyncio
async def test_widened_read_with_failing_probe_reports_one_refusal(
    canonical, caplog, metric_reader, monkeypatch
):
    """A failed staleness probe after a corrupt verdict is not a second refusal."""
    factory, tid = canonical
    caplog.set_level(logging.INFO)
    # An untagged (legacy) checkpoint: the lease-bound reader below widens.
    await save_checkpoint(tid, execution_id=str(tid))
    with factory() as db:
        set_payload_version(db, tid)
        lease = acquire_task_lease(db, tid, new_run=True)
    probes: list[int] = []
    original_scalar = Session.scalar

    def scalar(self, statement, *args, **kwargs):
        if "run_id IS NOT NULL" in str(statement):
            probes.append(1)
            if len(probes) > 1:
                raise OperationalError(
                    "SELECT", {"param": SENTINEL}, RuntimeError(f"offline {SENTINEL}")
                )
        return original_scalar(self, statement, *args, **kwargs)

    monkeypatch.setattr(Session, "scalar", scalar)
    store = TraceCheckpointStore(tracer_for(tid))
    with bind_task_lease_context(lease):
        with pytest.raises(CheckpointUnavailableError) as raised:
            await store.load_latest_checkpoint(str(tid))
    assert len(probes) == 2
    assert isinstance(raised.value.__cause__, OperationalError)
    assert_one(
        caplog,
        metric_reader,
        reader="checkpoint",
        outcome="corrupt",
        level=logging.ERROR,
        task_id=tid,
    )
    # No record, tracebacks included, renders the probe error's content.
    assert any(
        "run-tagged checkpoint probe failed: OperationalError" in r.getMessage()
        for r in caplog.records
    )
    assert SENTINEL not in caplog.text


# ------------------------------------------------------------ the reporter


def test_report_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("telemetry down")

    monkeypatch.setattr(execution_event_signals, "increment_counter", boom)
    report_recovery_refusal("checkpoint", "corrupt", task_id=1)

    monkeypatch.undo()
    monkeypatch.setattr(execution_event_signals.logger, "log", boom)
    report_recovery_refusal("checkpoint", "corrupt", task_id=1, tool_attempt_id="a")


def test_report_logs_the_documented_line(caplog, metric_reader):
    caplog.set_level(logging.INFO)
    report_recovery_refusal(
        "tool_outcome",
        "unknown_tool_effect",
        task_id=7,
        run_id="run-7",
        tool_attempt_id="attempt-7",
    )
    report_recovery_refusal("owner_check", "task_ended", task_id=8)
    report_recovery_refusal("checkpoint", "corrupt", task_id=9)
    report_recovery_refusal("checkpoint", "an_unlisted_outcome", task_id=10)
    lines = [(r.levelno, r.getMessage()) for r in records(caplog)]
    assert lines == [
        (
            logging.WARNING,
            "execution_event_recovery_refused task_id=7 run_id=run-7 storage=v2 "
            "component=execution-events reader=tool_outcome "
            "outcome=unknown_tool_effect tool_attempt_id=attempt-7",
        ),
        (
            logging.INFO,
            "execution_event_recovery_refused task_id=8 run_id=- storage=v2 "
            "component=execution-events reader=owner_check outcome=task_ended",
        ),
        (
            logging.ERROR,
            "execution_event_recovery_refused task_id=9 run_id=- storage=v2 "
            "component=execution-events reader=checkpoint outcome=corrupt",
        ),
        (
            logging.WARNING,
            "execution_event_recovery_refused task_id=10 run_id=- storage=v2 "
            "component=execution-events reader=checkpoint "
            "outcome=an_unlisted_outcome",
        ),
    ]
    assert refusal_counts(metric_reader) == {
        key(operation="tool_outcome", outcome="unknown_tool_effect", storage="v2"): 1,
        key(operation="owner_check", outcome="task_ended", storage="v2"): 1,
        key(operation="checkpoint", outcome="corrupt", storage="v2"): 1,
        key(operation="checkpoint", outcome="an_unlisted_outcome", storage="v2"): 1,
    }
