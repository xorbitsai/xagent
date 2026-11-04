"""Measurement and observe-only budgets for V2 execution-event reads.

Each of the five readers (model, display, reconnect, checkpoint, tool_outcome)
is driven for real over seeded V2 facts; its duration and row count are
asserted as exported by an in-memory metric reader. Budgets only report: the
reader's result never changes.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from contextlib import contextmanager
from typing import Any

import pytest

from tests.web.services.test_execution_event_signals import (
    SENTINEL,
    exported,
    key,
)
from tests.web.services.test_execution_event_signals import (
    metric_reader as metric_reader_fixture,
)
from tests.web.services.test_task_event_context_service import accept
from tests.web.services.test_task_event_display import fact
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent.checkpoint import (
    CheckpointUnavailableError,
    UnknownToolEffectError,
)
from xagent.web.api import websocket
from xagent.web.models.task import Task
from xagent.web.services import execution_event_signals as signals
from xagent.web.services.execution_event_signals import (
    ExecutionEventReadMeasurement,
    measure_execution_event_read,
)
from xagent.web.services.task_event_context_service import (
    _load_task_event_context,
    load_task_event_context,
)
from xagent.web.services.task_event_display import (
    _load_display_facts,
    display_horizon,
    load_display_facts,
)
from xagent.web.services.task_execution_event_recovery import (
    _read_committed_tool_outcome,
    _read_event_checkpoint,
    read_committed_tool_outcome,
    read_event_checkpoint,
)

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
metric_reader = metric_reader_fixture

ROW_ENV = "XAGENT_EXECUTION_EVENT_READ_ROW_BUDGETS"
DURATION_ENV = "XAGENT_EXECUTION_EVENT_READ_DURATION_BUDGETS_MS"
LOGGER = "xagent.web.services.execution_event_signals"
DURATION = "xagent.execution_event.read.duration"
ROWS = "xagent.execution_event.read.rows"
EXCEEDED = "xagent.execution_event.read.budget_exceeded"

TOOL_CALL = {
    "name": "search",
    "id": "attempt",
    "tool_attempt_id": "attempt",
    "assistant_message_id": "batch",
}


@pytest.fixture(autouse=True)
def no_budgets_by_default(monkeypatch):
    monkeypatch.delenv(ROW_ENV, raising=False)
    monkeypatch.delenv(DURATION_ENV, raising=False)
    signals._reported_invalid_budget_configs.clear()
    yield
    signals._reported_invalid_budget_configs.clear()


def attrs(**attributes: str) -> tuple[tuple[str, str], ...]:
    """Exported attributes of a read signal: always tagged ``storage=v2``."""
    return key(storage="v2", **attributes)


@contextmanager
def measurement_suppressed():
    """Run reads without recording a point, as if nested in a measured read."""
    token = signals._READ_MEASUREMENT_ACTIVE.set(True)
    try:
        yield
    finally:
        signals._READ_MEASUREMENT_ACTIVE.reset(token)


def histogram(reader, name: str) -> dict[tuple[tuple[str, str], ...], Any]:
    """Exported histogram points keyed by their sorted attributes."""
    data = reader.get_metrics_data()
    points: dict[tuple[tuple[str, str], ...], Any] = {}
    if data is None:
        return points
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name == name:
                    for point in metric.data.data_points:
                        points[tuple(sorted(point.attributes.items()))] = point
    return points


def budget_warnings(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == LOGGER
        and record.levelno == logging.WARNING
        and record.getMessage().startswith("execution_event_read_budget_exceeded")
    ]


def tool_fact(db, tid, kind, *, result=None, attempt="attempt", batch="batch"):
    data: dict[str, Any] = {
        "tool_name": "search",
        "tool_call_id": attempt,
        "tool_params": {"query": SENTINEL},
    }
    if kind != "tool_execution_start":
        data["result"] = result
    return fact(
        db,
        tid,
        kind,
        f"{attempt}:{kind}",
        data,
        tool_attempt_id=attempt,
        assistant_message_id=batch,
    )


def checkpoint_snapshot(**state: Any) -> dict[str, Any]:
    return {
        "protocol_event_id": "state",
        "data": {
            "checkpoint_type": "agent_execution_checkpoint",
            "snapshot_schema_version": 1,
            "snapshot": {
                "execution_id": "execution",
                "context": {"messages": []},
                "pattern_state": state,
            },
        },
    }


@pytest.fixture
def seeded(canonical):
    """A V2 history every reader can read, with the sentinel in its payloads."""
    factory, tid = canonical
    with factory() as db:
        accept(db, tid, "turn", SENTINEL)
        fact(
            db,
            tid,
            "agent_message",
            "answer",
            {"message": SENTINEL, "content": SENTINEL, "message_type": "assistant"},
        )
        tool_fact(db, tid, "tool_execution_start")
        tool_fact(db, tid, "tool_execution_end", result=SENTINEL)
        # An older state of this execution under a newer state of another one:
        # the checkpoint scan reads two rows before it finds the match.
        fact(
            db,
            tid,
            "recovery_state",
            "state:execution",
            checkpoint_snapshot(pending_tool_calls=[TOOL_CALL]),
            flat=True,
        )
        other = checkpoint_snapshot()
        other["data"]["snapshot"]["execution_id"] = "other"
        fact(db, tid, "recovery_state", "state:other", other, flat=True)
        db.commit()
    return factory, tid


class Readers:
    """The five readers as zero-argument calls, plus the unmeasured result."""

    def __init__(self, factory, tid, monkeypatch):
        self.factory = factory
        self.tid = tid
        self.cache: dict[str, Any] = {}
        monkeypatch.setattr(websocket, "cache_get", self.cache.get)
        monkeypatch.setattr(
            websocket, "cache_set", lambda k, v, ttl_seconds: self.cache.update({k: v})
        )

    def model(self):
        with self.factory() as db:
            context = load_task_event_context(db, self.tid)
            return context.messages, context.watermark, context.selected_skill_name

    def model_unmeasured(self):
        with self.factory() as db:
            context = _load_task_event_context(
                db,
                self.tid,
                before_message_id=None,
                before_turn_id=None,
                read=ExecutionEventReadMeasurement(),
            )
            return context.messages, context.watermark, context.selected_skill_name

    def display(self):
        with self.factory() as db:
            return [
                event.event_id
                for event in load_display_facts(
                    db,
                    self.tid,
                    through_sequence=display_horizon(db, self.tid),
                )
            ]

    def display_unmeasured(self):
        with self.factory() as db:
            return [
                event.event_id
                for event in _load_display_facts(
                    db,
                    self.tid,
                    scope_id="root",
                    through_sequence=display_horizon(db, self.tid),
                    after_sequence=0,
                    source=None,
                    kinds=None,
                )
            ]

    def reconnect(self):
        with self.factory() as db:
            task = db.get(Task, self.tid)
            return list(
                websocket._load_event_historical_stream_snapshot(db, task).events
            )

    def reconnect_unmeasured(self):
        # The snapshot builder also loads display facts, a measured read of its
        # own; suppress it so the baseline leaves no point behind.
        with self.factory() as db, measurement_suppressed():
            task = db.get(Task, self.tid)
            return list(
                websocket._build_event_historical_stream_snapshot(
                    db, task, ExecutionEventReadMeasurement()
                ).events
            )

    def checkpoint(self):
        with self.factory() as db:
            data = read_event_checkpoint(
                db,
                task_id=self.tid,
                scope_id="root",
                execution_id="execution",
                run_id=None,
                filter_run=False,
            )
            return data["snapshot"]["execution_id"]

    def checkpoint_unmeasured(self):
        with self.factory() as db:
            data = _read_event_checkpoint(
                db,
                task_id=self.tid,
                scope_id="root",
                execution_id="execution",
                run_id=None,
                filter_run=False,
                read=ExecutionEventReadMeasurement(),
            )
            return data["snapshot"]["execution_id"]

    def tool_outcome(self):
        with self.factory() as db:
            return read_committed_tool_outcome(
                db, task_id=self.tid, scope_id="root", tool_call=TOOL_CALL
            )

    def tool_outcome_unmeasured(self):
        with self.factory() as db:
            return _read_committed_tool_outcome(
                db,
                task_id=self.tid,
                scope_id="root",
                tool_call=TOOL_CALL,
                through_sequence=None,
                run_id=None,
                read=ExecutionEventReadMeasurement(),
            )


@pytest.fixture
def readers(seeded, monkeypatch):
    factory, tid = seeded
    return Readers(factory, tid, monkeypatch)


# The reconnect replay of the seeded history: the info frame, one converted
# event per displayed fact (input, answer, tool start, tool end) and the
# completion frame. Counted from the seed, not from the reader's own result.
RECONNECT_FRAMES = [
    "task_info",
    "user_message",
    "agent_message",
    "tool_execution_start",
    "tool_execution_end",
    "historical_data_complete",
]

# operation, reader method, expected row count of the seeded history
READERS = [
    ("model", "model", 4),
    ("display", "display", 4),
    ("reconnect", "reconnect", len(RECONNECT_FRAMES)),
    ("checkpoint", "checkpoint", 2),
    ("tool_outcome", "tool_outcome", 2),
]


def comparable(result):
    """Drop what differs between two reads of the same replay (random ids, clocks)."""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return [
            {k: v for k, v in event.items() if k not in {"event_id", "timestamp"}}
            for event in result
        ]
    return result


@pytest.mark.parametrize("operation,reader,rows", READERS)
def test_reader_records_duration_and_rows(
    readers, metric_reader, operation, reader, rows
):
    result = getattr(readers, reader)()
    if reader == "checkpoint":
        assert result == "execution"
    if reader == "tool_outcome":
        assert result == {"result": SENTINEL, "error": None}
    if reader == "reconnect":
        # The replay is exactly the frames the client receives.
        assert [event.get("event_type", event["type"]) for event in result] == (
            RECONNECT_FRAMES
        )

    durations = histogram(metric_reader, DURATION)
    row_points = histogram(metric_reader, ROWS)
    # Reconnect carries its outcome; the others have no second shape.
    attributes = (
        attrs(operation=operation, outcome="loaded")
        if operation == "reconnect"
        else attrs(operation=operation)
    )
    # The nested display-facts read is part of reconnect, not a second point,
    # and the nested tool-outcome reads are part of checkpoint.
    assert set(durations) == {attributes}
    assert set(row_points) == {attributes}
    assert durations[attributes].count == 1
    assert row_points[attributes].count == 1
    assert row_points[attributes].sum == rows
    assert durations[attributes].sum >= 0
    assert exported(metric_reader, EXCEEDED) == {}


def test_agent_execution_summary_rows_are_a_display_read(canonical, metric_reader):
    from xagent.web.services.task_agent_execution import load_agent_execution_detail

    factory, tid = canonical
    with factory() as db:
        fact(
            db,
            tid,
            "react_task_start",
            "child",
            {
                "source": "xagent-agent-tool-child",
                "worker_task_id": "child-scope",
                "agent_name": "Worker",
            },
            scope_id="child-scope",
        )
        for key_, worker in (
            ("summary-1", "child-scope"),
            ("summary-2", "child-scope"),
            ("other", "other-scope"),
        ):
            fact(
                db,
                tid,
                "task_update_general",
                key_,
                {"event_type": "workforce_delegation_start", "worker_task_id": worker},
            )
        db.commit()
    with factory() as db:
        task = db.get(Task, tid)
        detail = load_agent_execution_detail(db, task, "child-scope")
    assert detail["agent_name"] == "Worker"

    point = histogram(metric_reader, ROWS)[attrs(operation="display")]
    # One child-scope fact, then the agent's two root-scope summary rows.
    assert point.count == 2
    assert point.sum == 1 + 2


def test_seeded_row_counts_are_what_the_history_holds(readers):
    # Guards the expected counts above against drifting seed data.
    from sqlalchemy import func, select

    from xagent.web.models.task_execution_event import TaskExecutionEvent

    def count(db, *kinds):
        return db.scalar(
            select(func.count())
            .select_from(TaskExecutionEvent)
            .where(
                TaskExecutionEvent.task_id == readers.tid,
                TaskExecutionEvent.kind.in_(kinds),
            )
        )

    with readers.factory() as db:
        assert (
            count(
                db,
                "input_accepted",
                "agent_message",
                "tool_execution_start",
                "tool_execution_end",
            )
            == 4
        )
        assert count(db, "recovery_state") == 2
        assert count(db, "tool_execution_start", "tool_execution_end") == 2
    assert len(readers.display_unmeasured()) == 4
    # Four displayed facts plus the info and completion frames.
    assert len(readers.reconnect_unmeasured()) == 4 + 2


def test_reconnect_reports_cache_hit_and_loaded(readers, metric_reader):
    readers.reconnect()
    readers.reconnect()
    durations = histogram(metric_reader, DURATION)
    loaded = attrs(operation="reconnect", outcome="loaded")
    hit = attrs(operation="reconnect", outcome="cache_hit")
    assert set(durations) == {loaded, hit}
    assert durations[loaded].count == durations[hit].count == 1
    rows = histogram(metric_reader, ROWS)
    # The replay is the same size from the cache as from the database.
    assert rows[loaded].sum == rows[hit].sum > 0


def test_cache_hit_reconnect_is_measured_but_never_over_budget(
    readers, metric_reader, caplog, monkeypatch
):
    # A cache hit replays a stored snapshot and does not read the event log, so
    # the budgets guarding event-log reads must not fire for it.
    monkeypatch.setattr(signals, "time", Clock(step=0.25))  # each read: 250 ms
    monkeypatch.setenv(ROW_ENV, "reconnect=1")
    monkeypatch.setenv(DURATION_ENV, "reconnect=100")

    readers.reconnect()  # loaded: over both budgets
    assert exported(metric_reader, EXCEEDED) == {
        attrs(operation="reconnect", outcome="rows"): 1,
        attrs(operation="reconnect", outcome="duration"): 1,
    }
    assert len(budget_warnings(caplog)) == 1

    readers.reconnect()  # cache hit: same size, same duration, no new report
    assert exported(metric_reader, EXCEEDED) == {
        attrs(operation="reconnect", outcome="rows"): 1,
        attrs(operation="reconnect", outcome="duration"): 1,
    }
    assert len(budget_warnings(caplog)) == 1
    # Still measured.
    hit = attrs(operation="reconnect", outcome="cache_hit")
    assert histogram(metric_reader, DURATION)[hit].count == 1
    assert histogram(metric_reader, ROWS)[hit].sum == len(RECONNECT_FRAMES)


@pytest.mark.parametrize("operation,reader,rows", READERS)
def test_row_budget_below_count_reports_once_and_changes_nothing(
    readers, metric_reader, caplog, monkeypatch, operation, reader, rows
):
    baseline = comparable(getattr(readers, f"{reader}_unmeasured")())
    readers.cache.clear()
    monkeypatch.setenv(ROW_ENV, f"{operation}=1")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = getattr(readers, reader)()

    assert comparable(result) == baseline
    warnings = budget_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert message.startswith(
        f"execution_event_read_budget_exceeded task_id={readers.tid} storage=v2 "
        f"component=execution-events operation={operation} rows={rows} "
        "duration_ms="
    )
    assert message.endswith(" budget_rows=1 budget_ms=-")
    assert exported(metric_reader, EXCEEDED) == {
        attrs(operation=operation, outcome="rows"): 1
    }
    assert SENTINEL not in caplog.text


def test_default_star_budget_applies_to_unlisted_operations(
    readers, metric_reader, caplog, monkeypatch
):
    monkeypatch.setenv(ROW_ENV, "display=100000,*=1")
    readers.display()
    readers.tool_outcome()
    assert exported(metric_reader, EXCEEDED) == {
        attrs(operation="tool_outcome", outcome="rows"): 1
    }
    assert len(budget_warnings(caplog)) == 1


def test_row_budget_at_or_above_count_does_not_report(
    readers, metric_reader, caplog, monkeypatch
):
    monkeypatch.setenv(ROW_ENV, "tool_outcome=2")
    readers.tool_outcome()
    assert exported(metric_reader, EXCEEDED) == {}
    assert budget_warnings(caplog) == []
    # The measurement itself is still there.
    assert attrs(operation="tool_outcome") in histogram(metric_reader, ROWS)


@pytest.mark.parametrize("operation,reader,rows", READERS)
def test_no_budget_reports_nothing_but_measures(
    readers, metric_reader, caplog, operation, reader, rows
):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        getattr(readers, reader)()
    assert exported(metric_reader, EXCEEDED) == {}
    assert [r for r in caplog.records if r.name == LOGGER] == []
    assert histogram(metric_reader, DURATION)


class Clock:
    """A perf_counter whose reads advance by a fixed step."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def perf_counter(self) -> float:
        self.now += self.step
        return self.now


def test_duration_budget_reports_without_a_row_budget(
    readers, metric_reader, caplog, monkeypatch
):
    monkeypatch.setattr(signals, "time", Clock(step=0.25))  # each read: 250 ms
    monkeypatch.setenv(DURATION_ENV, "tool_outcome=100")
    result = readers.tool_outcome()
    assert result["result"] == SENTINEL
    assert exported(metric_reader, EXCEEDED) == {
        attrs(operation="tool_outcome", outcome="duration"): 1
    }
    (warning,) = budget_warnings(caplog)
    message = warning.getMessage()
    assert "rows=2 duration_ms=250.0 budget_rows=- budget_ms=100" in message


def test_both_budgets_exceeded_is_one_warning_two_counters(
    readers, metric_reader, caplog, monkeypatch
):
    monkeypatch.setattr(signals, "time", Clock(step=0.25))
    monkeypatch.setenv(DURATION_ENV, "*=100")
    monkeypatch.setenv(ROW_ENV, "*=1")
    readers.tool_outcome()
    assert exported(metric_reader, EXCEEDED) == {
        attrs(operation="tool_outcome", outcome="rows"): 1,
        attrs(operation="tool_outcome", outcome="duration"): 1,
    }
    assert len(budget_warnings(caplog)) == 1


def test_changing_the_budget_takes_effect_without_restart(
    readers, metric_reader, caplog, monkeypatch
):
    readers.tool_outcome()
    assert budget_warnings(caplog) == []
    monkeypatch.setenv(ROW_ENV, "tool_outcome=1")
    readers.tool_outcome()
    assert len(budget_warnings(caplog)) == 1
    monkeypatch.setenv(ROW_ENV, "tool_outcome=5")
    readers.tool_outcome()
    assert len(budget_warnings(caplog)) == 1


def test_nested_reads_are_measured_once_as_the_outer_operation(readers, metric_reader):
    # The checkpoint's pending tool call reads its committed outcome, and the
    # reconnect replay loads display facts.
    readers.checkpoint()
    readers.reconnect()
    assert set(histogram(metric_reader, DURATION)) == {
        attrs(operation="checkpoint"),
        attrs(operation="reconnect", outcome="loaded"),
    }


def test_reader_that_raises_still_records_duration_and_reraises(readers, metric_reader):
    with readers.factory() as db:
        with pytest.raises(CheckpointUnavailableError, match="no longer exists"):
            read_event_checkpoint(
                db,
                task_id=987654,
                scope_id="root",
                execution_id="execution",
                run_id=None,
            )
    assert histogram(metric_reader, DURATION)[attrs(operation="checkpoint")].count == 1
    # The reader never learned a row count, so none is recorded.
    assert histogram(metric_reader, ROWS) == {}


def test_expected_refusal_keeps_the_rows_it_read(readers, metric_reader):
    with readers.factory() as db:
        tool_fact(
            db, readers.tid, "tool_execution_start", attempt="unknown", batch="other"
        )
        db.commit()
        with pytest.raises(UnknownToolEffectError):
            read_committed_tool_outcome(
                db,
                task_id=readers.tid,
                scope_id="root",
                tool_call={
                    **TOOL_CALL,
                    "id": "unknown",
                    "tool_attempt_id": "unknown",
                    "assistant_message_id": "other",
                },
            )
    assert (
        histogram(metric_reader, DURATION)[attrs(operation="tool_outcome")].count == 1
    )
    assert histogram(metric_reader, ROWS)[attrs(operation="tool_outcome")].sum == 1


def test_helper_reraises_the_same_error_object_after_recording_and_reporting(
    metric_reader, caplog, monkeypatch
):
    monkeypatch.setenv(ROW_ENV, "model=1")
    error = RuntimeError(SENTINEL)
    with pytest.raises(RuntimeError) as raised:
        with measure_execution_event_read("model", task_id=5) as read:
            read.rows = 10
            raise error
    assert raised.value is error
    assert histogram(metric_reader, DURATION)[attrs(operation="model")].count == 1
    (warning,) = budget_warnings(caplog)
    assert "task_id=5" in warning.getMessage()
    assert SENTINEL not in caplog.text


def test_failing_telemetry_or_budget_lookup_never_changes_the_read(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("broken")

    monkeypatch.setattr(signals, "observe_value", broken)
    monkeypatch.setattr(signals, "get_execution_event_read_row_budgets", broken)
    with measure_execution_event_read("model", task_id=5) as read:
        read.rows = 3
        value = "unchanged"
    assert value == "unchanged"


def test_invalid_budget_config_logs_once_and_the_read_still_succeeds(
    readers, metric_reader, caplog, monkeypatch
):
    baseline = readers.tool_outcome()
    monkeypatch.setenv(ROW_ENV, "tool_outcome=abc")
    monkeypatch.setenv(DURATION_ENV, "bogus=1")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        results = [readers.tool_outcome() for _ in range(3)]
    assert results == [baseline] * 3
    invalid = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("execution_event_read_budget_config_invalid")
    ]
    # One warning per invalid variable, however many reads follow.
    assert len(invalid) == 2
    assert any(ROW_ENV in message for message in invalid)
    assert any(DURATION_ENV in message for message in invalid)
    assert budget_warnings(caplog) == []
    assert exported(metric_reader, EXCEEDED) == {}
    assert (
        histogram(metric_reader, DURATION)[attrs(operation="tool_outcome")].count == 4
    )

    # A different invalid value is a new configuration problem.
    monkeypatch.setenv(ROW_ENV, "tool_outcome=-1")
    readers.tool_outcome()
    readers.tool_outcome()
    assert (
        len(
            [
                r
                for r in caplog.records
                if r.getMessage().startswith(
                    "execution_event_read_budget_config_invalid"
                )
            ]
        )
        == 3
    )


def test_valid_duration_budget_still_applies_when_the_row_budget_is_invalid(
    readers, metric_reader, caplog, monkeypatch
):
    monkeypatch.setattr(signals, "time", Clock(step=0.25))
    monkeypatch.setenv(ROW_ENV, "nonsense")
    monkeypatch.setenv(DURATION_ENV, "*=100")
    readers.tool_outcome()
    assert exported(metric_reader, EXCEEDED) == {
        attrs(operation="tool_outcome", outcome="duration"): 1
    }


def test_sentinel_never_reaches_any_signal(readers, metric_reader, caplog, monkeypatch):
    monkeypatch.setenv(ROW_ENV, "*=1")
    monkeypatch.setenv(DURATION_ENV, "*=1")
    monkeypatch.setattr(signals, "time", Clock(step=1.0))
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        for name in ("model", "display", "reconnect", "checkpoint", "tool_outcome"):
            getattr(readers, name)()
    assert len(budget_warnings(caplog)) == 5
    assert SENTINEL not in caplog.text
    for name in (DURATION, ROWS):
        assert SENTINEL not in str(histogram(metric_reader, name))
    assert SENTINEL not in str(exported(metric_reader, EXCEEDED))


def test_measurement_works_in_a_worker_thread_and_nesting_is_per_context(
    readers, metric_reader
):
    box: dict[str, Any] = {}

    def run():
        box["value"] = readers.tool_outcome()

    def run_in_fresh_context():
        # Python 3.14 free-threaded builds start threads in a copy of the
        # caller's context; a fresh one keeps this test about nesting per
        # context on every build.
        contextvars.Context().run(run)

    # Reads inside a measured read on another thread are independent: the
    # suppression is per context, not process-wide.
    with measure_execution_event_read("model", task_id=readers.tid) as read:
        thread = threading.Thread(target=run_in_fresh_context)
        thread.start()
        thread.join()
        read.rows = 1
    assert box["value"]["result"] == SENTINEL
    assert set(histogram(metric_reader, DURATION)) == {
        attrs(operation="model"),
        attrs(operation="tool_outcome"),
    }
