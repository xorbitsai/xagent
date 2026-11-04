"""Stage-B read-budget measurements over template-driven V2 histories.

Measurement only: there are no thresholds here; budgets are decided
separately. The full set takes over an hour, so it is opt-in even for the
slow CI job: every test is slow and skipped unless ``XAGENT_READ_BUDGET=1``::

    XAGENT_READ_BUDGET=1 XAGENT_READ_BUDGET_RESULTS=results.jsonl \
        python -m pytest -m slow tests/web/services/test_execution_event_read_budget.py

Each scenario prints one ``READ_BUDGET {json}`` line and, when
``XAGENT_READ_BUDGET_RESULTS`` names a file, appends the same JSON line to it.
PostgreSQL scenarios need ``XAGENT_TEST_POSTGRES_URL``.

Workloads come from ``execution_event_workload`` (payload templates exported
from real E2E runs). Measured readers, as in the T1 plan:

M1 ``load_task_event_context``; M2 ``load_task_setup_snapshot_sync``;
M3 ``load_event_display_snapshot`` (with the ``load_display_facts`` fetch
timed separately); M4/M5 the WebSocket history replay, cold and warm, against
a JSON round-trip cache double; M6 ``read_event_checkpoint`` (settled run and
interrupted run); M7 ``read_committed_tool_outcome``; M8 one production
``recovery_state`` commit plus table size; M9 (PostgreSQL) ``EXPLAIN
(ANALYZE, BUFFERS)`` of the M1 queries over large covered prefixes.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
import uuid
from collections import Counter, defaultdict
from typing import Any, Callable

import pytest
import sqlalchemy as sa

from tests.web.services.execution_event_workload import (
    Generator,
    Workload,
    WriterSink,
    build,
    exported_sequences,
)
from tests.web.services.test_execution_event_history_baseline import (
    _CountFetchedRows,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.web.models.task import Task
from xagent.web.models.task_execution_event import TaskExecutionEvent

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture

# Large histories take minutes per scenario; the suite default is 300 s.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.timeout(3600),
    pytest.mark.skipif(
        os.getenv("XAGENT_READ_BUDGET") != "1",
        reason="read-budget measurements run only with XAGENT_READ_BUDGET=1",
    ),
]

RESULTS = os.getenv("XAGENT_READ_BUDGET_RESULTS")
PASSES = int(os.getenv("XAGENT_READ_BUDGET_PASSES", "5"))
RSS = os.getenv("XAGENT_READ_BUDGET_RSS", "1") != "0" and sys.platform != "win32"


def _publish(record_property, result: dict[str, Any]) -> None:
    line = json.dumps(result, default=str, sort_keys=True)
    record_property("read_budget", line)
    print("READ_BUDGET " + line)
    if RESULTS:
        with open(RESULTS, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class _JsonCache:
    """Process-local cache that round-trips JSON like the Redis backend."""

    def __init__(self) -> None:
        self.items: dict[str, str] = {}
        self.stored_bytes = 0

    def get_json(self, key):
        raw = self.items.get(key)
        return None if raw is None else json.loads(raw)

    def set_json(self, key, value, ttl_seconds):
        raw = json.dumps(value)
        self.stored_bytes = len(raw.encode())
        self.items[key] = raw

    def delete(self, *keys):
        for key in keys:
            self.items.pop(key, None)

    def delete_prefix(self, prefix):
        for key in [k for k in self.items if k.startswith(prefix)]:
            del self.items[key]


def measure(
    engine,
    read: Callable[[], Any],
    *,
    passes: int = PASSES,
    before_each: Callable[[], None] | None = None,
    capture: list | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Counting, tracemalloc and timing passes, each with fresh Sessions."""
    counted = {"queries": 0, "rows": 0, "serialized_value_bytes": 0}

    def count(_conn, cursor, statement, parameters, context, _many):
        counted["queries"] += 1
        context.cursor = _CountFetchedRows(cursor, counted)
        if capture is not None:
            capture.append((statement, parameters))

    prepare = before_each or (lambda: None)
    sa.event.listen(engine, "after_cursor_execute", count)
    try:
        prepare()
        value = read()
    finally:
        sa.event.remove(engine, "after_cursor_execute", count)
    gc.collect()
    prepare()
    tracemalloc.start()
    try:
        read()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    timings = []
    for _ in range(passes):
        prepare()
        gc.collect()
        started = time.perf_counter()
        read()
        timings.append((time.perf_counter() - started) * 1000)
    return value, {
        "queries": counted["queries"],
        "rows": counted["rows"],
        "fetched_bytes": counted["serialized_value_bytes"],
        "python_peak_bytes": peak,
        "median_ms": round(statistics.median(timings), 2),
        "max_ms": round(max(timings), 2),
    }


def _timed(read: Callable[[], Any], passes: int = PASSES) -> dict[str, float]:
    timings = []
    for _ in range(passes):
        gc.collect()
        started = time.perf_counter()
        read()
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms": round(statistics.median(timings), 2),
        "max_ms": round(max(timings), 2),
    }


def _explain(engine, statements) -> dict[str, Any]:
    """Re-run captured SELECTs under EXPLAIN (ANALYZE, BUFFERS); PG only."""
    plans = []
    with engine.connect() as connection:
        for statement, parameters in statements:
            if not statement.lstrip().upper().startswith("SELECT"):
                continue
            raw = connection.exec_driver_sql(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + statement, parameters
            ).scalar_one()
            plan = (raw if isinstance(raw, list) else json.loads(raw))[0]
            nodes, removed, stack = [], 0, [plan["Plan"]]
            while stack:
                node = stack.pop()
                removed += int(node.get("Rows Removed by Filter", 0))
                label = node["Node Type"]
                if node.get("Index Name"):
                    label += f" {node['Index Name']}"
                nodes.append(label)
                stack.extend(node.get("Plans", []))
            top = plan["Plan"]
            plans.append(
                {
                    "query": _classify(statement),
                    "execution_ms": round(plan["Execution Time"], 3),
                    "planning_ms": round(plan["Planning Time"], 3),
                    "rows_returned": top.get("Actual Rows"),
                    "rows_removed_by_filter": removed,
                    "shared_hit_blocks": top.get("Shared Hit Blocks"),
                    "shared_read_blocks": top.get("Shared Read Blocks"),
                    "nodes": nodes,
                }
            )
    total: dict[str, Any] = {
        "statements": len(plans),
        "execution_ms": round(sum(p["execution_ms"] for p in plans), 3),
        "rows_removed_by_filter": sum(p["rows_removed_by_filter"] for p in plans),
        "shared_hit_blocks": sum(p["shared_hit_blocks"] or 0 for p in plans),
        "shared_read_blocks": sum(p["shared_read_blocks"] or 0 for p in plans),
    }
    by_query: dict[str, dict[str, Any]] = {}
    for plan in plans:
        entry = by_query.setdefault(
            plan["query"],
            {
                "count": 0,
                "execution_ms": 0.0,
                "rows_removed_by_filter": 0,
                "rows_returned": 0,
                "shared_hit_blocks": 0,
                "nodes": plan["nodes"],
            },
        )
        entry["count"] += 1
        entry["execution_ms"] = round(entry["execution_ms"] + plan["execution_ms"], 3)
        entry["rows_removed_by_filter"] += plan["rows_removed_by_filter"]
        entry["rows_returned"] += plan["rows_returned"] or 0
        entry["shared_hit_blocks"] += plan["shared_hit_blocks"] or 0
    total["by_query"] = by_query
    return total


def _classify(statement: str) -> str:
    sql = " ".join(statement.split())
    if "FROM tasks" in sql and "task_execution_events" not in sql:
        return "task_row"
    if "task_execution_events.event_id IN" in sql:
        return "anchors_by_event_id"
    if "task_execution_events.turn_id IN" in sql:
        return "covered_inputs_by_turn_id"
    if "task_execution_events.assistant_message_id IN" in sql:
        return "covered_starts_by_batch"
    if "task_execution_events.tool_attempt_id IN" in sql:
        return "covered_by_tool_attempt_id"
    if "task_execution_events.tool_attempt_id =" in sql:
        return "tool_outcome_by_attempt"
    if "kind NOT IN" in sql:
        return "display_page"
    if "DESC" in sql and "kind =" in sql:
        return "latest_by_kind_desc"
    if "kind IN" in sql:
        return "kind_range_page"
    return "other"


def _analyze(engine) -> None:
    """PG statistics as autovacuum leaves them after a bulk load.

    SQLite is left without ``ANALYZE``, as deployments do not run it.
    """
    if engine.dialect.name == "postgresql":
        with engine.connect() as connection:
            connection.exec_driver_sql("ANALYZE task_execution_events")


def _table_size(engine) -> dict[str, Any]:
    with engine.connect() as connection:
        if engine.dialect.name == "postgresql":
            row = connection.execute(
                sa.text(
                    "SELECT pg_total_relation_size('task_execution_events'),"
                    " pg_relation_size('task_execution_events'),"
                    " pg_indexes_size('task_execution_events'),"
                    " COALESCE(pg_total_relation_size(reltoastrelid), 0)"
                    " FROM pg_class WHERE relname = 'task_execution_events'"
                )
            ).one()
            return {
                "total_bytes": row[0],
                "heap_bytes": row[1],
                "index_bytes": row[2],
                "toast_bytes": row[3],
            }
        try:
            size = connection.execute(
                sa.text(
                    "SELECT SUM(pgsize) FROM dbstat WHERE name = 'task_execution_events'"
                )
            ).scalar()
            index = connection.execute(
                sa.text(
                    "SELECT SUM(pgsize) FROM dbstat WHERE name IN (SELECT name FROM"
                    " sqlite_master WHERE type = 'index' AND tbl_name ="
                    " 'task_execution_events')"
                )
            ).scalar()
            return {"table_bytes": size, "index_bytes": index}
        except sa.exc.OperationalError:
            pages = connection.exec_driver_sql("PRAGMA page_count").scalar()
            page = connection.exec_driver_sql("PRAGMA page_size").scalar()
            return {"database_bytes": pages * page}


def _payload_stats(generator: Generator) -> dict[str, Any]:
    sink = generator.sink
    by_kind: dict[str, list[int]] = defaultdict(list)
    for kind, size in zip(sink.kinds, sink.payload_bytes):
        by_kind[kind].append(size)
    runs = max(generator.run_count, 1)
    return {
        "events": len(sink.kinds),
        "payload_bytes": sum(sink.payload_bytes),
        "runs": generator.run_count,
        "compactions": generator.compactions,
        "unusable_summaries": generator.unusable_summaries,
        "by_kind": {
            kind: {
                "count": len(sizes),
                "bytes": sum(sizes),
                "max": max(sizes),
                "per_run_bytes": sum(sizes) // runs,
            }
            for kind, sizes in sorted(by_kind.items())
        },
    }


def _rss(engine, task_id: int, measure_name: str, run_id: str) -> dict[str, Any]:
    from tests.web.services import execution_event_workload

    url = engine.url
    if engine.dialect.name == "postgresql":
        # Disposable engines connect through creator=; their URL has no
        # database. Rebuild it from the server URL and the actual database.
        with engine.connect() as connection:
            database = connection.exec_driver_sql("SELECT current_database()").scalar()
        url = sa.engine.make_url(os.environ["XAGENT_TEST_POSTGRES_URL"]).set(
            database=database
        )
    url = url.render_as_string(hide_password=False)
    completed = subprocess.run(
        [
            sys.executable,
            execution_event_workload.__file__,
            url,
            str(task_id),
            measure_name,
            run_id,
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        env=os.environ.copy(),
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if completed.returncode != 0 or not lines:
        return {"error": completed.stderr[-2000:]}
    return json.loads(lines[-1])


def _readers(factory, task_id: int):
    from xagent.web.api.websocket import _load_event_historical_stream_snapshot
    from xagent.web.services.task_event_context_service import (
        load_task_event_context,
    )
    from xagent.web.services.task_event_display import (
        display_horizon,
        load_display_facts,
        load_event_display_snapshot,
    )
    from xagent.web.services.task_setup_snapshot import load_task_setup_snapshot_sync

    def model():
        with factory() as db:
            context = load_task_event_context(db, task_id)
            return (len(context.messages), context.watermark)

    def setup():
        snapshot = load_task_setup_snapshot_sync(task_id, None)
        return len(snapshot.conversation_history)

    def display_fetch():
        with factory() as db:
            return len(
                load_display_facts(
                    db, task_id, through_sequence=display_horizon(db, task_id)
                )
            )

    def display():
        with factory() as db:
            snapshot = load_event_display_snapshot(db, task_id)
            return (len(snapshot.events), len(snapshot.messages))

    def display_bytes():
        with factory() as db:
            snapshot = load_event_display_snapshot(db, task_id)
            return len(
                json.dumps([snapshot.events, snapshot.messages], default=str).encode()
            )

    def websocket():
        with factory() as db:
            task = db.get(Task, task_id)
            return len(_load_event_historical_stream_snapshot(db, task).events)

    def websocket_bytes():
        with factory() as db:
            task = db.get(Task, task_id)
            events = _load_event_historical_stream_snapshot(db, task).events
            return len(json.dumps(list(events), default=str).encode())

    return {
        "model": model,
        "setup": setup,
        "display_fetch": display_fetch,
        "display": display,
        "display_bytes": display_bytes,
        "websocket": websocket,
        "websocket_bytes": websocket_bytes,
    }


def _checkpoint_reader(factory, task_id: int, run_id: str):
    from xagent.web.services.task_execution_event_recovery import (
        read_event_checkpoint,
    )

    def read():
        with factory() as db:
            try:
                data = read_event_checkpoint(
                    db,
                    task_id=task_id,
                    scope_id="root",
                    execution_id=str(task_id),
                    run_id=run_id,
                )
                return ("checkpoint", data["label"] if data else None)
            except Exception as error:  # noqa: BLE001 - a refusal is a result
                return ("refused", type(error).__name__)

    return read


def _measure_history(factory, engine, task_id, generator, result) -> None:
    """M1-M6 on a settled history, then M4/M6-M8 inside an interrupted run."""
    from xagent.web.services import hot_path_cache
    from xagent.web.services.task_execution_event_recovery import (
        read_committed_tool_outcome,
    )
    from xagent.web.services.task_execution_event_writer import (
        append_fact_no_commit,
        stage_applied_inputs_no_commit,
    )

    postgres = engine.dialect.name == "postgresql"
    readers = _readers(factory, task_id)
    cache = _JsonCache()
    hot_path_cache.set_cache_backend_for_testing(cache)
    try:
        captured: list = []
        model, result["M1_model"] = measure(engine, readers["model"], capture=captured)
        result["M1_model"]["messages"], watermark = model
        result["M1_model"]["watermark_sequence"] = (watermark or {}).get("sequence")
        if postgres:
            result["M1_model"]["explain"] = _explain(engine, captured)
        history, result["M2_setup"] = measure(engine, readers["setup"])
        # Setup installs exactly the model reader's messages, from this DB.
        assert result["M2_setup"]["queries"] > 0
        assert history == result["M1_model"]["messages"]
        captured = []
        _, result["M3_display"] = measure(engine, readers["display"], capture=captured)
        result["M3_display"]["fetch"] = _timed(readers["display_fetch"])
        result["M3_display"]["convert_ms"] = round(
            result["M3_display"]["median_ms"]
            - result["M3_display"]["fetch"]["median_ms"],
            2,
        )
        result["M3_display"]["output_bytes"] = readers["display_bytes"]()
        if postgres:
            result["M3_display"]["explain"] = _explain(engine, captured)
        _, result["M4_websocket_cold"] = measure(
            engine, readers["websocket"], before_each=lambda: cache.items.clear()
        )
        result["M4_websocket_cold"]["output_bytes"] = readers["websocket_bytes"]()
        result["M4_websocket_cold"]["cache_value_bytes"] = cache.stored_bytes
        cache.items.clear()
        readers["websocket"]()
        _, result["M5_websocket_warm"] = measure(engine, readers["websocket"])
        last_run = generator.runs[-1]
        captured = []
        refused, result["M6_checkpoint_settled"] = measure(
            engine, _checkpoint_reader(factory, task_id, last_run), capture=captured
        )
        result["M6_checkpoint_settled"]["outcome"] = refused
        if postgres:
            result["M6_checkpoint_settled"]["explain"] = _explain(engine, captured)
        result["table"] = _table_size(engine)
        if RSS:
            result["rss"] = {
                name: _rss(engine, task_id, name, last_run)
                for name in ("model", "display", "websocket")
            }

        # A crash inside the next run: its last tool outcome is committed,
        # the newest recovery state still lists the call as pending.
        pending = generator.interrupted_run(generator.w.tools or 1)
        run_id = generator.runs[-1]
        captured = []
        outcome, result["M6_checkpoint_interrupted"] = measure(
            engine, _checkpoint_reader(factory, task_id, run_id), capture=captured
        )
        result["M6_checkpoint_interrupted"]["outcome"] = outcome
        if postgres:
            result["M6_checkpoint_interrupted"]["explain"] = _explain(engine, captured)

        def tool_outcome():
            with factory() as db:
                return (
                    read_committed_tool_outcome(
                        db, task_id=task_id, scope_id="root", tool_call=pending
                    )
                    is not None
                )

        captured = []
        _, result["M7_tool_outcome"] = measure(engine, tool_outcome, capture=captured)
        if postgres:
            result["M7_tool_outcome"]["explain"] = _explain(engine, captured)
        _, result["M4_websocket_running_cold"] = measure(
            engine, readers["websocket"], before_each=lambda: cache.items.clear()
        )
        if RSS:
            result["rss"]["checkpoint_interrupted"] = _rss(
                engine, task_id, "checkpoint", run_id
            )

        # M8: the production writer commits one more recovery state of the
        # interrupted run's current context.
        payload = generator._recovery(
            "before_llm", generator.last_run, generator.last_context
        )
        payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode())

        def write():
            with factory() as db:
                event = append_fact_no_commit(
                    db,
                    task_id=task_id,
                    kind="recovery_state",
                    key=f"runtime:{uuid.uuid4()}",
                    payload=payload,
                    run_id=run_id,
                )
                stage_applied_inputs_no_commit(db, event)
                db.commit()

        before = _table_size(engine)
        result["M8_recovery_write"] = {"payload_bytes": payload_bytes, **_timed(write)}
        result["M8_recovery_write"]["table_growth"] = {
            key: (value or 0) - (before.get(key) or 0)
            for key, value in _table_size(engine).items()
        }
    finally:
        hot_path_cache.set_cache_backend_for_testing(None)


def _run(
    canonical, workload: Workload, record_property, monkeypatch, group: str
) -> None:
    factory, owner = canonical
    engine = factory.kw["bind"]
    # task_setup_snapshot binds get_session_local at import; canonical only
    # patches the database module, so point the setup reader here too.
    monkeypatch.setattr(
        "xagent.web.services.task_setup_snapshot.get_session_local", lambda: factory
    )
    started = time.perf_counter()
    task_id, generator = build(factory, owner, workload)
    _analyze(engine)
    result: dict[str, Any] = {
        "group": group,
        "dialect": engine.dialect.name,
        "workload": workload.label(),
        "params": workload.__dict__,
        "generate_s": round(time.perf_counter() - started, 1),
        "horizon": generator.sink.sequence,
        "payloads": _payload_stats(generator),
    }
    _measure_history(factory, engine, task_id, generator, result)
    _publish(record_property, result)


MATRIX = [
    Workload(
        rounds=n,
        tools=k,
        compaction=compaction,
        faithful_recovery_runs=20 if n >= 1000 else None,
    )
    for n in (10, 100, 1000)
    for k in (0, 3)
    for compaction in ("none", "summary", "drop")
    # Without compaction a 1,000-round context is far past the threshold, so
    # production compacts; drop-fallback covers its summary-less read path.
    if not (n == 1000 and compaction == "none")
    and not (compaction == "drop" and n == 10)
]
LONG = [
    # F8: the latest run is long and compacts repeatedly; every summary
    # carries that run's setup watermark, so the suffix is the whole run.
    *[
        Workload(rounds=10, tools=3, compaction="summary", long_run_tools=t)
        for t in (50, 300)
    ],
    # A long first run: SDK setup reads no event, so no summary is usable;
    # a WebSocket first run's coordinate is its own input.
    *[
        Workload(
            rounds=0,
            tools=3,
            compaction="summary",
            long_run_tools=t,
            long_run_first=True,
            first_entry=entry,
        )
        for t in (50, 300)
        for entry in ("sdk", "websocket")
    ],
]
EXTENDED = [
    Workload(rounds=100, tools=10, compaction="summary"),
    Workload(rounds=100, tools=3, result_bytes=128 * 1024, compaction="summary"),
    Workload(
        rounds=100, tools=3, compaction="summary", child_scope=True, attachments=True
    ),
    Workload(
        rounds=1000,
        tools=3,
        compaction="summary",
        child_scope=True,
        attachments=True,
        faithful_recovery_runs=20,
    ),
    Workload(
        rounds=10,
        tools=3,
        compaction="summary",
        long_run_tools=300,
        result_bytes=128 * 1024,
    ),
]


@pytest.mark.parametrize("workload", MATRIX, ids=Workload.label)
def test_matrix(canonical, workload, record_property, monkeypatch):
    _run(canonical, workload, record_property, monkeypatch, "matrix")


@pytest.mark.parametrize("workload", LONG, ids=Workload.label)
def test_long_runs(canonical, workload, record_property, monkeypatch):
    _run(canonical, workload, record_property, monkeypatch, "long")


@pytest.mark.parametrize("workload", EXTENDED, ids=Workload.label)
def test_extended(canonical, workload, record_property, monkeypatch):
    _run(canonical, workload, record_property, monkeypatch, "extended")


@pytest.mark.parametrize("covered_rows", [10_000, 100_000, 500_000])
def test_m9_covered_prefix_scan(canonical, covered_rows, record_property):
    """F9: T2's targeted queries filter non-indexed columns in the scope range."""
    factory, owner = canonical
    engine = factory.kw["bind"]
    if engine.dialect.name != "postgresql":
        pytest.skip("EXPLAIN (ANALYZE, BUFFERS) is PostgreSQL only")
    rows_per_round = 42  # one k=3 run of the template sequence
    rounds = max(1, covered_rows // rows_per_round)
    workload = Workload(
        rounds=rounds,
        tools=3,
        result_bytes=1024,
        compaction="summary",
        lean_runs=rounds,
        long_run_tools=12,
    )
    started = time.perf_counter()
    task_id, generator = build(factory, owner, workload)
    generated = round(time.perf_counter() - started, 1)
    _analyze(engine)
    readers = _readers(factory, task_id)
    captured: list = []
    model, metrics = measure(engine, readers["model"], capture=captured)
    with factory() as db:
        covered = db.scalar(
            sa.select(sa.func.count()).where(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.scope_id == "root",
                TaskExecutionEvent.sequence <= generator.summaries[-1][1],
            )
        )
    result = {
        "group": "m9",
        "dialect": engine.dialect.name,
        "workload": workload.label(),
        "generate_s": generated,
        "horizon": generator.sink.sequence,
        "covered_rows": covered,
        "summary_floor": generator.summaries[-1][1],
        "watermark": model[1],
        "M1_model": {**metrics, "explain": _explain(engine, captured)},
        "table": _table_size(engine),
    }
    pending = generator.interrupted_run(1)
    from xagent.web.services.task_execution_event_recovery import (
        read_committed_tool_outcome,
    )

    def tool_outcome():
        with factory() as db:
            return (
                read_committed_tool_outcome(
                    db, task_id=task_id, scope_id="root", tool_call=pending
                )
                is not None
            )

    captured = []
    _, result["M7_tool_outcome"] = measure(engine, tool_outcome, capture=captured)
    result["M7_tool_outcome"]["explain"] = _explain(engine, captured)
    _publish(record_property, result)


def test_generator_matches_writer_and_e2e_sequences(
    canonical, record_property, monkeypatch
):
    """Bulk rows equal the production-writer path at N=10, k=3 (plan 3.3)."""
    factory, owner = canonical
    engine = factory.kw["bind"]
    # A low threshold forces several compactions; child scope and file output
    # exercise every generated kind.
    workload = Workload(
        rounds=10,
        tools=3,
        compaction="summary",
        threshold_tokens=12_000,
        child_scope=True,
        attachments=True,
    )
    bulk_task, bulk = build(factory, owner, workload)
    writer_task, writer = build(factory, owner, workload, sink_type=WriterSink)
    assert bulk.compactions > 1 and writer.compactions == bulk.compactions
    assert writer.sink.watermark_mismatches == []
    assert writer.sink.kinds == bulk.sink.kinds

    from xagent.web.services.task_event_context_service import (
        load_task_event_context,
    )

    def stored(task):
        with factory() as db:
            rows = db.execute(
                sa.select(
                    TaskExecutionEvent.kind,
                    TaskExecutionEvent.scope_id,
                    TaskExecutionEvent.payload,
                )
                .where(TaskExecutionEvent.task_id == task)
                .order_by(TaskExecutionEvent.sequence)
            ).all()
        return [
            (k, s == "root", len(json.dumps(p, ensure_ascii=False).encode()))
            for k, s, p in rows
        ]

    bulk_rows, writer_rows = stored(bulk_task), stored(writer_task)
    assert [r[:2] for r in bulk_rows] == [r[:2] for r in writer_rows]
    per_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (kind, _, size), (_, _, other) in zip(bulk_rows, writer_rows):
        per_kind[kind][0] += size
        per_kind[kind][1] += other
    ratios = {kind: round(b / w, 4) for kind, (b, w) in per_kind.items() if w}
    # Chat facts are built by stage_chat_message_no_commit from its own
    # message fields; every other payload must be byte-identical in size.
    assert {
        kind: ratio
        for kind, ratio in ratios.items()
        if kind not in {"input_accepted", "assistant_message"} and ratio != 1.0
    } == {}

    with factory() as db:
        bulk_context = load_task_event_context(db, bulk_task)
        writer_context = load_task_event_context(db, writer_task)

    def shape(context):
        return [(m["role"], len(m.get("content") or "")) for m in context.messages]

    assert shape(bulk_context) == shape(writer_context)
    horizon = bulk.sink.sequence
    assert [m.body["role"] for m in bulk._history(horizon)] == [
        m["role"] for m in bulk_context.messages
    ]

    # Each generated run follows the committed E2E order, minus the kinds
    # those workflows emit that this model does not (memory retrieval etc.).
    sequences = exported_sequences()
    expected = {
        name: [e["kind"] for e in sequences[name] if e["scope"] == "root"]
        for name in ("sdk_two_runs", "react_file_tools")
    }
    _, k0 = build(factory, owner, Workload(rounds=2, tools=0))
    _, k2 = build(factory, owner, Workload(rounds=1, tools=2))
    assert k0.sink.kinds == expected["sdk_two_runs"]
    assert k2.sink.kinds == expected["react_file_tools"]
    # Byte fidelity: regenerate the E2E file-tool run with its own text and
    # result sizes (9-char input, 75-char answer, 20-byte read result, the
    # second call writing a file) and compare per-kind payload bytes.
    from tests.web.services import execution_event_workload as workload_module

    monkeypatch.setattr(workload_module, "USER_CHARS", 9)
    monkeypatch.setattr(workload_module, "ANSWER_CHARS", 75)
    _, e2e_sized = build(
        factory,
        owner,
        Workload(rounds=1, tools=2, result_bytes=20, attachments=True, file_every=1),
    )
    e2e_bytes, generated_bytes = Counter(), Counter()
    for event in sequences["react_file_tools"]:
        e2e_bytes[event["kind"]] += event["payload_bytes"]
    for kind, size in zip(e2e_sized.sink.kinds, e2e_sized.sink.payload_bytes):
        generated_bytes[kind] += size
    _publish(
        record_property,
        {
            "group": "validation",
            "dialect": engine.dialect.name,
            "events": len(bulk_rows),
            "compactions": bulk.compactions,
            "bulk_to_writer_payload_byte_ratio_by_kind": ratios,
            "bulk_payload_bytes": sum(r[2] for r in bulk_rows),
            "writer_payload_bytes": sum(r[2] for r in writer_rows),
            "model_messages": len(bulk_context.messages),
            "e2e_react_bytes_by_kind": dict(e2e_bytes),
            "generated_e2e_sized_bytes_by_kind": dict(generated_bytes),
            "summary_floors": [floor for _, floor in bulk.summaries],
        },
    )
