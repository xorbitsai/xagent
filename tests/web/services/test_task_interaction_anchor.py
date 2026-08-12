"""Contract tests for ``resolve_interaction_anchor``
(``task_interaction_anchor.py``) and the static zero-production-caller
assertion this module introduces for the resolver itself.

Two related cases live in other test files rather than here. The checkpoint
run-partition predicate's move into ``trace_event_staging.py`` is not about
the resolver at all and is tested in ``test_trace_event_staging.py``,
alongside the predicate itself. The zero-direct-construction guard for the
public ``InteractionHandoff`` class depends on that class's public rename,
which this module's own dependencies do not carry -- it is tested in
``test_interaction_handoff_surface.py``, where the rename is pinned.

All judgment-table cells run on a private, file-backed SQLite database per
test, following ``test_interaction_staging.py``'s own convention (see that
file's module docstring for why file-backed rather than in-memory).
"""

from __future__ import annotations

import ast
import logging
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import xagent
from tests.web.services.task_interaction_schema_shared import make_user
from xagent.core.agent.checkpoint import CHECKPOINT_TYPE, LEGACY_CHECKPOINT_TYPES
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TraceEvent
from xagent.web.services import interaction_rollout as ir
from xagent.web.services import ops_signals
from xagent.web.services.task_interaction_anchor import resolve_interaction_anchor
from xagent.web.services.task_interaction_staging import InteractionAnchor
from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD

_event_counter = count()


def _engine(tmp_path: Path):
    db_path = tmp_path / "task_interaction_anchor.db"
    engine = create_engine(f"sqlite:///{db_path}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    return engine


def _session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _make_task(db: Session, *, run_id: str | None) -> Task:
    user_id = make_user(db)
    task = Task(user_id=user_id, title="anchor resolver fixture task", run_id=run_id)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _make_row(
    db: Session,
    *,
    task_id: int,
    build_id: str | None = None,
    event_type: str = "system_update_general",
    checkpoint_type: str | None = CHECKPOINT_TYPE,
    run_partition: str | None = "run-a",
    execution_id: str | None = None,
) -> TraceEvent:
    data: dict[str, Any] = {}
    if checkpoint_type is not None:
        data["checkpoint_type"] = checkpoint_type
    if run_partition is not None:
        data[TASK_RUN_ID_TRACE_FIELD] = run_partition
    if execution_id is not None:
        data["execution_id"] = execution_id
    row = TraceEvent(
        task_id=task_id,
        build_id=build_id,
        event_id=f"anchor-event-{next(_event_counter)}",
        event_type=event_type,
        timestamp=datetime.now(timezone.utc),
        data=data,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _point(db: Session, task: Task, trace_event_id: int | None) -> None:
    task.last_checkpoint_trace_event_id = trace_event_id
    db.commit()
    db.refresh(task)


def _point_dangling(db: Session, task: Task, trace_event_id: int) -> None:
    """Points ``task`` at a ``trace_events`` row id that does not exist.

    ``trace_handlers.py``'s own docstring says this is only reachable on a
    database upgraded through Alembic rather than created fresh, because
    that path has no DB-level FK on this column. This fixture's schema is
    built fresh via ``create_all``, which does enforce the FK -- so
    foreign-key checking is switched off for this one write, reproducing
    the no-FK production shape directly rather than fighting the stronger
    guarantee ``create_all`` happens to give a test schema.
    """
    db.execute(sa.text("PRAGMA foreign_keys=OFF"))
    task.last_checkpoint_trace_event_id = trace_event_id
    db.commit()
    db.execute(sa.text("PRAGMA foreign_keys=ON"))
    db.refresh(task)


def _build_scenario(
    db: Session,
    *,
    task_run_id: str | None = "run-a",
    row_task_id: int | None = None,
    build_id: str | None = None,
    event_type: str = "system_update_general",
    checkpoint_type: str | None = CHECKPOINT_TYPE,
    run_partition: str | None = "run-a",
    execution_id: str | None = None,
) -> tuple[Task, TraceEvent]:
    """A task whose checkpoint pointer names a row satisfying every one of
    the six corrupt-check conditions by default -- callers pass one
    ``overrides``-style kwarg to break exactly one condition."""

    task = _make_task(db, run_id=task_run_id)
    target_task_id = row_task_id if row_task_id is not None else task.id
    row = _make_row(
        db,
        task_id=target_task_id,
        build_id=build_id,
        event_type=event_type,
        checkpoint_type=checkpoint_type,
        run_partition=run_partition,
        execution_id=execution_id,
    )
    _point(db, task, row.id)
    return task, row


@pytest.fixture(autouse=True)
def _reset_state():
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)
    ir._counters.clear()
    yield
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)
    ir._counters.clear()


def _count_anchor_constructions(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Instruments ``InteractionAnchor.__init__`` itself, not the return
    value of ``resolve_interaction_anchor`` -- the cross-task-row corrupt
    tests below need to prove the dataclass was never *built* on the
    corrupt path, which "the function returned None" alone does not
    distinguish from "it built one and then discarded it"."""

    calls = {"n": 0}
    original_init = InteractionAnchor.__init__

    def _wrapped(self: InteractionAnchor, *args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(InteractionAnchor, "__init__", _wrapped)
    return calls


# ---------------------------------------------------------------------------
# task.run_id is NOT NULL but task.last_checkpoint_trace_event_id IS NULL ->
# absence. Distinct from the no-run-id case below: this cell exercises step
# 2 of the judgment table specifically, with step 1 passed, so a
# pointer-NULL misclassification (e.g. falling into step 3's
# dangling-pointer branch) cannot hide behind step 1 short-circuiting first.
# ---------------------------------------------------------------------------


def test_ta1_no_checkpoint_pointer_is_absence(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    task = _make_task(db, run_id="run-a")

    result = resolve_interaction_anchor(db, task)

    assert result is None
    assert ops_signals.active_degradations() == {}
    # Step 2 registers no counter -- only step 1 (anchor_absent_no_run) and
    # step 3 (anchor_unavailable_dangling_pointer) do; see the resolver's
    # own judgment table.
    assert ir.counters_snapshot() == {}
    db.close()


# ---------------------------------------------------------------------------
# The pointer names no live trace_events row -> unavailable, reported as
# absence, counted separately from a NULL pointer.
# ---------------------------------------------------------------------------


def test_ta2_dangling_pointer_is_absence_and_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(
        logging.WARNING, logger="xagent.web.services.task_interaction_anchor"
    )
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    task = _make_task(db, run_id="run-a")
    # A pointer naming a row id that was never inserted -- the same shape
    # an Alembic-upgraded database with no DB-level FK on this column can
    # produce (see trace_handlers.py's own docstring on this exact case).
    _point_dangling(db, task, 999_999)

    result = resolve_interaction_anchor(db, task)

    assert result is None
    assert ops_signals.active_degradations() == {}
    assert ir.counters_snapshot() == {ir.COUNTER_ANCHOR_UNAVAILABLE_DANGLING_POINTER: 1}
    # Step 3's judgment-table entry is a logger.warning call, not info or
    # error -- pinned directly, since the judgment table's log levels are
    # contract, not incidental. Matched on the raw format string (not the
    # rendered message), so this stays exact even if the %-args change.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, caplog.records
    assert warnings[0].msg == (
        "task %s's checkpoint pointer %s has no matching trace_events row"
    )
    db.close()


# ---------------------------------------------------------------------------
# The row exists but fails one of six self-consistency conditions ->
# corrupt. Six parametrized cells, one broken condition each.
# ---------------------------------------------------------------------------

_CONDITION_BREAKS: dict[str, dict[str, Any]] = {
    "ownership": {"cross_task": True},
    "event_type": {"event_type": "system_update_partial"},
    "build_partition": {"build_id": "build-x"},
    "checkpoint_type": {"checkpoint_type": "not_a_checkpoint_type"},
    "run_partition": {"run_partition": "run-b"},
    "execution_identity": {"execution_id": "execution-mismatch"},
}


@pytest.mark.parametrize("condition", sorted(_CONDITION_BREAKS))
def test_ta3_corrupt_conditions(
    tmp_path: Path, condition: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(
        logging.ERROR, logger="xagent.web.services.task_interaction_anchor"
    )
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    overrides = dict(_CONDITION_BREAKS[condition])

    kwargs: dict[str, Any] = {}
    if overrides.pop("cross_task", False):
        other = _make_task(db, run_id=None)
        kwargs["row_task_id"] = other.id
    kwargs.update(overrides)

    task, _row = _build_scenario(db, **kwargs)

    result = resolve_interaction_anchor(db, task)

    assert result is None
    assert ops_signals.INTERACTION_ANCHOR_CORRUPT in ops_signals.active_degradations()
    # Corrupt registers no counter -- only the three absence/unavailable
    # outcomes do (see the resolver's own docstring: corrupt uses
    # ops_signals, not the counter registry).
    assert ir.counters_snapshot() == {}
    # Step 4's judgment-table entry is a logger.error call -- same rationale
    # as the dangling-pointer case's warning pin above, applied uniformly
    # across all six parametrized conditions since they all fall through
    # the same log call site.
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1, caplog.records
    assert errors[0].msg == "task %s's checkpoint pointer %s failed anchor validation"
    db.close()


# ---------------------------------------------------------------------------
# task.run_id IS NULL classifies as absence even when the pointer names a
# row that would otherwise fail a corrupt condition -- proving step 1
# short-circuits before the row is ever evaluated, not merely that its own
# outcome happens to be absence.
# ---------------------------------------------------------------------------


def test_ta4_no_run_id_short_circuits_before_corrupt_row_is_read(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    other = _make_task(db, run_id=None)
    task, _row = _build_scenario(
        db,
        task_run_id=None,
        row_task_id=other.id,  # ownership condition would fail
    )

    result = resolve_interaction_anchor(db, task)

    assert result is None
    assert ops_signals.active_degradations() == {}
    assert ir.counters_snapshot() == {ir.COUNTER_ANCHOR_ABSENT_NO_RUN: 1}
    db.close()


# ---------------------------------------------------------------------------
# A legacy-type row, same task, with no run-partition field at all -- the
# only shape a real legacy checkpoint row can have, since
# stage_trace_event_row (trace_event_staging.py) only ever writes that
# field for current-type checkpoints. Missing the field fails the
# run-partition self-consistency condition before the legacy-type check is
# even reached, so this classifies corrupt, not absent.
# ---------------------------------------------------------------------------


def test_legacy_type_row_without_run_partition_field_classifies_corrupt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(
        logging.ERROR, logger="xagent.web.services.task_interaction_anchor"
    )
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    legacy_type = next(iter(LEGACY_CHECKPOINT_TYPES))
    task, _row = _build_scenario(db, checkpoint_type=legacy_type, run_partition=None)

    result = resolve_interaction_anchor(db, task)

    assert result is None
    assert ops_signals.INTERACTION_ANCHOR_CORRUPT in ops_signals.active_degradations()
    # Corrupt registers no counter -- the legacy-absence counter this row
    # might look like it should reach is never incremented, because the
    # run-partition condition fails first.
    assert ir.counters_snapshot() == {}
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1, caplog.records
    assert errors[0].msg == "task %s's checkpoint pointer %s failed anchor validation"
    db.close()


# ---------------------------------------------------------------------------
# The one shape that does reach the legacy-absence outcome: a legacy-type
# row whose run-partition field is present and matches the task's run id.
# No current writer produces this shape -- stage_trace_event_row only ever
# writes the run-partition field for current-type checkpoints -- so this
# cell pins what the outcome does for a row shaped this way, rather than
# covering a path production data can take today.
# ---------------------------------------------------------------------------


def test_legacy_type_row_with_matching_run_partition_field_is_absence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="xagent.web.services.task_interaction_anchor")
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    legacy_type = next(iter(LEGACY_CHECKPOINT_TYPES))
    task, _row = _build_scenario(db, checkpoint_type=legacy_type)

    result = resolve_interaction_anchor(db, task)

    assert result is None
    assert ops_signals.active_degradations() == {}
    assert ir.counters_snapshot() == {
        ir.COUNTER_ANCHOR_ABSENT_LEGACY_CHECKPOINT_TYPE: 1
    }
    # Step 5's judgment-table entry is a logger.info call, distinct from
    # step 1's and step 2's own info-level messages (different format
    # strings) -- matched exactly, not just "any info record", so this
    # cell cannot pass on the strength of some other step's log call.
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1, caplog.records
    assert infos[0].msg == (
        "task %s's checkpoint pointer %s names a legacy checkpoint "
        "type %r; no interaction anchor to resolve"
    )
    db.close()


# ---------------------------------------------------------------------------
# A row belonging to another task is corrupt regardless of checkpoint_type
# -- ownership is checked (step 4) before the legacy-type absence
# classification (step 5), never bypassed by it. Both cells assert
# InteractionAnchor's constructor was never even called, not merely that
# the return value was None.
# ---------------------------------------------------------------------------


def test_ta6a_cross_task_row_modern_type_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _count_anchor_constructions(monkeypatch)
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    other = _make_task(db, run_id=None)
    task, _row = _build_scenario(
        db, row_task_id=other.id, checkpoint_type=CHECKPOINT_TYPE
    )

    result = resolve_interaction_anchor(db, task)

    assert result is None
    assert calls["n"] == 0
    assert ops_signals.INTERACTION_ANCHOR_CORRUPT in ops_signals.active_degradations()
    db.close()


def test_ta6b_cross_task_row_legacy_type_is_still_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Differs from the previous test in exactly one field: this row's
    owning task_id is the *other* task's, same as before, but here
    checkpoint_type is legacy. If step 5 (legacy -> absence) ran before
    step 4 (the six conditions, including ownership), this row would
    wrongly classify as absence instead of corrupt."""

    calls = _count_anchor_constructions(monkeypatch)
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    other = _make_task(db, run_id=None)
    legacy_type = next(iter(LEGACY_CHECKPOINT_TYPES))
    task, _row = _build_scenario(db, row_task_id=other.id, checkpoint_type=legacy_type)

    result = resolve_interaction_anchor(db, task)

    assert result is None
    assert calls["n"] == 0
    assert ops_signals.INTERACTION_ANCHOR_CORRUPT in ops_signals.active_degradations()
    db.close()


# ---------------------------------------------------------------------------
# Happy path -- all six conditions pass, checkpoint_type is current -> a
# resolved InteractionAnchor, every field asserted individually.
# ---------------------------------------------------------------------------


def test_ta7_happy_path_resolves_anchor(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    task, row = _build_scenario(
        db, task_run_id="run-a", run_partition="run-a", checkpoint_type=CHECKPOINT_TYPE
    )

    result = resolve_interaction_anchor(db, task)

    assert result is not None
    assert result.trace_event_id == row.id
    assert result.resume_event_id == row.event_id
    assert result.resume_execution_id == str(task.id)
    assert result.resume_run_partition == "run-a"
    assert result.resume_locator_format == "trace_event_pk_v1"
    assert result.resume_checkpoint_type == "agent_execution_checkpoint"
    assert ops_signals.active_degradations() == {}
    db.close()


# ---------------------------------------------------------------------------
# Happy path with a non-empty execution identity on both sides, matching --
# every other non-corrupt cell in this file leaves execution_id unset, which
# leaves the row's execution id empty and short-circuits condition 6's
# comparison before it ever runs. This cell builds the row directly (rather
# than through _build_scenario, which cannot take the task's own id before
# the task exists) so the row's execution id can be set to the task's id
# after creation -- the first cell where that comparison runs and comes out
# equal: two non-empty, equal execution ids should resolve, not corrupt.
# ---------------------------------------------------------------------------


def test_matching_non_empty_execution_identity_resolves_anchor(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    task = _make_task(db, run_id="run-a")
    row = _make_row(
        db,
        task_id=task.id,
        run_partition="run-a",
        execution_id=str(task.id),
    )
    _point(db, task, row.id)

    result = resolve_interaction_anchor(db, task)

    assert result is not None
    assert result.resume_execution_id == str(task.id)
    assert ops_signals.active_degradations() == {}
    db.close()


# ---------------------------------------------------------------------------
# resolve_interaction_anchor ships with zero production callers.
# ---------------------------------------------------------------------------

_ANCHOR_GATED_NAME = "resolve_interaction_anchor"
_ANCHOR_MODULE_STEM = "task_interaction_anchor"


def _anchor_production_uses(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if any(a.name == _ANCHOR_GATED_NAME for a in node.names):
                return True
            if (
                any(a.name == "*" for a in node.names)
                and (node.module or "").split(".")[-1] == _ANCHOR_MODULE_STEM
            ):
                return True
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == _ANCHOR_GATED_NAME:
                return True
            if isinstance(func, ast.Attribute) and func.attr == _ANCHOR_GATED_NAME:
                return True
    return False


def _scan_root(exclude_stem: str) -> list[Path]:
    root = Path(next(iter(xagent.__path__)))
    modules = [p for p in root.rglob("*.py") if p.stem != exclude_stem]
    assert modules, "production scan set is empty"
    return modules


def test_ta11_resolve_interaction_anchor_has_zero_production_callers() -> None:
    """This is a fresh assertion, not an extension of
    ``test_interaction_staging_production_gate.py``'s gate: that gate scans
    only for ``stage_interaction_request`` and ``interaction_handoff`` by
    name (``GATED_NAMES``), and this function is not one of them.

    The zero here is scoped to this module's introduction, not a permanent
    invariant to preserve by construction: the change that wires the first
    production caller changes this number to 3 (the three finalizers named
    in ``task_interaction_staging.py``'s own module docstring) and updates
    this test alongside that wiring, deliberately -- it is not a constant a
    later change edits in isolation without also reasoning about the wiring
    itself.
    """

    offenders: dict[str, bool] = {}
    for path in _scan_root(_ANCHOR_MODULE_STEM):
        source = path.read_text(encoding="utf-8")
        if _anchor_production_uses(source):
            offenders[str(path)] = True
    assert offenders == {}
