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
import inspect
import logging
import textwrap
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.interaction_static_scan_shared import _scan_root
from tests.web.services.task_interaction_schema_shared import make_user
from xagent.core.agent.checkpoint import CHECKPOINT_TYPE, LEGACY_CHECKPOINT_TYPES
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.api import trace_handlers
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TraceEvent
from xagent.web.services import interaction_rollout as ir
from xagent.web.services import ops_signals
from xagent.web.services.task_interaction_anchor import resolve_interaction_anchor
from xagent.web.services.task_interaction_staging import (
    InteractionAnchor,
    stage_interaction_request,
)
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
# The row's data column holds something other than a dict -- the shape
# resolve_interaction_anchor's isinstance(row.data, dict) guard exists to
# handle without raising. A list payload coerces to an empty dict inside
# the resolver, which has no checkpoint_type key, so it fails the
# checkpoint-type self-consistency condition immediately and lands on
# corrupt, the same outcome the six-condition table above produces. A None
# payload would exercise the same guard, but trace_events.data is a
# NOT NULL column, so committing a row with data=None raises an
# IntegrityError before this function ever runs -- that shape cannot be
# built through a committed row at all, so it is not covered here.
# ---------------------------------------------------------------------------


def test_non_dict_row_data_classifies_corrupt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(
        logging.ERROR, logger="xagent.web.services.task_interaction_anchor"
    )
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    task = _make_task(db, run_id="run-a")
    # Built directly rather than through _make_row, which always writes a
    # dict -- this is the one caller that needs a non-dict payload.
    row = TraceEvent(
        task_id=task.id,
        build_id=None,
        event_id=f"anchor-event-{next(_event_counter)}",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        data=["not", "a", "dict"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    _point(db, task, row.id)

    result = resolve_interaction_anchor(db, task)

    assert result is None
    assert ops_signals.INTERACTION_ANCHOR_CORRUPT in ops_signals.active_degradations()
    assert ir.counters_snapshot() == {}
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
# field for current-type checkpoints. Missing the field fails only the
# run-partition self-consistency condition, with every other condition
# passing -- the shape resolve_interaction_anchor's own docstring
# describes as a pre-existing row, not data corruption, so this
# classifies as absence and reuses the same counter step 5 uses for an
# ordinary legacy-type row.
# ---------------------------------------------------------------------------


def test_legacy_type_row_without_run_partition_field_classifies_absence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="xagent.web.services.task_interaction_anchor")
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    legacy_type = next(iter(LEGACY_CHECKPOINT_TYPES))
    task, _row = _build_scenario(db, checkpoint_type=legacy_type, run_partition=None)

    result = resolve_interaction_anchor(db, task)

    assert result is None
    assert ops_signals.active_degradations() == {}
    # Reclassified as absence, not corrupt -- the missing-run-partition
    # row reuses the existing legacy-type counter, not the new
    # missing-run-partition one, because its checkpoint_type is legacy.
    assert ir.counters_snapshot() == {
        ir.COUNTER_ANCHOR_ABSENT_LEGACY_CHECKPOINT_TYPE: 1
    }
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1, caplog.records
    assert infos[0].msg == (
        "task %s's checkpoint pointer %s is missing its run-partition "
        "field and names a legacy checkpoint type %r; no interaction "
        "anchor to resolve"
    )
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
    assert ir.counters_snapshot() == {ir.COUNTER_ANCHOR_RESOLVED: 1}
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
# A resolved anchor fed straight into stage_interaction_request: the two
# primitives have zero tests that exercise both together today, so nothing
# proves the write-direction resolver's output actually satisfies the
# staging primitive's own anchor validation and run-partition comparison.
# ---------------------------------------------------------------------------


def test_resolved_anchor_stages_successfully(tmp_path: Path) -> None:
    """Feeds resolve_interaction_anchor's output directly into
    stage_interaction_request with no intervening conversion, proving there
    is none for a caller to have to supply."""

    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    task, row = _build_scenario(
        db, task_run_id="run-a", run_partition="run-a", checkpoint_type=CHECKPOINT_TYPE
    )

    anchor = resolve_interaction_anchor(db, task)
    assert anchor is not None

    staged = stage_interaction_request(
        db,
        task_id=task.id,
        run_id=task.run_id,
        anchor=anchor,
        kind="clarification",
        protocol_version=1,
        origin="internal",
        request_payload={"message": "pick one", "interactions": []},
        request_idempotency_key=f"key-{next(_event_counter)}",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        now=datetime.now(timezone.utc),
    )
    db.commit()

    assert staged.created is True
    db.close()


# ---------------------------------------------------------------------------
# Structural equivalence between this module's six row-validity conditions
# and trace_handlers.py's own copy. The two stay separate functions on
# purpose -- their failure actions differ (this one returns None and
# degrades, that one raises CheckpointCorruptError) -- but the six-condition
# disjunction itself, and the order its six terms run in, must be kept
# identical by construction, not merely by review. This is a structural
# check, not a textual one: one known-legitimate rename (``task.id`` here
# versus ``self.task_id`` there, since one is a bare function and the other
# a bound method) is normalized away before comparing; everything else must
# match verbatim.
# ---------------------------------------------------------------------------


class _NormalizeSelfTaskId(ast.NodeTransformer):
    """Rewrites ``self.task_id`` to ``task.id`` wherever it appears, the one
    legitimate rename between trace_handlers.py's bound-method form and this
    module's bare-function form. Every other identifier must already match
    verbatim between the two copies."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "task_id"
        ):
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id="task", ctx=ast.Load()),
                    attr="id",
                    ctx=ast.Load(),
                ),
                node,
            )
        return node


def _six_condition_disjunction(func: Any) -> list[ast.expr]:
    """The six ``ast.BoolOp(op=Or)`` operands inside ``func``'s corrupt-row
    ``if`` statement. Located by shape (a 6-way ``or`` inside the function
    body), not by line number, since the two copies live at different
    offsets in their own files."""

    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.Or)
        and len(node.values) == 6
    ]
    assert len(candidates) == 1, (
        f"expected exactly one 6-way `or` in {func.__qualname__}, "
        f"found {len(candidates)}"
    )
    return candidates[0].values


def _normalized_dump(node: ast.expr) -> str:
    rewritten = _NormalizeSelfTaskId().visit(node)
    ast.fix_missing_locations(rewritten)
    return ast.dump(rewritten, annotate_fields=False)


def test_row_validity_conditions_match_trace_handlers_structurally() -> None:
    """If this test turns red, the six row-validity conditions
    ``resolve_interaction_anchor`` checks and the six
    ``_load_pk_anchored_checkpoint`` (``trace_handlers.py``) checks have
    drifted apart -- in content, or in the order they run in, which this
    module's own docstring states is a contract (condition 4 must precede
    condition 5). Bring the two back into alignment rather than editing
    this test to match whichever side changed, unless the divergence was
    deliberate -- in which case update the docstrings on both sides that
    say the two are kept identical, not just this test.
    """

    anchor_conditions = _six_condition_disjunction(resolve_interaction_anchor)
    trace_handler_conditions = _six_condition_disjunction(
        trace_handlers.DatabaseTraceHandler._load_pk_anchored_checkpoint
    )

    anchor_dumps = [_normalized_dump(node) for node in anchor_conditions]
    trace_handler_dumps = [_normalized_dump(node) for node in trace_handler_conditions]

    assert anchor_dumps == trace_handler_dumps


# ---------------------------------------------------------------------------
# The missing-run-partition reclassification inside the corrupt branch
# carries a second, hand-copied disjunction: the same six operands above,
# minus the partition-match one, combined with ``not (... or ...)`` instead
# of the six-condition ``or``. That copy cannot be factored into a shared
# helper (the six-condition ``or`` is pinned operand-for-operand against
# trace_handlers.py by the guard above), so this test pins the copy against
# the original directly instead of trusting the two to stay aligned by
# hand.
# ---------------------------------------------------------------------------

_PARTITION_TERM: ast.expr = ast.UnaryOp(
    op=ast.Not(), operand=ast.Name(id="partition_matches", ctx=ast.Load())
)


def _five_condition_disjunction(func: Any) -> list[ast.expr]:
    """The five ``ast.BoolOp(op=Or)`` operands inside ``func``'s
    ``only_partition_absent`` reclassification check. Located by shape (a
    5-way ``or`` inside the function body), the same way
    ``_six_condition_disjunction`` locates its own target."""

    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.Or)
        and len(node.values) == 5
    ]
    assert len(candidates) == 1, (
        f"expected exactly one 5-way `or` in {func.__qualname__}, found "
        f"{len(candidates)}"
    )
    return candidates[0].values


def test_missing_partition_discriminator_mirrors_the_main_disjunction() -> None:
    """The five conditions ``only_partition_absent`` checks must be the same
    five conditions the six-condition disjunction above carries, minus the
    partition-match term, in the same order. If this test turns red, the
    two copies have drifted apart -- bring the ``only_partition_absent``
    copy back into alignment with the six-condition ``or`` rather than
    editing this test, the same rule
    ``test_row_validity_conditions_match_trace_handlers_structurally``
    applies to its own pair.
    """

    outer = _six_condition_disjunction(resolve_interaction_anchor)
    inner = _five_condition_disjunction(resolve_interaction_anchor)

    partition_term_dump = _normalized_dump(_PARTITION_TERM)
    expected = [node for node in outer if _normalized_dump(node) != partition_term_dump]
    assert len(expected) == 5, (
        "expected the partition-match term to match exactly one of the six "
        f"outer operands, found {6 - len(expected)} matching operands"
    )

    assert [_normalized_dump(n) for n in inner] == [
        _normalized_dump(n) for n in expected
    ]


# ---------------------------------------------------------------------------
# W5-R3-3: every one of the six judgment-table outcomes, one cell each, in a
# single parametrized table. Each cell is also covered individually above;
# this suite exists so a missing or misplaced ``increment_counter`` call on
# any one outcome shows up as one obviously-named failing cell here, rather
# than only as a hard-to-spot assertion change buried in that outcome's own
# test.
# ---------------------------------------------------------------------------


def _six_outcome_no_run(db: Session) -> Task:
    return _make_task(db, run_id=None)


def _six_outcome_no_pointer(db: Session) -> Task:
    return _make_task(db, run_id="run-a")


def _six_outcome_dangling_pointer(db: Session) -> Task:
    task = _make_task(db, run_id="run-a")
    _point_dangling(db, task, 999_999)
    return task


def _six_outcome_corrupt(db: Session) -> Task:
    other = _make_task(db, run_id=None)
    task, _row = _build_scenario(db, row_task_id=other.id)
    return task


def _six_outcome_legacy_absence(db: Session) -> Task:
    legacy_type = next(iter(LEGACY_CHECKPOINT_TYPES))
    task, _row = _build_scenario(db, checkpoint_type=legacy_type)
    return task


def _six_outcome_resolved(db: Session) -> Task:
    task, _row = _build_scenario(
        db, task_run_id="run-a", run_partition="run-a", checkpoint_type=CHECKPOINT_TYPE
    )
    return task


_SIX_OUTCOME_CELLS: dict[str, tuple[Any, dict[str, int]]] = {
    "step1_no_run": (_six_outcome_no_run, {ir.COUNTER_ANCHOR_ABSENT_NO_RUN: 1}),
    "step2_no_pointer": (_six_outcome_no_pointer, {}),
    "step3_dangling_pointer": (
        _six_outcome_dangling_pointer,
        {ir.COUNTER_ANCHOR_UNAVAILABLE_DANGLING_POINTER: 1},
    ),
    "step4_corrupt": (_six_outcome_corrupt, {}),
    "step5_legacy_absence": (
        _six_outcome_legacy_absence,
        {ir.COUNTER_ANCHOR_ABSENT_LEGACY_CHECKPOINT_TYPE: 1},
    ),
    "step6_resolved": (_six_outcome_resolved, {ir.COUNTER_ANCHOR_RESOLVED: 1}),
}


@pytest.mark.parametrize("outcome", sorted(_SIX_OUTCOME_CELLS))
def test_ta12_six_outcome_counter_matrix(tmp_path: Path, outcome: str) -> None:
    engine = _engine(tmp_path)
    db = _session_factory(engine)()
    builder, expected_counters = _SIX_OUTCOME_CELLS[outcome]
    task = builder(db)

    resolve_interaction_anchor(db, task)

    if outcome == "step4_corrupt":
        assert ops_signals.INTERACTION_ANCHOR_CORRUPT in (
            ops_signals.active_degradations()
        )
    else:
        assert ops_signals.active_degradations() == {}
    assert ir.counters_snapshot() == expected_counters
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
