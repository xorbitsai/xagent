"""Coverage for ``task_interaction_read.get_pending_interaction_question``:
the tuple adapter over ``materialize_compatibility_view``.

Organized by the seven-tier projection table in the module's own
docstring (T0 through T3'''), plus the two hard shape requirements: step 0
costs no query into the interaction table (A5), and the adapter never
filters or reshapes what it receives (A14).
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Select, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

import xagent.web.services.chat_history_service as chat_history_service
import xagent.web.services.task_interaction_service as interaction_service_module
from tests.web.services.task_interaction_schema_shared import (
    anchor_event_id,
    make_task,
    make_user,
)
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import task_interaction_read as read_surface
from xagent.web.services.chat_history_service import persist_assistant_message
from xagent.web.services.interaction_rollout import (
    COUNTER_COMPAT_READ_FALLBACK,
    counters_snapshot,
)
from xagent.web.services.ops_signals import (
    CHECKPOINT_LOAD_UNAVAILABLE,
    CHECKPOINT_PK_ANCHOR_DANGLING,
    INTERACTION_READ_PAYLOAD_UNREADABLE,
    INTERACTION_READ_PROTOCOL_UNRECOGNIZED,
    INTERACTION_READ_TASK_MARKER_UNRECOGNIZED,
    active_degradations,
    clear_degradation,
)
from xagent.web.services.task_interaction_service import CompatibilityQuestionView
from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD

_DEGRADATION_SIGNALS_UNDER_TEST = (
    CHECKPOINT_PK_ANCHOR_DANGLING,
    CHECKPOINT_LOAD_UNAVAILABLE,
    INTERACTION_READ_PROTOCOL_UNRECOGNIZED,
    INTERACTION_READ_PAYLOAD_UNREADABLE,
    INTERACTION_READ_TASK_MARKER_UNRECOGNIZED,
)


@pytest.fixture(autouse=True)
def _clean_degradation_registry():
    """A10-A12 exercise materialize_compatibility_view's failure branches,
    which register process-global degradation signals; clear them around
    every test so they cannot leak into tests that read the shared
    registry (the /health suite asserts exact payloads and fails on any
    leftover entry). Same fixture as test_task_interaction_service.py's,
    duplicated rather than shared because pytest fixtures are file-local
    unless hoisted to a conftest.py, which is out of this delivery's
    scope."""
    for signal in _DEGRADATION_SIGNALS_UNDER_TEST:
        clear_degradation(signal)
    yield
    for signal in _DEGRADATION_SIGNALS_UNDER_TEST:
        clear_degradation(signal)


@pytest.fixture
def _engine(tmp_path: Path):
    db_path = tmp_path / "task_interaction_read.db"
    engine = create_engine(f"sqlite:///{db_path}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def _session_factory(_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture
def _db(_session_factory) -> Session:
    db = _session_factory()
    try:
        yield db
    finally:
        db.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_task(db: Session, *, marker: int | None) -> Task:
    """A loaded, committed Task row with the given
    interaction_protocol_version -- the object this adapter's own
    signature requires (a row, not an id)."""

    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.run_id = "run-a"
    task.interaction_protocol_version = marker
    db.commit()
    db.refresh(task)
    return task


def _make_trace_event(
    db: Session,
    *,
    task_id: int,
    run_partition: str = "run-a",
    execution_id: str = "exec-1",
    checkpoint_type: str = "agent_execution_checkpoint",
) -> int:
    event = TraceEvent(
        task_id=task_id,
        event_id=f"read-trace-event-{task_id}",
        event_type=str(CHECKPOINT_EVENT_TYPE),
        timestamp=_now(),
        build_id=None,
        data={
            TASK_RUN_ID_TRACE_FIELD: run_partition,
            "checkpoint_type": checkpoint_type,
            "execution_id": execution_id,
        },
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return int(event.id)


def _make_active_row(
    db: Session,
    *,
    task_id: int,
    resume_trace_event_id: int,
    run_id: str = "run-a",
    resume_run_partition: str = "run-a",
    resume_execution_id: str = "exec-1",
    protocol_version: int = 1,
    request_payload: dict[str, Any] | None = None,
) -> TaskInteractionRequest:
    now = _now()
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id=run_id,
        kind="clarification",
        protocol_version=protocol_version,
        status="active",
        active_slot=1,
        origin="internal",
        request_payload=request_payload
        if request_payload is not None
        else {
            "message": "Which environment?",
            "interactions": [
                {"type": "text_input", "field": "env", "label": "Environment"}
            ],
        },
        request_idempotency_key=f"read-key-{task_id}",
        resume_trace_event_id=resume_trace_event_id,
        resume_event_id=anchor_event_id(db, resume_trace_event_id),
        resume_execution_id=resume_execution_id,
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition=resume_run_partition,
        created_at=now,
        expires_at=now + timedelta(minutes=15),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ---------------------------------------------------------------------------
# T0 -- the protocol marker fast path. Every cell here must not depend on
# the task_interaction_requests table's contents at all.
# ---------------------------------------------------------------------------


def test_a1_marker_null_reads_the_live_legacy_question(_db: Session) -> None:
    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


def test_a2_marker_null_recovers_a_superseded_only_row(_db: Session) -> None:
    """A NULL marker means no native row holds this task's answer slot, so
    a question row a structured publication already relabelled to
    ``question_superseded`` is the only record of what this task is asking
    -- and the honest answer, not an empty pair. Step 0 is the one caller
    that opens ``allow_superseded``, which is what makes the second pass
    run here after the first finds no live row. Mutation: pass
    ``allow_superseded=False`` from step 0 and this turns red -- the pair
    comes back empty and the question is unreachable."""

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "An old question",
        message_type="question_superseded",
        interactions=[{"type": "text_input", "label": "Old"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("An old question")
    assert interactions == [{"type": "text_input", "label": "Old"}]


def test_a3_marker_null_prefers_the_live_row_over_a_higher_id_superseded_row(
    _db: Session,
) -> None:
    """A live question always wins over a higher-id superseded row --
    trivially true once superseded rows are never matched at all (see
    A2), but pinned here as its own cell so a future change that starts
    matching both message types in one predicate is caught at the level
    where it would actually invert priority. Mutation: match both
    ``QUESTION_MESSAGE_TYPE`` and ``SUPERSEDED_MESSAGE_TYPE`` in one
    ``message_type.in_(...)`` predicate ordered by id and this test turns
    red only if the superseded row is given the higher id, which it is
    here."""

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A newer superseded row",
        message_type="question_superseded",
        interactions=[{"type": "text_input", "label": "Superseded"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


def test_a4_unrecognized_marker_still_routes_through_the_rich_view(
    _db: Session,
) -> None:
    """A marker that is neither 1 nor NULL is not the fast-path's business
    to interpret -- only NULL means no native row was ever staged and skips
    the interaction table. Any other value, including one this reader does
    not otherwise recognize, is handed to the rich view, which is the only
    place that knows whether an active native row exists. Here there is no
    active row, so the rich view's own T1 fallback answers from the same
    single-pass legacy reader the NULL cell uses (A1), correctly, even
    though this marker value cannot occur on a persisted row (see below).
    The real behavioral difference from the old fast-path routing --
    whether an active native row's answer can be bypassed -- is not
    observable here since there is no active row to bypass; A4b below is
    the cell that pins that difference. The marker is set on the
    in-memory ORM object without a commit: ck_tasks_interaction_protocol_version
    pins the column to NULL-or-1 in the database, so persisting 2 would
    fail the CHECK; this adapter only ever reads the attribute, never
    writes it, so exercising the branch this way is faithful to what the
    adapter actually does."""

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )
    task.interaction_protocol_version = 2

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


def test_a4b_unrecognized_marker_with_an_active_native_row_never_leaks_the_legacy_question(
    _db: Session,
) -> None:
    """The invariant this fixes: once an active native row holds this
    task's current-run answer slot, the read surface must never surface a
    legacy transcript question instead, no matter what value happens to
    sit in ``tasks.interaction_protocol_version``. A marker of 2 used to
    take the fast path (``marker != 1``) and answer straight from the
    transcript without ever looking at the interaction table, exposing a
    stale legacy question while a native row was live. Mutation: change
    the gate condition back to ``marker != 1`` and this test turns red --
    the legacy question would be returned instead of the native one."""

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live legacy question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Legacy"}],
    )
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    _make_active_row(_db, task_id=int(task.id), resume_trace_event_id=trace_event_id)
    task.interaction_protocol_version = 2

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question == "Which environment?"
    assert interactions == [
        {
            "type": "text_input",
            "field": "env",
            "label": "Environment",
            "options": None,
            "placeholder": None,
            "multiline": False,
            "min": None,
            "max": None,
            "default_value": None,
            "accept": None,
            "multiple": False,
        }
    ]


def test_a4c_unrecognized_marker_registers_a_degradation_signal(
    _db: Session,
) -> None:
    """A4 pins where an unrecognized marker routes; this pins that it is
    also reported. Every corruption branch inside the compatibility view
    raises a signal and this one did not, leaving an operator nothing to
    look at. Keyed and process-local, so the assertion reads the registry
    rather than logs. Mutation: drop the register call and this turns red."""

    task = _make_task(_db, marker=None)
    _persist_live_question(_db, task)
    task.interaction_protocol_version = 2

    read_surface.get_pending_interaction_question(_db, task)

    detail = active_degradations()[INTERACTION_READ_TASK_MARKER_UNRECOGNIZED]
    assert f"task {task.id}" in detail


def test_a5_marker_fast_path_never_calls_the_rich_view(
    _db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 0 must cost zero queries into the interaction table -- the
    module docstring's central shape claim. Proven by making the rich
    view explode if it is ever called, then running A1's own scenario
    and confirming it still passes. Mutation: delete the marker check (so
    every task always calls the rich view) and this test turns red."""

    def _explode(db: Session, task_id: int, *, allow_superseded: bool = False):
        raise AssertionError("materialize_compatibility_view must not be called")

    monkeypatch.setattr(read_surface, "materialize_compatibility_view", _explode)

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


# ---------------------------------------------------------------------------
# T1 -- marker == 1, but the rich view itself falls back to the legacy
# transcript (table absent, or no active native row on this run).
# ---------------------------------------------------------------------------


def test_a6_marker_matches_no_active_row_reads_the_legacy_transcript(
    _db: Session,
) -> None:
    task = _make_task(_db, marker=1)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


def test_a7_marker_matches_table_absent_reads_the_legacy_transcript(
    _db: Session,
) -> None:
    task = _make_task(_db, marker=1)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )
    TaskInteractionRequest.__table__.drop(bind=_db.get_bind())

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


def test_a8_marker_matches_no_active_row_recovers_a_superseded_only_row(
    _db: Session,
) -> None:
    """T1's "no active row" fallback reaches relabelled rows for the same
    reason the NULL-marker cell does (see A2): no native row holds this
    task's answer slot, so the relabelled transcript row is the only record
    of the question and answering it lands nowhere a native row has
    claimed. The gate value travels marker -> adapter ->
    ``materialize_compatibility_view`` -> ``_legacy_view``. Mutation: stop
    threading ``allow_superseded`` into either legacy branch of the
    compatibility view and this turns red."""

    task = _make_task(_db, marker=1)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "An old question",
        message_type="question_superseded",
        interactions=[{"type": "text_input", "label": "Old"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("An old question")
    assert interactions == [{"type": "text_input", "label": "Old"}]


# ---------------------------------------------------------------------------
# Superseded recovery: what the read surface answers once a structured
# publication has relabelled the transcript row it replaced.
# ---------------------------------------------------------------------------


def _make_terminated_row(db: Session, *, task_id: int) -> None:
    """One interaction row in a terminal state: it holds no answer slot, so
    the active-row predicate does not match it and the compatibility view
    falls back to the transcript."""

    now = _now()
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id="run-a",
        kind="clarification",
        protocol_version=1,
        status="terminated",
        active_slot=None,
        origin="internal",
        request_payload={"message": "Which environment?", "interactions": []},
        request_idempotency_key=f"read-terminal-key-{task_id}",
        resume_trace_event_id=None,
        resume_event_id="resume-event-1",
        resume_execution_id="exec-1",
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition="run-a",
        terminal_reason="deadline_elapsed",
        terminated_at=now,
        created_at=now,
        expires_at=now + timedelta(minutes=15),
    )
    db.add(row)
    db.commit()


def test_read_surface_recovers_a_superseded_question(_db: Session) -> None:
    """The steady state a structured publication leaves behind on a task
    whose marker was never advanced: the task is waiting, its interaction
    row has reached a terminal state, and the transcript row that carried
    the question has been relabelled. Nothing holds the answer slot, so the
    relabelled row is what this task is asking and the read surface hands
    it back."""

    task = _make_task(_db, marker=None)
    task.status = TaskStatus.WAITING_FOR_USER
    _db.commit()
    _make_terminated_row(_db, task_id=int(task.id))
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "Which environment?",
        message_type="question_superseded",
        interactions=[{"type": "text_input", "label": "Environment"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("Which environment?")
    assert interactions == [{"type": "text_input", "label": "Environment"}]


def test_recovery_still_applies_when_an_active_row_exists_under_a_null_marker(
    _db: Session,
) -> None:
    """A known boundary, pinned as current behavior rather than as the
    behavior anyone wants: step 0 reads the marker off the task row and
    queries nothing else, so a NULL marker sitting on a task that does have
    an active interaction row still opens the gate and answers from the
    relabelled transcript row.

    The pair "active row staged" and "marker advanced to 1" is written by
    the staging side, which is what keeps this combination from occurring;
    the read surface does not second-guess it, and the superseded fallback
    does not take that duty on either."""

    task = _make_task(_db, marker=None)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    _make_active_row(_db, task_id=int(task.id), resume_trace_event_id=trace_event_id)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "An old question",
        message_type="question_superseded",
        interactions=[{"type": "text_input", "label": "Old"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("An old question")
    assert interactions == [{"type": "text_input", "label": "Old"}]


# ---------------------------------------------------------------------------
# T2 -- an active native row whose anchor resolves.
# ---------------------------------------------------------------------------


def test_a9_native_projection(_db: Session) -> None:
    task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    _make_active_row(_db, task_id=int(task.id), resume_trace_event_id=trace_event_id)

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question == "Which environment?"
    assert interactions == [
        {
            "type": "text_input",
            "field": "env",
            "label": "Environment",
            "options": None,
            "placeholder": None,
            "multiline": False,
            "min": None,
            "max": None,
            "default_value": None,
            "accept": None,
            "multiple": False,
        }
    ]


# ---------------------------------------------------------------------------
# T3 family -- an active native row this reader cannot answer with.
# ---------------------------------------------------------------------------


def test_a10_anchor_dangling_keeps_the_question_text_drops_controls(
    _db: Session,
) -> None:
    task = _make_task(_db, marker=1)
    # A trace event on a different run partition than the interaction
    # row's own resume_run_partition breaks the anchor's row-validity
    # judgment -- one of _resolve_read_direction_anchor's seven conditions
    # -- producing "anchor_dangling" through a real write, no monkeypatch.
    trace_event_id = _make_trace_event(
        _db, task_id=int(task.id), run_partition="a-different-run"
    )
    _make_active_row(_db, task_id=int(task.id), resume_trace_event_id=trace_event_id)

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question == "Which environment?"
    assert interactions is None


def test_a11_checkpoint_unavailable_keeps_the_question_text_drops_controls(
    _db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    _make_active_row(_db, task_id=int(task.id), resume_trace_event_id=trace_event_id)

    real_get = _db.get

    def _raising_get(model: Any, pk: Any, *args: Any, **kwargs: Any) -> Any:
        if model is TraceEvent:
            import sqlalchemy as sa

            raise sa.exc.OperationalError(
                "SELECT 1", {}, Exception("simulated session failure")
            )
        return real_get(model, pk, *args, **kwargs)

    monkeypatch.setattr(_db, "get", _raising_get)

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question == "Which environment?"
    assert interactions is None


def test_a12_unrecognized_protocol_version_drops_both_slots(
    _db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    row = _make_active_row(
        _db, task_id=int(task.id), resume_trace_event_id=trace_event_id
    )
    row.protocol_version = 2

    import xagent.web.services.task_interaction_service as svc

    def _fake_active_row(db: Session, task_id: int) -> TaskInteractionRequest:
        return row

    monkeypatch.setattr(svc, "_active_native_row", _fake_active_row)

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert (question, interactions) == (None, None)


def test_a13_unparseable_payload_drops_both_slots(_db: Session) -> None:
    task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    _make_active_row(
        _db,
        task_id=int(task.id),
        resume_trace_event_id=trace_event_id,
        request_payload={"not": "a valid v1 payload"},
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert (question, interactions) == (None, None)


def test_a13b_unanswerable_tier_never_leaks_interactions_even_if_the_view_carried_some(
    _db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the adapter's own ``if view.tier == "unanswerable": return
    view.question, None`` branch, not just the outcome it currently
    coincides with. Today all four ``materialize_compatibility_view`` code
    paths that produce the "unanswerable" tier (A10-A13) already
    hard-code ``interactions=None`` on the view itself, so a test that
    only drives the real view -- like A10-A13 -- cannot tell this
    branch apart from a naive ``return view.question, view.interactions``:
    both would return None either way. This test stubs the view directly
    to hand back an "unanswerable" tier that *does* carry interaction
    controls, so it fails unless the adapter itself refuses to forward
    them: this is the one place that guarantees a future
    materialize_compatibility_view change which starts populating
    interactions on this tier still cannot leak controls for a question
    the adapter has decided cannot be answered right now.

    Mutation: change the branch to ``return view.question,
    view.interactions`` and this test turns red; A10-A13 stay green."""

    def _fake_view(
        db: Session, task_id: int, *, allow_superseded: bool = False
    ) -> CompatibilityQuestionView:
        return CompatibilityQuestionView(
            tier="unanswerable",
            question="q",
            interactions=[{"type": "text_input", "field": "x", "label": "X"}],
            reason="anchor_dangling",
        )

    monkeypatch.setattr(read_surface, "materialize_compatibility_view", _fake_view)

    task = _make_task(_db, marker=1)

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert (question, interactions) == ("q", None)


# ---------------------------------------------------------------------------
# Shape invariants that hold across tiers.
# ---------------------------------------------------------------------------


def test_a14_adapter_does_not_filter_non_dict_interaction_elements(
    _db: Session,
) -> None:
    """The non-dict element filter belongs to the v1 layer, not this
    adapter or the shared reader -- see _filter_interaction_descriptors'
    own docstring for why it does not sink down. Mutation: have this
    adapter filter non-dict elements out of what it returns and this test
    turns red."""

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}, "not-a-dict"],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert interactions == [{"type": "text_input", "label": "Live"}, "not-a-dict"]


def test_a15_interaction_element_key_sets_differ_between_legacy_and_native_tiers(
    _db: Session,
) -> None:
    """The known gap this delivery does not converge (see the
    adapter's own module docstring): pins the two key sets so a future
    change that accidentally aligns or further diverges them is visible
    here first."""

    legacy_task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(legacy_task.id),
        int(legacy_task.user_id),
        "Which environment?",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Environment"}],
    )
    _, legacy_interactions = read_surface.get_pending_interaction_question(
        _db, legacy_task
    )
    assert legacy_interactions is not None
    assert set(legacy_interactions[0].keys()) == {"type", "label"}

    native_task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(native_task.id))
    _make_active_row(
        _db, task_id=int(native_task.id), resume_trace_event_id=trace_event_id
    )
    _, native_interactions = read_surface.get_pending_interaction_question(
        _db, native_task
    )
    assert native_interactions is not None
    assert set(native_interactions[0].keys()) == {
        "type",
        "field",
        "label",
        "options",
        "placeholder",
        "multiline",
        "min",
        "max",
        "default_value",
        "accept",
        "multiple",
    }


# ---------------------------------------------------------------------------
# The gate value itself: which of the read surface's outcomes lets the
# transcript reader reach relabelled rows, and which does not. Every cell
# below records the ``allow_superseded`` value that actually arrived at
# ``get_latest_waiting_question``, from whichever of its two call sites ran.
# ---------------------------------------------------------------------------


@contextmanager
def _recorded_gate_values(monkeypatch: pytest.MonkeyPatch):
    """Every ``allow_superseded`` value ``get_latest_waiting_question``
    receives while the block runs, in call order, patched at both of its
    call sites: step 0 in the adapter and ``_legacy_view`` in the
    compatibility view. An empty list means the transcript reader was never
    reached at all."""

    seen: list[bool] = []
    real = chat_history_service.get_latest_waiting_question

    def _record(db, task_id, *, allow_superseded: bool = False):
        seen.append(allow_superseded)
        return real(db, task_id, allow_superseded=allow_superseded)

    monkeypatch.setattr(read_surface, "get_latest_waiting_question", _record)
    monkeypatch.setattr(
        interaction_service_module, "get_latest_waiting_question", _record
    )
    yield seen


def _gate_marker_null(db: Session) -> Task:
    task = _make_task(db, marker=None)
    _persist_live_question(db, task)
    return task


def _gate_unrecognized_marker(db: Session) -> Task:
    task = _make_task(db, marker=None)
    _persist_live_question(db, task)
    # ck_tasks_interaction_protocol_version pins the column to NULL-or-1, so
    # this value is set on the in-memory row only -- the same construction
    # A4 uses, and faithful because the adapter only ever reads it.
    task.interaction_protocol_version = 2
    return task


def _gate_table_absent(db: Session) -> Task:
    task = _make_task(db, marker=1)
    _persist_live_question(db, task)
    TaskInteractionRequest.__table__.drop(bind=db.get_bind())
    return task


def _gate_no_active_row(db: Session) -> Task:
    task = _make_task(db, marker=1)
    _persist_live_question(db, task)
    return task


def _gate_unanswerable_payload(db: Session) -> Task:
    task = _make_task(db, marker=1)
    trace_event_id = _make_trace_event(db, task_id=int(task.id))
    _make_active_row(
        db,
        task_id=int(task.id),
        resume_trace_event_id=trace_event_id,
        request_payload={"not": "a valid v1 payload"},
    )
    return task


def _gate_unanswerable_anchor(db: Session) -> Task:
    task = _make_task(db, marker=1)
    trace_event_id = _make_trace_event(
        db, task_id=int(task.id), run_partition="a-different-run"
    )
    _make_active_row(db, task_id=int(task.id), resume_trace_event_id=trace_event_id)
    return task


def _persist_live_question(db: Session, task: Task) -> None:
    persist_assistant_message(
        db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )


_GATE_CASES: dict[str, tuple[Any, list[bool]]] = {
    # No native row can belong to this task, so the relabelled transcript
    # row is reachable.
    "marker_null": (_gate_marker_null, [True]),
    # A marker value this reader does not recognize means the slot's state
    # is unknown, and unknown closes the gate.
    "unrecognized_marker": (_gate_unrecognized_marker, [False]),
    # Both compatibility-view fallbacks mean nothing holds the slot.
    "table_absent": (_gate_table_absent, [True]),
    "no_active_row": (_gate_no_active_row, [True]),
    # The unanswerable tiers never reach the transcript reader at all.
    "unanswerable_payload": (_gate_unanswerable_payload, []),
    "unanswerable_anchor": (_gate_unanswerable_anchor, []),
}


@pytest.mark.parametrize("case", sorted(_GATE_CASES))
def test_gate_value_per_read_surface_outcome(
    _db: Session, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    build, expected = _GATE_CASES[case]
    task = build(_db)

    with _recorded_gate_values(monkeypatch) as seen:
        read_surface.get_pending_interaction_question(_db, task)

    assert seen == expected


def test_gate_is_closed_for_an_unrecognized_protocol_version_row(
    _db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The third unanswerable tier, which needs its own construction: an
    active row's protocol_version cannot be written to anything but 1
    (ck_task_interaction_requests_active_protocol), so it is mutated in
    memory behind the accessor that reads it, the same way A12 does."""

    task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    row = _make_active_row(
        _db, task_id=int(task.id), resume_trace_event_id=trace_event_id
    )
    row.protocol_version = 2
    monkeypatch.setattr(
        interaction_service_module, "_active_native_row", lambda db, task_id: row
    )

    with _recorded_gate_values(monkeypatch) as seen:
        read_surface.get_pending_interaction_question(_db, task)

    assert seen == []


# ---------------------------------------------------------------------------
# Statement count: step 0 still costs nothing beyond what the transcript
# reader itself costs.
# ---------------------------------------------------------------------------


@contextmanager
def _counted_selects(bind):
    seen: list[object] = []

    def _record(conn, clauseelement, multiparams, params, execution_options):
        if isinstance(clauseelement, Select):
            seen.append(clauseelement)

    event.listen(bind, "before_execute", _record)
    try:
        yield seen
    finally:
        event.remove(bind, "before_execute", _record)


@pytest.mark.parametrize("message_type", ["question", "question_superseded"])
def test_m3_marker_null_issues_no_statement_the_reader_would_not(
    _db: Session, message_type: str
) -> None:
    """Under a NULL marker the adapter must emit exactly the statements
    ``get_latest_waiting_question`` emits on its own -- one when the first
    pass hits, two when the gate is open and the first pass comes back
    empty -- and nothing else. Mutation: add any query to step 0, for
    instance a lookup into the interaction table, and this turns red."""

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A question",
        message_type=message_type,
    )
    # Load the task's own columns before counting: persist_assistant_message
    # commits, which expires them, and the refresh SELECT that first touch
    # would otherwise emit belongs to the fixture, not to either function
    # under measurement.
    _db.refresh(task)
    task_id = int(task.id)
    bind = _db.get_bind()

    with _counted_selects(bind) as through_adapter:
        read_surface.get_pending_interaction_question(_db, task)
    with _counted_selects(bind) as direct:
        chat_history_service.get_latest_waiting_question(
            _db, task_id, allow_superseded=True
        )

    assert len(through_adapter) == len(direct)
    assert len(direct) == (1 if message_type == "question" else 2)


# ---------------------------------------------------------------------------
# compat.read_fallback: where the counter is incremented, and where it must
# not be.
# ---------------------------------------------------------------------------


def _read_fallback_count() -> int:
    return counters_snapshot().get(COUNTER_COMPAT_READ_FALLBACK, 0)


@pytest.mark.parametrize("case", ["table_absent", "no_active_row"])
def test_each_compatibility_view_legacy_branch_counts_one_read_fallback(
    _db: Session, case: str
) -> None:
    task = _make_task(_db, marker=1)
    _persist_live_question(_db, task)
    if case == "table_absent":
        TaskInteractionRequest.__table__.drop(bind=_db.get_bind())

    before = _read_fallback_count()
    read_surface.get_pending_interaction_question(_db, task)

    assert _read_fallback_count() == before + 1


def test_the_marker_fast_path_counts_no_read_fallback(_db: Session) -> None:
    """The counter measures how often the compatibility view had to answer
    from the transcript, which is only meaningful next to how often it was
    asked. Counting step 0 as well would make it every read of a task whose
    marker is NULL -- today that is every task -- and the ratio would carry
    no information. Mutation: increment in step 0 and this turns red."""

    task = _make_task(_db, marker=None)
    _persist_live_question(_db, task)

    before = _read_fallback_count()
    read_surface.get_pending_interaction_question(_db, task)

    assert _read_fallback_count() == before


def test_a_recheck_that_finds_an_active_row_counts_no_read_fallback(
    _db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recheck exists to stop a legacy answer being returned, so the
    run it saves is not a fallback and must not be counted. The active row
    is made to appear on the second look only, which is the shape a
    concurrent publication produces.

    ``interactions`` is asserted rather than discarded: the
    ``anchor_dangling`` tier hands back the same question text with
    ``interactions=None``, so a test that only checked the text would stay
    green if the recheck's fallthrough landed there instead of on the
    native tier.
    """

    task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    row = _make_active_row(
        _db, task_id=int(task.id), resume_trace_event_id=trace_event_id
    )
    looks = 0

    def _appears_on_the_second_look(db, task_id: int):
        nonlocal looks
        looks += 1
        return None if looks == 1 else row

    monkeypatch.setattr(
        interaction_service_module, "_active_native_row", _appears_on_the_second_look
    )

    before = _read_fallback_count()
    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question == "Which environment?"
    assert interactions is not None
    assert _read_fallback_count() == before
