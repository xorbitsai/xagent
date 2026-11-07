"""Native clarification publication from the three task finalizers.

Each finalizer that settles a waiting round asks the rollout policy whether
native publication is allowed and, when it is, resolves the resume anchor
and the clarification draft, creates the interaction row through
``create()``, sets ``tasks.interaction_protocol_version`` to 1 and relabels
the task's transcript question rows -- all in its settlement transaction.

The three finalizers are driven directly, each in its own worker-owned
Session the way production calls them:

* ``fresh``: ``_finalize_task_execution_result_isolated``;
* ``resumed``: ``_finalize_resumed_task``;
* ``channel``: ``finalize_managed_task_lease_result_sync`` (the shared
  channel leaf's entry, which wraps ``finalize_managed_task_lease_result``
  in the commit witness driver), run as the coordinator that owns the lease
  unless a cell says otherwise.

For ``fresh`` and ``resumed`` the round's transcript question row is
written and committed before the finalizer runs, as the run's outbound
event does in production. ``channel`` writes it itself, in its settlement
transaction.

Runs on SQLite only: the PostgreSQL cells live in
``test_finalizer_marker_residue_postgresql.py``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from tests.web.services.coordinator_command_shared import owning_coordinator
from tests.web.services.interaction_marker_shared import (
    assert_interaction_marker_invariant,
    assert_marker_residue,
)
from tests.web.services.interaction_rollout_policy_shared import (
    install_interaction_rollout_policy,
)
from tests.web.services.test_task_clarification_draft import _draft
from tests.web.services.test_trace_fact_commit_retry import fail_commits
from xagent.core.agent.checkpoint import CHECKPOINT_TYPE
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.user import User
from xagent.web.services import chat_history_service
from xagent.web.services import interaction_rollout as ir
from xagent.web.services import managed_task_lease as managed
from xagent.web.services import ops_signals
from xagent.web.services import task_execution as execution
from xagent.web.services import task_execution_event_writer, task_interaction_service
from xagent.web.services.chat_history_service import (
    persist_assistant_message_no_commit,
)
from xagent.web.services.task_execution_controller import (
    TaskControlState,
    transition_task_control_state_sync,
)
from xagent.web.services.task_interaction_close import (
    close_legacy_resume_interaction_sync,
)
from xagent.web.services.task_interaction_service import InteractionPrincipalKind
from xagent.web.services.task_lease_service import (
    TASK_RUN_ID_TRACE_FIELD,
    TaskLease,
    acquire_task_lease,
)

SOURCE = "sdk"
QUESTION = "Which color do you want?"
SECOND_QUESTION = "Which size do you want?"
SECOND_EVENT_ID = "22222222-2222-4222-8222-222222222222"
LATE_EVENT_ID = "33333333-3333-4333-8333-333333333333"
FINALIZERS = ["fresh", "resumed", "channel"]
TASK_EXECUTION_FINALIZERS = ["fresh", "resumed"]
DECISION_PREFIX = "rollout.decision."


@pytest.fixture
def database(tmp_path: Path) -> Iterator[None]:
    init_db(db_url=f"sqlite:///{tmp_path / 'finalizer-publication.db'}")
    try:
        yield
    finally:
        # The file is discarded with tmp_path; the tasks <-> trace_events
        # pointer cycle makes drop_all refuse the tables.
        get_engine().dispose()


@pytest.fixture(autouse=True)
def _witness_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_BASE_DELAY_MS", "0")
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_MAX_DELAY_MS", "0")


def _native(monkeypatch: pytest.MonkeyPatch) -> None:
    install_interaction_rollout_policy(
        monkeypatch,
        mode="native",
        shared_task_execution=True,
        native_sources=frozenset({SOURCE}),
    )


@dataclass(frozen=True)
class Round:
    task_id: int
    user_id: int
    lease: TaskLease


def _seed_round(*, storage_version: int = 1) -> Round:
    """A running version-one task with a fresh lease and a checkpoint
    pointer the anchor resolver accepts for that lease's run."""
    with get_session_local()() as db:
        user = User(username="finalizer-publication", password_hash="x")
        db.add(user)
        db.flush()
        task = Task(
            user_id=user.id,
            title="Finalizer publication",
            description="Finalizer publication",
            status=TaskStatus.PENDING,
            source=SOURCE,
            conversation_storage_version=storage_version,
        )
        db.add(task)
        db.commit()
        task_id, user_id = int(task.id), int(user.id)
        lease = acquire_task_lease(db, task_id, new_run=True)
        assert lease is not None and lease.run_id is not None
        db.commit()
        checkpoint = TraceEvent(
            task_id=task_id,
            event_id=f"checkpoint-{task_id}",
            event_type="system_update_general",
            timestamp=datetime.now(timezone.utc),
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                TASK_RUN_ID_TRACE_FIELD: lease.run_id,
            },
        )
        db.add(checkpoint)
        db.flush()
        db.get(Task, task_id).last_checkpoint_trace_event_id = checkpoint.id
        db.commit()
    return Round(task_id=task_id, user_id=user_id, lease=lease)


def _waiting_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "waiting_for_user",
        "success": False,
        "output": QUESTION,
        "clarification_draft": _draft(message=QUESTION),
    }
    result.update(overrides)
    return result


def _commit_transcript_question(round_: Round) -> int:
    """The round's question row, as the run's outbound event commits it
    before a ``fresh`` or ``resumed`` finalizer runs."""
    with get_session_local()() as db:
        row = persist_assistant_message_no_commit(
            db,
            task_id=round_.task_id,
            user_id=round_.user_id,
            content=QUESTION,
            message_type=chat_history_service.QUESTION_MESSAGE_TYPE,
        )
        assert row is not None
        db.commit()
        return int(row.id)


def _finalize(
    finalizer: str,
    round_: Round,
    result: dict[str, Any],
    *,
    coordinator: bool = True,
) -> Any:
    if finalizer == "fresh":
        return execution._finalize_task_execution_result_isolated(
            task_id=round_.task_id,
            task_user_id=round_.user_id,
            pre_run_status=TaskStatus.RUNNING,
            result=result,
            expected_run_id=round_.lease.run_id,
            task_lease=round_.lease,
            resolved_scope_segments=(),
            prepared_outputs=execution._PreparedTaskFileOutputs((), (), ()),
        )
    if finalizer == "resumed":
        return execution._finalize_resumed_task(
            round_.task_id,
            status=str(result["status"]),
            success=bool(result.get("success")),
            output=str(result.get("output") or ""),
            task_owner_user_id=round_.user_id,
            result=result,
            task_lease=round_.lease,
            prepared_outputs=execution._PreparedTaskFileOutputs((), (), ()),
        )
    assert finalizer == "channel"
    scope = owning_coordinator(round_.lease) if coordinator else nullcontext()
    with scope:
        return managed.finalize_managed_task_lease_result_sync(
            round_.lease,
            status=TaskStatus.WAITING_FOR_USER,
            assistant_content=QUESTION,
            message_type=chat_history_service.QUESTION_MESSAGE_TYPE,
            execution_result=result,
        )


def _prepare(finalizer: str, round_: Round) -> None:
    if finalizer != "channel":
        _commit_transcript_question(round_)


def _second_question_result(
    round_: Round,
    *,
    message: str = SECOND_QUESTION,
    event_id: str = SECOND_EVENT_ID,
) -> dict[str, Any]:
    """What a run that asks again after its first question returns: a new
    checkpoint, the legacy question row the outbound event commits, and a
    waiting result whose draft carries a new ``event_id``."""
    with get_session_local()() as db:
        checkpoint = TraceEvent(
            task_id=round_.task_id,
            event_id=f"checkpoint-{round_.task_id}-{event_id}",
            event_type="system_update_general",
            timestamp=datetime.now(timezone.utc),
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                TASK_RUN_ID_TRACE_FIELD: round_.lease.run_id,
            },
        )
        db.add(checkpoint)
        db.flush()
        db.get(Task, round_.task_id).last_checkpoint_trace_event_id = checkpoint.id
        db.add(
            TaskChatMessage(
                task_id=round_.task_id,
                user_id=round_.user_id,
                role="assistant",
                content=message,
                message_type=chat_history_service.QUESTION_MESSAGE_TYPE,
            )
        )
        db.commit()
    draft = _draft(
        message=message,
        event_id=event_id,
        turn_message_count=5,
        origin_step_id="step-2",
    )
    return _waiting_result(output=message, clarification_draft=draft)


def _resume_same_run(round_: Round, *, close_id: int | None = None) -> Round:
    """Accept an answer on the round's run and resume it: the resume is
    requested, the legacy close runs with the id the message observed
    (``None`` when it observed no question), and a new lease attempt is
    taken the way the resumed execution takes it."""
    transition_task_control_state_sync(
        round_.task_id,
        TaskControlState.RESUME_REQUESTED,
        expected_run_id=round_.lease.run_id,
        fence_live_owner=True,
        owner_lease=round_.lease,
        expected_state_version=None,
        refuse_terminal_status=True,
    )
    close_legacy_resume_interaction_sync(
        task_id=round_.task_id,
        run_id=str(round_.lease.run_id),
        interaction_id=close_id,
    )
    lease = execution._acquire_resume_task_lease(
        round_.task_id,
        round_.user_id,
        round_.lease.run_id,
        refuse_terminal_status=True,
    )
    assert lease is not None
    return replace(round_, lease=lease)


@dataclass(frozen=True)
class Observed:
    rows: list[tuple[str, str]]
    marker: int | None
    message_types: list[str]
    status: TaskStatus
    control_state: str
    settled_facts: int


def _observe(task_id: int) -> Observed:
    with get_session_local()() as db:
        task = db.get(Task, task_id)
        rows = [
            (str(row.run_id), str(row.status))
            for row in db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.task_id == task_id)
            .order_by(TaskInteractionRequest.id)
        ]
        message_types = [
            str(row.message_type)
            for row in db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
            )
            .order_by(TaskChatMessage.id)
        ]
        settled = (
            db.query(TaskExecutionEvent)
            .filter(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.kind == "execution_settled",
            )
            .count()
        )
        return Observed(
            rows=rows,
            marker=task.interaction_protocol_version,
            message_types=message_types,
            status=task.status,
            control_state=str(task.control_state),
            settled_facts=settled,
        )


def _decision_delta(before: dict[str, int]) -> dict[str, int]:
    after = ir.counters_snapshot()
    return {
        name: after.get(name, 0) - before.get(name, 0)
        for name in set(after) | set(before)
        if name.startswith(DECISION_PREFIX)
        and after.get(name, 0) != before.get(name, 0)
    }


@contextmanager
def _statements() -> Iterator[list[str]]:
    seen: list[str] = []

    def record(_conn, _cursor, statement, _parameters, _context, _executemany):
        seen.append(statement)

    engine = get_engine()
    event.listen(engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        event.remove(engine, "before_cursor_execute", record)


# --------------------------------------------------------------------------
# The publication itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("finalizer", FINALIZERS)
def test_publication_writes_row_marker_and_relabel(
    database, monkeypatch: pytest.MonkeyPatch, finalizer: str
) -> None:
    _native(monkeypatch)
    round_ = _seed_round()
    _prepare(finalizer, round_)

    _finalize(finalizer, round_, _waiting_result())

    observed = _observe(round_.task_id)
    assert observed.rows == [(round_.lease.run_id, "active")]
    assert observed.marker == 1
    assert observed.message_types == [chat_history_service.SUPERSEDED_MESSAGE_TYPE]
    assert observed.status == TaskStatus.WAITING_FOR_USER
    with get_session_local()() as db:
        assert_marker_residue(db, task_id=round_.task_id, run_id=round_.lease.run_id)
        assert_interaction_marker_invariant(db, task_id=round_.task_id)


def test_publication_keeps_the_relabelled_row_otherwise_unchanged(
    database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relabel rewrites ``message_type`` only (``fresh``: the row was
    committed before the finalizer ran, so it can be compared)."""
    _native(monkeypatch)
    round_ = _seed_round()
    row_id = _commit_transcript_question(round_)
    with get_session_local()() as db:
        before = db.get(TaskChatMessage, row_id)
        snapshot = (before.content, before.interactions, before.created_at)

    _finalize("fresh", round_, _waiting_result())

    with get_session_local()() as db:
        after = db.get(TaskChatMessage, row_id)
        assert after.message_type == chat_history_service.SUPERSEDED_MESSAGE_TYPE
        assert (after.content, after.interactions, after.created_at) == snapshot


# --------------------------------------------------------------------------
# Rollout policy: the default mode and the single-process topology
# --------------------------------------------------------------------------


@pytest.mark.parametrize("finalizer", FINALIZERS)
def test_default_legacy_mode_publishes_nothing(
    database, monkeypatch: pytest.MonkeyPatch, finalizer: str
) -> None:
    install_interaction_rollout_policy(
        monkeypatch, mode="legacy", shared_task_execution=False
    )
    round_ = _seed_round()
    _prepare(finalizer, round_)
    before = ir.counters_snapshot()

    _finalize(finalizer, round_, _waiting_result())

    observed = _observe(round_.task_id)
    assert observed.rows == []
    assert observed.marker is None
    assert observed.message_types == [chat_history_service.QUESTION_MESSAGE_TYPE]
    assert _decision_delta(before) == {ir.COUNTER_ROLLOUT_DECISION_BLOCKED_MODE: 1}


def _publication_statements(statements: list[str]) -> list[str]:
    """Statements only the publication block issues: the anchor read, the
    interaction table's schema check and writes, and the relabel."""
    return [
        statement
        for statement in statements
        if "trace_events" in statement
        or "task_interaction_requests" in statement
        or "task_chat_messages SET message_type" in statement
    ]


@pytest.mark.parametrize("finalizer", FINALIZERS)
def test_legacy_mode_issues_no_publication_query(
    database, monkeypatch: pytest.MonkeyPatch, finalizer: str
) -> None:
    """In the default mode the gate answers before any query, so the
    settlement issues none of the statements the publication block would:
    no anchor read, no schema check, no interaction row, no relabel."""
    install_interaction_rollout_policy(
        monkeypatch, mode="legacy", shared_task_execution=False
    )
    round_ = _seed_round()
    _prepare(finalizer, round_)

    with _statements() as statements:
        _finalize(finalizer, round_, _waiting_result())

    assert _publication_statements(statements) == []


@pytest.mark.parametrize("finalizer", FINALIZERS)
def test_single_process_topology_never_publishes(
    database, monkeypatch: pytest.MonkeyPatch, finalizer: str
) -> None:
    """``native`` with the task's source allowed, but no shared task
    execution. ``fresh`` and ``resumed`` are stopped by the gate's topology
    step. ``channel`` runs without a coordinator, as it would in such a
    process (coordinators exist only under shared task execution), so it is
    stopped before the gate by its own entry condition, the coordinator that
    owns the lease; it counts no decision. In a single process the decision
    counters therefore count only the two task-execution finalizers."""
    install_interaction_rollout_policy(
        monkeypatch,
        mode="native",
        shared_task_execution=False,
        native_sources=frozenset({SOURCE}),
    )
    round_ = _seed_round()
    _prepare(finalizer, round_)
    before = ir.counters_snapshot()

    _finalize(finalizer, round_, _waiting_result(), coordinator=False)

    observed = _observe(round_.task_id)
    assert observed.rows == []
    assert observed.marker is None
    assert observed.message_types == [chat_history_service.QUESTION_MESSAGE_TYPE]
    expected = (
        {}
        if finalizer == "channel"
        else {ir.COUNTER_ROLLOUT_DECISION_BLOCKED_TOPOLOGY: 1}
    )
    assert _decision_delta(before) == expected


# --------------------------------------------------------------------------
# Entry conditions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("finalizer", FINALIZERS)
def test_version_two_task_skips_the_block(
    database, monkeypatch: pytest.MonkeyPatch, finalizer: str
) -> None:
    _native(monkeypatch)
    round_ = _seed_round(storage_version=2)
    _prepare(finalizer, round_)
    before = ir.counters_snapshot()

    with _statements() as statements:
        _finalize(finalizer, round_, _waiting_result())

    observed = _observe(round_.task_id)
    assert observed.rows == []
    assert observed.marker is None
    assert [s for s in statements if "task_chat_messages SET message_type" in s] == []
    with get_session_local()() as db:
        assert (
            db.query(TaskExecutionEvent)
            .filter(
                TaskExecutionEvent.task_id == round_.task_id,
                TaskExecutionEvent.kind == "interaction_requested",
            )
            .count()
            == 0
        )
    assert _decision_delta(before) == {}


@pytest.mark.parametrize("finalizer", TASK_EXECUTION_FINALIZERS)
def test_unknown_injection_outcome_pauses_without_publishing(
    database, monkeypatch: pytest.MonkeyPatch, finalizer: str
) -> None:
    _native(monkeypatch)
    round_ = _seed_round()
    _prepare(finalizer, round_)

    _finalize(finalizer, round_, _waiting_result(injection_outcome_unknown=True))

    observed = _observe(round_.task_id)
    assert observed.status == TaskStatus.PAUSED
    assert observed.rows == []
    assert observed.marker is None
    assert observed.message_types == [chat_history_service.QUESTION_MESSAGE_TYPE]


@pytest.mark.parametrize("finalizer", TASK_EXECUTION_FINALIZERS)
def test_externally_failed_task_publishes_nothing(
    database, monkeypatch: pytest.MonkeyPatch, finalizer: str
) -> None:
    _native(monkeypatch)
    round_ = _seed_round()
    _prepare(finalizer, round_)
    with get_session_local()() as db:
        db.get(Task, round_.task_id).status = TaskStatus.FAILED
        db.commit()
    before = ir.counters_snapshot()

    _finalize(finalizer, round_, _waiting_result())

    observed = _observe(round_.task_id)
    assert observed.rows == []
    assert observed.marker is None
    assert observed.status == TaskStatus.FAILED
    assert _decision_delta(before) == {}


@pytest.mark.parametrize(
    ("finalizer", "control_state", "published"),
    [
        ("fresh", "resume_requested", False),
        ("resumed", "resume_requested", False),
        ("fresh", "running", True),
        ("resumed", "running", True),
    ],
)
def test_resume_requested_wait_is_not_published(
    database,
    monkeypatch: pytest.MonkeyPatch,
    finalizer: str,
    control_state: str,
    published: bool,
) -> None:
    """Covers one ordering only: the resume was requested before the
    finalizer took the task row's lock. The ``running`` cells are the
    control: the same round publishes. The other ordering is covered by
    tests/web/api/test_message_answer_publication_race.py."""
    _native(monkeypatch)
    round_ = _seed_round()
    _prepare(finalizer, round_)
    with get_session_local()() as db:
        db.get(Task, round_.task_id).control_state = control_state
        db.commit()
    before = ir.counters_snapshot()

    _finalize(finalizer, round_, _waiting_result())

    observed = _observe(round_.task_id)
    if published:
        assert observed.rows == [(round_.lease.run_id, "active")]
        return
    assert observed.rows == []
    assert observed.marker is None
    assert observed.message_types == [chat_history_service.QUESTION_MESSAGE_TYPE]
    assert _decision_delta(before) == {}
    if finalizer == "fresh":
        assert observed.control_state == "resume_requested"
        assert observed.status == TaskStatus.WAITING_FOR_USER


def _rows(task_id: int) -> list[tuple[int, str, str | None, str]]:
    with get_session_local()() as db:
        return [
            (
                int(row.id),
                str(row.status),
                row.terminal_reason,
                str(row.request_idempotency_key),
            )
            for row in db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.task_id == task_id)
            .order_by(TaskInteractionRequest.id)
        ]


def test_a_replayed_draft_keeps_its_question(
    database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume that waits again on the same draft republishes the same
    question: the publication replays its row instead of retiring it."""
    _native(monkeypatch)
    round_ = _seed_round()
    _prepare("fresh", round_)
    _finalize("fresh", round_, _waiting_result())
    first = _rows(round_.task_id)
    assert [(status, reason) for _, status, reason, _ in first] == [("active", None)]

    resumed = _resume_same_run(round_)
    _prepare("resumed", resumed)
    ops_signals.clear_degradation(ops_signals.INTERACTION_HANDOFF_DEGRADED)
    _finalize("resumed", resumed, _waiting_result())

    assert _rows(round_.task_id) == first
    assert (
        ops_signals.INTERACTION_HANDOFF_DEGRADED
        not in ops_signals.active_degradations()
    )


@pytest.mark.parametrize("late", FINALIZERS)
def test_a_superseded_attempt_cannot_retire_the_current_question(
    database, monkeypatch: pytest.MonkeyPatch, late: str
) -> None:
    """A finalizer from an older attempt of the task's run is turned away
    before it publishes, so it cannot retire the question the current
    attempt published."""
    _native(monkeypatch)
    first = _seed_round()
    _prepare("fresh", first)
    _finalize("fresh", first, _waiting_result())
    second = _resume_same_run(first)
    assert second.lease.attempt_id != first.lease.attempt_id
    _prepare("resumed", second)
    _finalize("resumed", second, _second_question_result(second))
    before = _rows(first.task_id)
    assert [(status, reason) for _, status, reason, _ in before] == [
        ("terminated", "run_superseded"),
        ("active", None),
    ]

    _resume_same_run(second)
    _finalize(late, first, _second_question_result(first, event_id=LATE_EVENT_ID))

    assert _rows(first.task_id) == before


def test_a_close_that_lost_to_the_next_question_leaves_it_active(
    database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The close that observed the first question is bound to its id. When
    the next question has already retired that row, the close matches
    nothing and the new question stays active with its marker."""
    _native(monkeypatch)
    first = _seed_round()
    _prepare("fresh", first)
    _finalize("fresh", first, _waiting_result())
    ((first_id, _, _, _),) = _rows(first.task_id)

    resumed = _resume_same_run(first, close_id=None)
    _prepare("resumed", resumed)
    _finalize("resumed", resumed, _second_question_result(resumed))
    rowcount = close_legacy_resume_interaction_sync(
        task_id=first.task_id, run_id=str(first.lease.run_id), interaction_id=first_id
    )

    assert rowcount == 0
    rows = _rows(first.task_id)
    assert [(status, reason) for _, status, reason, _ in rows] == [
        ("terminated", "run_superseded"),
        ("active", None),
    ]
    assert rows[0][0] == first_id
    observed = _observe(first.task_id)
    assert observed.marker == 1
    with get_session_local()() as db:
        assert_marker_residue(db, task_id=first.task_id, run_id=first.lease.run_id)
        assert_interaction_marker_invariant(db, task_id=first.task_id)


# --------------------------------------------------------------------------
# Lock ordering and the principal create() runs as
# --------------------------------------------------------------------------


@pytest.mark.parametrize("finalizer", FINALIZERS)
def test_publication_gate_runs_after_the_row_lock(
    database, monkeypatch: pytest.MonkeyPatch, finalizer: str
) -> None:
    """The first statement that locks the task row (``fresh``/``resumed``:
    the settlement lock helper's ``UPDATE tasks`` on SQLite; ``channel``:
    the lease release's ``UPDATE tasks``) precedes the gate's schema check."""
    _native(monkeypatch)
    round_ = _seed_round()
    _prepare(finalizer, round_)

    with _statements() as statements:
        _finalize(finalizer, round_, _waiting_result())

    lock = next(
        index
        for index, statement in enumerate(statements)
        if statement.lstrip().startswith("UPDATE tasks")
    )
    schema_check = statements.index(
        'PRAGMA main.table_info("task_interaction_requests")'
    )
    assert lock < schema_check


@pytest.mark.parametrize("finalizer", FINALIZERS)
def test_create_runs_under_the_row_lock_as_system(
    database, monkeypatch: pytest.MonkeyPatch, finalizer: str, tmp_path: Path
) -> None:
    _native(monkeypatch)
    round_ = _seed_round()
    _prepare(finalizer, round_)
    seen: dict[str, Any] = {}
    original = task_interaction_service.create

    def create(db, **kwargs):
        seen["kind"] = kwargs["principal"].kind
        other = sqlite3.connect(
            tmp_path / "finalizer-publication.db", timeout=0, isolation_level=None
        )
        try:
            other.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            seen["error"] = str(exc)
        else:
            other.execute("ROLLBACK")
            seen["error"] = None
        finally:
            other.close()
        return original(db, **kwargs)

    monkeypatch.setattr(task_interaction_service, "create", create)

    _finalize(finalizer, round_, _waiting_result())

    assert seen == {
        "kind": InteractionPrincipalKind.SYSTEM,
        "error": "database is locked",
    }


# --------------------------------------------------------------------------
# A failing publication: the exception propagates and nothing is left
# --------------------------------------------------------------------------


class _InjectedFailure(Exception):
    pass


def _commit_earlier_question(round_: Round) -> None:
    """A question row from an earlier round, already committed, which a
    publication would relabel along with this round's."""
    _commit_transcript_question(round_)


def _lease_columns(task_id: int) -> tuple[Any, ...]:
    with get_session_local()() as db:
        task = db.get(Task, task_id)
        return (
            task.runner_id,
            task.run_id,
            task.lease_attempt_id,
            task.lease_expires_at,
        )


@pytest.mark.parametrize("finalizer", FINALIZERS)
def test_publication_failure_propagates_and_leaves_nothing(
    database, monkeypatch: pytest.MonkeyPatch, finalizer: str
) -> None:
    """The relabel raises after ``create()`` and the marker UPDATE have run.
    The finalizer does not catch it: it propagates unchanged and the
    settlement transaction is rolled back. How the caller then settles the
    run is not this test's subject (the outer handlers own it and have their
    own tests). ``channel`` passes through the commit witness driver, which
    re-raises an error that is not a retryable database error unchanged.

    ``fresh``/``resumed``: the round's question row was committed before the
    finalizer ran and stays a question; the task stays RUNNING with its
    lease. ``channel``: an earlier round's committed question row stays a
    question, and this round's row, written in the rolled-back transaction,
    does not exist."""
    _native(monkeypatch)
    round_ = _seed_round()
    if finalizer == "channel":
        _commit_earlier_question(round_)
    else:
        _prepare(finalizer, round_)
    lease_before = _lease_columns(round_.task_id)
    failure = _InjectedFailure()

    def supersede(_db, *, task_id):
        raise failure

    monkeypatch.setattr(
        chat_history_service, "supersede_legacy_question_rows", supersede
    )

    with pytest.raises(_InjectedFailure) as raised:
        _finalize(finalizer, round_, _waiting_result())

    assert raised.value is failure
    observed = _observe(round_.task_id)
    assert observed.rows == []
    assert observed.marker is None
    assert observed.message_types == [chat_history_service.QUESTION_MESSAGE_TYPE]
    assert observed.status == TaskStatus.RUNNING
    if finalizer != "channel":
        assert _lease_columns(round_.task_id) == lease_before


@pytest.mark.parametrize("failure", ["insert_conflict", "later_statement"])
def test_a_failed_publication_keeps_the_question_it_would_have_retired(
    database,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    """The retire and the new row share one transaction. When the new row
    cannot be published, the first question stays active with its marker,
    whether the handoff degrades or the whole settlement rolls back."""
    _native(monkeypatch)
    first = _seed_round()
    _prepare("fresh", first)
    _finalize("fresh", first, _waiting_result())
    before = _rows(first.task_id)
    resumed = _resume_same_run(first, close_id=None)
    _prepare("resumed", resumed)
    result = _second_question_result(resumed)

    if failure == "insert_conflict":

        def collide(session, _context, _instances):
            if any(isinstance(row, TaskInteractionRequest) for row in session.new):
                raise IntegrityError("simulated conflict", None, None)

        event.listen(Session, "before_flush", collide)
        try:
            with caplog.at_level(logging.INFO):
                _finalize("resumed", resumed, result)
        finally:
            event.remove(Session, "before_flush", collide)
    else:

        def supersede(_db, *, task_id):
            raise _InjectedFailure()

        monkeypatch.setattr(
            chat_history_service, "supersede_legacy_question_rows", supersede
        )
        with caplog.at_level(logging.INFO), pytest.raises(_InjectedFailure):
            _finalize("resumed", resumed, result)

    assert any(
        record.getMessage().startswith("Retiring interaction")
        for record in caplog.records
    )
    assert _rows(first.task_id) == before
    assert _observe(first.task_id).marker == 1


def test_channel_commit_failure_without_a_witness_propagates_and_leaves_nothing(
    database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version-one task stages no result fact, so the channel finalizer's
    commit witness driver has no witness to read back: a failed COMMIT is
    re-raised unchanged after one attempt, never read back or rerun."""
    _native(monkeypatch)
    round_ = _seed_round()
    _commit_earlier_question(round_)
    attempts: list[int] = []
    original_finalize = managed.finalize_managed_task_lease_result

    def finalize(db, lease, **kwargs):
        attempts.append(kwargs["attempt"].number)
        return original_finalize(db, lease, **kwargs)

    monkeypatch.setattr(managed, "finalize_managed_task_lease_result", finalize)
    read_backs: list[Any] = []
    original_read_back = task_execution_event_writer.fact_witness_committed

    def read_back(witness):
        read_backs.append(witness)
        return original_read_back(witness)

    monkeypatch.setattr(
        task_execution_event_writer, "fact_witness_committed", read_back
    )
    failures = fail_commits(monkeypatch, lambda _db: True, committed=False)

    with pytest.raises(OperationalError) as raised:
        _finalize("channel", round_, _waiting_result())

    assert raised.value is failures[0]
    assert read_backs == []
    assert attempts == [1]
    observed = _observe(round_.task_id)
    assert observed.rows == []
    assert observed.marker is None
    assert observed.message_types == [chat_history_service.QUESTION_MESSAGE_TYPE]
    assert observed.settled_facts == 0
