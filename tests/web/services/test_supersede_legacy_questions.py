"""Behavior tests for ``supersede_legacy_question_rows``.

Covers everything about the helper that is provable on SQLite: whole-set
rewrite semantics with negative controls (role, message_type, a
different task, and a nonexistent task_id), idempotency, the
degrade/propagate split on the catch clause, signal pairing,
transcript-field invariance, a caller-originated flush failure
propagating past the catch instead of being absorbed by it, and the
reader's and writer's compiled WHERE clauses matching at runtime.

One fact this file holds both halves of, and the reason the pairing
lives here: the relabel scans the whole task and carries no run
partition. Its WHERE is task, assistant role, message type, so a single
call relabels every round's question row on the task at once. That is
what makes the reader's superseded-row pass correct on a task that has
asked more than once -- ordering by id descending over rows the relabel
just swept up lands on the newest round, not on some earlier one.

One property is not provable here -- that
the savepoint is actually necessary, i.e. that a failed statement
without it would poison the caller's transaction. SQLite does not abort
an open transaction after a failed statement the way PostgreSQL does, so
that specific mutation cannot be made to fail on this backend; it is
covered instead in ``test_supersede_savepoint_postgresql.py`` (see that
file's module docstring).
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import Select, Update, create_engine, event
from sqlalchemy.exc import IntegrityError, InvalidRequestError, OperationalError
from sqlalchemy.orm import Query, Session, sessionmaker
from sqlalchemy.orm.session import SessionTransaction

from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import ops_signals
from xagent.web.services.chat_history_service import (
    QUESTION_MESSAGE_TYPE,
    SUPERSEDED_MESSAGE_TYPE,
    get_latest_waiting_question,
    persist_assistant_message,
    supersede_legacy_question_rows,
)


def _create_db_session():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal()


def _create_task(db_session, username="tester"):
    user = User(username=username, password_hash="hashed_password", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    task = Task(
        user_id=int(user.id),
        title="Chat task",
        description="Task chat",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


@pytest.fixture(autouse=True)
def _clean_signal():
    ops_signals.clear_degradation(ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED)
    yield
    ops_signals.clear_degradation(ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED)


def test_message_type_constants_have_their_documented_values():
    # The admin transcript export (conversation_logs.py) passes
    # message_type straight through, making "question_superseded" a de
    # facto external contract -- this is what reds if anyone edits
    # either value.
    assert QUESTION_MESSAGE_TYPE == "question"
    assert SUPERSEDED_MESSAGE_TYPE == "question_superseded"


def test_supersede_zeroes_the_whole_waiting_set_with_negative_controls():
    """Two simultaneously waiting questions on the same task both flip;
    a same-task row with ``role='user'``, a same-task assistant row with
    a different ``message_type``, and a same-shaped row on a different
    task are all untouched -- the negative controls prove all three
    predicate legs (``role``, ``message_type``, ``task_id``) actually
    gate the update, not just one of them."""
    db = _create_db_session()
    try:
        task = _create_task(db)
        other_task = _create_task(db, username="other-tester")

        first = persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "First question",
            message_type="question",
        )
        second = persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "Second question",
            message_type="question",
        )
        # Adversarial negative control: role='user' but message_type
        # matches -- if the helper's predicate ever dropped the role leg,
        # this row would wrongly flip too.
        user_row = TaskChatMessage(
            task_id=int(task.id),
            user_id=int(task.user_id),
            role="user",
            content="not actually a question",
            message_type="question",
        )
        db.add(user_row)
        db.commit()
        db.refresh(user_row)

        # Adversarial negative control: role='assistant' and same task,
        # but message_type is not "question" -- if the helper's predicate
        # ever dropped the message_type leg, this row would wrongly flip
        # too.
        non_question_row = persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "Just a status update, not a question",
        )

        other_question = persist_assistant_message(
            db,
            int(other_task.id),
            int(other_task.user_id),
            "Other task question",
            message_type="question",
        )

        updated = supersede_legacy_question_rows(db, task_id=int(task.id))

        assert updated == 2
        db.refresh(first)
        db.refresh(second)
        db.refresh(user_row)
        db.refresh(non_question_row)
        db.refresh(other_question)
        assert first.message_type == SUPERSEDED_MESSAGE_TYPE
        assert second.message_type == SUPERSEDED_MESSAGE_TYPE
        assert user_row.message_type == "question"
        assert non_question_row.message_type == "assistant_message"
        assert other_question.message_type == "question"
    finally:
        db.close()


def test_supersede_is_a_benign_no_op_for_a_task_with_no_rows():
    """A nonexistent ``task_id`` is a benign 0-row no-op, not an error --
    the same outcome as an empty projection where no assistant question
    row was ever persisted for a real task. Pins the ``task_id`` leg's
    behavior at its boundary, distinct from
    ``test_supersede_is_idempotent_on_a_second_call``, which pins
    behavior on a task that *did* have a row superseded already."""
    db = _create_db_session()
    try:
        updated = supersede_legacy_question_rows(db, task_id=999999)

        assert updated == 0
        signal_name = ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED
        assert signal_name not in ops_signals.active_degradations()
    finally:
        db.close()


def test_supersede_is_idempotent_on_a_second_call():
    db = _create_db_session()
    try:
        task = _create_task(db)
        row = persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
        )

        first_pass = supersede_legacy_question_rows(db, task_id=int(task.id))
        second_pass = supersede_legacy_question_rows(db, task_id=int(task.id))

        assert first_pass == 1
        assert second_pass == 0
        db.refresh(row)
        assert row.message_type == SUPERSEDED_MESSAGE_TYPE
    finally:
        db.close()


def test_supersede_degrades_on_dbapi_error_and_lets_the_caller_commit(monkeypatch):
    """A DBAPIError raised inside the update is swallowed: the helper
    returns 0, registers the degradation signal, and the legacy row is
    left exactly as it was. This proves the exception classification
    (DBAPIError degrades) and the signal path -- it does not by itself
    prove the savepoint is necessary, since SQLite does not poison the
    enclosing transaction on a failed statement the way PostgreSQL does.
    The savepoint's necessity is proven by the PostgreSQL-only test in
    ``test_supersede_savepoint_postgresql.py``."""
    db = _create_db_session()
    try:
        task = _create_task(db)
        row = persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
        )

        boom = OperationalError(
            "UPDATE task_chat_messages", {}, Exception("connection reset")
        )

        def failing_update(self, *args, **kwargs):
            raise boom

        monkeypatch.setattr(Query, "update", failing_update)
        result = supersede_legacy_question_rows(db, task_id=int(task.id))

        assert result == 0
        signal_name = ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED
        assert signal_name in ops_signals.active_degradations()
        detail = ops_signals.active_degradations()[signal_name]
        # Exact match, not a substring check: the detail is a composed
        # "task <id>: legacy question supersede failed (<class name>)"
        # string with nothing else in it -- no exception free-text, which
        # could leak DB connection info into a process-wide,
        # unauthenticated-adjacent registry (see ops_signals.py's own
        # docstring on that boundary).
        assert (
            detail
            == f"task {task.id}: legacy question supersede failed (OperationalError)"
        )

        # This confirms the caller's transaction is still usable after the
        # swallow -- it does not by itself prove the savepoint caused
        # that; on SQLite the commit succeeds even without one (see the
        # module docstring and the PostgreSQL-only test, which is what
        # actually proves the savepoint's necessity).
        db.commit()

        db.refresh(row)
        assert row.message_type == "question"
    finally:
        db.close()


def test_supersede_distinguishes_a_savepoint_exit_failure_from_a_statement_failure(
    monkeypatch,
):
    """A ``DBAPIError`` from the UPDATE itself and one from the SAVEPOINT's
    own RELEASE on the way out of the ``with`` block are different
    failures and now get different treatment. The statement failure is
    contained by the savepoint, so it degrades: return 0, signal
    registered. The exit failure is not contained -- on PostgreSQL it
    aborts the enclosing transaction -- so it propagates, and no signal is
    registered, because there is nothing here for an operator to act on
    that the exception does not already say. This test pins the second
    case; the statement-failure case is pinned by the test above.

    Patches ``SessionTransaction.__exit__`` (the class the ``with
    db.begin_nested():`` block's context manager protocol actually
    dispatches to -- an instance-level patch would not be seen by the
    ``with`` statement) so it raises only on a clean exit, i.e. only
    after the UPDATE inside the block has already run successfully."""
    db = _create_db_session()
    try:
        task = _create_task(db)
        persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
        )

        real_exit = SessionTransaction.__exit__

        def failing_release_exit(self, exc_type, exc_val, exc_tb):
            # Let the real RELEASE happen and finish cleanly first --
            # this test's own DB session must stay usable afterward --
            # then synthesize the exit-time DBAPIError on top of that
            # otherwise-successful exit, so the helper sees exactly the
            # "statement succeeded, then something failed on the way
            # out" shape without leaving the session wedged.
            result = real_exit(self, exc_type, exc_val, exc_tb)
            if exc_type is None:
                raise OperationalError(
                    "RELEASE SAVEPOINT", {}, Exception("release failed")
                )
            return result

        monkeypatch.setattr(SessionTransaction, "__exit__", failing_release_exit)

        signal_name = ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED

        with pytest.raises(OperationalError):
            supersede_legacy_question_rows(db, task_id=int(task.id))

        # No degradation signal: the exception is the notification, and a
        # signal here would tell an operator to look at a helper whose own
        # statement succeeded.
        assert signal_name not in ops_signals.active_degradations()
    finally:
        db.close()


def test_supersede_propagates_a_savepoint_that_could_not_be_created(monkeypatch):
    """The savepoint fails to be created, so nothing is holding the
    relabel and the caller's transaction is not sectioned off. This
    propagates and registers no signal, like the exit failure and unlike a
    failure of the UPDATE inside an open savepoint.

    The failure is injected at ``dialect.do_savepoint``, which is where the
    SAVEPOINT is actually issued. Patching ``Session.begin_nested`` instead
    would not exercise this path at all: that call emits no SQL."""
    db = _create_db_session()
    try:
        task = _create_task(db)
        persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
        )
        signal_name = ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED

        def refusing_do_savepoint(connection, name):
            raise OperationalError("SAVEPOINT", {}, Exception("savepoint refused"))

        monkeypatch.setattr(
            db.get_bind().dialect, "do_savepoint", refusing_do_savepoint
        )

        with pytest.raises(OperationalError):
            supersede_legacy_question_rows(db, task_id=int(task.id))

        assert signal_name not in ops_signals.active_degradations()
    finally:
        db.close()


def test_supersede_propagates_a_non_dbapi_sqlalchemy_error(monkeypatch):
    """``InvalidRequestError`` is a ``SQLAlchemyError`` but not a
    ``DBAPIError`` -- it must come straight out of the helper, and the
    catch clause must not register the degradation signal for it."""
    db = _create_db_session()
    try:
        task = _create_task(db)
        persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
        )

        def failing_update(self, *args, **kwargs):
            raise InvalidRequestError("bad bulk update usage")

        monkeypatch.setattr(Query, "update", failing_update)
        with pytest.raises(InvalidRequestError):
            supersede_legacy_question_rows(db, task_id=int(task.id))

        signal_name = ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED
        assert signal_name not in ops_signals.active_degradations()

        # The savepoint still rolled back cleanly on the way out; the
        # session is not wedged.
        db.commit()
    finally:
        db.close()


def test_supersede_clears_the_signal_on_the_next_successful_call(monkeypatch):
    """One failure registers the signal; a subsequent successful call on
    the same or another task clears it -- registration and clearing are
    both keyed on the signal name, not on the task."""
    db = _create_db_session()
    try:
        task = _create_task(db)
        persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
        )

        boom = OperationalError(
            "UPDATE task_chat_messages", {}, Exception("connection reset")
        )

        def failing_update(self, *args, **kwargs):
            raise boom

        monkeypatch.setattr(Query, "update", failing_update)
        first_result = supersede_legacy_question_rows(db, task_id=int(task.id))
        monkeypatch.undo()

        signal_name = ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED
        assert first_result == 0
        assert signal_name in ops_signals.active_degradations()

        second_result = supersede_legacy_question_rows(db, task_id=int(task.id))

        assert second_result == 1
        assert signal_name not in ops_signals.active_degradations()
    finally:
        db.close()


def test_supersede_never_commits_and_the_caller_can_still_roll_back(monkeypatch):
    """The helper never calls ``db.commit()`` itself -- it only stages the
    UPDATE inside the caller's own open transaction, inside its own
    SAVEPOINT, and leaves committing to the caller.

    Proven three ways: a commit spy on the session records zero calls
    during a successful supersede; a second write the caller stages
    *after* calling the helper can still be rolled back cleanly
    afterward, proving the helper left the session as a normal, live,
    abortable transaction rather than doing anything (an early commit, a
    wedged connection) that would have taken that ability away from the
    caller; and the superseded row itself is re-read after the caller's
    rollback and must be back to ``"question"``.

    That last assertion is the one that exercises the savepoint rather
    than the session bookkeeping around it. A released SAVEPOINT only
    discards the ability to roll back *to* that point -- it does not
    commit anything -- so the caller's rollback has to undo the relabel
    too. Reaching the case where that can fail needs the helper to run
    with nothing pending in the session, which is what happens here:
    ``persist_assistant_message`` above commits, so the helper starts from
    a clean session. The sentinel row cannot cover this, because staging
    it is itself DML and forces a real transaction either way.
    ``test_supersede_savepoint_postgresql.py`` proves a different property
    again (a CHECK-constraint failure inside the savepoint not poisoning
    the outer transaction), not rollback survival. What the spy and the
    sentinel add on top is narrower:
    the helper never commits or rolls back on its own, shown by the
    commit spy recording zero calls and by the fresh sentinel row above
    still being cleanly removable by the caller's own explicit rollback
    -- proof the session was left as a normal, abortable transaction.
    """
    db = _create_db_session()
    try:
        task = _create_task(db)
        question = persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
        )
        question_id = int(question.id)

        commit_calls = 0
        original_commit = db.commit
        rollback_calls = 0
        original_rollback = db.rollback

        def spy_commit():
            nonlocal commit_calls
            commit_calls += 1
            return original_commit()

        def spy_rollback():
            nonlocal rollback_calls
            rollback_calls += 1
            return original_rollback()

        monkeypatch.setattr(db, "commit", spy_commit)
        monkeypatch.setattr(db, "rollback", spy_rollback)

        updated = supersede_legacy_question_rows(db, task_id=int(task.id))

        assert updated == 1
        assert commit_calls == 0
        assert rollback_calls == 0
        monkeypatch.setattr(db, "rollback", original_rollback)

        sentinel = TaskChatMessage(
            task_id=int(task.id),
            user_id=int(task.user_id),
            role="assistant",
            content="staged after supersede",
            message_type="assistant_message",
        )
        db.add(sentinel)
        db.flush()
        sentinel_id = sentinel.id

        db.rollback()

        assert (
            db.query(TaskChatMessage).filter(TaskChatMessage.id == sentinel_id).first()
            is None
        )

        # The superseded row itself, not a sentinel staged afterwards. This is
        # the assertion that actually exercises the savepoint: the relabel must
        # be gone after the caller's rollback. It reaches the case the sentinel
        # cannot, because persist_assistant_message above commits, leaving the
        # session with no pending DML when the helper runs.
        assert (
            db.query(TaskChatMessage)
            .filter(TaskChatMessage.id == question_id)
            .one()
            .message_type
            == QUESTION_MESSAGE_TYPE
        )
    finally:
        db.close()


def test_supersede_touches_only_the_message_type_column():
    """Every other transcript field on the row is unchanged: no delete, no
    content rewrite, no interactions/attachments/turn_id/user_id/id/
    created_at drift."""
    db = _create_db_session()
    try:
        task = _create_task(db)
        row = persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
            interactions=[{"type": "text_input", "label": "Q"}],
        )
        before = {
            "id": row.id,
            "content": row.content,
            "interactions": row.interactions,
            "attachments": row.attachments,
            "turn_id": row.turn_id,
            "user_id": row.user_id,
            "created_at": row.created_at,
        }

        supersede_legacy_question_rows(db, task_id=int(task.id))

        db.refresh(row)
        assert row.message_type == SUPERSEDED_MESSAGE_TYPE
        assert row.id == before["id"]
        assert row.content == before["content"]
        assert row.interactions == before["interactions"]
        assert row.attachments == before["attachments"]
        assert row.turn_id == before["turn_id"]
        assert row.user_id == before["user_id"]
        assert row.created_at == before["created_at"]
    finally:
        db.close()


def test_supersede_opens_a_savepoint(monkeypatch):
    """The helper actually issues ``db.begin_nested()`` -- not just "some
    update inside a try/except" that happens to work on SQLite without a
    savepoint. Distinct from the PostgreSQL-only test in
    ``test_supersede_savepoint_postgresql.py``, which proves the
    savepoint is *necessary* (a failed statement without one poisons the
    caller's transaction); this test proves it is *present* at all, and
    that is provable on SQLite because it only needs to observe the call,
    not depend on SQLite reproducing PostgreSQL's abort-the-transaction
    behavior."""
    db = _create_db_session()
    try:
        task = _create_task(db)
        persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
        )

        calls = {"n": 0}
        real_begin_nested = Session.begin_nested

        def spy_begin_nested(self, *args, **kwargs):
            calls["n"] += 1
            return real_begin_nested(self, *args, **kwargs)

        monkeypatch.setattr(Session, "begin_nested", spy_begin_nested)
        updated = supersede_legacy_question_rows(db, task_id=int(task.id))

        assert updated == 1
        assert calls["n"] == 1
    finally:
        db.close()


def test_supersede_lets_a_callers_pending_write_failure_reach_the_caller():
    """``Session.begin_nested()`` flushes the session on entry, so a
    constraint violation among rows the *caller* staged and has not
    flushed would fire inside this helper. It must not be absorbed by the
    helper's own ``except DBAPIError``: the failure is the caller's, the
    UPDATE never ran, and reporting it as a supersede failure would send
    whoever reads the ops signal to the wrong function while the caller
    is left with an opaque ``PendingRollbackError`` at commit.

    The pending row here violates a real ``NOT NULL`` constraint at the
    database layer rather than through a patched method, so the failure
    is the one production would raise.
    """
    db = _create_db_session()
    try:
        task = _create_task(db)
        question = persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
        )
        question_id = int(question.id)

        # The caller's own staged, unflushed, constraint-violating row.
        db.add(
            TaskChatMessage(
                task_id=int(task.id),
                user_id=int(task.user_id),
                role="assistant",
                content=None,
                message_type="assistant_message",
            )
        )

        with pytest.raises(IntegrityError):
            supersede_legacy_question_rows(db, task_id=int(task.id))

        signal_name = ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED
        assert signal_name not in ops_signals.active_degradations()

        db.rollback()
        row = db.query(TaskChatMessage).filter(TaskChatMessage.id == question_id).one()
        assert row.message_type == QUESTION_MESSAGE_TYPE
    finally:
        db.close()


def test_supersede_end_to_end_with_the_reader():
    """Persist a waiting question, confirm the reader sees it, supersede
    it, then confirm the reader sees nothing -- proving the two functions
    agree end-to-end at runtime, on top of sharing
    ``_assistant_question_filters`` at the source level."""
    db = _create_db_session()
    try:
        task = _create_task(db)
        task.status = TaskStatus.WAITING_FOR_USER
        db.commit()
        persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
        )

        assert get_latest_waiting_question(db, int(task.id))[0] == "A question"
        supersede_legacy_question_rows(db, task_id=int(task.id))
        assert get_latest_waiting_question(db, int(task.id)) == (None, None)
    finally:
        db.close()


def test_the_reader_only_reaches_superseded_rows_when_the_caller_opens_it():
    """``allow_superseded`` is off unless a caller passes it. The default
    call -- the one every caller but the read surface adapter makes --
    sees a superseded row exactly as it always did: not at all. Mutation:
    default the parameter to True and the first assertion turns red."""

    db = _create_db_session()
    try:
        task = _create_task(db)
        persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A question",
            message_type="question",
            interactions=[{"type": "text_input", "label": "Env"}],
        )
        supersede_legacy_question_rows(db, task_id=int(task.id))

        assert get_latest_waiting_question(db, int(task.id)) == (None, None)

        question, interactions = get_latest_waiting_question(
            db, int(task.id), allow_superseded=True
        )
        assert question is not None
        assert question.startswith("A question")
        assert interactions == [{"type": "text_input", "label": "Env"}]
    finally:
        db.close()


def test_an_opened_reader_still_prefers_a_live_question_row():
    """Two sequential passes, not one ``message_type.in_(...)`` predicate:
    the live question row wins even when a superseded row carries a higher
    id, because the second pass only runs when the first came back empty.
    Mutation: match both message types in one predicate ordered by id and
    this turns red -- the superseded row here has the higher id."""

    db = _create_db_session()
    try:
        task = _create_task(db)
        persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "A live question",
            message_type="question",
        )
        persist_assistant_message(
            db,
            int(task.id),
            int(task.user_id),
            "An older question already superseded",
            message_type=SUPERSEDED_MESSAGE_TYPE,
        )

        question, _ = get_latest_waiting_question(
            db, int(task.id), allow_superseded=True
        )

        assert question is not None
        assert question.startswith("A live question")
    finally:
        db.close()


def test_an_opened_reader_returns_the_newest_round_after_a_whole_task_relabel():
    """``supersede_legacy_question_rows`` matches on task, role and message
    type with no run partition, so one call relabels every round's question
    row on the task at once (see its own docstring). The opened second pass
    orders by id descending, so what it hands back on a task that has asked
    three times is the third round's question, not the first.

    Mutation: order the second pass ascending and this turns red."""

    db = _create_db_session()
    try:
        task = _create_task(db)
        for round_number in (1, 2, 3):
            persist_assistant_message(
                db,
                int(task.id),
                int(task.user_id),
                f"Round {round_number} question",
                message_type="question",
            )

        relabelled = supersede_legacy_question_rows(db, task_id=int(task.id))
        assert relabelled == 3

        question, _ = get_latest_waiting_question(
            db, int(task.id), allow_superseded=True
        )

        assert question is not None
        assert question.startswith("Round 3 question")
    finally:
        db.close()


@contextmanager
def _captured_statements(bind):
    """Every SQL construct SQLAlchemy sends through ``bind`` while the
    block runs, as the ``ClauseElement`` objects themselves rather than
    rendered strings -- the WHERE clause has to be compared as a clause,
    not string-carved out of a full statement."""
    seen: list[object] = []

    def _record(conn, clauseelement, multiparams, params, execution_options):
        seen.append(clauseelement)

    event.listen(bind, "before_execute", _record)
    try:
        yield seen
    finally:
        event.remove(bind, "before_execute", _record)


def _rendered_where(bind, statement) -> str:
    return " ".join(
        str(
            statement.whereclause.compile(
                dialect=bind.dialect, compile_kwargs={"literal_binds": True}
            )
        ).split()
    )


def _reader_and_writer_where_clauses(db, task_id: int) -> tuple[str, str]:
    """Run both production functions for real and return the WHERE clause
    each one actually sent to the database.

    Neither side is reconstructed here: the SELECT is whatever
    ``get_latest_waiting_question`` emits and the UPDATE is whatever
    ``supersede_legacy_question_rows`` emits. So the comparison keeps
    measuring the real statements even if one function stops calling
    ``_assistant_question_filters`` -- what it detects is the two clauses
    differing, not the helper going unused. ``literal_binds`` renders the
    bound ``task_id`` inline, so the comparison covers the bound value
    too, not only the column/operator shape.

    The comparison is on rendered WHERE text, which makes it strict on
    purpose: conditions in a different order, or an equivalent rewrite
    such as ``message_type.in_(["question"])``, count as a difference.
    Two predicates that mean the same thing but do not render the same
    thing are exactly the state this abstraction exists to prevent.

    The reader is called with ``allow_superseded`` left at its default, so
    what this captures is its first pass and only its first pass. The
    second pass belongs on the other side of this pairing: its predicate
    matches ``SUPERSEDED_MESSAGE_TYPE``, which is what the writer
    *produces*. Folding it into the captured set would demand that the
    writer's input set equal its own output set, an assertion that can
    never hold. ``assert len(selects) == 1`` below therefore holds because
    the caller here left the second pass shut, not because the reader can
    only ever emit one statement.
    """
    bind = db.get_bind()
    with _captured_statements(bind) as read_seen:
        get_latest_waiting_question(db, task_id)
    with _captured_statements(bind) as write_seen:
        supersede_legacy_question_rows(db, task_id=task_id)

    selects = [s for s in read_seen if isinstance(s, Select)]
    updates = [s for s in write_seen if isinstance(s, Update)]
    assert len(selects) == 1, [type(s).__name__ for s in read_seen]
    assert len(updates) == 1, [type(s).__name__ for s in write_seen]
    assert selects[0].get_final_froms()[0].name == TaskChatMessage.__tablename__
    assert updates[0].table.name == TaskChatMessage.__tablename__

    return _rendered_where(bind, selects[0]), _rendered_where(bind, updates[0])


def test_the_shared_where_clause_still_has_all_three_legs():
    """Pins what the shared predicate compiles to: task, assistant role,
    question message_type -- and nothing else. This is the runtime
    SQL-equality check #1263 asks for: both sides are captured from the
    statements the production reader and writer actually emit, and each is
    required to equal the same fixed string, so it catches both kinds of
    drift -- one side diverging from the other, and a condition
    disappearing from the shared helper, which moves both sides together
    and would keep a bare equality check green.

    Scope: this pins the reader's first pass against the writer, nothing
    else. A live question row is persisted below, so the first pass hits
    and the reader's superseded-row pass -- which only runs when a caller
    opens it and the first pass comes back empty -- never runs here.
    """
    db = _create_db_session()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        persist_assistant_message(
            db,
            task_id,
            int(task.user_id),
            "A question",
            message_type="question",
        )

        read_where, write_where = _reader_and_writer_where_clauses(db, task_id)

        expected = (
            f"task_chat_messages.task_id = {task_id} "
            "AND task_chat_messages.role = 'assistant' "
            "AND task_chat_messages.message_type = 'question'"
        )
        assert read_where == expected
        assert write_where == expected
    finally:
        db.close()
