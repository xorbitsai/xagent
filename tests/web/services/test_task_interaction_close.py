"""Pin the retirement and marker-clear statements a legacy resume path issues.

``close_legacy_resume_interaction`` (and its short-transaction wrapper
``close_legacy_resume_interaction_sync``) and
``clear_interaction_marker_if_unpaired`` are exercised directly here at the
database level: rowcount classification across every input shape the close
statement can see, the no-op behavior on a deployment without the
interaction table, the ``NOT EXISTS`` guard the two marker-clear-only call
sites depend on, and a staging-primitive interaction proving the close is a
real behavior change, not a no-op. The production call sites that wire
these functions into the WebSocket and A2A resume paths are covered
separately in tests/web/api/test_websocket_owner_actor.py and
tests/web/api/test_a2a_api.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

from tests.web.services.task_interaction_schema_shared import (
    make_row,
    make_task,
    make_trace_event,
    make_user,
    row_state,
    seed_active_row,
    seed_task_with_run,
    tables_excluding_interaction_requests,
    task_marker,
)
from xagent.web.models import database as database_module
from xagent.web.models.database import (
    Base,
    configure_db,
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.task import Task
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import ops_signals
from xagent.web.services.task_interaction_close import (
    _classify_close_rowcount,
    active_interaction_id_sync,
    clear_interaction_marker_if_unpaired,
    close_legacy_resume_interaction,
    close_legacy_resume_interaction_sync,
)
from xagent.web.services.task_interaction_staging import (
    InteractionAnchor,
    InteractionSlotTaken,
    stage_interaction_request,
)

_CLOSE_MODULE_NAME = "xagent.web.services.task_interaction_close"


@pytest.fixture(autouse=True)
def _reset_ops_signals():
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)
    yield
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)


@pytest.fixture()
def db(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'interaction_close.db'}")
    session = next(get_db())
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=get_engine())


# --------------------------------------------------------------------------
# _classify_close_rowcount -- the one place every rowcount the close
# statement can produce gets classified, called directly, no database
# involved.
# --------------------------------------------------------------------------


def test_classify_close_rowcount_logs_info_for_the_expected_single_row_case(
    caplog,
) -> None:
    with caplog.at_level(logging.INFO, logger=_CLOSE_MODULE_NAME):
        _classify_close_rowcount(1, task_id=1, run_id="run-a", unmatched_row=None)

    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert ops_signals.active_degradations() == {}


@pytest.mark.parametrize(
    "unmatched_row", ["no_id_read", "row_absent", "row_status=terminated"]
)
def test_classify_close_rowcount_logs_debug_for_the_common_no_op_case(
    caplog, unmatched_row: str
) -> None:
    """The zero branch carries the description through to the log line
    verbatim: the level says a close matched nothing, and this says which
    of the situations that fold into that rowcount it was."""
    with caplog.at_level(logging.DEBUG, logger=_CLOSE_MODULE_NAME):
        _classify_close_rowcount(
            0, task_id=1, run_id="run-a", unmatched_row=unmatched_row
        )

    assert [record.levelno for record in caplog.records] == [logging.DEBUG]
    assert f"unmatched_row={unmatched_row}" in caplog.records[0].getMessage()
    assert ops_signals.active_degradations() == {}


def test_classify_close_rowcount_logs_error_and_registers_a_signal_for_an_impossible_rowcount(
    caplog,
) -> None:
    """rowcount > 1 needs either the primary key the close binds to or
    uq_task_interaction_active_slot to have stopped holding -- see the
    close module's docstring. Called directly here, because no database
    this suite can build produces that rowcount. Logged at error and
    surfaced on /health, not raised."""
    with caplog.at_level(logging.ERROR, logger=_CLOSE_MODULE_NAME):
        _classify_close_rowcount(2, task_id=7, run_id="run-b", unmatched_row=None)

    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert (
        ops_signals.INTERACTION_LEGACY_RESUME_CLOSE_ROWCOUNT_ANOMALY
        in ops_signals.active_degradations()
    )


# --------------------------------------------------------------------------
# Rowcount grid -- every input shape the close statement's WHERE fence sees.
# --------------------------------------------------------------------------


def test_close_retires_the_active_row_for_its_own_run(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=row_id
    )
    db.commit()

    assert rowcount == 1
    row = row_state(db, row_id)
    assert row.status == "terminated"
    assert row.active_slot is None
    assert row.terminal_reason == "answered_via_legacy_resume"
    assert row.terminated_at is not None
    assert task_marker(db, task_id) is None


def test_close_is_a_no_op_replaying_an_already_terminated_row(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)
    row = TaskInteractionRequest(
        **make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            run_id="run-a",
            status="terminated",
        )
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    row_id = int(row.id)
    original_terminal_reason = row.terminal_reason

    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=row_id
    )
    db.commit()

    assert rowcount == 0
    row = row_state(db, row_id)
    assert row.status == "terminated"
    assert row.terminal_reason == original_terminal_reason
    # The clear does not need this close to have matched anything -- it
    # needs nothing active to be left, and this row is already terminated.
    # A marker left dangling by an earlier, incomplete write still gets
    # zeroed here.
    assert task_marker(db, task_id) is None


@pytest.mark.parametrize("seeded_marker", [None, 1])
def test_close_is_a_no_op_with_no_interaction_rows_at_all(
    db, seeded_marker: int | None
) -> None:
    """No interaction row was ever staged for this run -- today's 100%
    case, since the table has no production writer yet. The condition on
    the clear is "no active row remains", not "the close matched
    something", so the marker is zeroed the same way whether it started
    unset or set."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=seeded_marker)

    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=None
    )
    db.commit()

    assert rowcount == 0
    assert task_marker(db, task_id) is None


def test_close_does_not_touch_a_different_runs_active_row(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    orphan_row_id = seed_active_row(db, task_id=task_id, run_id="run-b")

    # The orphan's own id is passed in deliberately, so the run predicate
    # is the only thing left that can reject it.
    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=orphan_row_id
    )
    db.commit()

    assert rowcount == 0
    orphan = row_state(db, orphan_row_id)
    assert orphan.status == "active"
    # This call's own run still gets its marker cleared -- the orphan row
    # belongs to a different run's marker, which this call never touches.
    assert task_marker(db, task_id) is None


def test_close_does_not_overwrite_a_row_already_recycled_by_another_terminal_reason(
    db,
) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=None)
    anchor_id = make_trace_event(db, task_id=task_id)
    row = TaskInteractionRequest(
        **make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            run_id="run-a",
            status="terminated",
            terminal_reason="run_superseded",
        )
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    row_id = int(row.id)

    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=row_id
    )
    db.commit()

    assert rowcount == 0
    assert row_state(db, row_id).terminal_reason == "run_superseded"


# What the primary-key predicate does on its own, holding everything else
# constant: one active row for this task and run, and only the id handed to
# the close varies. The marker follows the row, not the rowcount: it clears
# when nothing active is left, and survives when the close missed the live
# row -- whatever the reason it missed.
@pytest.mark.parametrize(
    ("id_to_pass", "expected_rowcount", "expected_marker"),
    [
        pytest.param("the_active_row", 1, None, id="the_row_observed_before_injection"),
        pytest.param(None, 0, 1, id="no_row_was_active_at_injection_time"),
        pytest.param("another_row", 0, 1, id="a_row_that_is_not_this_tasks_active_one"),
    ],
)
def test_close_retires_only_the_row_it_was_given(
    db, id_to_pass: str | None, expected_rowcount: int, expected_marker: int | None
) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    other_task_id = seed_task_with_run(db, run_id="run-z", marker=1)
    other_row_id = seed_active_row(db, task_id=other_task_id, run_id="run-z")

    interaction_id = {
        "the_active_row": row_id,
        "another_row": other_row_id,
        None: None,
    }[id_to_pass]
    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=interaction_id
    )
    db.commit()

    assert rowcount == expected_rowcount
    assert (row_state(db, row_id).status == "terminated") is (expected_rowcount == 1)
    # The other task's row is out of range of this close regardless.
    assert row_state(db, other_row_id).status == "active"
    assert task_marker(db, task_id) == expected_marker


# A zero rowcount is one number for situations an operator has to tell
# apart, so the debug line carries a description of which one it was,
# resolved from the database at the close's own call point. Four shapes,
# all of them misses, and a different set from the grid above: that one
# keeps the seeded rows fixed and varies only the id the close is handed,
# including the id that matches; these vary the seeded row too.
@pytest.mark.parametrize(
    ("row_to_seed", "id_to_pass", "expected_description"),
    [
        pytest.param(None, None, "no_id_read", id="the_read_produced_no_id"),
        pytest.param(None, 999_999_999, "row_absent", id="the_id_names_no_row"),
        pytest.param(
            "terminated",
            "the_seeded_row",
            "row_status=terminated",
            id="another_path_closed_the_row_first",
        ),
        pytest.param(
            "active",
            "the_seeded_row",
            "row_status=active",
            id="the_row_is_live_but_belongs_to_another_run",
        ),
    ],
)
def test_close_records_why_it_matched_no_row(
    db,
    caplog: pytest.LogCaptureFixture,
    row_to_seed: str | None,
    id_to_pass: object,
    expected_description: str,
) -> None:
    """The last case is the one that used to be indistinguishable from an
    empty table: a row that is still active, and still missed, because this
    close fences on run_id as well as on the id."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    seeded_row_id: int | None = None
    if row_to_seed == "terminated":
        anchor_id = make_trace_event(db, task_id=task_id)
        row = TaskInteractionRequest(
            **make_row(
                task_id=task_id,
                resume_trace_event_id=anchor_id,
                run_id="run-a",
                status="terminated",
            )
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        seeded_row_id = int(row.id)
    elif row_to_seed == "active":
        seeded_row_id = seed_active_row(db, task_id=task_id, run_id="run-b")

    interaction_id = seeded_row_id if id_to_pass == "the_seeded_row" else id_to_pass

    with caplog.at_level(logging.DEBUG, logger=_CLOSE_MODULE_NAME):
        rowcount = close_legacy_resume_interaction(
            db, task_id=task_id, run_id="run-a", interaction_id=interaction_id
        )
    db.commit()

    assert rowcount == 0
    assert [record.levelno for record in caplog.records] == [logging.DEBUG]
    assert f"unmatched_row={expected_description}" in caplog.records[0].getMessage()


def test_close_sync_opens_its_own_transaction_and_commits(db) -> None:
    """The short-transaction wrapper the two WebSocket injection sites
    share: no caller-held session."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    rowcount = close_legacy_resume_interaction_sync(
        task_id=task_id, run_id="run-a", interaction_id=row_id
    )

    assert rowcount == 1
    assert row_state(db, row_id).status == "terminated"
    assert task_marker(db, task_id) is None


# --------------------------------------------------------------------------
# Table absent -- a deployment not yet migrated to task_interaction_requests.
# --------------------------------------------------------------------------


@pytest.fixture()
def db_without_interaction_table(tmp_path):
    """A deployment shape missing task_interaction_requests -- bound as the
    *global* engine/session factory, not a private one.

    close_legacy_resume_interaction_sync (unlike the other functions this
    module tests) takes no db argument of its own: it opens its own session
    through get_session_local(), which reads the process-global factory.
    A fixture that built its own private engine here and handed back a
    session from it would leave that global factory pointed wherever the
    previous test left it, so close_legacy_resume_interaction_sync would run
    against a different database than the one this fixture seeds and
    asserts against -- the table-absence gate it is supposed to exercise
    would never actually see this fixture's schema. configure_db() only
    binds the engine and session factory; it does not create any tables
    (unlike init_db()), so the subset schema below is still built by hand.
    """
    previous_engine = database_module._engine
    previous_session_local = database_module._SessionLocal
    configure_db(db_url=f"sqlite:///{tmp_path / 'no_interaction_table.db'}")
    Base.metadata.create_all(
        bind=get_engine(), tables=tables_excluding_interaction_requests()
    )
    session = get_session_local()()
    try:
        yield session
    finally:
        session.close()
        # Restore the prior global factory so this fixture's rebinding does
        # not leak into whatever test runs next in this file (or module).
        database_module._engine = previous_engine
        database_module._SessionLocal = previous_session_local


def test_close_no_ops_when_the_interaction_table_does_not_exist(
    db_without_interaction_table,
) -> None:
    db = db_without_interaction_table
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    db.query(Task).filter(Task.id == task_id).update(
        {Task.run_id: "run-a", Task.interaction_protocol_version: 1}
    )
    db.commit()

    rowcount = close_legacy_resume_interaction_sync(
        task_id=task_id, run_id="run-a", interaction_id=None
    )

    assert rowcount == 0
    # The gate is checked before the marker clear too: close_legacy_resume_
    # interaction_sync returns before opening the lock read or the clear
    # statement, so a deployment without the table pays for neither.
    db.expire_all()
    assert (
        db.query(Task).filter(Task.id == task_id).one().interaction_protocol_version
        == 1
    )


def test_clear_marker_if_unpaired_no_ops_when_the_interaction_table_does_not_exist(
    db_without_interaction_table,
) -> None:
    db = db_without_interaction_table
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    db.query(Task).filter(Task.id == task_id).update(
        {Task.run_id: "run-a", Task.interaction_protocol_version: 1}
    )
    db.commit()

    clear_interaction_marker_if_unpaired(db, task_id=task_id, run_id="run-a")
    db.commit()

    db.expire_all()
    assert (
        db.query(Task).filter(Task.id == task_id).one().interaction_protocol_version
        == 1
    )


# --------------------------------------------------------------------------
# active_interaction_id_sync -- the pre-injection read whose result the close
# binds to. Every caller reaches it through a patched name, so the body is
# exercised here: the id it returns for a live row, and the two shapes that
# must degrade to None rather than to a wrong id.
# --------------------------------------------------------------------------


def test_active_interaction_id_sync_returns_the_live_rows_id(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    assert active_interaction_id_sync(task_id) == row_id


def test_active_interaction_id_sync_returns_none_without_the_interaction_table(
    db_without_interaction_table,
) -> None:
    db = db_without_interaction_table
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    db.query(Task).filter(Task.id == task_id).update(
        {Task.run_id: "run-a", Task.interaction_protocol_version: 1}
    )
    db.commit()

    assert active_interaction_id_sync(task_id) is None


def test_active_interaction_id_sync_returns_none_when_the_task_marker_is_null(
    db, caplog: pytest.LogCaptureFixture
) -> None:
    """``tasks.interaction_protocol_version`` being ``NULL`` means no native
    row was ever staged for this task's current wait -- the same first step
    ``get_pending_interaction_question`` takes on the read side -- so the
    interaction table goes unqueried even though a real active row exists
    here. Not a failure, so no warning."""

    task_id = seed_task_with_run(db, run_id="run-a", marker=None)
    seed_active_row(db, task_id=task_id, run_id="run-a")

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(task_id)

    assert result is None
    assert caplog.records == []


def test_active_interaction_id_sync_returns_none_for_an_absent_task(
    db, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(999_999_999)

    assert result is None
    assert caplog.records == []


# --------------------------------------------------------------------------
# active_interaction_id_sync's own four fail-open branches, migrated from
# tests/web/api/test_resume_interaction_seam.py where this function used to
# live as websocket.py's own _active_native_interaction_id_sync: no database
# configured yet, a session that fails to open, the interaction table not
# existing yet, and the row lookup itself raising. The module docstring
# argues at length for why each one resolves to "assume no active row"
# instead of propagating -- these pin that argument down to actual
# behavior, and distinguish the two branches that are expected in normal
# operation (no session factory yet, table not migrated yet -- no warning)
# from the two that represent a genuine failure worth a log line (session
# open failure, lookup failure).
#
# The last of the four is reached by two different schema states, and both
# are covered: a lookup that raises because the shared predicate raises
# (stubbed, below) and one that raises because the database really is
# missing a column this function reads -- the pre-migration state built by
# db_without_the_protocol_version_column further down.
# --------------------------------------------------------------------------


def test_active_interaction_id_sync_returns_none_without_a_session_factory(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``get_optional_session_local() is None`` -- no database configured yet
    for this process -- is the cheap, expected-in-tests case: it must return
    ``None`` without ever calling the (nonexistent) session factory, and
    without logging a warning. A caller that removed this branch would fall
    through to ``SessionLocal()`` with ``SessionLocal is None``, which raises
    ``TypeError`` and is instead caught by the *next* branch below -- still
    returning ``None``, but only after logging a warning this branch is
    specifically here to avoid."""

    monkeypatch.setattr(database_module, "get_optional_session_local", lambda: None)

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(1)

    assert result is None
    assert caplog.records == []


def test_active_interaction_id_sync_returns_none_when_opening_a_session_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A session factory that is installed but raises when called -- e.g. a
    prior test left a factory pointed at a since-removed temporary database
    file -- must also resolve to "no active row", but unlike the branch
    above this is a genuine failure and must be logged.

    What "closes nothing" costs is pinned separately, by
    test_close_keeps_the_marker_when_the_pre_injection_read_failed below:
    the close matches no row and the marker stays, so the live question the
    read could not see keeps its reader.
    """

    def _broken_session_local() -> None:
        raise RuntimeError("database file has been removed")

    monkeypatch.setattr(
        database_module, "get_optional_session_local", lambda: _broken_session_local
    )

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(1)

    assert result is None
    assert len(caplog.records) == 1
    assert "could not open a session" in caplog.records[0].message


def test_active_interaction_id_sync_returns_none_when_the_table_gate_reports_missing(
    db,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The table-existence gate, exercised against a database where the
    table does exist: ``interaction_requests_table_exists`` is stubbed to
    ``False`` while a real active row sits in a real table. What that
    isolates is the gate itself -- a caller that removed it would find the
    row and return its id, so ``None`` here can only have come from the
    gate, never from an empty table.

    The gate returning ``False`` for the reason it exists for -- a
    deployment that has not yet run the migration creating
    ``task_interaction_requests`` -- is covered without any stub by
    test_active_interaction_id_sync_returns_none_without_the_interaction_table
    above, which builds that schema shape for real. Both must resolve to
    ``None`` without a warning: a known deployment window is not a
    failure."""

    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    seed_active_row(db, task_id=task_id, run_id="run-a")

    monkeypatch.setattr(
        "xagent.web.services.task_interaction_close.interaction_requests_table_exists",
        lambda db: False,
    )

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(task_id)

    assert result is None
    assert caplog.records == []


def test_active_interaction_id_sync_returns_none_when_the_lookup_raises(
    db,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failure inside the row lookup itself -- reproduced here by making
    the shared active-row predicate raise, the same seam
    ``_answer_fence_stmt`` reuses -- must resolve to "no active row" and log
    a warning naming the lookup, not the session-open failure above's
    message."""

    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    seed_active_row(db, task_id=task_id, run_id="run-a")

    def _broken_criteria() -> list[object]:
        raise RuntimeError("criteria unavailable")

    monkeypatch.setattr(
        "xagent.web.services.task_interaction_service._active_native_row_criteria",
        _broken_criteria,
    )

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(task_id)

    assert result is None
    assert len(caplog.records) == 1
    assert "the active interaction row lookup failed" in caplog.records[0].message


def _schema_without_the_protocol_version_column() -> sa.MetaData:
    """This repo's full schema, minus ``tasks.interaction_protocol_version``.

    The column and the CHECK constraint that names it
    (``ck_tasks_interaction_protocol_version``) are both left out, which is
    what the tasks table looked like before the 2026-08-10 migration added
    the two together.

    Built by cloning every table into a fresh MetaData and editing the
    clone, rather than by creating the real schema and dropping the column
    afterwards: SQLite refuses ``ALTER TABLE tasks DROP COLUMN
    interaction_protocol_version`` outright while a CHECK constraint still
    names that column (measured on SQLite 3.53.4: "error in table tasks
    after drop column: no such column: interaction_protocol_version"), and
    dropping a CHECK constraint is not something SQLite's ALTER TABLE can
    do either. Every table is cloned, not just tasks, because a lone tasks
    clone cannot resolve its own foreign keys to users and the other tables
    it references.

    ``Table.to_metadata`` has no column filter, so the removal below reaches
    into the clone's private column collection. It is done to the clone and
    never to ``Base.metadata``, so the real mapping is untouched.
    """
    metadata = sa.MetaData()
    for table in Base.metadata.sorted_tables:
        table.to_metadata(metadata)
    tasks = metadata.tables[Task.__tablename__]
    tasks._columns.remove(tasks.c.interaction_protocol_version)
    for constraint in list(tasks.constraints):
        if isinstance(
            constraint, sa.CheckConstraint
        ) and "interaction_protocol_version" in str(constraint.sqltext):
            tasks.constraints.discard(constraint)
    return metadata


@pytest.fixture()
def db_without_the_protocol_version_column(tmp_path):
    """A deployment carrying task_interaction_requests but not the marker
    column active_interaction_id_sync reads first.

    Not a shape any single migration produces on its own -- the table and
    the column arrive one migration apart -- but it is what a partially
    migrated deployment can hold, and it is the one schema state that puts
    a failing statement in front of this function rather than an empty
    result set. Bound as the *global* engine and session factory for the
    same reason db_without_interaction_table above is: the function under
    test opens its own session through the process-global factory, so a
    private engine here would leave it reading some other database.
    """
    previous_engine = database_module._engine
    previous_session_local = database_module._SessionLocal
    configure_db(db_url=f"sqlite:///{tmp_path / 'no_marker_column.db'}")
    _schema_without_the_protocol_version_column().create_all(bind=get_engine())
    session = get_session_local()()
    try:
        yield session
    finally:
        session.close()
        database_module._engine = previous_engine
        database_module._SessionLocal = previous_session_local


def _seed_task_without_the_marker_column(db, *, run_id: str) -> int:
    """Insert one task into a tasks table that has no marker column.

    A Core INSERT naming its values explicitly, not the ORM helpers the
    other fixtures use: those end in ``db.refresh(task)``, which SELECTs
    every column the Task mapping declares -- including the one this
    schema does not have -- and would fail in the fixture instead of in the
    function under test. The INSERT below never names that column, so the
    statement is legal against this schema even though it is compiled from
    the full mapping.
    """
    user_id = make_user(db)
    result = db.execute(
        sa.insert(Task.__table__).values(
            user_id=user_id, title="pre-migration schema fixture task", run_id=run_id
        )
    )
    db.commit()
    return int(result.inserted_primary_key[0])


def test_active_interaction_id_sync_returns_none_when_the_marker_column_is_missing(
    db_without_the_protocol_version_column,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The marker read is the first statement this function issues, and on
    a deployment missing that column it does not come back empty -- it
    raises OperationalError before any gate has run. The catch-all around
    the lookup is what turns that into "assume no active row": the function
    returns None, logs one warning, and lets the caller's close match
    nothing, instead of failing a resume injection over a schema state the
    next migration fixes.

    A real active row is seeded to keep the None honest: the table is
    present and populated here, so nothing but the failing marker read can
    be producing it.
    """
    db = db_without_the_protocol_version_column
    task_id = _seed_task_without_the_marker_column(db, run_id="run-a")
    seed_active_row(db, task_id=task_id, run_id="run-a")

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(task_id)

    assert result is None
    assert len(caplog.records) == 1
    assert "the active interaction row lookup failed" in caplog.records[0].message


def test_close_keeps_the_marker_when_the_pre_injection_read_failed(
    db, monkeypatch
) -> None:
    """The two halves composed: the pre-injection read fails, so the close
    is handed ``None`` and matches nothing -- and the marker survives,
    because the question the read could not see is still active and still
    unanswered. Clearing it there would point every reader at the legacy
    transcript question while the native row it named waits for an answer.

    The failure is injected only for the read: the close below opens its
    own session through the same factory and has to reach a working
    database, which is exactly the production shape -- one unreadable
    query, not an unreachable database.
    """
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    def _broken_session_local():
        raise sa.exc.OperationalError("SELECT 1", {}, Exception("database is locked"))

    with monkeypatch.context() as failing_read:
        failing_read.setattr(
            database_module, "get_optional_session_local", lambda: _broken_session_local
        )
        observed_id = active_interaction_id_sync(task_id)

    assert observed_id is None

    rowcount = close_legacy_resume_interaction_sync(
        task_id=task_id, run_id="run-a", interaction_id=observed_id
    )

    assert rowcount == 0
    assert row_state(db, row_id).status == "active"
    assert task_marker(db, task_id) == 1


# --------------------------------------------------------------------------
# Compensation clear -- the NOT EXISTS guard.
# --------------------------------------------------------------------------


def test_clear_marker_if_unpaired_zeroes_a_marker_with_no_active_row(db) -> None:
    """Sequence 'close already committed, then a compensation path runs'.

    The row is already terminated (the close already ran); a marker left
    at 1 -- however that happened -- has nothing to protect and is zeroed.
    """
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)
    row = TaskInteractionRequest(
        **make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            run_id="run-a",
            status="terminated",
        )
    )
    db.add(row)
    db.commit()

    clear_interaction_marker_if_unpaired(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert task_marker(db, task_id) is None


def test_clear_marker_if_unpaired_leaves_a_still_active_row_untouched(db) -> None:
    """This is the mutation-testable half: removing the NOT EXISTS guard
    would zero a marker that still names a live question."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    clear_interaction_marker_if_unpaired(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert task_marker(db, task_id) == 1
    assert row_state(db, row_id).status == "active"


# --------------------------------------------------------------------------
# The close statement is a real behavior change, not a no-op: staging a
# second question on the same run behaves differently depending on whether
# it ran.
# --------------------------------------------------------------------------


def _stage(
    db,
    *,
    task_id: int,
    run_id: str,
    anchor_id: int,
    key: str,
):
    now = datetime.now(timezone.utc)
    return stage_interaction_request(
        db,
        task_id=task_id,
        run_id=run_id,
        anchor=InteractionAnchor(
            trace_event_id=anchor_id,
            resume_event_id="resume-event-1",
            resume_execution_id="resume-exec-1",
            resume_run_partition=run_id,
        ),
        kind="clarification",
        protocol_version=1,
        origin="internal",
        request_payload={"prompt": key},
        request_idempotency_key=key,
        expires_at=now + timedelta(minutes=15),
        now=now,
    )


def _stage_a_replacement_question_the_way_a_resumed_agent_would(
    db, *, task_id: int, run_id: str, anchor_id: int
):
    """Put a second question on the same run into the active slot, the way
    a resumed agent's own ``stage_interaction_request`` call does.

    The first question has to be reclaimable for that INSERT to land at
    all -- ``uq_task_interaction_active_slot`` allows one active row per
    task -- so its deadline is moved into the past first, standing in for
    the time that elapses while a resumed agent works. The second call
    then takes ``_reclaim_stale_slot_stmt``'s expired-row branch on its
    own; nothing here reclaims by hand. Returns ``(first_id, second_id)``.
    """
    first = _stage(db, task_id=task_id, run_id=run_id, anchor_id=anchor_id, key="q1")
    db.commit()
    # Both columns move together: ck_task_interaction_requests_expiry_
    # after_creation requires expires_at > created_at, so an already-lapsed
    # deadline has to belong to a row that was also created earlier.
    staged_at = datetime.now(timezone.utc) - timedelta(hours=2)
    db.execute(
        sa.update(TaskInteractionRequest)
        .where(TaskInteractionRequest.id == first.staged_db_id)
        .values(created_at=staged_at, expires_at=staged_at + timedelta(minutes=15))
    )
    db.commit()

    second = _stage(db, task_id=task_id, run_id=run_id, anchor_id=anchor_id, key="q2")
    db.commit()
    assert second.created is True
    return int(first.staged_db_id), int(second.staged_db_id)


def test_close_leaves_a_question_staged_after_the_injection_alone(db) -> None:
    """The window this close is keyed against. Injecting the user message
    is what resumes the agent, so between the observation and the close the
    resumed agent can ask something new. Retiring that new question as
    "answered via legacy resume" would silently discard a question nobody
    ever saw."""

    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)

    observed_id, staged_after_injection_id = (
        _stage_a_replacement_question_the_way_a_resumed_agent_would(
            db, task_id=task_id, run_id="run-a", anchor_id=anchor_id
        )
    )

    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=observed_id
    )
    db.commit()

    assert rowcount == 0
    survivor = row_state(db, staged_after_injection_id)
    assert survivor.status == "active"
    assert survivor.terminal_reason is None
    # The row that was observed before injection is terminal either way --
    # the reclaim retired it as expired when the new question took the slot.
    assert row_state(db, observed_id).status == "terminated"
    # The marker stays with the surviving question. This is the whole
    # reason the clear is conditioned: the run does have a live native
    # question, and zeroing the marker would send every reader to the
    # legacy transcript question instead of to this one.
    assert task_marker(db, task_id) == 1


def test_close_lets_a_second_question_on_the_same_run_become_active(db) -> None:
    """Deleting the close call must turn this red: without it, the second
    stage attempt collides with the first question's still-active slot and
    raises InteractionSlotTaken instead of ever becoming the active row."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)

    first = _stage(db, task_id=task_id, run_id="run-a", anchor_id=anchor_id, key="q1")
    db.commit()
    assert first.created is True

    close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=first.staged_db_id
    )
    db.commit()

    second = _stage(db, task_id=task_id, run_id="run-a", anchor_id=anchor_id, key="q2")
    db.commit()

    assert second.created is True
    active = (
        db.query(TaskInteractionRequest)
        .filter(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.status == "active",
        )
        .one()
    )
    assert active.id == second.staged_db_id
    assert active.request_idempotency_key == "q2"


def test_without_the_close_call_a_second_question_cannot_become_active(db) -> None:
    """The 'delete the close call' mutation, run for real: with no close in
    between, the first question's active row is still fresh (not expired,
    same run), so the second stage attempt's INSERT collides with the
    unique active-slot constraint and raises InteractionSlotTaken -- the
    first question remains the only active row."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)

    first = _stage(db, task_id=task_id, run_id="run-a", anchor_id=anchor_id, key="q1")
    db.commit()

    with pytest.raises(InteractionSlotTaken):
        _stage(db, task_id=task_id, run_id="run-a", anchor_id=anchor_id, key="q2")
    db.rollback()

    active = (
        db.query(TaskInteractionRequest)
        .filter(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.status == "active",
        )
        .one()
    )
    assert active.id == first.staged_db_id
    assert active.request_idempotency_key == "q1"
