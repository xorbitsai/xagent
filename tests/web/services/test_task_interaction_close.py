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

import ast
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import get_args

import pytest
import sqlalchemy as sa
from sqlalchemy import Select, event

from tests.web.services.active_interaction_read_shared import PRE_CHANGE_EQUIVALENT
from tests.web.services.interaction_static_scan_shared import _scan_root
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
    ACTIVE_INTERACTION_UNAVAILABLE_REASONS,
    ActiveInteractionAbsent,
    ActiveInteractionFound,
    ActiveInteractionRead,
    ActiveInteractionUnavailable,
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
# must resolve to ActiveInteractionAbsent rather than to a wrong id.
# --------------------------------------------------------------------------


def test_active_interaction_id_sync_returns_the_live_rows_id(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    assert active_interaction_id_sync(task_id) == ActiveInteractionFound(row_id)


def test_active_interaction_id_sync_reports_absence_without_the_interaction_table(
    db_without_interaction_table,
) -> None:
    db = db_without_interaction_table
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    db.query(Task).filter(Task.id == task_id).update(
        {Task.run_id: "run-a", Task.interaction_protocol_version: 1}
    )
    db.commit()

    assert active_interaction_id_sync(task_id) == ActiveInteractionAbsent()


def test_active_interaction_id_sync_reports_absence_when_the_task_marker_is_null(
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

    assert result == ActiveInteractionAbsent()
    assert caplog.records == []


def test_active_interaction_id_sync_reports_absence_for_an_absent_task(
    db, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(999_999_999)

    assert result == ActiveInteractionAbsent()
    assert caplog.records == []


# --------------------------------------------------------------------------
# active_interaction_id_sync's own four fail-open branches, migrated from
# tests/web/api/test_resume_interaction_seam.py where this function used to
# live as websocket.py's own _active_native_interaction_id_sync: no database
# configured yet, a session that fails to open, the interaction table not
# existing yet, and the row lookup itself raising. The module docstring
# argues at length for why each one resolves to "no active row to close"
# instead of propagating -- these pin that argument down to actual
# behavior, and distinguish the two branches that are expected in normal
# operation (no session factory yet, table not migrated yet -- no warning,
# ActiveInteractionAbsent) from the two that represent a genuine failure
# worth a log line (session open failure, lookup failure --
# ActiveInteractionUnavailable). "Fail-open" here describes the close
# sites, not this function's own return value any more: the two genuine
# failures now report ActiveInteractionUnavailable, distinct from
# ActiveInteractionAbsent, so that a caller can tell them apart even though
# every caller today takes the same action for both -- the type-level
# split is the seam a later change edits to make one particular caller
# (the resume command seam's refusal gate) act on the two differently,
# without having to first separate what this change already kept separate.
#
# The last of the four is reached by two different schema states, and both
# are covered: a lookup that raises because the shared predicate raises
# (stubbed, below) and one that raises because the database really is
# missing a column this function reads -- the pre-migration state built by
# db_without_the_protocol_version_column further down.
# --------------------------------------------------------------------------


def test_active_interaction_id_sync_reports_absence_without_a_session_factory(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``get_optional_session_local() is None`` -- no database configured yet
    for this process -- is the cheap, expected-in-tests case: it must report
    ``ActiveInteractionAbsent()`` without ever calling the (nonexistent)
    session factory, and without logging a warning. A caller that removed
    this branch would fall through to ``SessionLocal()`` with
    ``SessionLocal is None``, which raises ``TypeError`` and is instead
    caught by the *next* branch below -- still resolving to "no active
    row", but as ``ActiveInteractionUnavailable`` and only after logging a
    warning this branch is specifically here to avoid."""

    monkeypatch.setattr(database_module, "get_optional_session_local", lambda: None)

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(1)

    assert result == ActiveInteractionAbsent()
    assert caplog.records == []


def test_active_interaction_id_sync_reports_unavailable_when_opening_a_session_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A session factory that is installed but raises when called -- e.g. a
    prior test left a factory pointed at a since-removed temporary database
    file -- must resolve to ``ActiveInteractionUnavailable("session_unavailable")``,
    and unlike the branch above this is a genuine failure and must be
    logged.

    What the three legacy-resume close sites do with this outcome is
    pinned separately, by
    test_close_keeps_the_marker_when_the_pre_injection_read_failed below:
    translated to ``None``, the close matches no row and the marker stays,
    so the live question the read could not see keeps its reader.
    """

    def _broken_session_local() -> None:
        raise RuntimeError("database file has been removed")

    monkeypatch.setattr(
        database_module, "get_optional_session_local", lambda: _broken_session_local
    )

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(1)

    assert result == ActiveInteractionUnavailable("session_unavailable")
    assert len(caplog.records) == 1
    assert "could not open a session" in caplog.records[0].message


def test_active_interaction_id_sync_reports_absence_when_the_table_gate_reports_missing(
    db,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The table-existence gate, exercised against a database where the
    table does exist: ``interaction_requests_table_exists`` is stubbed to
    ``False`` while a real active row sits in a real table. What that
    isolates is the gate itself -- a caller that removed it would find the
    row and return ``ActiveInteractionFound``, so ``ActiveInteractionAbsent``
    here can only have come from the gate, never from an empty table.

    The gate returning ``False`` for the reason it exists for -- a
    deployment that has not yet run the migration creating
    ``task_interaction_requests`` -- is covered without any stub by
    test_active_interaction_id_sync_reports_absence_without_the_interaction_table
    above, which builds that schema shape for real. Both must resolve to
    ``ActiveInteractionAbsent()`` without a warning: a known deployment
    window is not a failure."""

    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    seed_active_row(db, task_id=task_id, run_id="run-a")

    monkeypatch.setattr(
        "xagent.web.services.task_interaction_close.interaction_requests_table_exists",
        lambda db: False,
    )

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(task_id)

    assert result == ActiveInteractionAbsent()
    assert caplog.records == []


def test_active_interaction_id_sync_reports_unavailable_when_the_lookup_raises(
    db,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failure inside the row lookup itself -- reproduced here by making
    the shared active-row predicate raise, the same seam
    ``_answer_fence_stmt`` reuses -- must resolve to
    ``ActiveInteractionUnavailable("lookup_failed")`` and log a warning
    naming the lookup, not the session-open failure above's message."""

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

    assert result == ActiveInteractionUnavailable("lookup_failed")
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


def test_active_interaction_id_sync_reports_unavailable_when_the_marker_column_is_missing(
    db_without_the_protocol_version_column,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The marker read is the first statement this function issues, and on
    a deployment missing that column it does not come back empty -- it
    raises OperationalError before any gate has run. The catch-all around
    the lookup is what turns that into ``ActiveInteractionUnavailable``:
    the function reports the lookup failure, logs one warning, and lets
    the caller's close match nothing, instead of failing a resume
    injection over a schema state the next migration fixes.

    A real active row is seeded to keep the result honest: the table is
    present and populated here, so nothing but the failing marker read can
    be producing it.
    """
    db = db_without_the_protocol_version_column
    task_id = _seed_task_without_the_marker_column(db, run_id="run-a")
    seed_active_row(db, task_id=task_id, run_id="run-a")

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        result = active_interaction_id_sync(task_id)

    assert result == ActiveInteractionUnavailable("lookup_failed")
    assert len(caplog.records) == 1
    assert "the active interaction row lookup failed" in caplog.records[0].message


def test_close_keeps_the_marker_when_the_pre_injection_read_failed(
    db, monkeypatch
) -> None:
    """The two halves composed: the pre-injection read comes back
    ``ActiveInteractionUnavailable``, the call site translates that to
    ``None`` (the same translation every legacy-resume close site performs
    -- see task_interaction_close.py's callers), and the close is handed
    ``None`` and matches nothing -- and the marker survives, because the
    question the read could not see is still active and still unanswered.
    Clearing it there would point every reader at the legacy transcript
    question while the native row it named waits for an answer.

    This is also this repo's behavioral pin for a translated
    ``ActiveInteractionUnavailable`` reaching one of the three
    legacy-resume close sites: the static gate in
    test_every_call_site_of_active_interaction_id_sync_names_unavailable
    below only checks that ``ActiveInteractionUnavailable`` is named in
    each caller's module, not that the translation is correct, so this is
    the one place asserting the translated value actually leaves the row
    untouched.

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
        active_interaction_read = active_interaction_id_sync(task_id)

    assert active_interaction_read == ActiveInteractionUnavailable(
        "session_unavailable"
    )

    # The value every production call site's translation produces for this
    # state, reached here by a two-branch stand-in rather than a copy of
    # the three branches themselves: what this test is about is the close
    # being handed `None`, not the shape of the translation. Whether each
    # site really keeps Unavailable on its own branch -- and logs it -- is
    # pinned at the sites, in the parameterized order tests in
    # tests/web/api/test_a2a_api.py, tests/web/api/v1/test_task_reply.py
    # and tests/web/api/test_websocket_owner_actor.py.
    if isinstance(active_interaction_read, ActiveInteractionFound):
        observed_id = active_interaction_read.interaction_id
    else:
        observed_id = None

    rowcount = close_legacy_resume_interaction_sync(
        task_id=task_id, run_id="run-a", interaction_id=observed_id
    )

    assert rowcount == 0
    assert row_state(db, row_id).status == "active"
    assert task_marker(db, task_id) == 1


# --------------------------------------------------------------------------
# ACTIVE_INTERACTION_UNAVAILABLE_REASONS -- the two-word reason
# vocabulary the two logger.warning call sites above are keyed to, and
# the table every caller-side test parametrizes over. A future change
# that merges those two log messages back into one would, if it also
# collapsed the reason vocabulary, slip past every assertion above (each
# only checks one reason string at a time); this pins the vocabulary's
# size directly.
# --------------------------------------------------------------------------


def test_active_interaction_unavailable_reasons_is_exactly_two_words() -> None:
    assert ACTIVE_INTERACTION_UNAVAILABLE_REASONS == {
        "session_unavailable",
        "lookup_failed",
    }


# --------------------------------------------------------------------------
# Static gate: every production module that calls active_interaction_id_sync
# must also name ActiveInteractionUnavailable. This does not check that a
# caller's Unavailable branch does anything sensible -- a branch reduced to
# `pass` still satisfies it -- only that the three-state return cannot be
# quietly narrowed back to a two-state one by a caller that pattern-matches
# on ActiveInteractionFound and treats everything else as absent. AST-based
# rather than a substring grep, following the same shape as this package's
# other zero-caller / production-use gates
# (test_task_interaction_service_create_gate.py,
# test_interaction_handoff_production_surface.py,
# test_task_interaction_anchor.py's _anchor_production_uses): a plain-text
# search would also match the name inside a docstring or comment, which
# proves nothing about whether the module's code actually handles the
# third state.
# --------------------------------------------------------------------------

_READER_NAME = "active_interaction_id_sync"
_UNAVAILABLE_NAME = "ActiveInteractionUnavailable"
_CLOSE_MODULE_STEM = "task_interaction_close"


def _references_name(tree: ast.AST, name: str) -> bool:
    """Whether ``name`` appears as a real reference in ``tree`` -- an
    import, a bare identifier, or an attribute access -- as opposed to
    merely appearing inside a string (a docstring or comment mentioning the
    name proves nothing about whether the module's code handles it)."""

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == name for alias in node.names):
                return True
        elif isinstance(node, ast.Name) and node.id == name:
            return True
        elif isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def test_every_call_site_of_active_interaction_id_sync_names_unavailable() -> None:
    """Every production module that calls ``active_interaction_id_sync``
    must also reference ``ActiveInteractionUnavailable`` somewhere in the
    same module. ``task_interaction_close.py`` itself is excluded from the
    scan: it defines both names but is not one of this function's callers.

    What this does not catch, by design: a caller that imports
    ``ActiveInteractionUnavailable`` and then does nothing with it (e.g. an
    ``isinstance`` branch reduced to ``pass``) still passes. That is a
    known, disclosed gap -- the behavioral pin for what the ``Unavailable``
    branch must actually do lives in test_resume_interaction_seam.py (the
    refusal gate) and in test_close_keeps_the_marker_when_the_pre_injection_
    read_failed above (the close sites). This gate only catches the most
    common regression shape: a fifth call site added later that pattern-
    matches on ``ActiveInteractionFound`` and folds everything else into
    "absent" without ever mentioning ``ActiveInteractionUnavailable`` at
    all.
    """

    offenders: list[str] = []
    for path in _scan_root(_CLOSE_MODULE_STEM):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        if _references_name(tree, _READER_NAME) and not _references_name(
            tree, _UNAVAILABLE_NAME
        ):
            offenders.append(str(path))
    assert offenders == []


# --------------------------------------------------------------------------
# Statement-count pin for the marker gate: on every deployment today (the
# protocol marker is NULL for every task -- see active_interaction_id_sync's
# own docstring), the marker gate short-circuits before the row lookup, so
# this reader issues exactly one statement and returns ActiveInteractionAbsent
# rather than paying for the uncached catalog inspection plus the two-table
# join below it. This test covers only that success path: the marker read
# itself can raise (see
# test_active_interaction_id_sync_reports_unavailable_when_the_marker_column_
# is_missing above), and when it does this function returns
# ActiveInteractionUnavailable instead, which this test's own assertion on
# the return value would catch, not paper over. The broader claim that this
# change leaves every caller's user-visible outcome unchanged is carried by
# test_the_pre_change_equivalent_table_covers_every_state and the
# parameterized order tests it feeds, not by this test alone.
# --------------------------------------------------------------------------


@contextmanager
def _counted_selects(bind):
    """Record every ``SELECT`` issued on ``bind`` while the context is
    open. Copied from
    tests/web/services/test_task_interaction_read.py::test_m3_marker_null_
    issues_no_statement_the_reader_would_not, which pins the analogous claim
    for the read surface this function's marker gate mirrors.
    """

    seen: list[object] = []

    def _record(conn, clauseelement, multiparams, params, execution_options):
        if isinstance(clauseelement, Select):
            seen.append(clauseelement)

    event.listen(bind, "before_execute", _record)
    try:
        yield seen
    finally:
        event.remove(bind, "before_execute", _record)


def test_active_interaction_id_sync_issues_only_the_marker_read_under_a_null_marker(
    db,
) -> None:
    """Under a NULL marker -- today's state for every task in every
    deployment, since the only writers of this column are this module's own
    two clears and both write NULL -- this function must return
    ``ActiveInteractionAbsent()`` after issuing exactly the one statement
    that reads the marker itself, and nothing more. A real active row is
    seeded so the assertion is not vacuous: if the marker gate were ever
    removed (or if ``ActiveInteractionUnavailable("lookup_failed")`` were
    ever produced from this branch instead), the row lookup below the gate
    would run and this would fail either on the statement count or on the
    returned value.

    This pins the marker gate's short-circuit and the cost it saves (one
    primary-key lookup instead of an uncached catalog inspection plus a
    two-table join). It covers only the success path -- the marker read can
    itself raise, which is a different situation pinned separately (see
    test_active_interaction_id_sync_reports_unavailable_when_the_marker_
    column_is_missing above) -- so it is not, on its own, the pin for "no
    user-visible behavior changed" across every caller; that claim is
    carried by test_the_pre_change_equivalent_table_covers_every_state and
    the parameterized order tests it feeds.

    Mutation: delete the ``if marker is None: return
    ActiveInteractionAbsent()`` short-circuit inside
    ``active_interaction_id_sync`` and this goes red, because the row
    lookup it guards would then issue a second statement this test asserts
    never happens.
    """
    task_id = seed_task_with_run(db, run_id="run-a", marker=None)
    seed_active_row(db, task_id=task_id, run_id="run-a")
    bind = db.get_bind()

    with _counted_selects(bind) as statements:
        result = active_interaction_id_sync(task_id)

    assert result == ActiveInteractionAbsent()
    assert len(statements) == 1


# --------------------------------------------------------------------------
# No user-visible behavior change: the claim in this change's description
# is not carried by any single test above, and cannot be, because it spans
# every production call site. It is instead a three-layer claim, each layer
# pinned separately:
#
#   1. Which state each situation produces -- pinned one test per situation
#      above (the seven tests from
#      test_active_interaction_id_sync_returns_the_live_rows_id through
#      test_active_interaction_id_sync_reports_unavailable_when_the_marker_
#      column_is_missing).
#   2. Which int | None value each state corresponds to in the shape this
#      reader returned before it became three-state -- pinned once by
#      PRE_CHANGE_EQUIVALENT, written in
#      tests/web/services/active_interaction_read_shared.py because the
#      layer-3 tests read it too, and by the test below.
#   3. That every call site really applies that same projection -- pinned
#      by the parameterized order tests in tests/web/api/test_a2a_api.py,
#      tests/web/api/v1/test_task_reply.py and
#      tests/web/api/test_websocket_owner_actor.py (the three legacy-resume
#      close sites), and by the refusal-gate cells in
#      tests/web/api/test_resume_interaction_seam.py (the fourth call site,
#      which does not close anything but must still let Absent and
#      Unavailable both through unchanged).
#
# All three layers have to hold for the claim to hold; this section is only
# layer 2.
# --------------------------------------------------------------------------


def test_the_pre_change_equivalent_table_covers_every_state() -> None:
    """``PRE_CHANGE_EQUIVALENT`` is what every production call site's
    translation is checked against (the parameterized order tests in the
    three close sites' own test files read the same table out of
    active_interaction_read_shared.py); this pins the table itself as
    exhaustive, so a state or a reason word added later without a
    corresponding row here fails loudly instead of silently narrowing what
    those other tests cover.

    Mutation: add a fourth member to ``ActiveInteractionRead`` without
    adding a row for it here, or add a third word to
    ``ACTIVE_INTERACTION_UNAVAILABLE_REASONS`` without adding a row for it
    here, and one of the two assertions below goes red.

    The first assertion reads the union's members out of the union itself
    (``typing.get_args``) rather than repeating the three class names: a
    hand-written literal set would have to be edited by the same change
    that adds a fourth member, so it would go green again on exactly the
    change this test exists to catch.
    """
    assert {type(state) for state, _ in PRE_CHANGE_EQUIVALENT} == set(
        get_args(ActiveInteractionRead)
    )
    assert {
        state.reason
        for state, _ in PRE_CHANGE_EQUIVALENT
        if isinstance(state, ActiveInteractionUnavailable)
    } == ACTIVE_INTERACTION_UNAVAILABLE_REASONS


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
