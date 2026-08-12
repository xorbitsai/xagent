"""Pin the retirement statements a legacy resume path issues on injection.

``close_legacy_resume_interaction`` (and its short-transaction wrapper
``close_legacy_resume_interaction_sync``) are exercised directly here at
the database level: rowcount classification across every input shape the
close statement can see, the no-op behavior on a deployment without the
interaction table, and a staging-primitive interaction proving the close
is a real behavior change, not a no-op. The two WebSocket injection sites
that wire this into production are covered separately in
tests/web/api/test_websocket_owner_actor.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.web.services.task_interaction_schema_shared import (
    make_row,
    make_task,
    make_trace_event,
    make_user,
    tables_excluding_interaction_requests,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services.task_interaction_close import (
    close_legacy_resume_interaction,
    close_legacy_resume_interaction_sync,
)
from xagent.web.services.task_interaction_staging import (
    InteractionAnchor,
    InteractionSlotTaken,
    stage_interaction_request,
)


@pytest.fixture()
def db(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'interaction_close.db'}")
    session = next(get_db())
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=get_engine())


def _seed_task_with_run(db, *, run_id: str, marker: int | None) -> int:
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    db.query(Task).filter(Task.id == task_id).update(
        {Task.run_id: run_id, Task.interaction_protocol_version: marker}
    )
    db.commit()
    return task_id


def _seed_active_row(db, *, task_id: int, run_id: str) -> int:
    anchor_id = make_trace_event(db, task_id=task_id)
    row = TaskInteractionRequest(
        **make_row(task_id=task_id, resume_trace_event_id=anchor_id, run_id=run_id)
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return int(row.id)


def _row_state(db, row_id: int) -> TaskInteractionRequest:
    db.expire_all()
    return (
        db.query(TaskInteractionRequest)
        .filter(TaskInteractionRequest.id == row_id)
        .one()
    )


def _task_marker(db, task_id: int) -> int | None:
    db.expire_all()
    return db.query(Task).filter(Task.id == task_id).one().interaction_protocol_version


# --------------------------------------------------------------------------
# Rowcount grid -- every input shape the close statement's WHERE fence sees.
# --------------------------------------------------------------------------


def test_close_retires_the_active_row_for_its_own_run(db) -> None:
    task_id = _seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = _seed_active_row(db, task_id=task_id, run_id="run-a")

    rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert rowcount == 1
    row = _row_state(db, row_id)
    assert row.status == "terminated"
    assert row.active_slot is None
    assert row.terminal_reason == "answered_via_legacy_resume"
    assert row.terminated_at is not None
    assert _task_marker(db, task_id) is None


def test_close_is_a_no_op_replaying_an_already_terminated_row(db) -> None:
    task_id = _seed_task_with_run(db, run_id="run-a", marker=1)
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

    rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert rowcount == 0
    row = _row_state(db, row_id)
    assert row.status == "terminated"
    assert row.terminal_reason == original_terminal_reason
    # The clear runs unconditionally: a marker left dangling by an earlier,
    # incomplete write still gets zeroed even though this close matched no
    # row of its own.
    assert _task_marker(db, task_id) is None


def test_close_is_a_no_op_with_no_interaction_rows_at_all(db) -> None:
    """Today's 100% case: the table has no production writer yet."""
    task_id = _seed_task_with_run(db, run_id="run-a", marker=None)

    rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert rowcount == 0
    assert _task_marker(db, task_id) is None


def test_close_does_not_touch_a_different_runs_active_row(db) -> None:
    task_id = _seed_task_with_run(db, run_id="run-a", marker=1)
    orphan_row_id = _seed_active_row(db, task_id=task_id, run_id="run-b")

    rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert rowcount == 0
    orphan = _row_state(db, orphan_row_id)
    assert orphan.status == "active"
    # This call's own run still gets its marker cleared -- the orphan row
    # belongs to a different run's marker, which this call never touches.
    assert _task_marker(db, task_id) is None


def test_close_does_not_overwrite_a_row_already_recycled_by_another_terminal_reason(
    db,
) -> None:
    task_id = _seed_task_with_run(db, run_id="run-a", marker=None)
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

    rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert rowcount == 0
    assert _row_state(db, row_id).terminal_reason == "run_superseded"


def test_close_sync_opens_its_own_transaction_and_commits(db) -> None:
    """The short-transaction wrapper the two WebSocket injection sites
    share: no caller-held session."""
    task_id = _seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = _seed_active_row(db, task_id=task_id, run_id="run-a")

    rowcount = close_legacy_resume_interaction_sync(task_id, "run-a")

    assert rowcount == 1
    assert _row_state(db, row_id).status == "terminated"
    assert _task_marker(db, task_id) is None


# --------------------------------------------------------------------------
# Table absent -- a deployment not yet migrated to task_interaction_requests.
# --------------------------------------------------------------------------


@pytest.fixture()
def db_without_interaction_table(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from xagent.db.sqlite import apply_sqlite_concurrency_pragmas

    engine = create_engine(f"sqlite:///{tmp_path / 'no_interaction_table.db'}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(
        bind=engine, tables=tables_excluding_interaction_requests()
    )
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield session
    finally:
        session.close()


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

    rowcount = close_legacy_resume_interaction_sync(task_id, "run-a")

    assert rowcount == 0
    # The gate is checked before the marker clear too: close_legacy_resume_
    # interaction_sync returns before opening the lock read or the clear
    # statement, so a deployment without the table pays for neither.
    db.expire_all()
    assert (
        db.query(Task).filter(Task.id == task_id).one().interaction_protocol_version
        == 1
    )


# --------------------------------------------------------------------------
# The close statement is a real behavior change, not a no-op: staging a
# second question on the same run behaves differently depending on whether
# it ran.
# --------------------------------------------------------------------------


def _stage(db, *, task_id: int, run_id: str, anchor_id: int, key: str):
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


def test_close_lets_a_second_question_on_the_same_run_become_active(db) -> None:
    """Deleting the close call must turn this red: without it, the second
    stage attempt collides with the first question's still-active slot and
    raises InteractionSlotTaken instead of ever becoming the active row."""
    task_id = _seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)

    first = _stage(db, task_id=task_id, run_id="run-a", anchor_id=anchor_id, key="q1")
    db.commit()
    assert first.created is True

    close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
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
    task_id = _seed_task_with_run(db, run_id="run-a", marker=1)
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
