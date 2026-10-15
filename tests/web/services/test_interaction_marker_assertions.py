"""Self-tests for the two marker pairing assertions, on SQLite.

Each case seeds a task on run ``run-a`` with a chosen marker, a chosen
interaction row and a chosen status, then checks what the assertion
accepts and the exact message it raises with.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.interaction_marker_shared import (
    assert_interaction_marker_invariant,
    assert_marker_residue,
)
from tests.web.services.task_interaction_schema_shared import (
    make_row,
    make_trace_event,
    seed_task_with_run,
)
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_interaction import TaskInteractionRequest


@contextmanager
def _sqlite_session(path: Path, *, check_constraints: bool) -> Iterator[Session]:
    engine = sa.create_engine(f"sqlite:///{path}")

    @sa.event.listens_for(engine, "connect")
    def configure(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")
        if not check_constraints:
            connection.execute("PRAGMA ignore_check_constraints=ON")

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def db(tmp_path) -> Iterator[Session]:
    with _sqlite_session(
        tmp_path / "marker_assertions.db", check_constraints=True
    ) as session:
        yield session


@pytest.fixture()
def unchecked_db(tmp_path) -> Iterator[Session]:
    """A database whose CHECK constraints are not enforced, for the one
    case today's constraints cannot hold: a protocol version other than 1."""
    with _sqlite_session(
        tmp_path / "marker_assertions.db", check_constraints=False
    ) as session:
        yield session


def _seed(
    db: Session,
    *,
    marker: int | None,
    row: tuple[str, str] | None,
    status: TaskStatus,
    task_run_id: str = "run-a",
    protocol_version: int = 1,
) -> tuple[int, int | None]:
    """Seed one task and at most one interaction row ``(run_id, status)``;
    return the task id and the row id."""
    task_id = seed_task_with_run(db, run_id=task_run_id, marker=marker)
    db.query(Task).filter(Task.id == task_id).update({Task.status: status})
    db.commit()
    if row is None:
        return task_id, None
    row_run_id, row_status = row
    seeded = TaskInteractionRequest(
        **make_row(
            task_id=task_id,
            resume_trace_event_id=make_trace_event(db, task_id=task_id),
            status=row_status,
            run_id=row_run_id,
            protocol_version=protocol_version,
        )
    )
    db.add(seeded)
    db.commit()
    return task_id, int(seeded.id)


@pytest.mark.parametrize(
    ("marker", "row"),
    [
        pytest.param(None, None, id="no_row_no_marker"),
        pytest.param(1, ("run-a", "active"), id="active_row_and_marker"),
        pytest.param(None, ("run-a", "terminated"), id="terminated_row_no_marker"),
        pytest.param(None, ("run-b", "active"), id="other_run_active_row_no_marker"),
    ],
)
def test_marker_residue_accepts_paired_states(
    db: Session, marker: int | None, row: tuple[str, str] | None
) -> None:
    task_id, _row_id = _seed(db, marker=marker, row=row, status=TaskStatus.RUNNING)

    assert assert_marker_residue(db, task_id=task_id, run_id="run-a") is None


@pytest.mark.parametrize(
    ("marker", "row", "expected"),
    [
        pytest.param(
            1,
            None,
            "task {t} run 'run-a': no active interaction row, "
            "interaction_protocol_version=1",
            id="marker_without_row",
        ),
        pytest.param(
            None,
            ("run-a", "active"),
            "task {t} run 'run-a': active interaction row {r} has "
            "protocol_version=1, interaction_protocol_version=None",
            id="row_without_marker",
        ),
    ],
)
def test_marker_residue_reports_the_half_left_behind(
    db: Session, marker: int | None, row: tuple[str, str] | None, expected: str
) -> None:
    task_id, row_id = _seed(db, marker=marker, row=row, status=TaskStatus.RUNNING)

    with pytest.raises(AssertionError) as excinfo:
        assert_marker_residue(db, task_id=task_id, run_id="run-a")

    assert str(excinfo.value) == expected.format(t=task_id, r=row_id)


def test_marker_residue_refuses_a_run_the_task_is_not_on(db: Session) -> None:
    """The pairing is defined for the task's current run only, so a run id
    the task has moved off is a failure, not a pass."""
    task_id, _row_id = _seed(
        db, marker=None, row=None, status=TaskStatus.RUNNING, task_run_id="run-b"
    )

    with pytest.raises(AssertionError) as excinfo:
        assert_marker_residue(db, task_id=task_id, run_id="run-a")

    assert str(excinfo.value) == f"task {task_id} is on run 'run-b', not run 'run-a'"


@pytest.mark.parametrize(
    ("status", "marker", "row", "expected"),
    [
        pytest.param(
            TaskStatus.WAITING_FOR_USER,
            1,
            ("run-a", "active"),
            None,
            id="waiting_paired",
        ),
        pytest.param(TaskStatus.WAITING_FOR_USER, None, None, None, id="waiting_empty"),
        pytest.param(
            TaskStatus.WAITING_FOR_USER,
            1,
            None,
            "task {t} is waiting_for_user with interaction_protocol_version=1 "
            "and 0 active interaction rows on run 'run-a'",
            id="waiting_marker_without_row",
        ),
        pytest.param(
            TaskStatus.WAITING_FOR_USER,
            None,
            ("run-a", "active"),
            "task {t} is waiting_for_user with active interaction row {r} on "
            "run 'run-a' at protocol_version=1 and interaction_protocol_version=None",
            id="waiting_row_without_marker",
        ),
        pytest.param(
            TaskStatus.RUNNING, 1, None, None, id="running_marker_without_row"
        ),
    ],
)
def test_interaction_marker_invariant_judges_only_a_waiting_task(
    db: Session,
    status: TaskStatus,
    marker: int | None,
    row: tuple[str, str] | None,
    expected: str | None,
) -> None:
    """``running_marker_without_row`` passes: this assertion checks nothing
    when the task is not WAITING_FOR_USER, so it cannot be used to guard a
    close path, which runs while the task is RUNNING."""
    task_id, row_id = _seed(db, marker=marker, row=row, status=status)

    if expected is None:
        assert assert_interaction_marker_invariant(db, task_id=task_id) is None
        return
    with pytest.raises(AssertionError) as excinfo:
        assert_interaction_marker_invariant(db, task_id=task_id)

    assert str(excinfo.value) == expected.format(t=task_id, r=row_id)


@pytest.mark.parametrize(
    ("marker", "row", "expected"),
    [
        pytest.param(2, ("run-a", "active"), None, id="marker_equals_the_row_version"),
        pytest.param(
            1,
            ("run-a", "active"),
            "task {t} is waiting_for_user with active interaction row {r} on "
            "run 'run-a' at protocol_version=2 and interaction_protocol_version=1",
            id="marker_differs_from_the_row_version",
        ),
        pytest.param(
            2,
            None,
            "task {t} is waiting_for_user with interaction_protocol_version=2 "
            "and 0 active interaction rows on run 'run-a'",
            id="later_version_marker_without_row",
        ),
    ],
)
def test_interaction_marker_invariant_compares_with_the_row_protocol_version(
    unchecked_db: Session,
    marker: int,
    row: tuple[str, str] | None,
    expected: str | None,
) -> None:
    """The pairing contract sets the marker to the active row's
    ``protocol_version``, not to a literal 1; with only version 1 allowed by
    today's constraints, version 2 is the only way to tell the two apart."""
    task_id, row_id = _seed(
        unchecked_db,
        marker=marker,
        row=row,
        status=TaskStatus.WAITING_FOR_USER,
        protocol_version=2,
    )

    if expected is None:
        assert (
            assert_interaction_marker_invariant(unchecked_db, task_id=task_id) is None
        )
        return
    with pytest.raises(AssertionError) as excinfo:
        assert_interaction_marker_invariant(unchecked_db, task_id=task_id)

    assert str(excinfo.value) == expected.format(t=task_id, r=row_id)
