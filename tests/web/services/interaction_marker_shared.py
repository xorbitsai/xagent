"""Assertions for the pairing between a task's interaction protocol marker
and its run's active interaction row.

The pairing production writers keep:

    Any production writer that sets or keeps an interaction row at
    ``status='active' AND active_slot IS NOT NULL`` must, in the same
    transaction, set ``tasks.interaction_protocol_version`` for that
    ``(task_id, run_id)`` to the row's ``protocol_version``; any production
    writer that retires the last active row for a ``(task_id, run_id)`` must
    clear it to NULL in the same transaction. The only exemption is an active
    row belonging to a run that is no longer ``tasks.run_id``, and it holds
    only while the seam, the read surface and the fence share one predicate
    carrying ``run_id = tasks.run_id``.

Two assertions, kept separate and kept together in this module:

``assert_marker_residue`` ignores the task's status and can be called after
any commit. It is the one that can see a close path leave the marker behind:
the close paths run while the task is RUNNING, where a check gated on
WAITING_FOR_USER has nothing to say.

``assert_interaction_marker_invariant`` states the invariant for a task that
is waiting for the user: the marker is set if and only if exactly one active
row exists for the task's current run, and it then equals that row's
``protocol_version``. In any other status it checks nothing, so it cannot
stand in for ``assert_marker_residue``.

Both read with their own Core selects and share no predicate with production
code, so changing a production predicate cannot change what they accept.
Pass them a session opened after the commit being checked.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Session

from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_interaction import TaskInteractionRequest


def _active_rows(
    db: Session, *, task_id: int, run_id: str | None
) -> list[tuple[int, int]]:
    """``(id, protocol_version)`` of each active row for this task and run."""
    result = db.execute(
        sa.select(TaskInteractionRequest.id, TaskInteractionRequest.protocol_version)
        .where(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.run_id == run_id,
            TaskInteractionRequest.status == "active",
            TaskInteractionRequest.active_slot.isnot(None),
        )
        .order_by(TaskInteractionRequest.id)
    )
    return [(int(row_id), int(version)) for row_id, version in result.all()]


def assert_marker_residue(db: Session, *, task_id: int, run_id: str) -> None:
    """Assert that the marker and the run's active row are both present or
    both absent, whatever the task's status.

    ``run_id`` is the run the code under test worked on, captured before it
    ran. It must still be the task's run: the pairing is defined for the
    task's current run only.
    """
    task_run_id, marker = db.execute(
        sa.select(Task.run_id, Task.interaction_protocol_version).where(
            Task.id == task_id
        )
    ).one()
    if task_run_id != run_id:
        raise AssertionError(
            f"task {task_id} is on run {task_run_id!r}, not run {run_id!r}"
        )
    rows = _active_rows(db, task_id=task_id, run_id=run_id)
    if not rows:
        if marker is not None:
            raise AssertionError(
                f"task {task_id} run {run_id!r}: no active interaction row, "
                f"interaction_protocol_version={marker!r}"
            )
        return
    # uq_task_interaction_active_slot allows one active row per task.
    row_id, protocol_version = rows[0]
    if marker != protocol_version:
        raise AssertionError(
            f"task {task_id} run {run_id!r}: active interaction row {row_id} "
            f"has protocol_version={protocol_version!r}, "
            f"interaction_protocol_version={marker!r}"
        )


def assert_interaction_marker_invariant(db: Session, *, task_id: int) -> None:
    """Assert, for a task waiting for the user, that the marker is set
    exactly when one active row exists for the task's current run, and that
    it equals that row's ``protocol_version``.

    Checks nothing when the task is not WAITING_FOR_USER.
    """
    status, run_id, marker = db.execute(
        sa.select(Task.status, Task.run_id, Task.interaction_protocol_version).where(
            Task.id == task_id
        )
    ).one()
    if status != TaskStatus.WAITING_FOR_USER:
        return
    rows = _active_rows(db, task_id=task_id, run_id=run_id)
    if marker is not None and len(rows) != 1:
        raise AssertionError(
            f"task {task_id} is waiting_for_user with "
            f"interaction_protocol_version={marker!r} and {len(rows)} active "
            f"interaction rows on run {run_id!r}"
        )
    if len(rows) == 1 and marker != rows[0][1]:
        row_id, protocol_version = rows[0]
        raise AssertionError(
            f"task {task_id} is waiting_for_user with active interaction row "
            f"{row_id} on run {run_id!r} at protocol_version={protocol_version!r} "
            f"and interaction_protocol_version={marker!r}"
        )
