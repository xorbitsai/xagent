"""Result holder for what a task settlement committed."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ...core.agent.interruption import InterruptionReason
from ..models.task_auto_recovery import TaskAutoRecovery, TaskAutoRecoveryState


@dataclass(frozen=True)
class RecordedInterruption:
    """What recording an interruption wrote to the task's recovery row.

    A detached snapshot, safe to read after the recording transaction
    commits. ``attempt`` is the automatic resume the row is scheduled for
    (``total_resumes + 1``), or for any other state the resumes already made.
    """

    task_id: int
    run_id: str
    reason: str
    state: str
    state_detail: str | None
    paused_state_version: int
    next_attempt_at: datetime | None
    attempt: int

    @classmethod
    def from_row(cls, row: TaskAutoRecovery) -> RecordedInterruption:
        total = int(row.total_resumes or 0)
        return cls(
            task_id=int(row.task_id),
            run_id=str(row.run_id),
            reason=str(row.reason),
            state=str(row.state),
            state_detail=row.state_detail,
            paused_state_version=int(row.paused_state_version),
            next_attempt_at=row.next_attempt_at,
            attempt=(
                total + 1
                if row.state == TaskAutoRecoveryState.SCHEDULED.value
                else total
            ),
        )


@dataclass
class SettlementReport:
    """What a settlement committed, for the caller that announces it.

    Two uses. A caller-supplied report is passed to a settle/finalize function
    (``settle_task_lease_isolated``, ``_settle_resumed_task_lease``,
    ``finalize_managed_task_lease_result``) and filled only after that call's
    commit succeeds. Callers create one report per settlement call: a fill
    writes only the fields its outcome sets. The function's bool return is the commit signal: an empty
    report does not by itself mean "not committed". A report is also returned
    by ``_finalize_resumed_task`` and ``_TaskExecutionFinalization``, which
    carry non-pause outcomes too.

    ``control_state`` is the committed V2 control identity, filled where a
    caller publishes it; it may be empty after a commit (a non-V2 row, or the
    managed path for a non-pause outcome). ``paused_for`` is the recorded
    reason of a committed interruption pause, and is set only then.
    """

    control_state: dict[str, Any] = field(default_factory=dict)
    paused_for: InterruptionReason | None = None

    @property
    def paused(self) -> bool:
        return self.paused_for is not None
