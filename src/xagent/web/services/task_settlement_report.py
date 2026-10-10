"""Result holder for what a task settlement committed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.agent.interruption import InterruptionReason


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
