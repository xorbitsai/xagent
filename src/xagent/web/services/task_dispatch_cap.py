"""Per-process cap on executions the command dispatcher takes from the queue.

The cap counts task ids, not handles: a task counts from the moment a dispatch
attempt reserves a slot for it until its last live execution handle ends, so
one task never occupies two slots. Executions registered outside the
dispatcher count as well, but the cap never holds them. ``BackgroundTaskManager``
owns the counted state; this module owns the limit, the command-kind
classification and the immutable view a dispatch attempt or an idle-release
check works from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ...config import (
    DISPATCHER_MAX_INFLIGHT_PER_PROCESS,
    get_dispatcher_max_inflight_per_process,
)

# Commands that start, resume or continue an execution in this process and
# therefore take a slot. Controls are applied to an execution that already
# exists (or to none) and never take one.
EXECUTION_OPENING_KINDS = frozenset({"start", "resume", "resume_input", "message"})
CONTROL_KINDS = frozenset({"pause", "cancel"})
# Controls a process at its dispatch cap claims ahead of the never-claimed
# commands it holds for their task (``_unfinished_earlier_command``). Only
# cancel: ``settle_cap_held_commands_no_commit`` defines what happens to the
# passed commands for a cancel alone, so adding a kind here also requires
# defining that kind's settlement, including when the control is rejected.
CAP_OVERTAKING_CONTROL_KINDS = frozenset({"cancel"})


@dataclass(frozen=True)
class DispatchCapSnapshot:
    """The active task ids one dispatch attempt or idle check saw, against the limit."""

    limit: int
    active_ids: frozenset[int]

    @property
    def closed(self) -> bool:
        return len(self.active_ids) >= self.limit

    def admits(self, task_id: int) -> bool:
        """Whether a command of ``task_id`` may open an execution."""

        return task_id in self.active_ids or not self.closed


_limit_memo: tuple[str | None, int] | None = None


def current_dispatch_limit() -> int:
    """The configured limit, parsed once per distinct raw environment value.

    An invalid value is reported once and then the default applies. A changed
    value is picked up on the next call. Never raises.
    """

    global _limit_memo
    raw = os.getenv(DISPATCHER_MAX_INFLIGHT_PER_PROCESS)
    memo = _limit_memo
    if memo is None or memo[0] != raw:
        memo = (raw, get_dispatcher_max_inflight_per_process())
        _limit_memo = memo
    return memo[1]


def build_dispatch_cap_snapshot() -> DispatchCapSnapshot:
    """Snapshot the process's active task ids. Event-loop thread only."""

    from .task_execution import background_task_manager

    return background_task_manager.dispatch_cap_snapshot(current_dispatch_limit())
