"""Per-process cap on executions the command dispatcher takes from the queue.

The cap counts task ids, not handles: a task counts from the moment a dispatch
attempt reserves a slot for it until its last live execution handle ends, so
one task never occupies two slots. Executions registered outside the
dispatcher count as well, but the cap never holds them. ``BackgroundTaskManager``
owns the counted state; this module owns the limit, the interactive reserve
(the slots only interactive commands may take), the command-kind classification
and the immutable view a dispatch attempt or an idle-release check works from.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from ...config import (
    DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS,
    DISPATCHER_MAX_INFLIGHT_PER_PROCESS,
    get_dispatcher_interactive_reserve_per_process,
    get_dispatcher_max_inflight_per_process,
)

logger = logging.getLogger(__name__)

# Commands that start, resume or continue an execution in this process and
# therefore take a slot. Controls are applied to an execution that already
# exists (or to none) and never take one.
EXECUTION_OPENING_KINDS = frozenset({"start", "resume", "resume_input", "message"})
CONTROL_KINDS = frozenset({"pause", "cancel"})
# Controls a process at its dispatch cap claims ahead of the never-claimed
# commands it holds for their task (``_unfinished_earlier_command``).
# ``settle_cap_held_commands_no_commit`` defines what happens to the passed
# commands for each kind here, including a pause rejected because nothing
# of its task runs; a kind added here needs its own rules there.
CAP_OVERTAKING_CONTROL_KINDS = frozenset({"cancel", "pause"})

# What the deployment classifier decided for a command at acceptance. It is
# stored on the command row; the dispatcher reads it, never recomputes it.
DispatchClass = Literal["interactive", "batch"]
DISPATCH_CLASS_INTERACTIVE: DispatchClass = "interactive"
DISPATCH_CLASS_BATCH: DispatchClass = "batch"
DISPATCH_CLASSES = frozenset({DISPATCH_CLASS_INTERACTIVE, DISPATCH_CLASS_BATCH})


@dataclass(frozen=True, kw_only=True)
class DispatchPolicy:
    """The limit and the interactive reserve one process dispatches under.

    ``interactive_reserve`` slots, out of ``limit``, are kept for interactive
    commands, so it is always smaller than the limit.
    """

    limit: int
    interactive_reserve: int

    def __post_init__(self) -> None:
        if not (self.limit >= 1 and 0 <= self.interactive_reserve < self.limit):
            raise ValueError(
                "Dispatch policy needs limit >= 1 and 0 <= interactive_reserve "
                f"< limit, got limit={self.limit} "
                f"interactive_reserve={self.interactive_reserve}"
            )


@dataclass(frozen=True, kw_only=True)
class DispatchCapSnapshot:
    """The active task ids one dispatch attempt or idle check saw, against its policy.

    Both lines count every active task, whichever class opened it: an
    interactive command may take a slot while fewer than ``limit`` tasks are
    active, a batch command only while fewer than ``limit - interactive_reserve``
    are.
    """

    policy: DispatchPolicy
    active_ids: frozenset[int]

    @property
    def limit(self) -> int:
        return self.policy.limit

    @property
    def closed(self) -> bool:
        return len(self.active_ids) >= self.limit

    @property
    def batch_closed(self) -> bool:
        return len(self.active_ids) >= self.limit - self.policy.interactive_reserve

    def admits(self, task_id: int, *, interactive: bool) -> bool:
        """Whether a command of ``task_id`` may open an execution.

        A task that is already active needs no further slot.
        """

        if task_id in self.active_ids:
            return True
        return not (self.closed if interactive else self.batch_closed)


# The last raw environment values and the policy parsed from them; the clamp
# warning is logged once per parse. Event-loop thread only.
_policy_memo: tuple[tuple[str | None, str | None], DispatchPolicy] | None = None
# The policy, classifier name and admission declaration the process last
# reported, so the report repeats only when one of them changes.
_logged_policy: tuple[DispatchPolicy, str, bool] | None = None


def current_dispatch_policy() -> DispatchPolicy:
    """The configured policy, parsed again only when a raw value changes.

    Only the last parse is kept, with the raw (limit, reserve) pair it came
    from. A call that sees the same pair returns that policy without parsing
    or reporting anything. A call that sees a different pair parses both
    values again: an invalid value is reported and its default applies, and a
    reserve that leaves no slot for batch commands is lowered to
    ``limit - 1`` and reported. Never raises.
    """

    global _policy_memo
    raw = (
        os.getenv(DISPATCHER_MAX_INFLIGHT_PER_PROCESS),
        os.getenv(DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS),
    )
    memo = _policy_memo
    if memo is None or memo[0] != raw:
        limit = get_dispatcher_max_inflight_per_process()
        reserve = get_dispatcher_interactive_reserve_per_process()
        if reserve > limit - 1:
            logger.warning(
                "%s=%s leaves no slot for batch commands under a limit of %s; using %s",
                DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS,
                reserve,
                limit,
                limit - 1,
            )
            reserve = limit - 1
        memo = (raw, DispatchPolicy(limit=limit, interactive_reserve=reserve))
        _policy_memo = memo
    return memo[1]


def build_dispatch_cap_snapshot() -> DispatchCapSnapshot:
    """Snapshot the process's active task ids. Event-loop thread only.

    Reports the effective policy, with the installed classifier, the first time
    a process builds a snapshot and whenever the policy, the classifier or its
    admission declaration changes.
    """

    from .task_execution import background_task_manager

    policy = current_dispatch_policy()
    _report_policy(policy)
    return background_task_manager.dispatch_cap_snapshot(policy)


def _report_policy(policy: DispatchPolicy) -> None:
    global _logged_policy
    # Imported here: the admission module imports this one.
    from .task_execution_admission import describe_installed_classifier

    classifier, admission = describe_installed_classifier()
    reported = (policy, classifier, admission)
    if reported == _logged_policy:
        return
    _logged_policy = reported
    logger.info(
        "dispatch policy limit=%s interactive_reserve=%s classifier=%s "
        "admission_enabled=%s",
        policy.limit,
        policy.interactive_reserve,
        classifier,
        admission,
    )
    if admission and policy.interactive_reserve > 0:
        logger.warning(
            "interactive commands of a team are not prioritized inside its "
            "admission bucket"
        )
