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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from ...config import (
    DISPATCHER_FAIR_SHARE,
    DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS,
    DISPATCHER_MAX_INFLIGHT_PER_PROCESS,
    get_dispatcher_fair_share_enabled,
    get_dispatcher_interactive_reserve_per_process,
    get_dispatcher_max_inflight_per_process,
)

logger = logging.getLogger(__name__)

# Commands that start, resume or continue an execution in this process and
# therefore take a slot. Controls are applied to an execution that already
# exists (or to none) and never take one.
EXECUTION_OPENING_KINDS = frozenset({"start", "resume", "resume_input", "message"})
CONTROL_KINDS = frozenset({"pause", "cancel"})
# Controls a process at its dispatch cap or its batch line claims ahead of the
# never-claimed commands it holds for their task
# (``_unfinished_earlier_command``).
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

    ``occupancy`` is how many slots each tenant key holds in this process, filled
    by ``BackgroundTaskManager.dispatch_cap_snapshot``; a snapshot built anywhere
    else, such as the one that only observes the cap state, leaves it empty.
    """

    policy: DispatchPolicy
    active_ids: frozenset[int]
    # A planned restart holds new execution (``release_hold``): the cap then
    # reads as closed whatever the count, so exactly the existing closed-cap
    # rules apply -- controls of any task and commands of tasks that already
    # hold a slot pass, everything that would open a new execution waits.
    held: bool = False
    occupancy: Mapping[str, int] = field(
        default_factory=dict, hash=False, compare=False
    )

    @property
    def limit(self) -> int:
        return self.policy.limit

    @property
    def closed(self) -> bool:
        return self.held or len(self.active_ids) >= self.limit

    @property
    def batch_closed(self) -> bool:
        return self.held or (
            len(self.active_ids) >= self.limit - self.policy.interactive_reserve
        )

    def admits(self, task_id: int, *, interactive: bool) -> bool:
        """Whether a command of ``task_id`` may open an execution.

        A task that is already active needs no further slot.
        """

        if task_id in self.active_ids:
            return True
        return not (self.closed if interactive else self.batch_closed)

    @property
    def in_reserve_band(self) -> bool:
        """Whether the process sits between its batch line and its limit.

        Only there can the tenant cap hold a tenant back: below the batch line
        every command may take a slot, at the limit none may, and with no
        interactive reserve the band is empty.
        """
        return self.batch_closed and not self.closed

    def capped_tenant(self, queued: Mapping[str, int] | None) -> str | None:
        """The tenant that may not take a new slot now, or None when nobody is held back.

        ``queued`` maps each queued tenant key to the id of its oldest queued
        command. Between the batch line and the limit, when at least two
        tenants are queued, the tenant holding the most slots in this process
        is held back. Every tenant that holds slots is compared, queued or not;
        when the top holders have nothing queued nobody is held back, and on a
        tie only the tied tenant with the earliest queued command is.

        None when ``queued`` is None or lists fewer than two tenants, when the
        process is not between the lines (``in_reserve_band``) and when no
        queued tenant is a top holder. The empty string is a valid tenant key
        and a valid result, so callers compare the result with ``is None``.
        """
        if queued is None or len(queued) < 2 or not self.in_reserve_band:
            return None
        top = max(self.occupancy.values(), default=0)
        if top <= 0:
            return None
        holders = [
            key for key, held in self.occupancy.items() if held == top and key in queued
        ]
        if not holders:
            return None
        return min(holders, key=lambda key: queued[key])


class TenantListingRequired(Exception):
    """A slot was asked for without the queued tenants, inside the reserve band.

    Raised by ``BackgroundTaskManager.reserve_dispatch_slot`` only when the
    caller passed ``require_listing=True``; it leaves no state behind. The
    dispatch attempt that gets it selects again, with the tenants listed.
    """


# The last raw environment values and the policy parsed from them; the clamp
# warning is logged once per parse. Event-loop thread only.
_policy_memo: tuple[tuple[str | None, str | None], DispatchPolicy] | None = None
# The policy, classifier name, admission declaration and raw fair-share value the
# process last reported, so the report repeats only when one of them changes.
_logged_policy: tuple[DispatchPolicy, str, bool, str | None] | None = None


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

    Reports the effective policy, with the installed classifier and the fair-share
    switch, the first time a process builds a snapshot and whenever the policy,
    the classifier, its admission declaration or the switch's raw value changes.
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
    # The raw switch value, not its parse: parsing an invalid value warns, and
    # this runs for every snapshot.
    reported = (policy, classifier, admission, os.getenv(DISPATCHER_FAIR_SHARE))
    if reported == _logged_policy:
        return
    _logged_policy = reported
    logger.info(
        "dispatch policy limit=%s interactive_reserve=%s classifier=%s "
        "admission_enabled=%s fair_share=%s",
        policy.limit,
        policy.interactive_reserve,
        classifier,
        admission,
        get_dispatcher_fair_share_enabled(),
    )
    if admission and policy.interactive_reserve > 0:
        logger.warning(
            "interactive commands of a team are not prioritized inside its "
            "admission bucket"
        )
