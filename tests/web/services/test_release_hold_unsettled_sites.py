"""Every "left for TTL recovery" exit records unsettled evidence; no success path does.

The existing resume scenarios in ``test_execution_scope_turn_wiring`` already
drive each retention branch (pool timeouts, unhealthy heartbeat, unresolvable
checkpoint, lease loss) and the normal settle paths with the real
``execute_resume_background``. They are re-run here unchanged, measuring the
process hold's ``unsettled_total`` before and after: one record per
retained lease, none for a proven release.

A source guard is only a refactor tripwire: it checks that each instrumented
function still routes through its recording call. It is a substring check,
so it cannot tell whether every branch records; a new retention branch is
covered only by a behavioural scenario. That proof for the Web-local sites
lives beside each site's own scenarios
(``test_managed_task_lease``, ``test_execute_task_manage_task_lease``,
``test_task_orchestrator``, ``test_channel_runtime``, v1 ``test_task_reply``
and ``test_a2a_api``), which own their fixtures.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.web.api import test_execution_scope_turn_wiring as wiring
from xagent.core.agent.checkpoint import CheckpointCorruptError
from xagent.web.models.task import TaskStatus
from xagent.web.services import (
    agent_service_manager,
    channel_runtime,
    managed_task_lease,
)
from xagent.web.services import release_hold as release_hold_module
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import (
    task_execution,
    task_orchestrator,
    task_resume,
)
from xagent.web.services.task_lease_service import TaskLease

# (scenario, unsettled records it must leave)
RESUME_SCENARIOS: list[tuple[Callable[..., Awaitable[None]], int]] = [
    (wiring.test_resume_background_adopts_preacquired_lease_without_reacquiring, 0),
    (wiring.test_resume_final_usage_and_heartbeat_finish_before_lease_release, 0),
    (wiring.test_resume_background_restores_prior_status_for_preacquired_lease, 0),
    (wiring.test_resume_pool_timeout_does_not_start_secondary_db_cleanup, 1),
    (wiring.test_resume_tracker_pool_timeout_skips_result_and_lease_checkouts, 1),
    (wiring.test_resume_heartbeat_pool_timeout_skips_result_and_lease_checkouts, 1),
    (wiring.test_resume_error_delivery_pool_timeout_skips_lease_checkout, 1),
    (
        wiring.test_resume_background_checkpoint_unavailable_from_pool_timeout_keeps_lease,
        1,
    ),
    (wiring.test_resume_lease_loss_cancels_execution_without_stale_side_effects, 1),
]


@pytest.mark.parametrize(
    ("scenario", "expected"),
    RESUME_SCENARIOS,
    ids=[scenario.__name__ for scenario, _ in RESUME_SCENARIOS],
)
async def test_resume_exits_record_unsettled_only_when_the_lease_is_retained(
    scenario: Callable[..., Awaitable[None]],
    expected: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    hold = release_hold_module.release_hold
    before = hold.unsettled_total
    kwargs: dict[str, Any] = (
        {"caplog": caplog} if "caplog" in inspect.signature(scenario).parameters else {}
    )
    await scenario(**kwargs)
    assert hold.unsettled_total - before == expected
    if expected:
        assert hold.snapshot()["unsettled"][-1]["kind"] == "resume_lease"


# (function, the call through which it records): direct calls, or the one
# helper a site routes its retained outcomes through. ``execute_task_background``
# is absent on purpose: its scheduler ``_schedule_bg`` records for it.
RETENTION_SITES: list[tuple[Callable[..., Any], str]] = [
    (task_execution.execute_resume_background, "release_hold.record_unsettled("),
    (runtime.TaskCoordinator._close, "release_hold.record_unsettled("),
    (transport._dispatch_shared_task_command, "release_hold.record_unsettled("),
    # Web-local retention sites (parent acceptance R2 audit).
    (managed_task_lease.ManagedTaskLease.finalize_result, "self._confirmed("),
    (managed_task_lease.ManagedTaskLease.close, "self._confirmed("),
    (
        managed_task_lease.ManagedTaskLease._confirmed,
        "release_hold.record_unsettled(",
    ),
    (
        agent_service_manager.AgentServiceManager.execute_task,
        "release_hold.record_unsettled(",
    ),
    (task_orchestrator._schedule_bg, "release_hold.record_unsettled("),
    (task_resume.resume_task_reply, "_record_unless_restored("),
    (task_resume.resume_a2a_task, "_record_unless_restored("),
    (task_resume._record_unless_restored, "release_hold.record_unsettled("),
    (channel_runtime.prepare_channel_task, "_record_unless_compensated("),
    (channel_runtime._record_unless_compensated, "release_hold.record_unsettled("),
]


@pytest.mark.parametrize(
    ("function", "marker"),
    RETENTION_SITES,
    ids=[function.__qualname__ for function, _ in RETENTION_SITES],
)
def test_each_retention_site_still_routes_through_its_record(
    function: Callable[..., Any], marker: str
) -> None:
    assert marker in inspect.getsource(function)


@pytest.mark.parametrize(
    ("outcome", "expected"), [("restored", 0), ("not_restored", 1), ("raises", 1)]
)
async def test_prelease_restore_records_unless_confirmed(
    outcome: str, expected: int
) -> None:
    hold = release_hold_module.release_hold
    before = hold.unsettled_total

    async def restore() -> bool:
        if outcome == "raises":
            raise RuntimeError("restore write fault")
        return outcome == "restored"

    if outcome == "raises":
        with pytest.raises(RuntimeError):
            await task_resume._record_unless_restored(41, restore)
    else:
        assert await task_resume._record_unless_restored(41, restore) is (
            outcome == "restored"
        )
    assert hold.unsettled_total - before == expected


async def test_prelease_restore_under_caller_cancel_records_its_own_outcome() -> None:
    """A restore that committed while its caller was cancelled is no evidence.

    The caller still sees ``CancelledError``; the record follows the drained
    restore's own result, not the cancellation that interrupted the wait.
    """
    hold = release_hold_module.release_hold
    before = hold.unsettled_total
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def restore() -> bool:
        entered.set()
        await proceed.wait()
        return True

    caller = asyncio.create_task(task_resume._record_unless_restored(43, restore))
    await asyncio.wait_for(entered.wait(), timeout=5)
    caller.cancel()
    await asyncio.sleep(0)
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=5)
    assert hold.unsettled_total - before == 0


@pytest.mark.parametrize(
    ("settle_outcome", "expected"), [(True, 0), (False, 0), ("raises", 1)]
)
async def test_resume_failure_settlement_records_only_when_it_raises(
    settle_outcome: bool | str, expected: int
) -> None:
    """Same rule as ``_schedule_bg``: a returned settlement is no retention.

    ``settle_task_lease_isolated`` returns ``False`` also when a pre-existing
    terminal/control outcome was reconciled and released, or when a cancel
    path (external, A2A, question cancel) already released the run, so only a
    settlement that raised leaves the lease to TTL recovery.
    """
    lease = TaskLease(task_id=44, runner_id="runner-a", run_id="run-a")

    def settle(*_args: Any, **_kwargs: Any) -> bool:
        if settle_outcome == "raises":
            raise RuntimeError("settlement commit failed")
        return bool(settle_outcome)

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            self.complete_tracking = AsyncMock()
            self.stop_periodic_updates = AsyncMock()

        async def start_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        side_effect=CheckpointCorruptError("all matching rows undecodable")
    )
    hold = release_hold_module.release_hold
    before = hold.unsettled_total
    with wiring._Patches(
        [
            patch(
                "xagent.web.services.task_execution._acquire_resume_task_lease",
                side_effect=wiring._fake_acquire_with_prior_status(
                    lease, TaskStatus.WAITING_FOR_USER
                ),
            ),
            patch(
                "xagent.web.services.task_execution._settle_resumed_task_lease",
                side_effect=settle,
            ),
            patch(
                "xagent.web.services.task_execution.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.services.task_execution.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.services.task_execution.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await task_execution.execute_resume_background(
            task_id=44,
            agent_service=agent_service,
            task_owner_user_id=1,
            expected_run_id="run-a",
            resolved_execution_scope=None,
        )
    assert hold.unsettled_total - before == expected
