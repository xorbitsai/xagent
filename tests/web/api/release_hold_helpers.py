"""Shared assertions for trigger entries under the planned-restart hold."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.trigger import TriggerRun, TriggerRunStatus
from xagent.web.services import release_hold as release_hold_module
from xagent.web.services.release_hold import ReleaseHold
from xagent.web.services.triggers import dispatch_pending_trigger_runs

from .conftest import _direct_db_session


def install_private_hold(monkeypatch: pytest.MonkeyPatch) -> ReleaseHold:
    """Replace the process hold for one test so its state cannot leak."""
    fresh = ReleaseHold()
    monkeypatch.setattr(release_hold_module, "release_hold", fresh)
    return fresh


def _run_states(run_id: int) -> tuple[str, str]:
    db = _direct_db_session()
    try:
        run = db.query(TriggerRun).filter(TriggerRun.id == run_id).one()
        task = db.query(Task).filter(Task.id == run.task_id).one()
        return str(run.status), task.status.value
    finally:
        db.close()


async def _dispatch_pending() -> int:
    db = _direct_db_session()
    try:
        return await dispatch_pending_trigger_runs(db)
    finally:
        db.close()


def assert_deferred_then_started(
    scheduled: MagicMock, hold: ReleaseHold, run_id: int
) -> None:
    """A held trigger run is committed PENDING, then started once on reopen."""
    # Deferred: committed and PENDING, not started, never FAILED.
    assert scheduled.call_count == 0
    assert _run_states(run_id) == (
        TriggerRunStatus.PENDING.value,
        TaskStatus.PENDING.value,
    )
    # The existing dispatcher owner starts it once admission reopens.
    hold.reopen()
    assert asyncio.run(_dispatch_pending()) == 1
    assert scheduled.call_count == 1
    assert asyncio.run(_dispatch_pending()) == 0
