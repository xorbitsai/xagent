"""A pause that passes the batch commands a process holds on its batch line is
handled and settled as it is at the cap: the pause scenarios of
``test_dispatch_cap_pause_overtake`` run again with the process between
limit-R and limit."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

import logging
from typing import Any

import pytest

from tests.web.services.test_dispatch_cap import (  # noqa: F401
    CapHost,
    cap_host,
    set_limit,
)
from tests.web.services.test_dispatch_cap_cancel_overtake import host  # noqa: F401
from tests.web.services.test_dispatch_cap_cancel_overtake import (  # noqa: F401
    SETTLE_LOGGER,
    pause_host,
)
from tests.web.services.test_dispatch_cap_pause_overtake import (
    PAUSE_SCENARIOS,
    accept_start,
    assert_settle_log,
    stage_pause,
)
from tests.web.services.test_dispatch_reserve import dispatch, set_reserve
from xagent.web.models.task import TaskStatus
from xagent.web.services import task_command_transport as transport
from xagent.web.services.task_dispatch_cap import DispatchPolicy

# The scenarios whose expectation does not depend on how the process got to
# its cap, and that the batch line must reproduce.
BATCH_LINE_SCENARIOS = [
    "S1-new-task-start",
    "S2-appended-start",
    "S4-waiting-reply",
    "S7-two-held-rows",
    "S8-shared-channel-stop",
    "S9-ticketed-start",
    "S11-other-users-pause",
    "S12c-precheck-reply-fails",
    "S16b-stale-web-pause-other-start",
    "S17-ticketed-start-stopped-by-channel",
]


@pytest.fixture
def batch_line_pause_host(
    pause_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> CapHost:
    """N=2, R=1: the one running task every scenario adds puts the process on
    its batch line. Tasks the scenarios create come from a batch source, so the
    commands staged for them are classified batch."""

    set_limit(monkeypatch, 2)
    set_reserve(monkeypatch, 1)
    real_new_task = pause_host.new_task

    def new_task(**fields: Any) -> int:
        fields.setdefault("source", "sdk")
        return real_new_task(**fields)

    monkeypatch.setattr(pause_host, "new_task", new_task)
    return pause_host


@pytest.mark.parametrize("scenario", BATCH_LINE_SCENARIOS)
async def test_pause_settles_held_batch_commands_below_cap(
    batch_line_pause_host: CapHost, caplog: pytest.LogCaptureFixture, scenario: str
) -> None:
    host = batch_line_pause_host
    with caplog.at_level(logging.INFO, logger=SETTLE_LOGGER):
        await PAUSE_SCENARIOS[scenario](host)

    cap = host.manager.dispatch_cap_snapshot(
        DispatchPolicy(limit=2, interactive_reserve=1)
    )
    assert cap.batch_closed and not cap.closed

    assert_settle_log(
        caplog,
        scenario,
        silent=(
            "S9-ticketed-start",
            "S11-other-users-pause",
            "S12c-precheck-reply-fails",
            "S16b-stale-web-pause-other-start",
        ),
    )


async def test_a_pause_does_not_pass_an_interactive_start_on_the_batch_line(
    batch_line_pause_host: CapHost,
) -> None:
    """The interactive variant of S1: the process may start the command, so it
    starts it first and the pause applies to the running task afterwards."""

    host = batch_line_pause_host
    host.live(host.new_task())
    task_id = host.new_task(
        status=TaskStatus.PENDING, run_id=None, state_version=0, source="internal"
    )
    start_id, _, _ = accept_start(host, task_id)
    pause_id = stage_pause(host, task_id)
    assert host.row(start_id).dispatch_class == "interactive"
    cap = host.manager.dispatch_cap_snapshot(
        DispatchPolicy(limit=2, interactive_reserve=1)
    )
    assert transport._find_command_candidate(pause_id, (), cap) is None

    assert await dispatch(host) is True
    assert host.claimed[-1] == (task_id, transport.TaskCommandKind.START)
    assert host.row(start_id).attempt_count == 1

    assert await dispatch(host) is True
    assert host.claimed[-1] == (task_id, transport.TaskCommandKind.PAUSE)
    assert host.row(pause_id).status == "completed"

    # The start ran; nothing settled it as a command the pause passed.
    row = host.row(start_id)
    assert row.status != "failed"
    assert "cancelled_before_admission" not in str(row.result)
