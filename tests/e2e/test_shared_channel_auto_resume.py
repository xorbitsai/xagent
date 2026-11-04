"""A shared channel reply held across an automatic resume, through real hosts.

The test process is the channel ingress: a Feishu bot whose platform network
I/O is stubbed accepts the message, and the recovery loop it runs in
production (``recover_channel_results``) delivers the reply. The worker is
killed with SIGKILL while the run waits on the model; lease recovery on the
web host pauses it and the sweeper resumes it, unless a newer channel
message replaced the resume or the reply stopped waiting first.

Backoff and polling are shortened so a resume is due a few seconds after the
lease (10s) expires: long enough for the reader to see the paused run.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from tests.e2e.shared_execution_harness import (
    SETTLE_TIMEOUT,
    auto_resume_commands,
    kill_host,
    model_calls,
    recovery_state,
    restart_hosts,
    task_state,
    wait_for,
)
from xagent.core.agent.execution_adapter import INTERRUPTED_USER_MESSAGE

pytestmark = pytest.mark.e2e

ANSWER = "Shared E2E answer"
PROCESSING = "still being processed"


@pytest.fixture(autouse=True)
def _fast_auto_resume(monkeypatch):
    # Read by the hosts the shared_app fixture spawns after this.
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS", "4")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_POLL_SECONDS", "1")
    monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_ENABLED", raising=False)
    monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_CHANNEL_WINDOW_SECONDS", raising=False)
    # This process is the ingress: its turns stop waiting quickly and reply
    # "still being processed", as after a long run.
    from xagent.web.services import shared_channel_execution

    monkeypatch.setattr(
        shared_channel_execution, "get_task_reply_wait_timeout_seconds", lambda: 0.5
    )


@pytest.fixture
async def ingress(shared_app):
    """A Feishu bot of the owner's channel; ``sent`` records every text."""
    from xagent.web.channels.feishu.bot import FeishuBotInstance
    from xagent.web.models.database import get_session_local
    from xagent.web.models.user_channel import UserChannel
    from xagent.web.services.task_event_bridge import (
        start_task_event_bridge,
        stop_task_event_bridge,
    )
    from xagent.web.services.task_events import (
        set_task_command_delivery,
        set_task_event_sink,
    )

    with get_session_local()() as db:
        channel = UserChannel(
            user_id=shared_app.user_id,
            channel_type="feishu",
            channel_name="E2E",
            is_active=True,
            config={"allowed_users": ["sender"]},
        )
        db.add(channel)
        db.commit()
        channel_id = int(channel.id)
    sent: list[str] = []

    async def send_text(_chat_id, text, **_kwargs):
        sent.append(text)
        return f"message-{len(sent)}"

    async def update_text(_chat_id, _message_id, text, **_kwargs):
        sent.append(text)
        return True

    await start_task_event_bridge()
    bot = FeishuBotInstance("test-id", "test-secret", "e2e", channel_id, "E2E")
    bot.api_client = Mock()
    bot._send_text = AsyncMock(side_effect=send_text)
    bot._update_text = AsyncMock(side_effect=update_text)
    try:
        yield SimpleNamespace(bot=bot, channel_id=channel_id, sent=sent)
    finally:
        await stop_task_event_bridge()
        set_task_command_delivery(None)
        set_task_event_sink(None)


async def _send(ingress, text: str, message_id: str) -> None:
    message = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                chat_id="chat",
                message_id=message_id,
                message_type="text",
                content=json.dumps({"text": text}),
            )
        )
    )
    await asyncio.wait_for(ingress.bot._process_messages_batch("sender", [message]), 30)


def _starts(task_id: int) -> list[dict[str, Any]]:
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_channel_delivery import TaskChannelDelivery
    from xagent.web.models.task_command import TaskExecutionCommand

    with get_session_local()() as db:
        starts = []
        for command in (
            db.query(TaskExecutionCommand)
            .filter_by(task_id=task_id, kind="start")
            .order_by(TaskExecutionCommand.id)
        ):
            delivery = db.get(TaskChannelDelivery, command.id)
            starts.append(
                {
                    "id": int(command.id),
                    "run_id": command.target_run_id,
                    "status": command.status,
                    "channel_result": (command.result or {}).get("channel_result"),
                    "delivery": delivery.status,
                    "delivered_at": delivery.delivered_at,
                }
            )
        return starts


async def _deliver_round(ingress) -> None:
    """One pass of the ingress recovery loop over every pending reply.

    A read that is still held puts the reply back for 30s; each pass makes it
    due again instead of waiting that out.
    """
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_channel_delivery import TaskChannelDelivery
    from xagent.web.services.channel_delivery import recover_channel_results

    with get_session_local()() as db:
        for row in db.query(TaskChannelDelivery).filter_by(status="pending"):
            row.available_at = None
        db.commit()
    await recover_channel_results(
        ingress.channel_id, ingress.bot._deliver_shared_result
    )


async def _deliver_until_done(ingress, task_id: int, app, timeout: float) -> int:
    """Poll replies every half second, as the reader would, until every
    reply of ``task_id`` is out. Returns how many rounds ran while the task
    rested PAUSED and left its first reply pending (held)."""
    held = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        paused = task_state(task_id)["status"] == "paused"
        await _deliver_round(ingress)
        starts = _starts(task_id)
        if paused and starts[0]["delivery"] == "pending":
            held += 1
        if all(start["delivery"] != "pending" for start in starts):
            return held
        await asyncio.sleep(0.5)
    pytest.fail(f"Replies still pending: {_starts(task_id)}\n{app.diagnostics()}")


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _count(sent: list[str], text: str) -> int:
    return sum(text in message for message in sent)


async def _gated_turn_killed(app, ingress) -> tuple[int, str]:
    """M1 waits on the model; its worker is killed and a replacement starts."""
    await _send(ingress, "e2e:gate", "m1")
    await asyncio.to_thread(
        wait_for, lambda: (app.root / "model-entered").exists(), app, "model call"
    )
    task_id = int(ingress.bot.active_tasks["sender"])
    run_id = task_state(task_id)["run_id"]
    kill_host(app, app.processes[1])
    # The resumed call may answer; the killed one never returns.
    (app.root / "model-release").touch()
    await asyncio.to_thread(app.start, "worker")
    return task_id, run_id


# --------------------------------------------------------------------------
# E9a: the channel gets the resumed run's answer
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_killed_channel_run_resumes_and_answers_its_channel(shared_app, ingress):
    app = shared_app
    task_id, run_id = await _gated_turn_killed(app, ingress)

    # The reader polls through the pause, the wait and the resumed run.
    delivering = asyncio.create_task(
        _deliver_until_done(ingress, task_id, app, timeout=SETTLE_TIMEOUT + 30)
    )
    settled = await asyncio.to_thread(app.wait_task, task_id, timeout=SETTLE_TIMEOUT)
    held = await delivering

    # The reader saw the paused run and kept waiting.
    assert held > 0
    assert settled["run_id"] == run_id
    assert ANSWER in (settled["output"] or "")
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["reason"], row["run_id"], row["total_resumes"]) == (
        "lease_expired",
        run_id,
        1,
    )
    assert events[:3] == ["interrupted", "scheduled", "auto_resumed"]
    assert len(auto_resume_commands(task_id)) == 1
    (start,) = _starts(task_id)
    assert start["run_id"] == run_id
    assert start["channel_result"]["status"] == "completed"
    assert start["delivery"] == "delivered"
    # One "still processing" notice, then the answer once; never "stopped".
    assert _count(ingress.sent, PROCESSING) == 1
    assert _count(ingress.sent, ANSWER) == 1
    assert _count(ingress.sent, INTERRUPTED_USER_MESSAGE) == 0
    # The killed run's routing call and its gated call, then the resumed
    # call from the checkpoint: nothing ran twice.
    assert len(model_calls(app)) == 3
    assert {call["role"] for call in model_calls(app)} == {"worker"}


# --------------------------------------------------------------------------
# E9b: a new channel message replaces the waiting resume
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_channel_message_while_scheduled_replaces_the_resume(shared_app, ingress):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_auto_recovery import TaskAutoRecovery

    app = shared_app
    # A wide window between recording and dispatch.
    await asyncio.to_thread(
        restart_hosts,
        app,
        "web",
        "worker",
        XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS="20",
    )
    task_id, killed_run_id = await _gated_turn_killed(app, ingress)
    await asyncio.to_thread(
        wait_for,
        lambda: (recovery_state(task_id)[0] or {}).get("state") == "scheduled",
        app,
        "the scheduled resume",
    )
    await _deliver_round(ingress)
    assert _starts(task_id)[0]["delivery"] == "pending"
    calls_before = len(model_calls(app))
    with get_session_local()() as db:
        due = db.get(TaskAutoRecovery, task_id).next_attempt_at

    await _send(ingress, "Channel followup", "m2")

    settled = await asyncio.to_thread(app.wait_task, task_id, timeout=SETTLE_TIMEOUT)
    await _deliver_until_done(ingress, task_id, app, timeout=30)
    assert settled["run_id"] != killed_run_id
    assert ANSWER in (settled["output"] or "")
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["state"], row["state_detail"]) == ("stale", "new_turn")
    assert "auto_resumed" not in events
    assert auto_resume_commands(task_id) == []
    first, second = _starts(task_id)
    assert (first["run_id"], second["run_id"]) == (killed_run_id, settled["run_id"])
    assert first["delivery"] == second["delivery"] == "delivered"
    # M1 is told it stopped, once; M2 gets its own answer, once.
    assert _count(ingress.sent, INTERRUPTED_USER_MESSAGE) == 1
    assert _count(ingress.sent, ANSWER) == 1
    # Only M2's run called the model after the kill; never the sweeper's.
    assert len(model_calls(app)) > calls_before
    # Past the dispatch time and two sweeper polls: it stood down.
    if due is None:
        await asyncio.sleep(35)
    else:
        remaining = _utc(due) + timedelta(seconds=2) - datetime.now(timezone.utc)
        await asyncio.sleep(max(remaining.total_seconds(), 0))
    assert auto_resume_commands(task_id) == []
    assert recovery_state(task_id)[0]["state"] == "stale"


# --------------------------------------------------------------------------
# E9c: past the channel window the reply stops waiting
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_channel_reply_past_the_window_is_interrupted(
    shared_app, ingress, monkeypatch
):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_auto_recovery import TaskAutoRecovery
    from xagent.web.services import task_auto_resume

    app = shared_app
    window = "15"
    # Never due within the test: only the reply's own deadline can end the
    # hold. The window is read by the hosts (recording) and by this ingress
    # (the hold's deadline), whose clock grace is cut to a second.
    await asyncio.to_thread(
        restart_hosts,
        app,
        "web",
        "worker",
        XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS="600",
        XAGENT_TASK_AUTO_RESUME_CHANNEL_WINDOW_SECONDS=window,
    )
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_CHANNEL_WINDOW_SECONDS", window)
    monkeypatch.setattr(task_auto_resume, "CHANNEL_HOLD_CLOCK_GRACE_SECONDS", 1)
    task_id, run_id = await _gated_turn_killed(app, ingress)
    await asyncio.to_thread(
        wait_for,
        lambda: (recovery_state(task_id)[0] or {}).get("state") == "scheduled",
        app,
        "the scheduled resume",
    )

    await _deliver_until_done(ingress, task_id, app, timeout=40)

    with get_session_local()() as db:
        interrupted_at = db.get(TaskAutoRecovery, task_id).interrupted_at
    (start,) = _starts(task_id)
    # Not before the window and the shortened grace (1s) ran out.
    assert _utc(start["delivered_at"]) - _utc(interrupted_at) >= timedelta(
        seconds=int(window) + 1
    )
    # Still scheduled: the sweeper never ran it, the reply gave up on it.
    assert recovery_state(task_id)[0]["state"] == "scheduled"
    assert task_state(task_id)["status"] == "paused"
    assert auto_resume_commands(task_id) == []
    assert (start["run_id"], start["delivery"]) == (run_id, "delivered")
    assert _count(ingress.sent, INTERRUPTED_USER_MESSAGE) == 1
    assert _count(ingress.sent, ANSWER) == 0


# --------------------------------------------------------------------------
# E10: /stop cancels the waiting resume
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_channel_stop_while_scheduled_cancels_the_resume(shared_app, ingress):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_auto_recovery import TaskAutoRecovery

    app = shared_app
    # A wide window between recording and dispatch.
    await asyncio.to_thread(
        restart_hosts,
        app,
        "web",
        "worker",
        XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS="20",
    )
    task_id, run_id = await _gated_turn_killed(app, ingress)
    await asyncio.to_thread(
        wait_for,
        lambda: (recovery_state(task_id)[0] or {}).get("state") == "scheduled",
        app,
        "the scheduled resume",
    )
    await _deliver_round(ingress)
    assert _starts(task_id)[0]["delivery"] == "pending"
    calls_before = len(model_calls(app))
    with get_session_local()() as db:
        row = db.get(TaskAutoRecovery, task_id)
        due, recorded_version = row.next_attempt_at, int(row.paused_state_version)

    stop = SimpleNamespace(
        event=SimpleNamespace(message=SimpleNamespace(chat_id="chat"))
    )
    await asyncio.wait_for(ingress.bot._handle_control("sender", stop, "/stop"), 30)

    # The /stop reply and M1's "stopped" reply, once each (their order is
    # best effort: the recovery loop may deliver M1 first).
    assert _count(ingress.sent, "will not resume automatically") == 1
    assert _count(ingress.sent, INTERRUPTED_USER_MESSAGE) == 1
    (start,) = _starts(task_id)
    assert (start["run_id"], start["delivery"]) == (run_id, "delivered")
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["state"], row["state_detail"]) == ("manual", "user_cancelled")
    assert "skipped_user_intent" in events and "auto_resumed" not in events
    state = task_state(task_id)
    assert (state["status"], state["run_id"]) == ("paused", run_id)
    assert state["state_version"] == recorded_version + 1
    # Past the dispatch time and two sweeper polls: nothing resumed the run.
    if due is None:
        await asyncio.sleep(35)
    else:
        remaining = _utc(due) + timedelta(seconds=2) - datetime.now(timezone.utc)
        await asyncio.sleep(max(remaining.total_seconds(), 0))
    await _deliver_round(ingress)
    assert auto_resume_commands(task_id) == []
    assert len(model_calls(app)) == calls_before
    assert _count(ingress.sent, INTERRUPTED_USER_MESSAGE) == 1
    assert task_state(task_id)["status"] == "paused"
