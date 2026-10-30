"""A channel /stop applied to the sender's selected task in the database.

``stop_channel_task_sync`` cancels an automatic resume the paused run waits
for (and makes its held reply "stopped", deliverable now), says IDLE when
nothing of the task runs, and pauses the run the chat awaits when the
ingress no longer tracks it. Runs on SQLite and, when
``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL; the lock case is
PostgreSQL only.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import sqlalchemy as sa

from tests.web.services.channel_delivery_shared import (
    database_url as database_url_fixture,
)
from tests.web.services.channel_delivery_shared import selected as selected_fixture
from tests.web.services.test_channel_reply_hold import (
    _hand_off,
    _postgresql_only,
    _record,
    _second_turn,
    _wait_for_a_lock_waiter,
)
from tests.web.services.test_channel_reply_hold import channel as channel_fixture
from tests.web.services.test_channel_reply_hold import (
    short_lock_timeout as short_lock_timeout_fixture,
)
from xagent.web.models.database import get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import channel_delivery, channel_stop
from xagent.web.services import shared_channel_execution as shared
from xagent.web.services.channel_runtime import (
    ChannelAuthorizationError,
    ChannelConfigurationError,
)
from xagent.web.services.channel_stop import ChannelStopOutcome, stop_channel_task_sync
from xagent.web.services.execution_result_projection import interrupted_channel_result
from xagent.web.services.task_auto_resume import (
    AutoResumeCancel,
    auto_resume_command_id,
    cancel_pending_auto_resume_no_commit,
)
from xagent.web.services.task_orchestrator import TaskTurnPayload

database_url = database_url_fixture
selected = selected_fixture
channel = channel_fixture
short_lock_timeout = short_lock_timeout_fixture

RACED = AutoResumeCancel(cancelled=False, raced=True)


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_CHANNEL_WINDOW_SECONDS", raising=False)
    monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_ENABLED", raising=False)
    notify = Mock()
    monkeypatch.setattr(channel_stop, "notify_task_command_dispatcher", notify)
    return notify


def _stop(ch, *, pause_awaited_run: bool = True, **kwargs: Any):
    return stop_channel_task_sync(
        **{
            "channel_id": ch.channel_id,
            "external_user_id": "sender",
            "task_id": ch.task_id,
            "pause_awaited_run": pause_awaited_run,
            **kwargs,
        }
    )


def _snapshot(ch) -> tuple[Any, ...]:
    with get_session_local()() as db:
        task = db.get(Task, ch.task_id)
        row = db.get(TaskAutoRecovery, ch.task_id)
        start = db.get(TaskExecutionCommand, ch.command_id)
        return (
            int(task.state_version),
            None if row is None else (row.state, row.state_detail),
            (start.result or {}).get("channel_result"),
            sorted(
                db.scalars(
                    sa.select(TaskExecutionCommand.command_id).where(
                        TaskExecutionCommand.task_id == ch.task_id
                    )
                )
            ),
        )


def _pauses(ch) -> list[TaskExecutionCommand]:
    with get_session_local()() as db:
        rows = list(
            db.scalars(
                sa.select(TaskExecutionCommand).where(
                    TaskExecutionCommand.task_id == ch.task_id,
                    TaskExecutionCommand.kind == "pause",
                )
            )
        )
        db.expunge_all()
        return rows


def _set_task(ch, **values: Any) -> None:
    with get_session_local()() as db:
        task = db.get(Task, ch.task_id)
        for name, value in values.items():
            setattr(task, name, value)
        db.commit()


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# S1 / S2. A cancel: the held reply becomes "stopped", sent once, now
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "recorded",
    [
        pytest.param({}, id="scheduled"),
        pytest.param(
            {"state": "dispatched", "command_status": "pending"}, id="dispatched"
        ),
    ],
)
async def test_a_stop_cancels_the_resume_and_releases_the_reply_once(
    channel, monkeypatch, recorded
):
    _record(channel, **recorded)
    sender = AsyncMock()
    # The hold: a delivery attempt leaves the reply pending for 30s.
    assert not await channel_delivery.deliver_channel_result(channel.command_id, sender)
    publish = AsyncMock()
    monkeypatch.setattr(channel_stop, "publish_recovery_notices", publish)

    result = await channel_stop.stop_channel_task(
        channel_id=channel.channel_id,
        external_user_id="sender",
        task_id=channel.task_id,
        pause_awaited_run=False,
    )

    assert result.outcome is ChannelStopOutcome.CANCELLED
    assert result.held_command_db_id == channel.command_id
    (notice,) = result.notices
    assert (notice.run_id, notice.state_version) == (
        channel.run_id,
        channel.version + 1,
    )
    assert notice.auto_resume["stop_reason"] == "cancelled"
    publish.assert_awaited_once_with(result.notices)
    version, row, channel_result, _ = _snapshot(channel)
    assert (version, row) == (channel.version + 1, ("manual", "user_cancelled"))
    assert channel_result == interrupted_channel_result()
    with get_session_local()() as db:
        delivery = db.get(TaskChannelDelivery, channel.command_id)
        assert delivery.status == "pending"
        assert _aware(delivery.available_at) <= datetime.now(timezone.utc)

    assert await channel_delivery.deliver_channel_result(channel.command_id, sender)
    assert not await channel_delivery.deliver_channel_result(channel.command_id, sender)
    sender.assert_awaited_once()
    assert sender.await_args.args[1] == interrupted_channel_result()
    assert not _pauses(channel)


def test_the_stopped_reply_wins_over_a_resume_by_hand(channel):
    """A person resumes the run on the web after the stop; its answer is not
    the channel's (the resumed run stores its answer only where none is)."""
    _record(channel)
    assert _stop(channel).outcome is ChannelStopOutcome.CANCELLED

    answer = {"success": True, "status": "completed", "output": "the answer"}
    with get_session_local()() as db:
        assert not shared.stamp_channel_result_no_commit(
            db, task_id=channel.task_id, run_id=channel.run_id, channel_result=answer
        )
        db.commit()
    assert _snapshot(channel)[2] == interrupted_channel_result()


def test_a_reply_another_sender_claimed_keeps_its_lease(channel):
    _record(channel)
    later = datetime.now(timezone.utc) + timedelta(seconds=60)
    with get_session_local()() as db:
        delivery = db.get(TaskChannelDelivery, channel.command_id)
        delivery.claim_token, delivery.available_at = "theirs", later
        db.commit()

    assert _stop(channel).held_command_db_id == channel.command_id

    with get_session_local()() as db:
        delivery = db.get(TaskChannelDelivery, channel.command_id)
        assert _aware(delivery.available_at) >= later - timedelta(seconds=1)


# --------------------------------------------------------------------------
# S3. Authorization and ownership
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "error"),
    [
        pytest.param("sender", ChannelAuthorizationError, id="not_allowed"),
        pytest.param("inactive", ChannelConfigurationError, id="inactive"),
    ],
)
def test_an_unauthorized_stop_raises_and_writes_nothing(channel, change, error):
    _record(channel)
    before = _snapshot(channel)
    sender = "sender"
    if change == "sender":
        sender = "intruder"
    else:
        with get_session_local()() as db:
            db.get(UserChannel, channel.channel_id).is_active = False
            db.commit()

    with pytest.raises(error):
        _stop(channel, external_user_id=sender)

    assert _snapshot(channel) == before


@pytest.mark.parametrize("change", ["other_channel", "other_owner", "missing"])
def test_a_task_outside_the_channel_and_owner_is_unavailable(channel, change):
    _record(channel)
    with get_session_local()() as db:
        task = db.get(Task, channel.task_id)
        if change == "other_channel":
            other = UserChannel(
                user_id=task.user_id,
                channel_type="feishu",
                channel_name="other",
                config={},
                is_active=True,
            )
            db.add(other)
            db.flush()
            task.channel_id = other.id
        elif change == "other_owner":
            user = User(username="someone", password_hash="unused")
            db.add(user)
            db.flush()
            task.user_id = user.id
        db.commit()
    before = _snapshot(channel)

    task_id = channel.task_id + 1000 if change == "missing" else channel.task_id
    result = _stop(channel, task_id=task_id)

    assert result.outcome is ChannelStopOutcome.UNAVAILABLE
    assert _snapshot(channel) == before


# --------------------------------------------------------------------------
# S4. Nothing to stop
# --------------------------------------------------------------------------


def _auto_resume_left_queued(ch) -> None:
    """The row gave up, but its RESUME is still queued: the guard skips it."""
    with get_session_local()() as db:
        db.add(
            TaskExecutionCommand(
                task_id=ch.task_id,
                actor_user_id=db.get(Task, ch.task_id).user_id,
                command_id=auto_resume_command_id(ch.version, 1, ch.run_id),
                kind="resume",
                payload={"type": "resume_task"},
                target_run_id=ch.run_id,
                status="pending",
            )
        )
        db.commit()


@pytest.mark.parametrize(
    "case",
    [
        "no_row",
        "manual",
        "user_cancelled",
        "expired",
        "exhausted",
        "superseded",
        "dispatch_failed",
        "ineligible",
        "stale",
        "disabled",
        "completed",
        "failed",
        "auto_resume_left_queued",
    ],
)
def test_a_task_with_nothing_to_stop_is_idle(channel, case):
    if case in ("completed", "failed"):
        _set_task(channel, status=TaskStatus(case), control_state=case)
    elif case == "user_cancelled":
        _record(channel, state="manual")
        with get_session_local()() as db:
            db.get(TaskAutoRecovery, channel.task_id).state_detail = "user_cancelled"
            db.commit()
    elif case == "auto_resume_left_queued":
        _record(channel, state="expired")
        _auto_resume_left_queued(channel)
    elif case != "no_row":
        _record(channel, state=case)
    before = _snapshot(channel)

    result = _stop(channel)

    assert result.outcome is ChannelStopOutcome.IDLE
    assert result.held_command_db_id is None and not result.notices
    assert _snapshot(channel) == before


@pytest.mark.parametrize(
    ("kind", "status"),
    [
        ("start", None),
        ("resume", None),
        ("resume_input", None),
        ("message", None),
        ("start", "completed"),
        ("start", "failed"),
    ],
)
def test_an_unfinished_run_command_is_active(channel, kind, status):
    _record(channel, state="expired")
    if status is not None:
        _set_task(channel, status=TaskStatus(status), control_state=status)
    with get_session_local()() as db:
        db.add(
            TaskExecutionCommand(
                task_id=channel.task_id,
                actor_user_id=db.get(Task, channel.task_id).user_id,
                command_id=f"{kind}-queued",
                kind=kind,
                payload={},
                target_run_id=channel.run_id,
                status="pending",
            )
        )
        db.commit()
    before = _snapshot(channel)

    assert _stop(channel).outcome is ChannelStopOutcome.ACTIVE
    assert _snapshot(channel) == before


def test_a_cancel_with_the_reply_already_sent_releases_nothing(channel):
    _record(channel)
    with get_session_local()() as db:
        db.get(TaskChannelDelivery, channel.command_id).status = "delivered"
        db.commit()

    result = _stop(channel)

    assert result.outcome is ChannelStopOutcome.CANCELLED
    assert result.held_command_db_id is None


def test_a_second_stop_finds_nothing_to_stop(channel):
    _record(channel, state="dispatched", command_status="pending")
    assert _stop(channel).outcome is ChannelStopOutcome.CANCELLED
    after = _snapshot(channel)

    # The cancelled RESUME is still queued until its guard skips it.
    assert _stop(channel).outcome is ChannelStopOutcome.IDLE
    assert _snapshot(channel) == after


def test_only_a_committed_cancel_drops_the_task_cache(channel, monkeypatch):
    invalidate = Mock()
    monkeypatch.setattr(channel_stop, "invalidate_task_cache_best_effort", invalidate)
    _record(channel, state="dispatched", command_status="pending")

    assert _stop(channel).outcome is ChannelStopOutcome.CANCELLED
    invalidate.assert_called_once_with(channel.task_id)

    # Nothing written: nothing to drop.
    assert _stop(channel).outcome is ChannelStopOutcome.IDLE
    invalidate.assert_called_once()


def test_a_cancel_that_keeps_racing_drops_nothing(channel, monkeypatch):
    invalidate = Mock()
    monkeypatch.setattr(channel_stop, "invalidate_task_cache_best_effort", invalidate)
    monkeypatch.setattr(
        channel_stop, "cancel_pending_auto_resume_no_commit", lambda *a, **k: RACED
    )
    _record(channel)

    assert _stop(channel).outcome is ChannelStopOutcome.ACTIVE
    invalidate.assert_not_called()


# --------------------------------------------------------------------------
# S5. No tracked turn: pause the run this chat waits for
# --------------------------------------------------------------------------


def test_a_running_awaited_run_is_paused(channel, _environment):
    _set_task(channel, status=TaskStatus.RUNNING, control_state="running")

    result = _stop(channel)

    assert result.outcome is ChannelStopOutcome.PAUSE_QUEUED
    (pause,) = _pauses(channel)
    assert pause.command_id.startswith("stop:")
    assert (pause.target_run_id, pause.status) == (channel.run_id, "pending")
    _environment.assert_called_once()


@pytest.mark.parametrize(
    "case", ["tracked_turn", "web_run", "reply_sent", "other_channel"]
)
def test_a_run_the_chat_does_not_await_is_left_to_the_ingress(
    channel, _environment, case
):
    _set_task(channel, status=TaskStatus.RUNNING, control_state="running")
    if case == "web_run":
        # A later run started on the web: this chat waits for no reply of it.
        _set_task(channel, run_id="web-run")
    elif case == "reply_sent":
        with get_session_local()() as db:
            db.get(TaskChannelDelivery, channel.command_id).status = "delivered"
            db.commit()
    elif case == "other_channel":
        # The reply is awaited, but by another channel of the owner.
        with get_session_local()() as db:
            other = UserChannel(
                user_id=db.get(Task, channel.task_id).user_id,
                channel_type="feishu",
                channel_name="other",
                config={},
                is_active=True,
            )
            db.add(other)
            db.flush()
            db.get(TaskChannelDelivery, channel.command_id).channel_id = other.id
            db.commit()

    result = _stop(channel, pause_awaited_run=case != "tracked_turn")

    assert result.outcome is ChannelStopOutcome.ACTIVE
    assert not _pauses(channel)
    _environment.assert_not_called()


@pytest.mark.parametrize(
    ("manual_resume", "outcome"),
    [
        pytest.param("pending", ChannelStopOutcome.PAUSE_QUEUED, id="can_complete"),
        pytest.param("completed", ChannelStopOutcome.ACTIVE, id="stranded"),
    ],
)
def test_a_hand_off_is_paused_only_while_it_can_complete(
    channel, manual_resume, outcome
):
    _hand_off(channel, row=None, manual_resume=manual_resume)

    assert _stop(channel).outcome is outcome
    assert len(_pauses(channel)) == int(outcome is ChannelStopOutcome.PAUSE_QUEUED)


@pytest.mark.parametrize(
    ("earlier", "staged"),
    [
        pytest.param("pause_requested", False, id="pause_requested"),
        pytest.param("pending", False, id="pause_pending"),
        pytest.param("processing", False, id="pause_processing"),
        pytest.param("completed", True, id="pause_done"),
        pytest.param("other_run", True, id="pause_of_another_run"),
    ],
)
def test_a_pause_already_queued_is_not_queued_again(
    channel, _environment, earlier, staged
):
    control_state = "pause_requested" if earlier == "pause_requested" else "running"
    _set_task(channel, status=TaskStatus.RUNNING, control_state=control_state)
    if earlier != "pause_requested":
        with get_session_local()() as db:
            db.add(
                TaskExecutionCommand(
                    task_id=channel.task_id,
                    actor_user_id=db.get(Task, channel.task_id).user_id,
                    command_id="stop:earlier",
                    kind="pause",
                    payload={},
                    target_run_id="older-run"
                    if earlier == "other_run"
                    else channel.run_id,
                    status="pending" if earlier == "other_run" else earlier,
                )
            )
            db.commit()
    before = len(_pauses(channel))

    assert _stop(channel).outcome is ChannelStopOutcome.PAUSE_QUEUED
    assert len(_pauses(channel)) == before + int(staged)
    assert _environment.called is staged


# --------------------------------------------------------------------------
# S6. A lost compare-and-swap is read again, at most three times
# --------------------------------------------------------------------------


def test_a_raced_cancel_is_rolled_back_and_retried(channel, monkeypatch):
    _record(channel)
    calls: list[int] = []

    def raced_once(db, task, *, at):
        calls.append(1)
        if len(calls) == 1:
            # Even a write staged before the race must not be kept.
            db.execute(
                sa.update(Task)
                .where(Task.id == task.id)
                .values(state_version=Task.state_version + 7)
            )
            return RACED
        return cancel_pending_auto_resume_no_commit(db, task, at=at)

    monkeypatch.setattr(
        channel_stop, "cancel_pending_auto_resume_no_commit", raced_once
    )

    assert _stop(channel).outcome is ChannelStopOutcome.CANCELLED
    assert len(calls) == 2
    assert _snapshot(channel)[0] == channel.version + 1


def test_a_cancel_that_keeps_racing_falls_back_to_the_ingress(channel, monkeypatch):
    _record(channel)
    before = _snapshot(channel)
    raced = Mock()

    def always_raced(db, task, *, at):
        raced()
        db.execute(
            sa.update(Task)
            .where(Task.id == task.id)
            .values(state_version=Task.state_version + 7)
        )
        return RACED

    monkeypatch.setattr(
        channel_stop, "cancel_pending_auto_resume_no_commit", always_raced
    )

    result = _stop(channel)

    assert result.outcome is ChannelStopOutcome.ACTIVE
    assert raced.call_count == 3
    assert _snapshot(channel) == before


# --------------------------------------------------------------------------
# P4. PostgreSQL: a stop waits for an acceptance that superseded the resume
# --------------------------------------------------------------------------


def test_postgresql_stop_waits_for_a_superseding_acceptance(
    channel, short_lock_timeout
):
    _postgresql_only()
    _record(channel)
    outcome: dict[str, Any] = {}

    def stop() -> None:
        try:
            outcome["result"] = _stop(channel)
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    accepting = get_session_local()()
    thread = threading.Thread(target=stop)
    try:
        shared.accept_channel_turn_no_commit(
            accepting, _second_turn(channel), TaskTurnPayload("again"), "ingress"
        )
        thread.start()
        with get_session_local()() as monitor:
            _wait_for_a_lock_waiter(monitor)
        accepting.commit()
    finally:
        accepting.close()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"].outcome is ChannelStopOutcome.ACTIVE
    with get_session_local()() as db:
        row = db.get(TaskAutoRecovery, channel.task_id)
        assert (row.state, row.state_detail) == ("stale", "new_turn")
        assert int(db.get(Task, channel.task_id).state_version) == channel.version + 1
    assert not _pauses(channel)
