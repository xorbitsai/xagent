"""How a channel /stop maps the database's answer to the in-memory stop.

``BatchChannelControl._handle_stop_request`` is shared by the Feishu and
Telegram ingresses; their own tests cover only the wiring (reply text,
selection, reply-then-deliver).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.web.channels import batch_control
from xagent.web.channels.batch_control import (
    BatchChannelControl,
    StopReply,
    selected_task_id,
    stop_reply_text,
)
from xagent.web.services.channel_runtime import (
    ChannelAuthorizationError,
    ChannelConfigurationError,
    SelectedChannelTask,
)
from xagent.web.services.channel_stop import ChannelStopOutcome, ChannelStopResult
from xagent.web.services.shared_channel_execution import SharedChannelTurn


class _Control(BatchChannelControl[str]):
    control_label = "Test"

    def _active_trace_handlers(self) -> dict[str, Any]:
        return {}

    async def _deliver_shared_result(self, delivery, result) -> None:
        raise AssertionError("delivered through deliver_channel_result")


@pytest.fixture
def control(monkeypatch):
    instance = _Control()
    instance._initialize_batch_control()
    instance._accepting = True
    instance.channel_id = 7
    monkeypatch.setattr(
        batch_control, "get_shared_task_execution_enabled", lambda: True
    )
    instance.stop = AsyncMock(return_value=ChannelStopResult(ChannelStopOutcome.ACTIVE))
    monkeypatch.setattr(batch_control, "stop_channel_task", instance.stop)
    return instance


def _turn(control) -> SharedChannelTurn:
    turn = SharedChannelTurn(
        SelectedChannelTask(5, 45, False, 7, "sender", None, 0), None
    )
    turn.request_stop = Mock(return_value=True)
    control.user_active_executions["sender"] = (45, turn)
    return turn


def _pending(control, pending: str | None) -> None:
    if pending == "queued":
        control.user_message_queues["sender"] = ["queued"]
    elif pending == "preparing":
        control.user_preparing_executions.add("sender")


def _dropped(control, pending: str | None) -> None:
    assert "sender" not in control.user_message_queues
    assert control._consume_user_stop_request("sender") is (pending == "preparing")


async def _handle(control, outcome=None, *, held=None, error=None):
    control.stop.return_value = ChannelStopResult(outcome, held)
    control.stop.side_effect = error
    return await control._handle_stop_request(
        "sender", external_user_id="sender", task_id=45
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [None, "queued", "preparing"])
async def test_a_cancel_drops_pending_input_but_not_the_tracked_turn(control, pending):
    turn = _turn(control)
    _pending(control, pending)

    assert await _handle(control, ChannelStopOutcome.CANCELLED, held=11) == (
        StopReply.CANCELLED,
        11,
    )

    # The tracked turn's stop would queue a PAUSE the paused run ignores.
    turn.request_stop.assert_not_called()
    _dropped(control, pending)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pending", "reply"),
    [
        (None, StopReply.NOTHING),
        ("queued", StopReply.STOPPED),
        ("preparing", StopReply.STOPPED),
    ],
)
async def test_idle_stops_only_pending_input(control, pending, reply):
    turn = _turn(control)
    _pending(control, pending)

    assert await _handle(control, ChannelStopOutcome.IDLE) == (reply, None)

    turn.request_stop.assert_not_called()
    _dropped(control, pending)


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", ["queued", "preparing"])
async def test_a_queued_pause_also_stops_pending_input(control, pending):
    _pending(control, pending)

    assert await _handle(control, ChannelStopOutcome.PAUSE_QUEUED) == (
        StopReply.STOPPED,
        None,
    )

    _dropped(control, pending)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "error"),
    [
        pytest.param(ChannelStopOutcome.ACTIVE, None, id="active"),
        pytest.param(ChannelStopOutcome.UNAVAILABLE, None, id="unavailable"),
        pytest.param(None, RuntimeError("db down"), id="error"),
        pytest.param(None, ChannelAuthorizationError("no"), id="unauthorized"),
        pytest.param(None, ChannelConfigurationError("no"), id="unconfigured"),
    ],
)
@pytest.mark.parametrize("tracked", [False, True])
async def test_anything_else_is_the_in_memory_stop(control, outcome, error, tracked):
    turn = _turn(control) if tracked else None

    reply = await _handle(control, outcome, error=error)

    assert reply == ((StopReply.STOPPED if tracked else StopReply.NOTHING), None)
    if turn is not None:
        turn.request_stop.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "awaited"),
    [
        pytest.param(None, True, id="nothing"),
        pytest.param("queued", True, id="queued"),
        pytest.param("preparing", True, id="preparing"),
        pytest.param("tracked", False, id="tracked_turn"),
    ],
)
async def test_the_awaited_run_is_paused_only_without_a_tracked_turn(
    control, state, awaited
):
    if state == "tracked":
        _turn(control)
    else:
        _pending(control, state)

    await _handle(control, ChannelStopOutcome.ACTIVE)

    control.stop.assert_awaited_once_with(
        channel_id=7, external_user_id="sender", task_id=45, pause_awaited_run=awaited
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["not_shared", "no_task", "no_channel"])
async def test_without_a_shared_selected_task_the_database_is_skipped(
    control, monkeypatch, case
):
    if case == "not_shared":
        monkeypatch.setattr(
            batch_control, "get_shared_task_execution_enabled", lambda: False
        )
    elif case == "no_channel":
        control.channel_id = None

    reply = await control._handle_stop_request(
        "sender", external_user_id="sender", task_id=None if case == "no_task" else 45
    )

    assert reply == (StopReply.NOTHING, None)
    control.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_released_reply_is_delivered_best_effort(control, monkeypatch):
    deliver = AsyncMock(side_effect=[None, RuntimeError("platform down")])
    monkeypatch.setattr(batch_control, "deliver_channel_result", deliver)

    await control._deliver_released_reply(None)
    await control._deliver_released_reply(11)
    await control._deliver_released_reply(12)

    assert [call.args for call in deliver.await_args_list] == [
        (11, control._deliver_shared_result),
        (12, control._deliver_shared_result),
    ]


@pytest.mark.parametrize(
    ("selected", "task_id"),
    [("45", 45), (45, 45), ("-1", None), (-1, None), (None, None), ("x", None)],
)
def test_selected_task_id(selected, task_id):
    assert selected_task_id(selected) == task_id


def test_stop_reply_text_names_new_as_the_bot_does():
    assert stop_reply_text(StopReply.NOTHING, new_hint="/new") == (
        "No active run to stop."
    )
    assert stop_reply_text(StopReply.STOPPED, new_hint="use /new") == (
        "Stopped the current run. Send another message to continue here, "
        "or use /new for a fresh task."
    )
    assert "will not resume automatically" in stop_reply_text(
        StopReply.CANCELLED, new_hint="/new"
    )
