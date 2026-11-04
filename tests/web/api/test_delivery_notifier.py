"""The host callback preserves logical turn identity and delivery outcomes."""

from unittest.mock import AsyncMock

import pytest

from xagent.web.services import task_command_execution


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_id", ["logical-turn", None])
@pytest.mark.parametrize("accepted", [True, False])
async def test_delivery_notifier_forwards_identity_and_outcome(
    monkeypatch, turn_id, accepted
):
    send = AsyncMock()
    monkeypatch.setattr(task_command_execution, "send_message_delivery", send)
    connection = AsyncMock()
    notifier = task_command_execution.make_delivery_notifier(
        connection, "client-message"
    )
    assert notifier is not None
    outcome = {
        "accepted": accepted,
        "message": "Accepted" if accepted else "Please retry",
        "error_code": None if accepted else "task_busy",
        "retry_with_new_id": not accepted,
        "rejection_outcome": None if accepted else "not_accepted",
    }
    await notifier(turn_id=turn_id, **outcome)
    send.assert_awaited_once_with(
        connection,
        client_message_id="client-message",
        turn_id=turn_id if turn_id is not None else "client-message",
        **outcome,
    )


def test_delivery_notifier_without_client_message_id():
    assert task_command_execution.make_delivery_notifier(AsyncMock(), None) is None
