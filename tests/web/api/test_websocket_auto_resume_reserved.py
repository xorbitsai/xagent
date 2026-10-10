"""Clients cannot forge the auto-resume sweeper's RESUME commands.

``resume_task`` trusts a RESUME that carries ``auto_resume`` under an
``auto-resume:`` command id as the sweeper's own, so the WebSocket ingress
refuses both the field and the id namespace before anything is stored.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.services import task_command_execution as command_execution_service
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services.task_command_transport import TaskCommandKind

_KINDS = (TaskCommandKind.MESSAGE, TaskCommandKind.PAUSE, TaskCommandKind.RESUME)


@pytest.fixture
def never_enqueued(monkeypatch):
    enqueue = MagicMock(side_effect=AssertionError("must not be enqueued"))
    monkeypatch.setattr(websocket_api, "_enqueue_websocket_task_command_sync", enqueue)
    return enqueue


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _KINDS)
async def test_client_frames_cannot_carry_auto_resume(never_enqueued, kind):
    with pytest.raises(
        task_execution_service.ClientVisibleValidationError, match="auto_resume"
    ):
        await websocket_api._enqueue_websocket_task_command(
            task_id=1,
            message_data={
                "user": SimpleNamespace(id=7, is_admin=False),
                "type": "resume_task",
                "auto_resume": {"expected_run_id": "run", "expected_state_version": 3},
            },
            kind=kind,
            command_id="client-1",
        )
    never_enqueued.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("command_id", ["auto-resume:3:1:run", " auto-resume:x"])
async def test_client_command_ids_cannot_use_the_reserved_prefix(
    never_enqueued, kind, command_id
):
    with pytest.raises(
        task_execution_service.ClientVisibleValidationError, match="auto-resume"
    ):
        await websocket_api._enqueue_websocket_task_command(
            task_id=1,
            message_data={"user": SimpleNamespace(id=7, is_admin=False)},
            kind=kind,
            command_id=command_id,
        )
    never_enqueued.assert_not_called()


@pytest.mark.asyncio
async def test_resume_frame_with_a_reserved_id_is_answered_with_an_error(
    never_enqueued, monkeypatch
):
    sent = AsyncMock()
    monkeypatch.setattr(websocket_api.manager, "send_personal_message", sent)

    await websocket_api.handle_resume_task(
        MagicMock(),
        1,
        {
            "user": SimpleNamespace(id=7, is_admin=False),
            "type": "resume_task",
            "command_id": "auto-resume:3:1:run",
        },
    )

    never_enqueued.assert_not_called()
    frame = sent.await_args.args[0]
    assert frame["type"] == "error"
    assert frame["error_code"] == "invalid_message"


def test_reserved_prefix_is_recognized_after_normalization():
    # ``_client_message_id`` strips whitespace before the id is used, so the
    # check must see the stripped id too.
    assert command_execution_service._client_message_id(" auto-resume:x ") == (
        "auto-resume:x"
    )
