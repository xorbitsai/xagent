"""Feishu progress projection keeps its display semantics with durable sending."""

from unittest.mock import AsyncMock, Mock

import pytest

from xagent.web.channels.feishu.trace_handler import FeishuTraceHandler


@pytest.mark.asyncio
async def test_shared_progress_preserves_updates_final_and_cancel():
    send = AsyncMock()
    client = Mock()
    handler = FeishuTraceHandler(1, client, "chat", send_update=send)
    await handler._update_message("answer")
    await handler._update_message("answer")
    await handler._update_message("answer", final=True)
    assert [call.args[0] for call in send.await_args_list] == [
        "answer ✍️",
        "answer",
    ]
    handler.cancel()
    await handler._update_message("late", final=True)
    assert send.await_count == 2
    client.im.v1.message.patch.assert_not_called()


@pytest.mark.asyncio
async def test_shared_progress_bounds_display_and_retries_failed_callback():
    send = AsyncMock(side_effect=[ConnectionError("unavailable"), None])
    handler = FeishuTraceHandler(1, Mock(), "chat", send_update=send)
    with pytest.raises(ConnectionError):
        await handler._update_message("x" * 5000)
    await handler._update_message("x" * 5000)
    assert send.await_count == 2
    assert send.await_args.args == ("x" * 4000,)


@pytest.mark.asyncio
async def test_local_progress_still_patches_existing_message():
    client = Mock()
    client.im.v1.message.patch.return_value.success.return_value = True
    handler = FeishuTraceHandler(1, client, "chat", "loading")
    await handler._update_message("answer")
    client.im.v1.message.patch.assert_called_once()
    assert client.im.v1.message.patch.call_args.args[0].message_id == "loading"
