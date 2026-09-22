"""Progress and final replies serialize through the same durable delivery claim."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from tests.web.services.channel_delivery_shared import accepted as accepted
from tests.web.services.channel_delivery_shared import (
    complete,
)
from tests.web.services.channel_delivery_shared import database_url as database_url
from tests.web.services.channel_delivery_shared import (
    expire_claim,
)
from tests.web.services.channel_delivery_shared import selected as selected
from xagent.web.models.database import get_session_local
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import channel_delivery as delivery
from xagent.web.services.channel_progress import DurableChannelProgress


@pytest.mark.asyncio
async def test_progress_persists_loading_destination_for_final_recovery(
    accepted, selected
):
    seen = []

    async def update(record, event):
        seen.append(dict(record.destination))
        record.destination["loading_message_id"] = "created-loading"

    progress = AsyncMock(side_effect=update)
    final = AsyncMock()
    observer = DurableChannelProgress(accepted, progress, final)
    await observer.send()
    event = Mock()
    await observer.send(event)
    assert progress.await_count == 2
    assert progress.await_args.args[1] is event
    assert seen[1]["loading_message_id"] == "created-loading"
    with get_session_local()() as db:
        row = db.get(TaskChannelDelivery, accepted)
        assert row.destination["loading_message_id"] == "created-loading"
        assert row.status == "pending" and row.claim_token is None
        assert row.failure_count == 0
    complete(accepted)
    await delivery.recover_channel_results(selected.selection.channel_id, final)
    await observer.send()
    final.assert_awaited_once()
    assert (
        final.await_args.args[0].destination["loading_message_id"] == "created-loading"
    )
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.get(TaskChannelDelivery, accepted).status == "delivered"


@pytest.mark.asyncio
@pytest.mark.parametrize("plain_final", [False, True])
async def test_final_and_progress_cannot_overtake_loading_send(accepted, plain_final):
    entered, release = asyncio.Event(), asyncio.Event()

    async def send_loading(record, event):
        entered.set()
        await release.wait()
        record.destination["loading_message_id"] = "created-loading"

    progress = AsyncMock(side_effect=send_loading)
    final = AsyncMock()
    observer = DurableChannelProgress(accepted, progress, final)
    pending = asyncio.create_task(observer.send())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        complete(accepted)
        await observer.send()
        await delivery.deliver_channel_result(accepted, final)
        final.assert_not_awaited()
        progress.assert_awaited_once()
    finally:
        release.set()
        await pending
    if plain_final:
        assert await delivery.deliver_channel_result(accepted, final)
    else:
        await observer.send()
    final.assert_awaited_once()
    assert (
        final.await_args.args[0].destination["loading_message_id"] == "created-loading"
    )
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 1


@pytest.mark.asyncio
async def test_progress_respects_failed_send_backoff(accepted):
    progress = AsyncMock(side_effect=[ConnectionError("platform unavailable"), None])
    final = AsyncMock()
    observer = DurableChannelProgress(accepted, progress, final)
    await observer.send()
    await observer.send()
    progress.assert_awaited_once()
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).failure_count == 0
    expire_claim(accepted)
    await observer.send()
    assert progress.await_count == 2
    final.assert_not_awaited()


def test_stale_progress_claim_cannot_overwrite_loading_destination(accepted):
    old, _ = delivery._claim(accepted, progress=True)
    expire_claim(accepted)
    current, _ = delivery._claim(accepted, progress=True)
    current.destination["loading_message_id"] = "current"
    delivery._settle(current)
    old.destination["loading_message_id"] = "stale"
    delivery._settle(old)
    with get_session_local()() as db:
        row = db.get(TaskChannelDelivery, accepted)
        assert row.destination["loading_message_id"] == "current"
        assert row.claim_token is None


@pytest.mark.asyncio
async def test_progress_failures_preserve_final_delivery_budget(accepted, selected):
    progress = AsyncMock(side_effect=ConnectionError("progress unavailable"))
    final = AsyncMock()
    observer = DurableChannelProgress(accepted, progress, final)
    for _ in range(12):
        expire_claim(accepted)
        await observer.send()
        await observer.send()
    assert progress.await_count == 12
    with get_session_local()() as db:
        row = db.get(TaskChannelDelivery, accepted)
        assert row.status == "pending"
        assert row.failure_count == 0
    complete(accepted)
    expire_claim(accepted)
    await delivery.recover_channel_results(selected.selection.channel_id, final)
    final.assert_awaited_once()
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "delivered"


@pytest.mark.asyncio
async def test_final_failure_via_progress_uses_final_retry_budget(accepted):
    complete(accepted)
    final = AsyncMock(side_effect=ConnectionError("final unavailable"))
    observer = DurableChannelProgress(accepted, AsyncMock(), final)
    for _ in range(10):
        expire_claim(accepted)
        await observer.send()
    assert final.await_count == 10
    with get_session_local()() as db:
        row = db.get(TaskChannelDelivery, accepted)
        assert row.status == "failed"
        assert row.failure_count == 10


@pytest.mark.asyncio
async def test_successful_progress_allows_immediate_plain_final(accepted):
    final = AsyncMock()
    observer = DurableChannelProgress(accepted, AsyncMock(), final)
    await observer.send()
    complete(accepted)
    assert await delivery.deliver_channel_result(accepted, final)
    await observer.send()
    final.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_notice", [False, True])
async def test_plain_pending_delivery_keeps_poll_delay(accepted, pending_notice):
    sender = AsyncMock()
    await delivery.deliver_channel_result(
        accepted, sender, pending_notice=pending_notice
    )
    await delivery.deliver_channel_result(
        accepted, sender, pending_notice=pending_notice
    )
    assert sender.await_count == int(pending_notice)
    with get_session_local()() as db:
        row = db.get(TaskChannelDelivery, accepted)
        assert row.available_at is not None
        assert row.claim_token is None


def test_durable_sender_has_no_raw_trace_handler_entrypoint():
    from xagent.core.agent.trace import TraceHandler

    sender = DurableChannelProgress(1, AsyncMock(), AsyncMock())
    assert not isinstance(sender, TraceHandler)
    assert not hasattr(sender, "handle_event")


@pytest.mark.asyncio
async def test_final_still_waits_for_failed_progress_backoff(accepted):
    final = AsyncMock()
    observer = DurableChannelProgress(
        accepted, AsyncMock(side_effect=ConnectionError("platform unavailable")), final
    )
    await observer.send()
    complete(accepted)
    assert not await delivery.deliver_channel_result(accepted, final)
    final.assert_not_awaited()
    expire_claim(accepted)
    assert await delivery.deliver_channel_result(accepted, final)
    final.assert_awaited_once()
