"""Cross-host routing, socket ACK semantics, and bounded fanout."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from xagent.web.services import task_event_bridge as module
from xagent.web.services.task_socket_writer import TaskSocketWriter


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6379/0")
    return module.TaskEventBridge()


@pytest.mark.asyncio
async def test_public_event_dedup_and_sequence_gap(bridge):
    delivered, statuses = [], []

    async def deliver(message, task_id):
        delivered.append((task_id, message))

    async def status(kind):
        statuses.append(kind)

    bridge.deliver = deliver
    bridge.stream_status = status
    body = {
        "version": 1,
        "kind": "event",
        "host": "worker-1",
        "sequence": 1,
        "task_id": 42,
        "message": {"type": "delta", "text": "hello"},
    }
    await bridge._receive(body)
    await bridge._receive(body)
    await bridge._receive({**body, "sequence": 3})
    assert len(delivered) == 2
    assert statuses == ["stream_resync_required"]
    await bridge.close()


@pytest.mark.asyncio
async def test_reply_ack_waits_for_socket_and_same_delivery_is_deduplicated(
    bridge, monkeypatch
):
    bridge.ready.set()
    sent, acks = [], []
    entered, release = asyncio.Event(), asyncio.Event()

    async def reply(message):
        entered.set()
        await release.wait()
        sent.append(message)

    async def publish(channel, body):
        acks.append(json.loads(body))

    monkeypatch.setattr(bridge, "_publish", publish)
    token = bridge.register_origin(42, "command-1", reply)
    body = {
        "version": 1,
        "kind": "reply",
        "origin": token,
        "task_id": 42,
        "command_id": "command-1",
        "delivery_id": "delivery-1",
        "reply_host": "worker",
        "message": {"type": "private"},
    }
    first = asyncio.create_task(bridge._deliver_reply(body))
    await entered.wait()
    second = asyncio.create_task(bridge._deliver_reply(body))
    assert not acks
    release.set()
    await asyncio.gather(first, second)
    assert sent == [{"type": "private"}]
    assert len(acks) == 2
    assert all(ack["delivered"] for ack in acks)
    await bridge.close()


@pytest.mark.asyncio
async def test_wrong_origin_identity_or_disconnected_recipient_cannot_receive(
    bridge, monkeypatch
):
    bridge.ready.set()
    sent, acks = [], []

    async def reply(message):
        sent.append(message)

    async def publish(channel, body):
        acks.append(json.loads(body))

    monkeypatch.setattr(bridge, "_publish", publish)
    socket = object()
    token = bridge.register_origin(42, "command-1", reply, recipient=socket)
    body = {
        "version": 1,
        "kind": "reply",
        "origin": token,
        "task_id": 43,
        "command_id": "command-1",
        "delivery_id": "delivery-1",
        "reply_host": "worker",
        "message": {},
    }
    await bridge._deliver_reply(body)
    bridge.discard_recipient(socket)
    await bridge._deliver_reply({**body, "task_id": 42})
    assert sent == []
    assert all(not ack["delivered"] for ack in acks)
    await bridge.close()


@pytest.mark.asyncio
async def test_transport_timeout_is_unknown_not_command_failure(
    bridge, monkeypatch, caplog
):
    monkeypatch.setattr(module, "_reply_route", lambda *args: ("web", "origin"))

    async def fail(*args):
        raise TimeoutError()

    monkeypatch.setattr(bridge, "_publish", fail)
    await bridge.reply_for("command-1", 42)({"type": "private"})
    assert "delivery unknown" in caplog.text
    assert not bridge._acks
    await bridge.close()


@pytest.mark.asyncio
async def test_negative_socket_ack_preserves_connection_error(bridge, monkeypatch):
    monkeypatch.setattr(module, "_reply_route", lambda *args: ("web", "origin"))

    async def publish(channel, raw):
        body = json.loads(raw)
        await bridge._receive(
            {
                "version": 1,
                "kind": "ack",
                "delivery_id": body["delivery_id"],
                "delivered": False,
            }
        )

    monkeypatch.setattr(bridge, "_publish", publish)
    with pytest.raises(ConnectionError):
        await bridge.reply_for("command-1", 42)({"type": "private"})
    await bridge.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_socket_writer_ack_follows_send_and_slow_queue_is_bounded(started):
    class Socket:
        def __init__(self):
            self.entered = asyncio.Event()
            self.close_codes = []

        async def send_text(self, text):
            self.entered.set()
            await asyncio.Event().wait()

        async def close(self, code):
            self.close_codes.append(code)

    socket = Socket()
    disconnected = []

    def disconnect():
        disconnected.append(True)
        # ConnectionManager.disconnect also stops the writer it removes.
        writer.stop()

    writer = TaskSocketWriter(socket, disconnect)
    ack = writer.enqueue("first", acknowledge=True)
    if started:
        await asyncio.wait_for(socket.entered.wait(), timeout=1)
    assert not ack.done()
    while not writer.queue.full():
        writer.enqueue("queued")
    with pytest.raises(ConnectionError):
        writer.enqueue("overflow")
    await asyncio.wait_for(
        asyncio.gather(writer.task, return_exceptions=True), timeout=2
    )
    assert writer.close_task is not None
    await asyncio.wait_for(writer.close_task, timeout=1)
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(ack, timeout=1)
    assert writer.closed
    assert writer.queue.empty()
    assert socket.close_codes == [1013]
    assert disconnected == [True]


@pytest.mark.asyncio
async def test_snapshot_failure_does_not_abort_execution_and_advances_sequence(
    bridge, monkeypatch, caplog
):
    from xagent.web.services import task_event_state

    async def fail(*args, **kwargs):
        raise RuntimeError("snapshot unavailable")

    monkeypatch.setattr(task_event_state, "_with_current_task_control_state", fail)
    await bridge.publish({"type": "task_completed"}, 42)
    assert bridge._sequence == 1
    assert "stream requires resync" in caplog.text
    await bridge.close()


@pytest.mark.asyncio
async def test_stream_identity_comes_from_execution_attempt(bridge, monkeypatch):
    from xagent.web.services.task_lease_service import (
        TaskLease,
        bind_task_lease_context,
    )

    events = []

    async def publish(channel, body):
        events.append(json.loads(body))

    monkeypatch.setattr(bridge, "_publish", publish)
    with bind_task_lease_context(TaskLease(42, "worker", "old-run", "attempt-1")):
        await bridge.publish({"type": "final_answer_delta", "delta": "hello"}, 42)
    frame = events[0]["message"]
    assert frame["stream_run_id"] == "old-run"
    assert frame["stream_attempt_id"] == "attempt-1"
    assert frame["transport_event_id"] == f"{bridge.host_id}:1"
    await bridge.close()


@pytest.mark.asyncio
async def test_real_redis_subscription_reconnect_exposes_gap(monkeypatch):
    from xagent.web.services.task_event_bridge import TaskEventBridge

    redis_url = os.getenv("XAGENT_TEST_REDIS_URL")
    if not redis_url:
        pytest.skip("XAGENT_TEST_REDIS_URL is not set")
    monkeypatch.setenv("XAGENT_REDIS_URL", redis_url)
    monkeypatch.setenv("XAGENT_TASK_EVENT_CHANNEL_PREFIX", f"reconnect:{uuid4().hex}")
    delivered, statuses = [], []
    unavailable = asyncio.Event()
    received = asyncio.Event()

    async def deliver(message, task_id):
        delivered.append(message["text"])
        received.set()

    async def status(kind):
        statuses.append(kind)
        if kind == "stream_unavailable":
            unavailable.set()

    subscriber = TaskEventBridge(deliver=deliver, stream_status=status)
    publisher = TaskEventBridge()
    try:
        await subscriber.start()
        await publisher.start()
        await publisher.publish({"type": "delta", "text": "before"}, 1)
        await asyncio.wait_for(received.wait(), 5)
        received.clear()
        # Disconnect only this test subscriber's connections; other Redis
        # clients and server data are untouched.
        await subscriber.redis.connection_pool.disconnect()
        await asyncio.wait_for(unavailable.wait(), 5)
        await publisher.publish({"type": "delta", "text": "lost"}, 1)
        await asyncio.wait_for(subscriber.ready.wait(), 5)
        await publisher.publish({"type": "delta", "text": "after"}, 1)
        await asyncio.wait_for(received.wait(), 5)
        assert delivered == ["before", "after"]
        assert "stream_unavailable" in statuses
        assert statuses.count("stream_resync_required") >= 2
    finally:
        await subscriber.close()
        await publisher.close()


@pytest.mark.parametrize("redis_url", [None, "", "   "])
def test_bridge_requires_explicit_redis_url(monkeypatch, redis_url):
    monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
    if redis_url is not None:
        monkeypatch.setenv("XAGENT_REDIS_URL", redis_url)
    with pytest.raises(ValueError, match="XAGENT_REDIS_URL must be configured"):
        module.TaskEventBridge()


@pytest.mark.asyncio
async def test_reply_overflow_logs_are_rate_limited(bridge, monkeypatch, caplog):
    from unittest.mock import AsyncMock

    clock = iter([100.0, 100.1, 160.0])
    monkeypatch.setattr(module, "monotonic", lambda: next(clock))
    deliver = AsyncMock()
    monkeypatch.setattr(bridge, "_deliver_reply", deliver)
    blocked = asyncio.Event()
    bridge._deliveries.update(asyncio.create_task(blocked.wait()) for _ in range(4096))
    try:
        for _ in range(3):
            await bridge._receive({"version": 1, "kind": "reply"})
        assert len(bridge._deliveries) == 4096
        deliver.assert_not_called()
        warnings = [
            record for record in caplog.records if "queue is full" in record.message
        ]
        assert len(warnings) == 2
        assert all("size=4096" in record.message for record in warnings)
    finally:
        await bridge.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "exception"])
async def test_single_send_failure_closes_socket_for_reconnection(failure):
    class Socket:
        def __init__(self):
            self.close_codes = []

        async def send_text(self, text):
            if failure == "timeout":
                await asyncio.Event().wait()
            raise RuntimeError("send failed")

        async def close(self, code):
            self.close_codes.append(code)

    socket = Socket()
    disconnected = []
    writer = TaskSocketWriter(socket, lambda: disconnected.append(True))
    acknowledgement = writer.enqueue("one message", acknowledge=True)
    await asyncio.wait_for(writer.task, timeout=8)
    with pytest.raises(ConnectionError, match="send failed"):
        await asyncio.wait_for(acknowledgement, timeout=1)
    assert socket.close_codes == [1013]
    assert disconnected == [True]
    assert writer.closed
    assert writer.queue.empty()


@pytest.mark.asyncio
async def test_reply_without_route_is_observable_without_publishing(
    bridge, monkeypatch, caplog
):
    monkeypatch.setattr(module, "_reply_route", lambda *args: None)
    publish = AsyncMock()
    counter = Mock()
    monkeypatch.setattr(bridge, "_publish", publish)
    monkeypatch.setattr(module, "increment_counter", counter)
    await bridge.reply_for("command-1", 42)({"content": "private reply content"})
    publish.assert_not_awaited()
    counter.assert_called_once_with(
        "xagent.task.reply.delivery", attributes={"outcome": "no_route"}
    )
    assert "no origin route task_id=42 command_id=command-1" in caplog.text
    assert "private reply content" not in caplog.text
    assert not bridge._acks
    await bridge.close()
