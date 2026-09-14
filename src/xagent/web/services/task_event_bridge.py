"""Redis task fanout and origin-bound replies; database commands remain the queue."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import RedisError

from ...config import get_redis_url, get_task_event_channel_prefix
from ...core.runtime_performance import increment_counter
from ..models.database import get_session_local
from ..models.task_command import TaskExecutionCommand
from .db_runtime import run_db_io_cancellation_safe
from .task_events import CommandReply

logger = logging.getLogger(__name__)

EventDelivery = Callable[[dict[str, Any], int], Awaitable[None]]
StreamStatus = Callable[[str], Awaitable[None]]


@dataclass
class _Origin:
    task_id: int
    command_id: str
    reply: CommandReply
    recipient: object | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # A command can reply several times. Retain only its recent identities;
    # entries disappear with the command or socket, and the registry is bounded.
    delivered: OrderedDict[str, bool] = field(default_factory=OrderedDict)


class TaskEventBridge:
    """One process boot identity; combined hosts use the same Redis path."""

    def __init__(
        self,
        *,
        deliver: EventDelivery | None = None,
        stream_status: StreamStatus | None = None,
    ) -> None:
        self.host_id = uuid4().hex
        self.prefix = get_task_event_channel_prefix()
        redis_url = get_redis_url()
        if not redis_url:
            raise ValueError("XAGENT_REDIS_URL must be configured for TaskEventBridge")
        self.redis = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            health_check_interval=10,
        )
        self.deliver = deliver
        self.stream_status = stream_status
        self.ready = asyncio.Event()
        self._reader: asyncio.Task[None] | None = None
        self._stopping = False
        self._publish_lock = asyncio.Lock()
        self._sequence = 0
        self._seen: OrderedDict[str, int] = OrderedDict()
        self._origins: OrderedDict[str, _Origin] = OrderedDict()
        self._acks: dict[str, asyncio.Future[bool]] = {}
        self._deliveries: set[asyncio.Task[None]] = set()
        self._last_reply_overflow_warning: float | None = None

    @property
    def private_channel(self) -> str:
        return f"{self.prefix}:host:{self.host_id}"

    async def start(self) -> None:
        self._reader = asyncio.create_task(self._listen())
        try:
            await asyncio.wait_for(self.ready.wait(), timeout=10)
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        self._stopping = True
        self.ready.clear()
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
        for pending in self._deliveries:
            pending.cancel()
        await asyncio.gather(*self._deliveries, return_exceptions=True)
        for future in self._acks.values():
            if not future.done():
                future.cancel()
        self._origins.clear()
        await self.redis.aclose()

    def require_ready(self) -> None:
        if not self.ready.is_set():
            raise ConnectionError(
                "Task event transport is unavailable; retry after reconnection"
            )

    def register_origin(
        self,
        task_id: int,
        command_id: str,
        reply: CommandReply,
        *,
        recipient: object | None = None,
    ) -> str:
        self.require_ready()
        token = uuid4().hex
        self._origins[token] = _Origin(task_id, command_id, reply, recipient)
        while len(self._origins) > 4096:
            self._origins.popitem(last=False)
        return token

    def discard_origin(self, token: str) -> None:
        self._origins.pop(token, None)

    def discard_recipient(self, recipient: object) -> None:
        for token in [
            token
            for token, origin in self._origins.items()
            if origin.recipient is recipient
        ]:
            self.discard_origin(token)

    async def _publish(self, channel: str, body: str) -> None:
        await asyncio.wait_for(self.redis.publish(channel, body), timeout=3)

    async def _status(self, kind: str) -> None:
        if self.stream_status is not None:
            await self.stream_status(kind)

    async def publish(self, message: dict[str, Any], task_id: int) -> None:
        from .task_event_state import _with_current_task_control_state

        # Enrich at the producer, before transport. Receivers never assign a
        # newer run to an old frame. Sequence numbers also advance on failure.
        async with self._publish_lock:
            self._sequence += 1
            try:
                message = await _with_current_task_control_state(
                    message, fallback_task_id=task_id
                )
                from .task_lease_service import current_task_lease

                lease = current_task_lease()
                message = dict(message)
                message["transport_event_id"] = f"{self.host_id}:{self._sequence}"
                if lease is not None and lease.task_id == task_id:
                    message["stream_run_id"] = lease.run_id
                    message["stream_attempt_id"] = lease.attempt_id
                envelope = {
                    "version": 1,
                    "kind": "event",
                    "host": self.host_id,
                    "sequence": self._sequence,
                    "task_id": task_id,
                    "message": message,
                }
                await self._publish(f"{self.prefix}:events", json.dumps(envelope))
            except Exception:
                # Publication is observational. Even snapshot/serialization
                # failures must not abort an accepted execution. The sequence
                # gap and persistent-state checks trigger client recovery.
                logger.warning(
                    "Task event publication failed task_id=%s; stream requires resync",
                    task_id,
                    exc_info=True,
                )

    async def _listen(self) -> None:
        channels = [self.private_channel]
        if self.deliver is not None:
            channels.append(f"{self.prefix}:events")
        while not self._stopping:
            try:
                async with self.redis.pubsub() as pubsub:
                    await pubsub.subscribe(*channels)
                    subscribed: set[str] = set()
                    async for frame in pubsub.listen():
                        if frame["type"] == "subscribe":
                            subscribed.add(frame["channel"])
                            if len(subscribed) == len(channels):
                                self.ready.set()
                                await self._status("stream_resync_required")
                            continue
                        if frame["type"] != "message":
                            continue
                        try:
                            body = json.loads(frame["data"])
                            await self._receive(body)
                        except (ValueError, TypeError, KeyError):
                            logger.warning("Invalid task event bridge envelope")
            except asyncio.CancelledError:
                raise
            except (TimeoutError, RedisError, OSError):
                logger.warning("Task event subscription interrupted; reconnecting")
            finally:
                self.ready.clear()
            await self._status("stream_unavailable")
            await asyncio.sleep(1)

    async def _receive(self, body: dict[str, Any]) -> None:
        if body.get("version") != 1:
            raise ValueError("Unsupported envelope")
        kind = body["kind"]
        if kind == "event" and self.deliver is not None:
            host, sequence = body["host"], body["sequence"]
            previous = self._seen.get(host)
            if previous is not None and sequence <= previous:
                return
            if previous is not None and sequence != previous + 1:
                await self._status("stream_resync_required")
            self._seen[host] = sequence
            self._seen.move_to_end(host)
            while len(self._seen) > 4096:
                self._seen.popitem(last=False)
            await self.deliver(body["message"], body["task_id"])
        elif kind == "ack":
            future = self._acks.get(body["delivery_id"])
            if future is not None and not future.done():
                future.set_result(body["delivered"] is True)
        elif kind == "reply":
            # Never wait for a socket in the Redis reader. Its bounded writer
            # queue preserves socket ordering while the reader accepts ACKs.
            if len(self._deliveries) >= 4096:
                now = monotonic()
                # Bound log volume while the receiver remains overloaded.
                if (
                    self._last_reply_overflow_warning is None
                    or now - self._last_reply_overflow_warning >= 60
                ):
                    logger.warning(
                        "Task reply delivery dropped: pending deliveries queue is full (size=%s)",
                        len(self._deliveries),
                    )
                    self._last_reply_overflow_warning = now
                return  # sender records delivery unknown, never re-execution
            pending = asyncio.create_task(self._deliver_reply(body))
            self._deliveries.add(pending)
            pending.add_done_callback(self._deliveries.discard)
        elif kind == "cleanup":
            origin = self._origins.get(body["origin"])
            if origin is not None and (origin.task_id, origin.command_id) == (
                body["task_id"],
                body["command_id"],
            ):
                self.discard_origin(body["origin"])

    async def _deliver_reply(self, body: dict[str, Any]) -> None:
        origin = self._origins.get(body["origin"])
        delivered = False
        if origin is not None and (origin.task_id, origin.command_id) == (
            body["task_id"],
            body["command_id"],
        ):
            async with origin.lock:
                delivery_id = body["delivery_id"]
                if delivery_id in origin.delivered:
                    delivered = origin.delivered[delivery_id]
                else:
                    try:
                        await origin.reply(body["message"])
                        delivered = True
                    except ConnectionError:
                        pass
                    origin.delivered[delivery_id] = delivered
                    while len(origin.delivered) > 128:
                        origin.delivered.popitem(last=False)
        try:
            await self._publish(
                f"{self.prefix}:host:{body['reply_host']}",
                json.dumps(
                    {
                        "version": 1,
                        "kind": "ack",
                        "delivery_id": body["delivery_id"],
                        "delivered": delivered,
                    }
                ),
            )
        except (TimeoutError, RedisError, OSError):
            logger.warning("Task reply ACK could not be published")

    def reply_for(self, command_id: str, task_id: int) -> CommandReply:
        async def reply(message: dict[str, Any]) -> None:
            route = await run_db_io_cancellation_safe(
                lambda: _reply_route(task_id, command_id)
            )
            if route is None:
                logger.warning(
                    "Task reply has no origin route task_id=%s command_id=%s",
                    task_id,
                    command_id,
                )
                increment_counter(
                    "xagent.task.reply.delivery", attributes={"outcome": "no_route"}
                )
                return
            host_id, origin = route
            delivery_id = uuid4().hex
            future = asyncio.get_running_loop().create_future()
            self._acks[delivery_id] = future
            try:
                await self._publish(
                    f"{self.prefix}:host:{host_id}",
                    json.dumps(
                        {
                            "version": 1,
                            "kind": "reply",
                            "task_id": task_id,
                            "command_id": command_id,
                            "origin": origin,
                            "delivery_id": delivery_id,
                            "reply_host": self.host_id,
                            "message": message,
                        }
                    ),
                )
                delivered = await asyncio.wait_for(future, timeout=5)
            except (TimeoutError, RedisError, OSError):
                # A send may have completed and its ACK been lost. This is a
                # transport outcome, never a reason to retry Agent execution.
                logger.warning(
                    "Task reply delivery unknown task_id=%s command_id=%s delivery_id=%s",
                    task_id,
                    command_id,
                    delivery_id,
                )
                return
            finally:
                self._acks.pop(delivery_id, None)
            if not delivered:
                raise ConnectionError("Original task command connection is unavailable")

        return reply

    def discard_command(self, command_id: str, task_id: int) -> None:
        async def cleanup() -> None:
            route = await run_db_io_cancellation_safe(
                lambda: _reply_route(task_id, command_id)
            )
            if route is None:
                return
            host, origin = route
            try:
                await self._publish(
                    f"{self.prefix}:host:{host}",
                    json.dumps(
                        {
                            "version": 1,
                            "kind": "cleanup",
                            "task_id": task_id,
                            "command_id": command_id,
                            "origin": origin,
                        }
                    ),
                )
            except (TimeoutError, RedisError, OSError):
                logger.warning("Task origin cleanup deferred to registry eviction")

        pending = asyncio.create_task(cleanup())
        self._deliveries.add(pending)
        pending.add_done_callback(self._deliveries.discard)


def _reply_route(task_id: int, command_id: str) -> tuple[str, str] | None:
    with get_session_local()() as db:
        row = (
            db.query(
                TaskExecutionCommand.reply_host_id, TaskExecutionCommand.reply_origin
            )
            .filter(
                TaskExecutionCommand.task_id == task_id,
                TaskExecutionCommand.command_id == command_id,
            )
            .first()
        )
        if row is None or row.reply_host_id is None or row.reply_origin is None:
            return None
        return row.reply_host_id, row.reply_origin


_bridge: TaskEventBridge | None = None


def get_task_event_bridge() -> TaskEventBridge:
    if _bridge is None:
        raise ConnectionError("Task event bridge has not started")
    return _bridge


async def start_task_event_bridge(
    *, deliver: EventDelivery | None = None, stream_status: StreamStatus | None = None
) -> TaskEventBridge:
    global _bridge
    from .task_events import set_task_command_delivery, set_task_event_sink

    bridge = TaskEventBridge(deliver=deliver, stream_status=stream_status)
    await bridge.start()
    _bridge = bridge
    set_task_event_sink(bridge.publish)
    set_task_command_delivery(bridge)
    return bridge


async def stop_task_event_bridge() -> None:
    global _bridge
    if _bridge is not None:
        await _bridge.close()
        _bridge = None
