"""Connection-owned serialization for outbound WebSocket text frames."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar


class WebSocketTextSink(Protocol):
    """Minimal WebSocket shape required by the ordered writer."""

    async def send_text(self, data: str) -> None: ...


class WebSocketWriterRetiredError(ConnectionError):
    """A queued write belongs to a connection generation that was retired."""


_SinkT = TypeVar("_SinkT", bound=WebSocketTextSink)


@dataclass
class _ConnectionTextWriter:
    websocket: WebSocketTextSink
    loop: asyncio.AbstractEventLoop | None = None
    tail: asyncio.Future[None] | None = field(default=None)
    retired_future: asyncio.Future[None] | None = field(default=None)
    active_write: asyncio.Task[None] | None = field(default=None)
    retired: bool = False

    def _bind_to_running_loop(
        self,
    ) -> tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]:
        running_loop = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = running_loop
            self.tail = running_loop.create_future()
            self.tail.set_result(None)
            self.retired_future = running_loop.create_future()
            if self.retired:
                self.retired_future.set_result(None)
        elif running_loop is not self.loop:
            raise RuntimeError("WebSocket writer cannot move between event loops")
        retired_future = self.retired_future
        if self.tail is None or retired_future is None:
            raise RuntimeError("WebSocket writer synchronization was not initialized")
        return running_loop, retired_future

    async def send_text(
        self,
        data: str,
        *,
        is_active: Callable[[], bool] | None = None,
    ) -> bool:
        if self.retired:
            raise WebSocketWriterRetiredError(
                "WebSocket connection generation is retired"
            )
        running_loop, retired_future = self._bind_to_running_loop()
        predecessor = self.tail
        if predecessor is None:
            raise RuntimeError("WebSocket writer queue was not initialized")
        turn = running_loop.create_future()
        self.tail = turn
        predecessor_released = False
        try:
            done, _pending = await asyncio.wait(
                {predecessor, retired_future},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if retired_future in done or self.retired:
                raise WebSocketWriterRetiredError(
                    "WebSocket connection generation is retired"
                )
            predecessor_released = True
            if is_active is not None and not is_active():
                return False
            write = asyncio.create_task(self.websocket.send_text(data))
            self.active_write = write
            try:
                await write
            except asyncio.CancelledError:
                if self.retired and write.cancelled():
                    raise WebSocketWriterRetiredError(
                        "WebSocket connection generation is retired"
                    ) from None
                raise
            finally:
                if self.active_write is write:
                    self.active_write = None
            return True
        finally:
            if predecessor_released or self.retired:
                if not turn.done():
                    turn.set_result(None)
            else:
                predecessor.add_done_callback(
                    lambda _future: turn.set_result(None) if not turn.done() else None
                )

    def retire(self) -> None:
        self.retired = True
        if self.retired_future is not None and not self.retired_future.done():
            self.retired_future.set_result(None)
        if self.active_write is not None and not self.active_write.done():
            self.active_write.cancel()


_WRITER_ATTRIBUTE = "_xagent_connection_text_writer"


def _writer_owner(websocket: WebSocketTextSink) -> Any:
    """Use Starlette connection state, with the socket itself for test doubles."""

    state = getattr(websocket, "state", None)
    return state if state is not None else websocket


def _current_writer(
    websocket: WebSocketTextSink,
) -> _ConnectionTextWriter | None:
    writer = getattr(_writer_owner(websocket), _WRITER_ATTRIBUTE, None)
    return writer if isinstance(writer, _ConnectionTextWriter) else None


def activate_websocket_writer(websocket: WebSocketTextSink) -> None:
    """Create a writer for a newly accepted or reactivated connection."""

    writer = _current_writer(websocket)
    if writer is not None:
        if writer.retired:
            raise WebSocketWriterRetiredError(
                "A retired WebSocket object cannot be reactivated"
            )
        return
    setattr(
        _writer_owner(websocket),
        _WRITER_ATTRIBUTE,
        _ConnectionTextWriter(websocket=websocket),
    )


def retire_websocket_writer(websocket: WebSocketTextSink) -> None:
    """Reject queued writes that belong to a disconnected connection."""

    writer = _current_writer(websocket)
    if writer is not None:
        writer.retire()


async def send_websocket_text(
    websocket: WebSocketTextSink,
    data: str,
    *,
    is_active: Callable[[], bool] | None = None,
) -> bool:
    """Send one text frame after all earlier writes for this connection."""

    writer = _current_writer(websocket)
    if writer is None:
        activate_websocket_writer(websocket)
        writer = _current_writer(websocket)
    if writer is None:
        raise RuntimeError("Failed to initialize WebSocket writer")
    return await writer.send_text(data, is_active=is_active)


async def fanout_websocket_text(
    websockets: list[_SinkT],
    data: str,
    *,
    is_active: Callable[[_SinkT], bool],
) -> list[tuple[_SinkT, Exception]]:
    """Write independently to an authorized connection snapshot.

    Operational failures are returned only after every connection has made
    progress. Caller cancellation still propagates directly.
    """

    async def send(websocket: _SinkT) -> Exception | None:
        if not is_active(websocket):
            return None
        try:
            await send_websocket_text(
                websocket,
                data,
                is_active=lambda: is_active(websocket),
            )
        except Exception as error:  # noqa: BLE001
            return error
        return None

    results = await asyncio.gather(*(send(websocket) for websocket in websockets))
    return [
        (websocket, result)
        for websocket, result in zip(websockets, results, strict=True)
        if result is not None
    ]
