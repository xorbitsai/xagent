"""Bounded, ordered writes for a single shared-mode WebSocket."""

import asyncio
from collections.abc import Callable
from typing import Any


class TaskSocketWriter:
    def __init__(self, socket: Any, disconnect: Callable[[], None]) -> None:
        self.socket = socket
        self.disconnect = disconnect
        self.queue: asyncio.Queue[tuple[str, asyncio.Future[None] | None]] = (
            asyncio.Queue(maxsize=256)
        )
        self.bytes = 0
        self.closed = False
        self.close_task: asyncio.Task[None] | None = None
        self.task = asyncio.create_task(self._run())

    def enqueue(
        self, text: str, *, acknowledge: bool = False
    ) -> asyncio.Future[None] | None:
        if self.closed:
            raise ConnectionError("Task connection is closed")
        if self.queue.full() or self.bytes + len(text.encode()) > 2 * 1024 * 1024:
            self.stop()
            # Overflow can happen before _run starts, so its finally cannot
            # own this close. Retain the bounded task until it completes.
            self.close_task = asyncio.create_task(self._close_socket())
            raise ConnectionError(
                "Task connection is too slow; reconnect to synchronize"
            )
        future = asyncio.get_running_loop().create_future() if acknowledge else None
        self.bytes += len(text.encode())
        self.queue.put_nowait((text, future))
        return future

    async def _close_socket(self) -> None:
        try:
            await asyncio.wait_for(self.socket.close(code=1013), timeout=2)
        except Exception:
            pass

    def stop(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.task is not asyncio.current_task():
            self.task.cancel()
        while not self.queue.empty():
            _, future = self.queue.get_nowait()
            if future is not None and not future.done():
                future.set_exception(ConnectionError("Task connection is closed"))
        self.bytes = 0
        self.disconnect()

    async def _run(self) -> None:
        pending = None
        send_failed = False
        try:
            while True:
                text, pending = await self.queue.get()
                self.bytes -= len(text.encode())
                await asyncio.wait_for(self.socket.send_text(text), timeout=5)
                if pending is not None and not pending.done():
                    pending.set_result(None)
                pending = None
        except asyncio.CancelledError:
            pass
        except Exception:
            send_failed = True
        finally:
            if pending is not None and not pending.done():
                pending.set_exception(ConnectionError("Task connection send failed"))
            self.stop()
            if send_failed:
                await self._close_socket()
