"""Process-local admission control for durable upload registration.

The gate lives at the asynchronous HTTP boundary so requests wait without
occupying the shared ``asyncio.to_thread`` executor. A lease covers the full
cancellation-safe registration call: if the client disconnects, capacity is
released only after the worker has settled and can no longer write storage.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock

from ...config import (
    get_file_upload_max_concurrency,
    get_file_upload_queue_timeout_seconds,
)


class UploadStorageCapacityError(TimeoutError):
    """Raised when an upload cannot enter durable registration in time."""


class UploadStorageGate:
    """Bound active durable upload registrations within one backend process."""

    def __init__(
        self,
        *,
        max_concurrency: int,
        queue_timeout_seconds: float,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be positive")
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._queue_timeout_seconds = queue_timeout_seconds
        self._active = 0
        self._waiting = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lifecycle_lock = Lock()

    @property
    def active(self) -> int:
        """Return the number of leases currently performing registration."""

        return self._active

    def _prepare_for_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind to one live loop, resetting only after its work is drained."""

        with self._lifecycle_lock:
            if self._loop is loop:
                return
            if self._loop is None:
                self._loop = loop
                return
            if not self._loop.is_closed():
                raise RuntimeError("Upload storage gate does not support concurrent event loops")
            if self._active or self._waiting:
                raise RuntimeError(
                    "Upload storage gate cannot leave a closed event loop until "
                    "all leases and waiters are drained"
                )

            # Contention binds asyncio.Semaphore to its event loop. Recreate
            # only that loop-bound primitive while retaining the one process
            # gate and its configured capacity.
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
            self._loop = loop

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[None]:
        """Acquire registration capacity and release it on every scope exit."""

        self._prepare_for_loop(asyncio.get_running_loop())
        with self._lifecycle_lock:
            self._waiting += 1
        try:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self._queue_timeout_seconds,
                )
            except TimeoutError as exc:
                raise UploadStorageCapacityError(
                    "Timed out waiting for durable upload capacity"
                ) from exc
        finally:
            with self._lifecycle_lock:
                self._waiting -= 1

        with self._lifecycle_lock:
            self._active += 1
        try:
            yield
        finally:
            with self._lifecycle_lock:
                self._active -= 1
            self._semaphore.release()


_gate: UploadStorageGate | None = None
_gate_lock = Lock()


def get_upload_storage_gate() -> UploadStorageGate:
    """Return the sole process gate, prepared for the current event loop.

    Sequential event-loop lifecycles reuse the same gate after the prior loop
    is closed and all of its leases and waiters have drained. Access from a
    second live loop is unsupported and raises ``RuntimeError`` rather than
    creating another independently usable capacity pool.
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = UploadStorageGate(
                max_concurrency=get_file_upload_max_concurrency(),
                queue_timeout_seconds=get_file_upload_queue_timeout_seconds(),
            )
        if loop is not None:
            _gate._prepare_for_loop(loop)
        return _gate
