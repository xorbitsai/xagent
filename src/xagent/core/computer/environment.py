from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from .schema import ComputerActionBatch, ComputerObservation


class ComputerEnvironmentError(RuntimeError):
    """Base error for a provider-neutral computer environment."""


class ComputerEnvironmentClosedError(ComputerEnvironmentError):
    """Raised when an operation targets a closed environment."""


class ComputerEnvironmentProtocolError(ComputerEnvironmentError):
    """Raised when an adapter violates the environment contract."""


class ComputerFrameMismatchError(ComputerEnvironmentError):
    """Raised when actions target an observation that is no longer current."""


class ComputerSessionMismatchError(ComputerEnvironmentError):
    """Raised when actions target another computer session."""


class ComputerTargetNotFoundError(ComputerEnvironmentError):
    """Raised when an action references an element outside the current frame."""


class ComputerEnvironment(ABC):
    """Provider-neutral environment with serialized stale-frame validation.

    Every state-changing call is checked against the last recorded frame while
    holding the same lock used for adapter execution. If execution fails after
    partially changing the external UI, the frame is invalidated so a caller
    must observe again instead of retrying against stale coordinates.
    """

    def __init__(self, session_id: str) -> None:
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise ValueError("session_id must not be empty")
        self.session_id = normalized_session_id
        self._current_observation: ComputerObservation | None = None
        self._seen_frame_ids: set[str] = set()
        self._operation_lock = asyncio.Lock()
        self._closed = False
        self._close_complete = False

    @property
    def current_observation(self) -> ComputerObservation | None:
        if self._current_observation is None:
            return None
        return self._current_observation.model_copy(deep=True)

    @property
    def closed(self) -> bool:
        return self._closed

    async def invalidate_observation(self) -> None:
        """Forget a frame whose target may have changed out of band."""

        async with self._operation_lock:
            self._invalidate_observation()

    def _invalidate_observation(self) -> None:
        self._current_observation = None

    def _record_observation(self, observation: ComputerObservation) -> None:
        self._ensure_open()
        if not isinstance(observation, ComputerObservation):
            raise ComputerEnvironmentProtocolError(
                "environment adapters must return ComputerObservation"
            )
        if observation.session_id != self.session_id:
            raise ComputerSessionMismatchError(
                f"observation session {observation.session_id!r} does not match "
                f"environment session {self.session_id!r}"
            )
        if observation.frame_id in self._seen_frame_ids:
            raise ComputerEnvironmentProtocolError(
                "environment adapters must return fresh frame_id values and "
                "must not reuse them within a session"
            )
        self._seen_frame_ids.add(observation.frame_id)
        self._current_observation = observation.model_copy(deep=True)

    def validate_action_batch(self, batch: ComputerActionBatch) -> None:
        """Validate a batch against the current frame without acquiring the lock.

        ``execute`` calls this synchronous helper while holding the operation
        lock. Direct callers that combine validation with a later operation
        must provide their own synchronization.
        """

        self._ensure_open()
        if batch.session_id != self.session_id:
            raise ComputerSessionMismatchError(
                f"action session {batch.session_id!r} does not match "
                f"environment session {self.session_id!r}"
            )
        if self._current_observation is None:
            raise ComputerFrameMismatchError(
                "cannot execute actions before an observation is recorded"
            )
        if batch.expected_frame_id != self._current_observation.frame_id:
            raise ComputerFrameMismatchError(
                f"actions target frame {batch.expected_frame_id!r}, but current "
                f"frame is {self._current_observation.frame_id!r}"
            )
        element_ids = {
            element.element_id for element in self._current_observation.elements
        }
        for action in batch.actions:
            target_id = action.target.element_id if action.target else None
            if target_id is not None and target_id not in element_ids:
                raise ComputerTargetNotFoundError(
                    f"element {target_id!r} is not present in frame "
                    f"{self._current_observation.frame_id!r}"
                )

    async def observe(self) -> ComputerObservation:
        """Capture and record the current environment state."""

        async with self._operation_lock:
            self._ensure_open()
            try:
                observation = await self._observe()
                self._record_observation(observation)
            except BaseException:
                self._invalidate_observation()
                raise
            current = self.current_observation
            if current is None:  # pragma: no cover - guarded by _record_observation.
                raise ComputerEnvironmentProtocolError(
                    "environment did not record an observation"
                )
            return current

    async def execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        """Validate and execute actions, then require a fresh observation."""

        async with self._operation_lock:
            detached_batch = batch.model_copy(deep=True)
            self.validate_action_batch(detached_batch)
            try:
                observation = await self._execute(detached_batch)
                self._record_observation(observation)
            except BaseException:
                self._invalidate_observation()
                raise
            current = self.current_observation
            if current is None:  # pragma: no cover - guarded by _record_observation.
                raise ComputerEnvironmentProtocolError(
                    "environment did not record an observation"
                )
            return current

    async def close(self) -> None:
        """Release adapter resources once and reject subsequent operations."""

        async with self._operation_lock:
            if self._close_complete:
                return
            self._closed = True
            self._invalidate_observation()
            await self._close()
            self._seen_frame_ids.clear()
            self._close_complete = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise ComputerEnvironmentClosedError(
                f"computer environment {self.session_id!r} is closed"
            )

    @abstractmethod
    async def _observe(self) -> ComputerObservation:
        """Adapter hook that captures the current environment state."""

    @abstractmethod
    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        """Adapter hook called only after base stale-frame validation."""

    async def _close(self) -> None:
        """Adapter hook for resource cleanup; stateless adapters may omit it."""
