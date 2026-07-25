from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import ComputerActionBatch, ComputerObservation


class ComputerFrameMismatchError(RuntimeError):
    """Raised when actions target an observation that is no longer current."""


class ComputerSessionMismatchError(RuntimeError):
    """Raised when actions target another computer session."""


class ComputerTargetNotFoundError(RuntimeError):
    """Raised when an action references an element outside the current frame."""


class ComputerEnvironment(ABC):
    """Provider-neutral environment contract with mandatory stale-frame checks."""

    def __init__(self, session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        self.session_id = session_id
        self._current_observation: ComputerObservation | None = None

    @property
    def current_observation(self) -> ComputerObservation | None:
        return self._current_observation

    def record_observation(self, observation: ComputerObservation) -> None:
        if observation.session_id != self.session_id:
            raise ComputerSessionMismatchError(
                f"observation session {observation.session_id!r} does not match "
                f"environment session {self.session_id!r}"
            )
        self._current_observation = observation

    def invalidate_observation(self) -> None:
        """Forget a frame whose target may have changed while unavailable."""

        self._current_observation = None

    def validate_action_batch(self, batch: ComputerActionBatch) -> None:
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
        observation = await self._observe()
        self.record_observation(observation)
        return observation

    async def execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        """Validate and execute actions, then return a new observation."""
        self.validate_action_batch(batch)
        observation = await self._execute(batch)
        self.record_observation(observation)
        return observation

    async def close(self) -> None:
        """Release environment resources. Stateless adapters may keep the no-op."""

    @abstractmethod
    async def _observe(self) -> ComputerObservation:
        """Adapter hook that captures the current environment state."""

    @abstractmethod
    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        """Adapter hook called only after the base stale-frame validation."""
