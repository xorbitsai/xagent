from __future__ import annotations

import pytest

from xagent.core.computer.environment import (
    ComputerEnvironment,
    ComputerFrameMismatchError,
    ComputerSessionMismatchError,
    ComputerTargetNotFoundError,
)
from xagent.core.computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerObservation,
    ComputerTarget,
    NormalizedRect,
    Viewport,
)
from xagent.core.context_ref import ContextReference, ContextReferencePurpose


def observation(
    *,
    session_id: str = "session-1",
    frame_id: str = "frame-1",
    with_element: bool = False,
) -> ComputerObservation:
    return ComputerObservation(
        session_id=session_id,
        frame_id=frame_id,
        environment=ComputerEnvironmentType.BROWSER,
        viewport=Viewport(width=100, height=100),
        screenshot=ContextReference(
            file_ref={
                "file_id": f"image-{frame_id}",
                "filename": f"{frame_id}.png",
                "mime_type": "image/png",
            },
            purpose=ContextReferencePurpose.OBSERVATION,
            frame_id=frame_id,
        ),
        elements=(
            [
                ComputerElement(
                    element_id="button-1",
                    source=ComputerElementSource.DOM,
                    bounds=NormalizedRect(x=0.1, y=0.1, width=0.2, height=0.1),
                )
            ]
            if with_element
            else []
        ),
    )


def action_batch(
    *,
    session_id: str = "session-1",
    frame_id: str = "frame-1",
) -> ComputerActionBatch:
    return ComputerActionBatch(
        session_id=session_id,
        expected_frame_id=frame_id,
        actions=[ComputerAction(type=ComputerActionType.SCREENSHOT)],
    )


class FakeEnvironment(ComputerEnvironment):
    async def _observe(self) -> ComputerObservation:
        assert self.current_observation is not None
        return self.current_observation

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        assert self.current_observation is not None
        return self.current_observation


def test_environment_accepts_actions_for_current_frame() -> None:
    environment = FakeEnvironment("session-1")
    environment.record_observation(observation())

    environment.validate_action_batch(action_batch())


def test_environment_rejects_stale_frame() -> None:
    environment = FakeEnvironment("session-1")
    environment.record_observation(observation(frame_id="frame-2"))

    with pytest.raises(ComputerFrameMismatchError, match="current frame"):
        environment.validate_action_batch(action_batch(frame_id="frame-1"))


@pytest.mark.asyncio
async def test_execute_always_applies_stale_frame_guard() -> None:
    environment = FakeEnvironment("session-1")
    environment.record_observation(observation(frame_id="frame-2"))

    with pytest.raises(ComputerFrameMismatchError):
        await environment.execute(action_batch(frame_id="frame-1"))


def test_environment_rejects_element_outside_current_frame() -> None:
    environment = FakeEnvironment("session-1")
    environment.record_observation(observation(with_element=True))
    batch = ComputerActionBatch(
        session_id="session-1",
        expected_frame_id="frame-1",
        actions=[
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(element_id="old-button"),
            )
        ],
    )

    with pytest.raises(ComputerTargetNotFoundError, match="old-button"):
        environment.validate_action_batch(batch)


def test_environment_rejects_other_session() -> None:
    environment = FakeEnvironment("session-1")

    with pytest.raises(ComputerSessionMismatchError):
        environment.record_observation(observation(session_id="session-2"))
