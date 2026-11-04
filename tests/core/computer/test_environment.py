from __future__ import annotations

import asyncio

import pytest

from xagent.core.computer.environment import (
    ComputerEnvironment,
    ComputerEnvironmentClosedError,
    ComputerEnvironmentProtocolError,
    ComputerFrameMismatchError,
    ComputerSessionMismatchError,
    ComputerTargetNotFoundError,
)
from xagent.core.computer.schema import (
    COMPUTER_FRAME_ID_METADATA_KEY,
    COMPUTER_SESSION_ID_METADATA_KEY,
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerElementSurface,
    ComputerEnvironmentType,
    ComputerObservation,
    ComputerTarget,
    NormalizedRect,
    Viewport,
)
from xagent.core.context_ref import ContextReference


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
            metadata={
                COMPUTER_SESSION_ID_METADATA_KEY: session_id,
                COMPUTER_FRAME_ID_METADATA_KEY: frame_id,
            },
        ),
        elements=(
            [
                ComputerElement(
                    element_id="button-1",
                    source=ComputerElementSource.DOM,
                    bounds=NormalizedRect(x=0.1, y=0.1, width=0.2, height=0.1),
                    surface=ComputerElementSurface.DOCUMENT,
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
    def __init__(self, session_id: str = "session-1") -> None:
        super().__init__(session_id)
        self.next_observation = observation(frame_id="frame-2")

    async def _observe(self) -> ComputerObservation:
        return self.next_observation

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        return self.next_observation


def test_environment_rejects_blank_session_id() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        FakeEnvironment("   ")


@pytest.mark.asyncio
async def test_environment_accepts_actions_for_current_frame() -> None:
    environment = FakeEnvironment()
    environment.next_observation = observation()
    await environment.observe()

    environment.validate_action_batch(action_batch())


@pytest.mark.asyncio
async def test_environment_rejects_stale_frame() -> None:
    environment = FakeEnvironment()
    await environment.observe()

    with pytest.raises(ComputerFrameMismatchError, match="current frame"):
        environment.validate_action_batch(action_batch(frame_id="frame-1"))


@pytest.mark.asyncio
async def test_environment_rejects_element_outside_current_frame() -> None:
    environment = FakeEnvironment()
    environment.next_observation = observation(with_element=True)
    await environment.observe()
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


@pytest.mark.asyncio
async def test_environment_rejects_other_session() -> None:
    environment = FakeEnvironment()
    environment.next_observation = observation(session_id="session-2")

    with pytest.raises(ComputerSessionMismatchError):
        await environment.observe()


@pytest.mark.asyncio
async def test_invalidate_observation_requires_a_fresh_observation() -> None:
    environment = FakeEnvironment()
    environment.next_observation = observation()
    await environment.observe()

    await environment.invalidate_observation()

    assert environment.current_observation is None
    with pytest.raises(ComputerFrameMismatchError, match="before an observation"):
        environment.validate_action_batch(action_batch())


@pytest.mark.asyncio
async def test_execute_requires_adapter_to_return_fresh_frame() -> None:
    environment = FakeEnvironment()
    environment.next_observation = observation()
    await environment.observe()
    environment.next_observation = observation()

    with pytest.raises(ComputerEnvironmentProtocolError, match="fresh frame_id"):
        await environment.execute(action_batch())

    assert environment.current_observation is None


@pytest.mark.asyncio
async def test_environment_rejects_frame_id_reuse_after_an_intermediate_frame() -> None:
    environment = FakeEnvironment()
    environment.next_observation = observation(frame_id="frame-1")
    await environment.observe()
    environment.next_observation = observation(frame_id="frame-2")
    await environment.execute(action_batch(frame_id="frame-1"))
    environment.next_observation = observation(frame_id="frame-1")

    with pytest.raises(ComputerEnvironmentProtocolError, match="must not reuse"):
        await environment.observe()

    assert environment.current_observation is None


class InvalidObservationEnvironment(FakeEnvironment):
    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        return None  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_execute_reports_invalid_observation_as_protocol_error() -> None:
    environment = InvalidObservationEnvironment()
    environment.next_observation = observation(frame_id="frame-1")
    await environment.observe()

    with pytest.raises(
        ComputerEnvironmentProtocolError,
        match="must return ComputerObservation",
    ):
        await environment.execute(action_batch(frame_id="frame-1"))


class FailingEnvironment(FakeEnvironment):
    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        raise RuntimeError("adapter failed after input")


@pytest.mark.asyncio
async def test_failed_execute_invalidates_possibly_stale_frame() -> None:
    environment = FailingEnvironment()
    environment.next_observation = observation()
    await environment.observe()

    with pytest.raises(RuntimeError, match="adapter failed"):
        await environment.execute(action_batch())

    assert environment.current_observation is None


class BlockingEnvironment(FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return observation(frame_id="frame-2")


@pytest.mark.asyncio
async def test_concurrent_actions_are_serialized_under_stale_frame_guard() -> None:
    environment = BlockingEnvironment()
    environment.next_observation = observation()
    await environment.observe()
    batch = action_batch()

    first = asyncio.create_task(environment.execute(batch))
    await environment.started.wait()
    second = asyncio.create_task(environment.execute(batch))
    await asyncio.sleep(0)
    environment.release.set()

    assert (await first).frame_id == "frame-2"
    with pytest.raises(ComputerFrameMismatchError):
        await second
    assert environment.calls == 1


class RetryCloseEnvironment(FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    async def _close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("temporary close failure")


@pytest.mark.asyncio
async def test_close_blocks_operations_and_can_retry_cleanup() -> None:
    environment = RetryCloseEnvironment()
    environment.next_observation = observation()
    await environment.observe()

    with pytest.raises(RuntimeError, match="temporary close failure"):
        await environment.close()
    assert environment.closed is True
    assert environment.current_observation is None
    with pytest.raises(ComputerEnvironmentClosedError):
        await environment.observe()

    await environment.close()
    assert environment.close_calls == 2
    assert environment._seen_frame_ids == set()
    await environment.close()
    assert environment.close_calls == 2
    with pytest.raises(ComputerEnvironmentClosedError):
        environment.validate_action_batch(action_batch())
