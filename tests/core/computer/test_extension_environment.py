from __future__ import annotations

import base64
from typing import Any

import pytest

from xagent.core.computer.extension import ExtensionComputerEnvironment
from xagent.core.computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerTarget,
)
from xagent.core.computer.session import ComputerSessionBinding
from xagent.core.context_ref import ContextReference, ContextReferencePurpose


class FakeObservationStore:
    def save_screenshot(self, **kwargs: Any) -> ContextReference:
        return ContextReference(
            file_ref={
                "file_id": "extension-shot",
                "filename": "extension.png",
                "mime_type": "image/png",
            },
            purpose=ContextReferencePurpose.OBSERVATION,
            frame_id=kwargs["frame_id"],
        )


def relay_observation() -> dict[str, Any]:
    return {
        "observation": {
            "screenshot_base64": base64.b64encode(b"extension-png").decode(),
            "viewport": {
                "width": 1200,
                "height": 800,
                "device_pixel_ratio": 2,
            },
            "elements": [
                {
                    "element_id": "dom-1",
                    "bounds": {
                        "x": 0.1,
                        "y": 0.2,
                        "width": 0.2,
                        "height": 0.1,
                    },
                    "label": "Password label secret",
                    "role": "input",
                    "text": "plaintext-secret",
                    "metadata": {
                        "tag": "input",
                        "input_type": "password",
                        "value": "metadata-secret",
                    },
                }
            ],
            "active_url": "https://example.com/login",
            "title": "Login",
        }
    }


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    async def request(
        self,
        command: str,
        payload: dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        self.calls.append((command, payload, timeout))
        return relay_observation()


class FakeRegistry:
    def __init__(self) -> None:
        self.connection = FakeConnection()
        self.acquisitions: list[tuple[int, str]] = []
        self.releases: list[tuple[int, str]] = []

    async def acquire(self, *, user_id: int, owner_task_id: str) -> FakeConnection:
        self.acquisitions.append((user_id, owner_task_id))
        return self.connection

    async def touch_claim(self, *, user_id: int, owner_task_id: str) -> None:
        return None

    async def release(self, *, user_id: int, owner_task_id: str) -> None:
        self.releases.append((user_id, owner_task_id))


def make_environment() -> tuple[ExtensionComputerEnvironment, FakeRegistry]:
    registry = FakeRegistry()
    binding = ComputerSessionBinding.from_values(
        runtime_kind="extension_relay",
        owner_task_id="task-1",
        user_id=8,
        profile_id="default",
        profile_root=None,
    )
    environment = ExtensionComputerEnvironment(
        session_id="task-1",
        workspace=object(),
        session_binding=binding,
        registry=registry,  # type: ignore[arg-type]
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
    )
    return environment, registry


@pytest.mark.asyncio
async def test_extension_environment_captures_and_redacts_observation() -> None:
    environment, registry = make_environment()

    observation = await environment.observe()

    assert observation.active_url == "https://example.com/login"
    assert observation.metadata["browser_runtime_kind"] == "extension_relay"
    assert observation.elements[0].text is None
    assert observation.elements[0].label == "Sensitive input"
    assert "plaintext-secret" not in observation.model_dump_json()
    assert "metadata-secret" not in observation.model_dump_json()
    assert "Password label secret" not in observation.model_dump_json()
    assert registry.acquisitions == [(8, "task-1")]
    assert registry.connection.calls[0][0] == "observe"


@pytest.mark.asyncio
async def test_extension_environment_serializes_element_target_as_point() -> None:
    environment, registry = make_environment()
    first = await environment.observe()

    second = await environment.execute(
        ComputerActionBatch(
            session_id="task-1",
            expected_frame_id=first.frame_id,
            actions=[
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(element_id="dom-1"),
                )
            ],
        )
    )

    command, payload, _timeout = registry.connection.calls[1]
    assert command == "act"
    assert payload["expected_frame_id"] == first.frame_id
    assert payload["action"]["target"] == pytest.approx({"x": 0.2, "y": 0.25})
    assert second.frame_id == payload["frame_id"]


@pytest.mark.asyncio
async def test_extension_environment_releases_task_claim() -> None:
    environment, registry = make_environment()

    await environment.close()

    assert registry.releases == [(8, "task-1")]
