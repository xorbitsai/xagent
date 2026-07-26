from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

import pytest

from xagent.core.computer.desktop import DesktopRelayEnvironment
from xagent.core.computer.relay import BrowserRelayMediaChunk
from xagent.core.computer.schema import (
    ELEMENT_EXTRACTION_INCOMPLETE_KEY,
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerMediaKind,
    ComputerTarget,
)
from xagent.core.computer.session import ComputerSessionBinding
from xagent.core.context_ref import ContextReference, ContextReferencePurpose


class FakeObservationStore:
    def save_screenshot(self, **kwargs: Any) -> ContextReference:
        return ContextReference(
            file_ref={
                "file_id": "desktop-shot",
                "filename": "desktop.png",
                "mime_type": "image/png",
            },
            purpose=ContextReferencePurpose.OBSERVATION,
            frame_id=kwargs["frame_id"],
        )


def relay_observation(
    *,
    paused: bool = False,
    emergency_stopped: bool = False,
) -> dict[str, Any]:
    return {
        "observation": {
            "screenshot_base64": base64.b64encode(b"desktop-png").decode(),
            "viewport": {
                "width": 900,
                "height": 600,
                "device_pixel_ratio": 2,
            },
            "elements": [
                {
                    "element_id": "ax-1",
                    "bounds": {
                        "x": 0.2,
                        "y": 0.1,
                        "width": 0.4,
                        "height": 0.2,
                    },
                    "label": "Account secret",
                    "role": "AXTextField",
                    "text": "must-not-leak",
                    "metadata": {
                        "role": "AXTextField",
                        "subrole": "AXSecureTextField",
                        "focused": True,
                        "sensitive": True,
                    },
                }
            ],
            "element_extraction_incomplete": True,
            "window_id": 81,
            "title": "Sign in",
            "application": "Example App",
            "platform": "macos",
            "supported_actions": [
                "screenshot",
                "capture_media",
                "click",
                "double_click",
                "move",
                "scroll",
                "type",
                "replace_text",
                "keypress",
                "drag",
                "wait",
            ],
            "paused": paused,
            "emergency_stopped": emergency_stopped,
        }
    }


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], float]] = []
        self.response = relay_observation()

    async def request(
        self,
        command: str,
        payload: dict[str, Any],
        *,
        timeout: float = 30.0,
        on_media_chunk: Any = None,
    ) -> dict[str, Any]:
        self.calls.append((command, payload, timeout))
        if command == "capture_media":
            media = b"desktop-media"
            await on_media_chunk(
                BrowserRelayMediaChunk(
                    type="media_chunk",
                    protocol_version=1,
                    request_id="request-1",
                    transfer_id=payload["transfer_id"],
                    chunk_index=0,
                    data_base64=base64.b64encode(media).decode(),
                )
            )
            return {
                "artifact": {
                    "transfer_id": payload["transfer_id"],
                    "mime_type": "audio/mp4",
                    "media_kind": "audio",
                    "duration_ms": 1_000,
                    "chunk_count": 1,
                    "size_bytes": len(media),
                    "sha256": hashlib.sha256(media).hexdigest(),
                }
            }
        return self.response


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


def make_environment(
    *, workspace: Any = None
) -> tuple[DesktopRelayEnvironment, FakeRegistry]:
    registry = FakeRegistry()
    binding = ComputerSessionBinding.from_values(
        runtime_kind="desktop_relay",
        owner_task_id="task-1",
        user_id=8,
        profile_id="default",
        profile_root=None,
    )
    environment = DesktopRelayEnvironment(
        session_id="task-1",
        workspace=workspace or object(),
        session_binding=binding,
        registry=registry,  # type: ignore[arg-type]
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
    )
    return environment, registry


class FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.workspace_dir = root
        self.output_dir = root / "output"

    def get_file_id_from_path(self, _path: str) -> None:
        return None

    def register_file(self, _path: str) -> str:
        return "desktop-media-file"


@pytest.mark.asyncio
async def test_desktop_environment_captures_window_and_redacts_secure_fields() -> None:
    environment, registry = make_environment()

    observation = await environment.observe()

    assert observation.environment is ComputerEnvironmentType.DESKTOP
    assert observation.metadata["computer_runtime_kind"] == "desktop_relay"
    assert observation.metadata["platform"] == "macos"
    assert observation.metadata["primary_modifier"] == "META"
    assert "replace_text" in observation.metadata["supported_actions"]
    assert observation.metadata["window_id"] == 81
    assert observation.metadata[ELEMENT_EXTRACTION_INCOMPLETE_KEY] is True
    assert observation.elements[0].source is ComputerElementSource.UI_AUTOMATION
    assert observation.elements[0].label == "Sensitive input"
    assert observation.elements[0].text is None
    assert "must-not-leak" not in observation.model_dump_json()
    assert "Account secret" not in observation.model_dump_json()
    assert registry.acquisitions == [(8, "task-1")]


@pytest.mark.asyncio
async def test_desktop_environment_resolves_element_target_in_authorized_window() -> (
    None
):
    environment, registry = make_environment()
    first = await environment.observe()

    second = await environment.execute(
        ComputerActionBatch(
            session_id="task-1",
            expected_frame_id=first.frame_id,
            actions=[
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(element_id="ax-1"),
                )
            ],
        )
    )

    command, payload, _timeout = registry.connection.calls[1]
    assert command == "act"
    assert payload["expected_frame_id"] == first.frame_id
    assert payload["action"]["target_element_id"] == "ax-1"
    assert payload["action"]["target"] == pytest.approx({"x": 0.4, "y": 0.2})
    assert second.frame_id == payload["frame_id"]


@pytest.mark.asyncio
async def test_desktop_environment_refuses_navigation_and_paused_input() -> None:
    environment, registry = make_environment()
    first = await environment.observe()

    with pytest.raises(ValueError, match="navigate is not supported"):
        await environment.execute(
            ComputerActionBatch(
                session_id="task-1",
                expected_frame_id=first.frame_id,
                actions=[
                    ComputerAction(
                        type=ComputerActionType.NAVIGATE,
                        url="https://example.com",
                    )
                ],
            )
        )

    registry.connection.response = relay_observation(paused=True)
    paused = await environment.observe()
    with pytest.raises(RuntimeError, match="paused by the user"):
        await environment.execute(
            ComputerActionBatch(
                session_id="task-1",
                expected_frame_id=paused.frame_id,
                actions=[
                    ComputerAction(
                        type=ComputerActionType.CLICK,
                        target=ComputerTarget(point={"x": 0.5, "y": 0.5}),
                    )
                ],
            )
        )


@pytest.mark.asyncio
async def test_desktop_environment_rejects_emergency_stop_and_releases_claim() -> None:
    environment, registry = make_environment()
    registry.connection.response = relay_observation(emergency_stopped=True)

    with pytest.raises(RuntimeError, match="emergency stop"):
        await environment.observe()

    await environment.close()
    assert registry.releases == [(8, "task-1")]


@pytest.mark.asyncio
async def test_desktop_environment_streams_media_to_a_file_ref(
    tmp_path: Path,
) -> None:
    environment, registry = make_environment(workspace=FakeWorkspace(tmp_path))
    first = await environment.observe()

    await environment.execute(
        ComputerActionBatch(
            session_id="task-1",
            expected_frame_id=first.frame_id,
            actions=[
                ComputerAction(
                    type=ComputerActionType.CAPTURE_MEDIA,
                    media_kind=ComputerMediaKind.AUDIO,
                    duration_ms=1_000,
                    output_filename="capture.m4a",
                )
            ],
        )
    )

    assert [call[0] for call in registry.connection.calls] == [
        "observe",
        "capture_media",
        "observe",
    ]
    assert environment.action_artifacts[0]["file_id"] == "desktop-media-file"
    assert (tmp_path / "output" / "capture.m4a").read_bytes() == b"desktop-media"
