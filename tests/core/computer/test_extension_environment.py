from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

import pytest

from xagent.core.computer.extension import ExtensionComputerEnvironment
from xagent.core.computer.relay import BrowserRelayMediaChunk
from xagent.core.computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerMediaKind,
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
                        "focused": True,
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
        on_media_chunk: Any = None,
    ) -> dict[str, Any]:
        self.calls.append((command, payload, timeout))
        if command == "capture_media":
            media = b"browser-media"
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
                    "mime_type": "video/webm",
                    "media_kind": "video",
                    "duration_ms": 1_000,
                    "chunk_count": 1,
                    "size_bytes": len(media),
                    "sha256": hashlib.sha256(media).hexdigest(),
                }
            }
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


def make_environment(
    *,
    navigation_allowlist: list[str] | None = None,
    workspace: Any = None,
) -> tuple[ExtensionComputerEnvironment, FakeRegistry]:
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
        workspace=workspace or object(),
        session_binding=binding,
        registry=registry,  # type: ignore[arg-type]
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
        navigation_allowlist=navigation_allowlist,
    )
    return environment, registry


class FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.workspace_dir = root
        self.output_dir = root / "output"

    def get_file_id_from_path(self, _path: str) -> None:
        return None

    def register_file(self, _path: str) -> str:
        return "browser-media-file"


@pytest.mark.asyncio
async def test_extension_environment_captures_and_redacts_observation() -> None:
    environment, registry = make_environment()

    observation = await environment.observe()

    assert observation.active_url == "https://example.com/login"
    assert observation.metadata["browser_runtime_kind"] == "extension_relay"
    assert observation.elements[0].text is None
    assert observation.elements[0].label == "Sensitive input"
    assert observation.elements[0].metadata["focused"] is True
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
    assert payload["navigation_policy"] == {"allowlist": [], "denylist": []}
    assert second.frame_id == payload["frame_id"]


@pytest.mark.asyncio
async def test_extension_environment_blocks_action_on_disallowed_current_host() -> None:
    environment, registry = make_environment(
        navigation_allowlist=["allowed.example"],
    )
    first = await environment.observe()

    with pytest.raises(ValueError, match="outside the configured allowlist"):
        await environment.execute(
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

    assert len(registry.connection.calls) == 1


@pytest.mark.asyncio
async def test_extension_environment_releases_task_claim() -> None:
    environment, registry = make_environment()

    await environment.close()

    assert registry.releases == [(8, "task-1")]


@pytest.mark.asyncio
async def test_extension_environment_streams_media_to_a_file_ref(
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
                    media_kind=ComputerMediaKind.VIDEO,
                    duration_ms=1_000,
                    output_filename="capture.webm",
                )
            ],
        )
    )

    assert [call[0] for call in registry.connection.calls] == [
        "observe",
        "capture_media",
        "observe",
    ]
    assert environment.action_artifacts[0]["file_id"] == "browser-media-file"
    assert (tmp_path / "output" / "capture.webm").read_bytes() == b"browser-media"

    await environment.observe()
    assert environment.action_artifacts == []
