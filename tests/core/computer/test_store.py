from __future__ import annotations

from pathlib import Path

import pytest

from xagent.core.computer.materializer import WorkspaceContextReferenceResolver
from xagent.core.computer.schema import Viewport
from xagent.core.computer.store import ObservationStore


class FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.workspace_dir = root
        self.temp_dir = root / "temp"
        self.temp_dir.mkdir(parents=True)
        self._paths: dict[str, Path] = {}

    def get_file_id_from_path(self, path: str) -> str | None:
        resolved = Path(path).resolve()
        return next(
            (file_id for file_id, value in self._paths.items() if value == resolved),
            None,
        )

    def register_file(self, path: str) -> str:
        file_id = f"file-{len(self._paths) + 1}"
        self._paths[file_id] = Path(path).resolve()
        return file_id

    def resolve_file_id(self, file_id: str) -> Path | None:
        return self._paths.get(file_id)


@pytest.mark.asyncio
async def test_observation_store_persists_file_ref_and_resolves_just_in_time(
    tmp_path: Path,
) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace)

    reference = store.save_screenshot(
        session_id="session/unsafe",
        frame_id="frame-1",
        image_bytes=b"fake-png-bytes",
        mime_type="image/png",
        viewport=Viewport(width=1280, height=720),
        text_fallback="browser screenshot",
    )

    assert reference.file_ref["file_id"] == "file-1"
    assert "file_path" not in reference.file_ref
    assert reference.metadata["sha256"]
    assert reference.metadata["viewport"]["width"] == 1280
    assert len(list(store.root.rglob("*.png"))) == 1

    resolver = WorkspaceContextReferenceResolver(workspace)
    materialized = await resolver.resolve_image(reference)

    assert materialized.startswith("data:image/png;base64,")
    assert "fake-png-bytes" not in materialized


def test_observation_store_is_idempotent_for_same_frame_and_bytes(
    tmp_path: Path,
) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace)

    first = store.save_screenshot(
        session_id="session-1",
        frame_id="frame-1",
        image_bytes=b"same-image",
        mime_type="image/png",
    )
    second = store.save_screenshot(
        session_id="session-1",
        frame_id="frame-1",
        image_bytes=b"same-image",
        mime_type="image/png",
    )

    assert first.file_id == second.file_id
    assert len(list(store.root.rglob("*.png"))) == 1
