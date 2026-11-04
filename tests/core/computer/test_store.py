from __future__ import annotations

import asyncio
import os
import threading
import time
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from xagent.core.computer.schema import (
    COMPUTER_FRAME_ID_METADATA_KEY,
    COMPUTER_SESSION_ID_METADATA_KEY,
    Viewport,
)
from xagent.core.computer.store import ObservationStore
from xagent.core.context_materializer import WorkspaceContextReferenceResolver
from xagent.core.workspace import TaskWorkspace


class FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.workspace_dir = root
        self.temp_dir = root / "temp"
        self.temp_dir.mkdir(parents=True)
        self.internal_temp_dir = self.temp_dir / ".fake-internal"
        self._paths: dict[str, Path] = {}
        self.registration_threads: list[int] = []
        self.durable_registration_calls = 0

    def get_file_id_from_path(self, path: str) -> str | None:
        resolved = Path(path).resolve()
        return next(
            (file_id for file_id, value in self._paths.items() if value == resolved),
            None,
        )

    def register_file(self, path: str) -> str:
        self.durable_registration_calls += 1
        return self._register(path)

    def register_internal_file(self, path: str) -> str:
        return self._register(path)

    def _register(self, path: str) -> str:
        self.registration_threads.append(threading.get_ident())
        existing = self.get_file_id_from_path(path)
        if existing is not None:
            return existing
        file_id = f"file-{len(self._paths) + 1}"
        self._paths[file_id] = Path(path).resolve()
        return file_id

    def resolve_file_id_detached(self, file_id: str) -> Path | None:
        return self._paths.get(file_id)

    def unregister_internal_file(self, path: str) -> str | None:
        resolved = Path(path).resolve()
        file_id = next(
            (key for key, value in self._paths.items() if value == resolved),
            None,
        )
        if file_id is not None:
            self._paths.pop(file_id, None)
        return file_id


class DurableOnlyWorkspace:
    def __init__(self, root: Path) -> None:
        self.workspace_dir = root
        self.temp_dir = root / "temp"
        self.temp_dir.mkdir(parents=True)

    def register_file(self, path: str) -> str:
        return "durable-file"


class MissingInternalTempWorkspace(FakeWorkspace):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        del self.internal_temp_dir


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(output, format="PNG")
    return output.getvalue()


@pytest.mark.asyncio
async def test_observation_store_persists_and_resolves_file_ref(
    tmp_path: Path,
) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace)
    event_loop_thread = threading.get_ident()
    image_bytes = _png_bytes()

    reference = await store.save_screenshot(
        session_id="session/unsafe",
        frame_id="frame-1",
        image_bytes=image_bytes,
        mime_type="image/png",
        viewport=Viewport(width=1280, height=720),
        text_fallback="browser screenshot",
    )

    assert reference.file_ref["file_id"] == "file-1"
    assert reference.file_ref["internal"] is True
    assert reference.file_ref["preview_url"] is None
    assert reference.file_ref["download_url"] is None
    assert reference.file_ref["markdown_link"] is None
    assert "file_path" not in reference.file_ref
    assert "relative_path" not in reference.file_ref
    assert reference.metadata[COMPUTER_SESSION_ID_METADATA_KEY] == "session/unsafe"
    assert reference.metadata[COMPUTER_FRAME_ID_METADATA_KEY] == "frame-1"
    assert reference.metadata["sha256"]
    assert reference.metadata["viewport"]["width"] == 1280
    assert workspace.registration_threads
    assert all(
        thread_id != event_loop_thread for thread_id in workspace.registration_threads
    )
    assert len(list(store.root.rglob("*.png"))) == 1
    assert workspace.durable_registration_calls == 0

    materialized = await WorkspaceContextReferenceResolver(workspace).resolve_image(
        reference
    )
    assert materialized.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_observation_store_is_idempotent_for_same_frame_and_bytes(
    tmp_path: Path,
) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace)

    first, second = await asyncio.gather(
        store.save_screenshot(
            session_id="session-1",
            frame_id="frame-1",
            image_bytes=b"same-image",
            mime_type="image/png",
        ),
        store.save_screenshot(
            session_id="session-1",
            frame_id="frame-1",
            image_bytes=b"same-image",
            mime_type="image/png",
        ),
    )

    assert first.file_id == second.file_id
    assert len(list(store.root.rglob("*.png"))) == 1


@pytest.mark.asyncio
async def test_observation_store_rejects_frame_reuse_with_different_image(
    tmp_path: Path,
) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace)
    await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-1",
        image_bytes=b"first-image",
        mime_type="image/png",
    )

    with pytest.raises(ValueError, match="already bound to different screenshot"):
        await store.save_screenshot(
            session_id="session-1",
            frame_id="frame-1",
            image_bytes=b"second-image",
            mime_type="image/png",
        )

    assert len(list(store.root.rglob("*.png"))) == 1


@pytest.mark.asyncio
async def test_observation_store_prunes_old_frame_bytes(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace, retention=2)

    first = await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-0",
        image_bytes=b"image-0",
        mime_type="image/png",
    )
    first_path = workspace.resolve_file_id_detached(first.file_id)
    assert first_path is not None
    os.utime(first_path, ns=(1, 1))

    second = await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-1",
        image_bytes=b"image-1",
        mime_type="image/png",
    )
    second_path = workspace.resolve_file_id_detached(second.file_id)
    assert second_path is not None
    os.utime(second_path, ns=(2, 2))

    latest = await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-2",
        image_bytes=b"image-2",
        mime_type="image/png",
    )

    assert len(list(store.root.rglob("*.png"))) == 2
    assert workspace.resolve_file_id_detached(first.file_id) is None
    assert workspace.resolve_file_id_detached(second.file_id) is not None
    latest_path = workspace.resolve_file_id_detached(latest.file_id)
    assert latest_path is not None
    assert latest_path.exists()


@pytest.mark.asyncio
async def test_observation_store_prunes_incomplete_temp_files(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace)
    await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-0",
        image_bytes=b"first-image",
        mime_type="image/png",
    )
    session_dir = next(path for path in store.root.iterdir() if path.is_dir())
    incomplete = session_dir / ".computer-frame-crash"
    incomplete.write_bytes(b"partial")
    old_timestamp = time.time() - 3_600
    os.utime(incomplete, (old_timestamp, old_timestamp))

    await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-1",
        image_bytes=b"image",
        mime_type="image/png",
    )

    assert not incomplete.exists()


@pytest.mark.asyncio
async def test_observation_store_preserves_fresh_incomplete_temp_files(
    tmp_path: Path,
) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace)
    await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-0",
        image_bytes=b"first-image",
        mime_type="image/png",
    )
    session_dir = next(path for path in store.root.iterdir() if path.is_dir())
    incomplete = session_dir / ".computer-frame-in-flight"
    incomplete.write_bytes(b"partial")

    await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-1",
        image_bytes=b"image",
        mime_type="image/png",
    )

    assert incomplete.exists()


@pytest.mark.asyncio
async def test_observation_store_unregisters_concurrently_deleted_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace, retention=1)
    first = await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-1",
        image_bytes=b"image-1",
        mime_type="image/png",
    )
    first_path = workspace.resolve_file_id_detached(first.file_id)
    assert first_path is not None

    original_unlink = Path.unlink
    externally_deleted = False

    def unlink_after_external_delete(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal externally_deleted
        if path == first_path and not externally_deleted:
            original_unlink(path)
            externally_deleted = True
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_after_external_delete)

    await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-2",
        image_bytes=b"image-2",
        mime_type="image/png",
    )

    assert externally_deleted is True
    assert workspace.resolve_file_id_detached(first.file_id) is None


@pytest.mark.asyncio
async def test_observation_store_prunes_other_frames_after_stat_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace, retention=3)
    first = await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-1",
        image_bytes=b"image-1",
        mime_type="image/png",
    )
    second = await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-2",
        image_bytes=b"image-2",
        mime_type="image/png",
    )
    first_path = workspace.resolve_file_id_detached(first.file_id)
    assert first_path is not None

    original_stat = Path.stat
    original_unlink = Path.unlink
    externally_deleted = False

    def stat_after_external_delete(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal externally_deleted
        if path == first_path and not externally_deleted:
            original_unlink(path)
            externally_deleted = True
            raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    store.retention = 1
    monkeypatch.setattr(Path, "stat", stat_after_external_delete)

    latest = await store.save_screenshot(
        session_id="session-1",
        frame_id="frame-3",
        image_bytes=b"image-3",
        mime_type="image/png",
    )

    assert externally_deleted is True
    assert workspace.resolve_file_id_detached(second.file_id) is None
    assert workspace.resolve_file_id_detached(latest.file_id) is not None


@pytest.mark.asyncio
async def test_observation_store_rejects_invalid_input(tmp_path: Path) -> None:
    store = ObservationStore(
        FakeWorkspace(tmp_path / "workspace"),
        max_image_bytes=5,
    )

    with pytest.raises(ValueError, match="must not be empty"):
        await store.save_screenshot(
            session_id="session-1",
            frame_id="frame-1",
            image_bytes=b"",
            mime_type="image/png",
        )
    with pytest.raises(ValueError, match="unsupported screenshot MIME"):
        await store.save_screenshot(
            session_id="session-1",
            frame_id="frame-1",
            image_bytes=b"image",
            mime_type="image/svg+xml",
        )
    with pytest.raises(ValueError, match="at most 5 bytes"):
        await store.save_screenshot(
            session_id="session-1",
            frame_id="frame-1",
            image_bytes=b"123456",
            mime_type="image/png",
        )


def test_observation_store_requires_internal_registration(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="internal temp storage and file registration"):
        ObservationStore(DurableOnlyWorkspace(tmp_path / "workspace"))


def test_observation_store_requires_internal_temp_capability(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="internal temp storage and file registration"):
        ObservationStore(MissingInternalTempWorkspace(tmp_path / "workspace"))


def test_observation_store_rejects_nested_internal_temp_root(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    workspace.internal_temp_dir = workspace.temp_dir / "nested" / ".fake-internal"

    with pytest.raises(ValueError, match="direct child"):
        ObservationStore(workspace)


def test_observation_store_rejects_invalid_limits(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    with pytest.raises(ValueError, match="retention must be positive"):
        ObservationStore(workspace, retention=0)
    with pytest.raises(ValueError, match="max_image_bytes must be positive"):
        ObservationStore(workspace, max_image_bytes=0)


@pytest.mark.asyncio
async def test_observation_store_compensates_failed_reference_validation(
    tmp_path: Path,
) -> None:
    workspace = FakeWorkspace(tmp_path / "workspace")
    store = ObservationStore(workspace)

    with pytest.raises(ValueError, match="must not contain a path"):
        await store.save_screenshot(
            session_id="session-1",
            frame_id="frame-1",
            image_bytes=b"image",
            mime_type="image/png",
            metadata={"path": "/private/secret.png"},
        )

    assert list(store.root.rglob("*.png")) == []
    assert workspace._paths == {}


def test_internal_file_registration_survives_workspace_reconstruction(
    tmp_path: Path,
) -> None:
    first = TaskWorkspace("task_41", base_dir=str(tmp_path))
    reconstructed: TaskWorkspace | None = None
    file_id: str | None = None
    try:
        frame = first.temp_dir / "frame.png"
        frame.write_bytes(b"frame")

        file_id = first.register_internal_file(str(frame))
        assert file_id.startswith("internal-")
        reconstructed = TaskWorkspace("task_41", base_dir=str(tmp_path))

        assert reconstructed.get_file_id_from_path(str(frame)) == file_id
        assert reconstructed.resolve_file_id_detached(file_id) == frame.resolve()
    finally:
        if reconstructed is not None:
            reconstructed.cleanup()
        first.cleanup()

    assert file_id is not None
    assert first._resolve_internal_file_id(file_id) is None


def test_internal_file_registration_rejects_external_paths(tmp_path: Path) -> None:
    workspace = TaskWorkspace("task_42", base_dir=str(tmp_path / "workspaces"))
    try:
        external = tmp_path / "external.png"
        external.write_bytes(b"external")

        with pytest.raises(ValueError, match="regular workspace temp files"):
            workspace.register_internal_file(str(external))
    finally:
        workspace.cleanup()


def test_internal_file_registration_rejects_output_files(tmp_path: Path) -> None:
    workspace = TaskWorkspace("task_46", base_dir=str(tmp_path))
    try:
        output = workspace.output_dir / "report.txt"
        output.write_text("report", encoding="utf-8")

        with pytest.raises(ValueError, match="regular workspace temp files"):
            workspace.register_internal_file(str(output))
    finally:
        workspace.cleanup()


def test_internal_file_registration_is_isolated_by_workspace_root(
    tmp_path: Path,
) -> None:
    first = TaskWorkspace("task_44", base_dir=str(tmp_path / "first"))
    second = TaskWorkspace("task_44", base_dir=str(tmp_path / "second"))
    try:
        frame = first.temp_dir / "frame.png"
        frame.write_bytes(b"frame")
        file_id = first.register_internal_file(str(frame))

        assert first.resolve_file_id_detached(file_id) == frame.resolve()
        assert second.resolve_file_id_detached(file_id) is None
    finally:
        first.cleanup()
        second.cleanup()


def test_internal_files_are_excluded_from_auto_registration_scan(
    tmp_path: Path,
) -> None:
    workspace = TaskWorkspace("task_45", base_dir=str(tmp_path))
    try:
        internal = workspace.temp_dir / "frame.png"
        internal.write_bytes(b"frame")
        visible = workspace.output_dir / "report.txt"
        visible.write_text("report", encoding="utf-8")
        workspace.register_internal_file(str(internal))

        scanned = workspace._scan_all_files()

        assert internal not in scanned
        assert visible in scanned
    finally:
        workspace.cleanup()


def test_internal_files_are_excluded_from_workspace_listing(tmp_path: Path) -> None:
    workspace = TaskWorkspace("task_47", base_dir=str(tmp_path))
    try:
        internal = workspace.temp_dir / "frame.png"
        internal.write_bytes(b"frame")
        workspace.register_internal_file(str(internal))

        files = workspace.get_all_files()

        assert files["temp"] == []
    finally:
        workspace.cleanup()


@pytest.mark.asyncio
async def test_observation_store_is_hidden_from_real_workspace_listing(
    tmp_path: Path,
) -> None:
    workspace = TaskWorkspace("task_48", base_dir=str(tmp_path))
    try:
        reference = await ObservationStore(workspace).save_screenshot(
            session_id="session-1",
            frame_id="frame-1",
            image_bytes=b"frame",
            mime_type="image/png",
        )

        assert workspace.resolve_file_id_detached(reference.file_id) is not None
        assert workspace.get_all_files()["temp"] == []
    finally:
        workspace.cleanup()


def test_unregistered_reserved_internal_temp_is_hidden(tmp_path: Path) -> None:
    workspace = TaskWorkspace("task_51", base_dir=str(tmp_path))
    try:
        frame = (
            workspace.internal_temp_dir
            / "computer_observations"
            / "session"
            / "frame.png"
        )
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"frame")

        assert workspace._is_internal_workspace_path(frame) is True
        assert workspace.get_all_files()["temp"] == []
    finally:
        workspace.cleanup()


def test_write_immutable_rejects_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    target.write_bytes(b"image")
    link = tmp_path / "frame.png"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        ObservationStore._write_immutable(link, b"image")


def test_write_immutable_rejects_existing_different_bytes(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"old")

    with pytest.raises(RuntimeError, match="contains other bytes"):
        ObservationStore._write_immutable(frame, b"new")


def test_internal_registration_rejects_temp_symlink_to_output(tmp_path: Path) -> None:
    workspace = TaskWorkspace("task_43", base_dir=str(tmp_path))
    try:
        target = workspace.output_dir / "frame.png"
        target.write_bytes(b"frame")
        link = workspace.temp_dir / "frame-link.png"
        link.symlink_to(target)

        with pytest.raises(ValueError, match="regular workspace temp files"):
            workspace.register_internal_file(str(link))
    finally:
        workspace.cleanup()


def test_clean_temp_files_unregisters_internal_files(tmp_path: Path) -> None:
    workspace = TaskWorkspace("task_50", base_dir=str(tmp_path))
    try:
        internal = workspace.temp_dir / "frame.png"
        internal.write_bytes(b"frame")
        file_id = workspace.register_internal_file(str(internal))

        workspace.clean_temp_files()

        assert not internal.exists()
        assert workspace.resolve_file_id_detached(file_id) is None
    finally:
        workspace.cleanup()


def test_internal_file_miss_warns_about_process_local_scope(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace = TaskWorkspace("task_49", base_dir=str(tmp_path))
    try:
        with caplog.at_level("WARNING"):
            assert workspace.resolve_file_id_detached("internal-missing") is None

        assert "Process-local internal file is unavailable" in caplog.text
    finally:
        workspace.cleanup()
