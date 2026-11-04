from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import stat
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

from ..context_ref import ContextReference
from ..file_ref import build_workspace_file_ref
from .schema import (
    COMPUTER_FRAME_ID_METADATA_KEY,
    COMPUTER_SESSION_ID_METADATA_KEY,
    Viewport,
)

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
_MIME_SUFFIXES = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_DEFAULT_RETENTION = 40
_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_EXECUTION_RETENTION_POLICY = "execution"
_INCOMPLETE_FILE_GRACE_SECONDS = 5 * 60


def _safe_component(value: str, *, label: str) -> tuple[str, str]:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) > 256:
        raise ValueError(f"{label} must be at most 256 characters")
    safe = _SAFE_ID.sub("_", normalized).strip("._") or label
    identity = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return normalized, f"{safe[:48]}-{identity}"


class ObservationStore:
    """Persist immutable screenshots as task-workspace FileRefs.

    ``save_screenshot`` offloads blocking disk and workspace registration from
    the event loop. A bounded per-session retention window keeps old durable
    references safe to persist while allowing their image bytes to expire and
    fall back to the reference's text description.
    """

    def __init__(
        self,
        workspace: Any,
        *,
        retention: int | None = None,
        max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
    ) -> None:
        required_workspace_members = (
            "internal_temp_dir",
            "temp_dir",
            "register_internal_file",
            "unregister_internal_file",
        )
        if any(not hasattr(workspace, member) for member in required_workspace_members):
            raise TypeError(
                "workspace must support internal temp storage and file registration"
            )
        if retention is not None and retention < 1:
            raise ValueError("retention must be positive or None")
        if max_image_bytes < 1:
            raise ValueError("max_image_bytes must be positive")
        self.workspace = workspace
        self.retention = _DEFAULT_RETENTION if retention is None else retention
        self.max_image_bytes = max_image_bytes
        workspace_temp = Path(workspace.temp_dir).resolve()
        # The reserved hidden parent keeps a just-published frame out of
        # workspace discovery even before its process-local registration lands.
        internal_temp = Path(workspace.internal_temp_dir).resolve()
        if internal_temp.parent != workspace_temp:
            raise ValueError(
                "internal temp root must be a direct child of workspace temp"
            )
        self.root = internal_temp / "computer_observations"
        self.root.mkdir(parents=True, exist_ok=True)
        # Re-resolve the child to reject a pre-existing symlink that escapes
        # the workspace after the parent capability check above.
        if not self.root.resolve().is_relative_to(workspace_temp):
            raise ValueError("computer observation root escapes workspace temp")
        self._lock = RLock()

    async def save_screenshot(
        self,
        *,
        session_id: str,
        frame_id: str,
        image_bytes: bytes,
        mime_type: str,
        viewport: Viewport | None = None,
        text_fallback: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContextReference:
        """Store a frame without blocking the caller's event loop."""

        payload = bytes(image_bytes)
        metadata_copy = dict(metadata or {})
        return await asyncio.to_thread(
            self._save_screenshot_sync,
            session_id=session_id,
            frame_id=frame_id,
            image_bytes=payload,
            mime_type=mime_type,
            viewport=viewport,
            text_fallback=text_fallback,
            metadata=metadata_copy,
        )

    def _save_screenshot_sync(
        self,
        *,
        session_id: str,
        frame_id: str,
        image_bytes: bytes,
        mime_type: str,
        viewport: Viewport | None,
        text_fallback: str | None,
        metadata: dict[str, Any],
    ) -> ContextReference:
        if not image_bytes:
            raise ValueError("image_bytes must not be empty")
        if len(image_bytes) > self.max_image_bytes:
            raise ValueError(
                f"image_bytes must be at most {self.max_image_bytes} bytes"
            )
        suffix = _MIME_SUFFIXES.get(mime_type)
        if suffix is None:
            raise ValueError(f"unsupported screenshot MIME type: {mime_type}")
        normalized_session, safe_session = _safe_component(
            session_id,
            label="session_id",
        )
        normalized_frame, safe_frame = _safe_component(
            frame_id,
            label="frame_id",
        )
        digest = hashlib.sha256(image_bytes).hexdigest()

        with self._lock:
            session_dir = self.root / safe_session
            session_dir.mkdir(parents=True, exist_ok=True)
            resolved_root = self.root.resolve()
            resolved_session_dir = session_dir.resolve()
            if resolved_session_dir.parent != resolved_root:
                raise ValueError("computer observation session escapes its root")
            image_path = resolved_session_dir / f"{safe_frame}-{digest}{suffix}"
            frame_prefix = f"{safe_frame}-"
            for existing_path in resolved_session_dir.iterdir():
                if (
                    existing_path.name.startswith(frame_prefix)
                    and not existing_path.name.startswith(".")
                    and existing_path.name != image_path.name
                ):
                    raise ValueError(
                        "frame_id is already bound to different screenshot bytes "
                        "or MIME type"
                    )
            created = self._write_immutable(image_path, image_bytes)

            try:
                file_ref = build_workspace_file_ref(
                    workspace=self.workspace,
                    file_path=image_path,
                    mime_type=mime_type,
                    internal=True,
                )

                ref_metadata: dict[str, Any] = {
                    **metadata,
                    COMPUTER_SESSION_ID_METADATA_KEY: normalized_session,
                    COMPUTER_FRAME_ID_METADATA_KEY: normalized_frame,
                    "sha256": digest,
                    "retention_policy": _EXECUTION_RETENTION_POLICY,
                }
                if viewport is not None:
                    ref_metadata["viewport"] = viewport.model_dump(mode="json")
                reference = ContextReference(
                    file_ref=file_ref,
                    text_fallback=text_fallback,
                    metadata=ref_metadata,
                )
            except Exception:
                if created:
                    try:
                        image_path.unlink(missing_ok=True)
                    except OSError:
                        logger.debug(
                            "Could not remove invalid observation %s",
                            image_path,
                            exc_info=True,
                        )
                    unregister_internal = getattr(
                        self.workspace,
                        "unregister_internal_file",
                        None,
                    )
                    if callable(unregister_internal):
                        try:
                            unregister_internal(str(image_path))
                        except (OSError, ValueError):
                            logger.debug(
                                "Could not compensate observation registration %s",
                                image_path,
                                exc_info=True,
                            )
                raise
            self._prune_session(resolved_session_dir, keep=image_path)

        logger.debug(
            "Stored computer frame %s for session %s as FileRef %s",
            normalized_frame,
            normalized_session,
            reference.file_id,
        )
        return reference

    @staticmethod
    def _write_immutable(path: Path, image_bytes: bytes) -> bool:
        if path.is_symlink():
            raise ValueError("computer observation path must not be a symlink")
        if path.exists():
            if path.read_bytes() != image_bytes:
                raise RuntimeError("immutable observation path contains other bytes")
            return False

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".computer-frame-",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(image_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                if path.read_bytes() != image_bytes:
                    raise RuntimeError(
                        "immutable observation path contains other bytes"
                    )
                return False
            return True
        finally:
            temporary_path.unlink(missing_ok=True)

    def _prune_session(self, session_dir: Path, *, keep: Path) -> None:
        try:
            paths = list(session_dir.iterdir())
        except OSError:
            logger.debug(
                "Could not list observations in %s",
                session_dir,
                exc_info=True,
            )
            return

        entries: list[tuple[int, str, Path]] = []
        try:
            # Derive "now" from the same filesystem as child mtimes so clock
            # skew on a mounted workspace cannot age an in-flight temp file.
            filesystem_now_ns = session_dir.stat().st_mtime_ns
        except OSError:
            filesystem_now_ns = None
        for path in paths:
            if path.name.startswith("."):
                if filesystem_now_ns is not None and path.name.startswith(
                    ".computer-frame-"
                ):
                    try:
                        path_stat = path.lstat()
                        age_ns = filesystem_now_ns - path_stat.st_mtime_ns
                        if age_ns > _INCOMPLETE_FILE_GRACE_SECONDS * 1_000_000_000:
                            path.unlink(missing_ok=True)
                    except OSError:
                        logger.debug(
                            "Could not prune incomplete observation %s",
                            path,
                            exc_info=True,
                        )
                continue
            try:
                path_stat = path.stat()
            except OSError:
                logger.debug("Could not inspect observation %s", path, exc_info=True)
                continue
            if stat.S_ISREG(path_stat.st_mode):
                entries.append((path_stat.st_mtime_ns, path.name, path))
        entries.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        frames = [entry[2] for entry in entries]
        ordered_frames = [keep, *(frame for frame in frames if frame != keep)]
        for frame in ordered_frames[self.retention :]:
            try:
                frame.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not prune observation %s", frame, exc_info=True)
                continue
            unregister_internal = getattr(
                self.workspace,
                "unregister_internal_file",
                None,
            )
            if callable(unregister_internal):
                try:
                    unregister_internal(str(frame))
                except (OSError, ValueError):
                    logger.debug(
                        "Could not unregister observation %s",
                        frame,
                        exc_info=True,
                    )
