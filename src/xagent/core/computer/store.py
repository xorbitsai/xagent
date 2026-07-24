from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ..context_ref import ContextReference, ContextReferencePurpose
from ..file_ref import build_workspace_file_ref
from .schema import Viewport

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
_MIME_SUFFIXES = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class ObservationStore:
    """Stores immutable screenshots and returns durable FileRef observations."""

    def __init__(self, workspace: Any) -> None:
        if not hasattr(workspace, "temp_dir") or not hasattr(
            workspace, "register_file"
        ):
            raise TypeError("workspace must expose temp_dir and register_file")
        self.workspace = workspace
        self.root = Path(workspace.temp_dir) / "computer_observations"
        self.root.mkdir(parents=True, exist_ok=True)

    def save_screenshot(
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
        if not image_bytes:
            raise ValueError("image_bytes must not be empty")
        suffix = _MIME_SUFFIXES.get(mime_type)
        if suffix is None:
            raise ValueError(f"unsupported screenshot MIME type: {mime_type}")

        safe_session = _SAFE_ID.sub("_", session_id).strip("._") or "session"
        safe_frame = _SAFE_ID.sub("_", frame_id).strip("._") or "frame"
        digest = hashlib.sha256(image_bytes).hexdigest()
        session_dir = self.root / safe_session
        session_dir.mkdir(parents=True, exist_ok=True)
        image_path = session_dir / f"{safe_frame}-{digest}{suffix}"
        if image_path.exists():
            if image_path.read_bytes() != image_bytes:
                raise RuntimeError("immutable observation path contains other bytes")
        else:
            with image_path.open("xb") as stream:
                stream.write(image_bytes)

        file_ref = build_workspace_file_ref(
            workspace=self.workspace,
            file_path=image_path,
            mime_type=mime_type,
        )
        ref_metadata = {
            **(metadata or {}),
            "sha256": digest,
            "retention": "execution",
        }
        if viewport is not None:
            ref_metadata["viewport"] = viewport.model_dump(mode="json")
        return ContextReference(
            file_ref=file_ref,
            purpose=ContextReferencePurpose.OBSERVATION,
            frame_id=frame_id,
            text_fallback=text_fallback,
            metadata=ref_metadata,
        )
