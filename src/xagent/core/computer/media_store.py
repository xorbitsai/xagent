from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..file_ref import build_workspace_file_ref
from .relay import BrowserRelayMediaChunk
from .schema import ComputerMediaKind

COMPUTER_MEDIA_CHUNK_BYTES = 256 * 1024
MAX_COMPUTER_MEDIA_BYTES = 32 * 1024 * 1024

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")
_MIME_SUFFIXES = {
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


class RelayMediaArtifact(BaseModel):
    """Manifest sent after all media chunks have arrived."""

    model_config = ConfigDict(extra="forbid")

    transfer_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    mime_type: str = Field(min_length=1, max_length=100)
    media_kind: ComputerMediaKind
    duration_ms: int = Field(ge=1_000, le=30_000)
    chunk_count: int = Field(ge=1, le=1_000)
    size_bytes: int = Field(ge=1, le=MAX_COMPUTER_MEDIA_BYTES)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class MediaArtifactTransfer:
    """Write ordered relay chunks to a temporary file and commit atomically."""

    def __init__(
        self,
        store: MediaArtifactStore,
        *,
        transfer_id: str,
        media_kind: ComputerMediaKind,
        output_filename: str | None,
    ) -> None:
        self.store = store
        self.transfer_id = transfer_id
        self.media_kind = media_kind
        self.output_filename = output_filename
        self._next_chunk_index = 0
        self._size_bytes = 0
        self._sha256 = hashlib.sha256()
        output_dir = Path(store.workspace.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._temporary_path = output_dir / f".computer-media-{transfer_id}.part"
        self._stream = self._temporary_path.open("xb")
        self._closed = False

    async def accept(self, chunk: BrowserRelayMediaChunk) -> None:
        if self._closed:
            raise ValueError("computer media transfer is already closed")
        if chunk.transfer_id != self.transfer_id:
            raise ValueError("computer relay returned the wrong media transfer")
        if chunk.chunk_index != self._next_chunk_index:
            raise ValueError(
                "computer relay returned out-of-order or duplicate media chunks"
            )
        try:
            payload = base64.b64decode(chunk.data_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("computer relay returned invalid media data") from exc
        if not payload or len(payload) > COMPUTER_MEDIA_CHUNK_BYTES:
            raise ValueError("computer relay returned an invalid media chunk size")
        if self._size_bytes + len(payload) > MAX_COMPUTER_MEDIA_BYTES:
            raise ValueError("computer media exceeds the 32 MiB transfer limit")
        self._stream.write(payload)
        self._sha256.update(payload)
        self._size_bytes += len(payload)
        self._next_chunk_index += 1

    def finish(self, artifact: RelayMediaArtifact) -> dict[str, Any]:
        if self._closed:
            raise ValueError("computer media transfer is already closed")
        try:
            if artifact.transfer_id != self.transfer_id:
                raise ValueError("computer relay returned the wrong media transfer")
            if artifact.media_kind != self.media_kind:
                raise ValueError("computer media kind does not match the request")
            mime_type = artifact.mime_type.partition(";")[0].strip().lower()
            suffix = _MIME_SUFFIXES.get(mime_type)
            if suffix is None:
                raise ValueError(f"unsupported computer media MIME type: {mime_type}")
            if not mime_type.startswith(f"{artifact.media_kind.value}/"):
                raise ValueError("computer media kind does not match its MIME type")
            if artifact.chunk_count != self._next_chunk_index:
                raise ValueError(
                    "computer media transfer is missing one or more chunks"
                )
            if artifact.size_bytes != self._size_bytes:
                raise ValueError("computer media transfer size check failed")
            if artifact.sha256 != self._sha256.hexdigest():
                raise ValueError("computer media transfer checksum failed")

            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._closed = True
            filename = self.store._filename(
                self.output_filename,
                artifact.media_kind,
                suffix,
            )
            final_path = self.store._unique_path(
                Path(self.store.workspace.output_dir) / filename
            )
            os.replace(self._temporary_path, final_path)
            try:
                return build_workspace_file_ref(
                    workspace=self.store.workspace,
                    file_path=final_path,
                    mime_type=mime_type,
                )
            except Exception:
                final_path.unlink(missing_ok=True)
                raise
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True
        self._temporary_path.unlink(missing_ok=True)


class MediaArtifactStore:
    """Receive bounded relay media and register it as a task output FileRef."""

    def __init__(self, workspace: Any) -> None:
        if not hasattr(workspace, "output_dir") or not hasattr(
            workspace, "register_file"
        ):
            raise TypeError("workspace must expose output_dir and register_file")
        self.workspace = workspace

    def begin(
        self,
        *,
        media_kind: ComputerMediaKind,
        output_filename: str | None = None,
    ) -> MediaArtifactTransfer:
        return MediaArtifactTransfer(
            self,
            transfer_id=uuid4().hex,
            media_kind=media_kind,
            output_filename=output_filename,
        )

    @classmethod
    def _filename(
        cls,
        requested: str | None,
        media_kind: ComputerMediaKind,
        suffix: str,
    ) -> str:
        if requested:
            basename = Path(requested.replace("\\", "/")).name
            sanitized = _SAFE_FILENAME.sub("_", basename).strip(" .")
            if sanitized:
                return str(Path(sanitized).with_suffix(suffix))
        return f"computer-{media_kind.value}-{uuid4().hex[:12]}{suffix}"

    @staticmethod
    def _unique_path(candidate: Path) -> Path:
        if not candidate.exists():
            return candidate
        for index in range(2, 10_000):
            alternative = candidate.with_name(
                f"{candidate.stem}-{index}{candidate.suffix}"
            )
            if not alternative.exists():
                return alternative
        raise RuntimeError("could not allocate a unique computer media filename")
