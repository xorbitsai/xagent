from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from xagent.core.computer.media_store import (
    COMPUTER_MEDIA_CHUNK_BYTES,
    MediaArtifactStore,
    RelayMediaArtifact,
)
from xagent.core.computer.relay import BrowserRelayMediaChunk
from xagent.core.computer.schema import ComputerMediaKind


class FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.workspace_dir = root
        self.output_dir = root / "output"
        self.registered: list[str] = []

    def get_file_id_from_path(self, _path: str) -> None:
        return None

    def register_file(self, path: str) -> str:
        self.registered.append(path)
        return "media-file"


def chunk(transfer_id: str, index: int, payload: bytes) -> BrowserRelayMediaChunk:
    return BrowserRelayMediaChunk(
        type="media_chunk",
        protocol_version=1,
        request_id="request-1",
        transfer_id=transfer_id,
        chunk_index=index,
        data_base64=base64.b64encode(payload).decode(),
    )


@pytest.mark.asyncio
async def test_media_store_merges_chunks_and_commits_atomically(
    tmp_path: Path,
) -> None:
    workspace = FakeWorkspace(tmp_path)
    transfer = MediaArtifactStore(workspace).begin(
        media_kind=ComputerMediaKind.VIDEO,
        output_filename="../../demo.txt",
    )
    payloads = [b"first-", b"second"]
    for index, payload in enumerate(payloads):
        await transfer.accept(chunk(transfer.transfer_id, index, payload))
    media = b"".join(payloads)

    file_ref = transfer.finish(
        RelayMediaArtifact(
            transfer_id=transfer.transfer_id,
            mime_type="video/webm;codecs=vp9,opus",
            media_kind=ComputerMediaKind.VIDEO,
            duration_ms=1_000,
            chunk_count=2,
            size_bytes=len(media),
            sha256=hashlib.sha256(media).hexdigest(),
        )
    )

    output = tmp_path / "output" / "demo.webm"
    assert output.read_bytes() == media
    assert file_ref["file_id"] == "media-file"
    assert not list((tmp_path / "output").glob("*.part"))


@pytest.mark.asyncio
async def test_media_store_rejects_out_of_order_and_oversized_chunks(
    tmp_path: Path,
) -> None:
    transfer = MediaArtifactStore(FakeWorkspace(tmp_path)).begin(
        media_kind=ComputerMediaKind.AUDIO
    )

    with pytest.raises(ValueError, match="out-of-order"):
        await transfer.accept(chunk(transfer.transfer_id, 1, b"wrong"))
    with pytest.raises(ValueError, match="chunk size"):
        await transfer.accept(
            chunk(
                transfer.transfer_id,
                0,
                b"x" * (COMPUTER_MEDIA_CHUNK_BYTES + 1),
            )
        )
    transfer.abort()
    assert not list((tmp_path / "output").glob("*.part"))
