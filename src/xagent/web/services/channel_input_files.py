"""Stage channel attachments before selecting a task, then compensate by reference."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from ...core.file_storage.keys import build_upload_storage_key
from .channel_runtime import DownloadedChannelFile
from .db_runtime import run_db_io_cancellation_safe
from .uploaded_file_store import (
    StagedUploadedFile,
    compensate_staged_uploaded_files,
    stage_uploaded_file_from_local_path,
)


@asynccontextmanager
async def stage_channel_input_files(
    files: list[DownloadedChannelFile],
    *,
    user_id: int,
    upload_source: str,
) -> AsyncIterator[tuple[tuple[StagedUploadedFile, ...], list[dict[str, Any]]]]:
    staged: list[StagedUploadedFile] = []

    def stage() -> None:
        for item in files:
            file_id = str(uuid4())
            staged.append(
                stage_uploaded_file_from_local_path(
                    local_path=item.path,
                    user_id=user_id,
                    filename=item.name,
                    file_id=file_id,
                    mime_type=item.mime_type,
                    storage_key=build_upload_storage_key(user_id, file_id, item.name),
                    upload_source=upload_source,
                    execution_scope=None,
                )
            )

    try:
        await run_db_io_cancellation_safe(stage)
        infos = [
            {
                "file_id": saved.file_id,
                "name": saved.filename,
                "path": saved.storage_path,
                "type": saved.mime_type,
                "size": saved.file_size,
                "source_id": original.source_id,
            }
            for saved, original in zip(staged, files, strict=True)
        ]
        yield tuple(staged), infos
    finally:
        if staged:
            await run_db_io_cancellation_safe(
                lambda: compensate_staged_uploaded_files(tuple(staged))
            )
