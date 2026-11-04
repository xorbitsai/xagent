"""Serialize source-reference publication with cleanup of local LanceDB files."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

from filelock import FileLock

_validator: Callable[[tuple[str, ...]], None] | None = None
_Operation = TypeVar("_Operation", bound=Callable[..., Any])


def _reference_directory() -> Path:
    from ......providers.vector_store.lancedb import LanceDBConnectionManager

    return Path(LanceDBConnectionManager().resolve_dir_from_env()) / ".file-references"


def _identity_path(file_id: str, suffix: str) -> Path:
    return (
        _reference_directory()
        / f"{hashlib.sha256(file_id.encode()).hexdigest()}.{suffix}"
    )


def has_cleanup_fence(file_id: str) -> bool:
    return _identity_path(file_id, "claimed").exists()


def persist_cleanup_fence(file_id: str) -> None:
    """Publish a conservative standalone fence before the SQL claim commits.

    A failed SQL claim may leave this marker. Web validation reconciles it
    against the surviving available upload; it never pins the file against GC.
    """
    path = _identity_path(file_id, "claimed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as marker:
        marker.write(file_id)
        marker.flush()
        os.fsync(marker.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def clear_cleanup_fence(file_id: str) -> None:
    """Caller holds the reference lock and has proved a live publication."""
    _identity_path(file_id, "claimed").unlink(missing_ok=True)


def set_file_reference_validator(
    validator: Callable[[tuple[str, ...]], None] | None,
) -> None:
    """Install the web metadata boundary; standalone RAG requires no SQL setup."""
    global _validator
    _validator = validator


@contextmanager
def file_cleanup_lock(file_id: str) -> Iterator[None]:
    """Serialize cleanup execution separately from reference publication.

    This inode also survives settlement. Never unlink it while workers run.
    """
    path = _identity_path(file_id, "cleanup.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path, timeout=15, thread_local=False):
        yield


@contextmanager
def file_reference_lock(file_ids: Iterable[str]) -> Iterator[None]:
    """Hold process-owned locks in stable order, without a SQL connection.

    Every process using a KB must share its LanceDB directory and filesystem
    locking semantics. Lock files must never be unlinked: replacing their inode
    would allow two owners. Process exit releases the OS lock automatically.
    """
    ids = sorted(set(file_ids))
    if not ids:
        yield
        return
    directory = _reference_directory()
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for file_id in ids:
            key = hashlib.sha256(file_id.encode()).hexdigest()
            stack.enter_context(
                FileLock(directory / f"{key}.lock", timeout=15, thread_local=False)
            )
        yield


@contextmanager
def protect_file_references(records: Iterable[dict[str, Any]]) -> Iterator[None]:
    ids = tuple(sorted({str(row["file_id"]) for row in records if row.get("file_id")}))
    with file_reference_lock(ids):
        if ids and _validator is not None:
            _validator(ids)
        if any(has_cleanup_fence(file_id) for file_id in ids):
            raise RuntimeError(
                "Source file was claimed; publish a new upload before referencing it"
            )
        yield


async def _drain(operation: Any) -> Any:
    """Do not release a publication lock while a cancelled native write runs."""
    task = asyncio.create_task(operation)
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if not asyncio.current_task().cancelling():  # type: ignore[union-attr]
                raise
            cancellation = exc
            if task.done():
                break
        except Exception:
            if cancellation is not None:
                raise cancellation
            raise
    if cancellation is not None:
        # Observe a late error as well as a late successful commit.
        try:
            task.result()
        except Exception as exc:
            raise cancellation from exc
        raise cancellation
    return result


def guard_document_upsert(operation: _Operation) -> _Operation:
    """Apply one boundary to both the sync and native async store APIs."""
    if asyncio.iscoroutinefunction(operation):

        @wraps(operation)
        async def asynchronous(self: Any, records: list[dict[str, Any]]) -> None:
            async def publish() -> None:
                guard = protect_file_references(records)
                await asyncio.to_thread(guard.__enter__)
                try:
                    await operation(self, records)
                finally:
                    await asyncio.to_thread(guard.__exit__, None, None, None)

            await _drain(publish())

        return cast(_Operation, asynchronous)

    @wraps(operation)
    def synchronous(self: Any, records: list[dict[str, Any]]) -> None:
        with protect_file_references(records):
            operation(self, records)

    return cast(_Operation, synchronous)


def guard_document_restore(operation: _Operation) -> _Operation:
    @wraps(operation)
    def restore(self: Any, snapshot: Any, **kwargs: Any) -> Any:
        with protect_file_references(snapshot.rows_by_table.get("documents", [])):
            return operation(self, snapshot, **kwargs)

    return cast(_Operation, restore)
