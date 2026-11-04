"""Prevent late local/preview publication across claimed-upload cleanup."""

from __future__ import annotations

import asyncio
import copy
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, TypeVar, cast

from filelock import Timeout
from sqlalchemy.engine import Connection
from sqlalchemy.orm import object_session, sessionmaker
from sqlalchemy.orm.exc import UnmappedInstanceError

from ...core.tools.core.RAG_tools.storage.file_reference import (
    _drain,
    file_cleanup_lock,
    has_cleanup_fence,
)
from ..models.database import get_optional_session_local, release_db_connection_if_clean
from ..models.uploaded_file import UploadedFile
from ..models.uploaded_file_cleanup_fence import UploadedFileCleanupFence

_Operation = TypeVar("_Operation", bound=Callable[..., Any])
_COPY_FIELDS = (
    "user_id",
    "file_id",
    "filename",
    "storage_path",
    "storage_key",
    "storage_status",
    "checksum",
    "etag",
    "storage_backend",
    "storage_uri",
)


class FilePublicationUnavailable(FileNotFoundError):
    """A claim or changed generation forbids publication and local fallback."""


def _validate_publication(file_id: str, sessions: Any, snapshot: Any = None) -> None:
    if sessions is None:
        if has_cleanup_fence(file_id):
            raise FilePublicationUnavailable("File is unavailable")
        return
    with sessions() as db:
        current = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
        if current is None:
            retired = db.get(UploadedFileCleanupFence, file_id) is not None
            if (
                retired
                or (snapshot is not None and snapshot.id is not None)
                or has_cleanup_fence(file_id)
            ):
                raise FilePublicationUnavailable("File is unavailable")
            return
        if current.storage_status not in {"available", "legacy"}:
            raise FilePublicationUnavailable("File is unavailable")
        if snapshot is not None and any(
            getattr(current, field) != getattr(snapshot, field)
            for field in (
                "id",
                "user_id",
                "filename",
                "storage_path",
                "storage_key",
                "storage_backend",
                "storage_uri",
                "checksum",
                "etag",
            )
            if snapshot.id is not None or getattr(snapshot, field) is not None
        ):
            raise FilePublicationUnavailable("File generation is unavailable")


def _prepare_copy(ref: Any) -> Any:
    row_id = getattr(ref.record, "id", None)
    if row_id is None:
        row_id = getattr(ref.record, "row_id", None)
    snapshot = SimpleNamespace(
        id=row_id, **{field: getattr(ref.record, field, None) for field in _COPY_FIELDS}
    )
    try:
        db = object_session(ref.record)
    except UnmappedInstanceError:
        db = None
    sessions = getattr(ref, "_cleanup_sessions", get_optional_session_local())
    read_only = getattr(ref, "_cleanup_read_only", False)
    if db is not None:
        bind = db.get_bind()
        if isinstance(bind, Connection):
            bind = bind.engine
        sessions = sessionmaker(bind=bind)
        read_only = not release_db_connection_if_clean(db)
    clone = copy.copy(ref)
    clone.record = snapshot
    clone._cleanup_sessions = sessions
    clone._cleanup_read_only = read_only
    return clone


async def async_managed_copy(
    ref: Any, *, restore_local: bool = False, allow_existing_local: bool = True
) -> Path:
    """Snapshot and release a clean request session before off-loop storage work."""
    clone = _prepare_copy(ref)
    operation = (
        clone.ensure_local
        if restore_local
        else lambda: clone.materialize(allow_existing_local=allow_existing_local)
    )
    return cast(Path, await _drain(asyncio.to_thread(operation)))


def preview_cache_hit(path: Path, cached: Path, file_id: str | None) -> Path | None:
    """Read an existing cache without serializing readers or publishing bytes."""
    if file_id:
        _validate_publication(file_id, get_optional_session_local())
    try:
        if cached.is_file() and cached.stat().st_mtime >= path.stat().st_mtime:
            return cached
    except OSError:
        pass
    return None


def guard_managed_copy_publication(operation: _Operation) -> _Operation:
    @wraps(operation)
    def publish(self: Any, *args: Any, **kwargs: Any) -> Any:
        clone = _prepare_copy(self)
        sessions = clone._cleanup_sessions
        read_only = clone._cleanup_read_only
        snapshot = clone.record
        if kwargs.get("allow_existing_local", True):
            _validate_publication(str(snapshot.file_id), sessions, snapshot)
            if clone.local_path.is_file():
                return cast(Any, clone.local_path)
        if read_only:
            from .managed_file_ref import DurableStorageOperationError

            raise DurableStorageOperationError(
                "Local publication requires a clean transaction",
                storage_key=snapshot.storage_key,
            )
        if operation.__name__ == "materialize" and clone.has_durable_object:
            from .managed_file_ref import (
                NAMESPACE_AUTHORITY_ERRORS,
                DurableObjectIntegrityError,
                DurableStorageOperationError,
            )

            _validate_publication(str(snapshot.file_id), sessions, snapshot)
            try:
                cache_locator = getattr(
                    clone._bound_storage(), "materialized_path", None
                )
                cached = (
                    cache_locator(clone.storage_key, clone.filename)
                    if cache_locator is not None
                    else None
                )
                if cached is not None and cached.is_file():
                    try:
                        clone._verify_content_checksum(cached)
                    except DurableObjectIntegrityError:
                        pass
                    else:
                        _validate_publication(str(snapshot.file_id), sessions, snapshot)
                        return cast(Any, cached)
            except NAMESPACE_AUTHORITY_ERRORS:
                raise
            except FilePublicationUnavailable:
                raise
            except OSError:
                # A disappearing cache is a miss; publication re-checks under lock.
                pass
            except Exception as exc:
                raise DurableStorageOperationError(
                    "Failed to materialize durable object",
                    storage_key=snapshot.storage_key,
                ) from exc
        try:
            with file_cleanup_lock(str(snapshot.file_id)):
                _validate_publication(str(snapshot.file_id), sessions, snapshot)
                return operation(clone, *args, **kwargs)
        except Timeout as exc:
            from .managed_file_ref import DurableStorageOperationError

            raise DurableStorageOperationError(
                "File publication is temporarily unavailable",
                storage_key=snapshot.storage_key,
            ) from exc

    return cast(_Operation, publish)


@contextmanager
def _preview_guard(file_id: str | None) -> Iterator[None]:
    if not file_id:
        yield
        return
    with file_cleanup_lock(file_id):
        _validate_publication(file_id, get_optional_session_local())
        yield


def guard_preview_publication(operation: _Operation) -> _Operation:
    if asyncio.iscoroutinefunction(operation):

        @wraps(operation)
        async def asynchronous(path: Any, file_id: str | None = None) -> Any:
            async def publish() -> Any:
                guard = _preview_guard(file_id)
                await asyncio.to_thread(guard.__enter__)
                try:
                    return await operation(path, file_id)
                finally:
                    await asyncio.to_thread(guard.__exit__, None, None, None)

            try:
                return await _drain(publish())
            except Timeout:
                # The async converter already uses None for retryable failure.
                return None

        return cast(_Operation, asynchronous)

    @wraps(operation)
    def synchronous(path: Any, file_id: str | None = None) -> Any:
        try:
            with _preview_guard(file_id):
                return operation(path, file_id)
        except Timeout as exc:
            from .managed_file_ref import DurableStorageOperationError

            raise DurableStorageOperationError(
                "File preview is temporarily unavailable", storage_key=None
            ) from exc

    return cast(_Operation, synchronous)
