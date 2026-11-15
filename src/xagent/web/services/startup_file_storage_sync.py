from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock, Timeout
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session, sessionmaker

from ...core.file_storage import (
    FsspecFileStorage,
    ScopedFileStorage,
    get_file_storage_backend,
    get_user_file_storage,
)
from ...core.file_storage.keys import build_upload_storage_key
from ..models.database import release_db_connection_if_clean
from ..models.uploaded_file import UploadedFile
from .managed_file_ref import (
    DurableStorageOperationError,
    ManagedFileRef,
    log_durable_storage_fault,
)
from .uploaded_file_store import (
    StagedUploadedFile,
    UploadedFileStore,
    UploadedFileVersionConflict,
    UploadedFileVersionSnapshot,
    snapshot_uploaded_file_version,
)

logger = logging.getLogger(__name__)

_sync_lock = threading.Lock()
_FILE_LOCK_RETRY_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class StartupFileStorageSyncResult:
    scanned: int = 0
    already_present: int = 0
    uploaded: int = 0
    skipped_missing_local: int = 0
    skipped_backend: int = 0
    failed: int = 0
    locked: bool = False


@dataclass
class _StartupFileStorageCandidate:
    """Detached mutable record used only during one storage reconciliation."""

    id: int
    file_id: str
    user_id: int
    task_id: int | None
    filename: str
    storage_path: str
    storage_backend: str | None
    storage_key: str | None
    storage_uri: str | None
    checksum: str | None
    etag: str | None
    workspace_relative_path: str | None
    workspace_category: str | None
    storage_status: str
    mime_type: str | None
    file_size: int

    @classmethod
    def from_record(cls, record: UploadedFile) -> "_StartupFileStorageCandidate":
        return cls(
            id=int(record.id),
            file_id=str(record.file_id),
            user_id=int(record.user_id),
            task_id=int(record.task_id) if record.task_id is not None else None,
            filename=str(record.filename),
            storage_path=str(record.storage_path),
            storage_backend=(
                str(record.storage_backend)
                if record.storage_backend is not None
                else None
            ),
            storage_key=(
                str(record.storage_key) if record.storage_key is not None else None
            ),
            storage_uri=(
                str(record.storage_uri) if record.storage_uri is not None else None
            ),
            checksum=str(record.checksum) if record.checksum is not None else None,
            etag=str(record.etag) if record.etag is not None else None,
            workspace_relative_path=(
                str(record.workspace_relative_path)
                if record.workspace_relative_path is not None
                else None
            ),
            workspace_category=(
                str(record.workspace_category)
                if record.workspace_category is not None
                else None
            ),
            storage_status=str(record.storage_status),
            mime_type=(str(record.mime_type) if record.mime_type is not None else None),
            file_size=int(record.file_size or 0),
        )

    def to_staged(self) -> StagedUploadedFile:
        storage_key = str(self.storage_key or "").strip()
        checksum = str(self.checksum or "").strip()
        if self.storage_status != "available" or not storage_key or not checksum:
            raise ValueError("Startup sync did not produce complete durable metadata")
        return StagedUploadedFile(
            file_id=self.file_id,
            user_id=self.user_id,
            task_id=self.task_id,
            filename=self.filename,
            storage_path=self.storage_path,
            storage_backend=self.storage_backend,
            storage_key=storage_key,
            storage_uri=self.storage_uri,
            checksum=checksum,
            etag=self.etag,
            workspace_relative_path=self.workspace_relative_path,
            workspace_category=self.workspace_category,
            mime_type=self.mime_type,
            file_size=self.file_size,
        )


_StartupSyncCursor = tuple[int, int]
_SessionFactory = Callable[[], Session]


class StartupSyncCancelled(RuntimeError):
    """The sync was told to stop (deadline exceeded or process shutdown)."""


class StartupSyncResume:
    """Where the last attempt got to, kept across the retry loop's attempts.

    ``cursor`` is the last candidate that finished without a failure; a retry
    continues after it instead of rescanning every user. It never moves past a
    failed candidate, so that candidate is attempted again.
    """

    def __init__(self) -> None:
        self.cursor: tuple[int, int] | None = None


class StartupSyncControl:
    """Cooperative cancel flag, progress marker and resume point for one attempt.

    The sync thread calls :meth:`checkpoint` before each file's storage I/O
    and between batches, so a watchdog can stop it at the next boundary and
    can tell a long but progressing scan from a stalled one
    (:attr:`last_progress`). A single storage or database call that never
    returns cannot be interrupted from here; the watchdog's thread dump is what
    makes that case diagnosable.
    """

    def __init__(self, resume: StartupSyncResume | None = None) -> None:
        self.cancel = threading.Event()
        self.scanned = 0
        self.phase = "starting"
        self.resume = resume if resume is not None else StartupSyncResume()
        self.last_progress = time.monotonic()

    def checkpoint(self, phase: str | None = None, *, progress: bool = True) -> None:
        if phase is not None:
            self.phase = phase
        if self.cancel.is_set():
            raise StartupSyncCancelled("Startup file storage sync was cancelled")
        if progress:
            self.last_progress = time.monotonic()


@dataclass(frozen=True)
class StartupFileStoragePreflight:
    """Outcome of the read-only pass: does any registered file need a write?"""

    scanned: int = 0
    needs_write: int = 0
    skipped_missing_local: int = 0
    skipped_backend: bool = False

    @property
    def clean(self) -> bool:
        return self.needs_write == 0


def preflight_registered_files(
    *,
    storage: FsspecFileStorage | Any | None = None,
    batch_size: int = 500,
    session_factory: _SessionFactory | None = None,
    control: StartupSyncControl | None = None,
) -> StartupFileStoragePreflight:
    """Classify every registered file without writing storage or the database.

    Uses the same candidate scan and per-user listing as the real sync and the
    same "already durable" test, but never adopts, uploads or persists. A
    standby runs it before activation so activation can skip the write pass
    when nothing needs writing.
    """
    backend = _detect_backend(storage)
    if backend != "s3":
        return StartupFileStoragePreflight(skipped_backend=True)

    factory = session_factory or _resolve_session_factory(None, None)
    scanned = 0
    needs_write = 0
    skipped_missing_local = 0
    cursor: _StartupSyncCursor | None = None
    current_user_id: int | None = None
    remote_objects: dict[str, Any] = {}
    user_storage: ScopedFileStorage | None = None
    while True:
        if control is not None:
            control.checkpoint("preflight scan")
        candidates, next_cursor = _load_startup_sync_candidates(
            factory, after=cursor, limit=max(1, int(batch_size))
        )
        if not candidates:
            break
        for candidate, _version in candidates:
            if control is not None:
                control.checkpoint()  # before this candidate's storage I/O
            scanned += 1
            if candidate.user_id != current_user_id or user_storage is None:
                current_user_id = candidate.user_id
                user_storage = _user_scoped_storage(storage, candidate.user_id)
                remote_objects = _list_remote_objects_for_user(user_storage)
            key = _expected_storage_key(candidate)
            if key in remote_objects:
                if not _has_complete_durable_metadata(candidate):
                    needs_write += 1  # adoption persists the row
                continue
            local_path = Path(candidate.storage_path)
            if not local_path.exists() or not local_path.is_file():
                skipped_missing_local += 1
                continue
            needs_write += 1
        cursor = next_cursor
        if control is not None:
            control.scanned = scanned
    result = StartupFileStoragePreflight(
        scanned=scanned,
        needs_write=needs_write,
        skipped_missing_local=skipped_missing_local,
    )
    logger.info(
        "Startup file storage preflight complete: scanned=%s needs_write=%s "
        "skipped_missing_local=%s",
        result.scanned,
        result.needs_write,
        result.skipped_missing_local,
    )
    return result


def sync_registered_files_to_durable_storage(
    db: Session | None = None,
    *,
    storage: FsspecFileStorage | Any | None = None,
    batch_size: int = 500,
    session_factory: _SessionFactory | None = None,
    control: StartupSyncControl | None = None,
) -> StartupFileStorageSyncResult:
    """Reconcile DB registrations without holding a connection during storage I/O.

    ``db`` remains a compatibility input for existing callers. A clean caller
    transaction is rolled back to release its pooled connection before a new
    short-session factory is derived; a session with pending writes is rejected
    rather than carried across storage I/O. New runtime callers should pass
    neither argument and use the configured session factory.
    """
    backend = _detect_backend(storage)

    if backend != "s3":
        logger.info(
            "Skipping startup file storage sync for non-S3 backend: %s",
            backend or "unknown",
        )
        return StartupFileStorageSyncResult(skipped_backend=1)

    if not _sync_lock.acquire(blocking=False):
        logger.info("Startup file storage sync is already running in this process")
        return StartupFileStorageSyncResult(locked=True)

    file_lock = None
    try:
        SessionLocal = _resolve_session_factory(db, session_factory)
        file_lock = _acquire_file_lock_after_contention(control)
        return _sync_registered_files(
            SessionLocal,
            storage=storage,
            batch_size=batch_size,
            control=control,
        )
    finally:
        if file_lock is not None:
            _release_file_lock(file_lock)
        _sync_lock.release()


def _resolve_session_factory(
    db: Session | None,
    session_factory: _SessionFactory | None,
) -> _SessionFactory:
    if db is not None:
        if not release_db_connection_if_clean(db):
            raise RuntimeError(
                "Cannot run startup file storage sync while the caller "
                "database session has pending writes"
            )
    if session_factory is not None:
        return session_factory
    if db is not None:
        return sessionmaker(bind=db.get_bind())
    from ..models.database import get_session_local

    return get_session_local()


def _acquire_file_lock_after_contention(
    control: StartupSyncControl | None = None,
) -> Any:
    file_lock = _acquire_file_lock()
    while file_lock is None:
        if control is not None:
            control.checkpoint("waiting for sync lock holder", progress=False)
        logger.info(
            "Startup file storage sync is already running in another process; waiting"
        )
        _wait_for_lock_holder()
        file_lock = _acquire_file_lock()
    return file_lock


def _wait_for_lock_holder() -> None:
    time.sleep(_FILE_LOCK_RETRY_INTERVAL_SECONDS)


def _detect_backend(storage: FsspecFileStorage | Any | None) -> str:
    if storage is None:
        return get_file_storage_backend()
    backend = str(getattr(storage, "backend", "") or "")
    if not backend:
        backend = str(getattr(storage, "_backend", "") or "")
    return backend


def _user_scoped_storage(
    storage: FsspecFileStorage | Any | None, user_id: int
) -> ScopedFileStorage:
    if storage is None:
        return get_user_file_storage(user_id)
    return ScopedFileStorage(storage=storage, prefix=f"users/{user_id}")


def _sync_registered_files(
    session_factory: _SessionFactory,
    *,
    storage: FsspecFileStorage | Any | None,
    batch_size: int,
    control: StartupSyncControl | None = None,
) -> StartupFileStorageSyncResult:
    scanned = 0
    already_present = 0
    uploaded = 0
    skipped_missing_local = 0
    failed = 0

    resolved_batch_size = max(1, int(batch_size))
    cursor: _StartupSyncCursor | None = (
        control.resume.cursor if control is not None else None
    )
    previous: _StartupFileStorageCandidate | None = None
    current_user_id: int | None = None
    user_storage: ScopedFileStorage | None = None
    remote_objects: dict[str, Any] = {}
    while True:
        if control is not None:
            control.checkpoint("sync scan")
        candidates, next_cursor = _load_startup_sync_candidates(
            session_factory,
            after=cursor,
            limit=resolved_batch_size,
        )
        if not candidates:
            break
        for candidate, expected_version in candidates:
            if control is not None:
                # The previous candidate finished cleanly: a retry resumes
                # after it. Frozen once any file failed, so that file (and
                # whatever follows) is attempted again.
                if previous is not None and failed == 0:
                    control.resume.cursor = (previous.user_id, previous.id)
                control.checkpoint()  # before this candidate's storage I/O
                control.scanned = scanned + 1
            previous = candidate
            scanned += 1
            user_id = candidate.user_id
            if user_id != current_user_id or user_storage is None:
                current_user_id = user_id
                user_storage = _user_scoped_storage(storage, user_id)
                remote_objects = _list_remote_objects_for_user(user_storage)

            expected_storage_key = _expected_storage_key(candidate)
            remote_object = remote_objects.get(expected_storage_key)
            if remote_object is not None:
                if not _has_complete_durable_metadata(candidate):
                    try:
                        adopt_result = ManagedFileRef(
                            candidate, storage=user_storage
                        ).adopt_existing_object(expected_storage_key)
                    except DurableStorageOperationError as exc:
                        # Named before the bare arm below so an adoption failure
                        # gets the classified provider fields, not just a
                        # traceback -- the stat/content-hash raise sites in
                        # ``adopt_existing_object`` are the ones #1467 asked to
                        # cover, and this is their only caller.
                        failed += 1
                        log_durable_storage_fault(
                            logger,
                            "startup durable adoption",
                            exc,
                            file_id=candidate.file_id,
                            storage_key=expected_storage_key,
                        )
                        continue
                    except Exception:
                        failed += 1
                        logger.exception(
                            "Failed startup durable adoption for file_id=%s key=%s",
                            candidate.file_id,
                            expected_storage_key,
                        )
                        continue
                    if adopt_result == "missing":
                        local_path = Path(candidate.storage_path)
                        if not local_path.exists() or not local_path.is_file():
                            skipped_missing_local += 1
                        else:
                            failed += 1
                        continue
                    if not _persist_startup_sync_candidate(
                        session_factory,
                        candidate,
                        expected=expected_version,
                    ):
                        failed += 1
                        continue
                    if adopt_result == "uploaded":
                        uploaded += 1
                already_present += 1
                continue

            local_path = Path(candidate.storage_path)
            if not local_path.exists() or not local_path.is_file():
                skipped_missing_local += 1
                logger.warning(
                    "Skipping startup durable sync for missing local file: "
                    "file_id=%s path=%s",
                    candidate.file_id,
                    local_path,
                )
                continue

            try:
                stored_object = ManagedFileRef(
                    candidate, storage=user_storage
                ).sync_to_durable(
                    storage_key=expected_storage_key,
                    mime_type=candidate.mime_type,
                )
            except DurableStorageOperationError as exc:
                # ``sync_to_durable`` wraps the provider fault, so this is a
                # reporting site rather than a raw boundary: it gets the
                # classified fields the adoption arm above already gets. Still
                # counted and skipped -- one file failing to sync at startup
                # must not stop the rest.
                failed += 1
                log_durable_storage_fault(
                    logger,
                    "startup durable sync",
                    exc,
                    file_id=candidate.file_id,
                    storage_key=expected_storage_key,
                )
                continue
            except Exception:
                failed += 1
                logger.exception(
                    "Failed startup durable sync for file_id=%s path=%s key=%s",
                    candidate.file_id,
                    local_path,
                    expected_storage_key,
                )
                continue
            if not _persist_startup_sync_candidate(
                session_factory,
                candidate,
                expected=expected_version,
            ):
                failed += 1
                continue
            remote_objects[expected_storage_key] = stored_object
            uploaded += 1

        cursor = next_cursor
        if control is not None and failed == 0 and next_cursor is not None:
            control.resume.cursor = next_cursor

    result = StartupFileStorageSyncResult(
        scanned=scanned,
        already_present=already_present,
        uploaded=uploaded,
        skipped_missing_local=skipped_missing_local,
        failed=failed,
    )
    logger.info(
        "Startup file storage sync complete: scanned=%s already_present=%s uploaded=%s skipped_missing_local=%s failed=%s",
        result.scanned,
        result.already_present,
        result.uploaded,
        result.skipped_missing_local,
        result.failed,
    )
    return result


def _load_startup_sync_candidates(
    session_factory: _SessionFactory,
    *,
    after: _StartupSyncCursor | None,
    limit: int,
) -> tuple[
    tuple[
        tuple[_StartupFileStorageCandidate, UploadedFileVersionSnapshot],
        ...,
    ],
    _StartupSyncCursor | None,
]:
    with session_factory() as db:
        query = db.query(UploadedFile).filter(
            UploadedFile.storage_status != "compensating"
        )
        if after is not None:
            after_user_id, after_row_id = after
            query = query.filter(
                or_(
                    UploadedFile.user_id > after_user_id,
                    and_(
                        UploadedFile.user_id == after_user_id,
                        UploadedFile.id > after_row_id,
                    ),
                )
            )
        records = (
            query.order_by(UploadedFile.user_id.asc(), UploadedFile.id.asc())
            .limit(limit)
            .all()
        )
        candidates = tuple(
            (
                _StartupFileStorageCandidate.from_record(record),
                snapshot_uploaded_file_version(record),
            )
            for record in records
        )
    next_cursor = (
        (candidates[-1][0].user_id, candidates[-1][0].id) if candidates else None
    )
    return candidates, next_cursor


def _persist_startup_sync_candidate(
    session_factory: _SessionFactory,
    candidate: _StartupFileStorageCandidate,
    *,
    expected: UploadedFileVersionSnapshot,
) -> bool:
    with session_factory() as db:
        try:
            UploadedFileStore(db).upsert_already_durable(
                candidate.to_staged(),
                expected=expected,
            )
            db.commit()
            return True
        except UploadedFileVersionConflict:
            db.rollback()
            logger.warning(
                "Uploaded file %s changed during startup storage sync; "
                "retrying on the next startup pass",
                candidate.file_id,
            )
            return False
        except Exception:
            db.rollback()
            raise


def _expected_storage_key(record: UploadedFile | _StartupFileStorageCandidate) -> str:
    existing_key = str(getattr(record, "storage_key", "") or "").strip()
    if existing_key:
        return existing_key
    return build_upload_storage_key(
        int(getattr(record, "user_id")),
        str(getattr(record, "file_id")),
        str(getattr(record, "filename")),
    )


def _has_complete_durable_metadata(
    record: UploadedFile | _StartupFileStorageCandidate,
) -> bool:
    return bool(
        getattr(record, "storage_key", None)
        and getattr(record, "storage_backend", None) == "s3"
        and getattr(record, "storage_status", None) == "available"
        and getattr(record, "checksum", None)
    )


def _list_remote_objects_for_user(storage: ScopedFileStorage) -> dict[str, Any]:
    return {stored.key: stored for stored in storage.list(storage.prefix)}


def _get_lock_file_path() -> str:
    return os.environ.get(
        "XAGENT_FILE_STORAGE_STARTUP_SYNC_LOCK_FILE",
        os.path.join(tempfile.gettempdir(), "xagent_file_storage_startup_sync.lock"),
    )


def _acquire_file_lock() -> Any | None:
    lock_path = _get_lock_file_path()
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    try:
        lock = FileLock(lock_path, timeout=0)
        lock.acquire()
        Path(lock_path).write_text(str(os.getpid()), encoding="utf-8")
        return lock
    except Timeout:
        return None


def _release_file_lock(lock_file: Any) -> None:
    lock_file.release()
