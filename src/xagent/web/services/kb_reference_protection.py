"""Web metadata checks inside the shared KB reference publication lock."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar, cast
from uuid import uuid4

from filelock import Timeout
from sqlalchemy import event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from ...core.tools.core.RAG_tools.storage.file_reference import (
    clear_cleanup_fence,
    file_cleanup_lock,
    file_reference_lock,
    has_cleanup_fence,
    persist_cleanup_fence,
    set_file_reference_validator,
)
from ..models.database import get_optional_session_local, release_db_connection_if_clean
from ..models.kb_ingest_target import KBIngestTarget
from ..models.uploaded_file import UploadedFile
from ..models.uploaded_file_cleanup_fence import UploadedFileCleanupFence

logger = logging.getLogger(__name__)

ADMISSION_BUSY_MESSAGE = "KB source is busy; retry the upload"
ADMISSION_CONFLICT_MESSAGE = "KB source is unavailable or the ingest was superseded"
_ADMISSION_CANCELLED_MESSAGE = "KB ingest admission was cancelled"


@dataclass(frozen=True)
class _IngestGeneration:
    payload: dict[str, Any]
    sessions: sessionmaker[Session]


_ingest_generation: ContextVar[_IngestGeneration | None] = ContextVar(
    "kb_reference_ingest_generation", default=None
)
_Operation = TypeVar("_Operation", bound=Callable[..., Any])


class FileReferenceConflict(RuntimeError):
    """The source was claimed, or the ingest generation no longer owns it."""


def select_new_ingest_file_id(db: Session, proposed: str) -> str:
    """Preserve a missing upload's deterministic ID unless cleanup retired it."""
    if db.get(UploadedFileCleanupFence, proposed) is not None or has_cleanup_fence(
        proposed
    ):
        return str(uuid4())
    return proposed


def validate_file_references(db: Session, file_ids: tuple[str, ...]) -> None:
    rows = db.query(UploadedFile).filter(UploadedFile.file_id.in_(file_ids)).all()
    present = {str(row.file_id) for row in rows}
    if any(str(row.storage_status) not in {"available", "legacy"} for row in rows):
        raise FileReferenceConflict("Uploaded file is not available for KB reference")
    missing = set(file_ids) - present
    if (
        missing
        and db.query(UploadedFileCleanupFence)
        .filter(UploadedFileCleanupFence.file_id.in_(missing))
        .first()
        is not None
    ):
        raise FileReferenceConflict(
            "Uploaded file has already been claimed for cleanup"
        )
    context = _ingest_generation.get()
    generation = context.payload if context else None
    if generation and generation.get("file_id") in file_ids:
        target = (
            db.query(KBIngestTarget)
            .filter(
                KBIngestTarget.user_id == int(generation["user_id"]),
                KBIngestTarget.collection == str(generation["collection"]),
                KBIngestTarget.target_path == str(generation["target_path"]),
                KBIngestTarget.latest_generation_id == str(generation["generation_id"]),
                KBIngestTarget.deleted_at.is_(None),
            )
            .first()
        )
        if target is None:
            raise FileReferenceConflict("KB ingest generation is no longer active")
    # A live upload is authoritative after rollback or fresh durable publication.
    for file_id in present:
        clear_cleanup_fence(file_id)
    if any(has_cleanup_fence(file_id) for file_id in missing):
        raise FileReferenceConflict("Source identity has an unresolved cleanup fence")


def install_file_reference_validator() -> None:
    def validate(file_ids: tuple[str, ...]) -> None:
        context = _ingest_generation.get()
        factory = context.sessions if context else get_optional_session_local()
        if factory is not None:
            with factory() as db:
                validate_file_references(db, file_ids)

    set_file_reference_validator(validate)


def record_cleanup_fence(db: Session, file_id: str) -> None:
    """The caller commits this identity together with its successful claim."""
    if db.get(UploadedFileCleanupFence, file_id) is None:
        db.add(UploadedFileCleanupFence(file_id=file_id))
        db.flush()
    persist_cleanup_fence(file_id)


def notify_file_publication(db: Session, file_id: str) -> None:
    """Reconcile a standalone fence only after new metadata actually commits."""
    if has_cleanup_fence(file_id):
        db.info.setdefault("kb_published_files", set()).add(file_id)


@event.listens_for(Session, "after_commit")
def _publication_committed(db: Session) -> None:
    if db.get_nested_transaction() is None:
        db.info["kb_publication_committed"] = True


@event.listens_for(Session, "after_transaction_end")
def _reconcile_publication(db: Session, transaction: Any) -> None:
    if transaction.parent is not None:
        return
    file_ids = db.info.pop("kb_published_files", set())
    committed = db.info.pop("kb_publication_committed", False)
    if not file_ids or not committed:
        return
    # Root transaction end has returned the caller's connection to the pool.
    bind = db.get_bind()
    if isinstance(bind, Connection):
        bind = bind.engine
    try:
        with file_reference_lock(file_ids), sessionmaker(bind=bind)() as check:
            validate_file_references(check, tuple(file_ids))
    except Exception:
        logger.warning("Deferred publication fence reconciliation", exc_info=True)


@contextmanager
def unreferenced_file(db: Session, file_id: str) -> Iterator[bool]:
    """Fail closed before a destructive action, and fence concurrent publication."""
    from .kb_file_service import find_referenced_file_ids

    if not release_db_connection_if_clean(db):
        raise RuntimeError("Cleanup requires a clean caller transaction")
    with file_reference_lock([file_id]):
        pending = (
            db.query(KBIngestTarget.id)
            .filter(
                KBIngestTarget.file_id == file_id,
                KBIngestTarget.deleted_at.is_(None),
            )
            .first()
            is not None
        )
        db.rollback()
        referenced = bool(find_referenced_file_ids([file_id]))
        yield not pending and not referenced


def guard_cleanup_claim(operation: _Operation) -> _Operation:
    @wraps(operation)
    def claim(db: Session, candidate: Any) -> Any:
        if not release_db_connection_if_clean(db):
            raise RuntimeError("Cleanup requires a clean caller transaction")
        with (
            file_cleanup_lock(candidate.file_id),
            unreferenced_file(db, candidate.file_id) as eligible,
        ):
            if not eligible:
                return None
            token = operation(db, candidate)
            if token is not None:
                record_cleanup_fence(db, candidate.file_id)
                db.commit()
            return token

    return cast(_Operation, claim)


def guard_ingest_admission(operation: _Operation) -> _Operation:
    @wraps(operation)
    def admit(db: Session, **kwargs: Any) -> Any:
        if not release_db_connection_if_clean(db):
            raise RuntimeError("KB admission requires a clean caller transaction")
        try:
            with file_reference_lock([kwargs["file_id"]]):
                validate_file_references(db, (kwargs["file_id"],))
                return operation(db, **kwargs)
        except Exception as exc:
            from ..models.background_job import BackgroundJob
            from .background_jobs import mark_job_failed
            from .kb_ingest_targets import release_kb_ingest_target_generation

            if isinstance(exc, Timeout):
                error_message = ADMISSION_BUSY_MESSAGE
            elif isinstance(exc, FileReferenceConflict):
                error_message = ADMISSION_CONFLICT_MESSAGE
            else:
                error_message = str(exc)
            try:
                db.rollback()
                release_kb_ingest_target_generation(
                    db,
                    user_id=kwargs["user_id"],
                    collection=kwargs["collection"],
                    target_path=kwargs["target_path"],
                    generation_id=kwargs["generation_id"],
                )
                job = db.get(BackgroundJob, kwargs["job_id"])
                if job is not None:
                    mark_job_failed(db, job, error_message=error_message)
            except Exception:
                logger.warning("Failed to settle rejected KB admission", exc_info=True)
            raise

    return cast(_Operation, admit)


async def async_admit_kb_ingest_target(db: Session, **kwargs: Any) -> Any:
    """Run target admission off-loop in a worker-owned short SQL Session."""
    if not release_db_connection_if_clean(db):
        raise RuntimeError("KB admission requires a clean caller transaction")
    bind = db.get_bind()
    if isinstance(bind, Connection):
        bind = bind.engine

    def admit() -> Any:
        from .kb_ingest_targets import admit_kb_ingest_target

        with sessionmaker(bind=bind, autoflush=False)() as worker_db:
            return admit_kb_ingest_target(worker_db, **kwargs)

    # A cancelled request must not remove its staged source while admission can
    # still commit in a detached worker. Drain the worker before propagating the
    # cancellation, matching the native async document-publication boundary.
    task = asyncio.create_task(run_in_threadpool(admit))
    cancellation: asyncio.CancelledError | None = None
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
        try:
            task.result()
        except Exception as exc:
            raise cancellation from exc

        def settle_cancelled_admission() -> None:
            from ..models.background_job import BackgroundJob
            from .background_jobs import mark_job_failed
            from .kb_ingest_targets import release_kb_ingest_target_generation

            with sessionmaker(bind=bind, autoflush=False)() as worker_db:
                try:
                    release_kb_ingest_target_generation(
                        worker_db,
                        user_id=kwargs["user_id"],
                        collection=kwargs["collection"],
                        target_path=kwargs["target_path"],
                        generation_id=kwargs["generation_id"],
                    )
                except Exception:
                    worker_db.rollback()
                    logger.warning(
                        "Failed to release cancelled KB ingest admission",
                        exc_info=True,
                    )
                job = worker_db.get(BackgroundJob, kwargs["job_id"])
                if job is not None:
                    mark_job_failed(
                        worker_db,
                        job,
                        error_message=_ADMISSION_CANCELLED_MESSAGE,
                    )

        settlement = asyncio.create_task(run_in_threadpool(settle_cancelled_admission))
        while True:
            try:
                await asyncio.shield(settlement)
                break
            except asyncio.CancelledError:
                if settlement.done():
                    logger.warning("Cancelled KB admission settlement during shutdown")
                    break
                continue
            except Exception:
                logger.warning(
                    "Failed to settle cancelled KB ingest admission", exc_info=True
                )
                break
        raise cancellation
    return result


def guard_upload_compensation(operation: _Operation) -> _Operation:
    @wraps(operation)
    def compensate(claims: Any) -> None:
        from .uploaded_file_store import get_session_local

        unique = tuple(
            dict.fromkeys(
                claim
                for claim in claims
                if str(claim.file_id).strip()
                and str(claim.expected_storage_key).strip()
            )
        )
        admitted: set[str] = set()
        with ExitStack() as claim_locks:
            for file_id in sorted({claim.file_id for claim in unique}):
                with ExitStack() as candidate_lock:
                    candidate_lock.enter_context(file_cleanup_lock(file_id))
                    with get_session_local()() as db:
                        eligible = candidate_lock.enter_context(
                            unreferenced_file(db, file_id)
                        )
                    if not eligible:
                        logger.warning(
                            "Skipping registered upload compensation for "
                            "referenced KB source %s",
                            file_id,
                        )
                        continue
                    claim_locks.enter_context(candidate_lock.pop_all())
                    admitted.add(file_id)
            try:
                operation(
                    [claim for claim in unique if claim.file_id in admitted],
                    _release_claim_locks=claim_locks.close,
                )
            finally:
                claim_locks.close()

    return cast(_Operation, compensate)


def coordinate_ingest_job(operation: _Operation) -> _Operation:
    @wraps(operation)
    def ingest(db: Session, job: Any) -> Any:
        from ..jobs.exceptions import BackgroundJobHandlerError
        from .kb_ingest_targets import release_kb_ingest_target_generation

        payload = dict(job.payload or {})
        final_attempt = int(job.attempts or 0) >= int(job.max_attempts or 1)
        bind = db.get_bind()
        if isinstance(bind, Connection):
            bind = bind.engine
        token = _ingest_generation.set(
            _IngestGeneration(payload, sessionmaker(bind=bind))
            if payload.get("generation_id")
            else None
        )
        release = False
        try:
            result = operation(db, job)
            release = True
            return result
        except BackgroundJobHandlerError as exc:
            release = final_attempt or not exc.retryable
            raise
        except Exception:
            release = final_attempt
            raise
        finally:
            _ingest_generation.reset(token)
            if release and payload.get("generation_id"):
                try:
                    db.rollback()
                    release_kb_ingest_target_generation(
                        db,
                        user_id=int(payload["user_id"]),
                        collection=str(payload["collection"]),
                        target_path=str(payload["target_path"]),
                        generation_id=str(payload["generation_id"]),
                    )
                except Exception:
                    logger.warning(
                        "Failed to release terminal KB ingest target", exc_info=True
                    )

    return cast(_Operation, ingest)
