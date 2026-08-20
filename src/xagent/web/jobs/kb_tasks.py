from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session

from ...core.tools.core.RAG_tools.core.schemas import (
    IngestionConfig,
    IngestionResult,
    WebCrawlConfig,
)
from ...core.tools.core.RAG_tools.kb import (
    KBApiCompatibilityFacade,
    KBApiOperationResult,
    get_kb_coordinator,
)
from ...core.tools.core.RAG_tools.pipelines.document_ingestion import (
    run_document_ingestion,
)
from ...core.tools.core.RAG_tools.pipelines.web_ingestion import (
    FileHandlerResult,
    run_web_ingestion,
)
from ...core.tools.core.RAG_tools.utils.user_scope import user_scope_context
from ..config import get_upload_path
from ..models.background_job import BackgroundJob
from ..models.database import get_session_local
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..services.background_jobs import update_job_progress
from ..services.kb_ingest_targets import is_latest_kb_ingest_generation
from ..tracking.standalone_usage import usage_scope
from .exceptions import BackgroundJobHandlerError
from .progress import BackgroundJobProgressManager

logger = logging.getLogger(__name__)

_SUPERSEDED_STAGED_INGEST_MESSAGE = "KB ingest job superseded by a newer upload"


class StagedDocumentIngestSuperseded(RuntimeError):
    pass


def _get_api_compatibility_facade() -> KBApiCompatibilityFacade:
    return get_kb_coordinator().api_compatibility


def _collection_existed_before(payload: dict[str, Any]) -> bool:
    """Read the flag stamped when the job was submitted.

    It only ever narrows what a failure may clean up; the destructive paths in
    api/kb.py re-check the collection itself, because this value goes stale
    while a sibling ingest runs.
    """
    return bool(payload.get("collection_existed_before", True))


def _get_job_user(
    db: Session,
    payload: dict[str, Any],
    *,
    context: str,
) -> User | None:
    user_id = int(payload["user_id"])
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        logger.warning("Cannot %s for missing user %s", context, user_id)
    return user


def _save_job_collection_config_after_ingest(
    db: Session,
    payload: dict[str, Any],
    ingestion_config: IngestionConfig,
    *,
    context: str,
    documents_created: int,
    result_payload: dict[str, Any] | None = None,
) -> None:
    if documents_created <= 0 and _collection_existed_before(payload):
        # Nothing to publish, so a missing user is not this job's problem and
        # raising here would skip the caller's failure cleanup.
        return

    user = _get_job_user(
        db, payload, context=f"save collection config during {context}"
    )
    if user is None:
        if documents_created <= 0:
            # Nothing landed, so there is nothing this user was needed for.
            # Raising would replace the real failure — the crawl's own message,
            # retryable when the crawl genuinely failed — with a non-retryable
            # "user no longer exists" that says nothing about what went wrong.
            return
        # Returning here would mark the job SUCCEEDED with nothing published:
        # a successful import the user cannot see and has no error to act on.
        raise BackgroundJobHandlerError(
            f"Cannot publish the collection config during {context}: "
            f"user {payload.get('user_id')} no longer exists",
            result=result_payload,
            retryable=False,
        )

    from ..api.kb import CollectionConfigSaveError, _save_collection_config_after_ingest

    try:
        asyncio.run(
            _save_collection_config_after_ingest(
                collection=str(payload["collection"]),
                config_json=ingestion_config.model_dump_json(exclude_unset=True),
                user=user,
                context=context,
                documents_created=documents_created,
                collection_existed_before=_collection_existed_before(payload),
            )
        )
    except CollectionConfigSaveError as exc:
        # The documents already landed; a retry would re-run against consumed
        # staging state and end up deleting the collection. The payload has to
        # ride along, or the job record is indistinguishable from a real failure.
        raise BackgroundJobHandlerError(
            str(exc), result=result_payload, retryable=False
        ) from exc


def _cleanup_failed_job_collection_metadata(
    db: Session,
    payload: dict[str, Any],
    *,
    context: str,
    successful_documents: int = 0,
    side_effects_may_remain: bool = False,
) -> None:
    user = _get_job_user(
        db,
        payload,
        context=f"clean up failed-ingest collection metadata during {context}",
    )
    if user is None:
        return

    from ..api.kb import _cleanup_collection_metadata_after_failed_ingest

    asyncio.run(
        _cleanup_collection_metadata_after_failed_ingest(
            collection_existed_before=_collection_existed_before(payload),
            collection_name=str(payload["collection"]),
            user=user,
            context=context,
            successful_documents=successful_documents,
            side_effects_may_remain=side_effects_may_remain,
        )
    )


def _cleanup_failed_job_collection_metadata_after_api_ingest(
    db: Session,
    payload: dict[str, Any],
    *,
    api_result: KBApiOperationResult[Any],
    context: str,
    successful_documents: int | None = None,
    rollback_complete: bool | None = None,
) -> None:
    user = _get_job_user(
        db,
        payload,
        context=f"clean up failed-ingest collection metadata during {context}",
    )
    if user is None:
        return

    from ..api.kb import _cleanup_collection_metadata_after_failed_api_ingest

    asyncio.run(
        _cleanup_collection_metadata_after_failed_api_ingest(
            api_result=api_result,
            collection_existed_before=_collection_existed_before(payload),
            collection_name=str(payload["collection"]),
            user=user,
            context=context,
            successful_documents=successful_documents,
            rollback_complete=rollback_complete,
        )
    )


def handle_kb_ingest_document(db: Session, job: BackgroundJob) -> dict[str, Any]:
    payload = dict(job.payload or {})
    ingestion_config = IngestionConfig.model_validate(payload["ingestion_config"])
    file_id = payload.get("file_id")
    target_path = payload.get("target_path")
    is_staged_input = bool(target_path)
    progress_manager = BackgroundJobProgressManager(db, job)

    update_job_progress(db, job, message="Ingesting document")
    if is_staged_input and not _is_staged_document_generation_latest(db, payload):
        update_job_progress(db, job, message="Superseded by newer upload")
        return _superseded_staged_document_result(payload)

    def _assert_latest_generation() -> None:
        if is_staged_input and not _is_staged_document_generation_latest(db, payload):
            raise StagedDocumentIngestSuperseded(_SUPERSEDED_STAGED_INGEST_MESSAGE)

    try:
        # This job runs in a Celery worker with no TaskTracker, so bind the
        # usage sink here or the whole background ingestion bills nothing.
        # Synchronous throughout, so contextvars are not at risk.
        with (
            usage_scope(int(payload["user_id"])),
            user_scope_context(
                user_id=int(payload["user_id"]),
                is_admin=bool(payload.get("is_admin", False)),
            ),
        ):
            api_result = _get_api_compatibility_facade().run_with_operation_outcome(
                lambda: run_document_ingestion(
                    collection=str(payload["collection"]),
                    source_path=str(payload["source_path"]),
                    ingestion_config=ingestion_config,
                    progress_manager=progress_manager,
                    user_id=int(payload["user_id"]),
                    is_admin=bool(payload.get("is_admin", False)),
                    file_id=str(file_id) if file_id else None,
                    metadata_source_path=str(target_path) if target_path else None,
                    commit_gate=_assert_latest_generation if is_staged_input else None,
                ),
                operation_type="document_ingestion",
                collection=str(payload["collection"]),
            )
            result = api_result.result
    except StagedDocumentIngestSuperseded:
        _cleanup_failed_staged_job_collection_metadata_if_current(
            db,
            payload,
            context="background staged document superseded exception",
        )
        return _superseded_staged_document_result(payload)
    except Exception:
        if is_staged_input:
            if int(job.attempts or 0) >= int(job.max_attempts or 1):
                _cleanup_staged_document_input(payload)
            _cleanup_failed_staged_job_collection_metadata_if_current(
                db,
                payload,
                context="background staged document ingest exception",
            )
        else:
            _cleanup_failed_job_collection_metadata(
                db,
                payload,
                context="background document ingest exception",
            )
        raise

    result_payload = result.model_dump(mode="json")
    if file_id:
        result_payload["file_id"] = file_id
    if result.status in {"error", "partial"}:
        if is_staged_input:
            if _is_superseded_ingestion_result(result):
                _cleanup_failed_staged_job_collection_metadata_if_current(
                    db,
                    payload,
                    context="background staged document superseded result",
                )
                return _superseded_staged_document_result(payload)
            rollback_api_result = _rollback_failed_staged_document_ingestion_if_current(
                db,
                payload,
                result,
                api_result=api_result,
            )
            if rollback_api_result is False:
                _cleanup_failed_staged_job_collection_metadata_if_current(
                    db,
                    payload,
                    context="background stale staged document ingest",
                )
                return _superseded_staged_document_result(payload)
            api_result = rollback_api_result
            _cleanup_failed_job_collection_metadata_after_api_ingest(
                db,
                payload,
                api_result=api_result,
                context="background staged document ingest",
            )
        else:
            api_result = _rollback_failed_document_ingestion(
                db,
                payload,
                result,
                api_result=api_result,
            )
            _cleanup_failed_job_collection_metadata_after_api_ingest(
                db,
                payload,
                api_result=api_result,
                context="background document ingest",
            )
        raise BackgroundJobHandlerError(
            result.message,
            result=result_payload,
            retryable=False,
        )
    if is_staged_input:
        if not _is_staged_document_generation_latest(db, payload):
            _cleanup_failed_staged_job_collection_metadata_if_current(
                db,
                payload,
                context="background stale staged document publish",
            )
            return _superseded_staged_document_result(payload)
        try:
            file_record = _publish_staged_document_ingestion(db, payload)
            result_payload["file_id"] = str(file_record.file_id)
        except StagedDocumentIngestSuperseded:
            _cleanup_failed_staged_job_collection_metadata_if_current(
                db,
                payload,
                context="background staged document publish superseded",
            )
            return _superseded_staged_document_result(payload)
        except Exception as exc:  # noqa: BLE001
            rollback_api_result = _rollback_failed_staged_document_ingestion_if_current(
                db,
                payload,
                result,
                api_result=api_result,
            )
            if rollback_api_result is False:
                _cleanup_failed_staged_job_collection_metadata_if_current(
                    db,
                    payload,
                    context="background stale staged document publish rollback",
                )
                return _superseded_staged_document_result(payload)
            api_result = rollback_api_result
            _cleanup_failed_job_collection_metadata_after_api_ingest(
                db,
                payload,
                api_result=api_result,
                context="background staged document publish",
                successful_documents=0,
            )
            raise BackgroundJobHandlerError(
                f"Document ingestion succeeded but publishing uploaded file failed: {exc}",
                result=result_payload,
                retryable=False,
            ) from exc
    else:
        _discard_ingest_backup(payload)
    # Reaching here means exactly one document landed, so publish the config.
    _save_job_collection_config_after_ingest(
        db,
        payload,
        ingestion_config,
        context="background document ingest",
        documents_created=result.produced_documents,
        result_payload=result_payload,
    )
    return result_payload


def handle_kb_ingest_web(db: Session, job: BackgroundJob) -> dict[str, Any]:
    payload = dict(job.payload or {})
    crawl_config = WebCrawlConfig.model_validate(payload["crawl_config"])
    ingestion_config = IngestionConfig.model_validate(payload["ingestion_config"])
    user_id = int(payload["user_id"])
    is_admin = bool(payload.get("is_admin", False))
    collection = str(payload["collection"])
    processed_urls: dict[str, str] = {}

    def _progress(message: str, completed: int, total: int) -> None:
        update_job_progress(
            db,
            job,
            message=message,
            completed=completed,
            total=total,
            extra={"source": "web_ingestion"},
        )

    def _file_handler_with_db(
        temp_file_path: Path,
        title: str,
        collection_name: str,
        url: str,
    ) -> FileHandlerResult:
        SessionLocal = get_session_local()
        db_session = SessionLocal()
        try:
            return _handle_web_file(
                temp_file_path=temp_file_path,
                title=title,
                collection_name=collection_name,
                url=url,
                db_session=db_session,
                user_id=user_id,
                is_admin=is_admin,
                processed_urls=processed_urls,
            )
        finally:
            db_session.close()

    update_job_progress(db, job, message="Crawling website")
    try:
        with user_scope_context(user_id=user_id, is_admin=is_admin):
            api_result = asyncio.run(
                _get_api_compatibility_facade().run_async_with_operation_outcome(
                    lambda: run_web_ingestion(
                        collection=collection,
                        crawl_config=crawl_config,
                        ingestion_config=ingestion_config,
                        progress_callback=_progress,
                        user_id=user_id,
                        is_admin=is_admin,
                        file_handler=_file_handler_with_db,
                    ),
                    operation_type="web_ingestion",
                    collection=collection,
                )
            )
            result = api_result.result
    except Exception:
        _cleanup_failed_web_collection_metadata_if_new(db, payload)
        raise

    from ..api.kb import _demote_empty_crawl_to_error

    crawl_status = result.status
    api_result = _demote_empty_crawl_to_error(
        api_result,
        collection_existed_before=_collection_existed_before(payload),
    )
    result = api_result.result
    crawled_nothing = result.status != crawl_status
    documents_created = int(result.documents_created or 0)
    result_payload = result.model_dump(mode="json")
    # Partial crawls still leave real documents behind, so publish on any of them.
    _save_job_collection_config_after_ingest(
        db,
        payload,
        ingestion_config,
        context="background web ingest",
        documents_created=documents_created,
        result_payload=result_payload,
    )
    if result.status in {"error", "partial"}:
        _cleanup_failed_web_collection_metadata_if_new(
            db,
            payload,
            api_result=api_result,
            successful_documents=documents_created,
        )
    if result.status == "error":
        # A crawl that recorded no failures has nothing to retry: re-running it
        # re-crawls the whole site to reach the same empty result.
        raise BackgroundJobHandlerError(
            result.message,
            result=result_payload,
            retryable=not crawled_nothing,
        )
    return result_payload


def _handle_web_file(
    *,
    temp_file_path: Path,
    title: str,
    collection_name: str,
    url: str,
    db_session: Session,
    user_id: int,
    is_admin: bool,
    processed_urls: dict[str, str],
) -> FileHandlerResult:
    from ..api.kb import (
        _create_new_web_file_handler_result,
        _normalize_web_title_for_filename,
        _recreate_missing_existing_file,
        _refresh_existing_file_if_changed,
        _WebFileLock,
    )

    url_hash = hashlib.sha256(f"{collection_name}:{url}".encode()).hexdigest()[:16]
    safe_title = _normalize_web_title_for_filename(title)
    filename = f"{url_hash}_{safe_title}.md"
    lock_key = f"{user_id}:{url_hash}"

    with _WebFileLock(lock_key):
        if url_hash in processed_urls:
            existing_file_id = processed_urls[url_hash]
            existing_record = (
                db_session.query(UploadedFile)
                .filter(UploadedFile.file_id == existing_file_id)
                .first()
            )
            if existing_record:
                result = _refresh_existing_file_if_changed(
                    existing_record=existing_record,
                    temp_file_path=temp_file_path,
                    db_session=db_session,
                    user_id=user_id,
                    is_admin=is_admin,
                    collection_name=collection_name,
                    url=url,
                    filename=filename,
                    url_hash=url_hash,
                    processed_urls=processed_urls,
                    context="background-job cache",
                )
                if result is not None:
                    return result

        existing_record = (
            db_session.query(UploadedFile)
            .filter(
                UploadedFile.user_id == user_id,
                UploadedFile.filename == filename,
            )
            .first()
        )
        if existing_record:
            result = _refresh_existing_file_if_changed(
                existing_record=existing_record,
                temp_file_path=temp_file_path,
                db_session=db_session,
                user_id=user_id,
                is_admin=is_admin,
                collection_name=collection_name,
                url=url,
                filename=filename,
                url_hash=url_hash,
                processed_urls=processed_urls,
                context="background-job cross-session",
            )
            if result is not None:
                processed_urls[url_hash] = str(existing_record.file_id)
                return result

            result = _recreate_missing_existing_file(
                existing_record=existing_record,
                temp_file_path=temp_file_path,
                db_session=db_session,
                user_id=user_id,
                is_admin=is_admin,
                collection_name=collection_name,
                filename=filename,
                url_hash=url_hash,
                processed_urls=processed_urls,
            )
            return result

        persistent_file = get_upload_path(
            filename,
            user_id=user_id,
            collection=collection_name,
            collection_is_sanitized=True,
        )
        persistent_file.parent.mkdir(parents=True, exist_ok=True)
        return _create_new_web_file_handler_result(
            temp_file_path=temp_file_path,
            persistent_file=persistent_file,
            db_session=db_session,
            user_id=user_id,
            is_admin=is_admin,
            collection_name=collection_name,
            filename=filename,
            url=url,
            url_hash=url_hash,
            processed_urls=processed_urls,
        )


def _cleanup_staged_document_input(payload: dict[str, Any]) -> None:
    from ..api.kb import _cleanup_background_ingest_staging_file

    _cleanup_background_ingest_staging_file(payload.get("source_path"))


def _has_generation_gate(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("target_path")
        and payload.get("generation_id")
        and payload.get("user_id") is not None
        and payload.get("collection")
    )


def _is_staged_document_generation_latest(
    db: Session,
    payload: dict[str, Any],
) -> bool:
    if not _has_generation_gate(payload):
        return True
    return is_latest_kb_ingest_generation(
        db,
        user_id=int(payload["user_id"]),
        collection=str(payload["collection"]),
        target_path=str(payload["target_path"]),
        generation_id=str(payload["generation_id"]),
    )


def _cleanup_failed_staged_job_collection_metadata_if_current(
    db: Session,
    payload: dict[str, Any],
    *,
    context: str,
) -> bool:
    if not _is_staged_document_generation_latest(db, payload):
        logger.info(
            "Skipping collection metadata cleanup for superseded KB ingest generation: "
            "%s/user_%s generation=%s",
            payload.get("collection"),
            payload.get("user_id"),
            payload.get("generation_id"),
        )
        return False

    _cleanup_failed_job_collection_metadata(
        db,
        payload,
        context=context,
    )
    return True


def _is_superseded_ingestion_result(result: IngestionResult) -> bool:
    return str(result.message) == _SUPERSEDED_STAGED_INGEST_MESSAGE


def _superseded_staged_document_result(payload: dict[str, Any]) -> dict[str, Any]:
    _cleanup_staged_document_input(payload)
    return {
        "status": "superseded",
        "published": False,
        "message": _SUPERSEDED_STAGED_INGEST_MESSAGE,
        "file_id": payload.get("file_id"),
        "generation_id": payload.get("generation_id"),
        "target_path": payload.get("target_path"),
    }


def _rollback_failed_staged_document_ingestion(
    db: Session,
    payload: dict[str, Any],
    result: IngestionResult,
    *,
    api_result: KBApiOperationResult[Any],
) -> KBApiOperationResult[Any]:
    from ..api.kb import _rollback_failed_cloud_ingestion

    user_id = int(payload["user_id"])
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        _cleanup_staged_document_input(payload)
        raise BackgroundJobHandlerError(
            f"Cannot roll back KB ingestion for missing user {user_id}",
            result=result.model_dump(mode="json"),
            retryable=False,
        )

    ingestion_config_payload = payload.get("ingestion_config")
    embedding_model_id = (
        ingestion_config_payload.get("embedding_model_id")
        if isinstance(ingestion_config_payload, dict)
        else None
    )
    try:
        rollback_execution = _get_api_compatibility_facade().run_failed_ingest_rollback(
            api_result,
            lambda: asyncio.run(
                _rollback_failed_cloud_ingestion(
                    db=db,
                    user=user,
                    collection_name=str(payload["collection"]),
                    result=result,
                    file_path=Path(str(payload["source_path"])),
                    file_record=None,
                    collection_existed_before=bool(
                        payload.get("collection_existed_before", True)
                    ),
                    uploaded_file_existed_before=False,
                    file_backup_path=None,
                    had_existing_file=False,
                    embedding_model_id=embedding_model_id,
                )
            ),
        )
        if rollback_execution.error is not None:
            raise BackgroundJobHandlerError(
                str(rollback_execution.error),
                result=result.model_dump(mode="json"),
                retryable=False,
            ) from rollback_execution.error
        return rollback_execution.operation_result
    finally:
        _cleanup_staged_document_input(payload)


def _rollback_failed_staged_document_ingestion_if_current(
    db: Session,
    payload: dict[str, Any],
    result: IngestionResult,
    *,
    api_result: KBApiOperationResult[Any],
) -> KBApiOperationResult[Any] | Literal[False]:
    if not _is_staged_document_generation_latest(db, payload):
        _cleanup_staged_document_input(payload)
        return False
    return _rollback_failed_staged_document_ingestion(
        db,
        payload,
        result,
        api_result=api_result,
    )


def _publish_staged_document_ingestion(
    db: Session,
    payload: dict[str, Any],
) -> UploadedFile:
    from ..api.kb import (
        _build_ingest_backup_path,
        _cleanup_background_ingest_staging_file,
        _restore_ingest_file_backup,
        _upsert_uploaded_file_record,
    )

    source_path = Path(str(payload["source_path"]))
    target_path = Path(str(payload["target_path"]))
    if not source_path.exists():
        raise FileNotFoundError(f"Missing staged ingest file: {source_path}")
    if not _is_staged_document_generation_latest(db, payload):
        raise StagedDocumentIngestSuperseded(_SUPERSEDED_STAGED_INGEST_MESSAGE)

    payload_file_id = str(payload["file_id"]) if payload.get("file_id") else None
    existing_record = (
        db.query(UploadedFile)
        .filter(UploadedFile.storage_path == str(target_path))
        .first()
    )
    if (
        existing_record is not None
        and payload_file_id
        and str(existing_record.file_id) != payload_file_id
    ):
        raise RuntimeError(
            "Canonical upload path was updated by another upload before this job published"
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    had_existing_file = target_path.exists()
    backup_path: Path | None = None
    if had_existing_file:
        backup_path = _build_ingest_backup_path(target_path)
        shutil.copy2(target_path, backup_path)

    try:
        shutil.copy2(source_path, target_path)
        file_record = _upsert_uploaded_file_record(
            db,
            user_id=int(payload["user_id"]),
            filename=str(payload["filename"]),
            storage_path=target_path,
            mime_type=payload.get("mime_type"),
            file_size=int(payload.get("file_size") or target_path.stat().st_size),
            file_id=payload_file_id,
        )
    except Exception:
        db.rollback()
        _restore_ingest_file_backup(
            file_path=target_path,
            backup_path=backup_path,
            had_existing_file=had_existing_file,
        )
        raise

    if backup_path is not None and backup_path.exists():
        try:
            backup_path.unlink()
        except OSError:
            logger.warning("Failed to remove ingest backup %s", backup_path)
    _cleanup_background_ingest_staging_file(source_path)
    return file_record


def _rollback_failed_document_ingestion(
    db: Session,
    payload: dict[str, Any],
    result: IngestionResult,
    *,
    api_result: KBApiOperationResult[Any],
) -> KBApiOperationResult[Any]:
    from ..api.kb import _rollback_failed_ingestion

    user_id = int(payload["user_id"])
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise BackgroundJobHandlerError(
            f"Cannot roll back KB ingestion for missing user {user_id}",
            result=result.model_dump(mode="json"),
            retryable=False,
        )

    file_record = None
    file_id = payload.get("file_id")
    if file_id:
        file_record = (
            db.query(UploadedFile).filter(UploadedFile.file_id == str(file_id)).first()
        )
    if file_record is None:
        file_record = (
            db.query(UploadedFile)
            .filter(UploadedFile.storage_path == str(payload["source_path"]))
            .first()
        )
    if file_record is None:
        raise BackgroundJobHandlerError(
            f"Cannot roll back KB ingestion for missing file {payload.get('file_id')}",
            result=result.model_dump(mode="json"),
            retryable=False,
        )

    backup_path = payload.get("file_backup_path")
    rollback_execution = _get_api_compatibility_facade().run_failed_ingest_rollback(
        api_result,
        lambda: asyncio.run(
            _rollback_failed_ingestion(
                db=db,
                user=user,
                collection_name=str(payload["collection"]),
                result=result,
                file_path=Path(str(payload["source_path"])),
                file_record=file_record,
                collection_existed_before=_collection_existed_before(payload),
                uploaded_file_existed_before=bool(
                    payload.get("uploaded_file_existed_before", True)
                ),
                file_backup_path=Path(str(backup_path)) if backup_path else None,
                had_existing_file=bool(payload.get("had_existing_file", True)),
            )
        ),
    )
    if rollback_execution.error is not None:
        raise BackgroundJobHandlerError(
            str(rollback_execution.error),
            result=result.model_dump(mode="json"),
            retryable=False,
        ) from rollback_execution.error
    return rollback_execution.operation_result


def _discard_ingest_backup(payload: dict[str, Any]) -> None:
    backup_path = payload.get("file_backup_path")
    if not backup_path:
        return
    backup = Path(str(backup_path))
    if not backup.exists():
        return
    try:
        backup.unlink()
    except OSError:
        logger.warning("Failed to remove ingest backup %s", backup)


def _cleanup_failed_web_collection_metadata_if_new(
    db: Session,
    payload: dict[str, Any],
    *,
    api_result: KBApiOperationResult[Any] | None = None,
    successful_documents: int | None = None,
) -> None:
    if api_result is None:
        _cleanup_failed_job_collection_metadata(
            db,
            payload,
            context="background web ingest",
            successful_documents=int(successful_documents or 0),
        )
        return

    _cleanup_failed_job_collection_metadata_after_api_ingest(
        db,
        payload,
        api_result=api_result,
        context="background web ingest",
        successful_documents=successful_documents,
    )
