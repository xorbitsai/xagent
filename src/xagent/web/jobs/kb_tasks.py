from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ...core.tools.core.RAG_tools.core.schemas import (
    IngestionConfig,
    WebCrawlConfig,
)
from ...core.tools.core.RAG_tools.pipelines.document_ingestion import (
    run_document_ingestion,
)
from ...core.tools.core.RAG_tools.pipelines.web_ingestion import (
    FileHandlerResult,
    run_web_ingestion,
)
from ...core.tools.core.RAG_tools.progress import get_progress_manager
from ...core.tools.core.RAG_tools.utils.user_scope import user_scope_context
from ..config import get_upload_path
from ..models.background_job import BackgroundJob
from ..models.database import get_session_local
from ..models.uploaded_file import UploadedFile
from ..services.background_jobs import update_job_progress
from .exceptions import BackgroundJobHandlerError

logger = logging.getLogger(__name__)


def handle_kb_ingest_document(db: Session, job: BackgroundJob) -> dict[str, Any]:
    payload = dict(job.payload or {})
    ingestion_config = IngestionConfig.model_validate(payload["ingestion_config"])
    file_id = payload.get("file_id")

    update_job_progress(db, job, message="Ingesting document")
    with user_scope_context(
        user_id=int(payload["user_id"]),
        is_admin=bool(payload.get("is_admin", False)),
    ):
        result = run_document_ingestion(
            collection=str(payload["collection"]),
            source_path=str(payload["source_path"]),
            ingestion_config=ingestion_config,
            progress_manager=get_progress_manager(),
            user_id=int(payload["user_id"]),
            is_admin=bool(payload.get("is_admin", False)),
            file_id=str(file_id) if file_id else None,
        )

    result_payload = result.model_dump(mode="json")
    if file_id:
        result_payload["file_id"] = file_id
    if result.status == "error":
        raise BackgroundJobHandlerError(result.message, result=result_payload)
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
                processed_urls=processed_urls,
            )
        finally:
            db_session.close()

    update_job_progress(db, job, message="Crawling website")
    with user_scope_context(user_id=user_id, is_admin=is_admin):
        result = asyncio.run(
            run_web_ingestion(
                collection=collection,
                crawl_config=crawl_config,
                ingestion_config=ingestion_config,
                progress_callback=_progress,
                user_id=user_id,
                is_admin=is_admin,
                file_handler=_file_handler_with_db,
            )
        )

    result_payload = result.model_dump(mode="json")
    if result.status == "error":
        raise BackgroundJobHandlerError(result.message, result=result_payload)
    return result_payload


def _handle_web_file(
    *,
    temp_file_path: Path,
    title: str,
    collection_name: str,
    url: str,
    db_session: Session,
    user_id: int,
    processed_urls: dict[str, str],
) -> FileHandlerResult:
    from ..api.kb import (
        _normalize_web_title_for_filename,
        _recreate_missing_existing_file,
        _refresh_existing_file_if_changed,
        _upsert_uploaded_file_record,
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
        try:
            shutil.copy2(temp_file_path, persistent_file)
            file_record = _upsert_uploaded_file_record(
                db_session,
                user_id=user_id,
                filename=filename,
                storage_path=persistent_file,
                mime_type="text/markdown",
                file_size=persistent_file.stat().st_size,
            )
            processed_urls[url_hash] = str(file_record.file_id)
            return FileHandlerResult(
                file_path=str(persistent_file),
                file_id=str(file_record.file_id),
            )
        except Exception:
            if persistent_file.exists():
                try:
                    persistent_file.unlink()
                except OSError:
                    logger.warning(
                        "Failed to clean up orphaned web-ingest file %s",
                        persistent_file,
                    )
            raise
