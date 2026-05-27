from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from xagent.config import CELERY_BROKER_URL, CELERY_ENABLED
from xagent.core.tools.core.RAG_tools.core.schemas import (
    IngestionConfig,
    IngestionResult,
    WebCrawlConfig,
)
from xagent.web.models.background_job import BackgroundJobStatus, BackgroundJobType
from xagent.web.models.database import get_session_local, init_db
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.background_jobs import (
    create_background_job,
    enqueue_background_job,
    is_background_job_enqueue_available,
    requeue_stale_background_jobs,
)
from xagent.web.services.triggers import enqueue_trigger_event_job


def _init_test_db(path: Path):
    init_db(f"sqlite:///{path}")
    return get_session_local()


def _create_user(db, username: str = "background-job-test") -> User:
    user = User(username=username, password_hash="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_enqueue_background_job_disabled_stays_pending(tmp_path, monkeypatch):
    monkeypatch.setenv(CELERY_ENABLED, "false")
    monkeypatch.delenv(CELERY_BROKER_URL, raising=False)

    SessionLocal = _init_test_db(tmp_path / "jobs-disabled.db")
    db = SessionLocal()
    try:
        user = _create_user(db)
        job = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.TRIGGER_EVENT,
            payload={"source_type": "email", "event_type": "message.received"},
        )

        enqueued = enqueue_background_job(db, job)

        assert enqueued.status == BackgroundJobStatus.PENDING.value
        assert enqueued.celery_task_id is None
    finally:
        db.close()


def test_background_job_enqueue_unavailable_without_worker(monkeypatch):
    monkeypatch.setenv(CELERY_ENABLED, "true")
    monkeypatch.setenv(CELERY_BROKER_URL, "memory://")

    assert is_background_job_enqueue_available(check_worker=False) is True
    assert is_background_job_enqueue_available(check_worker=True) is False


def test_job_capabilities_use_sync_without_worker(monkeypatch):
    monkeypatch.setenv(CELERY_ENABLED, "true")
    monkeypatch.setenv(CELERY_BROKER_URL, "memory://")

    from xagent.web.api.jobs import get_job_capabilities

    capabilities = get_job_capabilities(_user=object())  # type: ignore[arg-type]

    assert capabilities["kb_ingest_mode"] == "sync"
    assert capabilities["celery_enabled"] is True
    assert capabilities["broker_configured"] is True
    assert capabilities["broker_reachable"] is True
    assert capabilities["worker_available"] is False


def test_celery_worker_app_import_registers_tasks():
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    code = """
from xagent.web.jobs.celery_app import celery_app
expected = {
    "xagent.web.jobs.tasks.execute_background_job",
    "xagent.web.jobs.trigger_tasks.scan_due_triggers",
}
missing = expected.difference(celery_app.tasks)
assert not missing, missing
assert not celery_app.conf.task_always_eager
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )


def test_trigger_event_job_runs_with_eager_celery(tmp_path, monkeypatch):
    monkeypatch.setenv(CELERY_ENABLED, "true")
    monkeypatch.setenv(CELERY_BROKER_URL, "memory://")

    from xagent.web.jobs.celery_app import celery_app

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

    SessionLocal = _init_test_db(tmp_path / "jobs-eager.db")
    db = SessionLocal()
    try:
        user = _create_user(db, username="trigger-eager-test")

        job = enqueue_trigger_event_job(
            db,
            user_id=int(user.id),
            source_type="email",
            event_type="message.received",
            source_event_id="evt-1",
            event_payload={"subject": "hello"},
        )

        db.refresh(job)
        assert job.status == BackgroundJobStatus.SUCCEEDED.value
        assert job.result == {
            "status": "accepted",
            "source_type": "email",
            "event_type": "message.received",
            "processed_at": job.result["processed_at"],
        }
        assert job.celery_task_id
    finally:
        db.close()
        celery_app.conf.task_always_eager = False
        celery_app.conf.task_eager_propagates = False


def test_trigger_event_idempotency_is_scoped_by_user(tmp_path, monkeypatch):
    monkeypatch.setenv(CELERY_ENABLED, "false")
    monkeypatch.delenv(CELERY_BROKER_URL, raising=False)

    SessionLocal = _init_test_db(tmp_path / "trigger-idempotency-scope.db")
    db = SessionLocal()
    try:
        user_one = _create_user(db, username="trigger-user-one")
        user_two = _create_user(db, username="trigger-user-two")

        job_one = enqueue_trigger_event_job(
            db,
            user_id=int(user_one.id),
            source_type="email",
            event_type="message.received",
            source_event_id="evt-1",
            event_payload={"subject": "hello"},
        )
        job_two = enqueue_trigger_event_job(
            db,
            user_id=int(user_two.id),
            source_type="email",
            event_type="message.received",
            source_event_id="evt-1",
            event_payload={"subject": "hello"},
        )

        assert job_one.id != job_two.id
        assert job_one.user_id == int(user_one.id)
        assert job_two.user_id == int(user_two.id)
    finally:
        db.close()


def test_kb_idempotency_reuses_only_non_terminal_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv(CELERY_ENABLED, "false")
    monkeypatch.delenv(CELERY_BROKER_URL, raising=False)

    SessionLocal = _init_test_db(tmp_path / "kb-idempotency-terminal.db")
    db = SessionLocal()
    try:
        user = _create_user(db, username="kb-idempotency-test")
        idempotency_key = "kb.ingest.document:test"
        first_job = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT,
            payload={"collection": "kb", "version": 1},
            idempotency_key=idempotency_key,
            reuse_terminal_idempotency_key=False,
        )
        setattr(first_job, "status", BackgroundJobStatus.FAILED.value)
        db.add(first_job)
        db.commit()

        retry_job = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT,
            payload={"collection": "kb", "version": 2},
            idempotency_key=idempotency_key,
            reuse_terminal_idempotency_key=False,
        )
        duplicate_in_flight = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT,
            payload={"collection": "kb", "version": 3},
            idempotency_key=idempotency_key,
            reuse_terminal_idempotency_key=False,
        )

        db.refresh(first_job)
        assert first_job.idempotency_key is None
        assert retry_job.id != first_job.id
        assert retry_job.idempotency_key == idempotency_key
        assert duplicate_in_flight.id == retry_job.id
    finally:
        db.close()


def test_kb_document_job_reads_staged_file_and_publishes_canonical(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(CELERY_ENABLED, "false")
    monkeypatch.delenv(CELERY_BROKER_URL, raising=False)

    from xagent.web.jobs.kb_tasks import handle_kb_ingest_document

    SessionLocal = _init_test_db(tmp_path / "kb-staged-ingest.db")
    db = SessionLocal()
    try:
        user = _create_user(db, username="kb-staged-ingest-test")
        staged_file = tmp_path / "stage" / "doc.txt"
        target_file = tmp_path / "canonical" / "doc.txt"
        staged_file.parent.mkdir(parents=True)
        staged_file.write_text("staged content", encoding="utf-8")
        file_id = "11111111-1111-4111-8111-111111111111"
        ingestion_config = IngestionConfig()
        job = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT,
            payload={
                "collection": "kb",
                "source_path": str(staged_file),
                "target_path": str(target_file),
                "file_id": file_id,
                "filename": "doc.txt",
                "mime_type": "text/plain",
                "file_size": staged_file.stat().st_size,
                "user_id": int(user.id),
                "is_admin": False,
                "ingestion_config": ingestion_config.model_dump(mode="json"),
                "collection_existed_before": True,
            },
        )

        captured = {}

        def fake_run_document_ingestion(**kwargs):
            captured.update(kwargs)
            return IngestionResult(
                status="success",
                doc_id="doc-1",
                message="ok",
                completed_steps=[
                    {"name": "register_document", "metadata": {"created": True}}
                ],
            )

        monkeypatch.setattr(
            "xagent.web.jobs.kb_tasks.run_document_ingestion",
            fake_run_document_ingestion,
        )

        result = handle_kb_ingest_document(db, job)

        assert captured["source_path"] == str(staged_file)
        assert captured["metadata_source_path"] == str(target_file)
        assert result["file_id"] == file_id
        assert target_file.read_text(encoding="utf-8") == "staged content"
        assert not staged_file.exists()
        file_record = (
            db.query(UploadedFile)
            .filter(UploadedFile.storage_path == str(target_file))
            .first()
        )
        assert file_record is not None
        assert str(file_record.file_id) == file_id
    finally:
        db.close()


def test_kb_document_job_supersedes_older_generation_for_same_target(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(CELERY_ENABLED, "false")
    monkeypatch.delenv(CELERY_BROKER_URL, raising=False)

    from xagent.web.jobs.kb_tasks import handle_kb_ingest_document
    from xagent.web.services.kb_ingest_targets import admit_kb_ingest_target

    SessionLocal = _init_test_db(tmp_path / "kb-target-generation.db")
    db = SessionLocal()
    try:
        user = _create_user(db, username="kb-target-generation-test")
        stage_dir = tmp_path / "stage"
        target_file = tmp_path / "canonical" / "doc.txt"
        stage_dir.mkdir(parents=True)
        staged_a = stage_dir / "a.txt"
        staged_b = stage_dir / "b.txt"
        staged_a.write_text("older content", encoding="utf-8")
        staged_b.write_text("newer content", encoding="utf-8")
        file_id = "22222222-2222-4222-8222-222222222222"
        generation_a = "33333333-3333-4333-8333-333333333333"
        generation_b = "44444444-4444-4444-8444-444444444444"
        ingestion_config = IngestionConfig()

        def payload_for(path: Path, generation_id: str) -> dict:
            return {
                "collection": "kb",
                "source_path": str(path),
                "target_path": str(target_file),
                "file_id": file_id,
                "generation_id": generation_id,
                "file_sha256": generation_id,
                "filename": "doc.txt",
                "mime_type": "text/plain",
                "file_size": path.stat().st_size,
                "user_id": int(user.id),
                "is_admin": False,
                "ingestion_config": ingestion_config.model_dump(mode="json"),
                "collection_existed_before": True,
            }

        job_a = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT,
            payload=payload_for(staged_a, generation_a),
        )
        job_b = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT,
            payload=payload_for(staged_b, generation_b),
        )
        admit_kb_ingest_target(
            db,
            user_id=int(user.id),
            collection="kb",
            target_path=str(target_file),
            file_id=file_id,
            generation_id=generation_a,
            job_id=str(job_a.id),
            file_sha256=generation_a,
        )
        admit_kb_ingest_target(
            db,
            user_id=int(user.id),
            collection="kb",
            target_path=str(target_file),
            file_id=file_id,
            generation_id=generation_b,
            job_id=str(job_b.id),
            file_sha256=generation_b,
        )

        ingested_sources: list[str] = []

        def fake_run_document_ingestion(**kwargs):
            ingested_sources.append(kwargs["source_path"])
            return IngestionResult(
                status="success",
                doc_id="doc-1",
                message="ok",
                completed_steps=[
                    {"name": "register_document", "metadata": {"created": True}}
                ],
            )

        monkeypatch.setattr(
            "xagent.web.jobs.kb_tasks.run_document_ingestion",
            fake_run_document_ingestion,
        )

        result_b = handle_kb_ingest_document(db, job_b)
        result_a = handle_kb_ingest_document(db, job_a)

        assert result_b["file_id"] == file_id
        assert result_a["status"] == "superseded"
        assert result_a["published"] is False
        assert ingested_sources == [str(staged_b)]
        assert target_file.read_text(encoding="utf-8") == "newer content"
        assert not staged_a.exists()
        assert not staged_b.exists()
        file_record = (
            db.query(UploadedFile)
            .filter(UploadedFile.storage_path == str(target_file))
            .first()
        )
        assert file_record is not None
        assert str(file_record.file_id) == file_id
    finally:
        db.close()


def test_kb_document_job_skips_canonical_rollback_when_generation_turns_stale(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(CELERY_ENABLED, "false")
    monkeypatch.delenv(CELERY_BROKER_URL, raising=False)

    from xagent.web.jobs import kb_tasks
    from xagent.web.jobs.kb_tasks import handle_kb_ingest_document
    from xagent.web.services.kb_ingest_targets import admit_kb_ingest_target

    SessionLocal = _init_test_db(tmp_path / "kb-stale-rollback.db")
    db = SessionLocal()
    try:
        user = _create_user(db, username="kb-stale-rollback-test")
        staged_file = tmp_path / "stage" / "doc.txt"
        target_file = tmp_path / "canonical" / "doc.txt"
        staged_file.parent.mkdir(parents=True)
        staged_file.write_text("older content", encoding="utf-8")
        file_id = "55555555-5555-4555-8555-555555555555"
        generation_a = "66666666-6666-4666-8666-666666666666"
        generation_b = "77777777-7777-4777-8777-777777777777"
        ingestion_config = IngestionConfig()

        job = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT,
            payload={
                "collection": "kb",
                "source_path": str(staged_file),
                "target_path": str(target_file),
                "file_id": file_id,
                "generation_id": generation_a,
                "file_sha256": generation_a,
                "filename": "doc.txt",
                "mime_type": "text/plain",
                "file_size": staged_file.stat().st_size,
                "user_id": int(user.id),
                "is_admin": False,
                "ingestion_config": ingestion_config.model_dump(mode="json"),
                "collection_existed_before": True,
            },
        )
        admit_kb_ingest_target(
            db,
            user_id=int(user.id),
            collection="kb",
            target_path=str(target_file),
            file_id=file_id,
            generation_id=generation_a,
            job_id=str(job.id),
            file_sha256=generation_a,
        )

        def fake_run_document_ingestion(**kwargs):
            admit_kb_ingest_target(
                db,
                user_id=int(user.id),
                collection="kb",
                target_path=str(target_file),
                file_id=file_id,
                generation_id=generation_b,
                job_id="newer-job",
                file_sha256=generation_b,
            )
            return IngestionResult(
                status="partial",
                doc_id="doc-1",
                message="partial after stale generation",
                completed_steps=[
                    {"name": "register_document", "metadata": {"created": True}}
                ],
            )

        def fail_rollback(*args, **kwargs):
            raise AssertionError("stale staged jobs must not roll back canonical state")

        monkeypatch.setattr(
            "xagent.web.jobs.kb_tasks.run_document_ingestion",
            fake_run_document_ingestion,
        )
        monkeypatch.setattr(
            kb_tasks,
            "_rollback_failed_staged_document_ingestion",
            fail_rollback,
        )

        result = handle_kb_ingest_document(db, job)

        assert result["status"] == "superseded"
        assert result["published"] is False
        assert not staged_file.exists()
    finally:
        db.close()


def test_background_job_progress_manager_mirrors_rag_progress(tmp_path, monkeypatch):
    monkeypatch.setenv(CELERY_ENABLED, "false")
    monkeypatch.delenv(CELERY_BROKER_URL, raising=False)

    from xagent.web.jobs.progress import BackgroundJobProgressManager

    class Delegate:
        def create_task(self, **kwargs):
            return kwargs["task_id"]

        def update_task_progress(self, *args, **kwargs):
            return None

        def complete_task(self, *args, **kwargs):
            return None

        def track_task(self, *args, **kwargs):
            raise AssertionError("not used")

        def get_active_tasks(self, *args, **kwargs):
            return []

    SessionLocal = _init_test_db(tmp_path / "jobs-progress.db")
    db = SessionLocal()
    try:
        user = _create_user(db, username="progress-test")
        job = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT,
            payload={"collection": "kb"},
        )

        manager = BackgroundJobProgressManager(
            db,
            job,
            delegate=Delegate(),
            throttle_seconds=0,
        )
        task_id = manager.create_task("ingestion", task_id="task-1")
        manager.update_task_progress(
            task_id,
            current_step="parse_document",
            overall_progress=0.25,
            metadata={
                "steps": {
                    "parse_document": {
                        "message": "Parsing document",
                        "step_progress": 0.5,
                    }
                }
            },
        )

        db.refresh(job)
        assert job.progress["message"] == "Parsing document"
        assert job.progress["completed"] == 25
        assert job.progress["total"] == 100
        assert job.progress["current_step"] == "parse_document"
        assert (
            job.progress["metadata"]["steps"]["parse_document"]["step_progress"] == 0.5
        )
    finally:
        db.close()


def test_kb_web_job_cleans_new_collection_metadata_on_ingest_error(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(CELERY_ENABLED, "false")
    monkeypatch.delenv(CELERY_BROKER_URL, raising=False)

    from xagent.web.jobs.exceptions import BackgroundJobHandlerError
    from xagent.web.jobs.kb_tasks import handle_kb_ingest_web

    SessionLocal = _init_test_db(tmp_path / "web-ingest-cleanup.db")
    db = SessionLocal()
    try:
        user = _create_user(db, username="web-ingest-cleanup-test")
        job = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_WEB,
            payload={
                "collection": "web-kb",
                "crawl_config": WebCrawlConfig(
                    start_url="https://example.com"
                ).model_dump(mode="json"),
                "ingestion_config": IngestionConfig().model_dump(mode="json"),
                "user_id": int(user.id),
                "is_admin": False,
                "collection_existed_before": False,
            },
        )

        async def fake_run_web_ingestion(**kwargs):
            return IngestionResult(status="error", message="crawl failed")

        cleaned: list[tuple[str, int]] = []

        async def fake_cleanup(*, collection_name, user):
            cleaned.append((collection_name, int(user.id)))

        monkeypatch.setattr(
            "xagent.web.jobs.kb_tasks.run_web_ingestion",
            fake_run_web_ingestion,
        )
        monkeypatch.setattr(
            "xagent.web.api.kb._cleanup_failed_new_collection_metadata",
            fake_cleanup,
        )

        with pytest.raises(BackgroundJobHandlerError):
            handle_kb_ingest_web(db, job)

        assert cleaned == [("web-kb", int(user.id))]
    finally:
        db.close()


def test_requeue_stale_background_jobs_marks_old_running_pending(tmp_path, monkeypatch):
    monkeypatch.setenv(CELERY_ENABLED, "false")
    monkeypatch.delenv(CELERY_BROKER_URL, raising=False)

    SessionLocal = _init_test_db(tmp_path / "jobs-stale.db")
    db = SessionLocal()
    try:
        user = _create_user(db, username="stale-test")
        job = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_WEB,
            payload={"collection": "kb"},
        )
        old = datetime.now(timezone.utc) - timedelta(hours=3)
        setattr(job, "status", BackgroundJobStatus.RUNNING.value)
        setattr(job, "started_at", old)
        db.add(job)
        db.commit()
        db.refresh(job)

        requeued = requeue_stale_background_jobs(db, stale_after_seconds=60)

        assert [item.id for item in requeued] == [job.id]
        db.refresh(job)
        assert job.status == BackgroundJobStatus.PENDING.value
        assert job.celery_task_id is None
        assert job.started_at is None
        assert job.progress["message"] == "Requeued stale background job"
    finally:
        db.close()
