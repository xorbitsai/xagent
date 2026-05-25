from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from xagent.config import CELERY_BROKER_URL, CELERY_ENABLED
from xagent.web.models.background_job import BackgroundJobStatus, BackgroundJobType
from xagent.web.models.database import get_session_local, init_db
from xagent.web.models.user import User
from xagent.web.services.background_jobs import (
    create_background_job,
    enqueue_background_job,
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
