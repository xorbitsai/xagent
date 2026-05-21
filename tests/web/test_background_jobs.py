from __future__ import annotations

from pathlib import Path

from xagent.config import CELERY_BROKER_URL, CELERY_ENABLED
from xagent.web.models.background_job import BackgroundJobStatus, BackgroundJobType
from xagent.web.models.database import get_session_local, init_db
from xagent.web.models.user import User
from xagent.web.services.background_jobs import (
    create_background_job,
    enqueue_background_job,
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
    monkeypatch.delenv(CELERY_ENABLED, raising=False)
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
