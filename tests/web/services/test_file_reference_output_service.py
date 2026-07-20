from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.file_reference_output_service import (
    reconcile_assistant_file_references,
)


def _create_context():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    user = User(username="owner", password_hash="hashed", is_admin=False)
    db.add(user)
    db.flush()
    task = Task(
        user_id=int(user.id),
        title="FileRef task",
        description="Generate a video",
        status=TaskStatus.COMPLETED,
    )
    db.add(task)
    db.flush()
    return db, user, task


def _add_file(db, user, task, *, file_id: str, filename: str):
    record = UploadedFile(
        file_id=file_id,
        user_id=int(user.id),
        task_id=int(task.id) if task is not None else None,
        filename=filename,
        storage_path=f"/tmp/{file_id}/{filename}",
        mime_type="video/mp4",
        file_size=123,
    )
    db.add(record)
    db.flush()
    return record


def test_reconcile_keeps_valid_file_reference():
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_video.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[generated_video.mp4](file:real-id)",
        )

        assert content == "[generated_video.mp4](file:real-id)"
    finally:
        db.close()


def test_reconcile_repairs_invented_id_from_unique_filename():
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="c6553861-5bdd-4628-9b15-1310e34fe499",
            filename="generated_video_253a6da9.mp4",
        )
        _add_file(
            db,
            user,
            None,
            file_id="older-unbound-id",
            filename="generated_video_253a6da9.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=(
                "下载：[generated_video_253a6da9.mp4]"
                "(file:253a6da9-76e1-4b16-b26e-2eba2d8b0583)"
            ),
        )

        assert content == (
            "下载：[generated_video_253a6da9.mp4]"
            "(file:c6553861-5bdd-4628-9b15-1310e34fe499)"
        )
    finally:
        db.close()


def test_reconcile_unlinks_unknown_or_ambiguous_file_reference():
    db, user, task = _create_context()
    try:
        _add_file(db, user, task, file_id="first-id", filename="report.mp4")
        _add_file(db, user, task, file_id="second-id", filename="report.mp4")

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=(
                "[missing.mp4](file:invented-id) and "
                "[report.mp4](file:another-invented-id)"
            ),
        )

        assert content == "missing.mp4 and report.mp4"
        assert "file:" not in content
    finally:
        db.close()


def test_reconcile_unlinks_record_with_invalid_file_id():
    db, user, task = _create_context()
    try:
        _add_file(
            db,
            user,
            task,
            file_id="invalid/id",
            filename="generated_video.mp4",
        )

        content = reconcile_assistant_file_references(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            content="[generated_video.mp4](file:invented-id)",
        )

        assert content == "generated_video.mp4"
    finally:
        db.close()


def test_reconcile_reuses_prefetched_records_without_querying():
    db, user, task = _create_context()
    try:
        record = _add_file(
            db,
            user,
            task,
            file_id="real-id",
            filename="generated_video.mp4",
        )

        with patch.object(
            db,
            "query",
            side_effect=AssertionError("prefetched reconciliation must not query"),
        ):
            content = reconcile_assistant_file_references(
                db,
                task_id=int(task.id),
                user_id=int(user.id),
                content="[generated_video.mp4](file:invented-id)",
                records=[record],
            )

        assert content == "[generated_video.mp4](file:real-id)"
    finally:
        db.close()
