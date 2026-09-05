"""Row-level behaviour of the shared task purge helper."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.task_deletion import purge_task_rows

from .conftest import _admin_headers, _direct_db_session, _register_second_user

pytestmark = pytest.mark.usefixtures("_test_db")


def test_purge_task_rows_detaches_uploaded_files_instead_of_deleting_them() -> None:
    """Purging a task must leave its uploads behind, detached from the task.

    ``Task.uploaded_files`` is a relationship without a cascade, so the ORM
    ``db.delete(task)`` nulls ``UploadedFile.task_id`` on its own. Both callers
    -- the create-failure compensation path and ordinary task deletion -- rely
    on this: the task row goes away while the files the request already stored
    survive.
    """

    _admin_headers()
    _register_second_user("purge-owner", "purgepass1")
    db = _direct_db_session()
    try:
        db.execute(text("PRAGMA foreign_keys = ON"))
        owner = db.query(User).filter(User.username == "purge-owner").one()
        task = Task(user_id=int(owner.id), title="compensated", description="")
        db.add(task)
        db.flush()
        task_id = int(task.id)
        db.add_all(
            [
                UploadedFile(
                    user_id=int(owner.id),
                    task_id=task_id,
                    filename=f"kept-{index}.txt",
                    storage_path=f"/tmp/kept-{index}.txt",
                    file_size=1,
                )
                for index in range(2)
            ]
        )
        db.commit()

        assert (
            purge_task_rows(
                db,
                task_id=task_id,
                detached_reason="task_deleted",
            )
            is True
        )
        db.commit()

        assert db.query(Task).filter(Task.id == task_id).count() == 0
        surviving = (
            db.query(UploadedFile).filter(UploadedFile.user_id == owner.id).all()
        )
        assert len(surviving) == 2
        assert all(row.task_id is None for row in surviving)
        assert all(row.detached_reason == "task_deleted" for row in surviving)
        assert all(row.detached_at is not None for row in surviving)
        assert {row.filename for row in surviving} == {"kept-0.txt", "kept-1.txt"}
    finally:
        db.close()


def test_purge_task_rows_returns_false_for_a_missing_task() -> None:
    """A purge for an already-deleted task is a no-op, not an error."""

    _admin_headers()
    db = _direct_db_session()
    try:
        assert (
            purge_task_rows(
                db,
                task_id=987654321,
                detached_reason="task_create_failed",
            )
            is False
        )
    finally:
        db.close()
