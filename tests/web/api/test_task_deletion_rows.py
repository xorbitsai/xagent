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


def test_purge_task_rows_preserves_uploaded_files_by_detaching_them() -> None:
    """The create-failure compensation branch must keep the user's uploads.

    ``preserve_uploaded_files=True`` is the compensation path used after a
    runtime-extension binding failure: the just-created task row goes away but
    the files the request already stored must survive, detached from the task.
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
            purge_task_rows(db, task_id=task_id, preserve_uploaded_files=True) is True
        )
        db.commit()

        assert db.query(Task).filter(Task.id == task_id).count() == 0
        surviving = (
            db.query(UploadedFile).filter(UploadedFile.user_id == owner.id).all()
        )
        assert len(surviving) == 2
        assert all(row.task_id is None for row in surviving)
        assert {row.filename for row in surviving} == {"kept-0.txt", "kept-1.txt"}
    finally:
        db.close()


def test_purge_task_rows_without_preservation_also_detaches_uploaded_files() -> None:
    """Characterise the ``preserve_uploaded_files=False`` branch.

    ``Task.uploaded_files`` is a relationship without a cascade, so the ORM
    ``db.delete(task)`` nulls ``UploadedFile.task_id`` on its own. Both flag
    values therefore leave the rows detached rather than deleted; the flag only
    controls whether an extra bulk UPDATE runs first.
    """

    _admin_headers()
    _register_second_user("purge-owner-2", "purgepass1")
    db = _direct_db_session()
    try:
        db.execute(text("PRAGMA foreign_keys = ON"))
        owner = db.query(User).filter(User.username == "purge-owner-2").one()
        task = Task(user_id=int(owner.id), title="deleted", description="")
        db.add(task)
        db.flush()
        task_id = int(task.id)
        db.add(
            UploadedFile(
                user_id=int(owner.id),
                task_id=task_id,
                filename="detached.txt",
                storage_path="/tmp/detached.txt",
                file_size=1,
            )
        )
        db.commit()

        assert (
            purge_task_rows(db, task_id=task_id, preserve_uploaded_files=False) is True
        )
        db.commit()

        assert db.query(Task).filter(Task.id == task_id).count() == 0
        rows = db.query(UploadedFile).filter(UploadedFile.user_id == owner.id).all()
        assert [row.task_id for row in rows] == [None]
    finally:
        db.close()
