"""Real PostgreSQL contenders for attachment binding and task deletion."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from threading import Event

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.file_turn import bind_turn_files_no_commit
from xagent.web.services.task_deletion import purge_task_rows
from xagent.web.services.uploaded_file_store import (
    StagedUploadedFile,
    UploadedFileStore,
    snapshot_uploaded_file_version,
)

pytestmark = pytest.mark.postgresql


@pytest.fixture
def sessions():
    with disposable_database_factory("attachment_race") as make:
        engine = make("locks")
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, autoflush=False)
        with factory.begin() as db:
            db.add(User(id=1, username="owner", password_hash="unused"))
            db.flush()
            db.add_all([Task(id=i, user_id=1, title=str(i)) for i in (1, 2)])
            db.flush()
            db.add(
                UploadedFile(
                    file_id="file",
                    user_id=1,
                    filename="file",
                    storage_path="/unused/test-file",
                    storage_status="available",
                )
            )
        yield factory


def bind(db, target=1):
    return bind_turn_files_no_commit(
        db=db, file_ids=["file"], task_id=target, owner_user_id=1
    )


def delete(db):
    return purge_task_rows(db, task_id=1, detached_reason="task_deleted")


def wait_for_block(sessions, pid):
    deadline = time.monotonic() + 3
    with sessions() as observer:
        while time.monotonic() < deadline:
            if observer.execute(
                sa.text("SELECT cardinality(pg_blocking_pids(:pid))"), {"pid": pid}
            ).scalar():
                return
            time.sleep(0.01)
    pytest.fail("The competing operation never waited on the winning transaction")


@pytest.mark.parametrize("first", ["bind", "delete"])
def test_task_delete_and_bind_serialize_both_winners(sessions, first):
    operations = {"bind": bind, "delete": delete}
    second = "delete" if first == "bind" else "bind"
    ready = Event()
    pid = []

    def contender():
        with sessions() as db:
            db.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            pid.append(db.execute(sa.text("SELECT pg_backend_pid()")).scalar())
            ready.set()
            result = operations[second](db)
            db.commit()
            return result

    with sessions() as winner, ThreadPoolExecutor(max_workers=1) as executor:
        operations[first](winner)
        future = executor.submit(contender)
        try:
            assert ready.wait(3)
            wait_for_block(sessions, pid[0])
            winner.commit()
        finally:
            winner.rollback()
        assert future.result(timeout=6) == (True if second == "delete" else ["file"])
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.task_id is None
        assert row.detached_reason == ("task_deleted" if first == "bind" else None)


def test_rebind_rejects_uncommitted_source_delete_and_succeeds_after_commit(sessions):
    with sessions.begin() as db:
        assert bind(db) == []

    def rebind():
        with sessions.begin() as db:
            db.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            return bind(db, 2)

    with sessions() as deleter, ThreadPoolExecutor(max_workers=1) as executor:
        delete(deleter)
        try:
            # The committed snapshot still belongs to task 1, so task 2 must
            # reject it even though detachment is in progress.
            assert executor.submit(rebind).result(timeout=6) == ["file"]
            deleter.commit()
        finally:
            deleter.rollback()
    assert rebind() == []
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.task_id == 2
        assert row.detached_reason is None
        assert row.detached_at is None


@pytest.mark.parametrize("insert_first", [True, False])
def test_binding_does_not_block_child_foreign_key_inserts(sessions, insert_first):
    with sessions() as binder, sessions() as child:
        for db in (binder, child):
            db.execute(sa.text("SET LOCAL lock_timeout = '1s'"))

        def insert():
            child.add(
                TraceEvent(
                    task_id=1,
                    event_id="child",
                    event_type="test",
                    data={},
                    timestamp=datetime.now(UTC),
                )
            )
            child.flush()

        if insert_first:
            insert()
        assert bind(binder) == []
        if not insert_first:
            insert()
        child.commit()
        binder.commit()


def test_delete_fences_new_upload_publication_before_detachment_scan(sessions):
    with sessions() as deleter, sessions() as uploader:
        delete(deleter)
        uploader.execute(sa.text("SET LOCAL lock_timeout = '200ms'"))
        uploader.add(
            UploadedFile(
                file_id="late",
                user_id=1,
                task_id=1,
                filename="late",
                storage_path="/unused/late",
            )
        )
        with pytest.raises(sa.exc.OperationalError, match="lock timeout"):
            uploader.flush()
        uploader.rollback()
        deleter.commit()


def test_upsert_batch_locks_task_before_same_task_upload_update(sessions):
    with sessions.begin() as db:
        first = db.query(UploadedFile).filter_by(file_id="file").one()
        first.task_id = 1
        first.storage_backend = "file"
        first.storage_key = "users/1/tasks/1/outputs/file/first.txt"
        first.checksum = "first"
        second = UploadedFile(
            file_id="second",
            user_id=1,
            task_id=2,
            filename="second.txt",
            storage_path="/unused/second.txt",
            storage_backend="file",
            storage_key="users/1/tasks/2/outputs/second/second.txt",
            checksum="second",
            storage_status="available",
        )
        db.add(second)
        db.flush()
        first_expected = snapshot_uploaded_file_version(first)
        second_expected = snapshot_uploaded_file_version(second)
        first_staged = replace(
            StagedUploadedFile.from_record(first),
            storage_key="users/1/tasks/1/outputs/file/replacement.txt",
            checksum="first-replacement",
        )
        second_staged = replace(
            StagedUploadedFile.from_record(second),
            task_id=1,
            storage_key="users/1/tasks/1/outputs/second/replacement.txt",
            checksum="second-replacement",
        )

    ready = Event()
    pid = []

    def purge():
        with sessions() as db:
            db.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            db.execute(sa.text("SET LOCAL statement_timeout = '5s'"))
            pid.append(db.execute(sa.text("SELECT pg_backend_pid()")).scalar())
            ready.set()
            result = delete(db)
            db.commit()
            return result

    with sessions() as updater, ThreadPoolExecutor(max_workers=1) as executor:
        updater.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
        updater.execute(sa.text("SET LOCAL statement_timeout = '5s'"))
        store = UploadedFileStore(updater)
        store.upsert_already_durable(first_staged, expected=first_expected)
        future = executor.submit(purge)
        try:
            assert ready.wait(3)
            wait_for_block(sessions, pid[0])
            store.upsert_already_durable(
                second_staged,
                expected=second_expected,
                allow_task_rebind=True,
            )
            updater.commit()
        finally:
            updater.rollback()
        assert future.result(timeout=6) is True

    with sessions() as db:
        rows = db.query(UploadedFile).order_by(UploadedFile.file_id).all()
        assert [row.task_id for row in rows] == [None, None]
        assert [row.detached_reason for row in rows] == [
            "task_deleted",
            "task_deleted",
        ]
