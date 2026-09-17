"""Real PostgreSQL coverage for metadata-first checkpoint blob writes."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from xagent.web.models.database import Base
from xagent.web.models.task import Task, TraceCheckpointBlob, TraceMessageBlob
from xagent.web.models.user import User
from xagent.web.services import trace_message_storage as storage

pytestmark = pytest.mark.postgresql


@pytest.fixture
def engine():
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    schema = "trace_blob_test_" + uuid.uuid4().hex
    admin = create_engine(url)
    with admin.begin() as db:
        db.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        Base.metadata.create_all(engine)
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as db:
            db.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def make_task(db):
    user = User(username=uuid.uuid4().hex, password_hash="unused")
    db.add(user)
    db.flush()
    task = Task(user_id=user.id, title="Blob test", description="Blob test")
    db.add(task)
    db.commit()
    return int(task.id)


@pytest.fixture(params=["message", "checkpoint"])
def kind(request):
    return request.param


def upsert(db, task_id, kind, values):
    candidates = {
        key: storage.BlobCandidate(data=value, payload_bytes=len(value))
        for key, value in values.items()
    }
    if kind == "message":
        storage._upsert_message_blobs(
            db, task_id=task_id, execution_id="exec", blobs_by_hash=candidates
        )
    else:
        storage._upsert_checkpoint_blobs(
            db,
            task_id=task_id,
            execution_id="exec",
            blobs_by_ref={
                ("metadata", key): value for key, value in candidates.items()
            },
        )


def capture_inserts(monkeypatch):
    original = storage._insert_blob_rows_on_conflict_do_nothing
    counts = []

    def insert(db, **kwargs):
        counts.append(len(kwargs["values"]))
        return original(db, **kwargs)

    monkeypatch.setattr(storage, "_insert_blob_rows_on_conflict_do_nothing", insert)
    return counts


def test_only_missing_payloads_are_inserted_and_rollback_is_not_cached(
    engine, kind, monkeypatch
):
    counts = capture_inserts(monkeypatch)
    # Use fresh sessions after commits, as the actual trace worker does.
    with Session(engine, autoflush=False) as db:
        task_id = make_task(db)
        upsert(db, task_id, kind, {"old": "old-data"})
        db.commit()
    with Session(engine, autoflush=False) as db:
        upsert(db, task_id, kind, {"old": "old-data"})
        db.commit()
        upsert(db, task_id, kind, {"old": "old-data", "new": "new-data"})
        db.rollback()
    with Session(engine, autoflush=False) as db:
        upsert(db, task_id, kind, {"old": "old-data", "new": "new-data"})
        db.commit()
    assert counts == [1, 0, 1, 1]


def test_existing_size_mismatch_still_fails(engine, kind, monkeypatch):
    with Session(engine, autoflush=False) as db:
        task_id = make_task(db)
        upsert(db, task_id, kind, {"hash": "original"})
        db.commit()
        counts = capture_inserts(monkeypatch)
        with pytest.raises(ValueError, match="hash collision"):
            upsert(db, task_id, kind, {"hash": "wrong-size"})
        assert counts == []


def test_missed_insert_still_fails_validation(engine, kind, monkeypatch):
    with Session(engine, autoflush=False) as db:
        task_id = make_task(db)
        monkeypatch.setattr(
            storage, "_insert_blob_rows_on_conflict_do_nothing", lambda *a, **kw: True
        )
        with pytest.raises(RuntimeError, match="did not persist"):
            upsert(db, task_id, kind, {"missing": "data"})


def test_lookup_is_task_scoped(engine, kind, monkeypatch):
    counts = capture_inserts(monkeypatch)
    with Session(engine, autoflush=False) as db:
        first = make_task(db)
        second = make_task(db)
        upsert(db, first, kind, {"same-hash": "data"})
        db.commit()
        upsert(db, second, kind, {"same-hash": "data"})
        db.commit()
    assert counts == [1, 1]


def test_checkpoint_lookup_is_kind_scoped(engine, monkeypatch):
    counts = capture_inserts(monkeypatch)
    with Session(engine, autoflush=False) as db:
        task_id = make_task(db)
        candidate = storage.BlobCandidate(data="data", payload_bytes=4)
        for kind in ["metadata", "system_prompt"]:
            storage._upsert_checkpoint_blobs(
                db,
                task_id=task_id,
                execution_id="exec",
                blobs_by_ref={(kind, "same-hash"): candidate},
            )
            db.commit()
    assert counts == [1, 1]


@pytest.mark.parametrize("collision", [False, True])
def test_concurrent_misses_keep_conflict_and_size_checks(
    engine, kind, collision, monkeypatch
):
    with Session(engine, autoflush=False) as db:
        task_id = make_task(db)
        upsert(db, task_id, kind, {"old": "old-data"})
        db.commit()
    barrier = Barrier(2)
    original = storage._insert_blob_rows_on_conflict_do_nothing

    def insert(db, **kwargs):
        # Both workers have completed their lookup and see the new hash absent.
        assert len(kwargs["values"]) == 1
        barrier.wait(timeout=10)
        return original(db, **kwargs)

    monkeypatch.setattr(storage, "_insert_blob_rows_on_conflict_do_nothing", insert)

    def worker(index):
        with Session(engine, autoflush=False) as db:
            try:
                value = "different-size" if collision and index else "new-data"
                upsert(db, task_id, kind, {"old": "old-data", "new": value})
                db.commit()
                return "ok"
            except ValueError as exc:
                db.rollback()
                assert "hash collision" in str(exc)
                return "collision"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, index) for index in range(2)]
        outcomes = sorted(future.result(timeout=20) for future in futures)
    assert outcomes == (["collision", "ok"] if collision else ["ok", "ok"])
    model = TraceMessageBlob if kind == "message" else TraceCheckpointBlob
    with Session(engine) as db:
        assert db.query(model).filter_by(task_id=task_id).count() == 2
