"""Focused compensation behavior at the KB reference coordination boundary."""

from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Event, get_ident

import pytest
from sqlalchemy import event

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from tests.web.services.test_kb_reference_cleanup import boundary as _boundary
from tests.web.services.test_kb_reference_cleanup import (
    document,
)
from xagent.web.models.background_job import BackgroundJob
from xagent.web.models.kb_ingest_target import KBIngestTarget
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.kb_reference_protection import FileReferenceConflict
from xagent.web.services.uploaded_file_store import (
    RegisteredUploadCompensationClaim,
    compensate_registered_uploads_sync,
)

boundary = _boundary


def _add_upload(sessions, tmp_path, file_id: str) -> None:
    path = tmp_path / f"{file_id}.txt"
    path.write_text(file_id)
    with sessions.begin() as db:
        db.add(
            UploadedFile(
                user_id=1,
                file_id=file_id,
                filename=path.name,
                storage_path=str(path),
                storage_key=f"users/1/uploads/{file_id}/{file_id}.txt",
                checksum=hashlib.sha256(file_id.encode()).hexdigest(),
                storage_status="available",
                created_at=datetime.now(UTC),
            )
        )


def _claim(file_id: str) -> RegisteredUploadCompensationClaim:
    return RegisteredUploadCompensationClaim(
        user_id=1,
        file_id=file_id,
        expected_task_id=None,
        expected_storage_key=(
            "users/1/uploads/source/source.txt"
            if file_id == "source"
            else f"users/1/uploads/{file_id}/{file_id}.txt"
        ),
    )


def test_referenced_compensation_is_logged_while_eligible_sibling_finishes(
    boundary, tmp_path, monkeypatch, caplog
):
    sessions, store, _path = boundary
    _add_upload(sessions, tmp_path, "sibling")
    store.upsert_documents([document()])
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.delete_uploaded_file_compensation_object",
        lambda **_kwargs: "absent",
    )

    compensate_registered_uploads_sync([_claim("source"), _claim("sibling")])

    with sessions() as db:
        states = {
            row.file_id: row.storage_status for row in db.query(UploadedFile).all()
        }
    assert states == {"source": "available"}
    assert "referenced KB source source" in caplog.text


def test_compensation_releases_reference_lock_before_durable_delete(
    boundary, monkeypatch
):
    sessions, store, _path = boundary
    delete_started = Event()
    finish_delete = Event()

    def blocked_delete(**_kwargs):
        delete_started.set()
        assert finish_delete.wait(DB_PROGRESS_TIMEOUT)
        return "absent"

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.delete_uploaded_file_compensation_object",
        blocked_delete,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        compensation = pool.submit(
            compensate_registered_uploads_sync, [_claim("source")]
        )
        assert delete_started.wait(DB_PROGRESS_TIMEOUT)
        publication = pool.submit(store.upsert_documents, [document()])
        try:
            with pytest.raises(FileReferenceConflict):
                publication.result(timeout=DB_PROGRESS_TIMEOUT)
        finally:
            finish_delete.set()
        compensation.result(timeout=DB_PROGRESS_TIMEOUT)


def test_async_admission_drains_worker_owned_session_before_cancellation_returns(
    boundary, monkeypatch
):
    from xagent.web.services import kb_reference_protection
    from xagent.web.services.background_jobs import create_background_job

    sessions, _store, path = boundary
    lock_started = Event()
    release_lock = Event()
    lock_threads: list[int] = []

    @contextmanager
    def blocked_lock(_file_ids):
        lock_threads.append(get_ident())
        lock_started.set()
        assert release_lock.wait(DB_PROGRESS_TIMEOUT)
        yield

    monkeypatch.setattr(kb_reference_protection, "file_reference_lock", blocked_lock)

    async def scenario() -> None:
        with sessions() as db:
            job = create_background_job(
                db,
                user_id=1,
                job_type="kb.ingest.document",
                payload={},
            )
            caller_commits = 0

            def committed(_session):
                nonlocal caller_commits
                caller_commits += 1

            event.listen(db, "after_commit", committed)
            try:
                admission = asyncio.create_task(
                    kb_reference_protection.async_admit_kb_ingest_target(
                        db,
                        user_id=1,
                        collection="kb",
                        target_path=str(path),
                        file_id="source",
                        generation_id="one",
                        job_id=str(job.id),
                        file_sha256="hash",
                    )
                )
                assert await asyncio.to_thread(lock_started.wait, 5)
                admission.cancel()
                await asyncio.sleep(0)
                assert not admission.done()
                release_lock.set()
                with pytest.raises(asyncio.CancelledError):
                    await admission
            finally:
                event.remove(db, "after_commit", committed)
            assert caller_commits == 0

    event_loop_thread = get_ident()
    asyncio.run(scenario())
    assert lock_threads and lock_threads[0] != event_loop_thread
    with sessions() as db:
        target = db.query(KBIngestTarget).one()
        assert target.latest_generation_id == "one"
        assert target.deleted_at is not None
        job = db.query(BackgroundJob).one()
        assert job.status == "failed"
        assert job.error_message == "KB ingest admission was cancelled"


def test_cancelled_settlement_does_not_freeze_admission_or_shutdown():
    import subprocess
    import sys

    script = """
import asyncio
import hashlib
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from xagent.web.services import kb_reference_protection as protection

async def scenario():
    started, resume = asyncio.Event(), asyncio.Event()
    settlement = asyncio.get_running_loop().create_future()

    async def worker(operation):
        if not started.is_set():
            started.set()
            await resume.wait()
            return None
        settlement.set_result(asyncio.current_task())
        await asyncio.Event().wait()

    protection.run_in_threadpool = worker
    engine = create_engine("sqlite://")
    with Session(engine) as db:
        admission = asyncio.create_task(protection.async_admit_kb_ingest_target(db))
        await started.wait()
        admission.cancel()
        resume.set()
        (await settlement).cancel()
        try:
            await admission
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Admission lost the request cancellation")
    engine.dispose()

asyncio.run(scenario())
"""
    # Shutdown can cancel both owned tasks; bound a hot-loop regression externally.
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    exit_code = result.returncode
    assert exit_code == 0, result.stderr
