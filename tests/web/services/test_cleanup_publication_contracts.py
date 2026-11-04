"""Managed read contention preserves HTTP and transaction contracts."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from filelock import FileLock, Timeout

from tests.web.services.test_complete_upload_cleanup import (
    lifecycle as cleanup_lifecycle_fixture,
)
from xagent.core.tools.core.RAG_tools.storage.file_reference import file_cleanup_lock
from xagent.web.api import files
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.managed_file_ref import (
    DurableStorageOperationError,
    ManagedFileRef,
)
from xagent.web.services.uploaded_file_cleanup_publication import (
    FilePublicationUnavailable,
)

lifecycle = cleanup_lifecycle_fixture


@contextmanager
def held_cleanup_lock(file_id):
    entered, release = Event(), Event()

    def hold():
        with file_cleanup_lock(file_id):
            entered.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(hold)
        assert entered.wait(5)
        try:
            yield
        finally:
            release.set()
        worker.result(timeout=5)


@pytest.mark.asyncio
async def test_actual_preview_lock_wait_does_not_stop_event_loop(
    lifecycle, monkeypatch
):
    sessions, source, materialized, _, _, file_id = lifecycle
    source.unlink()
    materialized.unlink()
    original = FileLock.acquire

    def bounded(lock, *args, **kwargs):
        if str(lock.lock_file).endswith(".cleanup.lock"):
            kwargs["timeout"] = 0.15
        return original(lock, *args, **kwargs)

    monkeypatch.setattr(FileLock, "acquire", bounded)
    ticks = []
    finished = asyncio.Event()

    async def heartbeat():
        while not finished.is_set():
            await asyncio.sleep(0.01)
            if not finished.is_set():
                ticks.append(1)

    with held_cleanup_lock(file_id), sessions() as db:
        task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        try:
            with pytest.raises(HTTPException) as fault:
                await files.preview_file(
                    file_id, auth=(SimpleNamespace(id=1), False), db=db
                )
            assert fault.value.status_code == 503
        finally:
            finished.set()
            await task
    assert len(ticks) >= 3


def test_cached_managed_read_validates_without_waiting_for_execution(lifecycle):
    sessions, source, _, _, _, file_id = lifecycle
    with held_cleanup_lock(file_id), sessions() as db:
        ref = ManagedFileRef(db.query(UploadedFile).one())
        assert ref.ensure_local() == source
        assert ref.materialize() == source
    with sessions.begin() as db:
        db.query(UploadedFile).one().storage_status = "compensating"
    with sessions() as db:
        with pytest.raises(FilePublicationUnavailable):
            ManagedFileRef(db.query(UploadedFile).one()).ensure_local()


def test_dirty_transaction_keeps_changes_and_reports_typed_publication_error(lifecycle):
    sessions, source, _, _, _, _ = lifecycle
    with sessions() as db:
        row = db.query(UploadedFile).one()
        user = db.get(User, 1)
        user.username = "pending-owner-change"
        assert ManagedFileRef(row).ensure_local() == source
        source.unlink()
        with pytest.raises(DurableStorageOperationError):
            ManagedFileRef(row).ensure_local()
        assert user in db.dirty
        db.commit()
    with sessions() as db:
        assert db.get(User, 1).username == "pending-owner-change"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"storage_status": "compensating"},
        {"filename": "new-name.txt"},
        {"storage_backend": "s3"},
        {"storage_uri": "s3://new-publication/key"},
        {"etag": "replacement-etag"},
    ],
    ids=["claimed", "renamed", "backend", "locator", "previously-null-token"],
)
async def test_stale_preview_snapshot_does_not_fall_back_to_local_bytes(
    lifecycle, changes
):
    sessions, source, _, _, _, file_id = lifecycle
    with sessions() as db:
        stale = db.query(UploadedFile).one()
        with sessions.begin() as changed:
            row = changed.query(UploadedFile).one()
            for field, value in changes.items():
                setattr(row, field, value)
        assert stale.storage_status == "available"
        with pytest.raises(HTTPException) as fault:
            await files.preview_file(
                file_id, auth=(SimpleNamespace(id=1), False), db=db
            )
        assert fault.value.status_code == 404
        assert isinstance(fault.value.__cause__, FilePublicationUnavailable)
    assert source.exists()


@pytest.mark.asyncio
async def test_managed_pptx_conversion_allows_cached_reads_and_releases_connections(
    lifecycle, monkeypatch
):
    sessions, source, _, previews, _, file_id = lifecycle
    pptx = source.with_suffix(".pptx")
    source.rename(pptx)
    previews[0].unlink()
    with sessions.begin() as db:
        row = db.query(UploadedFile).one()
        row.storage_path, row.filename = str(pptx), pptx.name
    entered, release = asyncio.Event(), asyncio.Event()
    checked_out = [0]

    @sa.event.listens_for(sessions.kw["bind"], "checkout")
    def checkout(*_):
        checked_out[0] += 1

    @sa.event.listens_for(sessions.kw["bind"], "checkin")  # codespell:ignore checkin
    def returned(*_):
        checked_out[0] -= 1

    async def convert(*args, **kwargs):
        output = files.Path(args[args.index("--outdir") + 1])

        class Process:
            pid = 123_456_789
            returncode = None

            async def wait(self):
                entered.set()
                await release.wait()
                (output / "source.pdf").write_bytes(b"converted preview")
                self.returncode = 0
                return self.returncode

        return Process()

    monkeypatch.setattr(files.pptx_conversion.os, "killpg", lambda _pid, _signal: None)
    monkeypatch.setattr(files.asyncio, "create_subprocess_exec", convert)
    with sessions() as converting_db, sessions() as reading_db:
        task = asyncio.create_task(
            files.preview_pptx_as_pdf(
                file_id, user=SimpleNamespace(id=1), db=converting_db
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            assert checked_out[0] == 0
            response = await asyncio.wait_for(
                files.preview_file(
                    file_id, auth=(SimpleNamespace(id=1), False), db=reading_db
                ),
                timeout=2,
            )
            assert response.status_code == 200
            assert checked_out[0] == 0
        finally:
            release.set()
            response = await task
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_cancelled_restore_drains_native_copy_before_releasing_guard(
    lifecycle, monkeypatch
):
    from xagent.core.file_storage import storage
    from xagent.web.services.uploaded_file_cleanup_publication import async_managed_copy

    sessions, source, _, _, _, file_id = lifecycle
    source.unlink()
    entered, release = Event(), Event()
    original = storage.shutil.copyfileobj

    def copy_bytes(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage.shutil, "copyfileobj", copy_bytes)
    with sessions() as db:
        ref = ManagedFileRef(db.query(UploadedFile).one())
        task = asyncio.create_task(async_managed_copy(ref, restore_local=True))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            acquire = FileLock.acquire
            monkeypatch.setattr(
                FileLock,
                "acquire",
                lambda lock, *args, **kwargs: acquire(
                    lock, *args, **{**kwargs, "timeout": 0}
                ),
            )
            with pytest.raises(Timeout):
                with file_cleanup_lock(file_id):
                    pass
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert source.exists()
