"""Materialized reads stay lock-free without weakening publication fencing."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest
import sqlalchemy as sa
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
from xagent.web.services.uploaded_file_store import (
    UploadedFileStore,
    snapshot_uploaded_file_version,
)

lifecycle = cleanup_lifecycle_fixture


def _durable_path(row: UploadedFile) -> Path:
    return Path(unquote(urlparse(str(row.storage_uri)).path))


@contextmanager
def _held_execution_lock(file_id: str):
    entered, release = Event(), Event()

    def hold() -> None:
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


@contextmanager
def _bounded_execution_wait(monkeypatch):
    original_acquire = FileLock.acquire

    def bounded_acquire(lock, *args, **kwargs):
        if str(lock.lock_file).endswith(".cleanup.lock"):
            kwargs["timeout"] = 0.05
        return original_acquire(lock, *args, **kwargs)

    with monkeypatch.context() as bounded:
        bounded.setattr(FileLock, "acquire", bounded_acquire)
        yield


@pytest.mark.parametrize("allow_existing_local", [True, False])
def test_materialized_cache_hit_does_not_wait_for_execution_lock(
    lifecycle, monkeypatch, allow_existing_local
):
    sessions, source, materialized, _, _, file_id = lifecycle
    source.unlink()
    with sessions() as db:
        ref = ManagedFileRef(db.query(UploadedFile).one())

    with _held_execution_lock(file_id), _bounded_execution_wait(monkeypatch):
        assert (
            ref.materialize(allow_existing_local=allow_existing_local) == materialized
        )

    assert not source.exists()
    assert ref.ensure_local() == source


@pytest.mark.asyncio
async def test_preview_returns_materialized_cache_while_execution_lock_is_held(
    lifecycle, monkeypatch
):
    sessions, source, materialized, _, _, file_id = lifecycle
    source.unlink()
    with sessions() as db:
        durable = _durable_path(db.query(UploadedFile).one())

    checked_out = [0]

    @sa.event.listens_for(sessions.kw["bind"], "checkout")
    def checkout(*_args):
        checked_out[0] += 1

    @sa.event.listens_for(sessions.kw["bind"], "checkin")  # codespell:ignore checkin
    def returned(*_args):
        checked_out[0] -= 1

    hashing, release_hash = Event(), Event()
    original_open = Path.open

    def paused_open(path, *args, **kwargs):
        if path == durable:
            hashing.set()
            assert release_hash.wait(10)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", paused_open)
    with (
        _held_execution_lock(file_id),
        _bounded_execution_wait(monkeypatch),
        sessions() as db,
    ):
        response_task = asyncio.create_task(
            files.preview_file(file_id, auth=(SimpleNamespace(id=1), False), db=db)
        )
        try:
            assert await asyncio.to_thread(hashing.wait, 5)
            assert checked_out[0] == 0
        finally:
            release_hash.set()
        response = await asyncio.wait_for(response_task, timeout=5)

    assert Path(response.path) == materialized


@pytest.mark.parametrize(
    "changes",
    [{"storage_status": "compensating"}, {"filename": "replacement.txt"}],
    ids=["claimed", "new-generation"],
)
def test_stale_snapshot_rejects_materialized_cache_without_waiting(
    lifecycle, monkeypatch, changes
):
    sessions, source, _, _, _, file_id = lifecycle
    source.unlink()
    with sessions() as db:
        ref = ManagedFileRef(db.query(UploadedFile).one())
        db.expunge_all()
    with sessions.begin() as db:
        row = db.query(UploadedFile).one()
        for field, value in changes.items():
            setattr(row, field, value)

    with _held_execution_lock(file_id), _bounded_execution_wait(monkeypatch):
        with pytest.raises(FilePublicationUnavailable):
            ref.materialize(allow_existing_local=False)


def test_generation_is_rechecked_after_the_cache_probe(lifecycle, monkeypatch):
    sessions, source, _, _, _, file_id = lifecycle
    source.unlink()
    with sessions() as db:
        row = db.query(UploadedFile).one()
        durable = _durable_path(row)
        ref = ManagedFileRef(row)
        db.expunge_all()

    entered, release = Event(), Event()
    original_open = Path.open

    def paused_open(path, *args, **kwargs):
        if path == durable:
            entered.set()
            assert release.wait(10)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", paused_open)
    with _held_execution_lock(file_id), _bounded_execution_wait(monkeypatch):
        with ThreadPoolExecutor(max_workers=1) as pool:
            read = pool.submit(ref.materialize, allow_existing_local=False)
            try:
                assert entered.wait(5)
                with sessions.begin() as db:
                    db.query(UploadedFile).one().etag = "replacement-etag"
            finally:
                release.set()
            with pytest.raises(FilePublicationUnavailable):
                read.result(timeout=5)


def test_corrupt_cache_waits_for_guard_before_repair(lifecycle, monkeypatch):
    sessions, source, materialized, _, _, file_id = lifecycle
    source.unlink()
    materialized.write_bytes(b"corrupt cache")
    with sessions() as db:
        ref = ManagedFileRef(db.query(UploadedFile).one())

    with _held_execution_lock(file_id), _bounded_execution_wait(monkeypatch):
        with pytest.raises(DurableStorageOperationError) as fault:
            ref.materialize(allow_existing_local=False)
        assert isinstance(fault.value.__cause__, Timeout)
        assert materialized.read_bytes() == b"corrupt cache"

    assert ref.materialize(allow_existing_local=False) == materialized
    assert materialized.read_bytes() != b"corrupt cache"


def test_dirty_transaction_rejects_cache_probe_without_discarding_changes(
    lifecycle, monkeypatch
):
    sessions, source, _, _, _, _ = lifecycle
    source.unlink()
    with sessions() as db:
        user = db.get(User, 1)
        user.username = "pending-owner-change"
        row = db.query(UploadedFile).one()
        durable = _durable_path(row)
        ref = ManagedFileRef(row)
        durable_reads = []
        original_open = Path.open

        def observe_open(path, *args, **kwargs):
            if path == durable:
                durable_reads.append(path)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", observe_open)

        with pytest.raises(DurableStorageOperationError, match="clean transaction"):
            ref.materialize(allow_existing_local=False)
        assert durable_reads == []
        assert user in db.dirty
        db.commit()

    with sessions() as db:
        assert db.get(User, 1).username == "pending-owner-change"


def test_deleted_row_id_snapshot_rejects_surviving_local_path(lifecycle):
    sessions, source, _, _, _, _ = lifecycle
    callbacks = []
    with sessions.begin() as db:
        row = db.query(UploadedFile).one()
        ref = ManagedFileRef(snapshot_uploaded_file_version(row))
        UploadedFileStore(db).delete(
            row,
            delete_local=False,
            after_commit=callbacks,
        )

    assert source.is_file()
    with pytest.raises(FilePublicationUnavailable):
        ref.ensure_local()


def test_row_id_snapshot_compares_previously_null_generation_fields(lifecycle):
    sessions, source, _, _, _, _ = lifecycle
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.etag is None
        ref = ManagedFileRef(snapshot_uploaded_file_version(row))
    with sessions.begin() as db:
        db.query(UploadedFile).one().etag = "replacement-etag"

    assert source.is_file()
    with pytest.raises(FilePublicationUnavailable):
        ref.ensure_local()
