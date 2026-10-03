"""Connection-ownership tests for the three shared upload entry points."""

from __future__ import annotations

import asyncio
import builtins
import io
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.datastructures import UploadFile

from tests.shared.execution_scope import register_scope_resolver
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.storage import FsspecFileStorage
from xagent.core.workspace import TaskWorkspace
from xagent.web.api import files as files_api
from xagent.web.api.public_chat_access import (
    PublicChatAccessContext,
    ShareChatAccessContext,
    upload_public_chat_files,
    upload_share_chat_files,
)
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services.managed_file_ref import (
    DURABLE_FAULT_LOG_PREFIX,
    DurableStorageOperationError,
    ManagedFileRef,
)

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _install_one_slot_queue_pool,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture()
def isolated_upload_storage(monkeypatch: pytest.MonkeyPatch, tmp_path):
    upload_root = tmp_path / "uploads"
    object_root = tmp_path / "objects"
    upload_root.mkdir()
    monkeypatch.setenv(
        "XAGENT_FILE_STORAGE_URI",
        object_root.as_uri(),
    )
    get_unscoped_file_storage.cache_clear()
    monkeypatch.setattr(files_api, "get_uploads_dir", lambda: upload_root)
    try:
        yield upload_root, object_root
    finally:
        get_unscoped_file_storage.cache_clear()


def _record_durable_pool_ownership(
    monkeypatch: pytest.MonkeyPatch,
    engine,
) -> list[int]:
    checked_out: list[int] = []
    original_sync = ManagedFileRef.sync_to_durable

    def recording_sync(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        checked_out.append(engine.pool.checkedout())
        return original_sync(self, *args, **kwargs)

    monkeypatch.setattr(ManagedFileRef, "sync_to_durable", recording_sync)
    return checked_out


def _stage_path_in(root: Path, user_id: int):
    """Return a shared deterministic staging candidate for collision tests."""

    def get_upload_path(
        filename: str,
        task_id: str | None,
        folder: str | None,
        requested_user_id: int,
    ) -> Path:
        del task_id, folder
        assert requested_user_id == user_id
        path = root / f"user_{user_id}" / Path(filename).name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    return get_upload_path


class _SlowTrackingSource(io.BytesIO):
    """A synchronous source that reveals an accidental event-loop read."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.started = threading.Event()
        self.release = threading.Event()
        self.loop_thread = threading.get_ident()
        self.read_threads: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_threads.append(threading.get_ident())
        self.started.set()
        assert threading.get_ident() != self.loop_thread
        assert self.release.wait(timeout=30), "upload read was never released"
        return super().read(size)


class _BlockingSource(io.BytesIO):
    """Hold a worker-owned read until a cancellation test releases it."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.started = threading.Event()
        self.release = threading.Event()

    def read(self, size: int = -1) -> bytes:
        self.started.set()
        assert self.release.wait(timeout=2)
        return super().read(size)


def _admin_user_id() -> int:
    # The helper also creates the admin record when the test database is empty.
    _admin_headers()
    db = _direct_db_session()
    try:
        return int(db.query(User.id).filter(User.username == "admin").scalar())
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("evicted_base", [False, True])
async def test_concurrent_same_name_uploads_reserve_distinct_paths_and_contents(
    monkeypatch: pytest.MonkeyPatch,
    isolated_upload_storage,
    evicted_base: bool,
) -> None:
    """The shared staging boundary must make naming reservation atomic."""

    upload_root, _object_root = isolated_upload_storage
    user_id = _admin_user_id()
    monkeypatch.setattr(
        files_api, "get_upload_path", _stage_path_in(upload_root, user_id)
    )

    base_path = upload_root / f"user_{user_id}" / "same-name.txt"
    original_upload = None
    if evicted_base:
        original_upload = await files_api.store_uploaded_files(
            upload_items=[
                UploadFile(filename="same-name.txt", file=io.BytesIO(b"original"))
            ],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        )
        base_path.unlink()

    race_path = base_path.with_name("same-name_1.txt") if evicted_base else base_path
    original_lookup = files_api.upload_path_is_registered_sync
    workers_started = 0
    workers_ready = threading.Barrier(2)
    workers_lock = threading.Lock()

    def synchronized_lookup(path: Path) -> bool:
        nonlocal workers_started
        registered = original_lookup(path)
        if path == race_path:
            assert not registered
            with workers_lock:
                workers_started += 1
            # Both workers finish the pre-check before either can reserve the
            # candidate. Exclusive creation must resolve the actual race.
            workers_ready.wait(timeout=5)
        return registered

    monkeypatch.setattr(
        files_api,
        "upload_path_is_registered_sync",
        synchronized_lookup,
    )
    first = UploadFile(
        filename="same-name.txt",
        file=io.BytesIO(b"first exact contents"),
        headers={"content-type": "text/plain"},
    )
    second = UploadFile(
        filename="same-name.txt",
        file=io.BytesIO(b"second exact contents"),
        headers={"content-type": "text/plain"},
    )

    await asyncio.gather(
        files_api.store_uploaded_files(
            upload_items=[first],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        ),
        files_api.store_uploaded_files(
            upload_items=[second],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        ),
    )

    staged = sorted((upload_root / f"user_{user_id}").glob("same-name*.txt"))
    assert workers_started == 2
    assert len(staged) == 2
    assert {path.read_bytes() for path in staged} == {
        b"first exact contents",
        b"second exact contents",
    }
    if original_upload is not None:
        assert not base_path.exists()
        assert base_path not in staged
        response = client.get(
            f"/api/files/download/{original_upload['file_id']}",
            headers=_admin_headers(),
        )
        assert response.status_code == 200
        assert response.content == b"original"
        assert {path.read_bytes() for path in staged} == {
            b"first exact contents",
            b"second exact contents",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, FileExistsError])
async def test_upload_lookup_failure_propagates_without_staging_bytes(
    monkeypatch: pytest.MonkeyPatch,
    isolated_upload_storage,
    error_type: type[Exception],
) -> None:
    """A lookup failure is not a filename collision and must fail closed."""

    upload_root, object_root = isolated_upload_storage
    user_id = _admin_user_id()
    monkeypatch.setattr(
        files_api, "get_upload_path", _stage_path_in(upload_root, user_id)
    )

    def failing_lookup(_path: Path) -> bool:
        raise error_type("lookup unavailable")

    monkeypatch.setattr(files_api, "upload_path_is_registered_sync", failing_lookup)
    source = io.BytesIO(b"must not be staged")
    with pytest.raises(error_type, match="lookup unavailable"):
        await files_api.store_uploaded_files(
            upload_items=[UploadFile(filename="lookup-failure.txt", file=source)],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        )
    assert source.tell() == 0
    assert not list(upload_root.rglob("lookup-failure*"))
    assert not list(object_root.rglob("lookup-failure*"))
    with _direct_db_session() as db:
        assert db.query(UploadedFile).count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path_spelling", ["plain", "symlink", "parent-components"])
async def test_same_name_upload_preserves_registered_file_after_cache_eviction(
    monkeypatch: pytest.MonkeyPatch,
    isolated_upload_storage,
    path_spelling: str,
) -> None:
    """An absent local cache is not an unclaimed upload path."""

    upload_root, _object_root = isolated_upload_storage
    staging_root = upload_root
    if path_spelling == "symlink":
        staging_root = upload_root.with_name("upload-alias")
        staging_root.symlink_to(upload_root, target_is_directory=True)
    elif path_spelling == "parent-components":
        intermediate = upload_root / "nested"
        intermediate.mkdir()
        staging_root = intermediate / ".."
    if path_spelling != "plain":
        assert str(staging_root) != str(staging_root.resolve())
    user_id = _admin_user_id()
    monkeypatch.setattr(
        files_api, "get_upload_path", _stage_path_in(staging_root, user_id)
    )

    async def upload(payload: bytes):
        return await files_api.store_uploaded_files(
            upload_items=[
                UploadFile(filename="same-name.txt", file=io.BytesIO(payload))
            ],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        )

    originals = []
    for payload in (b"first contents", b"second contents"):
        result = await upload(payload)
        with _direct_db_session() as db:
            record = (
                db.query(UploadedFile)
                .filter(UploadedFile.file_id == result["file_id"])
                .one()
            )
            original_path = Path(record.storage_path)
        # Registration preserves the same spelling used by the reservation check.
        assert original_path.parent == staging_root / f"user_{user_id}"
        originals.append((result["file_id"], original_path, payload))
        original_path.unlink()

    new_upload = await upload(b"new contents")
    with _direct_db_session() as db:
        new_record = (
            db.query(UploadedFile)
            .filter(UploadedFile.file_id == new_upload["file_id"])
            .one()
        )
        new_path = Path(new_record.storage_path)
        assert db.query(UploadedFile).count() == 3
    assert new_path not in [path for _, path, _ in originals]
    assert new_path.read_bytes() == b"new contents"

    headers = _admin_headers()
    for file_id, original_path, payload in originals:
        assert file_id != new_upload["file_id"]
        assert not original_path.exists()
        response = client.get(f"/api/files/download/{file_id}", headers=headers)
        assert response.status_code == 200
        assert response.content == payload
        assert original_path.read_bytes() == payload
    assert new_path.read_bytes() == b"new contents"


@pytest.mark.asyncio
async def test_staging_copy_reads_off_loop_with_no_database_checkout(
    monkeypatch: pytest.MonkeyPatch,
    isolated_upload_storage,
) -> None:
    """Slow synchronous upload reads must not stall the request loop or hold DB."""

    upload_root, _object_root = isolated_upload_storage
    user_id = _admin_user_id()
    monkeypatch.setattr(
        files_api, "get_upload_path", _stage_path_in(upload_root, user_id)
    )
    engine = _install_one_slot_queue_pool(monkeypatch)
    source = _SlowTrackingSource(b"bounded-copy")
    upload = UploadFile(
        filename="off-loop.txt",
        file=source,
        headers={"content-type": "text/plain"},
    )
    loop_thread = threading.get_ident()
    task = asyncio.create_task(
        files_api.store_uploaded_files(
            upload_items=[upload],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        )
    )
    try:
        assert await asyncio.to_thread(source.started.wait, 30)
        assert not task.done()
        assert engine.pool.checkedout() == 0
        source.release.set()
        await asyncio.wait_for(task, timeout=30)
        assert source.read_threads
        assert all(thread_id != loop_thread for thread_id in source.read_threads)
        assert engine.pool.checkedout() == 0
    finally:
        source.release.set()
        try:
            await asyncio.wait_for(task, timeout=30)
        finally:
            engine.dispose()


def test_reserve_and_copy_enforces_max_size_and_cleans_partial_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The worker returns the path and removes MAX+1 partial output."""

    monkeypatch.setattr(files_api, "MAX_FILE_SIZE", 4)
    monkeypatch.setattr(files_api, "MAX_FILE_SIZE_LABEL", "4B")
    user_id = 7
    monkeypatch.setattr(files_api, "get_upload_path", _stage_path_in(tmp_path, user_id))

    exact_path = files_api._reserve_and_copy_upload(
        UploadFile(filename="exact.txt", file=io.BytesIO(b"1234")),
        task_id=None,
        folder=None,
        user_id=user_id,
    )
    assert exact_path.read_bytes() == b"1234"

    with pytest.raises(HTTPException, match="maximum limit"):
        files_api._reserve_and_copy_upload(
            UploadFile(filename="too-large.txt", file=io.BytesIO(b"12345")),
            task_id=None,
            folder=None,
            user_id=user_id,
        )
    assert not (tmp_path / f"user_{user_id}" / "too-large.txt").exists()


def test_reserve_and_copy_does_not_mistake_source_file_exists_error_for_collision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only an exclusive-open collision may select the next candidate."""

    class SourceThatFailsWithFileExists:
        def read(self, _size: int = -1) -> bytes:
            raise FileExistsError("source read failure")

    user_id = 7
    monkeypatch.setattr(files_api, "get_upload_path", _stage_path_in(tmp_path, user_id))
    with pytest.raises(FileExistsError, match="source read failure"):
        files_api._reserve_and_copy_upload(
            UploadFile(
                filename="source-error.txt", file=SourceThatFailsWithFileExists()
            ),
            task_id=None,
            folder=None,
            user_id=user_id,
        )
    assert not list(tmp_path.rglob("source-error*"))


def test_reserve_and_copy_cleans_exact_file_after_write_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """I/O errors after reservation must not leave an empty staged file."""

    user_id = 7
    monkeypatch.setattr(files_api, "get_upload_path", _stage_path_in(tmp_path, user_id))
    original_open = builtins.open

    class FailingWriter:
        def __init__(self, buffer) -> None:  # type: ignore[no-untyped-def]
            self.buffer = buffer

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
            self.buffer.close()

        def write(self, _chunk: bytes) -> int:
            raise OSError("write failed")

    def failing_open(path, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        buffer = original_open(path, mode, *args, **kwargs)
        return FailingWriter(buffer) if "x" in mode else buffer

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(OSError, match="write failed"):
        files_api._reserve_and_copy_upload(
            UploadFile(filename="write-error.txt", file=io.BytesIO(b"payload")),
            task_id=None,
            folder=None,
            user_id=user_id,
        )
    assert not (tmp_path / f"user_{user_id}" / "write-error.txt").exists()


def test_reserve_and_copy_propagates_open_error_without_creating_an_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exclusive-open failures are I/O failures, never candidate collisions."""

    user_id = 7
    monkeypatch.setattr(files_api, "get_upload_path", _stage_path_in(tmp_path, user_id))

    original_open = builtins.open

    def failing_open(path, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        if "x" in mode:
            raise PermissionError("open denied")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(PermissionError, match="open denied"):
        files_api._reserve_and_copy_upload(
            UploadFile(filename="open-error.txt", file=io.BytesIO(b"payload")),
            task_id=None,
            folder=None,
            user_id=user_id,
        )
    assert not list(tmp_path.rglob("open-error*"))


@pytest.mark.parametrize(
    "control_error",
    [SystemExit("stop"), KeyboardInterrupt()],
    ids=["system-exit", "keyboard-interrupt"],
)
def test_delete_staged_upload_preserves_process_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    control_error: BaseException,
) -> None:
    """Best-effort cleanup must not suppress process-control exceptions."""

    def interrupting_unlink(
        _path: Path,
        *,
        missing_ok: bool = False,
    ) -> None:
        del missing_ok
        raise control_error

    monkeypatch.setattr(Path, "unlink", interrupting_unlink)
    with pytest.raises(BaseException) as raised:
        files_api._delete_staged_upload(tmp_path / "staged.txt")
    assert raised.value is control_error


def test_delete_staged_upload_suppresses_operational_unlink_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unlink failure is logged without replacing the staging failure."""

    def failing_unlink(
        _path: Path,
        *,
        missing_ok: bool = False,
    ) -> None:
        del missing_ok
        raise OSError("unlink failed")

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    files_api._delete_staged_upload(tmp_path / "staged.txt")
    assert "Failed to clean up local upload" in caplog.text


def test_delete_staged_upload_does_not_hide_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only operational unlink failures are best-effort."""

    def invalid_unlink(
        _path: Path,
        *,
        missing_ok: bool = False,
    ) -> None:
        del missing_ok
        raise ValueError("invalid cleanup call")

    monkeypatch.setattr(Path, "unlink", invalid_unlink)
    with pytest.raises(ValueError, match="invalid cleanup call"):
        files_api._delete_staged_upload(tmp_path / "staged.txt")


def test_reserve_and_copy_uses_real_upload_path_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The worker remains compatible with the production path builder."""

    from xagent.web import config as web_config

    monkeypatch.setattr(web_config, "get_uploads_dir", lambda: tmp_path)
    target = files_api._reserve_and_copy_upload(
        UploadFile(filename="../real-path.txt", file=io.BytesIO(b"payload")),
        task_id="42",
        folder="documents",
        user_id=7,
    )

    assert target == tmp_path / "user_7" / "task_42" / "documents" / "real-path.txt"
    assert target.read_bytes() == b"payload"


@pytest.mark.asyncio
async def test_store_uploaded_files_preserves_source_value_error_and_cleans_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A source failure is not a client folder-validation error."""

    class SourceThatFailsWithValueError:
        def read(self, _size: int = -1) -> bytes:
            raise ValueError("source read failure")

    user_id = 7
    monkeypatch.setattr(files_api, "get_upload_path", _stage_path_in(tmp_path, user_id))

    with pytest.raises(ValueError, match="source read failure"):
        await files_api.store_uploaded_files(
            upload_items=[
                UploadFile(
                    filename="source-value-error.txt",
                    file=SourceThatFailsWithValueError(),
                )
            ],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        )

    assert not list(tmp_path.rglob("source-value-error*"))


@pytest.mark.asyncio
async def test_store_uploaded_files_preserves_write_value_error_and_cleans_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A destination failure is not a client folder-validation error."""

    user_id = 7
    monkeypatch.setattr(files_api, "get_upload_path", _stage_path_in(tmp_path, user_id))
    original_open = builtins.open

    class ValueErrorWriter:
        def __init__(self, buffer) -> None:  # type: ignore[no-untyped-def]
            self.buffer = buffer

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
            self.buffer.close()

        def write(self, _chunk: bytes) -> int:
            raise ValueError("destination write failure")

    def failing_open(path, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        buffer = original_open(path, mode, *args, **kwargs)
        return ValueErrorWriter(buffer) if "x" in mode else buffer

    monkeypatch.setattr(builtins, "open", failing_open)

    with pytest.raises(ValueError, match="destination write failure"):
        await files_api.store_uploaded_files(
            upload_items=[
                UploadFile(
                    filename="write-value-error.txt",
                    file=io.BytesIO(b"payload"),
                )
            ],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        )

    assert not list(tmp_path.rglob("write-value-error*"))


@pytest.mark.asyncio
async def test_store_uploaded_files_maps_path_validation_error_to_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only path construction turns a validation error into a client 422."""

    def invalid_upload_path(*_args, **_kwargs) -> Path:  # type: ignore[no-untyped-def]
        raise ValueError("invalid folder")

    monkeypatch.setattr(files_api, "get_upload_path", invalid_upload_path)

    with pytest.raises(HTTPException) as exc_info:
        await files_api.store_uploaded_files(
            upload_items=[
                UploadFile(
                    filename="invalid-folder.txt",
                    file=io.BytesIO(b"payload"),
                )
            ],
            task_type="general",
            task_id=None,
            folder="bad-folder",
            user_id=7,
            single_file_mode=True,
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Invalid folder name: invalid folder"


@pytest.mark.asyncio
async def test_cancelled_store_cleans_its_exact_late_result_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The request cleanup owner deletes a drained staging result exactly once."""

    user_id = 7
    monkeypatch.setattr(files_api, "get_upload_path", _stage_path_in(tmp_path, user_id))
    source = _BlockingSource(b"late worker result")
    target = tmp_path / f"user_{user_id}" / "cancelled.txt"
    unlink_calls: list[Path] = []
    original_unlink = Path.unlink

    def recording_unlink(path: Path, *, missing_ok: bool = False) -> None:
        unlink_calls.append(path)
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", recording_unlink)
    task = asyncio.create_task(
        files_api.store_uploaded_files(
            upload_items=[UploadFile(filename="cancelled.txt", file=source)],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        )
    )
    assert await asyncio.to_thread(source.started.wait, 2)
    task.cancel()
    source.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert unlink_calls.count(target) == 1
    assert not target.exists()


@pytest.mark.asyncio
async def test_failed_store_reuses_shared_staged_cleanup_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Request cleanup delegates operational error handling to one helper."""

    user_id = 7
    target = tmp_path / f"user_{user_id}" / "cleanup.txt"
    monkeypatch.setattr(files_api, "get_upload_path", _stage_path_in(tmp_path, user_id))
    cleanup_calls: list[Path] = []
    original_delete = files_api._delete_staged_upload

    def recording_delete(path: Path) -> None:
        cleanup_calls.append(path)
        original_delete(path)

    def fail_registration(_registrations) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError("registration failed")

    monkeypatch.setattr(files_api, "_delete_staged_upload", recording_delete)
    monkeypatch.setattr(files_api, "register_local_uploads_sync", fail_registration)

    with pytest.raises(RuntimeError, match="registration failed"):
        await files_api.store_uploaded_files(
            upload_items=[
                UploadFile(filename="cleanup.txt", file=io.BytesIO(b"payload"))
            ],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        )

    assert cleanup_calls == [target]
    assert not target.exists()


def test_jwt_upload_releases_auth_session_before_durable_io(
    monkeypatch: pytest.MonkeyPatch,
    isolated_upload_storage,
) -> None:
    headers = _admin_headers()
    engine = _install_one_slot_queue_pool(monkeypatch)
    checked_out = _record_durable_pool_ownership(monkeypatch, engine)
    try:
        response = client.post(
            "/api/files/upload",
            headers=headers,
            data={"task_type": "general"},
            files={"file": ("jwt.txt", b"payload", "text/plain")},
        )
        assert response.status_code == 200, response.text
        assert checked_out == [0]
        assert engine.pool.checkedout() == 0
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_widget_upload_releases_access_session_before_durable_io(
    monkeypatch: pytest.MonkeyPatch,
    isolated_upload_storage,
) -> None:
    _admin_headers()
    engine = _install_one_slot_queue_pool(monkeypatch)
    checked_out = _record_durable_pool_ownership(monkeypatch, engine)
    request_db = _direct_db_session()
    try:
        user = request_db.query(User).filter(User.username == "admin").one()
        context = PublicChatAccessContext(
            user=user,
            channel_id=None,
            guest_id="guest",
            widget_workforce_id=1,
        )
        response = await upload_public_chat_files(
            file=UploadFile(
                filename="widget.txt",
                file=io.BytesIO(b"payload"),
                headers={"content-type": "text/plain"},
            ),
            files=None,
            task_type="general",
            message="",
            task_id=None,
            folder=None,
            access_context=context,
            db=request_db,
        )
        assert response["file_id"]
        assert checked_out == [0]
        assert engine.pool.checkedout() == 0
    finally:
        request_db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_share_upload_releases_access_session_before_durable_io(
    monkeypatch: pytest.MonkeyPatch,
    isolated_upload_storage,
) -> None:
    _admin_headers()
    engine = _install_one_slot_queue_pool(monkeypatch)
    checked_out = _record_durable_pool_ownership(monkeypatch, engine)
    request_db = _direct_db_session()
    try:
        user = request_db.query(User).filter(User.username == "admin").one()
        context = ShareChatAccessContext(
            user=user,
            share_token="share",
            guest_id="guest-upload",
            workforce=object(),  # type: ignore[arg-type]
        )
        response = await upload_share_chat_files(
            file=UploadFile(
                filename="share.txt",
                file=io.BytesIO(b"payload"),
                headers={"content-type": "text/plain"},
            ),
            files=None,
            task_type="general",
            message="",
            task_id=None,
            folder=None,
            access_context=context,
            db=request_db,
        )
        assert response["file_id"]
        assert checked_out == [0]
        assert engine.pool.checkedout() == 0
    finally:
        request_db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_cancel_after_upload_registration_compensates_metadata_and_bytes(
    monkeypatch: pytest.MonkeyPatch,
    isolated_upload_storage,
) -> None:
    _upload_root, object_root = isolated_upload_storage
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = int(db.query(User.id).filter(User.username == "admin").scalar())
    finally:
        db.close()

    original_register = files_api.register_local_uploads_sync
    registration_committed = threading.Event()
    allow_worker_return = threading.Event()
    file_ids: list[str] = []
    local_paths = []

    def delayed_registered_return(registrations):  # type: ignore[no-untyped-def]
        local_paths.extend(item.local_path for item in registrations)
        file_ids.extend(item.file_id for item in registrations)
        snapshots = original_register(registrations)
        registration_committed.set()
        assert allow_worker_return.wait(timeout=2)
        return snapshots

    monkeypatch.setattr(
        files_api,
        "register_local_uploads_sync",
        delayed_registered_return,
    )
    upload_task = asyncio.create_task(
        files_api.store_uploaded_files(
            upload_items=[
                UploadFile(
                    filename="late-cancel.txt",
                    file=io.BytesIO(b"payload"),
                    headers={"content-type": "text/plain"},
                )
            ],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        )
    )

    registered = await asyncio.to_thread(registration_committed.wait, 10)
    assert registered, (
        repr(upload_task.exception())
        if upload_task.done() and not upload_task.cancelled()
        else "registration worker did not settle"
    )
    upload_task.cancel()
    allow_worker_return.set()
    with pytest.raises(asyncio.CancelledError):
        await upload_task

    assert file_ids
    check_db = _direct_db_session()
    try:
        assert (
            check_db.query(UploadedFile)
            .filter(UploadedFile.file_id.in_(file_ids))
            .count()
            == 0
        )
    finally:
        check_db.close()
    assert all(not path.exists() for path in local_paths)
    assert not any(path.is_file() for path in object_root.rglob("*"))


@pytest.mark.asyncio
async def test_failed_compensation_does_not_skip_request_local_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    isolated_upload_storage,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _upload_root, _object_root = isolated_upload_storage
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = int(db.query(User.id).filter(User.username == "admin").scalar())
    finally:
        db.close()

    local_paths: list[Path] = []

    def fail_registration(registrations):  # type: ignore[no-untyped-def]
        local_paths.extend(item.local_path for item in registrations)
        raise RuntimeError("registration failed")

    def fail_compensation(_claims) -> None:  # type: ignore[no-untyped-def]
        raise DurableStorageOperationError(
            "storage cleanup unavailable", storage_key=None
        )

    monkeypatch.setattr(
        files_api,
        "register_local_uploads_sync",
        fail_registration,
    )
    monkeypatch.setattr(
        files_api,
        "compensate_registered_uploads_sync",
        fail_compensation,
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        await files_api.store_uploaded_files(
            upload_items=[
                UploadFile(
                    filename="cleanup-after-compensation-error.txt",
                    file=io.BytesIO(b"payload"),
                    headers={"content-type": "text/plain"},
                )
            ],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=user_id,
            single_file_mode=True,
        )

    assert local_paths
    assert all(not path.exists() for path in local_paths)

    # The double fault -- the upload failed against durable storage and
    # compensating against it failed too -- is the incident this reporting
    # exists for, so it must not be recorded as an unlabelled traceback. The
    # AST sweep pins that the site declares these fields; this pins that the
    # record actually carries them, with the values from this request.
    fault_lines = [
        record.getMessage()
        for record in caplog.records
        if record.name == "xagent.web.api.files"
        and DURABLE_FAULT_LOG_PREFIX in record.getMessage()
    ]
    assert len(fault_lines) == 1, caplog.records
    assert "upload compensation" in fault_lines[0]
    assert f"user_id={user_id}" in fault_lines[0]


@pytest.mark.asyncio
async def test_store_uploaded_files_fails_closed_on_scope_authority_mismatch(
    isolated_upload_storage,
) -> None:
    """``store_uploaded_files`` selects the write namespace (workspace
    segments / storage key) for the bytes it writes, unlike the off-turn
    *read* paths (``resolve_execution_scope_off_turn``) that downgrade an
    authority mismatch to a warning and keep going. A registered
    resolver/persisted-snapshot disagreement here must propagate and fail
    the request instead of silently choosing a namespace."""
    from xagent.core.execution_scope import (
        ExecutionScope,
        ExecutionScopeAuthorityError,
        set_execution_scope_snapshot_loader,
    )
    from xagent.web.models.task import Task, TaskStatus
    from xagent.web.services.execution_scope_snapshot import (
        load_task_execution_scope_snapshot,
    )

    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = int(db.query(User.id).filter(User.username == "admin").scalar())
        task = Task(
            user_id=user_id,
            title="scope mismatch",
            description="scope mismatch",
            status=TaskStatus.COMPLETED,
            source="sdk",
            agent_config={
                "execution_scope": ExecutionScope(
                    workspace_segments=("snapshot-tenant",),
                ).to_dict()
            },
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    set_execution_scope_snapshot_loader(load_task_execution_scope_snapshot)
    register_scope_resolver(
        lambda _task_id: ExecutionScope(workspace_segments=("resolver-tenant",)),
    )
    try:
        with pytest.raises(ExecutionScopeAuthorityError):
            await files_api.store_uploaded_files(
                upload_items=[
                    UploadFile(
                        filename="mismatch.txt",
                        file=io.BytesIO(b"payload"),
                        headers={"content-type": "text/plain"},
                    )
                ],
                task_type="general",
                task_id=str(task_id),
                folder=None,
                user_id=user_id,
                single_file_mode=True,
            )
    finally:
        register_scope_resolver(None)
        set_execution_scope_snapshot_loader(None)


def test_http_durable_upload_is_bound_to_agent_workspace_without_second_put(
    monkeypatch: pytest.MonkeyPatch,
    isolated_upload_storage,
) -> None:
    upload_root, _object_root = isolated_upload_storage
    headers = _admin_headers()
    put_calls = 0
    original_put_file = FsspecFileStorage.put_file

    def counting_put_file(
        self: FsspecFileStorage,
        source,
        key: str,
        content_type: str | None = None,
    ):
        nonlocal put_calls
        put_calls += 1
        return original_put_file(self, source, key, content_type)

    monkeypatch.setattr(FsspecFileStorage, "put_file", counting_put_file)
    response = client.post(
        "/api/files/upload",
        headers=headers,
        data={"task_type": "general"},
        files={"file": ("already-durable.txt", b"payload", "text/plain")},
    )
    assert response.status_code == 200, response.text
    file_id = str(response.json()["file_id"])

    db = _direct_db_session()
    try:
        record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
        storage_path = str(record.storage_path)
        file_info = {
            "file_id": file_id,
            "name": str(record.filename),
            "path": storage_path,
            "workspace_path": None,
        }
    finally:
        db.close()

    workspace = TaskWorkspace(
        id="web_task_999",
        base_dir=str(tmp_path := upload_root / "agent-workspaces"),
        allowed_external_dirs=[str(Path(storage_path).parent)],
    )
    task_execution_service._register_uploaded_files_for_agent(
        type("AgentService", (), {"workspace": workspace})(),
        [file_info],
    )

    assert put_calls == 1
    assert workspace._recently_registered_files[str(Path(storage_path).resolve())] == (
        file_id
    )
    assert Path(str(file_info["workspace_path"])).exists()
    assert tmp_path.exists()
