"""The startup file-storage sync gate must never hold a host's control paths.

Covers the 2026-11 incident: a prewarmed standby's activation started the
sync, the gate held the host's lifecycle probe behind it, and a sync that stood
still left the host blind. Also covers the read-only standby preflight that
keeps a fully-synced standby from gating at all, and the sync's own bounds.
"""

import asyncio
import threading
import time
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.file_storage.types import StoredObject
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import startup_file_storage_sync as sync_module
from xagent.web.services.startup_file_storage_sync import (
    StartupFileStoragePreflight,
    StartupSyncCancelled,
    StartupSyncControl,
    preflight_registered_files,
)

app_module = import_module("xagent.web.app")


@pytest.fixture(autouse=True)
def _isolated_prefixes(monkeypatch):
    monkeypatch.setattr(app_module, "_file_storage_startup_sync_exempt_prefixes", set())
    monkeypatch.setattr(
        app_module, "get_file_storage_startup_sync_enabled", lambda: True
    )


def _app_with_stuck_sync(*, finish_after: float | None = None) -> FastAPI:
    test_app = FastAPI()
    test_app.add_middleware(app_module.FileStorageStartupSyncGateMiddleware)

    @test_app.on_event("startup")
    async def _startup() -> None:
        async def _sync() -> None:
            await asyncio.sleep(finish_after if finish_after is not None else 3600)

        test_app.state.file_storage_startup_sync_task = asyncio.create_task(_sync())

    @test_app.get("/internal/thin-gate/lifecycle")
    async def _lifecycle() -> dict[str, str]:
        return {"status": "ok"}

    @test_app.get("/api/client")
    async def _client() -> dict[str, str]:
        return {"status": "served"}

    return test_app


# --- M1: host-registered exemptions ------------------------------------------


def test_registered_prefix_answers_while_sync_is_stuck():
    app_module.register_file_storage_startup_sync_exempt_prefix("/internal/thin-gate/")
    with TestClient(_app_with_stuck_sync()) as client:
        started = time.monotonic()
        response = client.get("/internal/thin-gate/lifecycle")
    assert response.status_code == 200
    assert time.monotonic() - started < 1.0


def test_registered_prefix_answers_after_sync_failed():
    app_module.register_file_storage_startup_sync_exempt_prefix("/internal/thin-gate/")
    test_app = _app_with_stuck_sync()
    with TestClient(test_app) as client:
        test_app.state.file_storage_startup_sync_error = RuntimeError("s3 down")
        assert client.get("/internal/thin-gate/lifecycle").status_code == 200
        assert client.get("/api/client").status_code == 503


def test_unregistered_path_is_still_held_until_sync_completes():
    app_module.register_file_storage_startup_sync_exempt_prefix("/internal/thin-gate/")
    with TestClient(_app_with_stuck_sync(finish_after=0.3)) as client:
        started = time.monotonic()
        response = client.get("/api/client")
    assert response.status_code == 200
    assert time.monotonic() - started >= 0.25


def test_without_registration_lifecycle_is_held():
    """Pins the incident: the default gate holds every non-exempt path."""
    with TestClient(_app_with_stuck_sync(finish_after=0.3)) as client:
        started = time.monotonic()
        assert client.get("/internal/thin-gate/lifecycle").status_code == 200
        assert time.monotonic() - started >= 0.25


def test_prefix_must_be_absolute():
    with pytest.raises(ValueError):
        app_module.register_file_storage_startup_sync_exempt_prefix("internal/")


def test_prefix_match_is_a_prefix_not_a_substring():
    app_module.register_file_storage_startup_sync_exempt_prefix("/internal/thin-gate/")
    assert app_module._is_file_storage_startup_sync_exempt("/internal/thin-gate/x")
    assert not app_module._is_file_storage_startup_sync_exempt(
        "/api/internal/thin-gate/x"
    )
    assert not app_module._is_file_storage_startup_sync_exempt("/internal/thin-gates")


# --- status accessor ----------------------------------------------------------


@pytest.mark.asyncio
async def test_status_reports_running_failed_done_with_elapsed(monkeypatch):
    test_app = FastAPI()
    assert (
        app_module.get_file_storage_startup_sync_status(test_app)["state"] == "pending"
    )

    release = asyncio.Event()

    async def _attempt(*_args) -> None:
        await release.wait()

    monkeypatch.setattr(app_module, "_run_bounded_file_storage_sync_attempt", _attempt)
    task = app_module.start_file_storage_startup_sync_task(test_app)
    await asyncio.sleep(0.05)
    running = app_module.get_file_storage_startup_sync_status(test_app)
    assert running["state"] == "running"
    assert running["gating"] is True
    assert running["elapsed_seconds"] >= 0.04

    test_app.state.file_storage_startup_sync_error = RuntimeError("boom")
    assert (
        app_module.get_file_storage_startup_sync_status(test_app)["state"] == "failed"
    )
    test_app.state.file_storage_startup_sync_error = None

    release.set()
    await task
    done = app_module.get_file_storage_startup_sync_status(test_app)
    assert done["state"] == "done"
    assert done["elapsed_seconds"] is not None


def test_status_disabled(monkeypatch):
    monkeypatch.setattr(
        app_module, "get_file_storage_startup_sync_enabled", lambda: False
    )
    assert (
        app_module.get_file_storage_startup_sync_status(FastAPI())["state"]
        == "disabled"
    )


# --- S2: preflight and non-gating background sync ------------------------------


class _Storage:
    backend = "s3"

    def __init__(self, keys=()):
        self.keys = set(keys)
        self.put_calls = []

    def list(self, prefix):
        return [
            StoredObject(backend="s3", key=k, uri=f"s3://b/{k}", size=1)
            for k in sorted(self.keys)
            if k.startswith(prefix.rstrip("/") + "/")
        ]

    def put_file(self, *a, **k):  # pragma: no cover - must never run
        self.put_calls.append(a)
        raise AssertionError("preflight must not write storage")


def _db(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        user = User(username="u", password_hash="h", is_admin=False)
        db.add(user)
        db.commit()
        return factory, int(user.id)


def _row(factory, user_id, path: Path, file_id, **kw):
    with factory() as db:
        db.add(
            UploadedFile(
                file_id=file_id,
                user_id=user_id,
                filename=path.name,
                storage_path=str(path),
                mime_type="text/plain",
                file_size=0,
                storage_status=kw.pop("storage_status", "legacy"),
                **kw,
            )
        )
        db.commit()


def _durable_row(factory, user_id, path, file_id):
    key = f"users/{user_id}/uploads/{file_id}/{path.name}"
    _row(
        factory,
        user_id,
        path,
        file_id,
        storage_backend="s3",
        storage_key=key,
        storage_status="available",
        checksum="c",
    )
    return key


def test_preflight_clean_when_every_file_is_already_durable(tmp_path):
    factory, uid = _db(tmp_path)
    local = tmp_path / "a.txt"
    local.write_text("x")
    key = _durable_row(factory, uid, local, "f1")
    gone = tmp_path / "gone.txt"  # missing locally and remotely: nothing to write
    _row(factory, uid, gone, "f2")
    storage = _Storage({key})

    result = preflight_registered_files(storage=storage, session_factory=factory)

    assert result.clean
    assert result.scanned == 2
    assert result.skipped_missing_local == 1
    assert storage.put_calls == []


def test_preflight_reports_file_missing_remote_but_present_locally(tmp_path):
    factory, uid = _db(tmp_path)
    local = tmp_path / "a.txt"
    local.write_text("x")
    _row(factory, uid, local, "f1")
    result = preflight_registered_files(storage=_Storage(), session_factory=factory)
    assert not result.clean and result.needs_write == 1


def test_preflight_reports_remote_present_but_row_incomplete(tmp_path):
    factory, uid = _db(tmp_path)
    local = tmp_path / "a.txt"
    local.write_text("x")
    _row(factory, uid, local, "f1")
    key = f"users/{uid}/uploads/f1/a.txt"
    result = preflight_registered_files(
        storage=_Storage({key}), session_factory=factory
    )
    assert result.needs_write == 1


def test_preflight_writes_nothing_to_the_database(tmp_path):
    factory, uid = _db(tmp_path)
    local = tmp_path / "a.txt"
    local.write_text("x")
    _row(factory, uid, local, "f1")
    key = f"users/{uid}/uploads/f1/a.txt"
    preflight_registered_files(storage=_Storage({key}), session_factory=factory)
    with factory() as db:
        row = db.query(UploadedFile).one()
        assert row.storage_status == "legacy" and row.storage_key is None


def test_preflight_honours_cancel(tmp_path):
    factory, uid = _db(tmp_path)
    local = tmp_path / "a.txt"
    local.write_text("x")
    _row(factory, uid, local, "f1")
    control = StartupSyncControl()
    control.cancel.set()
    with pytest.raises(StartupSyncCancelled):
        preflight_registered_files(
            storage=_Storage(), session_factory=factory, control=control
        )


@pytest.mark.parametrize(
    ("task", "result", "expected_gate"),
    [
        (None, None, True),  # normal start: never ran a preflight
        (object(), None, True),  # preflight unfinished or failed
        (object(), StartupFileStoragePreflight(scanned=3, needs_write=1), True),
        (object(), StartupFileStoragePreflight(scanned=3, needs_write=0), False),
    ],
)
def test_should_gate_only_when_preflight_proved_nothing_to_write(
    task, result, expected_gate
):
    test_app = FastAPI()
    test_app.state.file_storage_preflight_task = task
    test_app.state.file_storage_preflight_result = result
    assert app_module._file_storage_sync_should_gate(test_app) is expected_gate


@pytest.mark.asyncio
async def test_clean_standby_activation_never_gates_requests(monkeypatch):
    """Activation with a clean preflight: sync runs in background, gate open."""
    test_app = FastAPI()
    test_app.state.file_storage_preflight_task = object()
    test_app.state.file_storage_preflight_result = StartupFileStoragePreflight(
        scanned=9
    )
    release = asyncio.Event()

    async def _attempt(*_args) -> None:
        await release.wait()

    monkeypatch.setattr(app_module, "_run_bounded_file_storage_sync_attempt", _attempt)
    monkeypatch.setattr(app_module, "get_shared_task_execution_enabled", lambda: True)
    for name in (
        "start_uploaded_file_recovery_task",
        "start_orphan_upload_gc_task",
        "start_retention_purge_task",
        "start_task_cleanup_retry_task",
    ):
        monkeypatch.setattr(app_module, name, Mock())

    await app_module._start_local_runtime_loops(test_app)
    try:
        assert test_app.state.file_storage_startup_sync_gating is False
        # Sync still running, yet the wait returns at once and /ready is not 503.
        await asyncio.wait_for(
            app_module.wait_for_file_storage_startup_sync(test_app), timeout=0.5
        )
        assert not test_app.state.file_storage_startup_sync_task.done()
    finally:
        release.set()
        await test_app.state.file_storage_startup_sync_task


@pytest.mark.asyncio
async def test_standby_with_files_to_write_still_gates(monkeypatch):
    test_app = FastAPI()
    test_app.state.file_storage_preflight_task = object()
    test_app.state.file_storage_preflight_result = StartupFileStoragePreflight(
        scanned=9, needs_write=2
    )
    release = asyncio.Event()

    async def _attempt(*_args) -> None:
        await release.wait()

    monkeypatch.setattr(app_module, "_run_bounded_file_storage_sync_attempt", _attempt)
    monkeypatch.setattr(app_module, "get_shared_task_execution_enabled", lambda: True)
    for name in (
        "start_uploaded_file_recovery_task",
        "start_orphan_upload_gc_task",
        "start_retention_purge_task",
        "start_task_cleanup_retry_task",
    ):
        monkeypatch.setattr(app_module, name, Mock())

    await app_module._start_local_runtime_loops(test_app)
    try:
        assert test_app.state.file_storage_startup_sync_gating is True
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                app_module.wait_for_file_storage_startup_sync(test_app), timeout=0.4
            )
    finally:
        release.set()
        await test_app.state.file_storage_startup_sync_task


def test_non_gating_sync_does_not_fail_requests_when_it_errors():
    test_app = FastAPI()
    test_app.state.file_storage_startup_sync_gating = False
    test_app.state.file_storage_startup_sync_error = RuntimeError("s3 down")
    asyncio.run(app_module.wait_for_file_storage_startup_sync(test_app))


# --- S1: bounded sync -----------------------------------------------------------


@pytest.mark.asyncio
async def test_attempt_logs_progress_dumps_stacks_and_cancels_when_stalled(monkeypatch):
    monkeypatch.setattr(
        app_module, "FILE_STORAGE_STARTUP_SYNC_PROGRESS_INTERVAL_SECONDS", 0.05
    )
    monkeypatch.setattr(app_module, "FILE_STORAGE_STARTUP_SYNC_DUMP_AFTER_SECONDS", 0.1)
    monkeypatch.setattr(app_module, "FILE_STORAGE_STARTUP_SYNC_STALL_SECONDS", 0.3)
    monkeypatch.setattr(
        app_module, "FILE_STORAGE_STARTUP_SYNC_CANCEL_GRACE_SECONDS", 2.0
    )
    dump = Mock()
    monkeypatch.setattr(app_module, "_dump_all_thread_stacks", dump)
    saw_cancel = threading.Event()

    def _stuck_sync() -> None:
        control = app_module._file_storage_sync_control.get()
        while True:
            try:
                control.checkpoint("stuck", progress=False)
            except StartupSyncCancelled:
                saw_cancel.set()
                raise
            time.sleep(0.01)

    monkeypatch.setattr(app_module, "run_startup_file_storage_sync", _stuck_sync)

    with pytest.raises(TimeoutError, match="no progress"):
        await app_module._run_bounded_file_storage_sync_attempt()

    dump.assert_called_once()
    assert saw_cancel.wait(1.0)


@pytest.mark.asyncio
async def test_attempt_runs_on_dedicated_executor_not_the_default_one(monkeypatch):
    names: list[str] = []
    monkeypatch.setattr(
        app_module,
        "run_startup_file_storage_sync",
        lambda: names.append(threading.current_thread().name),
    )
    await app_module._run_bounded_file_storage_sync_attempt()
    assert names and names[0].startswith("startup-file-sync")


@pytest.mark.asyncio
async def test_saturated_default_executor_does_not_stall_the_sync(monkeypatch):
    loop = asyncio.get_running_loop()
    gate = threading.Event()
    from concurrent.futures import ThreadPoolExecutor

    small = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(small)
    blocker = loop.run_in_executor(None, gate.wait)  # default pool now full
    ran = threading.Event()
    monkeypatch.setattr(app_module, "run_startup_file_storage_sync", ran.set)
    try:
        await asyncio.wait_for(app_module._run_bounded_file_storage_sync_attempt(), 2.0)
        assert ran.is_set()
    finally:
        gate.set()
        await blocker
        small.shutdown(wait=True)


def test_locked_result_from_an_earlier_attempt_is_not_success(monkeypatch):
    monkeypatch.setattr(
        sync_module,
        "sync_registered_files_to_durable_storage",
        Mock(return_value=sync_module.StartupFileStorageSyncResult(locked=True)),
    )
    token = app_module._file_storage_sync_control.set(StartupSyncControl())
    try:
        with pytest.raises(RuntimeError, match="previous attempt"):
            app_module.run_startup_file_storage_sync()
    finally:
        app_module._file_storage_sync_control.reset(token)


def test_stuck_sync_does_not_stop_lifecycle_from_answering(monkeypatch):
    """End to end through the real middleware with the real retry loop."""
    app_module.register_file_storage_startup_sync_exempt_prefix("/internal/thin-gate/")
    test_app = FastAPI()
    test_app.add_middleware(app_module.FileStorageStartupSyncGateMiddleware)
    release = threading.Event()
    monkeypatch.setattr(
        app_module, "run_startup_file_storage_sync", lambda: release.wait(30)
    )

    @test_app.on_event("startup")
    async def _startup() -> None:
        app_module.start_file_storage_startup_sync_task(test_app)

    @test_app.get("/internal/thin-gate/lifecycle")
    async def _lifecycle() -> dict:
        return app_module.get_file_storage_startup_sync_status(test_app)

    @test_app.get("/api/client")
    async def _client() -> dict:
        return {}

    try:
        with TestClient(test_app) as client:
            started = time.monotonic()
            body = client.get("/internal/thin-gate/lifecycle").json()
            assert time.monotonic() - started < 1.0
            assert body["state"] == "running"
            release.set()
            assert client.get("/api/client").status_code == 200
    finally:
        release.set()


@pytest.mark.asyncio
async def test_ready_is_not_held_by_a_non_gating_sync(monkeypatch):
    import xagent.web.models.database as database_module
    import xagent.web.services.interaction_rollout as ir

    monkeypatch.setattr(
        ir,
        "_policy",
        ir.InteractionRolloutPolicy(
            mode="legacy",
            native_sources=frozenset(),
            native_protocol_version=1,
            shared_task_execution=True,
        ),
    )
    monkeypatch.setattr(database_module, "_SessionLocal", None)
    state = app_module.app.state
    task = asyncio.create_task(asyncio.sleep(30))
    monkeypatch.setattr(state, "file_storage_startup_sync_task", task, raising=False)
    monkeypatch.setattr(state, "file_storage_startup_sync_error", None, raising=False)
    try:
        monkeypatch.setattr(
            state, "file_storage_startup_sync_gating", True, raising=False
        )
        assert (await app_module.readiness_check()).status_code == 503
        monkeypatch.setattr(
            state, "file_storage_startup_sync_gating", False, raising=False
        )
        assert (await app_module.readiness_check()).status_code == 200
    finally:
        task.cancel()


# --- review: stall (not total) timeout, resume, cancel before storage I/O --------


class _SlowListStorage(_Storage):
    """Every per-user listing succeeds but takes a while (listings cache off)."""

    def __init__(self, keys=(), delay=0.1, on_list=None):
        super().__init__(keys)
        self.delay = delay
        self.list_prefixes: list[str] = []
        self.on_list = on_list

    def list(self, prefix):
        self.list_prefixes.append(prefix)
        if self.on_list is not None:
            self.on_list(len(self.list_prefixes))
        time.sleep(self.delay)
        return super().list(prefix)


def _many_user_db(tmp_path, users=8):
    # A file database: the sync runs on executor threads.
    engine = create_engine(f"sqlite:///{tmp_path}/sync.sqlite")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    keys = set()
    ids = []
    with factory() as db:
        for n in range(users):
            user = User(username=f"u{n}", password_hash="h", is_admin=False)
            db.add(user)
            db.commit()
            ids.append(int(user.id))
    for uid in ids:
        path = tmp_path / f"f{uid}.txt"
        path.write_text("x")
        keys.add(_durable_row(factory, uid, path, f"file{uid}"))
    return factory, keys


@pytest.mark.asyncio
async def test_steadily_progressing_scan_longer_than_the_stall_window_completes(
    monkeypatch, tmp_path
):
    """Regression: a total deadline restarted a healthy long scan forever."""
    factory, keys = _many_user_db(tmp_path, users=8)
    storage = _SlowListStorage(keys, delay=0.1)  # ~0.8 s in total
    monkeypatch.setattr(
        app_module, "FILE_STORAGE_STARTUP_SYNC_PROGRESS_INTERVAL_SECONDS", 0.05
    )
    monkeypatch.setattr(app_module, "FILE_STORAGE_STARTUP_SYNC_STALL_SECONDS", 0.3)
    monkeypatch.setattr(app_module, "FILE_STORAGE_STARTUP_SYNC_DUMP_AFTER_SECONDS", 60)
    monkeypatch.setenv(
        "XAGENT_FILE_STORAGE_STARTUP_SYNC_LOCK_FILE", str(tmp_path / "l.lock")
    )
    real = sync_module.sync_registered_files_to_durable_storage
    monkeypatch.setattr(
        sync_module,
        "sync_registered_files_to_durable_storage",
        lambda **kw: real(storage=storage, session_factory=factory, **kw),
    )
    test_app = FastAPI()
    task = app_module.start_file_storage_startup_sync_task(
        test_app, retry_interval_seconds=0
    )
    await asyncio.wait_for(task, 10)

    assert test_app.state.file_storage_startup_sync_completed is True
    assert test_app.state.file_storage_startup_sync_error is None
    # one pass: no restarts, each user listed exactly once
    assert len(storage.list_prefixes) == 8
    await asyncio.wait_for(app_module.wait_for_file_storage_startup_sync(test_app), 1)


@pytest.mark.asyncio
async def test_retry_after_a_stall_resumes_instead_of_restarting(monkeypatch, tmp_path):
    factory, keys = _many_user_db(tmp_path, users=6)
    stall_once = {"done": False}

    def on_list(call_number):
        if call_number == 4 and not stall_once["done"]:
            stall_once["done"] = True
            time.sleep(0.6)  # a real stall inside the 4th listing

    storage = _SlowListStorage(keys, delay=0.01, on_list=on_list)
    monkeypatch.setattr(
        app_module, "FILE_STORAGE_STARTUP_SYNC_PROGRESS_INTERVAL_SECONDS", 0.05
    )
    monkeypatch.setattr(app_module, "FILE_STORAGE_STARTUP_SYNC_STALL_SECONDS", 0.3)
    monkeypatch.setattr(
        app_module, "FILE_STORAGE_STARTUP_SYNC_CANCEL_GRACE_SECONDS", 0.05
    )
    monkeypatch.setattr(app_module, "_dump_all_thread_stacks", Mock())
    monkeypatch.setenv(
        "XAGENT_FILE_STORAGE_STARTUP_SYNC_LOCK_FILE", str(tmp_path / "l.lock")
    )
    real = sync_module.sync_registered_files_to_durable_storage
    monkeypatch.setattr(
        sync_module,
        "sync_registered_files_to_durable_storage",
        lambda **kw: real(storage=storage, session_factory=factory, **kw),
    )
    test_app = FastAPI()
    task = app_module.start_file_storage_startup_sync_task(
        test_app, retry_interval_seconds=0.7
    )
    await asyncio.wait_for(task, 15)

    assert test_app.state.file_storage_startup_sync_completed is True
    # The stalled attempt finished user 4 before it saw the cancel, and the
    # retry continued after it: every user is listed exactly once (a restart
    # from the beginning would list users 1-3 again).
    assert sorted(storage.list_prefixes) == sorted(set(storage.list_prefixes))
    assert len(storage.list_prefixes) == 6


@pytest.mark.asyncio
async def test_a_real_stall_still_trips_the_timeout_and_dumps_stacks(monkeypatch):
    monkeypatch.setattr(
        app_module, "FILE_STORAGE_STARTUP_SYNC_PROGRESS_INTERVAL_SECONDS", 0.05
    )
    monkeypatch.setattr(app_module, "FILE_STORAGE_STARTUP_SYNC_DUMP_AFTER_SECONDS", 0.1)
    monkeypatch.setattr(app_module, "FILE_STORAGE_STARTUP_SYNC_STALL_SECONDS", 0.3)
    monkeypatch.setattr(
        app_module, "FILE_STORAGE_STARTUP_SYNC_CANCEL_GRACE_SECONDS", 0.05
    )
    dump = Mock()
    monkeypatch.setattr(app_module, "_dump_all_thread_stacks", dump)
    release = threading.Event()
    monkeypatch.setattr(
        app_module, "run_startup_file_storage_sync", lambda: release.wait(10)
    )
    try:
        with pytest.raises(TimeoutError, match="no progress"):
            await app_module._run_bounded_file_storage_sync_attempt()
        dump.assert_called_once()
    finally:
        release.set()


def test_cancel_during_the_first_listing_stops_before_the_next_users_listing(tmp_path):
    factory, keys = _many_user_db(tmp_path, users=3)
    control = StartupSyncControl()
    storage = _SlowListStorage(keys, delay=0, on_list=lambda n: control.cancel.set())
    with pytest.raises(StartupSyncCancelled):
        preflight_registered_files(
            storage=storage, session_factory=factory, control=control
        )
    assert len(storage.list_prefixes) == 1


def test_cancel_during_the_first_listing_stops_the_sync_too(tmp_path, monkeypatch):
    factory, keys = _many_user_db(tmp_path, users=3)
    control = StartupSyncControl()
    storage = _SlowListStorage(keys, delay=0, on_list=lambda n: control.cancel.set())
    monkeypatch.setenv(
        "XAGENT_FILE_STORAGE_STARTUP_SYNC_LOCK_FILE", str(tmp_path / "l.lock")
    )
    with pytest.raises(StartupSyncCancelled):
        sync_module.sync_registered_files_to_durable_storage(
            storage=storage, session_factory=factory, control=control
        )
    assert len(storage.list_prefixes) == 1


def test_resume_cursor_never_moves_past_a_failed_file(tmp_path, monkeypatch):
    factory, keys = _many_user_db(tmp_path, users=3)
    with factory() as db:  # user 1's row is not durable and its upload fails
        row = db.query(UploadedFile).order_by(UploadedFile.id).first()
        row.storage_status = "legacy"
        row.storage_key = None
        row.checksum = None
        row.storage_backend = None
        db.commit()
        # Read the identity while the session is open: the row detaches on exit.
        first_user_key = (
            f"users/{row.user_id}/uploads/{row.file_id}/{Path(row.storage_path).name}"
        )
    storage = _SlowListStorage(keys - {first_user_key}, delay=0)

    def failing_put(*args, **kwargs):
        raise RuntimeError("s3 down")

    storage.put_file = failing_put
    monkeypatch.setenv(
        "XAGENT_FILE_STORAGE_STARTUP_SYNC_LOCK_FILE", str(tmp_path / "l.lock")
    )
    control = StartupSyncControl()

    result = sync_module.sync_registered_files_to_durable_storage(
        storage=storage, session_factory=factory, control=control
    )

    assert result.failed == 1
    assert control.resume.cursor is None  # a retry starts at the failed file
