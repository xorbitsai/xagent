from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.storage import FsspecFileStorage
from xagent.core.workspace import TaskWorkspace
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import channel_runtime
from xagent.web.services.channel_runtime import (
    DownloadedChannelFile,
    load_active_channel_configs,
    prepare_channel_task,
    register_channel_uploaded_files,
)
from xagent.web.services.task_lease_service import TaskLease
from xagent.web.services.uploaded_file_store import UploadedFileStore


def _create_channel_session_local(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'channel-claim.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.2,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        bind=engine,
    )
    with SessionLocal() as db:
        user = User(username="channel-claim-user", password_hash="hash")
        db.add(user)
        db.flush()
        channel = UserChannel(
            user_id=int(user.id),
            channel_type="telegram",
            channel_name="Telegram claim",
            config={"allowed_users": ["telegram-user"]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        return engine, SessionLocal, int(user.id), int(channel.id)


@pytest.mark.asyncio
async def test_prepare_channel_task_commits_creation_and_exact_claim_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
    )

    assert prepared is not None
    assert prepared.user_id == user_id
    assert prepared.is_new_task is True
    assert prepared.managed_lease.lease.task_id == prepared.task_id
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == prepared.task_id).one()
        assert task.status == TaskStatus.RUNNING
        assert task.run_id == prepared.managed_lease.lease.run_id
        assert task.runner_id == prepared.managed_lease.lease.runner_id
        assert task.lease_expires_at is not None

    assert await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


def test_prepare_channel_task_rolls_back_new_task_when_claim_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, _user_id, channel_id = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.acquire_task_lease_no_commit",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    prepared = channel_runtime._prepare_channel_task_sync(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
        expected_owner_user_id=None,
    )

    assert prepared is None
    with SessionLocal() as db:
        assert db.query(Task).count() == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_prepare_channel_task_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()

    claim = channel_runtime._ChannelTaskClaimSnapshot(
        user_id=7,
        task_id=11,
        is_new_task=False,
        lease=TaskLease(task_id=11, runner_id="runner-a", run_id="run-a"),
    )
    managed = object()

    def blocking_prepare(**_kwargs):  # type: ignore[no-untyped-def]
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return claim

    monkeypatch.setattr(
        "xagent.web.services.channel_runtime._prepare_channel_task_sync",
        blocking_prepare,
    )
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.start_managed_task_lease",
        lambda _lease: managed,
    )

    preparation = asyncio.create_task(
        prepare_channel_task(
            channel_id=3,
            external_user_id="telegram-user",
            active_task_id=11,
            text="hello",
            channel_name="Telegram",
        )
    )
    await asyncio.to_thread(worker_started.wait, 2)
    await asyncio.sleep(0)

    assert preparation.done() is False
    allow_worker.set()
    prepared = await preparation
    assert prepared is not None
    assert prepared.user_id == 7
    assert prepared.task_id == 11
    assert prepared.is_new_task is False
    assert prepared.managed_lease is managed


@pytest.mark.asyncio
async def test_register_channel_uploaded_files_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()
    downloaded = DownloadedChannelFile(
        name="report.txt",
        path=tmp_path / "report.txt",
        mime_type="text/plain",
        size=6,
        source_id="remote-file",
    )

    def blocking_register(**_kwargs) -> tuple:  # type: ignore[no-untyped-def]
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return ()

    monkeypatch.setattr(
        "xagent.web.services.channel_runtime._register_channel_uploaded_files_sync",
        blocking_register,
    )

    registration = asyncio.create_task(
        register_channel_uploaded_files(
            workspace=object(),
            task_id=11,
            user_id=7,
            files=(downloaded,),
        )
    )
    await asyncio.to_thread(worker_started.wait, 2)
    await asyncio.sleep(0)

    assert registration.done() is False
    allow_worker.set()
    assert await registration == ()


@pytest.mark.asyncio
async def test_channel_durable_upload_does_not_hold_pool_connection_or_upload_twice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine = create_engine(
        f"sqlite:///{tmp_path / 'channel.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.2,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        bind=engine,
    )
    with SessionLocal() as db:
        user = User(username="channel-upload-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=71, user_id=int(user.id), title="Channel upload")
        db.add(task)
        db.commit()
        user_id = int(user.id)

    workspace = TaskWorkspace(
        id="web_task_71",
        base_dir=str(tmp_path / "workspaces"),
        db_task_id=71,
    )
    source = workspace.input_dir / "report.txt"
    source.write_text("report", encoding="utf-8")
    downloaded = DownloadedChannelFile(
        name=source.name,
        path=source,
        mime_type="text/plain",
        size=source.stat().st_size,
        source_id="remote-file",
    )

    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )

    put_started = threading.Event()
    allow_put = threading.Event()
    put_calls = 0
    original_put_file = FsspecFileStorage.put_file

    def blocking_put_file(
        self: FsspecFileStorage,
        source_path: Path,
        key: str,
        content_type: str | None = None,
    ):
        nonlocal put_calls
        put_calls += 1
        put_started.set()
        assert allow_put.wait(timeout=3)
        return original_put_file(self, source_path, key, content_type)

    monkeypatch.setattr(FsspecFileStorage, "put_file", blocking_put_file)

    registration = asyncio.create_task(
        register_channel_uploaded_files(
            workspace=workspace,
            task_id=71,
            user_id=user_id,
            files=(downloaded,),
        )
    )
    assert await asyncio.to_thread(put_started.wait, 2)

    def probe_pool() -> int:
        with SessionLocal() as db:
            return int(db.execute(text("SELECT 1")).scalar_one())

    try:
        assert await asyncio.wait_for(asyncio.to_thread(probe_pool), timeout=1) == 1
    finally:
        allow_put.set()

    registered = await registration
    assert len(registered) == 1
    assert put_calls == 1
    assert workspace.resolve_file_id(registered[0].file_id) == source
    with SessionLocal() as db:
        record = (
            db.query(UploadedFile)
            .filter(UploadedFile.file_id == registered[0].file_id)
            .one()
        )
        assert record.task_id == 71
        assert record.workspace_relative_path == "input/report.txt"
        assert record.workspace_category == "input"
        assert record.storage_status == "available"
        assert record.storage_key == (
            f"users/{user_id}/tasks/71/outputs/{record.file_id}/input/report.txt"
        )

    durable_files_before_failure = {
        path.relative_to(object_root)
        for path in object_root.rglob("*")
        if path.is_file()
    }
    failed_source = workspace.input_dir / "metadata-failure.txt"
    failed_source.write_text("must be compensated", encoding="utf-8")

    def fail_metadata_persistence(
        self: UploadedFileStore,
        file_record: UploadedFile,
    ) -> UploadedFile:
        raise RuntimeError("simulated metadata transaction failure")

    monkeypatch.setattr(
        UploadedFileStore,
        "add_already_durable",
        fail_metadata_persistence,
    )
    failed_registration = await register_channel_uploaded_files(
        workspace=workspace,
        task_id=71,
        user_id=user_id,
        files=(
            DownloadedChannelFile(
                name=failed_source.name,
                path=failed_source,
                mime_type="text/plain",
                size=failed_source.stat().st_size,
            ),
        ),
    )
    assert failed_registration == ()
    assert {
        path.relative_to(object_root)
        for path in object_root.rglob("*")
        if path.is_file()
    } == durable_files_before_failure
    with SessionLocal() as db:
        assert db.query(UploadedFile).count() == 1

    engine.dispose()
    get_unscoped_file_storage.cache_clear()


@pytest.mark.asyncio
async def test_prepare_channel_task_compensates_late_claim_before_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()
    claim = channel_runtime._ChannelTaskClaimSnapshot(
        user_id=7,
        task_id=19,
        is_new_task=True,
        lease=TaskLease(task_id=19, runner_id="runner-a", run_id="run-a"),
    )
    compensated: list[channel_runtime._ChannelTaskClaimSnapshot] = []

    def blocking_prepare(**_kwargs):  # type: ignore[no-untyped-def]
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return claim

    def compensate(snapshot: channel_runtime._ChannelTaskClaimSnapshot) -> bool:
        compensated.append(snapshot)
        return True

    monkeypatch.setattr(
        "xagent.web.services.channel_runtime._prepare_channel_task_sync",
        blocking_prepare,
    )
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime._compensate_channel_task_claim_sync",
        compensate,
    )

    preparation = asyncio.create_task(
        prepare_channel_task(
            channel_id=3,
            external_user_id="telegram-user",
            active_task_id=None,
            text="hello",
            channel_name="Telegram",
        )
    )
    await asyncio.to_thread(worker_started.wait, 2)
    preparation.cancel()
    allow_worker.set()

    with pytest.raises(asyncio.CancelledError):
        await preparation
    assert compensated == [claim]


@pytest.mark.asyncio
async def test_load_active_channel_configs_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()

    def blocking_load(*_args, **_kwargs) -> tuple:  # type: ignore[no-untyped-def]
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return ()

    monkeypatch.setattr(
        "xagent.web.services.channel_runtime._load_active_channel_configs_sync",
        blocking_load,
    )

    loading = asyncio.create_task(
        load_active_channel_configs(
            channel_type="telegram",
            required_config_keys=("bot_token",),
        )
    )
    await asyncio.to_thread(worker_started.wait, 2)
    await asyncio.sleep(0)

    assert loading.done() is False
    allow_worker.set()
    assert await loading == ()
