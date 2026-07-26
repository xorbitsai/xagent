"""Connection-ownership tests for the three shared upload entry points."""

from __future__ import annotations

import asyncio
import io
import threading
from pathlib import Path

import pytest
from fastapi.datastructures import UploadFile

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.storage import FsspecFileStorage
from xagent.core.workspace import TaskWorkspace
from xagent.web.api import files as files_api
from xagent.web.api import websocket as websocket_api
from xagent.web.api.public_chat_access import (
    PublicChatAccessContext,
    ShareChatAccessContext,
    upload_public_chat_files,
    upload_share_chat_files,
)
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.managed_file_ref import (
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
        raise DurableStorageOperationError("storage cleanup unavailable")

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
    websocket_api._register_uploaded_files_for_agent(
        type("AgentService", (), {"workspace": workspace})(),
        [file_info],
    )

    assert put_calls == 1
    assert workspace._recently_registered_files[str(Path(storage_path).resolve())] == (
        file_id
    )
    assert Path(str(file_info["workspace_path"])).exists()
    assert tmp_path.exists()
