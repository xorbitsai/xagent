"""Each /ingest-cloud file reads credentials in its own Session, and one file
raising cannot end the request (#2664).
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from tests.web.api import test_kb_dir as kb_dir
from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
from xagent.web.api import cloud_storage
from xagent.web.models.database import get_db
from xagent.web.models.user_oauth import UserOAuth

test_env = kb_dir.test_env
temp_uploads = kb_dir.temp_uploads

METADATA_STORE = "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store"


def _drive_files(count: int) -> list[dict[str, str]]:
    return [
        {"provider": "google-drive", "fileId": f"drive-{i}", "fileName": f"f{i}.csv"}
        for i in range(count)
    ]


def _connect_drive(test_env, monkeypatch) -> None:
    _app, _headers, user, sessions = test_env
    db = sessions()
    try:
        db.add(
            UserOAuth(
                user_id=int(user.id),
                provider="google-drive",
                access_token="token",
                refresh_token="refresh",
                scope="drive.file",
            )
        )
        db.commit()
    finally:
        db.close()
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")


def test_credentials_are_read_in_a_session_the_worker_owns(
    test_env, temp_uploads, monkeypatch
) -> None:
    app, headers, _user, _ = test_env
    _connect_drive(test_env, monkeypatch)
    request_sessions: list[Any] = []
    credential_sessions: list[Any] = []
    loop_threads: list[int] = []
    credential_threads: list[int] = []
    original_override = app.dependency_overrides[get_db]

    def _recording_get_db():
        for db in original_override():
            request_sessions.append(db)
            yield db

    def _on_loop(*_args, **_kwargs):
        loop_threads.append(threading.get_ident())

    def _spy(user_id, db, account_id=None):
        credential_sessions.append(db)
        credential_threads.append(threading.get_ident())
        return cloud_storage.get_google_credentials(user_id, db, account_id)

    app.dependency_overrides[get_db] = _recording_get_db
    try:
        with ExitStack() as stack:
            for p in (
                patch("xagent.web.api.kb.get_google_credentials", side_effect=_spy),
                patch("xagent.web.api.kb.build", side_effect=RuntimeError("stop")),
                patch(
                    "xagent.web.api.kb._ensure_collection_access",
                    new_callable=AsyncMock,
                    side_effect=_on_loop,
                ),
                patch("xagent.web.api.kb.get_collection_sync", return_value=object()),
            ):
                stack.enter_context(p)
            response = TestClient(app).post(
                "/api/kb/ingest-cloud",
                json={"collection": "cloud_coll", "files": _drive_files(2)},
                headers=headers,
            )
    finally:
        app.dependency_overrides[get_db] = original_override

    assert response.status_code == 200
    assert [e["message"] for e in response.json()] == [
        "Google Drive metadata lookup failed"
    ] * 2
    assert len(credential_sessions) == 2
    assert len({id(db) for db in credential_sessions}) == 2
    assert not {id(db) for db in credential_sessions} & {
        id(db) for db in request_sessions
    }
    assert not any(db.in_transaction() for db in credential_sessions)
    assert len(loop_threads) == 1
    assert loop_threads[0] not in credential_threads


def test_concurrent_credential_reads_all_see_the_connected_account(
    test_env, temp_uploads, monkeypatch
) -> None:
    app, headers, _user, _ = test_env
    _connect_drive(test_env, monkeypatch)
    messages: set[str] = set()
    with ExitStack() as stack:
        for p in (
            patch("xagent.web.api.kb.build", side_effect=RuntimeError("stop")),
            patch(
                "xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock
            ),
            patch("xagent.web.api.kb.get_collection_sync", return_value=object()),
        ):
            stack.enter_context(p)
        client = TestClient(app)
        for _ in range(20):
            response = client.post(
                "/api/kb/ingest-cloud",
                json={"collection": "cloud_coll", "files": _drive_files(5)},
                headers=headers,
            )
            assert response.status_code == 200
            messages.update(e["message"] for e in response.json())

    assert messages == {"Google Drive metadata lookup failed"}


def test_a_file_raising_outside_its_try_does_not_end_the_request(
    test_env, temp_uploads, monkeypatch
) -> None:
    app, headers, _user, _ = test_env
    sibling_ingesting = threading.Event()
    raised = threading.Event()
    lock = threading.Lock()
    calls: list[int] = []

    def _credentials(*_args, **_kwargs):
        with lock:
            calls.append(1)
            first = len(calls) == 1
        if first:
            return object()
        sibling_ingesting.wait(8)
        raised.set()
        raise OperationalError("SELECT user_oauth", {}, Exception("server closed"))

    def _ingest(**_kwargs):
        sibling_ingesting.set()
        raised.wait(8)
        return IngestionResult(
            status="success",
            doc_id="doc",
            message="ok",
            completed_steps=[
                {"name": "register_document", "metadata": {"created": True}}
            ],
        )

    class _Files:
        def get(self, fileId, **_kw):
            return kb_dir._fake_drive_metadata_request(fileId, "f.csv", "text/csv")

        def get_media(self, fileId, **_kw):
            return {"fileId": fileId}

    class _Downloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b"content")
            return None, True

    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value=None)
    metadata_store.save_collection_config = AsyncMock()
    with ExitStack() as stack:
        for p in (
            patch("xagent.web.api.kb.get_google_credentials", side_effect=_credentials),
            patch(
                "xagent.web.api.kb.build", return_value=SimpleNamespace(files=_Files)
            ),
            patch("xagent.web.api.kb.MediaIoBaseDownload", _Downloader),
            patch("xagent.web.api.kb.run_document_ingestion", side_effect=_ingest),
            patch(
                "xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock
            ),
            patch(
                "xagent.web.api.kb.get_collection_sync", side_effect=ValueError("new")
            ),
            patch(METADATA_STORE, return_value=metadata_store),
        ):
            stack.enter_context(p)
        response = TestClient(app).post(
            "/api/kb/ingest-cloud",
            json={"collection": "cloud_coll", "files": _drive_files(2)},
            headers=headers,
        )

    assert response.status_code == 200
    entries = sorted(response.json(), key=lambda e: e["status"])
    assert [e["status"] for e in entries] == ["error", "success"]
    assert entries[0]["message"].startswith("Unexpected error: ")
    assert "server closed" in entries[0]["message"]
    raised_at = [e["status"] for e in response.json()].index("error")
    assert entries[0]["doc_id"] == f"f{raised_at}.csv"
    metadata_store.save_collection_config.assert_awaited_once()


def test_a_new_collection_whose_only_file_raises_is_cleaned_up(
    test_env, temp_uploads
) -> None:
    app, headers, user, _ = test_env
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value=None)
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock(return_value={})
    raised = OperationalError("SELECT user_oauth", {}, Exception("server closed"))
    with (
        patch("xagent.web.api.kb.get_google_credentials", side_effect=raised),
        patch(METADATA_STORE, return_value=metadata_store),
    ):
        response = TestClient(app).post(
            "/api/kb/ingest-cloud",
            json={"collection": "cloud_new_coll", "files": _drive_files(1)},
            headers=headers,
        )

    assert response.status_code == 200
    assert [(e["status"], e["doc_id"]) for e in response.json()] == [
        ("error", "f0.csv")
    ]
    metadata_store.save_collection_config.assert_not_awaited()
    metadata_store.delete_collection_metadata.assert_awaited_once_with(
        collection_name="cloud_new_coll",
        user_id=int(user.id),
        is_admin=False,
        delete_orphaned_metadata=True,
    )
