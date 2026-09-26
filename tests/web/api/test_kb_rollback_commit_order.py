"""Row deletes commit before their file bytes are deleted (#2664)."""

from __future__ import annotations

import asyncio
import threading
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from tests.web.api import test_kb_dir as kb_dir
from tests.web.api import test_kb_local_rollback_contract as local
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.storage import FsspecFileStorage
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionOperationResult,
    IngestionResult,
)
from xagent.core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from xagent.web.api import kb as kb_module
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.managed_file_ref import ManagedFileRef

test_env = kb_dir.test_env
temp_uploads = kb_dir.temp_uploads

FILE_A = {"provider": "google-drive", "fileId": "drive-A", "fileName": "a.csv"}
FILE_B = {"provider": "google-drive", "fileId": "drive-B", "fileName": "b.csv"}
METADATA_STORE = "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store"


def _durable_storage(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()


def _rows(sessions) -> dict[str, tuple[str, str, str]]:
    db = sessions()
    try:
        return {
            str(r.filename): (str(r.file_id), str(r.storage_key), str(r.storage_path))
            for r in db.query(UploadedFile).all()
        }
    finally:
        db.close()


def _rows_without_bytes(sessions) -> list[str]:
    storage = get_unscoped_file_storage()
    return sorted(
        name for name, (_, key, _) in _rows(sessions).items() if not storage.exists(key)
    )


class _MetadataStore:
    """Config read hops to a worker thread, like the LanceDB store, and can be held."""

    def __init__(self, hold=None) -> None:
        self.hold = hold
        self.delete_collection_metadata = AsyncMock(return_value={})
        self.save_collection_config = AsyncMock()

    async def get_collection_config(self, **_kw: Any) -> None:
        def _read() -> None:
            if self.hold is not None:
                self.hold()

        return await asyncio.to_thread(_read)


def _post_cloud(test_env, files, ingest, *extra, downloader_gate=None):
    app, headers, _user, _ = test_env

    class _Files:
        def get(self, fileId, **_kw):
            name = {"drive-A": "a.csv", "drive-B": "b.csv"}[fileId]
            return kb_dir._fake_drive_metadata_request(fileId, name, "text/csv")

        def get_media(self, fileId, **_kw):
            return {"fileId": fileId}

    class _Downloader:
        def __init__(self, fh, request):
            self._fh = fh
            self._file_id = request["fileId"]

        def next_chunk(self):
            if downloader_gate is not None:
                downloader_gate(self._file_id)
            self._fh.write(self._file_id.encode())
            return None, True

    with ExitStack() as stack:
        for p in (
            patch("xagent.web.api.kb.get_google_credentials", return_value=object()),
            patch(
                "xagent.web.api.kb.build", return_value=SimpleNamespace(files=_Files)
            ),
            patch("xagent.web.api.kb.MediaIoBaseDownload", _Downloader),
            patch("xagent.web.api.kb.run_document_ingestion", side_effect=ingest),
            patch(
                "xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock
            ),
            *extra,
        ):
            stack.enter_context(p)
        return TestClient(app).post(
            "/api/kb/ingest-cloud",
            json={"collection": "cloud_coll", "files": files},
            headers=headers,
        )


def _broken_commit() -> None:
    raise OperationalError("COMMIT", {}, Exception("disk I/O error"))


def _is_file(kw: dict[str, Any], prefix: str) -> bool:
    return Path(kw["source_path"]).name.startswith(prefix)


def _failed(**_kw: Any) -> IngestionResult:
    return IngestionResult(status="error", message="embedding down")


def test_cloud_collection_failure_after_file_step_leaves_no_row_without_bytes(
    test_env, temp_uploads, monkeypatch, tmp_path
) -> None:
    _durable_storage(monkeypatch, tmp_path)
    sessions = test_env[3]

    class _BrokenStore:
        def list_document_records(self, **_kw):
            raise RuntimeError("lance down")

    response = _post_cloud(
        test_env,
        [FILE_A],
        _failed,
        patch("xagent.web.api.kb.get_collection_sync", return_value=object()),
        patch("xagent.web.api.kb.get_vector_index_store", _BrokenStore),
    )

    assert response.status_code == 200
    assert response.json()[0]["message"].endswith(
        "lance down. Original ingestion error: embedding down"
    )
    assert "a.csv" not in _rows(sessions)


def test_cloud_failed_row_delete_commit_keeps_row_and_bytes(
    test_env, temp_uploads, monkeypatch, tmp_path
) -> None:
    _durable_storage(monkeypatch, tmp_path)
    sessions = test_env[3]
    seen: dict[str, Any] = {}
    real_orphan = kb_module._delete_uploaded_file_if_orphaned

    def _orphan_then_break_commit(db, **kwargs):
        deleted = real_orphan(db, **kwargs)
        db.commit = _broken_commit
        return deleted

    def _ingest(**kw):
        seen.update(_rows(sessions))
        return _failed()

    response = _post_cloud(
        test_env,
        [FILE_A],
        _ingest,
        patch("xagent.web.api.kb.get_collection_sync", return_value=object()),
        patch(
            "xagent.web.api.kb._delete_uploaded_file_if_orphaned",
            side_effect=_orphan_then_break_commit,
        ),
    )

    assert "disk I/O error" in response.json()[0]["message"]
    assert _rows(sessions) == seen
    assert get_unscoped_file_storage().exists(seen["a.csv"][1])


def test_cloud_sibling_rollback_cannot_undo_this_files_row_delete(
    test_env, temp_uploads, monkeypatch, tmp_path
) -> None:
    _durable_storage(monkeypatch, tmp_path)
    sessions = test_env[3]
    a_in_collection_step = threading.Event()
    b_rolled_back = threading.Event()
    waited: list[bool] = []
    original_restore = kb_module._restore_ingest_file_backup

    def _restore(**kw):
        try:
            return original_restore(**kw)
        finally:
            if Path(kw["file_path"]).name.startswith("b__"):
                b_rolled_back.set()

    def _ingest(**kw):
        if _is_file(kw, "a__"):
            return _failed()
        waited.append(a_in_collection_step.wait(8))
        return IngestionResult(
            status="partial",
            doc_id="doc-b",
            completed_steps=[
                {"name": "register_document", "metadata": {"created": True}}
            ],
            message="embedding failed",
        )

    def _hold() -> None:
        a_in_collection_step.set()
        waited.append(b_rolled_back.wait(8))

    response = _post_cloud(
        test_env,
        [FILE_A, FILE_B],
        _ingest,
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("new")),
        patch(
            "xagent.web.api.kb.delete_collection",
            return_value=SimpleNamespace(status="success", message=""),
        ),
        patch(
            "xagent.web.api.kb.delete_document",
            return_value=SimpleNamespace(status="error", message="lance busy"),
        ),
        patch(METADATA_STORE, return_value=_MetadataStore(_hold)),
        patch("xagent.web.api.kb._restore_ingest_file_backup", side_effect=_restore),
    )

    a_entry, b_entry = response.json()
    assert a_entry["message"] == "embedding down"
    assert b_entry["message"].startswith("Failed to fully roll back cloud ingest")
    assert len(waited) >= 2 and all(waited)
    assert "a.csv" not in _rows(sessions)
    assert _rows_without_bytes(sessions) == []


def test_cloud_sibling_upsert_is_not_blocked_by_a_rollback_in_progress(
    test_env, temp_uploads, monkeypatch, tmp_path
) -> None:
    _durable_storage(monkeypatch, tmp_path)
    a_in_collection_step = threading.Event()
    b_ingested = threading.Event()
    waited: list[bool] = []

    def _gate(file_id: str) -> None:
        if file_id == "drive-B":
            waited.append(a_in_collection_step.wait(8))

    def _hold() -> None:
        a_in_collection_step.set()
        waited.append(b_ingested.wait(8))

    def _ingest(**kw):
        if _is_file(kw, "a__"):
            return _failed()
        b_ingested.set()
        return IngestionResult(status="success", doc_id="doc-b", message="ok")

    response = _post_cloud(
        test_env,
        [FILE_A, FILE_B],
        _ingest,
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("new")),
        patch(
            "xagent.web.api.kb.delete_collection",
            return_value=SimpleNamespace(status="success", message=""),
        ),
        patch(METADATA_STORE, return_value=_MetadataStore(_hold)),
        downloader_gate=_gate,
    )

    a_entry, b_entry = response.json()
    assert a_entry["message"] == "embedding down"
    assert (b_entry["status"], b_entry["message"]) == ("success", "ok")
    assert len(waited) >= 2 and all(waited)


def test_local_whole_collection_metadata_failure_leaves_no_row_without_bytes(
    test_env, temp_uploads, monkeypatch, tmp_path
) -> None:
    _durable_storage(monkeypatch, tmp_path)

    response = local._post_ingest(
        test_env,
        "x.txt",
        "fresh",
        patch("xagent.web.api.kb.run_document_ingestion", side_effect=_failed),
        patch(
            "xagent.web.api.kb.delete_collection",
            return_value=SimpleNamespace(status="success", message=""),
        ),
        patch(METADATA_STORE, return_value=_MetadataStore()),
        patch(
            "xagent.web.api.kb._cleanup_failed_new_collection_metadata",
            AsyncMock(side_effect=RuntimeError("metadata store down")),
        ),
    )

    assert response.status_code == 500
    assert "metadata store down" in response.json()["detail"]
    assert _rows(test_env[3]) == {}


def _seed(test_env, uploads, durable_by_name, collection="", owner_id=None):
    _app, _headers, user, sessions = test_env
    owner_id = owner_id or int(user.id)
    db = sessions()
    try:
        for name, durable in durable_by_name.items():
            path = uploads / f"user_{owner_id}" / collection / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
            record = UploadedFile(
                user_id=owner_id,
                filename=name,
                # Resolved, as the directory pass compares against a resolved dir.
                storage_path=str(path.resolve()),
                mime_type="text/plain",
                file_size=path.stat().st_size,
            )
            db.add(record)
            db.flush()
            if durable:
                ManagedFileRef(record).sync_to_durable()
        db.commit()
    finally:
        db.close()
    return {n: row for n, row in _rows(sessions).items() if n in durable_by_name}


def _collection_store(documents: list[DocumentRecord]) -> list[Any]:
    store = MagicMock()
    store.list_document_records.side_effect = lambda *_a, **_kw: list(documents)

    def _delete_collection(*_args, **_kwargs):
        documents.clear()
        return CollectionOperationResult(
            status="success",
            collection="demo",
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )

    return [
        patch("xagent.web.api.kb.get_vector_index_store", return_value=store),
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_vector_index_store",
            return_value=store,
        ),
        patch("xagent.web.api.kb.delete_collection", side_effect=_delete_collection),
    ]


def _delete_collection_api(test_env, documented, *extra):
    app, headers, _user, _ = test_env
    documents = [
        DocumentRecord(doc_id=name, file_id=file_id, source_path="x")
        for name, (file_id, _, _) in documented.items()
    ]
    with ExitStack() as stack:
        for p in (
            *_collection_store(documents),
            patch(
                "xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock
            ),
            *extra,
        ):
            stack.enter_context(p)
        return TestClient(app).delete("/api/kb/collections/demo", headers=headers)


def test_collection_delete_byte_failure_leaves_no_row_without_bytes(
    test_env, temp_uploads, monkeypatch, tmp_path
) -> None:
    _durable_storage(monkeypatch, tmp_path)
    rows = _seed(test_env, temp_uploads, {"one.txt": True, "two.txt": True})
    real_delete = FsspecFileStorage.delete
    failed: list[str] = []

    def _first_delete_fails(self, key: str) -> None:
        if not failed:
            failed.append(key)
            raise RuntimeError("object store 503")
        real_delete(self, key)

    monkeypatch.setattr(FsspecFileStorage, "delete", _first_delete_fails)
    response = _delete_collection_api(test_env, rows)

    assert response.status_code == 200
    assert _rows(test_env[3]) == {}
    storage = get_unscoped_file_storage()
    assert [key for _, key, _ in rows.values() if storage.exists(key)] == failed


def test_collection_delete_failed_commit_keeps_every_file_byte(
    test_env, temp_uploads, monkeypatch, tmp_path
) -> None:
    _durable_storage(monkeypatch, tmp_path)
    rows = _seed(test_env, temp_uploads, {"legacy.txt": False, "durable.txt": True})
    # No document names this row; the directory pass finds it by path.
    in_dir = _seed(test_env, temp_uploads, {"in_dir.txt": True}, collection="demo")
    real_cleanup = kb_module.delete_collection_uploaded_files

    def _cleanup_then_fail_commit(db, **kwargs):
        db.commit = _broken_commit
        return real_cleanup(db, **kwargs)

    response = _delete_collection_api(
        test_env,
        rows,
        patch(
            "xagent.web.api.kb.delete_collection_uploaded_files",
            side_effect=_cleanup_then_fail_commit,
        ),
    )

    assert response.status_code == 500
    assert _rows(test_env[3]) == {**rows, **in_dir}
    storage = get_unscoped_file_storage()
    assert storage.exists(in_dir["in_dir.txt"][1])
    assert storage.exists(rows["durable.txt"][1])
    assert Path(rows["legacy.txt"][2]).exists()


def test_collection_delete_commits_every_owners_row_deletes(
    test_env, temp_uploads, monkeypatch, tmp_path
) -> None:
    _durable_storage(monkeypatch, tmp_path)
    user_id = int(test_env[2].id)
    for owner_id in (user_id, user_id + 1):
        _seed(test_env, temp_uploads, {f"{owner_id}.txt": True}, "demo", owner_id)
    db = test_env[3]()
    try:
        with ExitStack() as stack:
            for p in _collection_store([]):
                stack.enter_context(p)
            kb_module._perform_kb_collection_delete("demo", user_id, True, db)
    finally:
        db.close()

    assert _rows(test_env[3]) == {}


def test_run_after_commit_runs_every_delete_and_logs_each_failure(caplog) -> None:
    ran: list[str] = []

    def _fail(name: str) -> Any:
        def _action() -> None:
            ran.append(name)
            raise OSError(name)

        return _action

    with pytest.raises(OSError, match="first"):
        kb_module._run_after_commit([_fail("first"), _fail("second")])

    assert ran == ["first", "second"]
    assert len([r for r in caplog.records if r.exc_info]) == 2


def test_cloud_byte_delete_failure_after_commit_reports_incomplete_rollback(
    test_env, temp_uploads, monkeypatch, tmp_path
) -> None:
    _durable_storage(monkeypatch, tmp_path)
    sessions = test_env[3]
    seen: dict[str, Any] = {}

    def _ingest(**kw):
        seen.update(_rows(sessions))
        return _failed()

    def _broken_delete(self, key: str) -> None:
        raise RuntimeError("object store 503")

    monkeypatch.setattr(FsspecFileStorage, "delete", _broken_delete)
    response = _post_cloud(
        test_env,
        [FILE_A],
        _ingest,
        patch("xagent.web.api.kb.get_collection_sync", return_value=object()),
    )

    assert "object store 503" in response.json()[0]["message"]
    assert "a.csv" not in _rows(sessions)
    assert get_unscoped_file_storage().exists(seen["a.csv"][1])
