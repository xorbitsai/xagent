"""Tests for uploaded files reconciliation helpers."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lancedb
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.web.services.kb_file_service as kb_file_service
from xagent.core.tools.core.RAG_tools.kb import KBCoordinator
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_chunks_table,
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.management.status import write_ingestion_status
from xagent.core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from xagent.core.tools.core.RAG_tools.storage.factory import (
    bind_storage_shim_for_current_context,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.kb_file_service import (
    aggregate_uploaded_file_statuses,
    list_documents_for_user,
    reconcile_uploaded_files,
)


@pytest.fixture
def reconcile_env(monkeypatch: pytest.MonkeyPatch):
    with (
        tempfile.TemporaryDirectory() as lancedb_dir,
        tempfile.TemporaryDirectory() as uploads_dir,
    ):
        monkeypatch.setenv("LANCEDB_DIR", lancedb_dir)
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", uploads_dir)

        conn = lancedb.connect(lancedb_dir)
        ensure_documents_table(conn)
        docs_table = conn.open_table("documents")

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        yield docs_table, SessionLocal, Path(uploads_dir)


def _create_user(session_local: sessionmaker, user_id: int = 1) -> None:
    db = session_local()
    db.add(
        User(
            id=user_id,
            username=f"user_{user_id}",
            password_hash="hash",
            is_admin=False,
        )
    )
    db.commit()
    db.close()


def test_file_status_cache_evicts_lru_entry_when_maxsize_exceeded():
    cache = kb_file_service._FileStatusCache(ttl_seconds=60, maxsize=2)

    cache.put(1, ["file-a"], {"file-a": "SUCCESS"})
    cache.put(1, ["file-b"], {"file-b": "FAILED"})
    assert cache.get(1, ["file-a"]) == {"file-a": "SUCCESS"}

    cache.put(1, ["file-c"], {"file-c": "UNKNOWN"})

    assert cache.get(1, ["file-a"]) == {"file-a": "SUCCESS"}
    assert cache.get(1, ["file-b"]) is None
    assert cache.get(1, ["file-c"]) == {"file-c": "UNKNOWN"}


def test_aggregate_uploaded_file_statuses_returns_expected_priority(reconcile_env):
    docs_table, session_local, uploads_dir = reconcile_env
    _create_user(session_local, user_id=1)

    file_path = uploads_dir / "user_1" / "kb" / "a.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content", encoding="utf-8")

    db = session_local()
    file_record = UploadedFile(
        user_id=1,
        filename="a.md",
        storage_path=str(file_path),
        mime_type="text/markdown",
        file_size=file_path.stat().st_size,
    )
    db.add(file_record)
    db.commit()
    db.refresh(file_record)

    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-agg",
                "file_id": file_record.file_id,
                "source_path": str(file_path),
                "user_id": 1,
            }
        ]
    )
    write_ingestion_status(
        "kb",
        "doc-agg",
        status="success",
        message="done",
        parse_hash="",
        user_id=1,
    )

    status_map = aggregate_uploaded_file_statuses(
        file_ids=[file_record.file_id],
        user_id=1,
        is_admin=False,
    )
    assert status_map[file_record.file_id] == "SUCCESS"
    db.close()


def test_aggregate_uploaded_file_statuses_treats_legacy_indexed_file_as_success(
    reconcile_env,
):
    docs_table, session_local, uploads_dir = reconcile_env
    ensure_chunks_table(get_connection_from_env())
    chunks_table = get_connection_from_env().open_table("chunks")
    _create_user(session_local, user_id=1)

    file_path = uploads_dir / "user_1" / "kb" / "legacy.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("legacy searchable content", encoding="utf-8")

    db = session_local()
    file_record = UploadedFile(
        user_id=1,
        filename="legacy.md",
        storage_path=str(file_path),
        mime_type="text/markdown",
        file_size=file_path.stat().st_size,
    )
    db.add(file_record)
    db.commit()
    db.refresh(file_record)

    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-legacy",
                "file_id": file_record.file_id,
                "source_path": str(file_path),
                "user_id": 1,
            }
        ]
    )
    chunks_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-legacy",
                "parse_hash": "parse-legacy",
                "chunk_id": "chunk-1",
                "index": 0,
                "text": "legacy searchable content",
                "page_number": 1,
                "section": "",
                "anchor": "",
                "json_path": "",
                "chunk_hash": "chunk-hash",
                "config_hash": "config-hash",
                "created_at": datetime.now(timezone.utc),
                "metadata": "{}",
                "user_id": 1,
            }
        ]
    )

    status_map = aggregate_uploaded_file_statuses(
        file_ids=[file_record.file_id],
        user_id=1,
        is_admin=False,
        use_cache=False,
    )

    assert status_map[file_record.file_id] == "SUCCESS"
    db.close()


def test_reconcile_uploaded_files_deletes_only_stale_failed_by_default(reconcile_env):
    docs_table, session_local, uploads_dir = reconcile_env
    _create_user(session_local, user_id=1)
    db = session_local()

    failed_path = uploads_dir / "user_1" / "kb" / "failed.md"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text("failed", encoding="utf-8")

    unknown_path = uploads_dir / "user_1" / "kb" / "unknown.md"
    unknown_path.write_text("unknown", encoding="utf-8")

    success_path = uploads_dir / "user_1" / "kb" / "success.md"
    success_path.write_text("success", encoding="utf-8")

    old_time = datetime.now(timezone.utc) - timedelta(days=10)

    failed_file = UploadedFile(
        user_id=1,
        filename="failed.md",
        storage_path=str(failed_path),
        mime_type="text/markdown",
        file_size=failed_path.stat().st_size,
        created_at=old_time,
    )
    unknown_file = UploadedFile(
        user_id=1,
        filename="unknown.md",
        storage_path=str(unknown_path),
        mime_type="text/markdown",
        file_size=unknown_path.stat().st_size,
        created_at=old_time,
    )
    success_file = UploadedFile(
        user_id=1,
        filename="success.md",
        storage_path=str(success_path),
        mime_type="text/markdown",
        file_size=success_path.stat().st_size,
        created_at=old_time,
    )
    db.add_all([failed_file, unknown_file, success_file])
    db.commit()
    db.refresh(failed_file)
    db.refresh(unknown_file)
    db.refresh(success_file)

    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-failed",
                "file_id": failed_file.file_id,
                "source_path": str(failed_path),
                "user_id": 1,
            },
            {
                "collection": "kb",
                "doc_id": "doc-success",
                "file_id": success_file.file_id,
                "source_path": str(success_path),
                "user_id": 1,
            },
        ]
    )
    write_ingestion_status(
        "kb",
        "doc-failed",
        status="failed",
        message="failed",
        parse_hash="",
        user_id=1,
    )
    write_ingestion_status(
        "kb",
        "doc-success",
        status="success",
        message="ok",
        parse_hash="",
        user_id=1,
    )

    result = reconcile_uploaded_files(
        db,
        user_id=1,
        is_admin=False,
        stale_ttl_hours=24,
        delete_stale=True,
    )

    assert result["stale_candidates"] == 2
    assert result["deleted"] == 1
    assert result["cleanup_errors"] == 0
    assert not failed_path.exists()
    assert unknown_path.exists()
    assert success_path.exists()

    remaining = db.query(UploadedFile).all()
    remaining_file_ids = {record.file_id for record in remaining}
    assert remaining_file_ids == {unknown_file.file_id, success_file.file_id}

    refreshed_docs_table = get_connection_from_env().open_table("documents")
    rows = refreshed_docs_table.search().where("collection == 'kb'").to_list()
    row_by_doc_id = {str(row.get("doc_id")): row for row in rows}
    assert "doc-failed" not in row_by_doc_id
    assert "doc-success" in row_by_doc_id
    db.close()


def test_reconcile_uploaded_files_does_not_commit_caller_session(
    reconcile_env,
    monkeypatch: pytest.MonkeyPatch,
):
    docs_table, session_local, uploads_dir = reconcile_env
    _create_user(session_local, user_id=1)
    db = session_local()

    failed_path = uploads_dir / "user_1" / "kb" / "failed-no-commit.md"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text("failed", encoding="utf-8")
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    failed_file = UploadedFile(
        user_id=1,
        filename="failed-no-commit.md",
        storage_path=str(failed_path),
        mime_type="text/markdown",
        file_size=failed_path.stat().st_size,
        created_at=old_time,
    )
    db.add(failed_file)
    db.commit()
    db.refresh(failed_file)

    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-no-commit",
                "file_id": failed_file.file_id,
                "source_path": str(failed_path),
                "user_id": 1,
            }
        ]
    )
    write_ingestion_status(
        "kb",
        "doc-no-commit",
        status="failed",
        message="failed",
        parse_hash="",
        user_id=1,
    )

    commit_calls: list[bool] = []
    monkeypatch.setattr(db, "commit", lambda: commit_calls.append(True))

    result = reconcile_uploaded_files(
        db,
        user_id=1,
        is_admin=False,
        stale_ttl_hours=24,
        delete_stale=True,
    )

    assert result["deleted"] == 1
    assert commit_calls == []
    assert (
        db.query(UploadedFile)
        .filter(UploadedFile.file_id == failed_file.file_id)
        .first()
        is None
    )
    db.close()


def test_reconcile_uploaded_files_preserves_record_when_durable_delete_fails(
    reconcile_env, monkeypatch: pytest.MonkeyPatch
):
    docs_table, session_local, uploads_dir = reconcile_env
    _create_user(session_local, user_id=1)
    db = session_local()

    failed_path = uploads_dir / "user_1" / "kb" / "failed-durable.md"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text("failed", encoding="utf-8")
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    failed_file = UploadedFile(
        user_id=1,
        filename="failed-durable.md",
        storage_path=str(failed_path),
        storage_backend="s3",
        storage_key="users/1/uploads/file-1/failed-durable.md",
        storage_status="available",
        mime_type="text/markdown",
        file_size=failed_path.stat().st_size,
        created_at=old_time,
    )
    db.add(failed_file)
    db.commit()
    db.refresh(failed_file)

    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-durable-delete",
                "file_id": failed_file.file_id,
                "source_path": str(failed_path),
                "user_id": 1,
            }
        ]
    )
    write_ingestion_status(
        "kb",
        "doc-durable-delete",
        status="failed",
        message="failed",
        parse_hash="",
        user_id=1,
    )

    def fail_durable_delete(self):
        raise RuntimeError("remote delete failed")

    monkeypatch.setattr(
        "xagent.web.services.managed_file_ref.ManagedFileRef.delete_durable",
        fail_durable_delete,
    )

    result = reconcile_uploaded_files(
        db,
        user_id=1,
        is_admin=False,
        stale_ttl_hours=24,
        delete_stale=True,
    )

    assert result["cleanup_errors"] == 1
    assert result["deleted"] == 0
    assert failed_path.exists()
    still_exists = (
        db.query(UploadedFile)
        .filter(UploadedFile.file_id == failed_file.file_id)
        .first()
    )
    assert still_exists is not None
    db.close()


def test_reconcile_uploaded_files_records_cleanup_error_when_documents_delete_fails(
    reconcile_env, monkeypatch: pytest.MonkeyPatch
):
    docs_table, session_local, uploads_dir = reconcile_env
    _create_user(session_local, user_id=1)
    db = session_local()

    failed_path = uploads_dir / "user_1" / "kb" / "failed-delete.md"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text("failed", encoding="utf-8")

    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    failed_file = UploadedFile(
        user_id=1,
        filename="failed-delete.md",
        storage_path=str(failed_path),
        mime_type="text/markdown",
        file_size=failed_path.stat().st_size,
        created_at=old_time,
    )
    db.add(failed_file)
    db.commit()
    db.refresh(failed_file)

    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-delete-failed",
                "file_id": failed_file.file_id,
                "source_path": str(failed_path),
                "user_id": 1,
            }
        ]
    )
    write_ingestion_status(
        "kb",
        "doc-delete-failed",
        status="failed",
        message="failed",
        parse_hash="",
        user_id=1,
    )

    def _failing_cascade_delete(self, **_kw):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(
        LanceDBVectorIndexStore, "cascade_delete", _failing_cascade_delete
    )

    result = reconcile_uploaded_files(
        db,
        user_id=1,
        is_admin=False,
        stale_ttl_hours=24,
        delete_stale=True,
    )

    assert result["cleanup_errors"] == 1
    assert result["deleted"] == 0
    assert failed_path.exists()
    still_exists = (
        db.query(UploadedFile)
        .filter(UploadedFile.file_id == failed_file.file_id)
        .first()
    )
    assert still_exists is not None
    db.close()


def test_file_statuses_ignore_documents_the_user_cannot_see(reconcile_env):
    docs_table, _session_local, _uploads_dir = reconcile_env
    docs_table.add(
        [
            {"collection": "kb", "doc_id": "own", "file_id": "f-own", "user_id": 1},
            {"collection": "kb", "doc_id": "other", "file_id": "f-other", "user_id": 2},
            {"collection": "kb", "doc_id": "legacy", "file_id": "f-legacy"},
        ]
    )
    for doc_id, owner in (("own", 1), ("other", 2), ("legacy", None)):
        write_ingestion_status(
            "kb", doc_id, status="success", message="ok", parse_hash="", user_id=owner
        )
    file_ids = ["f-own", "f-other", "f-legacy"]

    assert aggregate_uploaded_file_statuses(
        file_ids=file_ids, user_id=1, is_admin=False, use_cache=False
    ) == {"f-own": "SUCCESS", "f-other": "UNKNOWN", "f-legacy": "UNKNOWN"}
    assert aggregate_uploaded_file_statuses(
        file_ids=file_ids, user_id=1, is_admin=True, use_cache=False
    ) == {"f-own": "SUCCESS", "f-other": "SUCCESS", "f-legacy": "SUCCESS"}


def test_listed_documents_stop_at_ten_thousand(reconcile_env):
    docs_table, _session_local, _uploads_dir = reconcile_env
    docs_table.add(
        [
            {"collection": "bulk", "doc_id": f"d-{i}", "file_id": "f", "user_id": 7}
            for i in range(10_001)
        ]
    )

    assert len(list_documents_for_user(user_id=7, is_admin=False)) == 10_000


def _unreachable(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("read the ambient store")


_AMBIENT = SimpleNamespace(
    get_vector_index_store=_unreachable, get_ingestion_status_store=_unreachable
)


class _OtherBackend:
    """Vector and status store of another backend, owned by one coordinator."""

    def __init__(
        self,
        docs: list[DocumentRecord],
        statuses: dict[tuple[str, str], str],
        indexed: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.docs = docs
        self.statuses = statuses
        self.indexed = set(indexed)
        self.failing_lookups: set[str] = set()
        self.down = False
        self.lookups: list[list[str]] = []
        self.status_reads: list[tuple[Any, ...]] = []
        self.indexed_reads: list[tuple[Any, ...]] = []
        self.cascades: list[dict[str, Any]] = []

    def list_document_records_by_file_ids(self, file_ids):
        file_ids = list(file_ids)
        self.lookups.append(file_ids)
        if self.down or (len(file_ids) == 1 and file_ids[0] in self.failing_lookups):
            raise RuntimeError("lookup down")
        return [doc for doc in self.docs if doc.file_id in file_ids]

    def load_ingestion_status(
        self, collection, doc_id=None, user_id=None, is_admin=False
    ):
        self.status_reads.append((collection, user_id, is_admin))
        return [
            {"doc_id": ref_doc_id, "status": status}
            for (ref_collection, ref_doc_id), status in self.statuses.items()
            if ref_collection == collection
        ]

    def list_indexed_doc_refs(self, doc_refs, user_id, is_admin):
        doc_refs = sorted(doc_refs)
        self.indexed_reads.append((doc_refs, user_id, is_admin))
        return self.indexed & set(doc_refs)

    def cascade_delete(self, **kwargs):
        self.cascades.append(kwargs)
        return {"documents": 1}


def _facade(store: Any) -> Any:
    shim = SimpleNamespace(
        get_vector_index_store=lambda: store, get_ingestion_status_store=lambda: store
    )
    return KBCoordinator(storage_shim=shim).file_compatibility


def _doc(file_id: str, collection: str, doc_id: str, user_id: Any = 7):
    return DocumentRecord(
        doc_id=doc_id, file_id=file_id, user_id=user_id, collection=collection
    )


def test_file_statuses_come_from_the_facades_stores():
    store = _OtherBackend(
        docs=[
            _doc("f-run", "kb", "r-1"),
            _doc("f-run", "kb", "r-2"),
            _doc("f-fail", "kb", "x-1"),
            _doc("f-mix", "kb", "m-1"),
            _doc("f-mix", "kb2", "m-2"),
            _doc("f-pad", " kb ", " p-1 "),
            _doc("f-idx", "kb", "i-1"),
            _doc("f-unk", "kb", "u-1"),
            _doc("f-empty", "kb", ""),
        ],
        statuses={
            ("kb", "r-1"): "running",
            ("kb", "r-2"): "success",
            ("kb", "x-1"): "failed",
            ("kb", "m-1"): "failed",
            ("kb2", "m-2"): "success",
            ("kb", "p-1"): "success",
        },
        indexed=(("kb", "i-1"),),
    )
    file_ids = "f-run f-fail f-mix f-pad f-idx f-unk f-empty f-none".split()

    with bind_storage_shim_for_current_context(_AMBIENT):
        statuses = _facade(store).aggregate_uploaded_file_statuses(
            file_ids=file_ids, user_id=7, is_admin=False, use_cache=False
        )

    assert statuses == {
        "f-run": "RUNNING",
        "f-fail": "FAILED",
        "f-mix": "SUCCESS",
        "f-pad": "SUCCESS",
        "f-idx": "SUCCESS",
        "f-unk": "UNKNOWN",
        "f-empty": "UNKNOWN",
        "f-none": "UNKNOWN",
    }
    assert store.lookups == [sorted(file_ids)]
    assert store.status_reads == [("kb", 7, False), ("kb2", 7, False)]
    visible = [("kb", d) for d in "r-1 r-2 x-1 m-1 p-1 i-1 u-1".split()]
    assert store.indexed_reads == [(sorted([*visible, ("kb2", "m-2")]), 7, False)]


@pytest.mark.parametrize(
    ("user_id", "is_admin", "expected"),
    [
        (7, False, {"f-own": "SUCCESS", "f-other": "UNKNOWN", "f-legacy": "UNKNOWN"}),
        (
            None,
            False,
            {"f-own": "UNKNOWN", "f-other": "UNKNOWN", "f-legacy": "UNKNOWN"},
        ),
        (1, True, {"f-own": "SUCCESS", "f-other": "SUCCESS", "f-legacy": "SUCCESS"}),
    ],
)
def test_file_statuses_keep_the_owner_rule(user_id, is_admin, expected):
    store = _OtherBackend(
        docs=[
            _doc("f-own", "kb", "own"),
            _doc("f-other", "kb", "other", user_id=8),
            _doc("f-legacy", "kb", "legacy", user_id=None),
        ],
        statuses={
            ("kb", "own"): "success",
            ("kb", "other"): "success",
            ("kb", "legacy"): "success",
        },
    )

    assert (
        _facade(store).aggregate_uploaded_file_statuses(
            file_ids=list(expected), user_id=user_id, is_admin=is_admin, use_cache=False
        )
        == expected
    )
    assert [read[1:] for read in store.indexed_reads] == [(user_id, is_admin)]


def test_documents_are_listed_from_the_facades_store():
    rows = [{"collection": "kb", "doc_id": ""}]
    calls: list[dict[str, Any]] = []

    def list_document_rows(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return rows

    with bind_storage_shim_for_current_context(_AMBIENT):
        listed = _facade(
            SimpleNamespace(list_document_rows=list_document_rows)
        ).list_documents_for_user(user_id=7, is_admin=False)

    assert listed is rows
    assert calls == [{"user_id": 7, "is_admin": False}]


def _old_uploads(session_local, uploads_dir: Path, *names: str) -> list[UploadedFile]:
    _create_user(session_local, user_id=1)
    db = session_local()
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    records = []
    for offset, name in enumerate(names):
        path = uploads_dir / "user_1" / "kb" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        records.append(
            UploadedFile(
                user_id=1,
                filename=name,
                storage_path=str(path),
                mime_type="text/markdown",
                file_size=path.stat().st_size,
                created_at=old_time + timedelta(minutes=offset),
            )
        )
    db.add_all(records)
    db.commit()
    for record in records:
        db.refresh(record)
    db.close()
    return records


def test_reconcile_looks_up_and_cascades_through_the_facades_store(reconcile_env):
    _docs_table, session_local, uploads_dir = reconcile_env
    a, b, c = _old_uploads(session_local, uploads_dir, "a.md", "b.md", "c.md")
    store = _OtherBackend(
        docs=[
            _doc(a.file_id, " kb ", " d-a ", user_id=1),
            _doc(a.file_id, "", "d-x", user_id=1),
            _doc(a.file_id, "kb", "", user_id=1),
            _doc(a.file_id, "kb", "d-o", user_id=2),
            _doc(b.file_id, "kb", "d-b", user_id=1),
            _doc(c.file_id, "kb", "d-c", user_id=1),
        ],
        statuses={
            ("kb", "d-a"): "failed",
            ("kb", "d-b"): "failed",
            ("kb", "d-c"): "success",
        },
    )
    store.failing_lookups = {b.file_id}
    db = session_local()

    with bind_storage_shim_for_current_context(_AMBIENT):
        result = _facade(store).reconcile_uploaded_files(
            db, user_id=1, is_admin=False, stale_ttl_hours=24
        )

    assert result == dict(scanned=3, stale_candidates=2, deleted=1, cleanup_errors=1)
    assert store.lookups == [
        sorted([a.file_id, b.file_id, c.file_id]),
        [a.file_id],
        [b.file_id],
    ]
    assert [cascade["doc_id"] for cascade in store.cascades] == ["d-a", "d-o"]
    assert store.cascades[:1] == [
        {
            "target": "document",
            "collection": "kb",
            "doc_id": "d-a",
            "user_id": 1,
            "is_admin": False,
            "preview_only": False,
            "confirm": True,
        }
    ]
    assert {r.file_id for r in db.query(UploadedFile).all()} == {b.file_id, c.file_id}
    db.close()


def test_reconcile_counts_a_store_outage_after_cached_statuses(
    reconcile_env, monkeypatch: pytest.MonkeyPatch
):
    _docs_table, session_local, uploads_dir = reconcile_env
    monkeypatch.setattr(
        kb_file_service,
        "_file_status_cache",
        kb_file_service._FileStatusCache(ttl_seconds=3600),
    )
    a, b = _old_uploads(session_local, uploads_dir, "a.md", "b.md")
    store = _OtherBackend(
        docs=[
            _doc(a.file_id, "kb", "d-a", user_id=1),
            _doc(b.file_id, "kb", "d-b", user_id=1),
        ],
        statuses={("kb", "d-a"): "failed", ("kb", "d-b"): "failed"},
    )
    facade = _facade(store)
    facade.aggregate_uploaded_file_statuses(
        file_ids=[a.file_id, b.file_id], user_id=1, is_admin=False
    )
    store.down = True
    db = session_local()

    result = facade.reconcile_uploaded_files(
        db, user_id=1, is_admin=False, stale_ttl_hours=24
    )

    assert result == dict(scanned=2, stale_candidates=2, deleted=0, cleanup_errors=2)
    assert db.query(UploadedFile).count() == 2
    db.close()


def test_reconcile_reads_no_store_without_uploaded_files(reconcile_env):
    _docs_table, session_local, _uploads_dir = reconcile_env
    db = session_local()

    facade = KBCoordinator(storage_shim=_AMBIENT).file_compatibility
    result = facade.reconcile_uploaded_files(db, user_id=1, is_admin=False)

    assert result == dict(scanned=0, stale_candidates=0, deleted=0, cleanup_errors=0)
    db.close()
