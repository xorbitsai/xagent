"""Orphan checks count references past the scan cap and from any owner (#2662)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.web.api import test_kb_dir as kb_dir
from xagent.core.tools.core.RAG_tools.core.config import DEFAULT_VECTOR_STORE_SCAN_LIMIT
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionOperationResult,
    IngestionResult,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env
from xagent.web.api import kb as kb_module
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import kb_file_service

test_env = kb_dir.test_env
temp_uploads = kb_dir.temp_uploads

OTHER_TENANT = 424242


def _documents() -> Any:
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    return conn.open_table("documents")


def _doc(
    collection: str, doc_id: str, file_id: str, user_id: int | None
) -> dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "file_id": file_id,
        "user_id": user_id,
        "source_path": f"/src/{doc_id}",
    }


def _seed(filler_owner: int, *tail: dict[str, Any]) -> None:
    table = _documents()
    table.add(
        [
            _doc("filler", f"filler-{i}", f"filler-{i}", filler_owner)
            for i in range(DEFAULT_VECTOR_STORE_SCAN_LIMIT)
        ]
    )
    table.add(list(tail))


def _assert_capped_scan_misses(file_id: str, *, user_id: int, is_admin: bool) -> None:
    records = LanceDBVectorIndexStore().list_document_records(
        collection_name=None, user_id=user_id, is_admin=is_admin
    )
    assert file_id not in {record.file_id for record in records}


def _uploaded(sessions: Any, user_id: int, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("content")
    session = sessions()
    try:
        record = UploadedFile(
            user_id=user_id,
            filename=path.name,
            storage_path=str(path),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(record)
        session.commit()
        return str(record.file_id)
    finally:
        session.close()


def _row_exists(sessions: Any, file_id: str) -> bool:
    session = sessions()
    try:
        return (
            session.query(UploadedFile).filter_by(file_id=file_id).first() is not None
        )
    finally:
        session.close()


def _fake_delete_document(collection, doc_id, user_id, is_admin):
    _documents().delete(f"doc_id = '{doc_id}'")
    return kb_dir._successful_delete_result(collection, doc_id)


def _fake_delete_collection(collection, user_id, is_admin):
    _documents().delete(f"collection = '{collection}'")
    return CollectionOperationResult(
        status="success", collection=collection, message="deleted"
    )


def _delete_via_api(app: Any, headers: Any, url: str) -> Any:
    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.delete_document", side_effect=_fake_delete_document),
        patch(
            "xagent.web.api.kb.delete_collection", side_effect=_fake_delete_collection
        ),
    ):
        return TestClient(app).delete(url, headers=headers)


def _stub_rollback_leaves(monkeypatch: pytest.MonkeyPatch, *, may_delete: bool):
    for name, fake in {
        "delete_document": _fake_delete_document,
        "delete_collection": _fake_delete_collection,
        "_rollback_may_delete_collection": AsyncMock(return_value=may_delete),
        "_cleanup_failed_new_collection_metadata": AsyncMock(),
        "_restore_ingest_file_backup": lambda **_kw: None,
    }.items():
        monkeypatch.setattr(kb_module, name, fake)


async def _rollback(
    rollback: Any,
    sessions: Any,
    user: User,
    *,
    file_id: str,
    path: Path,
    collection: str,
    collection_existed_before: bool,
) -> None:
    db = sessions()
    try:
        await rollback(
            db=db,
            user=db.get(User, user.id),
            collection_name=collection,
            result=IngestionResult(
                status="partial",
                doc_id=f"doc-{collection}",
                completed_steps=[
                    {"name": "register_document", "metadata": {"created": True}}
                ],
                message="boom",
            ),
            file_path=path,
            file_record=db.query(UploadedFile).filter_by(file_id=file_id).one(),
            collection_existed_before=collection_existed_before,
            uploaded_file_existed_before=False,
            file_backup_path=None,
            had_existing_file=True,
        )
    finally:
        db.close()


def _break_reference_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args, **_kwargs):
        raise RuntimeError("refs down")

    monkeypatch.setattr(kb_file_service, "query_to_list", _raise)


def test_document_delete_keeps_file_referenced_past_scan_cap(test_env, temp_uploads):
    app, headers, user, sessions = test_env
    path = temp_uploads / f"user_{user.id}" / "demo" / "shared.txt"
    file_id = _uploaded(sessions, user.id, path)
    _seed(
        user.id,
        _doc("demo", "doc-demo", file_id, user.id),
        _doc("other", "doc-other", file_id, user.id),
    )
    _assert_capped_scan_misses(file_id, user_id=user.id, is_admin=False)

    response = _delete_via_api(
        app, headers, f"/api/kb/collections/demo/documents/shared.txt?file_id={file_id}"
    )

    assert response.status_code == 200
    assert response.json()["deleted_doc_ids"] == ["doc-demo"]
    assert _row_exists(sessions, file_id)
    assert path.exists()


@pytest.mark.parametrize(
    "rollback",
    [kb_module._rollback_failed_ingestion, kb_module._rollback_failed_cloud_ingestion],
    ids=["local", "cloud"],
)
async def test_file_rollback_keeps_file_referenced_past_scan_cap(
    test_env, temp_uploads, monkeypatch, rollback
):
    _app, _headers, user, sessions = test_env
    path = temp_uploads / f"user_{user.id}" / "demo" / "shared.txt"
    file_id = _uploaded(sessions, user.id, path)
    _seed(
        user.id,
        _doc("demo", "doc-demo", file_id, user.id),
        _doc("other", "doc-other", file_id, user.id),
    )
    _assert_capped_scan_misses(file_id, user_id=user.id, is_admin=False)
    _stub_rollback_leaves(monkeypatch, may_delete=False)

    await _rollback(
        rollback,
        sessions,
        user,
        file_id=file_id,
        path=path,
        collection="demo",
        collection_existed_before=True,
    )

    assert _row_exists(sessions, file_id)
    assert path.exists()


def test_collection_delete_keeps_file_referenced_past_scan_cap(test_env, temp_uploads):
    app, headers, user, sessions = test_env
    path = temp_uploads / f"user_{user.id}" / "shared" / "shared.txt"
    file_id = _uploaded(sessions, user.id, path)
    _seed(
        user.id,
        _doc("team", "doc-team", file_id, user.id),
        _doc("other", "doc-other", file_id, user.id),
    )
    _assert_capped_scan_misses(file_id, user_id=user.id, is_admin=False)

    response = _delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert _row_exists(sessions, file_id)
    assert path.exists()


def test_admin_collection_delete_keeps_each_owner_file_referenced_past_scan_cap(
    test_env, temp_uploads
):
    app, headers, user, sessions = test_env
    owners = [OTHER_TENANT + 1, OTHER_TENANT + 2]
    session = sessions()
    try:
        session.add_all(
            User(id=owner, username=f"owner-{owner}", password_hash="x")
            for owner in owners
        )
        session.get(User, user.id).is_admin = True
        session.commit()
    finally:
        session.close()

    def _path(owner: int, name: str) -> Path:
        return temp_uploads / f"user_{owner}" / "shared" / name

    kept, orphaned, tail = {}, {}, []
    for owner in owners:
        kept[owner] = _uploaded(sessions, owner, _path(owner, "kept.txt"))
        orphaned[owner] = _uploaded(sessions, owner, _path(owner, "orphan.txt"))
        tail += [
            _doc("team", f"team-kept-{owner}", kept[owner], owner),
            _doc("team", f"team-orphan-{owner}", orphaned[owner], owner),
            _doc("other", f"other-{owner}", kept[owner], owner),
        ]
    _seed(OTHER_TENANT, *tail)
    for file_id in kept.values():
        _assert_capped_scan_misses(file_id, user_id=user.id, is_admin=True)

    response = _delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    for owner in owners:
        assert _row_exists(sessions, kept[owner])
        assert _path(owner, "kept.txt").exists()
        assert not _row_exists(sessions, orphaned[owner])
        assert not _path(owner, "orphan.txt").exists()


@pytest.mark.parametrize(
    "other_owner", [OTHER_TENANT, None], ids=["other-tenant", "unowned"]
)
def test_document_delete_keeps_file_another_owner_references(
    test_env, temp_uploads, other_owner
):
    app, headers, user, sessions = test_env
    path = temp_uploads / f"user_{user.id}" / "demo" / "shared.txt"
    file_id = _uploaded(sessions, user.id, path)
    _documents().add(
        [
            _doc("demo", "doc-demo", file_id, user.id),
            _doc("theirs", "doc-theirs", file_id, other_owner),
        ]
    )

    response = _delete_via_api(
        app, headers, f"/api/kb/collections/demo/documents/shared.txt?file_id={file_id}"
    )

    assert response.status_code == 200
    assert response.json()["deleted_doc_ids"] == ["doc-demo"]
    assert _row_exists(sessions, file_id)
    assert path.exists()


@pytest.mark.parametrize(
    "rollback",
    [kb_module._rollback_failed_ingestion, kb_module._rollback_failed_cloud_ingestion],
    ids=["local", "cloud"],
)
async def test_file_rollback_keeps_file_another_tenant_references(
    test_env, temp_uploads, monkeypatch, rollback
):
    _app, _headers, user, sessions = test_env
    path = temp_uploads / f"user_{user.id}" / "demo" / "shared.txt"
    file_id = _uploaded(sessions, user.id, path)
    _documents().add(
        [
            _doc("demo", "doc-demo", file_id, user.id),
            _doc("theirs", "doc-theirs", file_id, OTHER_TENANT),
        ]
    )
    _stub_rollback_leaves(monkeypatch, may_delete=False)

    await _rollback(
        rollback,
        sessions,
        user,
        file_id=file_id,
        path=path,
        collection="demo",
        collection_existed_before=True,
    )

    assert _row_exists(sessions, file_id)
    assert path.exists()


def test_collection_delete_keeps_file_another_tenant_references(test_env, temp_uploads):
    app, headers, user, sessions = test_env
    path = temp_uploads / f"user_{user.id}" / "shared" / "shared.txt"
    file_id = _uploaded(sessions, user.id, path)
    _documents().add(
        [
            _doc("team", "doc-team", file_id, user.id),
            _doc("theirs", "doc-theirs", file_id, OTHER_TENANT),
        ]
    )

    response = _delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert _row_exists(sessions, file_id)
    assert path.exists()


def test_admin_collection_delete_counts_references_from_any_owner(
    test_env, temp_uploads
):
    app, headers, user, sessions = test_env
    session = sessions()
    try:
        session.add(User(id=OTHER_TENANT, username="owner", password_hash="x"))
        session.get(User, user.id).is_admin = True
        session.commit()
    finally:
        session.close()
    kept_path = temp_uploads / f"user_{OTHER_TENANT}" / "shared" / "kept.txt"
    orphan_path = temp_uploads / f"user_{OTHER_TENANT}" / "shared" / "orphan.txt"
    kept = _uploaded(sessions, OTHER_TENANT, kept_path)
    orphan = _uploaded(sessions, OTHER_TENANT, orphan_path)
    _documents().add(
        [
            _doc("team", "team-kept", kept, OTHER_TENANT),
            _doc("team", "team-orphan", orphan, OTHER_TENANT),
            _doc("mine", "admin-copy", kept, user.id),
        ]
    )

    response = _delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert _row_exists(sessions, kept)
    assert kept_path.exists()
    assert not _row_exists(sessions, orphan)
    assert not orphan_path.exists()


def test_deleting_own_document_never_deletes_another_owners_file(
    test_env, temp_uploads
):
    app, headers, user, sessions = test_env
    session = sessions()
    try:
        session.add(User(id=OTHER_TENANT, username="owner", password_hash="x"))
        session.commit()
    finally:
        session.close()
    path = temp_uploads / f"user_{OTHER_TENANT}" / "uploads" / "theirs.txt"
    file_id = _uploaded(sessions, OTHER_TENANT, path)
    _documents().add([_doc("demo", "doc-demo", file_id, user.id)])

    response = _delete_via_api(
        app, headers, f"/api/kb/collections/demo/documents/theirs.txt?file_id={file_id}"
    )

    assert response.status_code == 200
    assert response.json()["deleted_doc_ids"] == ["doc-demo"]
    assert _row_exists(sessions, file_id)
    assert path.exists()


def test_collection_delete_still_removes_referenced_file_in_its_directory(
    test_env, temp_uploads
):
    app, headers, user, sessions = test_env
    # Resolved: the directory pass compares against the resolved collection path.
    path = temp_uploads.resolve() / f"user_{user.id}" / "team" / "shared.txt"
    file_id = _uploaded(sessions, user.id, path)
    _documents().add(
        [
            _doc("team", "doc-team", file_id, user.id),
            _doc("theirs", "doc-theirs", file_id, OTHER_TENANT),
        ]
    )

    response = _delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    # Current behavior, not desired: #2662 item 7.
    assert not _row_exists(sessions, file_id)
    assert not path.exists()


def test_reference_lookup_returns_referenced_candidates_of_any_owner(
    test_env, monkeypatch
):
    _app, _headers, user, _sessions = test_env
    monkeypatch.setattr(kb_file_service, "_ORPHAN_LOOKUP_BATCH_SIZE", 2)
    table = _documents()
    table.add(
        [
            _doc("bulk", f"bulk-{i}", "f-1", user.id)
            for i in range(DEFAULT_VECTOR_STORE_SCAN_LIMIT)
        ]
    )
    table.add(
        [
            _doc("demo", "d-2", "f-2", OTHER_TENANT),
            _doc("demo", "d-3", "f-3", None),
            _doc("demo", "d-x", "f-x", user.id),
        ]
    )

    assert kb_module._find_referenced_file_ids(["f-1", "f-2", "f-3", "f-4"]) == {
        "f-1",
        "f-2",
        "f-3",
    }


def test_document_delete_skips_cleanup_when_reference_lookup_fails(
    test_env, temp_uploads, monkeypatch
):
    app, headers, user, sessions = test_env
    path = temp_uploads / f"user_{user.id}" / "demo" / "shared.txt"
    file_id = _uploaded(sessions, user.id, path)
    _documents().add([_doc("demo", "doc-demo", file_id, user.id)])
    _break_reference_lookup(monkeypatch)

    response = _delete_via_api(
        app, headers, f"/api/kb/collections/demo/documents/shared.txt?file_id={file_id}"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["deleted_doc_ids"] == ["doc-demo"]
    assert _row_exists(sessions, file_id)
    assert path.exists()


async def test_rollback_fails_closed_when_reference_lookup_fails(
    test_env, temp_uploads, monkeypatch
):
    _app, _headers, user, sessions = test_env
    path = temp_uploads / f"user_{user.id}" / "demo" / "shared.txt"
    file_id = _uploaded(sessions, user.id, path)
    _documents().add([_doc("demo", "doc-demo", file_id, user.id)])
    _stub_rollback_leaves(monkeypatch, may_delete=False)
    _break_reference_lookup(monkeypatch)

    with pytest.raises(kb_module.RollbackFailureError) as info:
        await _rollback(
            kb_module._rollback_failed_ingestion,
            sessions,
            user,
            file_id=file_id,
            path=path,
            collection="demo",
            collection_existed_before=True,
        )

    assert str(info.value) == (
        "Failed to fully roll back ingest for demo/shared.txt: refs down. "
        "Original ingestion error: boom"
    )
    assert _row_exists(sessions, file_id)
    assert path.exists()


def test_collection_delete_fails_closed_when_reference_lookup_fails(
    test_env, temp_uploads, monkeypatch
):
    app, headers, user, sessions = test_env
    path = temp_uploads / f"user_{user.id}" / "shared" / "shared.txt"
    file_id = _uploaded(sessions, user.id, path)
    _documents().add([_doc("team", "doc-team", file_id, user.id)])
    _break_reference_lookup(monkeypatch)

    response = _delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to delete collection: refs down"
    assert _row_exists(sessions, file_id)
    assert path.exists()
