"""Contract for ``_rollback_failed_ingestion`` and the text it surfaces (#795).

Leaf fakes only record calls; every assertion runs after the wrapper returns or
raises, because an assert inside a fake would be swallowed by the rollback's
own ``except Exception``.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.web.api import test_kb_dir as kb_dir
from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
from xagent.core.tools.core.RAG_tools.utils.string_utils import (
    generate_deterministic_doc_id,
)
from xagent.web.api import kb as kb_module
from xagent.web.api.kb import RollbackFailureError

test_env = kb_dir.test_env
temp_uploads = kb_dir.temp_uploads

REGISTERED = [{"name": "register_document", "metadata": {"created": True}}]
WHOLE = [
    "list:coll",
    "may_delete",
    "delete_collection",
    "physdir",
    "refs:[]",
    "del_coll_files",
    "query",
    "store.delete:refreshed",
    "metadata",
    "commit",
    "restore",
]
KEPT = [
    "list:coll",
    "may_delete",
    "delete_document",
    "refs:['file-1']",
    "orphan",
    "commit",
    "restore",
]
SHAPES = [
    pytest.param(True, WHOLE, id="whole-collection"),
    pytest.param(False, KEPT, id="registered-kept"),
]
PREFIX = "Failed to fully roll back ingest for coll/failed.txt"
ORIGINAL = "Original ingestion error: partial failure"


def _result(
    *,
    doc_id: Optional[str] = "doc-1",
    steps: Optional[list[dict[str, Any]]] = None,
    message: str = "partial failure",
) -> IngestionResult:
    return IngestionResult(
        status="partial",
        doc_id=doc_id,
        completed_steps=REGISTERED if steps is None else steps,
        message=message,
    )


def _install_leaves(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    *,
    records: tuple[Any, ...] | list[Any] = (),
    may_delete: bool = False,
    refreshed: bool = True,
    physdir: tuple[str, Optional[str]] = ("success", None),
    collection_status: str = "success",
    document_status: str = "success",
    raises: Optional[dict[str, Exception]] = None,
) -> tuple[Any, dict[str, Any]]:
    failures = raises or {}
    seen: dict[str, Any] = {}

    def _hit(name: str) -> None:
        calls.append(name)
        if name in failures:
            raise failures[name]

    class _Store:
        def list_document_records(self, *, collection_name, user_id, is_admin):
            _hit(f"list:{collection_name}")
            return list(records) if collection_name else []

    async def _may_delete(**kwargs):
        seen["may_delete"] = kwargs
        _hit("may_delete")
        return may_delete

    def _delete_collection(collection, user_id, is_admin):
        _hit("delete_collection")
        return SimpleNamespace(status=collection_status, message="boom")

    def _physdir(*, user_id, collection_name):
        _hit("physdir")
        return SimpleNamespace(
            status=physdir[0], error=physdir[1], collection_dir=Path("/uploads/coll")
        )

    def _del_coll_files(
        db, *, user_id, collection_file_ids, remaining_file_ids, collection_dir
    ):
        seen["collection_file_ids"] = collection_file_ids
        seen["remaining_file_ids"] = remaining_file_ids
        seen["collection_dir"] = collection_dir
        _hit("del_coll_files")

    class _FileStore:
        def __init__(self, db):
            pass

        def delete(self, record, *, delete_local):
            _hit(f"store.delete:{record.tag}")

    async def _metadata(*, collection_name, user):
        _hit("metadata")

    def _delete_document(collection, doc_id, user_id, is_admin):
        _hit("delete_document")
        return SimpleNamespace(status=document_status, message="boom")

    def _orphan(db, *, file_id, user_id, remaining_file_ids):
        _hit("orphan")

    def _refs(file_ids):
        _hit(f"refs:{sorted(file_ids)}")
        seen["refs_answer"] = set(file_ids)
        return seen["refs_answer"]

    def _clear_status(collection, doc_id, *, user_id, is_admin):
        _hit("clear_status")

    def _restore(**_kwargs):
        _hit("restore")

    fakes = {
        "get_vector_index_store": _Store,
        "_rollback_may_delete_collection": _may_delete,
        "delete_collection": _delete_collection,
        "delete_collection_physical_dir": _physdir,
        "delete_collection_uploaded_files": _del_coll_files,
        "UploadedFileStore": _FileStore,
        "_cleanup_failed_new_collection_metadata": _metadata,
        "delete_document": _delete_document,
        "_delete_uploaded_file_if_orphaned": _orphan,
        "_find_referenced_file_ids": _refs,
        "clear_ingestion_status": _clear_status,
        "_restore_ingest_file_backup": _restore,
        "_delete_web_rag_side_effects_for_file_id": lambda **_kw: _hit("web_cleanup"),
        "_restore_rag_document_snapshot": lambda *_a, **_kw: _hit("web_snapshot"),
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(kb_module, name, fake)

    class _Db:
        def query(self, _model):
            return self

        def filter(self, *_criteria):
            return self

        def first(self):
            _hit("query")
            return SimpleNamespace(tag="refreshed") if refreshed else None

        def commit(self):
            _hit("commit")

        def rollback(self):
            calls.append("rollback")

    return _Db(), seen


async def _rollback(
    db: Any,
    *,
    result: Optional[IngestionResult] = None,
    collection_existed_before: bool = False,
    uploaded_file_existed_before: bool = False,
    embedding_model_id: Optional[str] = None,
) -> None:
    await kb_module._rollback_failed_ingestion(
        db=db,
        user=SimpleNamespace(id=7, is_admin=False),
        collection_name="coll",
        result=result or _result(),
        file_path=Path("/uploads/failed.txt"),
        file_record=SimpleNamespace(file_id="file-1", tag="passed"),
        collection_existed_before=collection_existed_before,
        uploaded_file_existed_before=uploaded_file_existed_before,
        file_backup_path=None,
        had_existing_file=False,
        embedding_model_id=embedding_model_id,
    )


@pytest.mark.parametrize(
    ("leaves", "rollback_kwargs", "expected"),
    [
        pytest.param({"may_delete": True}, {}, WHOLE, id="whole-collection"),
        pytest.param(
            {"may_delete": True, "physdir": ("not_found", None)},
            {},
            WHOLE,
            id="whole-collection-dir-already-gone",
        ),
        pytest.param(
            {"may_delete": True},
            {"uploaded_file_existed_before": True},
            [c for c in WHOLE if c not in {"query", "store.delete:refreshed"}],
            id="whole-collection-existing-upload",
        ),
        pytest.param(
            {"may_delete": True, "refreshed": False},
            {},
            [c for c in WHOLE if c != "store.delete:refreshed"],
            id="whole-collection-row-gone",
        ),
        pytest.param({}, {}, KEPT, id="registered-kept"),
        pytest.param(
            {},
            {"uploaded_file_existed_before": True},
            ["list:coll", "may_delete", "delete_document", "restore"],
            id="registered-kept-existing-upload",
        ),
        pytest.param(
            {},
            {"result": _result(steps=[])},
            ["list:coll", "may_delete", "clear_status", "store.delete:passed"]
            + ["commit", "restore"],
            id="unregistered-new-upload",
        ),
        pytest.param(
            {},
            {"result": _result(steps=[]), "uploaded_file_existed_before": True},
            ["list:coll", "may_delete", "clear_status", "restore"],
            id="unregistered-existing-upload",
        ),
        pytest.param(
            {},
            {"result": _result(doc_id=None)},
            ["list:coll", "may_delete", "store.delete:passed", "commit", "restore"],
            id="no-doc-id",
        ),
        pytest.param(
            {},
            {"result": _result(doc_id=None), "uploaded_file_existed_before": True},
            ["list:coll", "may_delete", "restore"],
            id="nothing-to-compensate",
        ),
        pytest.param(
            {},
            {"collection_existed_before": True},
            KEPT[1:],
            id="collection-existed-before",
        ),
    ],
)
async def test_rollback_runs_leaves_in_order(
    monkeypatch, leaves, rollback_kwargs, expected
) -> None:
    calls: list[str] = []
    db, _ = _install_leaves(monkeypatch, calls, **leaves)

    await _rollback(db, **rollback_kwargs)

    assert calls == expected


@pytest.mark.parametrize(
    ("records", "other_present", "file_ids"),
    [
        pytest.param([], False, set(), id="empty"),
        pytest.param(
            [{"doc_id": "doc-1", "file_id": "f-1"}], False, {"f-1"}, id="own-doc"
        ),
        pytest.param(
            [{"doc_id": "other", "file_id": "file-1"}],
            True,
            {"file-1"},
            id="sibling-sharing-the-file-id",
        ),
        pytest.param(
            [SimpleNamespace(doc_id="other", file_id="f-2")],
            True,
            {"f-2"},
            id="object-record",
        ),
    ],
)
async def test_collection_decision_compares_doc_ids(
    monkeypatch, records, other_present, file_ids
) -> None:
    calls: list[str] = []
    db, seen = _install_leaves(monkeypatch, calls, records=records, may_delete=True)

    await _rollback(db)

    assert seen["may_delete"] == {
        "collection_name": "coll",
        "user_id": 7,
        "collection_existed_before": False,
        "other_document_present": other_present,
        "context": "failed-ingest rollback",
    }
    assert seen["collection_file_ids"] == file_ids
    assert f"refs:{sorted(file_ids)}" in calls


@pytest.mark.parametrize(
    ("existed", "offered"),
    [
        pytest.param(False, {"file-1"}, id="new-row"),
        pytest.param(True, set(), id="pre-existing-row"),
    ],
)
async def test_whole_collection_offers_only_a_row_this_run_created(
    monkeypatch, existed, offered
) -> None:
    calls: list[str] = []
    db, seen = _install_leaves(
        monkeypatch,
        calls,
        records=[{"doc_id": "doc-1", "file_id": "file-1"}],
        may_delete=True,
    )

    await _rollback(db, uploaded_file_existed_before=existed)

    assert seen["collection_file_ids"] == offered
    assert seen["remaining_file_ids"] is seen["refs_answer"]
    assert seen["collection_dir"] is None
    assert f"refs:{sorted(offered)}" in calls


async def test_collection_existed_before_still_asks_the_decision(monkeypatch) -> None:
    calls: list[str] = []
    db, seen = _install_leaves(monkeypatch, calls)

    await _rollback(db, collection_existed_before=True)

    assert seen["may_delete"]["collection_existed_before"] is True
    assert seen["may_delete"]["other_document_present"] is False


@pytest.mark.parametrize(
    ("leaves", "rollback_kwargs", "ran", "detail"),
    [
        pytest.param(
            {"may_delete": True, "collection_status": "error"},
            {},
            WHOLE[:3],
            "delete collection 'coll' during rollback failed: boom",
            id="delete-collection",
        ),
        pytest.param(
            {"may_delete": True, "physdir": ("failed", "trash lock busy")},
            {},
            WHOLE[:4],
            "delete collection physical directory during rollback failed: "
            "trash lock busy",
            id="physical-dir",
        ),
        pytest.param(
            {"may_delete": True, "physdir": ("failed", None)},
            {},
            WHOLE[:4],
            "delete collection physical directory during rollback failed: "
            "unknown physical cleanup failure",
            id="physical-dir-without-detail",
        ),
        pytest.param(
            {"may_delete": True, "raises": {"refs:[]": RuntimeError("list down")}},
            {},
            WHOLE[:5],
            "list down",
            id="remaining-records",
        ),
        pytest.param(
            {
                "may_delete": True,
                "raises": {"del_coll_files": RuntimeError("uploads locked")},
            },
            {},
            WHOLE[:6],
            "uploads locked",
            id="collection-uploads",
        ),
        pytest.param(
            {
                "may_delete": True,
                "raises": {"store.delete:refreshed": RuntimeError("row locked")},
            },
            {},
            WHOLE[:8],
            "row locked",
            id="refreshed-row-delete",
        ),
        pytest.param(
            {"may_delete": True, "raises": {"metadata": RuntimeError("meta down")}},
            {},
            WHOLE[:9],
            "meta down",
            id="metadata-before-commit",
        ),
        pytest.param(
            {"document_status": "error"},
            {},
            KEPT[:3],
            "delete document 'doc-1' during rollback failed: boom",
            id="delete-document",
        ),
        pytest.param(
            {"raises": {"orphan": OSError("disk")}},
            {},
            KEPT[:5],
            "disk",
            id="orphan-before-commit",
        ),
        pytest.param(
            {"raises": {"refs:['file-1']": RuntimeError("refs down")}},
            {},
            KEPT[:4],
            "refs down",
            id="kept-refs",
        ),
        pytest.param(
            {"raises": {"clear_status": RuntimeError("status down")}},
            {"result": _result(steps=[])},
            ["list:coll", "may_delete", "clear_status"],
            "status down",
            id="clear-status",
        ),
        pytest.param(
            {"raises": {"store.delete:passed": RuntimeError("row locked")}},
            {"result": _result(steps=[])},
            ["list:coll", "may_delete", "clear_status", "store.delete:passed"],
            "row locked",
            id="store-delete-before-commit",
        ),
        pytest.param(
            {"raises": {"list:coll": RuntimeError("lance down")}},
            {},
            ["list:coll"],
            "lance down",
            id="records-scan",
        ),
    ],
)
async def test_first_failure_stops_rollback_with_verbatim_text(
    monkeypatch, leaves, rollback_kwargs, ran, detail
) -> None:
    calls: list[str] = []
    db, _ = _install_leaves(monkeypatch, calls, **leaves)

    with pytest.raises(RollbackFailureError) as info:
        await _rollback(db, **rollback_kwargs)

    assert str(info.value) == f"{PREFIX}: {detail}. {ORIGINAL}"
    assert calls == [*ran, "rollback", "restore"]


async def test_failing_exception_is_chained_unwrapped(monkeypatch) -> None:
    calls: list[str] = []
    boom = OSError("disk")
    db, _ = _install_leaves(monkeypatch, calls, raises={"orphan": boom})

    with pytest.raises(RollbackFailureError) as info:
        await _rollback(db)

    assert info.value.__cause__ is boom


@pytest.mark.parametrize(("may_delete", "ran"), SHAPES)
async def test_restore_failure_after_commit_reports_both(
    monkeypatch, may_delete, ran
) -> None:
    calls: list[str] = []
    db, _ = _install_leaves(
        monkeypatch,
        calls,
        may_delete=may_delete,
        raises={"restore": OSError("restore exploded")},
    )

    with pytest.raises(RollbackFailureError) as info:
        await _rollback(db)

    assert str(info.value) == (
        f"{PREFIX}: restore exploded. {ORIGINAL}; "
        "backup restore also failed: restore exploded"
    )
    assert calls == [*ran, "rollback", "restore"]


async def test_vector_store_failure_escapes_before_the_rollback(monkeypatch) -> None:
    calls: list[str] = []
    db, _ = _install_leaves(monkeypatch, calls)
    boom = RuntimeError("no store")

    def _raise() -> None:
        raise boom

    monkeypatch.setattr(kb_module, "get_vector_index_store", _raise)

    with pytest.raises(RuntimeError) as info:
        await _rollback(db)

    assert info.value is boom
    assert calls == []


async def test_rollback_failure_text_carries_embedding_hint(monkeypatch) -> None:
    calls: list[str] = []
    db, _ = _install_leaves(monkeypatch, calls, document_status="error")

    with pytest.raises(RollbackFailureError) as info:
        await _rollback(
            db,
            result=_result(message="No embedding model available"),
            embedding_model_id="emb-1",
        )

    assert "Current embedding_model_id: 'emb-1'." in str(info.value)


@pytest.mark.parametrize(("may_delete", "ran"), SHAPES)
async def test_rollback_does_not_yield_to_sibling_coroutines(
    monkeypatch, may_delete, ran
) -> None:
    calls: list[str] = []
    db, _ = _install_leaves(monkeypatch, calls, may_delete=may_delete)

    async def _sibling() -> None:
        calls.append("sibling")

    sibling = asyncio.create_task(_sibling())
    await _rollback(db)
    await sibling

    assert calls == [*ran, "sibling"]


def test_wrapper_stays_a_coroutine_function() -> None:
    assert inspect.iscoroutinefunction(kb_module._rollback_failed_ingestion)


# --- Text surfaced by /ingest ---


def _post_ingest(test_env, filename: str, collection: str, *patches: Any) -> Any:
    app, headers, _user, _ = test_env
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return TestClient(app).post(
            "/api/kb/ingest",
            files={"file": (filename, b"new content", "text/plain")},
            data={"collection": collection},
            headers=headers,
        )


def _existing_collection() -> Any:
    return patch("xagent.web.api.kb.get_collection_sync", return_value=object())


def test_ingest_returns_document_rollback_failure_verbatim(
    test_env, temp_uploads
) -> None:
    response = _post_ingest(
        test_env,
        "failed.txt",
        "existing_collection",
        _existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            return_value=_result(message="embedding failed"),
        ),
        patch(
            "xagent.web.api.kb.delete_document",
            return_value=SimpleNamespace(status="error", message="boom"),
        ),
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to fully roll back ingest for "
        "existing_collection/failed.txt: delete document 'doc-1' during rollback "
        "failed: boom. Original ingestion error: embedding failed"
    }


def test_ingest_returns_collection_rollback_failure_verbatim(
    test_env, temp_uploads
) -> None:
    response = _post_ingest(
        test_env,
        "failed.txt",
        "new_collection",
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            return_value=_result(message="ingest failed"),
        ),
        patch(
            "xagent.web.api.kb.delete_collection",
            return_value=SimpleNamespace(status="error", message="boom"),
        ),
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to fully roll back ingest for new_collection/failed.txt: "
        "delete collection 'new_collection' during rollback failed: boom. "
        "Original ingestion error: ingest failed"
    }


def test_ingest_setup_failure_clears_status_by_real_doc_id(
    test_env, temp_uploads
) -> None:
    _, _, user, _ = test_env
    clear_status = MagicMock()
    seen: dict[str, Any] = {}

    def _raise(**kwargs: Any) -> None:
        seen["file_id"] = kwargs["file_id"]
        raise RuntimeError("parser crashed")

    response = _post_ingest(
        test_env,
        "x.txt",
        "coll",
        _existing_collection(),
        patch("xagent.web.api.kb.run_document_ingestion", side_effect=_raise),
        patch("xagent.web.api.kb.clear_ingestion_status", clear_status),
    )

    assert response.status_code == 500
    assert not response.json()["detail"].startswith("Failed to fully roll back")
    clear_status.assert_called_once_with(
        "coll",
        generate_deterministic_doc_id("coll", seen["file_id"]),
        user_id=int(user.id),
        is_admin=False,
    )


def test_ingest_setup_failure_returns_rollback_failure_verbatim(
    test_env, temp_uploads
) -> None:
    response = _post_ingest(
        test_env,
        "x.txt",
        "coll",
        _existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=RuntimeError("parser crashed"),
        ),
        patch(
            "xagent.web.api.kb.clear_ingestion_status",
            side_effect=RuntimeError("status down"),
        ),
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to fully roll back ingest for coll/x.txt: status down. "
        "Original ingestion error: Ingestion setup failed before completion."
    }


def test_ingest_keeps_failed_result_after_clean_rollback(
    test_env, temp_uploads
) -> None:
    ingested = _result(message="embedding failed")

    response = _post_ingest(
        test_env,
        "failed.txt",
        "existing_collection",
        _existing_collection(),
        patch("xagent.web.api.kb.run_document_ingestion", return_value=ingested),
        patch(
            "xagent.web.api.kb.delete_document",
            side_effect=kb_dir._make_delete_tracker()[1],
        ),
    )

    assert response.status_code == 500
    assert response.json() == {**ingested.model_dump(mode="json"), "status": "error"}
