"""Contract for ``_rollback_failed_cloud_ingestion`` and the text it surfaces (#795).

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
from xagent.core.tools.core.RAG_tools.core.schemas import (
    IngestionConfig,
    IngestionResult,
)
from xagent.core.tools.core.RAG_tools.utils.string_utils import (
    generate_deterministic_doc_id,
)
from xagent.web.api import kb as kb_module
from xagent.web.api.kb import RollbackFailureError
from xagent.web.jobs.exceptions import BackgroundJobHandlerError
from xagent.web.jobs.kb_tasks import handle_kb_ingest_document
from xagent.web.models.background_job import BackgroundJobType
from xagent.web.services.background_jobs import create_background_job

test_env = kb_dir.test_env
temp_uploads = kb_dir.temp_uploads

REGISTERED = [{"name": "register_document", "metadata": {"created": True}}]
FULL_CHAIN = [
    "delete_document",
    "refs:['file-1']",
    "orphan",
    "list:coll",
    "may_delete",
    "delete_collection",
    "metadata_cleanup",
    "commit",
    "restore",
]
PREFIX = "Failed to fully roll back cloud ingest for coll/cloud__abc.csv"


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
    may_delete: bool = True,
    document_status: str = "success",
    collection_error: Optional[Exception] = None,
    restore_error: Optional[Exception] = None,
    refs_error: Optional[Exception] = None,
) -> Any:
    class _Store:
        def list_document_records(self, *, collection_name, user_id, is_admin, **_kw):
            calls.append(f"list:{collection_name}")
            return []

    def _delete_document(collection, doc_id, user_id, is_admin):
        calls.append("delete_document")
        return SimpleNamespace(status=document_status, message="boom")

    def _clear_status(collection, doc_id, *, user_id, is_admin):
        calls.append("clear_status")

    def _orphan(db, *, file_id, user_id, remaining_file_ids):
        calls.append("orphan")

    def _refs(file_ids):
        calls.append(f"refs:{sorted(file_ids)}")
        if refs_error is not None:
            raise refs_error
        return set()

    async def _may_delete(**_kwargs):
        calls.append("may_delete")
        return may_delete

    def _delete_collection(collection, user_id, is_admin):
        calls.append("delete_collection")
        if collection_error is not None:
            raise collection_error
        return SimpleNamespace(status="success", message="")

    async def _metadata_cleanup(**_kwargs):
        calls.append("metadata_cleanup")

    def _restore(**_kwargs):
        calls.append("restore")
        if restore_error is not None:
            raise restore_error

    fakes = {
        "get_vector_index_store": _Store,
        "delete_document": _delete_document,
        "clear_ingestion_status": _clear_status,
        "_delete_uploaded_file_if_orphaned": _orphan,
        "_find_referenced_file_ids": _refs,
        "_rollback_may_delete_collection": _may_delete,
        "delete_collection": _delete_collection,
        "_cleanup_failed_new_collection_metadata": _metadata_cleanup,
        "_restore_ingest_file_backup": _restore,
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(kb_module, name, fake)
    return SimpleNamespace(
        commit=lambda: calls.append("commit"),
        rollback=lambda: calls.append("rollback"),
    )


async def _rollback(
    db: Any,
    *,
    result: Optional[IngestionResult] = None,
    with_file_record: bool = True,
    embedding_model_id: Optional[str] = None,
    uploaded_file_existed_before: bool = False,
) -> None:
    await kb_module._rollback_failed_cloud_ingestion(
        db=db,
        user=SimpleNamespace(id=7, is_admin=False),
        collection_name="coll",
        result=result or _result(),
        file_path=Path("/uploads/cloud__abc.csv"),
        file_record=SimpleNamespace(file_id="file-1") if with_file_record else None,
        collection_existed_before=False,
        uploaded_file_existed_before=uploaded_file_existed_before,
        file_backup_path=None,
        had_existing_file=False,
        embedding_model_id=embedding_model_id,
    )


@pytest.mark.parametrize(
    ("result_kwargs", "with_file_record", "may_delete", "expected"),
    [
        pytest.param({}, True, True, FULL_CHAIN, id="registered"),
        pytest.param(
            {"steps": []},
            True,
            True,
            ["clear_status", *FULL_CHAIN[1:]],
            id="unregistered-clears-status-before-file",
        ),
        pytest.param({"doc_id": None}, True, True, FULL_CHAIN[1:], id="no-doc-id"),
        pytest.param(
            {},
            False,
            True,
            [FULL_CHAIN[0], "refs:[]", *FULL_CHAIN[3:]],
            id="no-file-record-still-lists",
        ),
        pytest.param(
            {},
            True,
            False,
            [*FULL_CHAIN[:5], "commit", "restore"],
            id="collection-kept",
        ),
    ],
)
async def test_rollback_runs_leaves_in_order(
    monkeypatch, result_kwargs, with_file_record, may_delete, expected
) -> None:
    calls: list[str] = []
    db = _install_leaves(monkeypatch, calls, may_delete=may_delete)

    await _rollback(
        db, result=_result(**result_kwargs), with_file_record=with_file_record
    )

    assert calls == expected


@pytest.mark.parametrize(
    ("result_kwargs", "expected"),
    [
        pytest.param({}, [FULL_CHAIN[0], *FULL_CHAIN[3:]], id="registered"),
        pytest.param(
            {"steps": []}, ["clear_status", *FULL_CHAIN[3:]], id="unregistered"
        ),
        pytest.param({"doc_id": None}, FULL_CHAIN[3:], id="no-doc-id"),
    ],
)
async def test_pre_existing_row_skips_the_file_step(
    monkeypatch, result_kwargs, expected
) -> None:
    calls: list[str] = []
    db = _install_leaves(monkeypatch, calls)

    await _rollback(
        db, result=_result(**result_kwargs), uploaded_file_existed_before=True
    )

    assert calls == expected


async def test_document_failure_stops_rollback_with_verbatim_text(monkeypatch) -> None:
    calls: list[str] = []
    db = _install_leaves(monkeypatch, calls, document_status="error")

    with pytest.raises(RollbackFailureError) as info:
        await _rollback(db)

    assert str(info.value) == (
        f"{PREFIX}: delete document 'doc-1' during cloud rollback failed: boom. "
        "Original ingestion error: partial failure"
    )
    assert calls == ["delete_document", "rollback", "restore"]
    assert type(info.value.__cause__) is RuntimeError
    assert str(info.value.__cause__) == (
        "delete document 'doc-1' during cloud rollback failed: boom"
    )


async def test_collection_failure_rolls_back_session_and_chains_the_exception(
    monkeypatch,
) -> None:
    calls: list[str] = []
    boom = RuntimeError("lance down")
    db = _install_leaves(monkeypatch, calls, collection_error=boom)

    with pytest.raises(RollbackFailureError) as info:
        await _rollback(db)

    assert info.value.__cause__ is boom
    assert str(info.value) == (
        f"{PREFIX}: lance down. Original ingestion error: partial failure"
    )
    assert calls == [*FULL_CHAIN[:6], "rollback", "restore"]


async def test_refs_failure_stops_before_orphan_and_collection(monkeypatch) -> None:
    calls: list[str] = []
    db = _install_leaves(monkeypatch, calls, refs_error=RuntimeError("refs down"))

    with pytest.raises(RollbackFailureError) as info:
        await _rollback(db)

    assert str(info.value) == (
        f"{PREFIX}: refs down. Original ingestion error: partial failure"
    )
    assert calls == [*FULL_CHAIN[:2], "rollback", "restore"]


async def test_restore_failure_after_commit_reports_both(monkeypatch) -> None:
    calls: list[str] = []
    db = _install_leaves(monkeypatch, calls, restore_error=OSError("restore exploded"))

    with pytest.raises(RollbackFailureError) as info:
        await _rollback(db)

    assert str(info.value) == (
        f"{PREFIX}: restore exploded. Original ingestion error: partial failure; "
        "backup restore also failed: restore exploded"
    )
    assert calls == [*FULL_CHAIN, "rollback", "restore"]


async def test_rollback_failure_text_carries_embedding_hint(monkeypatch) -> None:
    calls: list[str] = []
    db = _install_leaves(monkeypatch, calls, document_status="error")

    with pytest.raises(RollbackFailureError) as info:
        await _rollback(
            db,
            result=_result(message="No embedding model available"),
            embedding_model_id="emb-1",
        )

    assert "Current embedding_model_id: 'emb-1'." in str(info.value)


async def test_rollback_does_not_yield_to_sibling_coroutines(monkeypatch) -> None:
    calls: list[str] = []
    db = _install_leaves(monkeypatch, calls)

    async def _sibling() -> None:
        calls.append("sibling")

    sibling = asyncio.create_task(_sibling())
    await _rollback(db)
    await sibling

    assert calls == [*FULL_CHAIN, "sibling"]


def test_wrapper_stays_a_coroutine_function() -> None:
    assert inspect.iscoroutinefunction(kb_module._rollback_failed_cloud_ingestion)


# --- Text surfaced by /ingest-cloud and the staged document job ---

CLOUD_FILE = {
    "provider": "google-drive",
    "fileId": "drive-file-1",
    "fileName": "cloud.csv",
}


def _post_cloud(test_env, ingest: Any, *extra_patches: Any) -> Any:
    app, headers, _user, _ = test_env

    class _Files:
        def get(self, fileId: str, **_kwargs):
            return kb_dir._fake_drive_metadata_request(fileId, "cloud.csv", "text/csv")

        def get_media(self, fileId: str, **_kwargs):
            return {"fileId": fileId}

    class _Downloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b"cloud-content")
            return None, True

    with ExitStack() as stack:
        for p in (
            patch("xagent.web.api.kb.get_google_credentials", return_value=object()),
            patch(
                "xagent.web.api.kb.build", return_value=SimpleNamespace(files=_Files)
            ),
            patch("xagent.web.api.kb.MediaIoBaseDownload", _Downloader),
            patch("xagent.web.api.kb.run_document_ingestion", side_effect=ingest),
            *extra_patches,
        ):
            stack.enter_context(p)
        return TestClient(app).post(
            "/api/kb/ingest-cloud",
            json={"collection": "cloud_coll", "files": [CLOUD_FILE]},
            headers=headers,
        )


def test_ingest_cloud_returns_rollback_failure_text_verbatim(
    test_env, temp_uploads
) -> None:
    response = _post_cloud(
        test_env,
        lambda **_kw: _result(doc_id="cloud-doc-id"),
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("new")),
        patch(
            "xagent.web.api.kb.delete_document",
            return_value=SimpleNamespace(status="error", message="boom"),
        ),
    )

    assert response.status_code == 200
    entry = response.json()[0]
    assert entry["status"] == "error"
    assert entry["doc_id"] == "cloud.csv"
    assert entry["message"] == (
        "Failed to fully roll back cloud ingest for "
        "cloud_coll/cloud__30b487d9d8d6.csv: delete document 'cloud-doc-id' "
        "during cloud rollback failed: boom. Original ingestion error: "
        "partial failure"
    )


def test_ingest_cloud_raised_ingestion_clears_status_by_real_doc_id(
    test_env, temp_uploads
) -> None:
    _, _, user, _ = test_env
    clear_status = MagicMock()
    seen: dict[str, Any] = {}

    def _raise(**kwargs):
        seen["file_id"] = kwargs["file_id"]
        raise RuntimeError("parser crashed")

    response = _post_cloud(
        test_env,
        _raise,
        patch("xagent.web.api.kb.get_collection_sync", return_value=MagicMock()),
        patch("xagent.web.api.kb.clear_ingestion_status", clear_status),
    )

    assert response.status_code == 200
    assert response.json()[0]["doc_id"] == "cloud.csv"
    assert response.json()[0]["message"] == "Ingestion failed: parser crashed"
    clear_status.assert_called_once_with(
        "cloud_coll",
        generate_deterministic_doc_id("cloud_coll", seen["file_id"]),
        user_id=int(user.id),
        is_admin=False,
    )


def test_ingest_cloud_keeps_partial_result_after_clean_rollback(
    test_env, temp_uploads
) -> None:
    response = _post_cloud(
        test_env,
        lambda **_kw: _result(doc_id="cloud-doc-id"),
        patch("xagent.web.api.kb.get_collection_sync", return_value=MagicMock()),
        patch(
            "xagent.web.api.kb.delete_document",
            side_effect=kb_dir._make_delete_tracker()[1],
        ),
    )

    assert response.status_code == 200
    entry = response.json()[0]
    assert entry["status"] == "partial"
    assert entry["message"] == "partial failure"


def _run_staged_job(test_env, tmp_path: Path) -> BackgroundJobHandlerError:
    _, _, user, session_local = test_env
    staged = tmp_path / "stage" / "doc.txt"
    staged.parent.mkdir(parents=True)
    staged.write_text("staged content", encoding="utf-8")
    db = session_local()
    try:
        job = create_background_job(
            db,
            user_id=int(user.id),
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT,
            payload={
                "collection": "existing-kb",
                "source_path": str(staged),
                "target_path": str(tmp_path / "canonical" / "doc.txt"),
                "file_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "filename": "doc.txt",
                "mime_type": "text/plain",
                "file_size": staged.stat().st_size,
                "user_id": int(user.id),
                "is_admin": False,
                "ingestion_config": IngestionConfig().model_dump(mode="json"),
                "collection_existed_before": True,
            },
        )
        with pytest.raises(BackgroundJobHandlerError) as info:
            handle_kb_ingest_document(db, job)
        return info.value
    finally:
        db.close()


def test_staged_job_returns_rollback_failure_text_verbatim(
    test_env, tmp_path, monkeypatch
) -> None:
    ingested = _result(message="ingestion failed")
    monkeypatch.setattr(
        "xagent.web.jobs.kb_tasks.run_document_ingestion", lambda **_kw: ingested
    )
    monkeypatch.setattr(kb_module, "get_vector_index_store", MagicMock())
    monkeypatch.setattr(
        kb_module,
        "delete_document",
        lambda *_a: SimpleNamespace(status="error", message="boom"),
    )

    err = _run_staged_job(test_env, tmp_path)

    assert str(err) == (
        "Failed to fully roll back cloud ingest for existing-kb/doc.txt: "
        "delete document 'doc-1' during cloud rollback failed: boom. "
        "Original ingestion error: ingestion failed"
    )
    assert err.retryable is False
    assert err.result == ingested.model_dump(mode="json")


def test_staged_job_routes_through_the_patched_wrapper(
    test_env, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "xagent.web.jobs.kb_tasks.run_document_ingestion",
        lambda **_kw: _result(message="ingestion failed"),
    )

    with patch(
        "xagent.web.api.kb._rollback_failed_cloud_ingestion",
        side_effect=RollbackFailureError("x"),
    ):
        err = _run_staged_job(test_env, tmp_path)

    assert str(err) == "x"
