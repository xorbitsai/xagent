"""A raised ingest is rolled back under its real document identity (#2662)."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from tests.web.api import test_kb_cloud_rollback_contract as cloud
from tests.web.api import test_kb_dir as kb_dir
from tests.web.api import test_kb_local_rollback_contract as local
from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.kb import KBApiCompatibilityFacade
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    update_collection_stats_sync,
)
from xagent.web.api import kb as kb_module
from xagent.web.api.kb import CollectionConfigSaveError
from xagent.web.models.uploaded_file import UploadedFile

test_env = kb_dir.test_env
temp_uploads = kb_dir.temp_uploads

_list_refs = kb_module._list_document_refs_for_uploaded_file
_get_collection = kb_module.get_collection_sync
_INCOMPLETE = "Failed to fully roll back"


def _register(**kwargs: Any) -> dict[str, Any]:
    return register_document(
        collection=kwargs["collection"],
        source_path=kwargs["source_path"],
        user_id=kwargs["user_id"],
        file_id=kwargs["file_id"],
    )


def _register_then_raise(seen: dict[str, Any]) -> Any:
    def _ingest(**kwargs: Any) -> IngestionResult:
        seen.update(file_id=kwargs["file_id"], **_register(**kwargs))
        raise RuntimeError("binding write failed")

    return _ingest


def _raise_before_registration(seen: dict[str, Any]) -> Any:
    def _ingest(**kwargs: Any) -> IngestionResult:
        seen["file_id"] = kwargs["file_id"]
        raise RuntimeError("before registration")

    return _ingest


def _register_and_succeed(seen: dict[str, Any]) -> Any:
    def _ingest(**kwargs: Any) -> IngestionResult:
        registered = _register(**kwargs)
        seen.update(file_id=kwargs["file_id"], **registered)
        return IngestionResult(
            status="success",
            doc_id=registered["doc_id"],
            completed_steps=[
                {"name": "register_document", "metadata": registered},
            ],
            message="ok",
        )

    return _ingest


def _file_ids(test_env: Any) -> list[str]:
    session = test_env[3]()
    try:
        return [str(row.file_id) for row in session.query(UploadedFile).all()]
    finally:
        session.close()


def _refs_down(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("refs down")


def _refs_down_once() -> Any:
    calls: list[str] = []

    def _lookup(file_id: str) -> Any:
        calls.append(file_id)
        if len(calls) == 1:
            raise RuntimeError("refs down")
        return _list_refs(file_id)

    return _lookup


def _collection_exists(name: str) -> bool:
    try:
        _get_collection(name)
    except ValueError:
        return False
    return True


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _post_bytes(test_env: Any, content: bytes, *patches: Any) -> Any:
    app, headers, _user, _ = test_env
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return TestClient(app).post(
            "/api/kb/ingest",
            files={"file": ("x.txt", content, "text/plain")},
            data={"collection": "coll"},
            headers=headers,
        )


def _assert_rolled_back(response: Any) -> None:
    assert response.status_code == 500
    assert not response.json()["detail"].startswith(_INCOMPLETE)


def test_ingest_raise_after_registration_removes_the_new_document(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise(seen),
        ),
    )

    _assert_rolled_back(response)
    assert seen["created"] is True
    assert _list_refs(seen["file_id"]) == []
    assert _file_ids(test_env) == []


def test_ingest_raise_keeps_a_document_that_existed_before(
    test_env, temp_uploads
) -> None:
    first: dict[str, Any] = {}
    local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_and_succeed(first),
        ),
    )
    second: dict[str, Any] = {}

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise(second),
        ),
    )

    _assert_rolled_back(response)
    assert second["file_id"] == first["file_id"]
    assert _list_refs(first["file_id"]) == [("coll", first["doc_id"])]
    assert _file_ids(test_env) == [first["file_id"]]


def test_ingest_identity_checks_run_off_the_event_loop(test_env, temp_uploads) -> None:
    local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_and_succeed({}),
        ),
    )
    on_loop: list[bool] = []
    real_check = kb_module._file_document_registered

    def _check(*args: Any) -> bool:
        on_loop.append(_on_event_loop())
        return real_check(*args)

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise({}),
        ),
        patch("xagent.web.api.kb._file_document_registered", side_effect=_check),
    )

    _assert_rolled_back(response)
    assert on_loop == [False, False]


def test_ingest_raise_keeps_the_document_when_the_pre_check_fails(
    test_env, temp_uploads
) -> None:
    first: dict[str, Any] = {}
    local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_and_succeed(first),
        ),
    )

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise({}),
        ),
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            side_effect=_refs_down,
        ),
    )

    _assert_rolled_back(response)
    assert _list_refs(first["file_id"]) == [("coll", first["doc_id"])]
    assert _file_ids(test_env) == [first["file_id"]]


def test_ingest_raise_removes_the_document_when_the_post_check_fails(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise(seen),
        ),
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            side_effect=_refs_down,
        ),
    )

    _assert_rolled_back(response)
    assert _list_refs(seen["file_id"]) == []
    assert _file_ids(test_env) == []


def test_ingest_raise_before_registration_keeps_the_row_when_the_post_check_fails(
    test_env, temp_uploads, caplog
) -> None:
    seen: dict[str, Any] = {}

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_raise_before_registration(seen),
        ),
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            side_effect=_refs_down,
        ),
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail.startswith(f"{_INCOMPLETE} ingest for coll/x.txt: delete document ")
    assert detail.endswith(
        "Original ingestion error: Ingestion setup failed before completion."
    )
    assert _file_ids(test_env) == [seen["file_id"]]
    assert "Could not check whether document" in caplog.text


def test_ingest_raise_into_a_new_collection_compares_real_doc_ids(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}
    may_delete = AsyncMock(return_value=False)

    response = local._post_ingest(
        test_env,
        "x.txt",
        "fresh",
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("new")),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise(seen),
        ),
        patch.object(
            KBApiCompatibilityFacade, "failed_ingest_may_delete_collection", may_delete
        ),
    )

    _assert_rolled_back(response)
    decision = may_delete.await_args.kwargs
    assert (decision["register_created"], decision["doc_id"]) == (True, seen["doc_id"])
    assert [r.doc_id for r in decision["collection_records"]] == [seen["doc_id"]]
    assert _list_refs(seen["file_id"]) == []


def test_ingest_raise_into_a_new_collection_deletes_it_whole(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}

    def _ingest(**kwargs: Any) -> IngestionResult:
        update_collection_stats_sync(collection_name=kwargs["collection"])
        seen["collection_created"] = _collection_exists(kwargs["collection"])
        return _register_then_raise(seen)(**kwargs)

    response = local._post_ingest(
        test_env,
        "x.txt",
        "fresh",
        patch("xagent.web.api.kb.run_document_ingestion", side_effect=_ingest),
    )

    _assert_rolled_back(response)
    assert seen["collection_created"] is True
    assert not _collection_exists("fresh")
    assert not (temp_uploads / f"user_{test_env[2].id}" / "fresh").exists()
    assert _list_refs(seen["file_id"]) == []
    assert _file_ids(test_env) == []


def test_ingest_raise_keeps_the_only_document_of_a_configless_collection(
    test_env, temp_uploads
) -> None:
    first: dict[str, Any] = {}
    _post_bytes(
        test_env,
        b"old content",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_and_succeed(first),
        ),
        patch(
            "xagent.web.api.kb._save_collection_config_after_ingest",
            AsyncMock(side_effect=CollectionConfigSaveError("config write failed")),
        ),
    )

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("read")),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise({}),
        ),
    )

    _assert_rolled_back(response)
    assert _list_refs(first["file_id"]) == [("coll", first["doc_id"])]
    assert _file_ids(test_env) == [first["file_id"]]
    stored = temp_uploads / f"user_{test_env[2].id}" / "coll" / "x.txt"
    assert stored.read_bytes() == b"old content"


def test_ingest_raise_before_registration_may_still_delete_a_new_collection(
    test_env, temp_uploads
) -> None:
    may_delete = AsyncMock(return_value=False)

    response = local._post_ingest(
        test_env,
        "x.txt",
        "fresh",
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("new")),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=RuntimeError("before registration"),
        ),
        patch.object(
            KBApiCompatibilityFacade, "failed_ingest_may_delete_collection", may_delete
        ),
    )

    _assert_rolled_back(response)
    assert may_delete.await_args.kwargs["collection_records"] == []


def test_ingest_cloud_raise_after_registration_removes_the_new_document(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}

    response = cloud._post_cloud(
        test_env,
        _register_then_raise(seen),
        patch("xagent.web.api.kb.get_collection_sync", return_value=object()),
    )

    assert response.status_code == 200
    entry = response.json()[0]
    assert entry["status"] == "error"
    assert entry["doc_id"] == "cloud.csv"
    assert entry["message"] == "Ingestion failed: binding write failed"
    assert _list_refs(seen["file_id"]) == []
    assert _file_ids(test_env) == []


def test_ingest_cloud_raise_keeps_a_document_that_existed_before(
    test_env, temp_uploads
) -> None:
    existing = patch("xagent.web.api.kb.get_collection_sync", return_value=object())
    first: dict[str, Any] = {}
    cloud._post_cloud(test_env, _register_and_succeed(first), existing)

    response = cloud._post_cloud(test_env, _register_then_raise({}), existing)

    entry = response.json()[0]
    assert entry["status"] == "error"
    assert entry["message"] == "Ingestion failed: binding write failed"
    assert _list_refs(first["file_id"]) == [("cloud_coll", first["doc_id"])]
    assert _file_ids(test_env) == [first["file_id"]]


def test_ingest_cloud_raise_keeps_the_document_when_the_pre_check_fails(
    test_env, temp_uploads
) -> None:
    existing = patch("xagent.web.api.kb.get_collection_sync", return_value=object())
    first: dict[str, Any] = {}
    cloud._post_cloud(test_env, _register_and_succeed(first), existing)

    response = cloud._post_cloud(
        test_env,
        _register_then_raise({}),
        existing,
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            side_effect=_refs_down_once(),
        ),
    )

    entry = response.json()[0]
    assert entry["status"] == "error"
    assert entry["message"] == "Ingestion failed: binding write failed"
    assert _list_refs(first["file_id"]) == [("cloud_coll", first["doc_id"])]
    assert _file_ids(test_env) == [first["file_id"]]


def test_ingest_cloud_raise_before_registration_keeps_the_row_when_the_post_check_fails(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}

    response = cloud._post_cloud(
        test_env,
        _raise_before_registration(seen),
        patch("xagent.web.api.kb.get_collection_sync", return_value=object()),
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            side_effect=_refs_down,
        ),
    )

    assert response.status_code == 200
    entry = response.json()[0]
    assert entry["status"] == "error"
    assert entry["doc_id"] == "cloud.csv"
    assert entry["message"].startswith(
        f"{_INCOMPLETE} cloud ingest for cloud_coll/cloud__30b487d9d8d6.csv: "
        "delete document "
    )
    assert entry["message"].endswith(
        "Original ingestion error: Ingestion failed: before registration"
    )
    assert _file_ids(test_env) == [seen["file_id"]]
