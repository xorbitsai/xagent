"""A failed ingest keeps an UploadedFile row that existed before the run (#2662)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tests.web.api import test_kb_cloud_rollback_contract as cloud
from tests.web.api import test_kb_dir as kb_dir
from tests.web.api import test_kb_local_rollback_contract as local
from tests.web.api import test_kb_raised_ingest_identity as raised
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    update_collection_stats_sync,
)
from xagent.web.api.kb import _document_existed_before_ingest
from xagent.web.models.uploaded_file import UploadedFile

test_env = kb_dir.test_env
temp_uploads = kb_dir.temp_uploads


@pytest.fixture
def durable_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()


def _rows(test_env: Any) -> list[tuple[str, str]]:
    session = test_env[3]()
    try:
        return [
            (str(row.file_id), str(row.storage_key))
            for row in session.query(UploadedFile).all()
        ]
    finally:
        session.close()


def _succeed_without_document(**_kwargs: Any) -> IngestionResult:
    return IngestionResult(status="success", doc_id="unused", message="ok")


def _register_then_fail(**kwargs: Any) -> IngestionResult:
    update_collection_stats_sync(collection_name=kwargs["collection"])
    registered = register_document(
        collection=kwargs["collection"],
        source_path=kwargs["source_path"],
        user_id=kwargs["user_id"],
        file_id=kwargs["file_id"],
    )
    assert registered["created"] is True
    return IngestionResult(
        status="partial",
        doc_id=registered["doc_id"],
        completed_steps=[{"name": "register_document", "metadata": registered}],
        message="embedding failed",
    )


def _no_config() -> Any:
    return patch("xagent.web.api.kb._save_collection_config_after_ingest", AsyncMock())


def _ingest(test_env: Any, collection: str, ingest: Any, *patches: Any) -> Any:
    return local._post_ingest(
        test_env,
        "x.txt",
        collection,
        patch("xagent.web.api.kb.run_document_ingestion", side_effect=ingest),
        _no_config(),
        *patches,
    )


def _assert_kept(test_env: Any, before: list[tuple[str, str]]) -> None:
    assert _rows(test_env) == before
    assert get_unscoped_file_storage().exists(before[0][1])


@pytest.mark.parametrize(
    "ingest",
    [
        pytest.param(
            lambda **_kw: IngestionResult(status="error", message="embedding down"),
            id="unregistered",
        ),
        pytest.param(_register_then_fail, id="registered"),
    ],
)
def test_ingest_cloud_failure_keeps_a_pre_existing_row(
    test_env, temp_uploads, durable_storage, ingest
) -> None:
    cloud._post_cloud(test_env, _succeed_without_document, local._existing_collection())
    before = _rows(test_env)

    response = cloud._post_cloud(
        test_env, ingest, local._existing_collection(), _no_config()
    )

    assert response.json()[0]["status"] in {"error", "partial"}
    _assert_kept(test_env, before)


def test_ingest_failure_after_registration_keeps_a_pre_existing_row(
    test_env, temp_uploads, durable_storage
) -> None:
    _ingest(test_env, "coll", _succeed_without_document, local._existing_collection())
    before = _rows(test_env)

    response = _ingest(
        test_env, "coll", _register_then_fail, local._existing_collection()
    )

    assert response.status_code == 500
    _assert_kept(test_env, before)


def test_whole_collection_rollback_keeps_a_pre_existing_row(
    test_env, temp_uploads, durable_storage
) -> None:
    _ingest(test_env, "fresh", _succeed_without_document, local._existing_collection())
    before = _rows(test_env)

    response = _ingest(test_env, "fresh", _register_then_fail)

    assert response.status_code == 500
    _assert_kept(test_env, before)


def _delete_row_after_lookup(test_env: Any) -> Any:
    async def _then_delete(*args: Any) -> bool:
        existed = await _document_existed_before_ingest(*args)
        session = test_env[3]()
        try:
            session.query(UploadedFile).delete()
            session.commit()
        finally:
            session.close()
        return existed

    return patch("xagent.web.api.kb._document_existed_before_ingest", _then_delete)


def _post_local(test_env: Any, ingest: Any, *patches: Any) -> Any:
    return _ingest(test_env, "coll", ingest, local._existing_collection(), *patches)


def _post_cloud(test_env: Any, ingest: Any, *patches: Any) -> Any:
    return cloud._post_cloud(
        test_env, ingest, local._existing_collection(), _no_config(), *patches
    )


@pytest.mark.parametrize("post", [_post_local, _post_cloud], ids=["local", "cloud"])
def test_a_row_reinserted_after_the_lookup_is_rolled_back_as_new(
    test_env, temp_uploads, durable_storage, post
) -> None:
    post(test_env, _succeed_without_document)
    [(old_file_id, _)] = _rows(test_env)
    seen: list[tuple[str, str]] = []

    def _record_then_fail(**kwargs: Any) -> IngestionResult:
        seen.extend(_rows(test_env))
        return _register_then_fail(**kwargs)

    post(test_env, _record_then_fail, _delete_row_after_lookup(test_env))

    [(new_file_id, new_key)] = seen
    assert new_file_id != old_file_id
    assert _rows(test_env) == []
    assert not get_unscoped_file_storage().exists(new_key)


@pytest.mark.parametrize("post", [_post_local, _post_cloud], ids=["local", "cloud"])
def test_a_raise_after_reinsertion_removes_the_new_document(
    test_env, temp_uploads, durable_storage, post
) -> None:
    first: dict[str, Any] = {}
    post(test_env, raised._register_and_succeed(first))
    second: dict[str, Any] = {}

    post(
        test_env,
        raised._register_then_raise(second),
        _delete_row_after_lookup(test_env),
    )

    assert second["file_id"] != first["file_id"]
    assert raised._list_refs(second["file_id"]) == []
    assert _rows(test_env) == []
