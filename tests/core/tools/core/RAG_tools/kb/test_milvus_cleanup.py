"""Leftover Milvus row cleanup against the running Milvus at ``MILVUS_URI``.

Each test drops its own collection so one test's rows never affect another.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from xagent.core.tools.core.RAG_tools.core.schemas import RegisterDocumentRequest
from xagent.core.tools.core.RAG_tools.kb import milvus_cleanup
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    KBHandleProvider,
    ensure_milvus_collection,
    milvus_collection_name,
)
from xagent.core.tools.core.RAG_tools.kb.kb_ids import get_or_create_kb_id
from xagent.core.tools.core.RAG_tools.kb.milvus_cleanup import (
    purge_leftover_milvus_rows,
)
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    _safe_close_table,
    ensure_ingestion_runs_table,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)
from xagent.providers.vector_store.milvus import MilvusConnectionManager

pytestmark = pytest.mark.milvus

COLLECTION = "kb"
GRACE = 60


@cache
def _milvus() -> Any:
    from pymilvus import MilvusClient

    return MilvusClient(uri=os.environ["MILVUS_URI"])


@pytest.fixture(autouse=True)
def milvus_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")


@pytest.fixture
def client() -> Any:
    return _milvus()


@pytest.fixture(autouse=True)
def only_own_collections(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Scope the purge to this test's collections in the shared CI Milvus."""
    names: list[str] = []
    monkeypatch.setattr(
        milvus_cleanup, "_milvus_collections", lambda _client: list(names)
    )
    return names


@pytest.fixture
def model(client: Any, only_own_collections: list[str]) -> str:
    model = f"cleanup-{uuid.uuid4().hex[:12]}"
    name = ensure_milvus_collection(client, model, 3)
    only_own_collections.append(name)
    yield model
    client.drop_collection(name)


def _open(user_id: int | None = 1) -> KBCollectionHandle:
    return KBHandleProvider().open(
        KBCollectionContext(
            collection=COLLECTION,
            user_scope=KBUserScope(user_id=user_id, is_admin=user_id is None),
            access_mode=KBAccessMode.WRITE,
            allow_create=True,
            hide_missing=True,
            metadata_store=get_metadata_store(),
            vector_index_store=get_vector_index_store(),
            ingestion_status_store=get_ingestion_status_store(),
            main_pointer_store=get_main_pointer_store(),
            backend=KBStorageBackend.MILVUS,
            capabilities=KBBackendCapabilities.milvus(),
        )
    )


def _kb_id(user_id: int) -> str:
    return get_or_create_kb_id(
        get_vector_index_store().get_raw_connection(), COLLECTION, user_id
    )


def _register(user_id: int, doc_id: str, tmp_path: Path) -> None:
    source = tmp_path / f"{doc_id}.txt"
    source.write_text("kiwi", encoding="utf-8")
    _open(user_id).register_document(
        RegisterDocumentRequest(
            collection=COLLECTION,
            source_path=str(source),
            doc_id=doc_id,
            user_id=user_id,
        )
    )


def _run(
    doc_id: str,
    user_id: int,
    *,
    status: str,
    updated_at: datetime,
) -> None:
    conn = get_vector_index_store().get_raw_connection()
    ensure_ingestion_runs_table(conn)
    table = conn.open_table("ingestion_runs")
    try:
        table.add(
            [
                {
                    "collection": COLLECTION,
                    "doc_id": doc_id,
                    "status": status,
                    "message": "",
                    "parse_hash": "",
                    "created_at": updated_at,
                    "updated_at": updated_at,
                    "user_id": user_id,
                }
            ]
        )
    finally:
        _safe_close_table(table)


def _row(
    chunk_id: str,
    kb_id: str,
    doc_id: str,
    *,
    user_id: int = 1,
    visible: bool = False,
    created_at: int = 1,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "kb_id": kb_id,
        "user_id": user_id,
        "doc_id": doc_id,
        "parse_hash": "ph",
        "config_hash": "",
        "text": f"kiwi {chunk_id}",
        "dense": [1.0, 0.0, 0.0],
        "visible": visible,
        "created_at": created_at,
        "metadata": {},
    }


def _upsert(client: Any, model: str, rows: list[dict[str, Any]]) -> None:
    name = milvus_collection_name(model)
    client.upsert(name, rows)
    # Bounded reads follow a write only after a Strong read of the same data.
    client.query(
        name,
        filter="kb_id in {kb_ids}",
        filter_params={"kb_ids": sorted({row["kb_id"] for row in rows})},
        limit=1,
        consistency_level="Strong",
    )


def _live(client: Any, model: str, kb_ids: list[str]) -> set[str]:
    rows = client.query(
        milvus_collection_name(model),
        filter="kb_id in {kb_ids}",
        filter_params={"kb_ids": kb_ids},
        output_fields=["chunk_id"],
        consistency_level="Strong",
    )
    return {row["chunk_id"] for row in rows}


def _old() -> int:
    return int(time.time()) - 3600


def _purge() -> dict[str, int]:
    return purge_leftover_milvus_rows(grace_seconds=GRACE, now=time.time())


def test_an_invisible_row_with_an_old_terminal_run_is_deleted(
    client: Any, model: str, tmp_path: Path
) -> None:
    kb_id = _kb_id(1)
    _register(1, "doc", tmp_path)
    _run(
        "doc",
        1,
        status="success",
        updated_at=datetime.fromtimestamp(_old(), tz=timezone.utc),
    )
    _upsert(client, model, [_row("c1", kb_id, "doc", created_at=_old())])

    assert _purge().get(milvus_collection_name(model)) == 1
    assert _live(client, model, [kb_id]) == set()


def test_an_invisible_row_with_a_running_run_is_kept(
    client: Any, model: str, tmp_path: Path
) -> None:
    kb_id = _kb_id(1)
    _register(1, "doc", tmp_path)
    _run(
        "doc",
        1,
        status="running",
        updated_at=datetime.fromtimestamp(_old(), tz=timezone.utc),
    )
    _upsert(client, model, [_row("c1", kb_id, "doc", created_at=_old())])

    assert _purge().get(milvus_collection_name(model), 0) == 0
    assert _live(client, model, [kb_id]) == {"c1"}


def test_an_invisible_row_with_a_terminal_run_inside_the_grace_is_kept(
    client: Any, model: str, tmp_path: Path
) -> None:
    kb_id = _kb_id(1)
    _register(1, "doc", tmp_path)
    _run(
        "doc",
        1,
        status="failed",
        updated_at=datetime.now(timezone.utc),
    )
    _upsert(client, model, [_row("c1", kb_id, "doc", created_at=_old())])

    assert _purge().get(milvus_collection_name(model), 0) == 0
    assert _live(client, model, [kb_id]) == {"c1"}


def test_an_invisible_row_without_a_run_row_but_with_a_document_is_kept(
    client: Any, model: str, tmp_path: Path
) -> None:
    kb_id = _kb_id(1)
    _register(1, "doc", tmp_path)
    _upsert(client, model, [_row("c1", kb_id, "doc", created_at=_old())])

    assert _purge().get(milvus_collection_name(model), 0) == 0
    assert _live(client, model, [kb_id]) == {"c1"}


@pytest.mark.parametrize("visible", [True, False])
def test_a_row_whose_document_is_missing_is_deleted(
    client: Any, model: str, tmp_path: Path, visible: bool
) -> None:
    kb_id = _kb_id(1)
    _register(1, "kept", tmp_path)
    _upsert(
        client,
        model,
        [_row("c1", kb_id, "gone", visible=visible, created_at=_old())],
    )

    assert _purge().get(milvus_collection_name(model)) == 1
    assert _live(client, model, [kb_id]) == set()


def test_a_row_of_a_kb_id_absent_from_the_ledger_is_deleted(
    client: Any, model: str
) -> None:
    _kb_id(1)
    orphan = uuid.uuid4().hex
    _upsert(client, model, [_row("c1", orphan, "doc", visible=True, created_at=_old())])

    assert _purge().get(milvus_collection_name(model)) == 1
    assert _live(client, model, [orphan]) == set()


def test_an_orphan_delete_leaves_a_live_kb_id_with_the_same_doc_id(
    client: Any, model: str, tmp_path: Path
) -> None:
    live = _kb_id(1)
    orphan = uuid.uuid4().hex
    doc_id = f"shared-{uuid.uuid4().hex[:8]}"
    _register(1, doc_id, tmp_path)
    # Same doc_id, same collection: the orphan's old row must go, the live
    # kb_id's old row must stay, so the delete must be scoped by kb_id.
    _upsert(
        client,
        model,
        [
            _row("c-orphan", orphan, doc_id, visible=True, created_at=_old()),
            _row("c-live", live, doc_id, visible=True, created_at=_old()),
        ],
    )

    assert _purge().get(milvus_collection_name(model)) == 1
    assert _live(client, model, [orphan]) == set()
    assert _live(client, model, [live]) == {"c-live"}


def test_an_empty_kb_ids_ledger_deletes_nothing(
    client: Any, model: str, caplog: pytest.LogCaptureFixture
) -> None:
    orphan = uuid.uuid4().hex
    _upsert(client, model, [_row("c1", orphan, "doc", visible=True, created_at=_old())])

    with caplog.at_level("ERROR"):
        assert _purge() == {}

    assert _live(client, model, [orphan]) == {"c1"}
    assert "kb_ids" in caplog.text


def test_a_row_rewritten_between_the_scan_and_the_delete_is_kept(
    client: Any, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _kb_id(1)
    orphan = uuid.uuid4().hex
    _upsert(client, model, [_row("c1", orphan, "doc", created_at=_old())])
    name = milvus_collection_name(model)
    real = milvus_cleanup._delete_rows

    def delete_after_rewrite(*args: Any, **kwargs: Any) -> int:
        client.upsert(name, [_row("c1", orphan, "doc", created_at=int(time.time()))])
        client.query(
            name,
            filter="kb_id == {kb_id}",
            filter_params={"kb_id": orphan},
            limit=1,
            consistency_level="Strong",
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(milvus_cleanup, "_delete_rows", delete_after_rewrite)

    assert _purge().get(name, 0) == 0
    assert _live(client, model, [orphan]) == {"c1"}


def test_another_users_same_doc_id_is_untouched(
    client: Any, model: str, tmp_path: Path
) -> None:
    mine = _kb_id(1)
    theirs = _kb_id(2)
    _register(2, "shared", tmp_path)
    _upsert(
        client,
        model,
        [
            _row("c1", mine, "shared", user_id=1, created_at=_old()),
            _row("c2", theirs, "shared", user_id=2, created_at=_old()),
        ],
    )

    # documents keys by (collection, doc_id) only, so the shared row keeps both
    # owners' Milvus rows even though the ledger row names user 2.
    assert _purge().get(milvus_collection_name(model), 0) == 0
    assert _live(client, model, [mine, theirs]) == {"c1", "c2"}


def test_a_failed_ledger_read_aborts_without_deleting(
    client: Any, model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb_id = _kb_id(1)
    _register(1, "doc", tmp_path)
    _upsert(client, model, [_row("c1", kb_id, "doc", created_at=_old())])

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("ledger is unreadable")

    monkeypatch.setattr(get_vector_index_store(), "iter_batches", explode)

    with pytest.raises(DatabaseOperationError):
        _purge()
    assert _live(client, model, [kb_id]) == {"c1"}


def test_the_cap_is_respected(
    client: Any, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _kb_id(1)
    monkeypatch.setattr(milvus_cleanup, "MAX_DELETES_PER_RUN", 2)
    kb_ids = [uuid.uuid4().hex for _ in range(5)]
    _upsert(
        client,
        model,
        [
            _row(f"c{i}", kb_id, f"doc-{i}", created_at=_old())
            for i, kb_id in enumerate(kb_ids)
        ],
    )

    assert _purge().get(milvus_collection_name(model)) == 2
    remaining = set().union(*[_live(client, model, [kb_id]) for kb_id in kb_ids])
    assert len(remaining) == 3


def test_a_failing_collection_lets_the_others_purge_then_raises(
    client: Any,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
    only_own_collections: list[str],
) -> None:
    _kb_id(1)
    other = f"cleanup-{uuid.uuid4().hex[:12]}"
    other_name = ensure_milvus_collection(client, other, 3)
    only_own_collections.append(other_name)
    try:
        orphan = uuid.uuid4().hex
        _upsert(client, model, [_row("c1", orphan, "doc-a", created_at=_old())])
        _upsert(client, other, [_row("c2", orphan, "doc-b", created_at=_old())])
        real = milvus_cleanup._scan_rows
        failing = milvus_collection_name(model)

        def scan(*args: Any, **kwargs: Any) -> Any:
            if args[1] == failing:
                raise RuntimeError("boom")
            return real(*args, **kwargs)

        monkeypatch.setattr(milvus_cleanup, "_scan_rows", scan)

        with pytest.raises(RuntimeError):
            _purge()
        assert _live(client, other, [orphan]) == set()
    finally:
        client.drop_collection(milvus_collection_name(other))


def test_lancedb_engine_returns_empty(
    client: Any, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _kb_id(1)
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "lancedb")
    touched: list[str] = []
    monkeypatch.setattr(
        MilvusConnectionManager,
        "get_shared_client_from_env",
        lambda _self: touched.append("client"),
    )

    assert purge_leftover_milvus_rows() == {}
    assert touched == []
