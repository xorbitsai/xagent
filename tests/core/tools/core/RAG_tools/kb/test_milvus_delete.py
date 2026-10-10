"""Milvus deletes and the rollback steps, on the server at ``MILVUS_URI``.

A delete removes the Milvus rows first and the ledger once that succeeds; a failed
Milvus delete leaves both stores and ``kb_ids`` as they were. Rows are read back
with Strong consistency: searches use Bounded, which may still return a row for a
moment after its delete.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DocumentProcessingStatus,
    RegisterDocumentRequest,
)
from xagent.core.tools.core.RAG_tools.kb import collection_handle
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    KBHandleProvider,
    ensure_milvus_collection,
    milvus_collection_name,
)
from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
from xagent.core.tools.core.RAG_tools.kb.kb_ids import get_or_create_kb_id, read_kb_ids
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBDocumentRowsSnapshot,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_name,
    to_model_tag,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.providers.vector_store.milvus import MilvusConnectionManager

pytestmark = pytest.mark.milvus

COLLECTION = "kb"
PARSE = "ph-1"
NEXT_PARSE = "ph-2"
SEEN_KB_IDS: set[str] = set()


class Faulty:
    """Forwards every call to a Milvus client; ``fail`` makes named calls raise."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.fail: dict[str, Exception] = {}
        self.after: dict[str, Callable[[], None]] = {}

    def __getattr__(self, name: str) -> Any:
        target = getattr(self.client, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            if name in self.fail:
                raise self.fail[name]
            result = target(*args, **kwargs)
            if name in self.after:
                self.after.pop(name)()
            return result

        return call

    def names(self, method: str) -> list[str]:
        return [args[0] for name, args, _ in self.calls if name == method]

    def count(self, method: str) -> int:
        return sum(1 for name, _, _ in self.calls if name == method)


@pytest.fixture(scope="module")
def client() -> Any:
    return _milvus()


def _shared_model(client: Any) -> Iterator[str]:
    """One collection for the whole module: rows are scoped by each test's kb_ids."""
    model = f"delete-{uuid.uuid4().hex[:12]}"
    ensure_milvus_collection(client, model, 3)
    yield model
    client.drop_collection(milvus_collection_name(model))


model = pytest.fixture(scope="module")(_shared_model)
other_model = pytest.fixture(scope="module")(_shared_model)


@contextmanager
def released(client: Any, *models: str) -> Iterator[None]:
    for name in models:
        client.release_collection(milvus_collection_name(name))
    try:
        yield
    finally:
        for name in models:
            client.load_collection(milvus_collection_name(name))


@pytest.fixture(autouse=True)
def milvus_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")
    SEEN_KB_IDS.clear()


@pytest.fixture
def faulty(client: Any, monkeypatch: pytest.MonkeyPatch) -> Faulty:
    faulty = Faulty(client)
    monkeypatch.setattr(
        MilvusConnectionManager, "get_shared_client_from_env", lambda _self: faulty
    )
    return faulty


def _open(
    user_id: int | None = 1, *, collection: str = COLLECTION
) -> KBCollectionHandle:
    return KBHandleProvider().open(
        KBCollectionContext(
            collection=collection,
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


def _chunks(
    handle: KBCollectionHandle,
    doc_id: str,
    ids: list[str],
    *,
    user_id: int = 1,
    parse_hash: str = PARSE,
) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    document = handle.load_document(doc_id, is_admin=True)
    made_from = {"content_hash": document.content_hash} if document else {}
    handle.write_chunks(
        doc_id,
        parse_hash,
        "cfg",
        {},
        [
            {
                "chunk_id": c,
                "index": i,
                "text": f"kiwi {c}",
                "created_at": now,
                "metadata": made_from,
            }
            for i, c in enumerate(ids)
        ],
        user_id=user_id,
    )


def _embed(
    handle: KBCollectionHandle,
    model: str,
    doc_id: str,
    *,
    user_id: int = 1,
    parse_hash: str = PARSE,
    visible: bool = False,
) -> None:
    """Write the ledger's chunks Milvus does not hold, as ``write_embeddings`` does."""
    client = _milvus()
    name = milvus_collection_name(model)
    kb_id = _kb_id(user_id, handle.context.collection)
    held = {
        row["chunk_id"]
        for row in client.query(
            name,
            filter="kb_id == {kb_id} and doc_id == {doc_id}",
            filter_params={"kb_id": kb_id, "doc_id": doc_id},
            output_fields=["chunk_id"],
            consistency_level="Strong",
        )
    }
    chunks = handle.read_existing_chunks(doc_id, parse_hash, "cfg", user_id=user_id)
    rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "kb_id": kb_id,
            "user_id": user_id,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            "config_hash": "",
            "text": chunk["text"],
            "dense": [1.0, 0.0, 0.0],
            "visible": visible,
            "created_at": 1,
            "metadata": {},
        }
        for chunk in chunks
        if chunk["chunk_id"] not in held
    ]
    if rows:
        client.upsert(name, rows)
        # Bounded counts follow a write only after a Strong read of the same data.
        client.query(
            name,
            filter="kb_id == {kb_id}",
            filter_params={"kb_id": kb_id},
            limit=1,
            consistency_level="Strong",
        )


def _ingest(
    handle: KBCollectionHandle,
    model: str,
    doc_id: str,
    ids: list[str],
    tmp_path: Path,
    *,
    user_id: int = 1,
    parse_hash: str = PARSE,
    commit: bool = True,
) -> None:
    source = tmp_path / f"{doc_id}.txt"
    source.write_text("kiwi", encoding="utf-8")
    handle.register_document(
        RegisterDocumentRequest(
            collection=COLLECTION,
            source_path=str(source),
            doc_id=doc_id,
            user_id=user_id,
        )
    )
    _chunks(handle, doc_id, ids, user_id=user_id, parse_hash=parse_hash)
    _embed(
        handle, model, doc_id, user_id=user_id, parse_hash=parse_hash, visible=commit
    )


@cache
def _milvus() -> Any:
    from pymilvus import MilvusClient

    return MilvusClient(uri=os.environ["MILVUS_URI"])


def _rows(client: Any, model: str) -> dict[str, dict[str, Any]]:
    """The rows of this test's kb_ids; other tests' rows share the collection."""
    rows = client.query(
        milvus_collection_name(model),
        filter="kb_id in {kb_ids}",
        filter_params={"kb_ids": sorted(SEEN_KB_IDS)},
        output_fields=["kb_id", "user_id", "doc_id", "visible"],
        consistency_level="Strong",
    )
    return {row.pop("chunk_id"): row for row in rows}


def _visible(client: Any, model: str) -> set[str]:
    return {id_ for id_, row in _rows(client, model).items() if row["visible"]}


def _ledger(handle: KBCollectionHandle) -> tuple[list[str], int]:
    """The documents and the chunk count the ledger holds, for every owner."""
    return (
        handle.list_collection_documents(None, True),
        handle.collection_stats(None, True)["chunks"],
    )


def _kb_id(user_id: int, collection: str = COLLECTION) -> str:
    kb_id = get_or_create_kb_id(
        get_vector_index_store().get_raw_connection(), collection, user_id
    )
    SEEN_KB_IDS.add(kb_id)
    return kb_id


def _kb_id_owners(collection: str = COLLECTION) -> set[str]:
    conn = get_vector_index_store().get_raw_connection()
    return set(read_kb_ids(conn, user_id=None, is_admin=True).get(collection, []))


def _delete_documents(
    handle: KBCollectionHandle, doc_ids: list[str], *, admin: bool = False
) -> dict[str, int]:
    return handle.delete_documents_data(
        doc_ids, user_id=None if admin else 1, is_admin=admin
    )


# --- document delete ---------------------------------------------------------


def test_a_document_delete_removes_its_rows_in_both_stores_and_nothing_else(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "gone", ["a", "b"], tmp_path)
    _ingest(handle, model, "kept", ["c"], tmp_path)
    _ingest(_open(2), model, "other", ["d"], tmp_path, user_id=2)

    counts = _delete_documents(handle, ["gone"])

    assert _visible(client, model) == {"c", "d"}
    assert _ledger(handle) == (["kept", "other"], 2)
    assert counts[embeddings_table_name(to_model_tag(model))] == 2
    assert counts["documents"] == 1 and counts["chunks"] == 2


def test_a_delete_reaches_every_model_collection(
    client: Any, model: str, other_model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "gone", ["a"], tmp_path)
    _chunks(handle, "gone", ["b"], parse_hash=NEXT_PARSE)
    _embed(handle, other_model, "gone", parse_hash=NEXT_PARSE, visible=True)
    _ingest(handle, other_model, "kept", ["c"], tmp_path)

    counts = _delete_documents(handle, ["gone"])

    assert _visible(client, model) == set()
    assert _visible(client, other_model) == {"c"}
    assert counts[embeddings_table_name(to_model_tag(model))] == 1
    assert counts[embeddings_table_name(to_model_tag(other_model))] == 1


def test_an_admin_delete_reaches_the_rows_of_every_owner(
    client: Any, model: str, tmp_path: Path
) -> None:
    _ingest(_open(1), model, "mine", ["a"], tmp_path)
    _ingest(_open(2), model, "theirs", ["b"], tmp_path, user_id=2)
    _ingest(_open(2), model, "kept", ["c"], tmp_path, user_id=2)
    admin = _open(None)

    _delete_documents(admin, ["mine", "theirs"], admin=True)

    assert _visible(client, model) == {"c"}
    assert _ledger(admin) == (["kept"], 1)


def test_a_tenant_delete_leaves_another_owners_rows_in_both_stores(
    client: Any, model: str, tmp_path: Path
) -> None:
    _ingest(_open(1), model, "doc", ["a"], tmp_path)
    stranger = _open(2)
    _chunks(stranger, "doc", ["x"], user_id=2)
    _embed(stranger, model, "doc", user_id=2, visible=True)

    stranger.delete_documents_data(["doc"], user_id=2, is_admin=False)

    assert _visible(client, model) == {"a"}
    assert _ledger(_open(None))[0] == ["doc"]
    assert _open(1).collection_stats(1, False)["chunks"] == 1


def test_a_failed_milvus_delete_leaves_the_ledger_and_the_rows_and_a_retry_finishes(
    client: Any, faulty: Faulty, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a", "b"], tmp_path)
    faulty.fail["delete"] = RuntimeError("milvus is down")

    with pytest.raises(DatabaseOperationError, match="milvus is down") as raised:
        _delete_documents(handle, ["doc"])

    assert raised.value.__cause__ is None
    assert raised.value.details == {
        "deleted_counts": {},
        "deleted_doc_ids": [],
        "failed_batch_index": 1,
    }
    assert _visible(client, model) == {"a", "b"}
    assert _ledger(handle) == (["doc"], 2)

    del faulty.fail["delete"]
    _delete_documents(handle, ["doc"])

    assert _visible(client, model) == set()
    assert _ledger(handle) == ([], 0)


def test_a_delete_in_batches_looks_up_the_collections_and_models_once(
    client: Any, faulty: Faulty, model: str
) -> None:
    kb_id = _kb_id(1)
    client.upsert(
        milvus_collection_name(model),
        [
            {
                "chunk_id": f"c{doc_id}",
                "kb_id": kb_id,
                "user_id": 1,
                "doc_id": doc_id,
                "parse_hash": PARSE,
                "config_hash": "",
                "text": "kiwi",
                "dense": [1.0, 0.0, 0.0],
                "visible": True,
                "created_at": 1,
                "metadata": {},
            }
            for doc_id in ("doc-000", "doc-100", "doc-200")
        ],
    )
    ids = [f"doc-{n:03d}" for n in range(250)]
    faulty.calls.clear()

    counts = _open().delete_documents_data(ids, user_id=1, is_admin=False)

    assert counts == {embeddings_table_name(to_model_tag(model)): 3}
    assert faulty.count("delete") == 3 * len(
        collection_handle._milvus_collections(client)
    )
    collections = len(collection_handle._milvus_collections(client))
    assert faulty.count("list_collections") == 1
    assert faulty.count("describe_collection") == collections
    assert _visible(client, model) == set()


def test_document_ids_with_quotes_and_multibyte_text_are_deleted(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    odd = "d'q\"文档 x"
    _ingest(handle, model, odd, ["a"], tmp_path)
    _ingest(handle, model, "kept", ["b"], tmp_path)

    _delete_documents(handle, [odd])

    assert _visible(client, model) == {"b"} and _ledger(handle) == (["kept"], 1)


def test_a_failed_batch_leaves_the_ledger_untouched_and_names_the_batch(
    client: Any, faulty: Faulty, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc-000", ["a"], tmp_path)
    ids = [f"doc-{n:03d}" for n in range(250)]
    first_batch = len(collection_handle._milvus_collections(client))
    deletes: list[int] = []

    def fail_the_second_batch() -> None:
        deletes.append(1)
        if len(deletes) < first_batch:
            faulty.after["delete"] = fail_the_second_batch
        else:
            faulty.fail["delete"] = RuntimeError("batch failed")

    faulty.after["delete"] = fail_the_second_batch

    warnings: list[str] = []
    with pytest.raises(DatabaseOperationError) as raised:
        handle.delete_documents_data(
            ids, user_id=1, is_admin=False, warnings_out=warnings
        )

    assert raised.value.details == {
        "deleted_counts": {},
        "deleted_doc_ids": [],
        "failed_batch_index": 2,
    }
    assert warnings and "batch failed" in warnings[0]
    assert _ledger(handle) == (["doc-000"], 1)


def test_a_collection_that_is_not_loaded_is_loaded_for_the_delete(
    client: Any, model: str, other_model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a"], tmp_path)
    _ingest(handle, other_model, "kept", ["b"], tmp_path)
    with released(client, model, other_model):
        counts = _delete_documents(handle, ["doc"])

        for name in (model, other_model):
            state = client.get_load_state(milvus_collection_name(name))["state"]
            assert state.name == "Loaded"
    assert counts[embeddings_table_name(to_model_tag(model))] == 1
    assert embeddings_table_name(to_model_tag(other_model)) not in counts
    assert _visible(client, model) == set() and _visible(client, other_model) == {"b"}
    assert _ledger(handle) == (["kept"], 1)


def test_a_collection_that_cannot_be_loaded_fails_the_delete_and_leaves_the_ledger(
    client: Any, faulty: Faulty, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a"], tmp_path)
    with released(client, model):
        faulty.fail["load_collection"] = RuntimeError("no memory")

        with pytest.raises(DatabaseOperationError, match="no memory") as raised:
            _delete_documents(handle, ["doc"])

        assert milvus_collection_name(model) in str(raised.value)
        assert _ledger(handle) == (["doc"], 1)
        del faulty.fail["load_collection"]
        _delete_documents(handle, ["doc"])
    assert _visible(client, model) == set() and _ledger(handle) == ([], 0)


def test_a_delete_of_a_collection_never_written_to_milvus_calls_no_delete(
    faulty: Faulty, model: str, tmp_path: Path
) -> None:
    handle = _open()
    source = tmp_path / "doc.txt"
    source.write_text("kiwi", encoding="utf-8")
    handle.register_document(
        RegisterDocumentRequest(
            collection=COLLECTION, source_path=str(source), doc_id="doc", user_id=1
        )
    )

    counts = _delete_documents(handle, ["doc"])

    assert counts == {"documents": 1}
    assert faulty.names("delete") == []


# --- collection delete -------------------------------------------------------


def _two_owners_two_models(
    model: str, other_model: str, tmp_path: Path
) -> KBCollectionHandle:
    _ingest(_open(1), model, "mine", ["a"], tmp_path)
    _ingest(_open(1), other_model, "mine-too", ["b"], tmp_path)
    _ingest(_open(2), model, "theirs", ["c"], tmp_path, user_id=2)
    _ingest(_open(1, collection="other"), model, "elsewhere", ["z"], tmp_path)
    return _open(None)


def test_an_admin_collection_delete_covers_every_owner_and_model_collection(
    client: Any, model: str, other_model: str, tmp_path: Path
) -> None:
    admin = _two_owners_two_models(model, other_model, tmp_path)
    other_kb_id = _kb_id(1, "other")

    counts = admin.delete_collection_data(user_id=None, is_admin=True)

    assert _visible(client, model) == {"z"} and _visible(client, other_model) == set()
    assert _ledger(admin) == ([], 0)
    assert counts[embeddings_table_name(to_model_tag(model))] == 2
    assert counts[embeddings_table_name(to_model_tag(other_model))] == 1
    assert counts["documents"] == 3 and counts["chunks"] == 3
    assert _rows(client, model)["z"]["kb_id"] == other_kb_id


def test_a_tenant_collection_delete_covers_only_the_callers_rows(
    client: Any, model: str, other_model: str, tmp_path: Path
) -> None:
    admin = _two_owners_two_models(model, other_model, tmp_path)

    _open(1).delete_collection_data(user_id=1, is_admin=False)

    assert _visible(client, model) == {"c", "z"}
    assert _visible(client, other_model) == set()
    assert _ledger(admin) == (["theirs"], 1)


def test_a_failed_collection_delete_leaves_the_ledger_and_the_kb_ids(
    client: Any, faulty: Faulty, model: str, other_model: str, tmp_path: Path
) -> None:
    admin = _two_owners_two_models(model, other_model, tmp_path)
    owners = _kb_id_owners()
    faulty.fail["delete"] = RuntimeError("milvus is down")

    with pytest.raises(DatabaseOperationError, match="milvus is down"):
        admin.delete_collection_data(user_id=None, is_admin=True)
    result = asyncio.run(
        KBCoordinator().delete_collection(COLLECTION, None, is_admin=True)
    )

    assert result.status == "error" and "milvus is down" in result.message
    assert _visible(client, model) == {"a", "c", "z"}
    assert _ledger(admin) == (["mine", "mine-too", "theirs"], 3)
    assert _kb_id_owners() == owners and len(owners) == 2


def test_the_rollback_cleanup_of_a_new_collection_deletes_its_rows(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a"], tmp_path, commit=False)

    handle.cleanup_collection_data_after_rollback(user_id=1, is_admin=False)

    assert _rows(client, model) == {} and _ledger(handle) == ([], 0)


async def _save_configs(*owners: int) -> None:
    store = get_metadata_store()
    for owner in owners:
        await store.save_collection_config(COLLECTION, "{}", owner)


def test_an_admin_collection_delete_removes_the_kb_ids_with_the_config_rows(
    client: Any, model: str, other_model: str, tmp_path: Path
) -> None:
    _two_owners_two_models(model, other_model, tmp_path)
    asyncio.run(_save_configs(1, 2))

    result = asyncio.run(
        KBCoordinator().delete_collection(COLLECTION, None, is_admin=True)
    )

    assert result.status == "success"
    assert get_metadata_store().list_collection_config_owner_ids(COLLECTION) == set()
    assert _kb_id_owners() == set()
    assert _kb_id(1, "other") and _visible(client, model) == {"z"}


def test_a_tenant_collection_delete_removes_only_the_callers_kb_id_and_config(
    client: Any, model: str, other_model: str, tmp_path: Path
) -> None:
    _two_owners_two_models(model, other_model, tmp_path)
    asyncio.run(_save_configs(1, 2))
    theirs = _kb_id(2)

    result = asyncio.run(KBCoordinator().delete_collection(COLLECTION, 1, False))

    assert result.status == "success"
    assert get_metadata_store().list_collection_config_owner_ids(COLLECTION) == {2}
    assert _kb_id_owners() == {theirs}
    assert _visible(client, model) == {"c", "z"}


def test_an_admin_delete_that_leaves_ledger_documents_keeps_the_kb_ids_until_a_retry(
    client: Any,
    model: str,
    other_model: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin = _two_owners_two_models(model, other_model, tmp_path)
    asyncio.run(_save_configs(1, 2))
    owners = _kb_id_owners()
    delete_ledger = LanceDBVectorIndexStore.delete_collection_data
    monkeypatch.setattr(
        LanceDBVectorIndexStore, "delete_collection_data", lambda *_, **__: {}
    )

    asyncio.run(KBCoordinator().delete_collection(COLLECTION, None, is_admin=True))

    assert _visible(client, model) == {"z"} and _visible(client, other_model) == set()
    assert _ledger(admin) == (["mine", "mine-too", "theirs"], 3)
    assert _kb_id_owners() == owners
    monkeypatch.setattr(
        LanceDBVectorIndexStore, "delete_collection_data", delete_ledger
    )

    retried = asyncio.run(
        KBCoordinator().delete_collection(COLLECTION, None, is_admin=True)
    )

    assert retried.status == "success" and _ledger(admin) == ([], 0)
    assert _kb_id_owners() == set()


# --- rename ------------------------------------------------------------------


def _rename(new: str, user_id: int | None, is_admin: bool) -> list[str]:
    return asyncio.run(
        KBCoordinator().rename_collection(COLLECTION, new, user_id, is_admin)
    )


def test_a_rename_keeps_the_kb_id_and_the_rows_follow_the_name(
    client: Any, model: str, tmp_path: Path
) -> None:
    _ingest(_open(1), model, "mine", ["a"], tmp_path)
    _ingest(_open(2), model, "theirs", ["b"], tmp_path, user_id=2)
    kb_id, before = _kb_id(1), _rows(client, model)

    assert _rename("renamed", 1, False) == []

    assert _rows(client, model) == before
    assert _kb_id(1, "renamed") == kb_id
    assert _open(1, collection="renamed").collection_stats(1, False)["embeddings"] == 1
    assert _open(1).collection_stats(1, False)["embeddings"] == 0
    assert _open(2).collection_stats(2, False)["embeddings"] == 1


def test_an_admin_rename_moves_every_owners_kb_id(
    client: Any, model: str, tmp_path: Path
) -> None:
    _ingest(_open(1), model, "mine", ["a"], tmp_path)
    _ingest(_open(2), model, "theirs", ["b"], tmp_path, user_id=2)
    owners = _kb_id_owners()

    assert _rename("renamed", None, True) == []

    assert _kb_id_owners("renamed") == owners and _kb_id_owners() == set()
    admin = _open(None, collection="renamed")
    assert admin.collection_stats(None, True)["embeddings"] == 2


def test_a_rename_replaces_the_owners_leftover_kb_id_under_the_target_name(
    client: Any, model: str, tmp_path: Path
) -> None:
    _ingest(_open(1), model, "mine", ["a"], tmp_path)
    kb_id = _kb_id(1)
    leftover = _kb_id(1, "renamed")

    assert _rename("renamed", 1, False) == []

    assert _kb_id_owners("renamed") == {kb_id} and kb_id != leftover


@pytest.mark.parametrize("failure", ["raises", "warns"])
def test_a_failed_data_rename_moves_the_kb_ids_back(
    client: Any,
    model: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _ingest(_open(1), model, "mine", ["a"], tmp_path)
    owners = _kb_id_owners()

    def broken(*_: Any, **__: Any) -> list[str]:
        if failure == "raises":
            raise RuntimeError("no disk")
        return ["chunks: no disk"]

    monkeypatch.setattr(LanceDBVectorIndexStore, "rename_collection_data", broken)
    if failure == "raises":
        with pytest.raises(RuntimeError, match="no disk"):
            _rename("renamed", 1, False)
    else:
        assert _rename("renamed", 1, False) == ["chunks: no disk"]

    assert _kb_id_owners() == owners and _kb_id_owners("renamed") == set()
    assert _open(1).collection_stats(1, False)["embeddings"] == 1


# --- failed-ingest rollback --------------------------------------------------


def test_the_discard_removes_invisible_rows_and_keeps_every_committed_row(
    client: Any, model: str, other_model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a", "b"], tmp_path)
    _ingest(handle, model, "other-doc", ["c"], tmp_path)
    _ingest(_open(2), model, "doc", ["s"], tmp_path, user_id=2, commit=False)
    _chunks(handle, "doc", ["x", "y"], parse_hash=NEXT_PARSE)
    _embed(handle, model, "doc", parse_hash=NEXT_PARSE)
    _embed(handle, other_model, "doc", parse_hash=NEXT_PARSE)
    _chunks(handle, "other-doc", ["q"], parse_hash=NEXT_PARSE)
    _embed(handle, model, "other-doc", parse_hash=NEXT_PARSE)

    assert handle.discard_uncommitted_embeddings("doc", user_id=1) == 4

    assert set(_rows(client, model)) == {"a", "b", "c", "q", "s"}
    assert _visible(client, model) == {"a", "b", "c"}
    assert _rows(client, other_model) == {}
    assert handle.discard_uncommitted_embeddings("doc", user_id=1) == 0


def test_the_discard_skips_a_collection_that_is_not_loaded(
    client: Any, model: str, other_model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a"], tmp_path, commit=False)
    _ingest(handle, other_model, "doc", ["b"], tmp_path, commit=False)
    with released(client, other_model):
        assert handle.discard_uncommitted_embeddings("doc", user_id=1) == 1

    assert _rows(client, model) == {}


def test_a_failed_discard_raises(faulty: Faulty, model: str, tmp_path: Path) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a"], tmp_path, commit=False)
    faulty.fail["delete"] = RuntimeError("milvus is down")

    with pytest.raises(DatabaseOperationError, match="milvus is down"):
        handle.discard_uncommitted_embeddings("doc", user_id=1)


def test_a_commit_that_fails_after_the_flip_hides_its_rows_again_for_the_discard(
    client: Any, faulty: Faulty, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a", "b"], tmp_path)
    _chunks(handle, "doc", ["x", "y", "z"], parse_hash=NEXT_PARSE)
    _embed(handle, model, "doc", parse_hash=NEXT_PARSE)

    def delete_one_after_the_flip() -> None:
        client.delete(
            milvus_collection_name(model),
            filter="chunk_id == {id}",
            filter_params={"id": "z"},
        )

    faulty.after["upsert"] = delete_one_after_the_flip
    with pytest.raises(DatabaseOperationError, match="missing or invisible"):
        handle.commit_embeddings("doc", NEXT_PARSE, model, user_id=1)

    assert _visible(client, model) == {"a", "b"}
    assert handle.discard_uncommitted_embeddings("doc", user_id=1) == 2
    assert set(_rows(client, model)) == {"a", "b"}


def test_a_failed_commit_leaves_the_rows_it_did_not_show_as_they_were(
    client: Any, faulty: Faulty, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a", "b"], tmp_path)
    _chunks(handle, "doc", ["b", "x", "z"], parse_hash=NEXT_PARSE)
    _embed(handle, model, "doc", parse_hash=NEXT_PARSE)
    faulty.after["upsert"] = lambda: client.delete(
        milvus_collection_name(model),
        filter="chunk_id == {id}",
        filter_params={"id": "z"},
    )

    with pytest.raises(DatabaseOperationError, match="missing or invisible"):
        handle.commit_embeddings("doc", NEXT_PARSE, model, user_id=1)

    assert _visible(client, model) == {"a", "b"}
    assert set(_rows(client, model)) == {"a", "b", "x"}


def test_a_superseded_commit_does_not_hide_the_rows_of_the_newer_run(
    client: Any,
    faulty: Faulty,
    model: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a", "b", "c"], tmp_path, commit=False)
    name = milvus_collection_name(model)
    faulty.after["upsert"] = lambda: client.upsert(
        name, [{"chunk_id": "c", "visible": False}], partial_update=True
    )
    gates: list[int] = []

    def newer_run_commits_then_supersedes() -> None:
        gates.append(1)
        if len(gates) == 2:
            client.upsert(
                name, [{"chunk_id": "c", "visible": True}], partial_update=True
            )
            raise RuntimeError("superseded")

    with pytest.raises(DatabaseOperationError, match="missing or invisible"):
        handle.commit_embeddings(
            "doc",
            PARSE,
            model,
            commit_gate=newer_run_commits_then_supersedes,
            user_id=1,
        )

    assert len(gates) == 2 and _visible(client, model) == {"a", "b", "c"}
    assert "Not hiding the chunks of doc again: RuntimeError('superseded')" in (
        caplog.text
    )


def test_a_failed_hide_does_not_replace_the_commit_error(
    client: Any, faulty: Faulty, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a", "b"], tmp_path, commit=False)

    def lose_a_row_and_break_updates() -> None:
        client.delete(
            milvus_collection_name(model),
            filter="chunk_id == {id}",
            filter_params={"id": "b"},
        )
        faulty.fail["upsert"] = RuntimeError("cannot hide")

    faulty.after["upsert"] = lose_a_row_and_break_updates
    with pytest.raises(
        DatabaseOperationError, match="1 of 2 chunks of doc are missing"
    ):
        handle.commit_embeddings("doc", PARSE, model, user_id=1)

    assert _visible(client, model) == {"a"}


# --- web-file refresh rollback -----------------------------------------------


def _snapshot(handle: KBCollectionHandle, doc_ids: list[str]) -> KBDocumentRowsSnapshot:
    return handle.capture_document_rows(doc_ids, user_id=1, is_admin=False)


def _status(handle: KBCollectionHandle, doc_id: str) -> str:
    (row,) = handle.load_ingestion_status(doc_id=doc_id, user_id=1)
    return str(row["status"])


def _refresh(
    handle: KBCollectionHandle,
    model: str,
    doc_id: str,
    ids: list[str],
    *,
    commit: bool,
    user_id: int = 1,
) -> None:
    _chunks(handle, doc_id, ids, user_id=user_id, parse_hash=NEXT_PARSE)
    _embed(handle, model, doc_id, user_id=user_id, parse_hash=NEXT_PARSE)
    if commit:
        handle.commit_embeddings(doc_id, NEXT_PARSE, model, user_id=user_id)


def _ingested_with_status(
    handle: KBCollectionHandle, model: str, tmp_path: Path
) -> KBDocumentRowsSnapshot:
    _ingest(handle, model, "doc", ["a", "b"], tmp_path)
    _ingest(handle, model, "kept", ["k"], tmp_path)
    _ingest(_open(2), model, "doc", ["s"], tmp_path, user_id=2)
    handle.write_ingestion_status("doc", status="success", parse_hash=PARSE, user_id=1)
    return _snapshot(handle, ["doc"])


def test_a_restore_after_a_run_that_stopped_before_its_commit_deletes_its_rows(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    snapshot = _ingested_with_status(handle, model, tmp_path)
    handle.clear_ingestion_status("doc", user_id=1)
    _refresh(handle, model, "doc", ["x", "y"], commit=False)

    assert handle.restore_document_rows(snapshot, user_id=1, is_admin=False) == []

    assert _visible(client, model) == {"a", "b", "k", "s"}
    assert set(_rows(client, model)) == {"a", "b", "k", "s"}
    assert _ledger(handle) == (["doc", "kept"], 3 + 1)
    assert _status(handle, "doc") == "success"


def test_a_restore_after_a_run_that_committed_marks_the_missing_old_chunks(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    snapshot = _ingested_with_status(handle, model, tmp_path)
    _refresh(handle, model, "doc", ["x", "y"], commit=True)
    assert _visible(client, model) == {"x", "y", "k", "s"}

    assert handle.restore_document_rows(snapshot, user_id=1, is_admin=False) == ["doc"]

    assert set(_rows(client, model)) == {"k", "s"}
    assert _status(handle, "doc") == DocumentProcessingStatus.PARTIALLY_EMBEDDED.value
    (row,) = handle.load_ingestion_status(doc_id="doc", user_id=1)
    assert row["parse_hash"] == PARSE and "re-ingest" in row["message"]
    assert _ledger(handle) == (["doc", "kept"], 3 + 1)


def test_a_restore_marks_a_document_whose_old_set_lost_a_row(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    snapshot = _ingested_with_status(handle, model, tmp_path)
    client.delete(
        milvus_collection_name(model),
        filter="chunk_id == {id}",
        filter_params={"id": "b"},
    )
    _refresh(handle, model, "doc", ["x"], commit=False)

    assert handle.restore_document_rows(snapshot, user_id=1, is_admin=False) == ["doc"]

    assert _visible(client, model) == {"a", "k", "s"}
    assert _status(handle, "doc") == DocumentProcessingStatus.PARTIALLY_EMBEDDED.value


def test_a_restore_checks_the_latest_parse_not_older_chunk_sets_in_the_ledger(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a", "b"], tmp_path)
    _refresh(handle, model, "doc", ["c", "d"], commit=True)
    handle.write_ingestion_status(
        "doc", status="success", parse_hash=NEXT_PARSE, user_id=1
    )
    snapshot = _snapshot(handle, ["doc"])
    _chunks(handle, "doc", ["x"], parse_hash="ph-3")
    _embed(handle, model, "doc", parse_hash="ph-3")

    assert handle.restore_document_rows(snapshot, user_id=1, is_admin=False) == []

    assert _visible(client, model) == {"c", "d"}
    assert _status(handle, "doc") == "success"


def test_a_restore_keeps_vectors_of_older_parse_chunks_the_ledger_still_holds(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a", "b"], tmp_path)
    _chunks(handle, "doc", ["c"], parse_hash=NEXT_PARSE)
    handle.write_ingestion_status(
        "doc", status="success", parse_hash=NEXT_PARSE, user_id=1
    )
    snapshot = _snapshot(handle, ["doc"])
    _chunks(handle, "doc", ["x"], parse_hash="ph-3")
    _embed(handle, model, "doc", parse_hash="ph-3")

    assert handle.restore_document_rows(snapshot, user_id=1, is_admin=False) == ["doc"]

    assert _visible(client, model) == {"a", "b"}
    assert set(_rows(client, model)) == {"a", "b"}


def test_a_restore_counts_a_hidden_old_row_as_missing(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    snapshot = _ingested_with_status(handle, model, tmp_path)
    client.upsert(
        milvus_collection_name(model),
        [{"chunk_id": "b", "visible": False}],
        partial_update=True,
    )

    assert handle.restore_document_rows(snapshot, user_id=1, is_admin=False) == ["doc"]

    assert _visible(client, model) == {"a", "k", "s"}
    assert "b" in _rows(client, model)


def test_a_restore_marks_nothing_for_a_document_without_a_recorded_parse(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    _ingest(handle, model, "doc", ["a"], tmp_path)
    snapshot = _snapshot(handle, ["doc"])
    _refresh(handle, model, "doc", ["x"], commit=True)

    assert handle.restore_document_rows(snapshot, user_id=1, is_admin=False) == []

    assert _rows(client, model) == {}
    assert handle.load_ingestion_status(doc_id="doc", user_id=1) == []


def test_a_restore_leaves_other_documents_and_owners_rows_and_aligns_every_model(
    client: Any, model: str, other_model: str, tmp_path: Path
) -> None:
    handle = _open()
    snapshot = _ingested_with_status(handle, model, tmp_path)
    _refresh(handle, model, "doc", ["x"], commit=False)
    _refresh(handle, other_model, "doc", ["y"], commit=False)
    _refresh(_open(2), model, "doc", ["t"], commit=False, user_id=2)

    handle.restore_document_rows(snapshot, user_id=1, is_admin=False)

    assert set(_rows(client, model)) == {"a", "b", "k", "s", "t"}
    assert _rows(client, other_model) == {}


def test_a_restore_loads_a_collection_that_is_not_loaded(
    client: Any, model: str, tmp_path: Path
) -> None:
    handle = _open()
    snapshot = _ingested_with_status(handle, model, tmp_path)
    _refresh(handle, model, "doc", ["x"], commit=False)
    with released(client, model):
        assert handle.restore_document_rows(snapshot, user_id=1, is_admin=False) == []

    assert set(_rows(client, model)) == {"a", "b", "k", "s"}
