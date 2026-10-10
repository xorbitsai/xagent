"""Behavior contract every ``KBCollectionHandle`` engine must satisfy.

Tests drive the handle only through the abstract interface; an engine joins by
adding an ``ENGINES`` entry. On Milvus the cases in ``MILVUS_SKELETON`` (the ledger
and the query-vector check) run, the cases in ``MILVUS_ROWS`` run against the server
at ``MILVUS_URI``, the cascade cases in ``MILVUS_UNSUPPORTED`` are skipped because
Milvus does not support cascade cleanup by design, and every other case needs the
Milvus embedding snapshot, restore or cleanup: those are not implemented yet, so the
cases are strict xfails on ``NotImplementedError`` until they are. Deletes, rename
and the document-row restore are also checked through per-document vector counts,
which both engines serve without search. Milvus counts and searches are Bounded, so
a change shows after a moment (``_eventually``). Not covered here:

- async search: LanceDB async search returns no rows today;
- untagged main pointers and ``list_main_pointers``: on LanceDB the untagged
  ``IS NULL`` / ``= ''`` alternatives are passed as a tuple and ANDed, so they
  never match, and ``list_main_pointers`` raises ``MainPointerError``
  (#2858);
- main pointers under tenant deletes: ``main_pointers`` has no owner column, so
  a tenant-scoped document delete or document cascade on another owner's doc_id
  removes that owner's pointers (#2859). New engines should not take
  LanceDB as the reference for either main-pointer item;
- version candidates, promotion and the candidate-cleanup snapshots: covered on
  LanceDB by ``test_collection_handle_version.py``;
- count-based reads right after a parse, chunk or embeddings cascade: LanceDB
  keeps returning pre-delete values from ``collection_stats``, ``chunk_exists``,
  ``parse_exists`` (parse cascade) and ``read_chunks_needing_embedding()``'s
  ``total_count`` (parse or chunk cascade), because ``count_rows`` reads through
  the cached ``_get_table`` handle and only the document cascade invalidates it
  (#2860).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VectorValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkEmbeddingData,
    ParsedParagraph,
    RegisterDocumentRequest,
)
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    KBHandleProvider,
    milvus_collection_name,
)
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)

COLLECTION = "contract"
MODEL = "contract-model"
TAG = to_model_tag(MODEL)
PARSE = "ph-1"
PARSE_V2 = "ph-2"
CONFIG = "cfg-1"
VECTORS = {
    "kiwi apple": [1.0, 0.0, 0.0],
    "cherry plum": [0.0, 1.0, 0.0],
    "kiwi banana": [0.0, 0.0, 1.0],
    "kiwi grape": [0.6, 0.0, 0.8],
}
SEARCH_MODES = ("dense", "sparse", "hybrid")

OpenHandle = Callable[..., KBCollectionHandle]


def _opener(
    backend: KBStorageBackend, capabilities: KBBackendCapabilities
) -> OpenHandle:
    def open_handle(collection: str, user_id: int | None = None) -> KBCollectionHandle:
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
                backend=backend,
                capabilities=capabilities,
                collection_info=None,
            )
        )

    return open_handle


ENGINES: dict[str, OpenHandle] = {
    "lancedb": _opener(KBStorageBackend.LANCEDB, KBBackendCapabilities.lancedb()),
    "milvus": _opener(KBStorageBackend.MILVUS, KBBackendCapabilities.milvus()),
}
MILVUS_SKELETON = {
    "test_register_document_is_idempotent_and_owner_scoped",
    "test_parse_rows_round_trip_within_owner_scope",
    "test_ingestion_status_round_trip_and_snapshot",
    "test_ingestion_status_async_round_trip",
    "test_main_pointer_round_trip_and_snapshot",
    "test_rename_collection_status_moves_only_the_callers_rows",
    "test_rename_collection_metadata_moves_only_the_callers_config",
    "test_delete_collection_config_follows_tenant_scope",
    "test_validate_query_vector_rejects_malformed_vectors",
}
MILVUS_ROWS = {
    "test_rewriting_embeddings_does_not_duplicate_rows",
    "test_search_returns_only_rows_the_caller_may_see",
    "test_search_never_crosses_collections",
    "test_dense_search_ranks_nearest_first_with_unit_scores",
    "test_hybrid_results_carry_per_route_scores",
    "test_chunk_rows_round_trip_in_index_order",
    "test_chunks_needing_embedding_resume_after_partial_write",
    "test_stats_and_listings_follow_owner_scope",
    "test_delete_documents_data_removes_document_chunks_and_vectors",
    "test_tenant_delete_leaves_other_owners_rows",
    "test_delete_collection_data_empties_only_this_collection",
    "test_rename_collection_data_moves_rows_to_new_name",
    "test_document_rows_restore_drops_rows_written_after_capture",
    "test_a_deleted_document_is_not_found_by_any_search",
    "test_restored_and_deleted_rows_are_not_found_by_any_search",
}
MILVUS_UNSUPPORTED = {
    "test_cascade_deletes_its_scope_only_when_confirmed",
    "test_cascade_leaves_other_owners_rows",
}


def _engine_marks(engine: str, case: str) -> tuple[pytest.MarkDecorator, ...]:
    if engine != "milvus" or case in MILVUS_SKELETON:
        return ()
    if case in MILVUS_UNSUPPORTED:
        return (pytest.mark.skip(reason="Milvus does not support cascade cleanup"),)
    if case in MILVUS_ROWS:
        return (pytest.mark.milvus,)
    return (
        pytest.mark.milvus,
        pytest.mark.xfail(
            strict=True,
            raises=NotImplementedError,
            reason="needs the Milvus embedding snapshot, restore or cleanup, not implemented yet",
        ),
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "open_handle" in metafunc.fixturenames:
        metafunc.parametrize(
            "open_handle",
            [
                pytest.param(
                    engine, marks=_engine_marks(engine, metafunc.function.__name__)
                )
                for engine in sorted(ENGINES)
            ],
            indirect=True,
        )


@pytest.fixture
def open_handle(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> OpenHandle:
    if request.param == "milvus":
        monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")
        if uri := os.environ.get("MILVUS_URI"):
            from pymilvus import MilvusClient

            request.addfinalizer(
                lambda: MilvusClient(uri=uri).drop_collection(
                    milvus_collection_name(MODEL)
                )
            )
    return ENGINES[request.param]


def test_milvus_case_lists_name_real_cases() -> None:
    cases = {name for name in globals() if name.startswith("test_")}

    assert MILVUS_SKELETON | MILVUS_ROWS | MILVUS_UNSUPPORTED <= cases
    assert not MILVUS_SKELETON & MILVUS_ROWS
    assert not (MILVUS_SKELETON | MILVUS_ROWS) & MILVUS_UNSUPPORTED


def _write_chunks(
    handle: KBCollectionHandle,
    doc_id: str,
    texts: list[str],
    *,
    user_id: int,
    start: int = 0,
    parse_hash: str = PARSE,
) -> None:
    now = datetime.now(timezone.utc)
    document = handle.load_document(doc_id, is_admin=True)
    made_from = {"content_hash": document.content_hash} if document else {}
    chunks = [
        {
            "chunk_id": f"{doc_id}-c{index}",
            "index": index,
            "text": text,
            "created_at": now,
            "metadata": {"page": index + 1, **made_from},
        }
        for index, text in enumerate(texts, start=start)
    ]
    handle.write_chunks(doc_id, parse_hash, CONFIG, {}, chunks, user_id=user_id)


def _embed(
    handle: KBCollectionHandle,
    doc_id: str,
    *,
    user_id: int,
    limit: int | None = None,
    parse_hash: str = PARSE,
) -> list[ChunkEmbeddingData]:
    pending = handle.read_chunks_needing_embedding(
        doc_id, parse_hash, MODEL, user_id=user_id
    ).chunks
    embeddings = [
        ChunkEmbeddingData(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            parse_hash=chunk.parse_hash,
            model=MODEL,
            vector=VECTORS[chunk.text],
            text=chunk.text,
            chunk_hash=chunk.chunk_hash,
            metadata=chunk.metadata,
        )
        for chunk in sorted(pending, key=lambda chunk: chunk.index)[:limit]
    ]
    handle.write_embeddings(embeddings, user_id=user_id)
    if limit is None:
        handle.commit_embeddings(doc_id, parse_hash, MODEL, user_id=user_id)
    return embeddings


def _ingest(
    handle: KBCollectionHandle,
    collection: str,
    doc_id: str,
    texts: list[str],
    *,
    user_id: int,
    source_dir: Path,
) -> list[ChunkEmbeddingData]:
    source = source_dir / f"{collection}-{doc_id}.txt"
    source.write_text("\n".join(texts), encoding="utf-8")
    handle.register_document(
        RegisterDocumentRequest(
            collection=collection,
            source_path=str(source),
            doc_id=doc_id,
            user_id=user_id,
        )
    )
    _write_chunks(handle, doc_id, texts, user_id=user_id)
    return _embed(handle, doc_id, user_id=user_id)


@pytest.fixture
def seeded(open_handle: OpenHandle, tmp_path: Path) -> KBCollectionHandle:
    handle = open_handle(COLLECTION)
    _ingest(
        handle,
        COLLECTION,
        "doc-1",
        ["kiwi apple", "cherry plum"],
        user_id=1,
        source_dir=tmp_path,
    )
    _ingest(
        handle, COLLECTION, "doc-2", ["kiwi banana"], user_id=2, source_dir=tmp_path
    )
    return handle


def _search(
    handle: KBCollectionHandle,
    mode: str,
    *,
    query: str = "kiwi",
    user_id: int | None = None,
    is_admin: bool = True,
) -> dict[str, str]:
    vector = VECTORS["kiwi apple"]
    scope = {"top_k": 10, "user_id": user_id, "is_admin": is_admin}
    if mode == "dense":
        response = handle.search_dense(MODEL, vector, **scope)
    elif mode == "sparse":
        response = handle.search_sparse(MODEL, query, **scope)
    else:
        response = handle.search_hybrid(MODEL, query, vector, **scope)
    assert response.status == "success", response.warnings
    return {result.chunk_id: result.doc_id for result in response.results}


def _vector_docs(handle: KBCollectionHandle) -> dict[str, int]:
    """Visible vectors per document, counted without search."""
    counts = handle.count_rows_by_document(user_id=None, is_admin=True)
    vectors = {
        doc_id: sum(n for table, n in tables.items() if table.startswith("embeddings_"))
        for doc_id, tables in counts.items()
    }
    return {doc_id: n for doc_id, n in vectors.items() if n}


def _assert_vector_docs(handle: KBCollectionHandle, expected: dict[str, int]) -> None:
    assert _vector_docs(handle) == expected


def _eventually(check: Callable[[], None], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return check()
        except AssertionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


HANDLE_CALLER_MODULES = (
    "xagent.core.tools.core.RAG_tools.kb.coordinator",
    "xagent.core.tools.core.RAG_tools.kb.legacy_step_compatibility",
    "xagent.core.tools.core.RAG_tools.kb.parse_display_compatibility",
    "xagent.core.tools.core.RAG_tools.kb.vector_storage_compatibility",
    "xagent.core.tools.core.RAG_tools.chunk.chunk_document",
    "xagent.core.tools.core.RAG_tools.parse.parse_display",
    "xagent.core.tools.core.RAG_tools.parse.parse_document",
)
HANDLE_OPENERS = {"open_collection", "open_collection_sync", "_open_collection_handle"}


def _opens_handle(node: ast.AST) -> bool:
    if isinstance(node, ast.Await):
        node = node.value
    return (
        isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) in HANDLE_OPENERS
    )


def _handle_attributes(module_name: str) -> set[str]:
    tree = ast.parse(inspect.getsource(importlib.import_module(module_name)))
    found: set[str] = set()
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = [*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs]
        handles = {
            param.arg
            for param in params
            if param.annotation is not None
            and "KBCollectionHandle" in ast.unparse(param.annotation)
        }
        for node in ast.walk(func):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and (
                node.value is not None and _opens_handle(node.value)
            ):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                handles |= {t.id for t in targets if isinstance(t, ast.Name)}
        for node in ast.walk(func):
            if isinstance(node, ast.Attribute) and (
                (isinstance(node.value, ast.Name) and node.value.id in handles)
                or _opens_handle(node.value)
            ):
                found.add(node.attr)
    return found


def test_interface_declares_every_handle_attribute_its_callers_use() -> None:
    """Supplements mypy, which is the primary check that callers stay on the ABC."""
    used = set().union(*map(_handle_attributes, HANDLE_CALLER_MODULES))
    assert used
    assert {name for name in used if not hasattr(KBCollectionHandle, name)} == set()


def test_register_document_is_idempotent_and_owner_scoped(
    open_handle: OpenHandle, tmp_path: Path
) -> None:
    handle = open_handle(COLLECTION)
    source = tmp_path / "a.txt"
    source.write_text("hello", encoding="utf-8")
    request = RegisterDocumentRequest(
        collection=COLLECTION, source_path=str(source), doc_id="doc-1", user_id=1
    )

    assert handle.register_document(request).created is True
    assert handle.register_document(request).created is False

    loaded = handle.load_document("doc-1", user_id=1)
    assert loaded is not None and loaded.doc_id == "doc-1"
    assert handle.load_document("doc-1", user_id=2) is None
    assert handle.load_document("doc-1", is_admin=True) is not None
    listed = handle.list_documents(user_id=1).documents
    assert [record.doc_id for record in listed] == ["doc-1"]
    assert handle.list_documents(user_id=2).documents == []

    assert handle.delete_document_record("doc-1", user_id=1) == 1
    assert handle.load_document("doc-1", is_admin=True) is None


def test_parse_rows_round_trip_within_owner_scope(open_handle: OpenHandle) -> None:
    handle = open_handle(COLLECTION)
    paragraph = ParsedParagraph(text="first paragraph", metadata={"page": 1})
    handle.write_parse("doc-1", PARSE, "default", {}, [paragraph], user_id=1)

    assert handle.parse_exists("doc-1", PARSE, user_id=1)
    assert not handle.parse_exists("doc-1", "other-hash", user_id=1)
    assert not handle.parse_exists("doc-1", PARSE, user_id=2)
    assert handle.read_parse_paragraphs("doc-1", PARSE, user_id=1) == [paragraph]
    assert handle.read_parse_paragraph_dicts("doc-1", PARSE, user_id=1) == [
        {"text": "first paragraph", "metadata": {"page": 1}}
    ]
    latest = handle.read_latest_parse_record("doc-1", user_id=1)
    assert latest is not None and latest.parse_hash == PARSE


def test_chunk_rows_round_trip_in_index_order(seeded: KBCollectionHandle) -> None:
    assert seeded.chunk_exists("doc-1", PARSE, CONFIG, user_id=1)
    assert not seeded.chunk_exists("doc-1", PARSE, "other-config", user_id=1)
    assert not seeded.chunk_exists("doc-1", PARSE, CONFIG, user_id=2)

    chunks = seeded.read_existing_chunks("doc-1", PARSE, CONFIG, user_id=1)
    assert [(c["chunk_id"], c["text"]) for c in chunks] == [
        ("doc-1-c0", "kiwi apple"),
        ("doc-1-c1", "cherry plum"),
    ]
    assert [
        {k: v for k, v in c["metadata"].items() if k != "content_hash"} for c in chunks
    ] == [{"page": 1}, {"page": 2}]


def test_chunks_needing_embedding_resume_after_partial_write(
    open_handle: OpenHandle,
) -> None:
    handle = open_handle(COLLECTION)
    _write_chunks(
        handle, "doc-1", ["kiwi apple", "cherry plum", "kiwi banana"], user_id=1
    )

    first = _embed(handle, "doc-1", user_id=1, limit=1)
    pending = handle.read_chunks_needing_embedding("doc-1", PARSE, MODEL, user_id=1)
    assert (pending.total_count, pending.pending_count) == (3, 2)
    assert first[0].chunk_id not in {chunk.chunk_id for chunk in pending.chunks}

    _embed(handle, "doc-1", user_id=1)
    done = handle.read_chunks_needing_embedding("doc-1", PARSE, MODEL, user_id=1)
    assert (done.total_count, done.pending_count, done.chunks) == (3, 0, [])


def test_rewriting_embeddings_does_not_duplicate_rows(
    open_handle: OpenHandle, tmp_path: Path
) -> None:
    handle = open_handle(COLLECTION)
    embeddings = _ingest(
        handle,
        COLLECTION,
        "doc-1",
        ["kiwi apple", "cherry plum"],
        user_id=1,
        source_dir=tmp_path,
    )

    handle.write_embeddings(embeddings, user_id=1)
    handle.commit_embeddings("doc-1", PARSE, MODEL, user_id=1)

    assert handle.collection_stats(None, True)["embeddings"] == 2
    assert set(_search(handle, "dense")) == {"doc-1-c0", "doc-1-c1"}


@pytest.mark.parametrize(
    "vector", [[], [float("nan"), 1.0], [float("inf")], ["x"], (0.1, 0.2)]
)
def test_validate_query_vector_rejects_malformed_vectors(
    open_handle: OpenHandle, vector: list[float]
) -> None:
    handle = open_handle(COLLECTION)
    handle.validate_query_vector([0.1, 0.2])
    with pytest.raises(VectorValidationError):
        handle.validate_query_vector(vector)


@pytest.mark.parametrize("mode", SEARCH_MODES)
@pytest.mark.parametrize(
    ("user_id", "is_admin", "expected"),
    [
        (1, False, {"doc-1"}),
        (2, False, {"doc-2"}),
        (3, False, set()),
        (None, True, {"doc-1", "doc-2"}),
    ],
)
def test_search_returns_only_rows_the_caller_may_see(
    seeded: KBCollectionHandle,
    mode: str,
    user_id: int | None,
    is_admin: bool,
    expected: set[str],
) -> None:
    hits = _search(seeded, mode, user_id=user_id, is_admin=is_admin)
    assert set(hits.values()) == expected


@pytest.mark.parametrize("mode", SEARCH_MODES)
def test_search_never_crosses_collections(
    seeded: KBCollectionHandle, open_handle: OpenHandle, tmp_path: Path, mode: str
) -> None:
    other = open_handle("other")
    _ingest(other, "other", "doc-x", ["kiwi grape"], user_id=1, source_dir=tmp_path)

    assert set(_search(seeded, mode).values()) == {"doc-1", "doc-2"}
    assert set(_search(other, mode).values()) == {"doc-x"}


def test_dense_search_ranks_nearest_first_with_unit_scores(
    seeded: KBCollectionHandle,
) -> None:
    response = seeded.search_dense(
        MODEL, VECTORS["cherry plum"], top_k=10, is_admin=True
    )

    top = response.results[0]
    assert (top.chunk_id, top.doc_id, top.text) == ("doc-1-c1", "doc-1", "cherry plum")
    assert top.parse_hash == PARSE
    assert {k: v for k, v in top.metadata.items() if k != "content_hash"} == {"page": 2}
    scores = [result.score for result in response.results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_hybrid_results_carry_per_route_scores(seeded: KBCollectionHandle) -> None:
    response = seeded.search_hybrid(
        MODEL, "apple", VECTORS["kiwi apple"], top_k=10, is_admin=True
    )

    top = response.results[0]
    assert top.chunk_id == "doc-1-c0"
    assert top.vector_score is not None and top.fts_score is not None


def test_delete_documents_data_removes_document_chunks_and_vectors(
    seeded: KBCollectionHandle,
) -> None:
    seeded.delete_documents_data(["doc-1"], user_id=None, is_admin=True)

    def removed() -> None:
        assert _vector_docs(seeded) == {"doc-2": 1}
        stats = seeded.collection_stats(None, True)
        assert (stats["documents"], stats["chunks"], stats["embeddings"]) == (1, 1, 1)

    _eventually(removed)
    assert seeded.list_collection_documents(None, True) == ["doc-2"]
    gone = seeded.read_chunks_needing_embedding("doc-1", PARSE, MODEL, is_admin=True)
    assert gone.total_count == 0


def test_tenant_delete_leaves_other_owners_rows(seeded: KBCollectionHandle) -> None:
    seeded.delete_documents_data(["doc-1", "doc-2"], user_id=2, is_admin=False)

    assert seeded.list_collection_documents(None, True) == ["doc-1"]
    _eventually(lambda: _assert_vector_docs(seeded, {"doc-1": 2}))


def test_delete_collection_data_empties_only_this_collection(
    seeded: KBCollectionHandle, open_handle: OpenHandle, tmp_path: Path
) -> None:
    other = open_handle("other")
    _ingest(other, "other", "doc-x", ["kiwi grape"], user_id=1, source_dir=tmp_path)

    seeded.delete_collection_data(user_id=None, is_admin=True)

    def emptied() -> None:
        stats = seeded.collection_stats(None, True)
        assert (stats["documents"], stats["chunks"], stats["embeddings"]) == (0, 0, 0)
        assert _vector_docs(seeded) == {}

    _eventually(emptied)
    other_stats = other.collection_stats(None, True)
    assert (other_stats["documents"], other_stats["embeddings"]) == (1, 1)


def test_rename_collection_data_moves_rows_to_new_name(
    seeded: KBCollectionHandle, open_handle: OpenHandle
) -> None:
    assert seeded.rename_collection_data("renamed", None, True) == []

    renamed = open_handle("renamed")
    _assert_vector_docs(renamed, {"doc-1": 2, "doc-2": 1})
    _assert_vector_docs(seeded, {})
    assert renamed.list_collection_documents(None, True) == ["doc-1", "doc-2"]
    assert seeded.count_documents(None, True) == 0


def test_stats_and_listings_follow_owner_scope(seeded: KBCollectionHandle) -> None:
    def counts(user_id: int | None, is_admin: bool) -> tuple[int, int, int]:
        stats = seeded.collection_stats(user_id, is_admin)
        return stats["documents"], stats["chunks"], stats["embeddings"]

    assert counts(None, True) == (2, 3, 3)
    assert counts(1, False) == (1, 2, 2)
    assert counts(3, False) == (0, 0, 0)
    assert seeded.count_documents(2, False) == 1
    assert seeded.list_collection_documents(1, False) == ["doc-1"]
    assert seeded.list_collection_documents(None, True) == ["doc-1", "doc-2"]


def test_document_rows_restore_drops_rows_written_after_capture(
    seeded: KBCollectionHandle,
) -> None:
    snapshot = seeded.capture_document_rows(["doc-1"], user_id=1, is_admin=False)
    _write_chunks(seeded, "doc-1", ["kiwi grape"], user_id=1, start=2)
    _embed(seeded, "doc-1", user_id=1)
    _eventually(lambda: _assert_vector_docs(seeded, {"doc-1": 3, "doc-2": 1}))

    assert seeded.restore_document_rows(snapshot, user_id=1, is_admin=False) == []

    _eventually(lambda: _assert_vector_docs(seeded, {"doc-1": 2, "doc-2": 1}))
    assert seeded.collection_stats(None, True)["chunks"] == 3


def _assert_every_search_finds(handle: KBCollectionHandle, expected: set[str]) -> None:
    def found() -> None:
        for mode in SEARCH_MODES:
            assert set(_search(handle, mode).values()) == expected, mode

    _eventually(found)


def test_a_deleted_document_is_not_found_by_any_search(
    seeded: KBCollectionHandle,
) -> None:
    seeded.delete_documents_data(["doc-1"], user_id=None, is_admin=True)

    _assert_every_search_finds(seeded, {"doc-2"})


def test_restored_and_deleted_rows_are_not_found_by_any_search(
    seeded: KBCollectionHandle,
) -> None:
    def sparse_grape() -> set[str]:
        return set(_search(seeded, "sparse", query="grape"))

    def grape_is(expected: set[str]) -> None:
        assert sparse_grape() == expected

    snapshot = seeded.capture_document_rows(["doc-1"], user_id=1, is_admin=False)
    _write_chunks(seeded, "doc-1", ["kiwi grape"], user_id=1, start=2)
    _embed(seeded, "doc-1", user_id=1)
    _eventually(lambda: grape_is({"doc-1-c2"}))

    seeded.restore_document_rows(snapshot, user_id=1, is_admin=False)
    _eventually(lambda: grape_is(set()))
    _assert_every_search_finds(seeded, {"doc-1", "doc-2"})

    seeded.delete_documents_data(["doc-1", "doc-2"], user_id=2, is_admin=False)
    _assert_every_search_finds(seeded, {"doc-1"})

    seeded.delete_collection_data(user_id=None, is_admin=True)
    _assert_every_search_finds(seeded, set())


def test_embedding_snapshot_restores_deleted_rows(seeded: KBCollectionHandle) -> None:
    snapshot = seeded.snapshot_embeddings("doc-1", PARSE, user_id=1)
    assert snapshot is not None

    assert seeded.delete_created_embeddings("doc-1", PARSE, user_id=1) == 2
    assert set(_search(seeded, "dense").values()) == {"doc-2"}
    pending = seeded.read_chunks_needing_embedding("doc-1", PARSE, MODEL, user_id=1)
    assert pending.pending_count == 2

    seeded.restore_embeddings(snapshot)
    assert set(_search(seeded, "dense").values()) == {"doc-1", "doc-2"}


def test_ingestion_status_round_trip_and_snapshot(open_handle: OpenHandle) -> None:
    handle = open_handle(COLLECTION)

    def statuses() -> list[str]:
        rows = handle.load_ingestion_status(doc_id="doc-1", user_id=1)
        return [row["status"] for row in rows]

    handle.write_ingestion_status(
        "doc-1", status="running", parse_hash=PARSE, user_id=1
    )
    assert statuses() == ["running"]
    assert handle.load_ingestion_status(doc_id="doc-1", user_id=2) == []

    snapshot = handle.capture_status_snapshot("doc-1", user_id=1)
    handle.write_ingestion_status(
        "doc-1", status="failed", parse_hash="ph-2", user_id=1
    )
    assert statuses() == ["failed"]
    handle.restore_status_snapshot("doc-1", snapshot, user_id=1)
    (restored,) = handle.load_ingestion_status(doc_id="doc-1", user_id=1)
    assert (restored["status"], restored["parse_hash"]) == ("running", PARSE)

    handle.restore_status_snapshot("doc-1", [], user_id=1)
    assert statuses() == []
    handle.write_ingestion_status("doc-1", status="running", user_id=1)
    handle.clear_status_snapshot("doc-1", user_id=1)
    assert statuses() == []


async def test_ingestion_status_async_round_trip(open_handle: OpenHandle) -> None:
    handle = open_handle(COLLECTION)

    await handle.write_ingestion_status_async("doc-1", status="running", user_id=1)
    rows = await handle.load_ingestion_status_async(doc_id="doc-1", user_id=1)
    assert [row["status"] for row in rows] == ["running"]
    assert handle.load_ingestion_status(doc_id="doc-1", user_id=1) == rows

    await handle.clear_ingestion_status_async("doc-1", user_id=1)
    assert await handle.load_ingestion_status_async(doc_id="doc-1", user_id=1) == []


def test_main_pointer_round_trip_and_snapshot(open_handle: OpenHandle) -> None:
    handle = open_handle(COLLECTION)

    def technical_id() -> str | None:
        pointer = handle.get_main_pointer("doc-1", "embed", TAG)
        return None if pointer is None else pointer["technical_id"]

    absent = handle.capture_main_pointer_snapshot("doc-1", "embed", TAG)
    assert absent.pointer is None
    handle.set_main_pointer("doc-1", "embed", "v1", "hash-1", TAG, "tester")
    assert technical_id() == "hash-1"

    snapshot = handle.capture_main_pointer_snapshot("doc-1", "embed", TAG)
    handle.set_main_pointer("doc-1", "embed", "v2", "hash-2", TAG)
    assert technical_id() == "hash-2"
    assert handle.restore_main_pointer_snapshot(snapshot) is True
    assert technical_id() == "hash-1"

    assert handle.restore_main_pointer_snapshot(absent) is True
    assert technical_id() is None
    handle.set_main_pointer("doc-1", "embed", "v1", "hash-1", TAG)
    assert handle.delete_main_pointer("doc-1", "embed", TAG) is True
    assert handle.delete_main_pointer("doc-1", "embed", TAG) is False
    assert technical_id() is None


@pytest.fixture
def two_versions(seeded: KBCollectionHandle) -> KBCollectionHandle:
    for parse_hash in (PARSE, PARSE_V2):
        paragraphs = [ParsedParagraph(text=parse_hash)]
        seeded.write_parse("doc-1", parse_hash, "default", {}, paragraphs, user_id=1)
    _write_chunks(
        seeded, "doc-1", ["kiwi grape"], user_id=1, start=5, parse_hash=PARSE_V2
    )
    _embed(seeded, "doc-1", user_id=1, parse_hash=PARSE_V2)
    return seeded


def _doc1_rows(handle: KBCollectionHandle) -> tuple[list[str], ...]:
    versions = (PARSE, PARSE_V2)
    parses = [
        p for p in versions if handle.read_parse_paragraphs("doc-1", p, is_admin=True)
    ]
    chunks = sorted(
        chunk["chunk_id"]
        for p in versions
        for chunk in handle.read_existing_chunks("doc-1", p, CONFIG, is_admin=True)
    )
    vectors = sorted(c for c, d in _search(handle, "dense").items() if d == "doc-1")
    return parses, chunks, vectors


ALL_CHUNKS = ["doc-1-c0", "doc-1-c1", "doc-1-c5"]
CASCADES = [
    pytest.param("document", {}, ([], [], []), id="document"),
    pytest.param(
        "parse",
        {"new_parse_hash": PARSE_V2},
        ([PARSE_V2], ["doc-1-c5"], ["doc-1-c5"]),
        id="parse",
    ),
    pytest.param(
        "chunk",
        {"new_parse_hash": PARSE_V2},
        ([PARSE, PARSE_V2], ["doc-1-c5"], ["doc-1-c5"]),
        id="chunk",
    ),
    pytest.param(
        "embeddings",
        {"model_tag": TAG},
        ([PARSE, PARSE_V2], ALL_CHUNKS, []),
        id="embed",
    ),
]
DEDICATED_CASCADES = {
    "document": "cleanup_document_cascade",
    "parse": "cleanup_parse_cascade",
    "chunk": "cleanup_chunk_cascade",
    "embeddings": "cleanup_embed_cascade",
}


def _cascade(
    handle: KBCollectionHandle, scope: str, via: str, **kwargs: object
) -> dict[str, int]:
    if via == "scope":
        return handle.cleanup_cascade("doc-1", scope, **kwargs)
    return getattr(handle, DEDICATED_CASCADES[scope])("doc-1", **kwargs)


@pytest.mark.parametrize("via", ["dedicated", "scope"])
@pytest.mark.parametrize(("scope", "kwargs", "expected"), CASCADES)
def test_cascade_deletes_its_scope_only_when_confirmed(
    two_versions: KBCollectionHandle,
    via: str,
    scope: str,
    kwargs: dict[str, object],
    expected: tuple[list[str], ...],
) -> None:
    before = _doc1_rows(two_versions)
    assert before == ([PARSE, PARSE_V2], ALL_CHUNKS, ALL_CHUNKS)

    preview = _cascade(two_versions, scope, via, **kwargs)
    _cascade(two_versions, scope, via, preview_only=False, **kwargs)
    assert _doc1_rows(two_versions) == before

    deleted = _cascade(
        two_versions, scope, via, preview_only=False, confirm=True, **kwargs
    )
    assert deleted == preview
    assert _doc1_rows(two_versions) == expected
    assert "doc-2-c0" in _search(two_versions, "dense")


@pytest.mark.parametrize("via", ["dedicated", "scope"])
@pytest.mark.parametrize(("scope", "kwargs", "expected"), CASCADES)
def test_cascade_leaves_other_owners_rows(
    two_versions: KBCollectionHandle,
    via: str,
    scope: str,
    kwargs: dict[str, object],
    expected: tuple[list[str], ...],
) -> None:
    before = _doc1_rows(two_versions)
    _cascade(
        two_versions,
        scope,
        via,
        user_id=2,
        is_admin=False,
        preview_only=False,
        confirm=True,
        **kwargs,
    )
    assert _doc1_rows(two_versions) == before


def test_operation_cleanup_deletes_only_the_named_vectors(
    seeded: KBCollectionHandle, open_handle: OpenHandle
) -> None:
    target = {
        "doc_id": "doc-1",
        "parse_hash": PARSE,
        "chunk_ids": ["doc-1-c0"],
        "model_tag": TAG,
    }
    stranger = open_handle(COLLECTION, user_id=2)
    blocked = stranger.cleanup_embeddings_for_operation(
        preview_only=False, confirm=True, **target
    )
    assert blocked.deleted_count == 0

    preview = seeded.cleanup_embeddings_for_operation(**target)
    assert (preview.status, preview.deleted_count) == ("planned", 1)
    assert "doc-1-c0" in _search(seeded, "dense")

    done = seeded.cleanup_embeddings_for_operation(
        preview_only=False, confirm=True, **target
    )
    assert (done.status, done.deleted_count) == ("complete", 1)
    assert set(_search(seeded, "dense")) == {"doc-1-c1", "doc-2-c0"}
    assert len(seeded.read_existing_chunks("doc-1", PARSE, CONFIG, user_id=1)) == 2


def test_rename_collection_status_moves_only_the_callers_rows(
    open_handle: OpenHandle,
) -> None:
    handle = open_handle(COLLECTION)
    handle.write_ingestion_status("doc-1", status="running", user_id=1)
    handle.write_ingestion_status("doc-2", status="running", user_id=2)

    assert handle.rename_collection_status("renamed", 2, False) == []

    def doc_ids(collection: str) -> list[str]:
        rows = open_handle(collection).load_ingestion_status(is_admin=True)
        return [row["doc_id"] for row in rows]

    assert doc_ids("renamed") == ["doc-2"]
    assert doc_ids(COLLECTION) == ["doc-1"]


async def test_rename_collection_metadata_moves_only_the_callers_config(
    open_handle: OpenHandle,
) -> None:
    store = get_metadata_store()
    for owner in (1, 2):
        await store.save_collection_config(COLLECTION, f'{{"owner": {owner}}}', owner)

    await open_handle(COLLECTION).rename_collection_metadata("renamed", 2, False)

    assert store.list_collection_config_owner_ids(COLLECTION) == {1}
    assert store.list_collection_config_owner_ids("renamed") == {2}
    assert await store.get_collection_config("renamed", 2) == '{"owner": 2}'


async def test_delete_collection_config_follows_tenant_scope(
    open_handle: OpenHandle,
) -> None:
    store = get_metadata_store()
    for owner in (1, 2):
        await store.save_collection_config(COLLECTION, "{}", owner)

    tenant = open_handle(COLLECTION, user_id=2)
    assert await tenant.delete_collection_config(tenant_only=True) == 1
    assert store.list_collection_config_owner_ids(COLLECTION) == {1}

    assert await open_handle(COLLECTION).delete_collection_config() == 1
    assert store.list_collection_config_owner_ids(COLLECTION) == set()
