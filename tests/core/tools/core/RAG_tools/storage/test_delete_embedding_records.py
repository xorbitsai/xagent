"""Tests for the row-only multi-table ``delete_embedding_records`` store
primitive (#510).

This primitive deletes ONLY ``embeddings_{model_tag}`` rows for a document
(optionally narrowed by parse_hash / chunk_ids / model_tag). It must enumerate
per-model embedding tables, not cascade into documents/parses/chunks, be
idempotent, preserve document-scoped tenant safety, and span all embedding
tables when ``model_tag`` is ``None``.

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

from datetime import datetime, timezone

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store


def _doc_row(collection: str, doc_id: str, user_id=None) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "file_id": None,
        "source_path": f"/uploads/{doc_id}.txt",
        "file_type": "txt",
        "content_hash": "a" * 64,
        "uploaded_at": datetime.now(timezone.utc),
        "title": None,
        "language": None,
        "user_id": user_id,
    }


def _parse_row(collection: str, doc_id: str, parse_hash: str, user_id=None) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "parse_hash": parse_hash,
        "parser": "test_parser",
        "created_at": datetime.now(timezone.utc),
        "params_json": "{}",
        "parsed_content": "parsed text",
        "user_id": user_id,
    }


def _chunk_row(
    collection: str, doc_id: str, parse_hash: str, chunk_id: str, user_id=None
) -> dict:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "parse_hash": parse_hash,
        "chunk_id": chunk_id,
        "index": 0,
        "text": "chunk text",
        "page_number": None,
        "section": None,
        "anchor": None,
        "json_path": None,
        "chunk_hash": "ch" + chunk_id,
        "config_hash": "cfg1",
        "created_at": datetime.now(timezone.utc),
        "metadata": "{}",
        "user_id": user_id,
    }


def _embedding_row(
    collection: str,
    doc_id: str,
    parse_hash: str,
    chunk_id: str,
    model: str,
    *,
    vector=None,
    user_id=None,
) -> dict:
    vec = vector or [0.1, 0.2]
    return {
        "collection": collection,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "parse_hash": parse_hash,
        "model": model,
        "vector": vec,
        "vector_dimension": len(vec),
        "text": f"text-{chunk_id}",
        "chunk_hash": "ch" + chunk_id,
        "created_at": datetime.now(timezone.utc),
        "metadata": "{}",
        "user_id": user_id,
    }


def _count_embeddings(model: str, collection: str) -> int:
    store = get_vector_index_store()
    return store.count_rows(
        f"embeddings_{to_model_tag(model)}", {"collection": collection}, is_admin=True
    )


class TestDeleteEmbeddingRecords:
    def test_deletes_only_targeted_model_table(self) -> None:
        store = get_vector_index_store()
        store.upsert_documents([_doc_row("coll", "d1")])
        store.upsert_parses([_parse_row("coll", "d1", "h1")])
        store.upsert_chunks([_chunk_row("coll", "d1", "h1", "c0")])
        store.upsert_embeddings("m1", [_embedding_row("coll", "d1", "h1", "c0", "m1")])
        store.upsert_embeddings("m2", [_embedding_row("coll", "d1", "h1", "c0", "m2")])

        deleted = store.delete_embedding_records(
            "coll", "d1", parse_hash="h1", model_tag="m1", user_id=None, is_admin=True
        )

        assert deleted == 1
        assert _count_embeddings("m1", "coll") == 0
        # The other model table is untouched.
        assert _count_embeddings("m2", "coll") == 1
        # No cascade into documents/parses/chunks.
        assert store.count_rows("documents", {"collection": "coll"}, is_admin=True) == 1
        assert store.count_rows("parses", {"collection": "coll"}, is_admin=True) == 1
        assert store.count_rows("chunks", {"collection": "coll"}, is_admin=True) == 1

    def test_model_tag_none_spans_all_embedding_tables(self) -> None:
        store = get_vector_index_store()
        store.upsert_embeddings("m1", [_embedding_row("coll", "d1", "h1", "c0", "m1")])
        store.upsert_embeddings("m2", [_embedding_row("coll", "d1", "h1", "c0", "m2")])

        deleted = store.delete_embedding_records(
            "coll", "d1", user_id=None, is_admin=True
        )

        assert deleted == 2
        assert _count_embeddings("m1", "coll") == 0
        assert _count_embeddings("m2", "coll") == 0

    def test_narrows_by_chunk_ids(self) -> None:
        store = get_vector_index_store()
        store.upsert_embeddings(
            "m1",
            [
                _embedding_row("coll", "d1", "h1", "c0", "m1"),
                _embedding_row("coll", "d1", "h1", "c1", "m1"),
            ],
        )

        deleted = store.delete_embedding_records(
            "coll", "d1", chunk_ids=["c0"], model_tag="m1", user_id=None, is_admin=True
        )

        assert deleted == 1
        assert _count_embeddings("m1", "coll") == 1

    def test_idempotent_and_missing_table_returns_zero(self) -> None:
        store = get_vector_index_store()
        # No embeddings table yet.
        assert (
            store.delete_embedding_records(
                "coll", "d1", model_tag="m1", user_id=None, is_admin=True
            )
            == 0
        )
        store.upsert_embeddings("m1", [_embedding_row("coll", "d1", "h1", "c0", "m1")])
        assert (
            store.delete_embedding_records(
                "coll", "d1", model_tag="m1", user_id=None, is_admin=True
            )
            == 1
        )
        assert (
            store.delete_embedding_records(
                "coll", "d1", model_tag="m1", user_id=None, is_admin=True
            )
            == 0
        )

    def test_user_scoping_blocks_other_tenant(self) -> None:
        store = get_vector_index_store()
        store.upsert_embeddings(
            "m1", [_embedding_row("coll", "d1", "h1", "c0", "m1", user_id=5)]
        )

        assert (
            store.delete_embedding_records(
                "coll", "d1", model_tag="m1", user_id=6, is_admin=False
            )
            == 0
        )
        assert _count_embeddings("m1", "coll") == 1
        assert (
            store.delete_embedding_records(
                "coll", "d1", model_tag="m1", user_id=5, is_admin=False
            )
            == 1
        )

    def test_collection_scoped(self) -> None:
        store = get_vector_index_store()
        store.upsert_embeddings("m1", [_embedding_row("coll", "d1", "h1", "c0", "m1")])
        store.upsert_embeddings("m1", [_embedding_row("other", "d1", "h1", "c0", "m1")])

        deleted = store.delete_embedding_records(
            "coll", "d1", model_tag="m1", user_id=None, is_admin=True
        )

        assert deleted == 1
        assert _count_embeddings("m1", "coll") == 0
        assert _count_embeddings("m1", "other") == 1
