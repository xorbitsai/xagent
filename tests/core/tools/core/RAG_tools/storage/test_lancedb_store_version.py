"""Tests for LanceDBVectorIndexStore.list_version_candidate_rows (#513 Task 4)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    """Provide an isolated LanceDBVectorIndexStore backed by a temp directory."""
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "lancedb"))
    # Force fresh store instance via the factory reset path.
    from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory

    StorageFactory.get_factory().reset_all()
    store = get_vector_index_store()
    yield store
    StorageFactory.get_factory().reset_all()


def _make_parse_row(
    collection: str,
    doc_id: str,
    parse_hash: str,
    parse_method: str = "unstructured",
    parser: str = "local:UnstructuredParser@v1",
) -> dict:
    """Build a minimal parse row matching the actual parses table schema."""
    return {
        "collection": collection,
        "doc_id": doc_id,
        "parse_hash": parse_hash,
        "parser": parser,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "params_json": json.dumps({"parse_method": parse_method}),
        "parsed_content": "test content",
        "user_id": None,
    }


def _make_chunk_row(
    collection: str,
    doc_id: str,
    parse_hash: str,
    chunk_id: str,
    text: str = "Hello world",
) -> dict:
    """Build a minimal chunk row matching the actual chunks table schema."""
    return {
        "collection": collection,
        "doc_id": doc_id,
        "parse_hash": parse_hash,
        "chunk_id": chunk_id,
        "index": 0,
        "text": text,
        "page_number": None,
        "section": None,
        "anchor": None,
        "json_path": None,
        "chunk_hash": "ch_" + chunk_id,
        "config_hash": "cfg1",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "metadata": "{}",
        "user_id": None,
    }


class TestListVersionCandidateRowsStore:
    """Tests for VectorIndexStore.list_version_candidate_rows – store-level (real LanceDB)."""

    def test_parse_candidate_row_shape(self, isolated_store):
        """Seeded parse row produces a candidate with expected dict shape + semantic_id format."""
        store = isolated_store
        collection = "coll1"
        doc_id = "doc1"
        parse_hash = "abcdef1234567890"

        store.upsert_parses(
            [
                _make_parse_row(
                    collection, doc_id, parse_hash, parse_method="unstructured"
                )
            ]
        )

        rows = store.list_version_candidate_rows(collection, doc_id, "parse")
        assert len(rows) == 1
        row = rows[0]

        # Dict shape matches test_list_candidates.py:101,161,228
        assert "semantic_id" in row
        assert "technical_id" in row
        assert "params_brief" in row
        assert "stats" in row
        assert "state" in row
        assert "created_at" in row
        assert "operator" in row

        # technical_id is the parse_hash
        assert row["technical_id"] == parse_hash

        # semantic_id format: parse_{method}_{hash8}
        # Note: parse_method is not a direct column in the parses table (it lives in
        # params_json), so _lc_generate_semantic_id receives "unknown" for method.
        assert row["semantic_id"].startswith("parse_")
        assert row["semantic_id"].endswith(parse_hash[:8])

        # state is "candidate"
        assert row["state"] == "candidate"

    def test_chunk_candidate_row_shape(self, isolated_store):
        """Seeded chunk rows produce candidate with expected shape (grouped by parse_hash)."""
        store = isolated_store
        collection = "coll2"
        doc_id = "doc2"
        parse_hash = "aabbcc1234567890"

        store.upsert_chunks(
            [
                _make_chunk_row(
                    collection, doc_id, parse_hash, "chunk1", "Hello world"
                ),
                _make_chunk_row(
                    collection, doc_id, parse_hash, "chunk2", "Goodbye world"
                ),
            ]
        )

        rows = store.list_version_candidate_rows(collection, doc_id, "chunk")
        assert len(rows) == 1  # Grouped by parse_hash
        row = rows[0]

        assert row["technical_id"] == parse_hash
        # semantic_id format: chunk_{strategy}_{size}_{hash8}
        assert row["semantic_id"].startswith("chunk_")
        assert parse_hash[:8] in row["semantic_id"]

        assert row["state"] == "candidate"
        assert "chunks_count" in row["stats"]
        assert row["stats"]["chunks_count"] == 2

    def test_embed_candidate_row_shape(self, isolated_store):
        """Seeded embedding rows produce candidate with expected shape."""
        store = isolated_store
        collection = "coll3"
        doc_id = "doc3"
        parse_hash = "eeff1234abcd5678"
        model_tag = "bge_large"

        store.upsert_embeddings(
            model_tag,
            [
                {
                    "collection": collection,
                    "doc_id": doc_id,
                    "parse_hash": parse_hash,
                    "chunk_id": "chunk1",
                    "model": "BAAI/bge-large-zh-v1.5",
                    "vector": [0.1, 0.2, 0.3],
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                }
            ],
        )

        rows = store.list_version_candidate_rows(
            collection, doc_id, "embed", model_tag=model_tag
        )
        assert len(rows) == 1
        row = rows[0]

        assert row["technical_id"] == parse_hash
        # semantic_id format: embed_{model}_{hash8}
        assert row["semantic_id"].startswith("embed_")
        assert parse_hash[:8] in row["semantic_id"]

        assert row["state"] == "candidate"
        assert "upsert_count" in row["stats"]
        assert row["stats"]["upsert_count"] == 1
        assert row["stats"]["vector_dim"] == 3

    def test_model_tag_required_for_embed(self, isolated_store):
        """Calling with step_type='embed' and no model_tag raises VersionManagementError."""
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            VersionManagementError,
        )

        store = isolated_store
        with pytest.raises((VersionManagementError, Exception)):
            store.list_version_candidate_rows("coll", "doc", "embed", model_tag=None)

    def test_async_variant_returns_same(self, isolated_store):
        """list_version_candidate_rows_async returns same results as sync."""
        import asyncio

        store = isolated_store
        collection = "coll_async"
        doc_id = "doc_async"
        parse_hash = "asynchash1234567"

        store.upsert_parses(
            [
                _make_parse_row(
                    collection,
                    doc_id,
                    parse_hash,
                    parse_method="pypdf",
                    parser="local:PyPDFParser@v1",
                )
            ]
        )

        rows_async = asyncio.run(
            store.list_version_candidate_rows_async(collection, doc_id, "parse")
        )
        rows_sync = store.list_version_candidate_rows(collection, doc_id, "parse")
        assert rows_async == rows_sync


class TestCleanupCascadeByScope:
    """Tests for VectorIndexStore.cleanup_cascade_by_scope – store-level (real LanceDB)."""

    def test_parse_scope_preview_counts_collapsed(self, isolated_store):
        """cleanup_cascade_by_scope scope=parse returns collapsed embeddings key."""
        store = isolated_store
        collection = "cc_coll1"
        doc_id = "cc_doc1"
        parse_hash = "parsehash11223344"

        # Seed a parse row
        store.upsert_parses([_make_parse_row(collection, doc_id, parse_hash)])
        # Seed a chunk
        store.upsert_chunks([_make_chunk_row(collection, doc_id, parse_hash, "ck1")])

        result = store.cleanup_cascade_by_scope(
            collection,
            doc_id,
            "parse",
            new_parse_hash="newhash1234567890",
            preview_only=True,
            confirm=False,
        )

        assert isinstance(result, dict)
        assert "embeddings" in result
        assert "chunks" in result
        assert "parses" in result

    @pytest.mark.parametrize(
        ("scope", "expected"),
        [
            ("parse", {"embeddings": 2, "chunks": 2}),
            ("chunk", {"embeddings": 2}),
        ],
    )
    def test_no_hashes_targets_untagged_main_pointer_rows_only(
        self, isolated_store, scope, expected
    ):
        """With neither hash, the untagged main pointer picks the rows to delete."""
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_main_pointer_store,
        )

        store = isolated_store
        coll, doc = "cc_ptr", "cc_ptr_doc"
        store.upsert_parses(
            [_make_parse_row(coll, doc, "main_hash"), _make_parse_row(coll, doc, "x")]
        )
        store.upsert_chunks(
            [
                _make_chunk_row(coll, doc, "main_hash", "m1"),
                _make_chunk_row(coll, doc, "main_hash", "m2"),
                _make_chunk_row(coll, doc, "x", "x1"),
            ]
        )
        store.upsert_embeddings(
            "m1",
            [
                {
                    "collection": coll,
                    "doc_id": doc,
                    "chunk_id": c,
                    "parse_hash": h,
                    "model": "m1",
                    "vector": [0.1, 0.2],
                    "vector_dimension": 2,
                    "text": c,
                    "chunk_hash": "ch_" + c,
                    "created_at": datetime.now(timezone.utc),
                    "metadata": "{}",
                    "user_id": None,
                }
                for h, c in [("main_hash", "m1"), ("main_hash", "m2"), ("x", "x1")]
            ],
        )

        def preview():
            return store.cleanup_cascade_by_scope(
                coll, doc, scope, preview_only=True, confirm=False
            )

        assert preview() == {"embeddings": 0}

        get_main_pointer_store().set_main_pointer(
            coll, doc, scope, f"{scope}_v1", "main_hash"
        )
        assert preview() == expected

    def test_document_scope_returns_6_keys(self, isolated_store):
        """cleanup_cascade_by_scope scope=document returns 6-key dict."""
        store = isolated_store
        collection = "cc_coll2"
        doc_id = "cc_doc2"

        result = store.cleanup_cascade_by_scope(
            collection,
            doc_id,
            "document",
            preview_only=True,
            confirm=False,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "embeddings",
            "chunks",
            "parses",
            "main_pointers",
            "documents",
            "ingestion_runs",
        }

    def test_embeddings_scope_collapse(self, isolated_store):
        """cleanup_cascade_by_scope scope=embeddings returns embeddings key."""
        store = isolated_store
        collection = "cc_coll3"
        doc_id = "cc_doc3"

        result = store.cleanup_cascade_by_scope(
            collection,
            doc_id,
            "embeddings",
            preview_only=True,
            confirm=False,
        )

        assert isinstance(result, dict)
        assert "embeddings" in result

    def test_async_variant_delegates_to_sync(self, isolated_store):
        """cleanup_cascade_by_scope_async returns same result as sync."""
        import asyncio

        store = isolated_store
        collection = "cc_coll4"
        doc_id = "cc_doc4"

        sync_result = store.cleanup_cascade_by_scope(
            collection, doc_id, "parse", preview_only=True, confirm=False
        )
        async_result = asyncio.run(
            store.cleanup_cascade_by_scope_async(
                collection, doc_id, "parse", preview_only=True, confirm=False
            )
        )
        assert sync_result == async_result
