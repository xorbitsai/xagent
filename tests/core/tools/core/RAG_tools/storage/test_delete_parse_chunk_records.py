"""Tests for the row-only ``delete_parse_records`` / ``delete_chunk_records``
store primitives (#509).

These primitives delete ONLY ``parses`` / ``chunks`` rows for a document
(optionally narrowed by parse_hash / config_hash). They must not cascade into
documents/embeddings, must be idempotent, and must preserve document-scoped
tenant safety.

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

from datetime import datetime, timezone

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
    collection: str,
    doc_id: str,
    parse_hash: str,
    config_hash: str,
    chunk_id: str,
    user_id=None,
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
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc),
        "metadata": "{}",
        "user_id": user_id,
    }


class TestDeleteParseRecords:
    def test_deletes_by_parse_hash_no_cascade(self) -> None:
        store = get_vector_index_store()
        store.upsert_documents([_doc_row("coll", "d1")])
        store.upsert_parses(
            [_parse_row("coll", "d1", "h1"), _parse_row("coll", "d1", "h2")]
        )
        store.upsert_chunks([_chunk_row("coll", "d1", "h1", "cfg1", "c0")])

        deleted = store.delete_parse_records(
            "coll", "d1", parse_hash="h1", user_id=None, is_admin=True
        )

        assert deleted == 1
        assert (
            store.count_rows(
                "parses", {"collection": "coll", "doc_id": "d1"}, is_admin=True
            )
            == 1
        )
        # documents and chunks untouched (no cascade).
        assert store.count_rows("documents", {"collection": "coll"}, is_admin=True) == 1
        assert store.count_rows("chunks", {"collection": "coll"}, is_admin=True) == 1

    def test_delete_all_parses_for_doc_when_hash_none(self) -> None:
        store = get_vector_index_store()
        store.upsert_parses(
            [_parse_row("coll", "d1", "h1"), _parse_row("coll", "d1", "h2")]
        )
        deleted = store.delete_parse_records(
            "coll", "d1", parse_hash=None, user_id=None, is_admin=True
        )
        assert deleted == 2
        assert store.count_rows("parses", {"collection": "coll"}, is_admin=True) == 0

    def test_idempotent_and_missing_table_returns_zero(self) -> None:
        store = get_vector_index_store()
        # No parses table yet.
        assert (
            store.delete_parse_records(
                "coll", "d1", parse_hash="h1", user_id=None, is_admin=True
            )
            == 0
        )
        store.upsert_parses([_parse_row("coll", "d1", "h1")])
        assert (
            store.delete_parse_records(
                "coll", "d1", parse_hash="h1", user_id=None, is_admin=True
            )
            == 1
        )
        assert (
            store.delete_parse_records(
                "coll", "d1", parse_hash="h1", user_id=None, is_admin=True
            )
            == 0
        )

    def test_user_scoping_blocks_other_tenant(self) -> None:
        store = get_vector_index_store()
        store.upsert_parses([_parse_row("coll", "d1", "h1", user_id=5)])

        assert (
            store.delete_parse_records(
                "coll", "d1", parse_hash="h1", user_id=6, is_admin=False
            )
            == 0
        )
        assert store.count_rows("parses", {"collection": "coll"}, is_admin=True) == 1
        assert (
            store.delete_parse_records(
                "coll", "d1", parse_hash="h1", user_id=5, is_admin=False
            )
            == 1
        )


class TestDeleteChunkRecords:
    def test_deletes_by_parse_and_config_hash_no_cascade(self) -> None:
        store = get_vector_index_store()
        store.upsert_documents([_doc_row("coll", "d1")])
        store.upsert_parses([_parse_row("coll", "d1", "h1")])
        store.upsert_chunks(
            [
                _chunk_row("coll", "d1", "h1", "cfg1", "c0"),
                _chunk_row("coll", "d1", "h1", "cfg1", "c1"),
                _chunk_row("coll", "d1", "h1", "cfg2", "c2"),
            ]
        )

        deleted = store.delete_chunk_records(
            "coll",
            "d1",
            parse_hash="h1",
            config_hash="cfg1",
            user_id=None,
            is_admin=True,
        )

        assert deleted == 2
        # The cfg2 chunk survives.
        assert store.count_rows("chunks", {"collection": "coll"}, is_admin=True) == 1
        # documents and parses untouched.
        assert store.count_rows("documents", {"collection": "coll"}, is_admin=True) == 1
        assert store.count_rows("parses", {"collection": "coll"}, is_admin=True) == 1

    def test_delete_all_chunks_for_doc_when_hashes_none(self) -> None:
        store = get_vector_index_store()
        store.upsert_chunks(
            [
                _chunk_row("coll", "d1", "h1", "cfg1", "c0"),
                _chunk_row("coll", "d1", "h2", "cfg2", "c1"),
            ]
        )
        deleted = store.delete_chunk_records(
            "coll",
            "d1",
            parse_hash=None,
            config_hash=None,
            user_id=None,
            is_admin=True,
        )
        assert deleted == 2
        assert store.count_rows("chunks", {"collection": "coll"}, is_admin=True) == 0

    def test_idempotent_and_missing_table_returns_zero(self) -> None:
        store = get_vector_index_store()
        assert (
            store.delete_chunk_records(
                "coll",
                "d1",
                parse_hash="h1",
                config_hash="cfg1",
                user_id=None,
                is_admin=True,
            )
            == 0
        )
        store.upsert_chunks([_chunk_row("coll", "d1", "h1", "cfg1", "c0")])
        assert (
            store.delete_chunk_records(
                "coll",
                "d1",
                parse_hash="h1",
                config_hash="cfg1",
                user_id=None,
                is_admin=True,
            )
            == 1
        )
        assert (
            store.delete_chunk_records(
                "coll",
                "d1",
                parse_hash="h1",
                config_hash="cfg1",
                user_id=None,
                is_admin=True,
            )
            == 0
        )

    def test_user_scoping_blocks_other_tenant(self) -> None:
        store = get_vector_index_store()
        store.upsert_chunks([_chunk_row("coll", "d1", "h1", "cfg1", "c0", user_id=5)])

        assert (
            store.delete_chunk_records(
                "coll",
                "d1",
                parse_hash="h1",
                config_hash="cfg1",
                user_id=6,
                is_admin=False,
            )
            == 0
        )
        assert store.count_rows("chunks", {"collection": "coll"}, is_admin=True) == 1
        assert (
            store.delete_chunk_records(
                "coll",
                "d1",
                parse_hash="h1",
                config_hash="cfg1",
                user_id=5,
                is_admin=False,
            )
            == 1
        )
