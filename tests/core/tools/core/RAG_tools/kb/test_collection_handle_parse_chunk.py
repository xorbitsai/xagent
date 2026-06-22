"""Tests for the collection handle parse/chunk lifecycle (#509).

The handle owns collection-scoped parse/chunk storage mechanics (existence,
reuse read, write, latest-parse selection, chunk read/write, cleanup, and
rollback compensation). Parser/chunker algorithms stay in their modules.

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    ParsedParagraph,
    ParseMethod,
)
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_metadata_store,
    get_vector_index_store,
)


def make_handle(collection: str = "coll") -> LanceDBCollectionHandle:
    """Build a LanceDB-backed handle bound to the current test stores."""
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context)


def _seed_parse(
    collection: str,
    doc_id: str,
    parse_hash: str,
    *,
    paragraphs=None,
    parser: str = "local:default@v1.0.0",
    created_at=None,
    user_id=None,
) -> None:
    if paragraphs is None:
        paragraphs = [{"text": "hello world", "metadata": {"layout_type": "text"}}]
    get_vector_index_store().upsert_parses(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "parser": parser,
                "created_at": created_at or datetime.now(timezone.utc),
                "params_json": "{}",
                "parsed_content": json.dumps(paragraphs, ensure_ascii=False),
                "user_id": user_id,
            }
        ]
    )


class TestHandleParseExists:
    def test_true_when_parse_present(self) -> None:
        handle = make_handle("coll")
        _seed_parse("coll", "d1", "h1")
        assert handle.parse_exists("d1", "h1", is_admin=True) is True

    def test_false_when_absent_or_other_collection(self) -> None:
        handle = make_handle("coll")
        _seed_parse("other", "d1", "h1")
        assert handle.parse_exists("d1", "h1", is_admin=True) is False
        assert handle.parse_exists("d1", "nope", is_admin=True) is False


class TestHandleReadParseParagraphs:
    def test_returns_parsed_paragraphs(self) -> None:
        handle = make_handle("coll")
        _seed_parse(
            "coll",
            "d1",
            "h1",
            paragraphs=[
                {"text": "alpha", "metadata": {"layout_type": "text"}},
                {"text": "beta", "metadata": {"page": 2}},
            ],
        )
        paras = handle.read_parse_paragraphs("d1", "h1", is_admin=True)
        assert [p.text for p in paras] == ["alpha", "beta"]
        assert all(isinstance(p, ParsedParagraph) for p in paras)
        assert paras[1].metadata == {"page": 2}

    def test_empty_when_absent(self) -> None:
        handle = make_handle("coll")
        assert handle.read_parse_paragraphs("d1", "h1", is_admin=True) == []


class TestHandleWriteParse:
    def test_persists_exact_parse_row(self) -> None:
        handle = make_handle("coll")
        paragraphs = [
            ParsedParagraph(text="alpha", metadata={"layout_type": "text"}),
            ParsedParagraph(text="beta", metadata={}),
        ]

        written = handle.write_parse(
            "d1",
            "h1",
            ParseMethod.DEFAULT,
            {"foo": "bar"},
            paragraphs,
            user_id=7,
        )
        assert written is True

        store = get_vector_index_store()
        rows = []
        for batch in store.iter_batches(
            table_name="parses",
            filters={"collection": "coll", "doc_id": "d1", "parse_hash": "h1"},
            is_admin=True,
        ):
            rows.extend(batch.to_pylist())
        assert len(rows) == 1
        row = rows[0]
        assert row["collection"] == "coll"
        assert row["doc_id"] == "d1"
        assert row["parse_hash"] == "h1"
        assert row["parser"] == f"local:{ParseMethod.DEFAULT}@v1.0.0"
        assert row["user_id"] == 7
        assert json.loads(row["params_json"]) == {"foo": "bar"}
        assert [p["text"] for p in json.loads(row["parsed_content"])] == [
            "alpha",
            "beta",
        ]

    def test_persists_into_context_collection(self) -> None:
        # The handle is collection-scoped; writes land in the bound collection.
        handle = make_handle("coll_a")
        handle.write_parse(
            "d1", "h1", ParseMethod.DEFAULT, {}, [ParsedParagraph(text="x")]
        )
        store = get_vector_index_store()
        assert store.count_rows("parses", {"collection": "coll_a"}, is_admin=True) == 1
        assert store.count_rows("parses", {"collection": "coll_b"}, is_admin=True) == 0


class TestHandleReadLatestParseRecord:
    def test_selects_latest_by_created_at(self) -> None:
        handle = make_handle("coll")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        _seed_parse("coll", "d1", "old", paragraphs=[{"text": "old"}], created_at=base)
        _seed_parse(
            "coll",
            "d1",
            "new",
            paragraphs=[{"text": "new"}],
            created_at=base + timedelta(days=1),
        )

        record = handle.read_latest_parse_record("d1", is_admin=True)
        assert record is not None
        assert record.parse_hash == "new"
        assert json.loads(record.parsed_content) == [{"text": "new"}]

    def test_honors_parse_hash_filter(self) -> None:
        handle = make_handle("coll")
        _seed_parse("coll", "d1", "h1", paragraphs=[{"text": "one"}])
        _seed_parse("coll", "d1", "h2", paragraphs=[{"text": "two"}])

        record = handle.read_latest_parse_record("d1", parse_hash="h1", is_admin=True)
        assert record is not None
        assert record.parse_hash == "h1"

    def test_none_when_absent(self) -> None:
        handle = make_handle("coll")
        assert handle.read_latest_parse_record("d1", is_admin=True) is None


def _seed_chunk(
    collection: str,
    doc_id: str,
    parse_hash: str,
    config_hash: str,
    chunk_id: str,
    *,
    index: int = 0,
    metadata=None,
    user_id=None,
) -> None:
    from xagent.core.tools.core.RAG_tools.utils.metadata_utils import serialize_metadata

    get_vector_index_store().upsert_chunks(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "chunk_id": chunk_id,
                "index": index,
                "text": f"text-{chunk_id}",
                "page_number": None,
                "section": None,
                "anchor": None,
                "json_path": None,
                "chunk_hash": f"ch-{chunk_id}",
                "config_hash": config_hash,
                "created_at": datetime.now(timezone.utc),
                "metadata": serialize_metadata(metadata or {"k": "v"}),
                "user_id": user_id,
            }
        ]
    )


class TestHandleChunkExists:
    def test_true_when_chunks_present(self) -> None:
        handle = make_handle("coll")
        _seed_chunk("coll", "d1", "h1", "cfg1", "c0")
        assert handle.chunk_exists("d1", "h1", "cfg1", is_admin=True) is True

    def test_false_when_absent(self) -> None:
        handle = make_handle("coll")
        _seed_chunk("coll", "d1", "h1", "cfg1", "c0")
        assert handle.chunk_exists("d1", "h1", "other", is_admin=True) is False


class TestHandleReadExistingChunks:
    def test_returns_normalized_chunks(self) -> None:
        handle = make_handle("coll")
        _seed_chunk("coll", "d1", "h1", "cfg1", "c0", index=0, metadata={"a": 1})
        _seed_chunk("coll", "d1", "h1", "cfg1", "c1", index=1, metadata={"b": 2})

        chunks = handle.read_existing_chunks("d1", "h1", "cfg1", is_admin=True)
        assert {c["chunk_id"] for c in chunks} == {"c0", "c1"}
        sample = next(c for c in chunks if c["chunk_id"] == "c0")
        # metadata is deserialized back to a dict; optional fields preserved.
        assert sample["metadata"] == {"a": 1}
        assert sample["index"] == 0
        assert sample["page_number"] is None
        assert set(sample.keys()) == {
            "chunk_id",
            "index",
            "text",
            "page_number",
            "section",
            "anchor",
            "json_path",
            "created_at",
            "metadata",
        }

    def test_empty_when_absent(self) -> None:
        handle = make_handle("coll")
        assert handle.read_existing_chunks("d1", "h1", "cfg1", is_admin=True) == []


class TestHandleReadParseParagraphDicts:
    def test_returns_text_metadata_dicts(self) -> None:
        handle = make_handle("coll")
        _seed_parse(
            "coll",
            "d1",
            "h1",
            paragraphs=[
                {"text": "alpha", "metadata": {"layout_type": "text"}},
                {"text": "beta", "metadata": {}},
            ],
        )
        paras = handle.read_parse_paragraph_dicts("d1", "h1", is_admin=True)
        assert paras == [
            {"text": "alpha", "metadata": {"layout_type": "text"}},
            {"text": "beta", "metadata": {}},
        ]

    def test_empty_when_absent(self) -> None:
        handle = make_handle("coll")
        assert handle.read_parse_paragraph_dicts("d1", "h1", is_admin=True) == []


class TestHandleWriteChunks:
    def test_persists_exact_chunk_rows(self) -> None:
        from xagent.core.tools.core.RAG_tools.utils.hash_utils import compute_chunk_hash

        handle = make_handle("coll")
        params = {"chunk_strategy": "recursive", "chunk_size": 1000}
        indexed_chunks = [
            {
                "chunk_id": "c0",
                "index": 0,
                "text": "hello",
                "page_number": 1,
                "section": "Intro",
                "anchor": "a0",
                "json_path": None,
                "created_at": datetime.now(timezone.utc),
                "metadata": {"layout_type": "text"},
            }
        ]

        written = handle.write_chunks(
            "d1", "h1", "cfg1", params, indexed_chunks, user_id=7
        )
        assert written is True

        store = get_vector_index_store()
        rows = []
        for batch in store.iter_batches(
            table_name="chunks",
            filters={"collection": "coll", "doc_id": "d1", "parse_hash": "h1"},
            is_admin=True,
        ):
            rows.extend(batch.to_pylist())
        assert len(rows) == 1
        row = rows[0]
        assert row["collection"] == "coll"
        assert row["chunk_id"] == "c0"
        assert row["config_hash"] == "cfg1"
        assert row["user_id"] == 7
        assert row["chunk_hash"] == compute_chunk_hash("hello", params)
        assert json.loads(row["metadata"]) == {"layout_type": "text"}

    def test_empty_chunks_returns_false(self) -> None:
        handle = make_handle("coll")
        assert handle.write_chunks("d1", "h1", "cfg1", {}, []) is False

    def test_persists_into_context_collection(self) -> None:
        handle = make_handle("coll_a")
        handle.write_chunks(
            "d1",
            "h1",
            "cfg1",
            {},
            [
                {
                    "chunk_id": "c0",
                    "index": 0,
                    "text": "x",
                    "created_at": datetime.now(timezone.utc),
                    "metadata": None,
                }
            ],
        )
        store = get_vector_index_store()
        assert store.count_rows("chunks", {"collection": "coll_a"}, is_admin=True) == 1
        assert store.count_rows("chunks", {"collection": "coll_b"}, is_admin=True) == 0


class TestHandleParseChunkCleanup:
    def test_delete_parse_records_by_hash_collection_scoped(self) -> None:
        handle = make_handle("coll")
        _seed_parse("coll", "d1", "h1")
        _seed_parse("coll", "d1", "h2")
        # A parse in another collection must not be touched.
        _seed_parse("other", "d1", "h1")

        deleted = handle.delete_parse_records("d1", parse_hash="h1", is_admin=True)
        assert deleted == 1

        store = get_vector_index_store()
        assert store.count_rows("parses", {"collection": "coll"}, is_admin=True) == 1
        assert store.count_rows("parses", {"collection": "other"}, is_admin=True) == 1

    def test_delete_all_parses_for_doc_and_idempotent(self) -> None:
        handle = make_handle("coll")
        _seed_parse("coll", "d1", "h1")
        _seed_parse("coll", "d1", "h2")

        assert handle.delete_parse_records("d1", is_admin=True) == 2
        assert handle.delete_parse_records("d1", is_admin=True) == 0

    def test_delete_chunk_records_by_config_no_cascade(self) -> None:
        handle = make_handle("coll")
        _seed_parse("coll", "d1", "h1")
        _seed_chunk("coll", "d1", "h1", "cfg1", "c0")
        _seed_chunk("coll", "d1", "h1", "cfg1", "c1")
        _seed_chunk("coll", "d1", "h1", "cfg2", "c2")

        deleted = handle.delete_chunk_records(
            "d1", parse_hash="h1", config_hash="cfg1", is_admin=True
        )
        assert deleted == 2

        store = get_vector_index_store()
        assert store.count_rows("chunks", {"collection": "coll"}, is_admin=True) == 1
        # Parse rows are not cascaded by the chunk cleanup.
        assert store.count_rows("parses", {"collection": "coll"}, is_admin=True) == 1

    def test_delete_all_chunks_for_doc_and_idempotent(self) -> None:
        handle = make_handle("coll")
        _seed_chunk("coll", "d1", "h1", "cfg1", "c0")
        _seed_chunk("coll", "d1", "h2", "cfg2", "c1")

        assert handle.delete_chunk_records("d1", is_admin=True) == 2
        assert handle.delete_chunk_records("d1", is_admin=True) == 0


class TestHandleParseRollback:
    def test_new_parse_rollback_idempotent_no_cascade(self) -> None:
        handle = make_handle("coll")
        _seed_parse("coll", "d1", "h1")
        _seed_chunk("coll", "d1", "h1", "cfg1", "c0")

        assert handle.delete_created_parse("d1", "h1", is_admin=True) == 1
        store = get_vector_index_store()
        assert store.count_rows("parses", {"collection": "coll"}, is_admin=True) == 0
        # No cascade into chunks.
        assert store.count_rows("chunks", {"collection": "coll"}, is_admin=True) == 1
        # Idempotent.
        assert handle.delete_created_parse("d1", "h1", is_admin=True) == 0

    def test_snapshot_then_restore_preserves_fields(self) -> None:
        handle = make_handle("coll")
        _seed_parse(
            "coll",
            "d1",
            "h1",
            paragraphs=[{"text": "original"}],
            parser="local:custom@v1.0.0",
            user_id=7,
        )
        snapshot = handle.snapshot_parse("d1", "h1", is_admin=True)
        assert snapshot is not None

        # Overwrite the parse row with different content.
        handle.write_parse(
            "d1", "h1", ParseMethod.DEFAULT, {"x": 1}, [ParsedParagraph(text="changed")]
        )
        latest = handle.read_latest_parse_record("d1", parse_hash="h1", is_admin=True)
        assert latest is not None
        assert json.loads(latest.parsed_content) == [
            {"text": "changed", "metadata": {}}
        ]

        # Restore brings every field back.
        handle.restore_parse(snapshot)
        restored = handle.read_latest_parse_record("d1", parse_hash="h1", is_admin=True)
        assert restored is not None
        assert restored.parser == "local:custom@v1.0.0"
        assert restored.user_id == 7
        assert json.loads(restored.parsed_content) == [{"text": "original"}]

    def test_snapshot_none_when_absent(self) -> None:
        handle = make_handle("coll")
        assert handle.snapshot_parse("d1", "h1", is_admin=True) is None

    def test_restore_rejects_snapshot_from_other_collection(self) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DocumentValidationError,
        )

        source = make_handle("coll_a")
        _seed_parse("coll_a", "d1", "h1")
        snapshot = source.snapshot_parse("d1", "h1", is_admin=True)
        assert snapshot is not None

        other = make_handle("coll_b")
        with pytest.raises(DocumentValidationError, match="cannot restore"):
            other.restore_parse(snapshot)


class TestHandleChunkRollback:
    def test_new_chunks_rollback_idempotent(self) -> None:
        handle = make_handle("coll")
        _seed_chunk("coll", "d1", "h1", "cfg1", "c0")
        _seed_chunk("coll", "d1", "h1", "cfg1", "c1")

        assert handle.delete_created_chunks("d1", "h1", "cfg1", is_admin=True) == 2
        store = get_vector_index_store()
        assert store.count_rows("chunks", {"collection": "coll"}, is_admin=True) == 0
        assert handle.delete_created_chunks("d1", "h1", "cfg1", is_admin=True) == 0

    def test_snapshot_then_restore_preserves_all_rows(self) -> None:
        handle = make_handle("coll")
        _seed_chunk("coll", "d1", "h1", "cfg1", "c0", index=0, metadata={"a": 1})
        _seed_chunk("coll", "d1", "h1", "cfg1", "c1", index=1, metadata={"b": 2})

        snapshot = handle.snapshot_chunks("d1", "h1", "cfg1", is_admin=True)
        assert snapshot is not None
        assert [c.chunk_id for c in snapshot.chunks] == ["c0", "c1"]

        # Destroy then restore.
        assert handle.delete_chunk_records("d1", is_admin=True) == 2
        handle.restore_chunks(snapshot)

        restored = handle.read_existing_chunks("d1", "h1", "cfg1", is_admin=True)
        assert {c["chunk_id"] for c in restored} == {"c0", "c1"}
        by_id = {c["chunk_id"]: c for c in restored}
        assert by_id["c0"]["metadata"] == {"a": 1}
        assert by_id["c1"]["index"] == 1

    def test_snapshot_none_when_absent(self) -> None:
        handle = make_handle("coll")
        assert handle.snapshot_chunks("d1", "h1", "cfg1", is_admin=True) is None

    def test_restore_rejects_snapshot_from_other_collection(self) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DocumentValidationError,
        )

        source = make_handle("coll_a")
        _seed_chunk("coll_a", "d1", "h1", "cfg1", "c0")
        snapshot = source.snapshot_chunks("d1", "h1", "cfg1", is_admin=True)
        assert snapshot is not None

        other = make_handle("coll_b")
        with pytest.raises(DocumentValidationError, match="cannot restore"):
            other.restore_chunks(snapshot)
