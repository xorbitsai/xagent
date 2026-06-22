"""Tests for the collection handle parse/chunk lifecycle (#509).

The handle owns collection-scoped parse/chunk storage mechanics (existence,
reuse read, write, latest-parse selection, chunk read/write, cleanup, and
rollback compensation). Parser/chunker algorithms stay in their modules.

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

import json
from datetime import datetime, timedelta, timezone

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
