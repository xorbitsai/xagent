"""Main pointer store against a real LanceDB table (#2858).

Storage isolation is provided by the autouse ``isolate_rag_storage`` fixture.
"""

import time

from xagent.core.tools.core.RAG_tools.storage.factory import get_main_pointer_store

COLL = "c"


def _set(store, doc_id="d", step="parse", sem="v1", tech="h1", tag=None):
    store.set_main_pointer(COLL, doc_id, step, sem, tech, model_tag=tag)


def test_untagged_pointer_round_trip() -> None:
    store = get_main_pointer_store()
    _set(store)

    first = store.get_main_pointer(COLL, "d", "parse")
    assert first is not None
    assert first["technical_id"] == "h1"

    time.sleep(0.01)
    _set(store, sem="v2", tech="h2")
    second = store.get_main_pointer(COLL, "d", "parse")
    assert second["technical_id"] == "h2"
    assert second["created_at"] == first["created_at"]
    assert len(store.list_main_pointers(COLL)) == 1

    assert store.delete_main_pointer(COLL, "d", "parse") is True
    assert store.get_main_pointer(COLL, "d", "parse") is None
    assert store.delete_main_pointer(COLL, "d", "parse") is False


def test_tagged_pointer_is_not_matched_by_untagged_lookup() -> None:
    store = get_main_pointer_store()
    _set(store, step="embed", tag="tag-a")

    assert store.get_main_pointer(COLL, "d", "embed") is None
    assert store.delete_main_pointer(COLL, "d", "embed") is False
    assert store.get_main_pointer(COLL, "d", "embed", "tag-a") is not None
    assert store.delete_main_pointer(COLL, "d", "embed", "tag-a") is True


def test_list_main_pointers_with_and_without_doc_id() -> None:
    store = get_main_pointer_store()
    _set(store, doc_id="d1")
    _set(store, doc_id="d2", step="embed", tag="tag-a")

    assert {p["doc_id"] for p in store.list_main_pointers(COLL)} == {"d1", "d2"}
    only = store.list_main_pointers(COLL, doc_id="d2")
    assert [(p["doc_id"], p["model_tag"]) for p in only] == [("d2", "tag-a")]
    assert store.list_main_pointers("other") == []


def _add_null_tag_row(doc_id: str, tech: str = "legacy") -> None:
    from datetime import datetime

    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_main_pointers_table,
    )
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        get_connection_from_env,
    )

    conn = get_connection_from_env()
    ensure_main_pointers_table(conn)
    now = datetime(2026, 1, 1)
    conn.open_table("main_pointers").add(
        [
            {
                "collection": COLL,
                "doc_id": doc_id,
                "step_type": "parse",
                "model_tag": None,
                "semantic_id": "legacy-v",
                "technical_id": tech,
                "created_at": now,
                "updated_at": now,
                "operator": "legacy",
            }
        ]
    )


def test_legacy_null_model_tag_row_is_matched_and_deleted() -> None:
    store = get_main_pointer_store()
    _add_null_tag_row("legacy")

    pointer = store.get_main_pointer(COLL, "legacy", "parse")
    assert pointer is not None
    assert pointer["technical_id"] == "legacy"
    assert pointer["model_tag"] is None
    assert [p["doc_id"] for p in store.list_main_pointers(COLL)] == ["legacy"]

    assert store.delete_main_pointer(COLL, "legacy", "parse") is True
    assert store.get_main_pointer(COLL, "legacy", "parse") is None
    assert store.delete_main_pointer(COLL, "legacy", "parse") is False


def test_set_on_legacy_null_row_is_read_back() -> None:
    store = get_main_pointer_store()
    _add_null_tag_row("legacy")

    _set(store, doc_id="legacy", sem="v2", tech="h2")

    pointer = store.get_main_pointer(COLL, "legacy", "parse")
    assert pointer is not None
    assert pointer["technical_id"] == "h2"
    assert store.delete_main_pointer(COLL, "legacy", "parse") is True
    assert store.get_main_pointer(COLL, "legacy", "parse") is None
