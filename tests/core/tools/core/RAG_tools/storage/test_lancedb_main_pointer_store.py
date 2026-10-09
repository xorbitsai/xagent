"""Store-level tests for LanceDBMainPointerStore against a real LanceDB table."""

from __future__ import annotations

from pathlib import Path

import pytest

from xagent.core.tools.core.RAG_tools.storage import reset_rag_storage_for_tests
from xagent.core.tools.core.RAG_tools.storage.factory import get_main_pointer_store


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    lancedb_dir = tmp_path / "lancedb"
    lancedb_dir.mkdir()
    monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))
    reset_rag_storage_for_tests()
    yield get_main_pointer_store()
    reset_rag_storage_for_tests()


def test_untagged_main_pointer_round_trip(store) -> None:
    store.set_main_pointer(
        collection="c",
        doc_id="d",
        step_type="parse",
        semantic_id="v1",
        technical_id="h1",
    )

    pointer = store.get_main_pointer(collection="c", doc_id="d", step_type="parse")
    assert pointer is not None
    assert pointer["semantic_id"] == "v1"
    assert pointer["technical_id"] == "h1"
    created_at = pointer["created_at"]

    store.set_main_pointer(
        collection="c",
        doc_id="d",
        step_type="parse",
        semantic_id="v2",
        technical_id="h2",
    )
    updated = store.get_main_pointer(collection="c", doc_id="d", step_type="parse")
    assert updated is not None
    assert updated["semantic_id"] == "v2"
    assert updated["created_at"] == created_at

    assert store.delete_main_pointer(collection="c", doc_id="d", step_type="parse")
    assert store.get_main_pointer(collection="c", doc_id="d", step_type="parse") is None


def test_list_main_pointers(store) -> None:
    assert store.list_main_pointers(collection="c") == []

    for doc_id in ("d1", "d2"):
        store.set_main_pointer(
            collection="c",
            doc_id=doc_id,
            step_type="parse",
            semantic_id="v1",
            technical_id="h1",
        )
    store.set_main_pointer(
        collection="other",
        doc_id="d1",
        step_type="parse",
        semantic_id="v1",
        technical_id="h1",
    )

    pointers = store.list_main_pointers(collection="c")
    assert sorted(p["doc_id"] for p in pointers) == ["d1", "d2"]

    pointers = store.list_main_pointers(collection="c", doc_id="d2")
    assert [p["doc_id"] for p in pointers] == ["d2"]
