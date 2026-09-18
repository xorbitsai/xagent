"""Real-LanceDB coverage for dormant streaming retrieval compatibility (#2346)."""

import json

import pyarrow as pa  # type: ignore
import pytest

from xagent.core.memory import retrieval_compatibility
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.memory.retrieval_compatibility import (
    stream_lexical_top_k,
    stream_list_all,
    stream_stats,
)
from xagent.core.memory.scope_columns import SCOPE_DIMS_COLUMN, USER_ID_COLUMN
from xagent.core.memory.storage_admission import DormantLanceDBMemoryHandle
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.providers.vector_store.lancedb import clear_connection_cache


@pytest.fixture
def store(tmp_path):
    """A real table whose rows all carry NULL vectors (no embedding model)."""
    clear_connection_cache()
    result = LanceDBMemoryStore(str(tmp_path), collection_name="memories")
    yield result
    clear_connection_cache()


@pytest.fixture
def observed(monkeypatch):
    """Rows materialised per scan batch and per retained set, in the style of
    the ``scanned`` counter in ``test_storage_admission_matrix.py``."""
    seen: dict[str, list[int]] = {"scan_batch": [], "retained": []}
    monkeypatch.setattr(
        retrieval_compatibility,
        "_checkpoint",
        lambda stage, batch=None: seen.setdefault(stage, []).append(batch),
    )
    return seen


def _add(store, note_id, text, *, user_id=7, metadata=None, **fields):
    assert store.add(
        MemoryNote(
            id=note_id,
            content=text,
            metadata={"user_id": user_id, **(metadata or {})},
            **fields,
        )
    ).success


def _add_raw(store, rows):
    """Insert rows the store's own writer would reject (malformed metadata)."""
    table = store._vector_store.get_raw_connection().open_table("memories")
    table.add(pa.Table.from_pylist(rows, schema=table.schema))
    _safe_close_table(table)


def _raw(note_id, text, metadata, *, user_id=7):
    return {
        "id": note_id,
        "text": text,
        "metadata": json.dumps(metadata),
        "vector": None,
        USER_ID_COLUMN: user_id,
        SCOPE_DIMS_COLUMN: [],
    }


def _handle(store):
    return DormantLanceDBMemoryHandle(
        store._vector_store.get_raw_connection(), "memories"
    )


def _stream_kwargs(store):
    return dict(
        row_to_note=store._dict_to_memory_note,
        note_filter_factory=store._residual_note_filter,
    )


def _top_k(store, query, k, **kwargs):
    return [
        note.id
        for note in stream_lexical_top_k(
            _handle(store),
            query,
            k,
            **_stream_kwargs(store),
            **kwargs,
        )
    ]


def test_tail_exact_winner_in_the_last_batch_beats_earlier_partial_matches(store):
    _add(store, "p1", "alpha beta")
    _add(store, "p2", "alpha gamma")
    _add(store, "tail", "alpha")
    # batch_size=1 puts the exact match alone in the final batch: it still wins,
    # so nothing capped the raw rows before ranking.
    assert _top_k(store, "alpha", 2, batch_size=1) == ["tail", "p1"]
    assert _top_k(store, "alpha", 1, batch_size=1) == ["tail"]


def test_residual_filter_rejection_does_not_consume_a_slot(store):
    _add(store, "drop", "alpha", metadata={"priority": "drop"})
    _add(store, "keep", "alpha alpha", metadata={"priority": "keep"})
    # "drop" is the better lexical match and is read first, but fails a residual
    # filter, so the single slot must still go to "keep".
    assert _top_k(
        store,
        "alpha",
        1,
        filters={"metadata": {"user_id": 7}, "priority": "keep"},
        batch_size=1,
    ) == ["keep"]


def test_cross_batch_ordering_is_identical_for_every_batch_size(store):
    for index, text in enumerate(
        ["alpha", "alpha alpha", "beta alpha", "alpha beta", "alpha alpha alpha"]
    ):
        _add(store, f"n{index}", text)
    orderings = {
        batch_size: tuple(_top_k(store, "alpha", 4, batch_size=batch_size))
        for batch_size in (1, 2, 3, 64)
    }
    assert len(set(orderings.values())) == 1
    assert orderings[1] == ("n0", "n4", "n1", "n3")


def test_malformed_row_is_skipped_without_aborting_the_scan(store):
    _add_raw(store, [_raw("bad", "alpha", {"user_id": 7, "timestamp": "not-a-date"})])
    _add(store, "good", "alpha")
    assert _top_k(store, "alpha", 5, batch_size=1) == ["good"]


def test_ann_duplicate_does_not_consume_lexical_quota(store):
    _add(store, "dup", "alpha")
    _add(store, "win", "alpha beta")
    # "dup" outranks "win", so without the ANN-id skip it would take the slot.
    assert _top_k(store, "alpha", 1, exclude_ids={"dup"}, batch_size=1) == ["win"]
    assert _top_k(store, "alpha", 1, batch_size=1) == ["dup"]


def test_user_and_scope_isolation_hold_on_a_real_two_principal_table(store, observed):
    _add(store, "mine", "alpha", metadata={"execution_scope_agent": "x"})
    _add(
        store, "other-user", "alpha", user_id=8, metadata={"execution_scope_agent": "x"}
    )
    _add(store, "other-scope", "alpha", metadata={"execution_scope_agent": "y"})
    filters = {"metadata": {"user_id": 7, "execution_scope_agent": "x"}}
    assert _top_k(store, "alpha", 5, filters=filters, batch_size=1) == ["mine"]
    # Both dimensions were pushed into `where`: the foreign rows were never read.
    assert sum(observed["scan_batch"]) == 1


def test_streaming_list_and_stats_match_the_existing_paths(store, monkeypatch):
    _add(store, "a", "alpha", category="c1", tags=["t1"])
    _add(store, "b", "beta", category="c2", tags=["t1", "t2"])
    _add(store, "c", "gamma", category="c1")
    expected_ids = [note.id for note in store.list_all()]
    expected_stats = store.get_stats()

    def _no_ranking(*args, **kwargs):
        raise AssertionError("streaming paths must not route through search()")

    monkeypatch.setattr(store, "search", _no_ranking)
    listed = stream_list_all(_handle(store), batch_size=2, **_stream_kwargs(store))
    assert [note.id for note in listed] == expected_ids
    assert (
        stream_stats(_handle(store), batch_size=2, **_stream_kwargs(store))
        == expected_stats
    )


def test_scan_and_selection_residency_stay_bounded(store, observed):
    for index in range(5):
        _add(store, f"n{index}", "alpha")
    assert _top_k(store, "alpha", 2, batch_size=2) == ["n0", "n1"]
    # Never more than batch_size rows resident, nothing skipped, and the
    # selection buffer never grows past k.
    assert max(observed["scan_batch"]) <= 2
    assert sum(observed["scan_batch"]) == 5
    assert max(observed["retained"]) <= 2
