"""Real-LanceDB coverage for dormant memory admission primitives."""

import json

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest

from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.memory.scope_columns import SCOPE_DIMS_COLUMN, USER_ID_COLUMN
from xagent.core.memory.vector_compatibility import (
    VECTOR_IDENTITY_METADATA_KEY,
    EmbeddingIdentity,
    VectorCompatibility,
    create_or_recreate_vector_capable_table,
)
from xagent.core.model.embedding import BaseEmbedding
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.providers.vector_store.lancedb import clear_connection_cache

IDENTITY = EmbeddingIdentity(
    "openai", "text-embedding-3-small", "https://api.openai.com/v1/embeddings", 4, None
)


class ConstantEmbedding(BaseEmbedding):
    def encode(self, text, dimension=None, instruct=None):
        vector = [0.5] * 4
        return vector if isinstance(text, str) else [vector for _ in text]

    def get_dimension(self):
        return 4

    @property
    def abilities(self):
        return ["embed"]


@pytest.fixture
def store(tmp_path):
    clear_connection_cache()
    result = LanceDBMemoryStore(
        str(tmp_path), collection_name="memories", embedding_model=ConstantEmbedding()
    )
    yield result
    clear_connection_cache()


def _raw_row(note_id, text, user_id, *, vector=None, priority="keep"):
    return {
        "id": note_id,
        "text": text,
        "metadata": json.dumps(
            {"content": text or "", "user_id": user_id, "priority": priority}
        ),
        "vector": vector,
        USER_ID_COLUMN: user_id,
        SCOPE_DIMS_COLUMN: [],
    }


def _add_raw_rows(table, rows):
    table.add(pa.Table.from_pylist(rows, schema=table.schema))


def test_ann_then_null_vector_fallback_is_filtered_deduplicated_and_stable(store):
    assert store.add(
        MemoryNote(
            id="ann",
            content="semantic winner",
            metadata={"user_id": 7, "priority": "keep"},
        )
    ).success
    table = store._vector_store.get_raw_connection().open_table("memories")
    _add_raw_rows(
        table,
        [
            _raw_row("z", "alpha alpha", 7),
            _raw_row("a", "alpha", 7),
            _raw_row("foreign", "alpha", 8),
            _raw_row("filtered", "alpha", 7, priority="drop"),
            _raw_row("ann", "alpha", 7),
            _raw_row("null-text", None, 7),
        ],
    )
    _safe_close_table(table)

    filters = {"metadata": {"user_id": 7}, "priority": "keep"}
    first = store.search_with_null_vector_fallback("alpha", k=3, filters=filters)
    second = store.search_with_null_vector_fallback("alpha", k=3, filters=filters)
    assert [note.id for note in first] == ["ann", "a", "z"]
    assert [note.id for note in second] == ["ann", "a", "z"]

    # Better ANN hits are never displaced merely to expose a lexical fallback.
    assert [
        note.id
        for note in store.search_with_null_vector_fallback(
            "alpha", k=1, filters=filters
        )
    ] == ["ann"]


def test_standard_search_does_not_enable_null_vector_supplement(store):
    assert store.add(MemoryNote(id="ann", content="winner")).success
    table = store._vector_store.get_raw_connection().open_table("memories")
    _add_raw_rows(table, [_raw_row("fallback", "alpha", None)])
    _safe_close_table(table)
    assert [note.id for note in store.search("alpha", k=2)] == ["ann"]


def test_real_null_vector_fallback_streams_to_tail_exact_winner(store):
    table = store._vector_store.get_raw_connection().open_table("memories")
    _add_raw_rows(
        table,
        [
            *[
                _raw_row(f"row-{index:03d}", "alpha broad match", 7)
                for index in range(300)
            ],
            _raw_row("tail-exact", "alpha", 7),
        ],
    )
    _safe_close_table(table)
    store._embedding_model = None

    result = store.search_with_null_vector_fallback(
        "alpha", k=1, filters={"metadata": {"user_id": 7}}
    )

    assert [note.id for note in result] == ["tail-exact"]


def test_real_scope_pushdown_prevents_foreign_rows_from_consuming_scan_cap(store):
    table = store._vector_store.get_raw_connection().open_table("memories")
    _add_raw_rows(
        table,
        [
            *[_raw_row(f"foreign-{index:03d}", "alpha", 8) for index in range(150)],
            _raw_row("tenant-hit", "alpha", 7),
        ],
    )
    _safe_close_table(table)
    store._embedding_model = None

    result = store.search_with_null_vector_fallback(
        "alpha", k=1, filters={"metadata": {"user_id": 7}}
    )

    assert [note.id for note in result] == ["tenant-hit"]


def test_null_fallback_streams_projected_backend_filtered_batches(store):
    observed = {}

    class Table:
        def to_batches(self, **kwargs):
            observed.update(kwargs)
            return iter([pa.record_batch({"id": [], "text": [], "metadata": []})])

    assert (
        store._lexical_candidates(
            Table(),
            "alpha",
            {},
            scope_where="user_id = 7",
            null_vectors_only=True,
            candidate_limit=1,
        )
        == []
    )
    assert observed == {
        "filter": "(user_id = 7) AND vector IS NULL",
        "columns": ["id", "text", "metadata"],
        "batch_size": 256,
    }


def test_streaming_fallback_skips_ineligible_early_batches_and_malformed_rows(store):
    table = store._vector_store.get_raw_connection().open_table("memories")
    rows = [
        _raw_row(f"drop-{index:03d}", "alpha", 7, priority="drop")
        for index in range(280)
    ]
    malformed = _raw_row("malformed", "alpha", 7)
    malformed["metadata"] = json.dumps(
        {"content": "alpha", "user_id": 7, "timestamp": "not-a-date"}
    )
    rows.extend([malformed, _raw_row("eligible", "alpha", 7)])
    _add_raw_rows(table, rows)
    _safe_close_table(table)
    store._embedding_model = None

    result = store.search_with_null_vector_fallback(
        "alpha",
        k=1,
        filters={"metadata": {"user_id": 7}, "priority": "keep"},
    )

    assert [note.id for note in result] == ["eligible"]


def test_streaming_fallback_ranks_stably_across_batches(store):
    table = store._vector_store.get_raw_connection().open_table("memories")
    _add_raw_rows(
        table,
        [
            *[
                _raw_row(f"broad-{index:03d}", "prefix alpha", 7)
                for index in range(260)
            ],
            _raw_row("z-prefix", "alpha suffix", 7),
            _raw_row("a-prefix", "alpha suffix", 7),
        ],
    )
    _safe_close_table(table)
    store._embedding_model = None

    first = store.search_with_null_vector_fallback("alpha", k=2)
    second = store.search_with_null_vector_fallback("alpha", k=2)

    assert [note.id for note in first] == ["a-prefix", "z-prefix"]
    assert [note.id for note in second] == ["a-prefix", "z-prefix"]


def test_ann_duplicate_does_not_consume_streaming_lexical_quota(store):
    assert store.add(MemoryNote(id="ann", content="semantic winner")).success
    table = store._vector_store.get_raw_connection().open_table("memories")
    _add_raw_rows(
        table,
        [
            _raw_row("ann", "alpha", None),
            _raw_row("fallback", "alpha", None),
        ],
    )
    _safe_close_table(table)

    result = store.search_with_null_vector_fallback("alpha", k=2)

    assert [note.id for note in result] == ["ann", "fallback"]


def test_create_and_recreate_use_typed_vectors_and_canonical_identity(tmp_path):
    connection = lancedb.connect(tmp_path)
    assert (
        create_or_recreate_vector_capable_table(connection, "new", IDENTITY)
        is VectorCompatibility.MATCHING
    )
    table = connection.open_table("new")
    assert table.count_rows() == 0
    assert table.schema.field("vector").type == pa.list_(pa.float32(), 4)
    assert VECTOR_IDENTITY_METADATA_KEY in (table.schema.metadata or {})
    _safe_close_table(table)

    vectorless = connection.create_table(
        "legacy",
        pa.table(
            {
                "id": ["kept"],
                "text": pa.array([None], pa.string()),
                "metadata": [json.dumps({"user_id": 9})],
            }
        ),
    )
    _safe_close_table(vectorless)
    assert (
        create_or_recreate_vector_capable_table(connection, "legacy", IDENTITY)
        is VectorCompatibility.MATCHING
    )
    table = connection.open_table("legacy")
    assert table.to_arrow().to_pylist() == [
        {
            "id": "kept",
            "text": None,
            "metadata": json.dumps({"user_id": 9}),
            "vector": None,
            USER_ID_COLUMN: 9,
            SCOPE_DIMS_COLUMN: [],
        }
    ]
    _safe_close_table(table)


@pytest.mark.parametrize("failure_point", ["open", "create"])
def test_recreation_propagates_real_io_errors(tmp_path, failure_point):
    connection = lancedb.connect(tmp_path)
    table = connection.create_table(
        "memories", pa.table({"id": ["x"], "text": ["x"], "metadata": ["{}"]})
    )
    _safe_close_table(table)

    class FailingConnection:
        def list_tables(self):
            return ["memories"]

        def open_table(self, name):
            if failure_point == "open":
                raise OSError("real open failure")
            return connection.open_table(name)

        def create_table(self, *args, **kwargs):
            raise OSError("real create failure")

    with pytest.raises(OSError, match=f"real {failure_point} failure"):
        create_or_recreate_vector_capable_table(
            FailingConnection(), "memories", IDENTITY
        )
