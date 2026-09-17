"""Real-LanceDB coverage for dormant memory admission primitives."""

import json
from types import SimpleNamespace

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
from xagent.web.dynamic_memory_store import DynamicMemoryStoreManager

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
        uri = connection.uri
        list_tables = connection.list_tables

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


def test_failed_manager_replacement_preserves_all_previous_state(monkeypatch):
    manager = DynamicMemoryStoreManager()
    previous_store = manager._memory_store
    manager._is_lancedb = True
    manager._last_embedding_model_id = 1
    manager._last_embedding_model_fingerprint = (1, "old")
    model = SimpleNamespace(id=2, updated_at="new")
    monkeypatch.setattr(manager, "_get_embedding_model_from_db", lambda: model)
    monkeypatch.setattr(
        manager,
        "_create_lancedb_store",
        lambda _model: (_ for _ in ()).throw(OSError("construction failed")),
    )

    manager._check_and_update_store()

    assert manager._memory_store is previous_store
    assert manager._is_lancedb is True
    assert manager._last_embedding_model_id == 1
    assert manager._last_embedding_model_fingerprint == (1, "old")
