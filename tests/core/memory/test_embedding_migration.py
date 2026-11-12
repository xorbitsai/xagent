"""Offline re-embedding migration of the LanceDB memory table."""

from __future__ import annotations

import json
import sqlite3

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest

import xagent.core.memory.embedding_migration as migration
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.embedding_migration import (
    MemoryReembedStatus,
    backup_table_name,
    reembed_memory_table,
    staging_table_name,
)
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.memory.scope_columns import SCOPE_DIMS_COLUMN, USER_ID_COLUMN
from xagent.core.memory.storage_admission import (
    DormantLanceDBMemoryHandle,
    admit_lancedb_memory_storage,
)
from xagent.core.memory.vector_compatibility import (
    VECTOR_IDENTITY_METADATA_KEY,
    EmbeddingIdentity,
    VectorCompatibility,
    _is_fully_admitted,
    classify_vector_compatibility,
    create_or_recreate_vector_capable_table,
)
from xagent.core.model.embedding import BaseEmbedding
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.providers.vector_store.lancedb import clear_connection_cache

OLD = EmbeddingIdentity(
    "openai", "text-embedding-3-small", "https://api.openai.com/v1/embeddings", 4, None
)
# Same dimension as OLD on purpose: nothing but the declared space can tell them
# apart, which is the case a config change alone silently mixes.
NEW = EmbeddingIdentity(
    "bailian", "qwen-embedding-v4", "https://example.test/v1", 4, None
)


class Encoder:
    """Records every batch it is asked to embed and returns a fixed value."""

    def __init__(self, value: float = 0.9, dimension: int = 4):
        self.value = value
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[self.value] * self.dimension for _ in texts]


def _row(note_id, text, user_id, value):
    return {
        "id": note_id,
        "text": text,
        "metadata": json.dumps({"content": text, "user_id": user_id}),
        "vector": [value] * 4,
        USER_ID_COLUMN: user_id,
        SCOPE_DIMS_COLUMN: [],
    }


def _live(connection, rows=None, identity=OLD):
    create_or_recreate_vector_capable_table(connection, "memories", identity)
    if rows:
        _append(connection, rows)
    return connection.open_table("memories")


def _append(connection, rows):
    table = connection.open_table("memories")
    try:
        table.add(pa.Table.from_pylist(rows, schema=table.schema))
    finally:
        _safe_close_table(table)


def _vectors(connection, name="memories"):
    table = connection.open_table(name)
    try:
        return table.to_arrow().to_pylist()
    finally:
        _safe_close_table(table)


def _names(connection):
    return {str(name) for name in connection.table_names()}


@pytest.fixture
def connection(tmp_path):
    clear_connection_cache()
    yield lancedb.connect(str(tmp_path))
    clear_connection_cache()


def _three_notes():
    return [
        _row("a", "alpha", 1, 0.1),
        _row("b", "beta", 1, 0.1),
        _row("c", "gamma", 2, 0.1),
    ]


def test_same_dimension_swap_replaces_the_declared_space_and_every_vector(connection):
    _live(connection, _three_notes())
    encoder = Encoder(0.9)

    outcome = reembed_memory_table(connection, "memories", NEW, encode=encoder)

    assert outcome.status is MemoryReembedStatus.COMPLETE
    assert (outcome.source_rows, outcome.reembedded_rows) == (3, 3)
    rows = _vectors(connection)
    assert [row["id"] for row in rows] == ["a", "b", "c"]
    assert all(row["vector"] == pytest.approx([0.9] * 4) for row in rows)
    # Owners and scope survive the rebuild; they still come from the metadata.
    assert [(row[USER_ID_COLUMN], row[SCOPE_DIMS_COLUMN]) for row in rows] == [
        (1, []),
        (1, []),
        (2, []),
    ]
    live = connection.open_table("memories")
    try:
        assert (
            classify_vector_compatibility(live.schema, NEW)
            is VectorCompatibility.MATCHING
        )
        assert (
            classify_vector_compatibility(live.schema, OLD)
            is VectorCompatibility.MISMATCHING
        )
        assert json.loads(live.schema.metadata[VECTOR_IDENTITY_METADATA_KEY]) == (
            NEW.as_dict()
        )
        assert _is_fully_admitted(live)
    finally:
        _safe_close_table(live)

    backup = _vectors(connection, backup_table_name("memories", NEW))
    assert all(row["vector"] == pytest.approx([0.1] * 4) for row in backup)
    assert backup_table_name("memories", NEW) in _names(connection)
    assert staging_table_name("memories", NEW) not in _names(connection)


def test_a_second_run_is_a_no_op(connection):
    _live(connection, _three_notes())
    assert (
        reembed_memory_table(connection, "memories", NEW, encode=Encoder()).status
        is MemoryReembedStatus.COMPLETE
    )
    encoder = Encoder()

    outcome = reembed_memory_table(connection, "memories", NEW, encode=encoder)

    assert outcome.status is MemoryReembedStatus.ALREADY_TARGET
    assert encoder.calls == []


def test_a_failed_staging_table_aborts_before_the_live_table_is_touched(
    connection, monkeypatch
):
    _live(connection, _three_notes())

    def drop_one(stage, batch=None):
        if stage == "batch_filled" and batch == 1:
            staging = connection.open_table(staging_table_name("memories", NEW))
            try:
                staging.delete("id = 'b'")
            finally:
                _safe_close_table(staging)

    monkeypatch.setattr(migration, "_checkpoint", drop_one)

    outcome = reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert outcome.status is MemoryReembedStatus.VERIFICATION_FAILED
    assert "missing" in outcome.detail
    live = connection.open_table("memories")
    try:
        assert live.count_rows() == 3
        assert (
            classify_vector_compatibility(live.schema, OLD)
            is VectorCompatibility.MATCHING
        )
    finally:
        _safe_close_table(live)
    assert backup_table_name("memories", NEW) not in _names(connection)


def test_staging_ids_must_match_and_not_merely_add_up(connection, monkeypatch):
    _live(connection, _three_notes())

    def swap_ids(stage, batch=None):
        if stage == "batch_filled" and batch == 1:
            staging = connection.open_table(staging_table_name("memories", NEW))
            try:
                staging.delete("id = 'b'")
                staging.add(
                    pa.Table.from_pylist(
                        [_row("ghost", "beta", 1, 0.9)], schema=staging.schema
                    )
                )
            finally:
                _safe_close_table(staging)

    monkeypatch.setattr(migration, "_checkpoint", swap_ids)

    outcome = reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert outcome.status is MemoryReembedStatus.VERIFICATION_FAILED
    assert "missing" in outcome.detail
    assert [row["id"] for row in _vectors(connection)] == ["a", "b", "c"]
    assert all(
        row["vector"] == pytest.approx([0.1] * 4) for row in _vectors(connection)
    )


def test_a_metadata_only_change_between_passes_is_carried(connection, monkeypatch):
    _live(connection, _three_notes())
    changed = []

    def change_metadata(stage, batch=None):
        if stage == "batch_filled" and not changed:
            changed.append(True)
            table = connection.open_table("memories")
            try:
                table.delete("id = 'b'")
            finally:
                _safe_close_table(table)
            _append(
                connection,
                [
                    {
                        **_row("b", "beta", 9, 0.1),
                        "metadata": json.dumps(
                            {"content": "beta", "user_id": 9, "tags": ["carried"]}
                        ),
                    }
                ],
            )

    monkeypatch.setattr(migration, "_checkpoint", change_metadata)

    outcome = reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert outcome.status is MemoryReembedStatus.COMPLETE
    assert outcome.passes == 2
    rows = {row["id"]: row for row in _vectors(connection)}
    assert rows["b"][USER_ID_COLUMN] == 9
    assert json.loads(rows["b"]["metadata"])["tags"] == ["carried"]


def _staging_table(connection):
    live = connection.open_table("memories")
    try:
        return migration._staging_table(
            connection, staging_table_name("memories", NEW), live.schema, NEW
        )
    finally:
        _safe_close_table(live)


def _staged_row(note_id, text, user_id, value):
    row = _row(note_id, text, user_id, 0.9)
    row["vector"] = None if value is None else [value] * 4
    return row


def test_a_staged_row_without_a_vector_fails_verification(connection):
    _live(connection, _three_notes())
    staging = _staging_table(connection)
    staging.add(
        pa.Table.from_pylist(
            [
                _staged_row("a", "alpha", 1, 0.9),
                _staged_row("b", "beta", 1, None),
                _staged_row("c", "gamma", 2, 0.9),
            ],
            schema=staging.schema,
        )
    )
    seen = sqlite3.connect(":memory:")
    seen.execute("CREATE TABLE source_ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
    for note_id in ("a", "b", "c"):
        seen.execute("INSERT INTO source_ids VALUES (?)", (note_id,))

    detail = migration._verify(
        connection,
        staging_table_name("memories", NEW),
        identity=NEW,
        source_rows=3,
        seen=seen,
        batch_size=512,
    )

    _safe_close_table(staging)
    seen.close()
    assert "carry no vector" in detail
    assert len(_vectors(connection)) == 3


def test_a_staged_row_without_a_vector_is_embedded_by_the_next_run(connection):
    _live(connection, _three_notes())
    staging = _staging_table(connection)
    staging.add(
        pa.Table.from_pylist(
            [
                _staged_row("a", "alpha", 1, 0.9),
                _staged_row("b", "beta", 1, None),
                _staged_row("c", "gamma", 2, 0.9),
            ],
            schema=staging.schema,
        )
    )
    _safe_close_table(staging)

    outcome = reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert outcome.status is MemoryReembedStatus.COMPLETE
    assert all(
        row["vector"] == pytest.approx([0.9] * 4) for row in _vectors(connection)
    )


def test_an_aborted_swap_leaves_no_backup_behind(connection, monkeypatch):
    _live(connection, _three_notes())

    def stop(stage, batch=None):
        if stage == "before_swap":
            raise RuntimeError("operator stopped the run")

    monkeypatch.setattr(migration, "_checkpoint", stop)

    with pytest.raises(RuntimeError, match="operator stopped"):
        reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert backup_table_name("memories", NEW) not in _names(connection)
    assert all(
        row["vector"] == pytest.approx([0.1] * 4) for row in _vectors(connection)
    )


def test_a_write_before_the_swap_joins_the_new_table_with_a_new_vector(
    connection, monkeypatch
):
    _live(connection, _three_notes())
    added = []

    def add_note(stage, batch=None):
        if stage == "batch_filled" and not added:
            added.append("later")
            _append(connection, [_row("later", "delta", 3, 0.1)])

    monkeypatch.setattr(migration, "_checkpoint", add_note)

    outcome = reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert outcome.status is MemoryReembedStatus.COMPLETE
    assert outcome.passes == 2
    rows = {row["id"]: row for row in _vectors(connection)}
    assert set(rows) == {"a", "b", "c", "later"}
    assert all(row["vector"] == pytest.approx([0.9] * 4) for row in rows.values())
    assert rows["later"][USER_ID_COLUMN] == 3


def test_notes_deleted_before_the_swap_do_not_survive(connection, monkeypatch):
    _live(connection, _three_notes())

    def delete_note(stage, batch=None):
        if stage == "batch_filled" and batch == 1:
            table = connection.open_table("memories")
            try:
                table.delete("id = 'c'")
            finally:
                _safe_close_table(table)

    monkeypatch.setattr(migration, "_checkpoint", delete_note)

    outcome = reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert outcome.status is MemoryReembedStatus.COMPLETE
    assert [row["id"] for row in _vectors(connection)] == ["a", "b"]


def test_writers_that_never_stop_leave_the_old_table_in_place(connection, monkeypatch):
    _live(connection, _three_notes())
    counter = []

    def keep_writing(stage, batch=None):
        if stage == "batch_filled":
            counter.append(1)
            _append(connection, [_row(f"w{len(counter)}", "delta", 4, 0.1)])

    monkeypatch.setattr(migration, "_checkpoint", keep_writing)

    outcome = reembed_memory_table(
        connection, "memories", NEW, encode=Encoder(), max_passes=3
    )

    assert outcome.status is MemoryReembedStatus.NOT_CONVERGED
    assert outcome.passes == 3
    live = connection.open_table("memories")
    try:
        assert (
            classify_vector_compatibility(live.schema, OLD)
            is VectorCompatibility.MATCHING
        )
        assert live.count_rows() == 3 + len(counter)
    finally:
        _safe_close_table(live)
    assert backup_table_name("memories", NEW) not in _names(connection)


def test_interrupted_run_resumes_and_only_embeds_what_is_missing(
    connection, monkeypatch
):
    _live(connection, _three_notes() + [_row("d", "delta", 2, 0.1)])

    def interrupt(stage, batch=None):
        if stage == "batch_filled" and batch == 1:
            raise RuntimeError("interrupted")

    monkeypatch.setattr(migration, "_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        reembed_memory_table(
            connection, "memories", NEW, encode=Encoder(), batch_size=2
        )
    assert staging_table_name("memories", NEW) in _names(connection)
    assert len(_vectors(connection)) == 4

    monkeypatch.setattr(migration, "_checkpoint", lambda *args: None)
    encoder = Encoder()
    outcome = reembed_memory_table(
        connection, "memories", NEW, encode=encoder, batch_size=2
    )

    assert outcome.status is MemoryReembedStatus.COMPLETE
    assert sorted(text for call in encoder.calls for text in call) == ["delta", "gamma"]
    assert all(
        row["vector"] == pytest.approx([0.9] * 4) for row in _vectors(connection)
    )


def test_embedding_failure_keeps_the_live_table(connection):
    _live(connection, _three_notes())

    def broken(_texts):
        raise RuntimeError("provider unavailable")

    outcome = reembed_memory_table(connection, "memories", NEW, encode=broken)

    assert outcome.status is MemoryReembedStatus.EMBEDDING_FAILED
    assert "provider unavailable" in outcome.detail
    assert all(
        row["vector"] == pytest.approx([0.1] * 4) for row in _vectors(connection)
    )
    assert backup_table_name("memories", NEW) not in _names(connection)


def test_a_vector_of_the_wrong_width_is_refused(connection):
    _live(connection, _three_notes())

    outcome = reembed_memory_table(
        connection, "memories", NEW, encode=Encoder(dimension=8)
    )

    assert outcome.status is MemoryReembedStatus.EMBEDDING_FAILED
    live = connection.open_table("memories")
    try:
        assert (
            classify_vector_compatibility(live.schema, OLD)
            is VectorCompatibility.MATCHING
        )
    finally:
        _safe_close_table(live)


def test_duplicate_source_ids_abort_the_migration(connection):
    _live(connection, _three_notes() + [_row("a", "alpha again", 1, 0.1)])

    outcome = reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert outcome.status is MemoryReembedStatus.INVALID_SOURCE_DATA
    assert "unique" in outcome.detail
    assert len(_vectors(connection)) == 4


def test_a_vectorless_table_is_rebuilt_with_vectors(connection):
    table = connection.create_table(
        "memories",
        pa.table(
            {
                "id": ["a", "b"],
                "text": ["alpha", "beta"],
                "metadata": [json.dumps({"user_id": 1})] * 2,
            }
        ),
    )
    _safe_close_table(table)

    outcome = reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert outcome.status is MemoryReembedStatus.COMPLETE
    rows = _vectors(connection)
    assert [row["vector"] for row in rows] == [
        pytest.approx([0.9] * 4),
        pytest.approx([0.9] * 4),
    ]
    assert [row[USER_ID_COLUMN] for row in rows] == [1, 1]


def test_an_absent_table_is_reported_not_created(connection):
    outcome = reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert outcome.status is MemoryReembedStatus.ABSENT
    assert "memories" not in _names(connection)


def test_a_staging_table_from_another_space_is_refused(connection):
    _live(connection, _three_notes())
    staging = connection.create_table(
        staging_table_name("memories", NEW), data=[_row("x", "x", 1, 0.9)]
    )
    _safe_close_table(staging)

    outcome = reembed_memory_table(connection, "memories", NEW, encode=Encoder())

    assert outcome.status is MemoryReembedStatus.INCOMPATIBLE_SCHEMA
    assert "does not hold the target vector space" in outcome.detail
    assert len(_vectors(connection)) == 3


class ConstantEmbedding(BaseEmbedding):
    def __init__(self, value: float = 0.5):
        self.value = value

    def encode(self, text, dimension=None, instruct=None):
        vector = [self.value] * 4
        return vector if isinstance(text, str) else [vector for _ in text]

    def get_dimension(self):
        return 4

    @property
    def abilities(self):
        return ["embed"]


def _admitted_store(connection, tmp_path):
    outcome = admit_lancedb_memory_storage(
        DormantLanceDBMemoryHandle(connection, "memories"),
        OLD,
        writers_quiesced=True,
    )
    assert outcome.admitted is not None
    return LanceDBMemoryStore.open_admitted(
        outcome.admitted,
        OLD,
        db_dir=str(tmp_path),
        embedding_model=ConstantEmbedding(),
    )


def test_a_store_that_lost_its_table_to_a_reembed_writes_no_old_vectors(
    connection, tmp_path
):
    _live(connection)
    store = _admitted_store(connection, tmp_path)
    assert store.add(MemoryNote(id="before", content="alpha")).success

    assert (
        reembed_memory_table(connection, "memories", NEW, encode=Encoder()).status
        is MemoryReembedStatus.COMPLETE
    )

    assert store.add(MemoryNote(id="after", content="beta")).success
    rows = {row["id"]: row for row in _vectors(connection)}
    assert rows["before"]["vector"] == pytest.approx([0.9] * 4)
    assert rows["after"]["vector"] is None
    # The store's old adapter never ranked the new table either.
    assert [note.id for note in store.search("beta", k=1)] == ["after"]
    assert all(
        row["vector"] in (None, pytest.approx([0.9] * 4)) for row in rows.values()
    )
