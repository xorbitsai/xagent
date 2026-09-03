from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from xagent.core.memory.base import MemoryBackendUnavailableError
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.model.embedding import BaseEmbedding
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore


class RecordingEmbedding(BaseEmbedding):
    def __init__(self, dimension: int = 8, *, fail: bool = False) -> None:
        self.dimension = dimension
        self.fail = fail
        self.inputs: list[Any] = []

    def encode(self, text, dimension=None, instruct=None):
        self.inputs.append(text)
        if self.fail:
            raise RuntimeError("embedding unavailable")
        if isinstance(text, str):
            return [0.1] * self.dimension
        return [[0.1] * self.dimension for _ in text]

    def get_dimension(self):
        return self.dimension

    @property
    def abilities(self):
        return ["embed"]


def _identity(
    model: str = "embedding-a", dimension: int = 8, endpoint: str = "https://a"
) -> dict[str, Any]:
    return {
        "provider": "test",
        "model": model,
        "endpoint": endpoint,
        "dimension": dimension,
    }


def _store(tmp_path, embedding, identity=None):
    return LanceDBMemoryStore(
        db_dir=str(tmp_path),
        collection_name="memories",
        embedding_model=embedding,
        vector_space_identity=identity,
        allow_schema_migration=False,
    )


def test_strict_vector_success_uses_real_store(tmp_path) -> None:
    embedding = RecordingEmbedding()
    store = _store(tmp_path, embedding, _identity())

    store.ensure_required_vector_search()
    added = store.add_required_vector(MemoryNote(id="one", content="alpha"))
    results = store.search_required_vector("alpha")

    assert added.success
    assert [note.id for note in results] == ["one"]
    assert embedding.inputs == ["alpha", "alpha"]


def test_admission_is_read_only_and_operation_embedding_failure_is_typed(
    tmp_path,
) -> None:
    initial = _store(tmp_path, RecordingEmbedding(), _identity())
    assert initial.add_required_vector(MemoryNote(id="one", content="alpha")).success
    failing = RecordingEmbedding(fail=True)
    strict = _store(tmp_path, failing, _identity())

    strict.ensure_required_vector_search()
    assert failing.inputs == []
    with pytest.raises(MemoryBackendUnavailableError):
        strict.add_required_vector(MemoryNote(id="two", content="beta"))
    with pytest.raises(MemoryBackendUnavailableError):
        strict.search_required_vector("alpha")
    assert strict.get("one").success


def test_identity_mismatch_is_unavailable_but_ordinary_add_is_text_only(
    tmp_path,
) -> None:
    first = _store(tmp_path, RecordingEmbedding(), _identity("model-a"))
    assert first.add_required_vector(MemoryNote(id="old", content="old")).success
    replacement_embedding = RecordingEmbedding()
    replacement = _store(tmp_path, replacement_embedding, _identity("model-b"))

    with pytest.raises(MemoryBackendUnavailableError):
        replacement.ensure_required_vector_search()
    assert replacement.add(MemoryNote(id="new", content="new")).success
    assert replacement_embedding.inputs == []

    table = replacement._vector_store.get_raw_connection().open_table("memories")
    rows = {row["id"]: row["vector"] for row in table.to_arrow().to_pylist()}
    assert rows["old"] is not None
    assert rows["new"] is None
    assert first.ensure_required_vector_search() is None


def test_missing_vector_column_is_strictly_unavailable_but_text_still_works(
    tmp_path,
) -> None:
    text_store = _store(tmp_path, None)
    assert text_store.add(MemoryNote(id="old", content="old text")).success
    strict = _store(tmp_path, RecordingEmbedding(), _identity())

    with pytest.raises(MemoryBackendUnavailableError):
        strict.ensure_required_vector_search()
    with pytest.raises(MemoryBackendUnavailableError):
        strict.search_required_vector("old")
    assert [note.id for note in strict.search("old")] == ["old"]


def test_strict_search_converts_terminal_ann_failure_to_typed_unavailable(
    monkeypatch, tmp_path
) -> None:
    store = _store(tmp_path, RecordingEmbedding(), _identity())
    assert store.add_required_vector(MemoryNote(id="one", content="old")).success
    connection = store._vector_store.get_raw_connection()
    table = connection.open_table("memories")
    monkeypatch.setattr(connection, "open_table", lambda _name: table)
    monkeypatch.setattr(
        table,
        "search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ANN failed")),
    )

    with pytest.raises(MemoryBackendUnavailableError):
        store.search_required_vector("old")


def test_dimension_change_before_first_new_write_is_read_only(tmp_path) -> None:
    first = _store(tmp_path, RecordingEmbedding(8), _identity(dimension=8))
    assert first.add_required_vector(MemoryNote(id="old", content="old")).success
    changed = _store(
        tmp_path,
        RecordingEmbedding(16),
        _identity(dimension=16),
    )

    with pytest.raises(MemoryBackendUnavailableError):
        changed.ensure_required_vector_search()

    table = first._vector_store.get_raw_connection().open_table("memories")
    arrow = table.to_arrow()
    assert arrow.column("id").to_pylist() == ["old"]
    assert arrow.schema.field("vector").type.list_size == 8


def test_populated_shared_table_admission_never_scans_or_migrates_tenants(
    monkeypatch, tmp_path
) -> None:
    store = UserIsolatedMemoryStore(_store(tmp_path, RecordingEmbedding(), _identity()))
    with UserContext(1):
        assert store.add_required_vector(
            MemoryNote(id="one", content="secret-a")
        ).success
    with UserContext(2):
        assert store.add_required_vector(
            MemoryNote(id="two", content="secret-b")
        ).success

    def forbidden(*_args, **_kwargs):
        raise AssertionError("strict admission must not migrate or re-embed")

    monkeypatch.setattr("xagent.core.memory.lancedb.migrate_table_swap", forbidden)
    restarted_embedding = RecordingEmbedding()
    restarted = _store(tmp_path, restarted_embedding, _identity())
    restarted.ensure_required_vector_search()

    assert restarted_embedding.inputs == []
    assert {note.id for note in restarted.list_all()} == {"one", "two"}


def test_strict_update_failure_preserves_old_row(tmp_path) -> None:
    initial = _store(tmp_path, RecordingEmbedding(), _identity())
    assert initial.add_required_vector(MemoryNote(id="one", content="old")).success
    failing = _store(tmp_path, RecordingEmbedding(fail=True), _identity())

    with pytest.raises(MemoryBackendUnavailableError):
        failing.update_required_vector(MemoryNote(id="one", content="new"))

    existing = initial.get("one")
    assert existing.success
    assert existing.content.content == "old"


def test_strict_update_replaces_row_after_embedding_succeeds(tmp_path) -> None:
    store = _store(tmp_path, RecordingEmbedding(), _identity())
    assert store.add_required_vector(MemoryNote(id="one", content="old")).success

    updated = store.update_required_vector(MemoryNote(id="one", content="new"))

    assert updated.success
    assert store.get("one").content.content == "new"


def test_concurrent_read_only_admission_does_not_lose_writer(tmp_path) -> None:
    store = _store(tmp_path, RecordingEmbedding(), _identity())

    def admit() -> None:
        for _ in range(20):
            store.ensure_required_vector_search()

    def write() -> None:
        for index in range(20):
            result = store.add_required_vector(
                MemoryNote(id=f"row-{index}", content=f"text-{index}")
            )
            assert result.success

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda fn: fn(), (admit, write)))

    assert {note.id for note in store.list_all()} == {
        f"row-{index}" for index in range(20)
    }


def test_strict_wrapper_routes_add_search_and_update(tmp_path) -> None:
    base = _store(tmp_path, RecordingEmbedding(), _identity())
    strict = UserIsolatedMemoryStore(base, require_vector_search=True)

    assert strict.add(MemoryNote(id="one", content="old")).success
    assert strict.search("old")[0].id == "one"
    assert strict.update(MemoryNote(id="one", content="new")).success
    assert strict.get("one").content.content == "new"
