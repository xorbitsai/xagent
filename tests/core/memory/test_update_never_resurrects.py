"""``update`` replaces a stored note only while it still exists.

``update_memory`` reads a note and then writes it back, and a delete (a user's or
an administrator's forget) can commit at any point of that, including while the
store's own ``update`` is in flight. ``update`` must then leave the note deleted:
it may never re-create the id. The interleaving tests below delete from inside the
concrete update -- while LanceDB computes the replacement embedding, before its
write; between the in-process store's membership check and its assignment -- so a
check-then-delete-then-add update, or an unserialized check-then-set one, fails
them.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import pytest

from xagent.core.agent.context.memory_tool import UpdateMemoryTool
from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.lancedb import LanceDBMemoryStore, _safe_close_table
from xagent.core.model.embedding import BaseEmbedding
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore


class _Embedding(BaseEmbedding):
    """A constant vector; ``during_next_encode`` runs once, inside the next encode,
    and ``fail`` makes encodes raise (after that hook)."""

    def __init__(self) -> None:
        self.during_next_encode: Callable[[], None] | None = None
        self.fail = False

    def encode(self, text, dimension=None, instruct=None):
        hook, self.during_next_encode = self.during_next_encode, None
        if hook is not None:
            hook()
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [0.1] * 8 if isinstance(text, str) else [[0.1] * 8 for _ in text]

    def get_dimension(self):
        return 8

    @property
    def abilities(self):
        return ["embed"]


def _lancedb(tmp_path, embedding: _Embedding) -> LanceDBMemoryStore:
    return LanceDBMemoryStore(
        db_dir=str(tmp_path), collection_name="memories", embedding_model=embedding
    )


def _column_names(store: LanceDBMemoryStore) -> set[str]:
    table = store._vector_store.get_raw_connection().open_table("memories")
    try:
        return set(table.schema.names)
    finally:
        _safe_close_table(table)


@pytest.fixture(params=["lancedb", "lancedb_text_only", "in_memory"])
def store(request, tmp_path) -> MemoryStore:
    if request.param == "in_memory":
        return InMemoryMemoryStore()
    if request.param == "lancedb_text_only":
        # No embedding adapter at all: the supported text-only schema, which has
        # no ``vector`` column (a failing adapter would still get one).
        text_only = LanceDBMemoryStore(
            db_dir=str(tmp_path), collection_name="memories", embedding_model=None
        )
        assert "vector" not in _column_names(text_only)
        return text_only
    return _lancedb(tmp_path, _Embedding())


def _ids(store: MemoryStore) -> list[str]:
    return sorted(str(note.id) for note in store.list_all())


def test_an_update_replaces_a_note_that_exists(store):
    kept = store.add(MemoryNote(content="kept")).memory_id
    memory_id = store.add(
        MemoryNote(content="prefers tea", metadata={"v": 1})
    ).memory_id

    response = store.update(
        MemoryNote(id=memory_id, content="prefers coffee", metadata={"v": 2})
    )

    assert response.success and response.memory_id == memory_id
    stored = store.get(memory_id).content
    assert (stored.content, stored.metadata["v"]) == ("prefers coffee", 2)
    assert _ids(store) == sorted([kept, memory_id])


def test_an_update_after_a_delete_reports_not_found(store):
    """The plain contract: an update of an id that is gone writes nothing."""
    memory_id = store.add(MemoryNote(content="prefers tea")).memory_id
    read = store.get(memory_id).content
    assert store.delete(memory_id).success

    read.content = "prefers coffee"
    response = store.update(read)

    assert not response.success
    assert _ids(store) == []


@pytest.mark.parametrize(
    "failing_embedding", [False, True], ids=["embedded", "unembedded"]
)
def test_a_delete_during_a_lancedb_update_is_not_undone(tmp_path, failing_embedding):
    """The forget commits while the update computes the replacement's embedding --
    after any existence check an update could make, before its write."""
    embedding = _Embedding()
    store = _lancedb(tmp_path, embedding)
    memory_id = store.add(MemoryNote(content="prefers tea")).memory_id
    read = store.get(memory_id).content
    embedding.fail = failing_embedding
    embedding.during_next_encode = lambda: store.delete(memory_id)

    read.content = "prefers coffee"
    response = store.update(read)

    assert embedding.during_next_encode is None  # the delete ran inside the update
    assert not response.success
    assert _ids(store) == []
    assert not store.get(memory_id).success


def test_a_delete_during_an_in_memory_update_is_not_undone():
    """The forget arrives between the update's membership check and its write:
    serialized behind the update, it removes the replacement too."""
    store = InMemoryMemoryStore()
    memory_id = store.add(MemoryNote(content="prefers tea")).memory_id
    forget = threading.Thread(target=store.delete, args=(memory_id,))

    class _Notes(dict):
        def __contains__(self, key):
            found = super().__contains__(key)
            if key == memory_id and forget.ident is None:
                forget.start()
                # Long enough for an unserialized delete to land before the write.
                forget.join(timeout=0.5)
            return found

    store._store = _Notes(store._store)

    store.update(MemoryNote(id=memory_id, content="prefers coffee"))
    forget.join(timeout=5)

    assert forget.ident is not None and not forget.is_alive()
    assert _ids(store) == []


def test_update_memory_racing_a_forget_reports_not_found(tmp_path):
    """The tool's own read-then-write, with the forget landing inside the store's
    update."""
    embedding = _Embedding()
    store = _lancedb(tmp_path, embedding)
    memory_id = store.add(MemoryNote(content="prefers tea")).memory_id
    embedding.during_next_encode = lambda: store.delete(memory_id)
    tool = UpdateMemoryTool(memory_store=store)

    result = asyncio.run(tool.execute(memory_id=memory_id, content="prefers coffee"))

    assert result["success"] is False
    assert _ids(store) == []


def test_the_user_isolated_wrapper_never_recreates_a_deleted_note(tmp_path):
    embedding = _Embedding()
    base = _lancedb(tmp_path, embedding)
    isolated = UserIsolatedMemoryStore(base)
    with UserContext(7):
        memory_id = isolated.add(MemoryNote(content="prefers tea")).memory_id
        read = isolated.get(memory_id).content
        embedding.during_next_encode = lambda: base.delete(memory_id)

        read.content = "prefers coffee"
        response = isolated.update(read)

    assert not response.success
    assert _ids(base) == []


def test_an_update_without_a_replacement_embedding_clears_the_old_vector(tmp_path):
    """A matched merge keeps a column the update leaves out; the new text must not
    be ranked by its former content's vector."""
    embedding = _Embedding()
    store = _lancedb(tmp_path, embedding)
    memory_id = store.add(MemoryNote(content="prefers tea")).memory_id
    embedding.fail = True

    response = store.update(MemoryNote(id=memory_id, content="prefers coffee"))

    assert response.success
    table = store._vector_store.get_raw_connection().open_table("memories")
    try:
        rows = table.to_arrow().to_pylist()
    finally:
        _safe_close_table(table)
    (row,) = rows
    assert (row["text"], row["vector"]) == ("prefers coffee", None)
