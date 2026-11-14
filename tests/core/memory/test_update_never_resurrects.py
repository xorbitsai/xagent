"""``update`` replaces a stored note only while it still exists.

``update_memory`` reads a note and then writes it back; a delete (a user's or an
administrator's forget) can commit in between. ``update`` must then report the
note as not found and leave it deleted: it may never re-create the id. Pinned for
every store, directly and through the user-isolation wrapper the runtime uses.
"""

from __future__ import annotations

import asyncio

import pytest

from xagent.core.agent.context.memory_tool import UpdateMemoryTool
from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.model.embedding import BaseEmbedding
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore


class _ConstantEmbedding(BaseEmbedding):
    def encode(self, text, dimension=None, instruct=None):
        return [0.1] * 8 if isinstance(text, str) else [[0.1] * 8 for _ in text]

    def get_dimension(self):
        return 8

    @property
    def abilities(self):
        return ["embed"]


class _FailingEmbedding(_ConstantEmbedding):
    """No vector for any note: the update's row carries no vector column."""

    def encode(self, text, dimension=None, instruct=None):
        raise RuntimeError("embedding unavailable")


@pytest.fixture(params=["lancedb", "lancedb_without_vectors", "in_memory"])
def store(request, tmp_path) -> MemoryStore:
    if request.param == "in_memory":
        return InMemoryMemoryStore()
    embedding = (
        _FailingEmbedding()
        if request.param == "lancedb_without_vectors"
        else _ConstantEmbedding()
    )
    return LanceDBMemoryStore(
        db_dir=str(tmp_path), collection_name="memories", embedding_model=embedding
    )


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


def test_an_update_of_a_deleted_note_never_recreates_it(store):
    """The forget lands between the updater's read and its write."""
    memory_id = store.add(MemoryNote(content="prefers tea")).memory_id
    read = store.get(memory_id).content
    assert store.delete(memory_id).success

    read.content = "prefers coffee"
    response = store.update(read)

    assert not response.success
    assert _ids(store) == []
    assert not store.get(memory_id).success


def test_update_memory_racing_a_forget_reports_not_found(store):
    """The tool's own read-then-write, with the forget committing in between."""
    memory_id = store.add(MemoryNote(content="prefers tea")).memory_id
    update = store.update

    def forgotten_first(note: MemoryNote):
        assert store.delete(str(note.id)).success
        return update(note)

    store.update = forgotten_first  # type: ignore[method-assign]
    tool = UpdateMemoryTool(memory_store=store)

    result = asyncio.run(tool.execute(memory_id=memory_id, content="prefers coffee"))

    assert result["success"] is False
    assert _ids(store) == []


def test_the_user_isolated_wrapper_never_recreates_a_deleted_note(store):
    isolated = UserIsolatedMemoryStore(store)
    with UserContext(7):
        memory_id = isolated.add(MemoryNote(content="prefers tea")).memory_id
        read = isolated.get(memory_id).content
        assert store.delete(memory_id).success

        read.content = "prefers coffee"
        response = isolated.update(read)

    assert not response.success
    assert _ids(store) == []
