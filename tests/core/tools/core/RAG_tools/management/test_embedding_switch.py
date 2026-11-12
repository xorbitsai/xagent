"""Switch/rollback primitive and every writer that could put a stale model back."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import CollectionNotFrozenError
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.kb import (
    KBApiCompatibilityFacade,
    KBPipelineCompatibilityFacade,
    KBToolCompatibilityFacade,
)
from xagent.core.tools.core.RAG_tools.management import collection_manager as cm
from xagent.core.tools.core.RAG_tools.management import (
    collections,
)
from xagent.core.tools.core.RAG_tools.management.embedding_switch import (
    switch_collection_embedding_model,
)
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    IDLE,
    CollectionMigrationState,
    read_migration_state,
    write_migration_state,
)
from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store
from xagent.core.tools.core.RAG_tools.utils import migration_utils

OLD_MODEL, OLD_DIM = "model-old", 4
NEW_MODEL, NEW_DIM = "model-new", 8


def _bind_models(monkeypatch: pytest.MonkeyPatch) -> None:
    models = {
        "old": (OLD_MODEL, OLD_DIM),
        "new": (NEW_MODEL, NEW_DIM),
        "new-alias": (NEW_MODEL, NEW_DIM),
    }

    def _resolve(model_id: str, **_: Any) -> tuple[SimpleNamespace, object]:
        model, dimension = models[model_id]
        return SimpleNamespace(id=model, dimension=dimension), object()

    monkeypatch.setattr(cm, "resolve_embedding_adapter", _resolve)
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.embedding_switch."
        "resolve_embedding_adapter",
        _resolve,
    )


async def _save(
    name: str = "kb", model: str = OLD_MODEL, dim: int = OLD_DIM, **kw: Any
):
    await cm.collection_manager.save_collection(
        CollectionInfo(
            name=name, embedding_model_id=model, embedding_dimension=dim, **kw
        )
    )


def _save_sync(name: str = "kb", **kw: Any) -> None:
    asyncio.run(_save(name, **kw))


def _freeze(name: str = "kb") -> None:
    write_migration_state(name, CollectionMigrationState(state=FROZEN))


def _switch(monkeypatch: pytest.MonkeyPatch, name: str = "kb") -> None:
    _save_sync(name)
    _freeze(name)
    _bind_models(monkeypatch)
    switch_collection_embedding_model(name, "new")


def _binding(name: str = "kb") -> tuple[Optional[str], Optional[int]]:
    row = asyncio.run(cm.collection_manager.get_collection(name))
    return row.embedding_model_id, row.embedding_dimension


def test_switch_sets_both_binding_fields_and_records_the_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_sync(documents=3)
    _freeze()
    _bind_models(monkeypatch)

    result = switch_collection_embedding_model("kb", "new-alias")

    assert _binding() == (NEW_MODEL, NEW_DIM)
    assert asyncio.run(cm.collection_manager.get_collection("kb")).documents == 3
    state = read_migration_state("kb")
    assert state.state == FROZEN
    assert state.source_model_id == OLD_MODEL
    assert state.target_model_id == NEW_MODEL
    assert result.target_dimension == NEW_DIM


def test_switch_writes_only_the_binding_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_sync()
    _freeze()
    _bind_models(monkeypatch)
    store = get_metadata_store()
    real_update = store.update_collection_fields
    seen: list[tuple[str, ...]] = []

    async def _spy(collection: CollectionInfo, field_names: Any, **kw: Any) -> bool:
        seen.append(tuple(field_names))
        return await real_update(collection, field_names, **kw)

    monkeypatch.setattr(store, "update_collection_fields", _spy)

    switch_collection_embedding_model("kb", "new")

    assert seen == [("embedding_model_id", "embedding_dimension")]


def test_switch_refuses_a_collection_that_is_not_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_sync()
    _bind_models(monkeypatch)

    with pytest.raises(CollectionNotFrozenError):
        switch_collection_embedding_model("kb", "new")

    assert _binding() == (OLD_MODEL, OLD_DIM)


def test_switch_refuses_an_idle_migration_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_sync()
    write_migration_state("kb", CollectionMigrationState(state=IDLE))
    _bind_models(monkeypatch)

    with pytest.raises(CollectionNotFrozenError):
        switch_collection_embedding_model("kb", "new")

    assert _binding() == (OLD_MODEL, OLD_DIM)


def test_switch_refuses_a_model_without_a_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_sync()
    _freeze()
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.embedding_switch."
        "resolve_embedding_adapter",
        lambda model_id, **_: (SimpleNamespace(id=model_id, dimension=None), object()),
    )

    with pytest.raises(ValueError, match="no dimension"):
        switch_collection_embedding_model("kb", "new")

    assert _binding() == (OLD_MODEL, OLD_DIM)
    assert read_migration_state("kb").target_model_id is None


def test_switch_refuses_a_missing_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze("ghost")
    _bind_models(monkeypatch)

    with pytest.raises(ValueError):
        switch_collection_embedding_model("ghost", "new")


def test_rollback_is_the_same_call_with_the_previous_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    switch_collection_embedding_model("kb", "old")

    assert _binding() == (OLD_MODEL, OLD_DIM)
    state = read_migration_state("kb")
    assert state.source_model_id == NEW_MODEL
    assert state.target_model_id == OLD_MODEL


def test_partial_updates_retry_commit_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    store = get_metadata_store()
    real = store.update_collection_fields
    calls = {"n": 0}

    async def _flaky(collection, field_names, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("commit conflict")
        return await real(collection, field_names, **kw)

    monkeypatch.setattr(store, "update_collection_fields", _flaky)

    cm.update_collection_stats_sync("kb", documents_delta=1)

    assert calls["n"] == 2
    row = asyncio.run(store.get_collection("kb"))
    assert row.documents == 1
    assert (row.embedding_model_id, row.embedding_dimension) == (NEW_MODEL, NEW_DIM)


def test_stats_update_cannot_revert_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    real_get = cm.collection_manager.get_collection
    stale = CollectionInfo(
        name="kb", embedding_model_id=OLD_MODEL, embedding_dimension=OLD_DIM
    )
    calls = {"n": 0}

    async def _stale_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        return stale if calls["n"] == 1 else await real_get(name)

    monkeypatch.setattr(cm.collection_manager, "get_collection", _stale_once)

    cm.update_collection_stats_sync("kb", documents_delta=1)

    row = asyncio.run(real_get("kb"))
    assert (row.embedding_model_id, row.embedding_dimension) == (NEW_MODEL, NEW_DIM)
    assert row.documents == 1


def test_access_timestamp_update_touches_only_that_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    before = asyncio.run(cm.collection_manager.get_collection("kb"))

    cm.mark_collection_accessed_sync("kb")

    row = asyncio.run(cm.collection_manager.get_collection("kb"))
    assert row.last_accessed_at > before.last_accessed_at
    assert (row.embedding_model_id, row.embedding_dimension) == (NEW_MODEL, NEW_DIM)
    assert row.documents == before.documents


def test_stats_rebuild_cannot_revert_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    store = get_metadata_store()
    stale = CollectionInfo(
        name="kb",
        embedding_model_id=OLD_MODEL,
        embedding_dimension=OLD_DIM,
        documents=2,
    )

    async def _stale(_: str) -> CollectionInfo:
        return stale

    real_get = store.get_collection
    calls = {"n": 0}

    async def _stale_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        return stale if calls["n"] == 1 else await real_get(name)

    monkeypatch.setattr(store, "get_collection", _stale_once)
    coordinator = SimpleNamespace(
        aggregate_collection_stats_sync=lambda **_: {
            "kb": {"documents": 5, "parses": 5, "chunks": 5, "embeddings": 5}
        }
    )

    asyncio.run(cm._rebuild_collection_stats_impl("kb", coordinator))

    assert _binding() == (NEW_MODEL, NEW_DIM)


def test_inferred_model_cannot_overwrite_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    real_get = cm.get_collection_sync
    stale = CollectionInfo(name="kb", embeddings=5)
    calls = {"n": 0}

    def _stale_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        return stale if calls["n"] == 1 else real_get(name)

    monkeypatch.setattr(cm, "get_collection_sync", _stale_once)
    monkeypatch.setattr(
        migration_utils,
        "_infer_embedding_config_from_collection",
        lambda _name: (OLD_MODEL, OLD_DIM),
    )

    resolved = cm.resolve_effective_embedding_model_sync("kb", None)

    assert resolved == NEW_MODEL
    assert _binding() == (NEW_MODEL, NEW_DIM)


def test_initialization_cannot_revert_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    real_get = cm.collection_manager.get_collection
    stale = CollectionInfo(name="kb")
    calls = {"n": 0}

    async def _stale_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        return stale if calls["n"] == 1 else await real_get(name)

    monkeypatch.setattr(cm.collection_manager, "get_collection", _stale_once)

    with pytest.raises(ValueError, match="already initialized"):
        asyncio.run(cm.collection_manager.initialize_collection_embedding("kb", "old"))

    assert _binding() == (NEW_MODEL, NEW_DIM)


def test_initialization_cannot_revert_a_switch_when_the_row_looks_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    _bind_models(monkeypatch)
    real_get = cm.collection_manager.get_collection
    calls = {"n": 0}

    async def _missing_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(f"Collection '{name}' not found")
        return await real_get(name)

    monkeypatch.setattr(cm.collection_manager, "get_collection", _missing_once)

    with pytest.raises(ValueError, match="already initialized"):
        asyncio.run(cm.collection_manager.initialize_collection_embedding("kb", "old"))

    assert _binding() == (NEW_MODEL, NEW_DIM)


def test_alias_canonicalization_cannot_revert_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    real_get = cm.collection_manager.get_collection
    stale = CollectionInfo(
        name="kb", embedding_model_id=OLD_MODEL, embedding_dimension=OLD_DIM
    )
    calls = {"n": 0}

    async def _stale_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        return stale if calls["n"] == 1 else await real_get(name)

    monkeypatch.setattr(cm.collection_manager, "get_collection", _stale_once)
    monkeypatch.setattr(
        cm, "_embedding_model_ids_equivalent", lambda _a, _b: (True, OLD_MODEL, OLD_DIM)
    )

    with pytest.raises(ValueError, match="already initialized"):
        asyncio.run(
            cm.collection_manager.initialize_collection_embedding("kb", "old-alias")
        )

    assert _binding() == (NEW_MODEL, NEW_DIM)


def test_metadata_rebuild_cannot_clear_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    stale = CollectionInfo(name="kb")
    store = get_metadata_store()
    real_get = store.get_collection
    calls = {"n": 0}

    async def _stale_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        return stale if calls["n"] == 1 else await real_get(name)

    async def _list_collections(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(status="success", collections=[stale], message="")

    monkeypatch.setattr(store, "get_collection", _stale_once)
    monkeypatch.setattr(collections, "list_collections", _list_collections)
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.kb.collection_handle.ledger_holds_vectors",
        lambda: False,
    )
    monkeypatch.setattr(cm, "get_vector_index_store", lambda: SimpleNamespace())
    monkeypatch.setattr(
        collections,
        "_resolve_coordinator",
        lambda _coordinator: SimpleNamespace(
            aggregate_collection_stats_sync=lambda **_: {
                "kb": {"documents": 0, "parses": 0, "chunks": 0, "embeddings": 0}
            }
        ),
    )

    asyncio.run(cm._rebuild_collection_metadata_impl())

    row = asyncio.run(real_get("kb"))
    assert (row.embedding_model_id, row.embedding_dimension) == (NEW_MODEL, NEW_DIM)


def test_reapplying_a_switch_keeps_the_rollback_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)

    result = switch_collection_embedding_model("kb", "new")

    assert result.source_model_id == OLD_MODEL
    state = read_migration_state("kb")
    assert state.source_model_id == OLD_MODEL
    assert state.target_model_id == NEW_MODEL


def test_missing_row_update_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = get_metadata_store()
    real = store.update_collection_fields
    calls = {"n": 0}

    async def _spy(*args, **kwargs):
        calls["n"] += 1
        return await real(*args, **kwargs)

    monkeypatch.setattr(store, "update_collection_fields", _spy)

    cm.mark_collection_accessed_sync("ghost")

    assert calls["n"] == 1


def test_same_model_initialization_race_returns_the_stored_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_sync(documents=3)
    _bind_models(monkeypatch)
    real_get = cm.collection_manager.get_collection
    stale = CollectionInfo(name="kb")
    calls = {"n": 0}

    async def _stale_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        return stale if calls["n"] == 1 else await real_get(name)

    monkeypatch.setattr(cm.collection_manager, "get_collection", _stale_once)

    result = asyncio.run(
        cm.collection_manager.initialize_collection_embedding("kb", "old")
    )

    assert result.embedding_model_id == OLD_MODEL
    assert result.documents == 3
    assert _binding() == (OLD_MODEL, OLD_DIM)


def test_stats_auto_create_cannot_clear_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    real_get = cm.collection_manager.get_collection

    async def _missing(_: str) -> CollectionInfo:
        raise ValueError("Collection 'kb' not found")

    monkeypatch.setattr(cm.collection_manager, "get_collection", _missing)

    cm.update_collection_stats_sync("kb", documents_delta=1)

    row = asyncio.run(real_get("kb"))
    assert (row.embedding_model_id, row.embedding_dimension) == (NEW_MODEL, NEW_DIM)
    assert row.documents == 1


def test_backend_bindings_create_a_missing_row() -> None:
    store = get_metadata_store()

    asyncio.run(
        KBApiCompatibilityFacade()._ensure_collection_backend_binding_with_store(
            "api-demo", store
        )
    )
    asyncio.run(
        KBToolCompatibilityFacade().ensure_agent_collection_backend_binding("tool-demo")
    )

    for name in ("api-demo", "tool-demo"):
        row = asyncio.run(store.get_collection(name))
        assert row.extra_metadata["kb_storage"] == {"backend": "lancedb"}


def test_backend_binding_keeps_an_existing_binding() -> None:
    _save_sync(name="demo")
    store = get_metadata_store()

    asyncio.run(
        KBApiCompatibilityFacade()._ensure_collection_backend_binding_with_store(
            "demo", store
        )
    )

    assert _binding("demo") == (OLD_MODEL, OLD_DIM)
    assert asyncio.run(store.get_collection("demo")).extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }


def test_api_binding_does_not_overwrite_a_row_that_appeared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_sync(name="demo")
    store = get_metadata_store()
    real_get = store.get_collection
    calls = {"n": 0}

    async def _missing_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(f"Collection '{name}' not found")
        return await real_get(name)

    monkeypatch.setattr(store, "get_collection", _missing_once)

    asyncio.run(
        KBApiCompatibilityFacade()._ensure_collection_backend_binding_with_store(
            "demo", store
        )
    )

    assert _binding("demo") == (OLD_MODEL, OLD_DIM)
    assert asyncio.run(real_get("demo")).extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }


def test_tool_binding_does_not_overwrite_a_row_that_appeared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_sync(name="demo")
    store = get_metadata_store()
    real_get = store.get_collection
    calls = {"n": 0}

    async def _missing_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(f"Collection '{name}' not found")
        return await real_get(name)

    monkeypatch.setattr(store, "get_collection", _missing_once)

    asyncio.run(
        KBToolCompatibilityFacade().ensure_agent_collection_backend_binding("demo")
    )

    assert _binding("demo") == (OLD_MODEL, OLD_DIM)
    assert asyncio.run(real_get("demo")).extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }


def test_metadata_rebuild_creation_cannot_clear_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    stale = CollectionInfo(name="kb")

    async def _list_collections(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(status="success", collections=[stale], message="")

    async def _missing(_name: str) -> None:
        return None

    monkeypatch.setattr(cm, "_read_stored_collection", _missing)
    monkeypatch.setattr(collections, "list_collections", _list_collections)
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.kb.collection_handle.ledger_holds_vectors",
        lambda: False,
    )
    monkeypatch.setattr(cm, "get_vector_index_store", lambda: SimpleNamespace())
    monkeypatch.setattr(
        collections,
        "_resolve_coordinator",
        lambda _coordinator: SimpleNamespace(
            aggregate_collection_stats_sync=lambda **_: {
                "kb": {"documents": 0, "parses": 0, "chunks": 0, "embeddings": 0}
            }
        ),
    )

    asyncio.run(cm._rebuild_collection_metadata_impl())

    assert _binding() == (NEW_MODEL, NEW_DIM)


def test_stats_rebuild_creation_cannot_clear_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _switch(monkeypatch)
    store = get_metadata_store()
    real_get = store.get_collection
    calls = {"n": 0}

    async def _missing_once(name: str) -> CollectionInfo:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(f"Collection '{name}' not found")
        return await real_get(name)

    monkeypatch.setattr(store, "get_collection", _missing_once)
    coordinator = SimpleNamespace(
        aggregate_collection_stats_sync=lambda **_: {
            "kb": {"documents": 5, "parses": 5, "chunks": 5, "embeddings": 5}
        }
    )

    asyncio.run(cm._rebuild_collection_stats_impl("kb", coordinator))

    assert _binding() == (NEW_MODEL, NEW_DIM)


class _UpdatingStore:
    def __init__(self, collection: CollectionInfo) -> None:
        self.collection = collection
        self.updated: list[tuple[tuple[str, ...], Any]] = []
        self.saved = 0

    async def get_collection(self, _name: str) -> CollectionInfo:
        return self.collection

    async def update_collection_fields(self, collection, field_names, **kw):
        self.updated.append((tuple(field_names), collection.extra_metadata))
        return True

    async def save_collection(self, collection) -> None:
        self.saved += 1


def test_backend_binding_updates_only_extra_metadata() -> None:
    store = _UpdatingStore(
        CollectionInfo(
            name="demo", embedding_model_id=NEW_MODEL, embedding_dimension=NEW_DIM
        )
    )
    facade = KBPipelineCompatibilityFacade(
        storage_shim=SimpleNamespace(get_metadata_store=lambda: store)
    )

    asyncio.run(facade.ensure_collection_backend_binding_async("demo"))

    assert store.saved == 0
    assert store.updated[0][0] == ("extra_metadata",)
