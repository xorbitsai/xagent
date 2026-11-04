"""Tests for the KB maintenance compatibility facade."""

from __future__ import annotations

from inspect import signature
from typing import cast

import pytest


def test_kb_maintenance_compatibility_public_surface_imports() -> None:
    """Given the KB package, the maintenance facade is publicly importable."""
    import xagent.core.tools.core.RAG_tools.kb as kb

    coordinator = kb.get_kb_coordinator()

    assert hasattr(kb, "CollectionConfigSnapshot")
    assert hasattr(kb, "CollectionRollbackMaintenanceResult")
    assert hasattr(kb, "KBMaintenanceCompatibilityFacade")
    assert hasattr(coordinator, "maintenance_compatibility")
    assert coordinator.maintenance_compat is coordinator.maintenance_compatibility


def test_kb_maintenance_compatibility_methods_match_public_helper_signatures() -> None:
    """Given legacy helpers, facade methods preserve their call signatures."""
    from xagent.core.tools.core.RAG_tools.kb import KBMaintenanceCompatibilityFacade
    from xagent.core.tools.core.RAG_tools.management import collection_manager

    facade = KBMaintenanceCompatibilityFacade()
    pairs = [
        (facade.get_collection_sync, collection_manager.get_collection_sync),
        (
            facade.initialize_collection_embedding_sync,
            collection_manager.initialize_collection_embedding_sync,
        ),
        (
            facade.validate_document_processing_sync,
            collection_manager.validate_document_processing_sync,
        ),
        (
            facade.update_collection_stats_sync,
            collection_manager.update_collection_stats_sync,
        ),
        (
            facade.mark_collection_accessed_sync,
            collection_manager.mark_collection_accessed_sync,
        ),
        (
            facade.delete_collection_metadata_sync,
            collection_manager.delete_collection_metadata_sync,
        ),
        (
            facade.capture_collection_config_snapshot,
            collection_manager.capture_collection_config_snapshot,
        ),
        (
            facade.capture_collection_config_snapshot_sync,
            collection_manager.capture_collection_config_snapshot_sync,
        ),
        (
            facade.restore_collection_config_snapshot,
            collection_manager.restore_collection_config_snapshot,
        ),
        (
            facade.restore_collection_config_snapshot_sync,
            collection_manager.restore_collection_config_snapshot_sync,
        ),
        (
            facade.cleanup_collection_metadata_after_rollback,
            collection_manager.cleanup_collection_metadata_after_rollback,
        ),
        (
            facade.cleanup_collection_metadata_after_rollback_sync,
            collection_manager.cleanup_collection_metadata_after_rollback_sync,
        ),
        (
            facade.rebuild_collection_stats,
            collection_manager.rebuild_collection_stats,
        ),
        (
            facade.rebuild_collection_stats_sync,
            collection_manager.rebuild_collection_stats_sync,
        ),
        (
            facade.resolve_effective_embedding_model_sync,
            collection_manager.resolve_effective_embedding_model_sync,
        ),
        (
            facade.rebuild_collection_metadata,
            collection_manager.rebuild_collection_metadata,
        ),
    ]

    for facade_method, public_helper in pairs:
        assert signature(facade_method) == signature(public_helper)


def test_public_maintenance_helper_delegates_through_facade(monkeypatch) -> None:
    """Given a public maintenance helper call, it routes through the facade."""
    from xagent.core.tools.core.RAG_tools.management import collection_manager

    class _FakeFacade:
        def get_collection_sync(self, collection_name: str) -> str:
            assert collection_name == "kb"
            return "facade-collection"

    monkeypatch.setattr(
        collection_manager,
        "_get_maintenance_compatibility_facade",
        lambda: _FakeFacade(),
    )

    assert collection_manager.get_collection_sync("kb") == "facade-collection"


def test_reset_helpers_keep_maintenance_facade_reusable(monkeypatch) -> None:
    """Given reset helpers, the manager/facade can be reused in one process."""
    from xagent.core.tools.core.RAG_tools.kb import (
        get_kb_coordinator,
        reset_kb_coordinator_for_tests,
    )
    from xagent.core.tools.core.RAG_tools.management import collection_manager

    class _FakeManager:
        async def get_collection(self, collection_name: str) -> str:
            return f"collection:{collection_name}"

    monkeypatch.setattr(collection_manager, "collection_manager", _FakeManager())

    first_facade = get_kb_coordinator().maintenance_compatibility
    assert first_facade.get_collection_sync("before") == "collection:before"

    collection_manager.reset_locks_for_testing()
    reset_kb_coordinator_for_tests()

    second_facade = get_kb_coordinator().maintenance_compatibility
    assert second_facade.get_collection_sync("after") == "collection:after"
    assert second_facade is not first_facade


@pytest.mark.asyncio
async def test_coordinator_maintenance_facade_uses_instance_storage_for_config_snapshot() -> (
    None
):
    """Given injected storage, maintenance calls use coordinator-owned stores."""
    from xagent.core.tools.core.RAG_tools.kb import KBCoordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory

    class MetadataStore:
        def __init__(self) -> None:
            self.config_calls: list[dict[str, object]] = []

        async def get_collection_config(
            self, collection_name: str, user_id: int | None, is_admin: bool = False
        ) -> str | None:
            self.config_calls.append(
                {
                    "collection_name": collection_name,
                    "user_id": user_id,
                    "is_admin": is_admin,
                }
            )
            return '{"source":"injected"}'

    class StorageFactoryStub:
        def __init__(self, metadata_store: MetadataStore) -> None:
            self.metadata_store = metadata_store

        def get_metadata_store(self) -> MetadataStore:
            return self.metadata_store

    metadata_store = MetadataStore()
    coordinator = KBCoordinator(
        storage_factory=cast(StorageFactory, StorageFactoryStub(metadata_store))
    )

    snapshot = (
        await coordinator.maintenance_compatibility.capture_collection_config_snapshot(
            "docs", 7
        )
    )

    assert snapshot.config_json == '{"source":"injected"}'
    assert snapshot.existed
    assert metadata_store.config_calls == [
        {
            "collection_name": "docs",
            "user_id": 7,
            "is_admin": False,
        }
    ]


@pytest.mark.asyncio
async def test_coordinator_maintenance_sync_helper_keeps_storage_binding_in_running_loop() -> (
    None
):
    """Given an active event loop, sync maintenance helpers keep storage binding."""
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.kb import KBCoordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory

    class MetadataStore:
        def __init__(self) -> None:
            self.collection_calls: list[str] = []

        async def get_collection(self, collection_name: str) -> CollectionInfo:
            self.collection_calls.append(collection_name)
            return CollectionInfo(name=collection_name, documents=12)

    class StorageFactoryStub:
        def __init__(self, metadata_store: MetadataStore) -> None:
            self.metadata_store = metadata_store

        def get_metadata_store(self) -> MetadataStore:
            return self.metadata_store

    metadata_store = MetadataStore()
    coordinator = KBCoordinator(
        storage_factory=cast(StorageFactory, StorageFactoryStub(metadata_store))
    )

    collection = coordinator.maintenance_compatibility.get_collection_sync(
        "thread_bound_docs"
    )

    assert collection.name == "thread_bound_docs"
    assert collection.documents == 12
    assert metadata_store.collection_calls == ["thread_bound_docs"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("collection_name", "original_config", "mutated_config"),
    [
        (
            "direct_restore",
            '{"source":"direct","chunk_size":512}',
            '{"source":"direct","chunk_size":1024}',
        ),
        (
            "cloud_restore",
            '{"source":"cloud","bucket":"old"}',
            '{"source":"cloud","bucket":"new"}',
        ),
        ("web_restore", '{"source":"web","depth":1}', '{"source":"web","depth":2}'),
    ],
)
async def test_collection_config_snapshot_restores_failed_ingest_configs(
    collection_name: str, original_config: str, mutated_config: str
) -> None:
    """Given failed direct/cloud/web ingest, config snapshot restore reverts changes."""
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    facade = get_kb_coordinator().maintenance_compatibility
    user_id = 7

    await metadata_store.save_collection_config(
        collection_name, original_config, user_id
    )
    snapshot = await facade.capture_collection_config_snapshot(collection_name, user_id)
    await metadata_store.save_collection_config(
        collection_name, mutated_config, user_id
    )

    result = await facade.restore_collection_config_snapshot(
        snapshot,
        rollback_complete=True,
        side_effects_may_remain=False,
    )

    assert result.status == "restored"
    assert not result.skipped
    assert result.warnings == ()
    assert (
        await metadata_store.get_collection_config(
            collection_name, user_id, is_admin=False
        )
        == original_config
    )


@pytest.mark.asyncio
async def test_collection_config_snapshot_removes_new_config_after_complete_rollback() -> (
    None
):
    """Given no previous config, complete rollback removes the newly created config."""
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    facade = get_kb_coordinator().maintenance_compatibility
    collection_name = "new_config_removed"
    user_id = 19

    snapshot = await facade.capture_collection_config_snapshot(collection_name, user_id)
    assert not snapshot.existed

    await metadata_store.save_collection_config(
        collection_name, '{"source":"direct"}', user_id
    )
    result = await facade.restore_collection_config_snapshot(
        snapshot,
        rollback_complete=True,
        side_effects_may_remain=False,
    )

    assert result.status == "removed"
    assert not result.skipped
    assert result.cleanup_counts["config_rows"] == 1
    assert (
        await metadata_store.get_collection_config(
            collection_name, user_id, is_admin=False
        )
        is None
    )


@pytest.mark.asyncio
async def test_collection_config_restore_skips_when_side_effects_may_remain() -> None:
    """Given residual side effects, config cleanup/restore is left visible."""
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    facade = get_kb_coordinator().maintenance_compatibility
    collection_name = "config_restore_guarded"
    user_id = 23

    await metadata_store.save_collection_config(
        collection_name, '{"source":"direct","version":1}', user_id
    )
    snapshot = await facade.capture_collection_config_snapshot(collection_name, user_id)
    mutated_config = '{"source":"direct","version":2}'
    await metadata_store.save_collection_config(
        collection_name, mutated_config, user_id
    )

    result = await facade.restore_collection_config_snapshot(
        snapshot,
        rollback_complete=True,
        side_effects_may_remain=True,
    )

    assert result.status == "skipped"
    assert result.skipped
    assert result.reason == "side_effects_may_remain"
    assert result.side_effects_may_remain
    assert result.warnings
    assert (
        await metadata_store.get_collection_config(
            collection_name, user_id, is_admin=False
        )
        == mutated_config
    )


@pytest.mark.asyncio
async def test_metadata_cleanup_skips_when_rollback_incomplete_and_keeps_rows() -> None:
    """Given incomplete rollback, metadata/config stay visible with warning outcome."""
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    facade = get_kb_coordinator().maintenance_compatibility
    collection_name = "cleanup_incomplete_visible"
    user_id = 31

    await metadata_store.save_collection(
        CollectionInfo(name=collection_name, documents=1)
    )
    await metadata_store.save_collection_config(
        collection_name, '{"source":"web"}', user_id
    )

    result = await facade.cleanup_collection_metadata_after_rollback(
        collection_name,
        user_id,
        rollback_complete=False,
        side_effects_may_remain=False,
        delete_orphaned_metadata=True,
    )

    assert result.status == "skipped"
    assert result.skipped
    assert result.reason == "rollback_not_complete"
    assert result.warnings
    assert (
        await metadata_store.get_collection(collection_name)
    ).name == collection_name
    assert (
        await metadata_store.get_collection_config(
            collection_name, user_id, is_admin=False
        )
        == '{"source":"web"}'
    )


@pytest.mark.asyncio
async def test_metadata_cleanup_skips_when_side_effects_may_remain_and_keeps_rows() -> (
    None
):
    """Given possible residual artifacts, metadata/config cleanup is skipped."""
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    facade = get_kb_coordinator().maintenance_compatibility
    collection_name = "cleanup_side_effects_visible"
    user_id = 37

    await metadata_store.save_collection(
        CollectionInfo(name=collection_name, documents=2)
    )
    await metadata_store.save_collection_config(
        collection_name, '{"source":"cloud"}', user_id
    )

    result = await facade.cleanup_collection_metadata_after_rollback(
        collection_name,
        user_id,
        rollback_complete=True,
        side_effects_may_remain=True,
        delete_orphaned_metadata=True,
    )

    assert result.status == "skipped"
    assert result.skipped
    assert result.reason == "side_effects_may_remain"
    assert result.side_effects_may_remain
    assert result.warnings
    assert (
        await metadata_store.get_collection(collection_name)
    ).name == collection_name
    assert (
        await metadata_store.get_collection_config(
            collection_name, user_id, is_admin=False
        )
        == '{"source":"cloud"}'
    )


@pytest.mark.asyncio
async def test_metadata_cleanup_deletes_new_rows_after_complete_rollback() -> None:
    """Given complete rollback, newly created metadata/config can be removed."""
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    facade = get_kb_coordinator().maintenance_compatibility
    collection_name = "cleanup_complete_removed"
    user_id = 41

    await metadata_store.save_collection(
        CollectionInfo(name=collection_name, documents=1)
    )
    await metadata_store.save_collection_config(
        collection_name, '{"source":"direct"}', user_id
    )

    result = await facade.cleanup_collection_metadata_after_rollback(
        collection_name,
        user_id,
        rollback_complete=True,
        side_effects_may_remain=False,
        delete_orphaned_metadata=True,
    )

    assert result.status == "cleaned"
    assert not result.skipped
    assert result.cleanup_counts == {"metadata_rows": 1, "config_rows": 1}
    assert (
        await metadata_store.get_collection_config(
            collection_name, user_id, is_admin=False
        )
        is None
    )
    with pytest.raises(ValueError, match="not found"):
        await metadata_store.get_collection(collection_name)


@pytest.mark.asyncio
async def test_rebuild_collection_stats_refreshes_metadata_from_storage_state() -> None:
    """Given stale metadata, stats rebuild recomputes counts from storage state."""
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    facade = get_kb_coordinator().maintenance_compatibility
    collection_name = "stats_rebuild_after_rollback"
    extra_metadata = {"kb_storage": {"backend": "lancedb"}}

    await metadata_store.save_collection(
        CollectionInfo(
            name=collection_name,
            documents=9,
            processed_documents=8,
            parses=7,
            chunks=6,
            embeddings=5,
            document_names=["stale.pdf"],
            extra_metadata=extra_metadata,
        )
    )

    rebuilt = await facade.rebuild_collection_stats(collection_name)

    assert rebuilt is not None
    assert rebuilt.documents == 0
    assert rebuilt.processed_documents == 0
    assert rebuilt.parses == 0
    assert rebuilt.chunks == 0
    assert rebuilt.embeddings == 0
    assert rebuilt.document_names == []
    assert rebuilt.extra_metadata == extra_metadata

    stored = await metadata_store.get_collection(collection_name)
    assert stored.documents == 0
    assert stored.processed_documents == 0
    assert stored.parses == 0
    assert stored.chunks == 0
    assert stored.embeddings == 0
    assert stored.extra_metadata == extra_metadata


@pytest.mark.asyncio
async def test_rebuild_collection_stats_counts_through_the_facades_storage() -> None:
    """Given a coordinator with its own shim, stats rebuild counts through it."""
    from types import SimpleNamespace

    from xagent.core.tools.core.RAG_tools.kb import KBCoordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    reads: list[tuple[int | None, bool]] = []

    def aggregate(*, user_id: int | None, is_admin: bool) -> dict:
        reads.append((user_id, is_admin))
        return {"kb_shim": {"documents": 2, "parses": 1, "chunks": 3, "embeddings": 4}}

    metadata_store = get_metadata_store()
    shim = SimpleNamespace(
        get_vector_index_store=lambda: SimpleNamespace(
            aggregate_collection_stats=aggregate
        ),
        get_metadata_store=lambda: metadata_store,
    )
    facade = KBCoordinator(storage_shim=shim).maintenance_compatibility

    rebuilt = await facade.rebuild_collection_stats("kb_shim")
    synced = facade.rebuild_collection_stats_sync("kb_shim")

    assert rebuilt is not None and synced is not None
    assert (rebuilt.documents, rebuilt.chunks, rebuilt.embeddings) == (2, 3, 4)
    assert synced.embeddings == 4
    assert reads == [(None, True), (None, True)]


@pytest.mark.asyncio
async def test_rebuild_collection_stats_treats_null_storage_counts_as_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given nullable aggregate counts, stats rebuild stores zero counts."""
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator
    from xagent.core.tools.core.RAG_tools.kb.collection_handle import KBHandleProvider
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    metadata_store = get_metadata_store()
    facade = get_kb_coordinator().maintenance_compatibility
    collection_name = "stats_rebuild_nullable_counts"

    await metadata_store.save_collection(
        CollectionInfo(
            name=collection_name,
            documents=9,
            processed_documents=8,
            parses=7,
            chunks=6,
            embeddings=5,
            document_names=["stale.pdf"],
        )
    )

    class _FakeVectorIndexStore:
        def aggregate_collection_stats(
            self, *, user_id: int | None, is_admin: bool
        ) -> dict[str, dict[str, int | None]]:
            assert user_id is None
            assert is_admin is True
            return {
                collection_name: {
                    "documents": None,
                    "parses": None,
                    "chunks": None,
                    "embeddings": None,
                }
            }

    monkeypatch.setattr(
        KBHandleProvider,
        "aggregate_collection_stats",
        lambda _provider, **kwargs: _FakeVectorIndexStore().aggregate_collection_stats(
            **kwargs
        ),
    )

    rebuilt = await facade.rebuild_collection_stats(collection_name)

    assert rebuilt is not None
    assert rebuilt.documents == 0
    assert rebuilt.processed_documents == 0
    assert rebuilt.parses == 0
    assert rebuilt.chunks == 0
    assert rebuilt.embeddings == 0
    assert rebuilt.document_names == []

    stored = await metadata_store.get_collection(collection_name)
    assert stored.documents == 0
    assert stored.processed_documents == 0
    assert stored.parses == 0
    assert stored.chunks == 0
    assert stored.embeddings == 0
