"""Tests for the KB core management compatibility facade."""

from __future__ import annotations

import inspect
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_kb_management_facade_public_surface_imports() -> None:
    import xagent.core.tools.core.RAG_tools.kb as kb
    from xagent.core.tools.core.RAG_tools.kb import (
        KBCoreManagementCompatibilityFacade,
        get_kb_coordinator,
        reset_kb_coordinator_for_tests,
    )

    assert hasattr(kb, "KBCoreManagementCompatibilityFacade")
    reset_kb_coordinator_for_tests()
    assert isinstance(
        get_kb_coordinator().management,
        KBCoreManagementCompatibilityFacade,
    )


def test_management_package_exports_sync_and_async_status_helpers() -> None:
    import xagent.core.tools.core.RAG_tools.management as management

    expected = {
        "write_ingestion_status",
        "load_ingestion_status",
        "clear_ingestion_status",
        "write_ingestion_status_async",
        "load_ingestion_status_async",
        "clear_ingestion_status_async",
    }

    assert expected.issubset(set(management.__all__))
    for name in expected:
        assert hasattr(management, name)

    assert inspect.iscoroutinefunction(management.write_ingestion_status_async)
    assert inspect.iscoroutinefunction(management.load_ingestion_status_async)
    assert inspect.iscoroutinefunction(management.clear_ingestion_status_async)
    assert not inspect.iscoroutinefunction(management.write_ingestion_status)
    assert not inspect.iscoroutinefunction(management.load_ingestion_status)
    assert not inspect.iscoroutinefunction(management.clear_ingestion_status)


@pytest.mark.asyncio
async def test_list_collections_routes_through_management_facade(
    monkeypatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import collections

    sentinel = object()
    calls: list[dict[str, Any]] = []

    class Facade:
        async def list_collections(
            self,
            user_id: Optional[int] = None,
            is_admin: Optional[bool] = None,
            force_realtime: bool = False,
        ) -> object:
            calls.append(
                {
                    "user_id": user_id,
                    "is_admin": is_admin,
                    "force_realtime": force_realtime,
                }
            )
            return sentinel

    monkeypatch.setattr(collections, "_get_management_facade", lambda: Facade())

    result = await collections.list_collections(
        user_id=123,
        is_admin=False,
        force_realtime=True,
    )

    assert result is sentinel
    assert calls == [
        {"user_id": 123, "is_admin": False, "force_realtime": True},
    ]


def test_delete_document_remains_sync_and_routes_through_management_facade(
    monkeypatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import collections

    sentinel = object()
    calls: list[tuple[str, str, int, bool]] = []

    class Facade:
        def delete_document(
            self,
            collection: str,
            doc_id: str,
            user_id: int,
            is_admin: bool = False,
        ) -> object:
            calls.append((collection, doc_id, user_id, is_admin))
            return sentinel

    monkeypatch.setattr(collections, "_get_management_facade", lambda: Facade())

    assert not inspect.iscoroutinefunction(collections.delete_document)
    result = collections.delete_document("docs", "doc-1", user_id=7, is_admin=True)

    assert result is sentinel
    assert calls == [("docs", "doc-1", 7, True)]


@pytest.mark.asyncio
async def test_async_status_helpers_stay_awaitable_and_route_through_facade(
    monkeypatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import status

    calls: list[tuple[str, str, str]] = []

    class Facade:
        async def write_ingestion_status_async(
            self,
            collection: str,
            doc_id: str,
            *,
            status: str,
            message: Optional[str] = None,
            parse_hash: Optional[str] = None,
            user_id: Optional[int] = None,
        ) -> None:
            calls.append((collection, doc_id, status))

    monkeypatch.setattr(status, "_get_management_facade", lambda: Facade())

    assert inspect.iscoroutinefunction(status.write_ingestion_status_async)
    await status.write_ingestion_status_async("docs", "doc-1", status="pending")

    assert calls == [("docs", "doc-1", "pending")]


def test_coordinator_management_facade_uses_instance_storage() -> None:
    from xagent.core.tools.core.RAG_tools.kb import KBCoordinator
    from xagent.core.tools.core.RAG_tools.management import collections
    from xagent.core.tools.core.RAG_tools.storage.contracts import DocumentRecord
    from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory

    class VectorStore:
        def __init__(self) -> None:
            self.list_calls: list[dict[str, object]] = []

        def list_document_records(self, **kwargs: object) -> list[DocumentRecord]:
            self.list_calls.append(dict(kwargs))
            return [DocumentRecord(doc_id="doc-local", source_path="/tmp/local.txt")]

        def aggregate_document_counts(self, **_: object) -> dict[str, int]:
            return {}

        def list_table_names(self) -> list[str]:
            return []

    class IngestionStatusStore:
        def __init__(self) -> None:
            self.load_calls: list[dict[str, object]] = []

        def load_ingestion_status(self, **kwargs: object) -> list[dict[str, object]]:
            self.load_calls.append(dict(kwargs))
            return [
                {
                    "collection": "docs",
                    "doc_id": "doc-local",
                    "status": "failed",
                    "message": "Loaded from injected status store",
                    "updated_at": None,
                }
            ]

    class StorageFactoryStub:
        def __init__(
            self,
            vector_store: VectorStore,
            status_store: IngestionStatusStore,
        ) -> None:
            self.vector_store = vector_store
            self.status_store = status_store

        def get_vector_index_store(self) -> VectorStore:
            return self.vector_store

        def get_ingestion_status_store(self) -> IngestionStatusStore:
            return self.status_store

    vector_store = VectorStore()
    status_store = IngestionStatusStore()

    coordinator = KBCoordinator(
        storage_factory=cast(
            StorageFactory,
            StorageFactoryStub(vector_store, status_store),
        )
    )
    result = coordinator.management.list_documents(
        "docs",
        user_id=42,
        is_admin=False,
    )

    assert [document.doc_id for document in result.documents] == ["doc-local"]
    assert result.documents[0].message == "Loaded from injected status store"
    assert vector_store.list_calls == [
        {
            "collection_name": "docs",
            "user_id": 42,
            "is_admin": False,
            "max_results": collections.DEFAULT_VECTOR_STORE_EXTENDED_SCAN_LIMIT,
        }
    ]
    assert status_store.load_calls == [
        {
            "collection": "docs",
            "doc_id": None,
            "user_id": None,
            "is_admin": False,
        }
    ]


def test_nested_management_facade_rebinds_inner_coordinator_storage() -> None:
    from xagent.core.tools.core.RAG_tools.kb import KBCoordinator
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        StorageFactory,
        bind_storage_shim_for_current_context,
        get_ingestion_status_store,
    )

    class IngestionStatusStore:
        def __init__(self, name: str) -> None:
            self.name = name
            self.load_calls: list[dict[str, object]] = []

        def load_ingestion_status(self, **kwargs: object) -> list[dict[str, object]]:
            self.load_calls.append(dict(kwargs))
            return [
                {
                    "collection": kwargs.get("collection"),
                    "doc_id": self.name,
                    "status": "success",
                }
            ]

    class StorageFactoryStub:
        def __init__(self, status_store: IngestionStatusStore) -> None:
            self.status_store = status_store

        def get_ingestion_status_store(self) -> IngestionStatusStore:
            return self.status_store

        def get_metadata_store(self):
            metadata = MagicMock()
            # hide_missing=True swallows this; _is_missing_collection_error matches
            # exactly "Collection 'docs' not found" (coordinator.py:1558-1561).
            metadata.get_collection = AsyncMock(
                side_effect=ValueError("Collection 'docs' not found")
            )
            return metadata

        def get_vector_index_store(self):
            return MagicMock()

        def get_main_pointer_store(self):
            return MagicMock()

    outer_store = IngestionStatusStore("outer-store")
    inner_store = IngestionStatusStore("inner-store")
    outer = KBCoordinator(
        storage_factory=cast(StorageFactory, StorageFactoryStub(outer_store))
    )
    inner = KBCoordinator(
        storage_factory=cast(StorageFactory, StorageFactoryStub(inner_store))
    )

    with bind_storage_shim_for_current_context(outer.storage_shim):
        assert get_ingestion_status_store() is outer_store
        records = inner.management.load_ingestion_status(
            collection="docs",
            doc_id="doc-inner",
        )
        assert records[0]["doc_id"] == "inner-store"
        assert get_ingestion_status_store() is outer_store

    assert outer_store.load_calls == []
    assert inner_store.load_calls == [
        {
            "collection": "docs",
            "doc_id": "doc-inner",
            "user_id": None,
            "is_admin": False,
        }
    ]


def test_facade_delegates_document_cleanup_to_existing_management_impl(
    monkeypatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.kb import (
        KBCoreManagementCompatibilityFacade,
    )
    from xagent.core.tools.core.RAG_tools.management import collections

    sentinel = object()
    calls: list[tuple[str, str, int, bool]] = []

    def fake_delete_document_impl(
        collection: str,
        doc_id: str,
        user_id: int,
        is_admin: bool = False,
    ) -> object:
        calls.append((collection, doc_id, user_id, is_admin))
        return sentinel

    monkeypatch.setattr(
        collections,
        "_delete_document_impl",
        fake_delete_document_impl,
    )

    result = KBCoreManagementCompatibilityFacade().delete_document(
        "docs",
        "doc-1",
        user_id=9,
        is_admin=False,
    )

    assert result is sentinel
    assert calls == [("docs", "doc-1", 9, False)]


@pytest.mark.asyncio
async def test_facade_delegates_status_cleanup_to_existing_async_impl(
    monkeypatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.kb import (
        KBCoreManagementCompatibilityFacade,
    )
    from xagent.core.tools.core.RAG_tools.management import status

    calls: list[tuple[str, str, Optional[int], bool]] = []

    async def fake_clear_status_impl(
        collection: str,
        doc_id: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        calls.append((collection, doc_id, user_id, is_admin))

    monkeypatch.setattr(
        status,
        "_clear_ingestion_status_async_impl",
        fake_clear_status_impl,
    )

    await KBCoreManagementCompatibilityFacade().clear_ingestion_status_async(
        "docs",
        "doc-1",
        user_id=3,
        is_admin=True,
    )

    assert calls == [("docs", "doc-1", 3, True)]
