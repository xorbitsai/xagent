"""#514 - the coordinator reaches collection data only through a handle.

Two complementary guards:

* A structural one over ``coordinator.py``: no method other than context
  resolution may reach a store, by any of the routes that are reachable from
  the coordinator -- the shim attribute, its public property, the storage
  factory, a ``getattr`` by name, or the module-level store accessors.
* Behavioural ones driving the coordinator with a fake backend handle: one
  representative method from several operation families, proving each routes
  through ``KBHandleProvider.open`` and that the resolved capabilities reach
  the handle. A non-LanceDB backend is not bindable yet, so the fake provider
  stands in for the external-vector case the split was designed for.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest

import xagent.core.tools.core.RAG_tools.kb.coordinator as coordinator_module
from xagent.core.tools.core.RAG_tools.kb import (
    KBAccessMode,
    KBBackendCapabilities,
    KBContextRequest,
    KBCoordinator,
    KBStorageBackend,
)

# Every route from a coordinator method to a store: the shim, the public
# property that returns it, the factory the shim is built from, and the
# module-level accessors the sibling facades already use.
SHIM_ATTRIBUTES = frozenset({"_storage_shim", "storage_shim", "_storage_factory"})
# Every accessor KBStorageShimCompatibilityFacade exposes, not just the four
# the coordinator happens to use today.
STORE_ACCESSORS = frozenset(
    {
        "get_kb_write_coordinator",
        "get_metadata_store",
        "get_vector_index_store",
        "get_vector_store_raw_connection",
        "get_ingestion_status_store",
        "get_prompt_template_store",
        "get_main_pointer_store",
    }
)

# Context resolution owns store lookup; the reset hook and the accessor only
# forward. Everything else must reach data through a handle.
SHIM_ALLOWED_METHODS = frozenset(
    {
        "__init__",
        "storage_shim",
        "get_context",
        "reset_compatibility_caches",
    }
)


def _methods_touching_shim(source: str) -> set[str]:
    tree = ast.parse(source)
    offenders: set[str] = set()
    for klass in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        if klass.name != "KBCoordinator":
            continue
        for method in klass.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(method):
                if isinstance(node, ast.Attribute) and (
                    node.attr in SHIM_ATTRIBUTES or node.attr in STORE_ACCESSORS
                ):
                    offenders.add(method.name)
                elif isinstance(node, ast.Constant) and node.value in SHIM_ATTRIBUTES:
                    offenders.add(method.name)
                elif isinstance(node, ast.Name) and node.id in STORE_ACCESSORS:
                    offenders.add(method.name)
                elif (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").split(".")[0] == "storage"
                ):
                    # Catches an accessor renamed on import, which no name
                    # match can see.
                    offenders.add(method.name)
    return offenders


def test_only_context_resolution_touches_the_storage_shim() -> None:
    source = Path(coordinator_module.__file__).read_text()

    offenders = _methods_touching_shim(source) - SHIM_ALLOWED_METHODS

    assert offenders == set(), (
        "KBCoordinator must reach collection data through a handle; these "
        f"methods touch the storage shim directly: {sorted(offenders)}"
    )


def test_shim_guard_flags_a_method_that_reaches_for_a_store() -> None:
    """The guard fails on a coordinator that grew its own store access."""
    source = """
class KBCoordinator:
    def get_context(self):
        return self._storage_shim.get_metadata_store()

    def direct_shim(self):
        return self._storage_shim.get_metadata_store().scan()

    def via_public_property(self):
        return self.storage_shim.get_vector_index_store()

    def via_getattr(self):
        return getattr(self, "_storage_shim").get_vector_index_store()

    def via_factory(self):
        return self._storage_factory.get_metadata_store()

    def via_module_accessor(self):
        from ..storage.factory import get_metadata_store

        return get_metadata_store()

    def via_unlisted_accessor(self):
        return self._storage_factory.get_vector_store_raw_connection()

    def via_renamed_import(self):
        from ..storage.factory import get_metadata_store as _grab

        return _grab()

    def via_imported_module(self):
        from ..storage import factory

        return factory.get_metadata_store()

    def routed(self, request):
        handle = self.open_collection(request)
        return handle.list_documents()
"""

    offenders = _methods_touching_shim(source) - SHIM_ALLOWED_METHODS

    assert offenders == {
        "direct_shim",
        "via_public_property",
        "via_getattr",
        "via_factory",
        "via_module_accessor",
        "via_imported_module",
        "via_unlisted_accessor",
        "via_renamed_import",
    }


@dataclass
class _FakeCollectionInfo:
    extra_metadata: dict[str, Any]


class _FakeMetadataStore:
    def __init__(self, collection_info: Optional[_FakeCollectionInfo] = None) -> None:
        self._collection_info = collection_info
        self.calls: list[str] = []

    async def get_collection(self, collection: str) -> Optional[_FakeCollectionInfo]:
        self.calls.append(collection)
        return self._collection_info


class _FakeStorageShim:
    """Storage shim that hands out sentinels instead of backend stores."""

    def __init__(self, metadata_store: _FakeMetadataStore) -> None:
        self.metadata_store = metadata_store
        self.vector_index_store = object()
        self.ingestion_status_store = object()
        self.main_pointer_store = object()

    def get_metadata_store(self) -> _FakeMetadataStore:
        return self.metadata_store

    def get_vector_index_store(self) -> object:
        return self.vector_index_store

    def get_ingestion_status_store(self) -> object:
        return self.ingestion_status_store

    def get_main_pointer_store(self) -> object:
        return self.main_pointer_store

    def reset_coordinator_caches(self) -> None:
        return None


class _RecordingHandle:
    """Stands in for an external-vector backend handle."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def _record(*args: Any, **kwargs: Any) -> str:
            self.calls.append((name, args, kwargs))
            return f"{name}-result"

        return _record


class _RecordingHandleProvider:
    def __init__(self) -> None:
        self.contexts: list[Any] = []
        self.handles: list[_RecordingHandle] = []

    def open(self, context: Any) -> _RecordingHandle:
        handle = _RecordingHandle(context)
        self.contexts.append(context)
        self.handles.append(handle)
        return handle

    def reset_for_tests(self) -> None:
        return None


def _coordinator(
    collection_info: Optional[_FakeCollectionInfo] = None,
) -> tuple[KBCoordinator, _RecordingHandleProvider, _FakeMetadataStore]:
    metadata_store = _FakeMetadataStore(collection_info)
    provider = _RecordingHandleProvider()
    coordinator = KBCoordinator(
        handle_provider=provider,  # type: ignore[arg-type]
        storage_shim=_FakeStorageShim(metadata_store),  # type: ignore[arg-type]
    )
    return coordinator, provider, metadata_store


# (coordinator method, kwargs, handle method the call must land on)
ROUTED_OPERATIONS = [
    (
        "list_document_records",
        {"collection": "c"},
        "list_documents",
    ),
    (
        "delete_document_record",
        {"collection": "c", "doc_id": "d"},
        "delete_document_record",
    ),
    (
        "load_ingestion_status",
        {"collection": "c", "doc_id": "d"},
        "load_ingestion_status",
    ),
    (
        "get_main_pointer",
        {"collection": "c", "doc_id": "d", "step_type": "parse"},
        "get_main_pointer",
    ),
    (
        "list_candidates",
        {"collection": "c", "doc_id": "d", "step_type": "parse"},
        "list_candidates",
    ),
    (
        "cleanup_document_cascade",
        {"collection": "c", "doc_id": "d"},
        "cleanup_document_cascade",
    ),
    (
        "capture_status_snapshot",
        {"collection": "c", "doc_id": "d"},
        "capture_status_snapshot",
    ),
    (
        "cleanup_vectors_for_operation",
        {"collection": "c", "doc_id": "d"},
        "cleanup_embeddings_for_operation",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("coordinator_method", "kwargs", "handle_method"),
    ROUTED_OPERATIONS,
    ids=[row[0] for row in ROUTED_OPERATIONS],
)
async def test_collection_scoped_operations_route_through_the_handle(
    coordinator_method: str, kwargs: dict[str, Any], handle_method: str
) -> None:
    coordinator, provider, metadata_store = _coordinator()

    result = await getattr(coordinator, coordinator_method)(**kwargs)

    assert result == f"{handle_method}-result"
    assert len(provider.handles) == 1, (
        f"{coordinator_method} must open exactly one handle, "
        f"opened {len(provider.handles)}"
    )
    called = [name for name, _args, _kwargs in provider.handles[0].calls]
    assert called == [handle_method]
    # The only store the coordinator may touch itself is the metadata store,
    # and only to resolve the backend binding.
    assert metadata_store.calls == ["c"]


@pytest.mark.asyncio
async def test_resolved_capabilities_reach_the_handle() -> None:
    coordinator, provider, _metadata_store = _coordinator()

    await coordinator.list_document_records(collection="c")

    context = provider.contexts[0]
    assert context.backend is KBStorageBackend.LANCEDB
    assert context.capabilities == KBBackendCapabilities.lancedb()
    assert len(provider.contexts) == 1


@pytest.mark.asyncio
async def test_declared_backend_binding_drives_capability_resolution() -> None:
    """An explicit lancedb binding resolves the same way an absent one does."""
    # Production writers persist the nested object (pipeline_compatibility.py:171).
    collection_info = _FakeCollectionInfo(
        extra_metadata={
            coordinator_module.KB_STORAGE_METADATA_KEY: {"backend": "lancedb"}
        }
    )
    coordinator, provider, _metadata_store = _coordinator(collection_info)

    await coordinator.list_document_records(collection="c")

    context = provider.contexts[0]
    assert context.backend is KBStorageBackend.LANCEDB
    assert context.capabilities.supports_search is True
    assert context.collection_info is collection_info


@pytest.mark.asyncio
async def test_unknown_backend_binding_fails_before_a_handle_is_opened() -> None:
    """An unbindable backend must not reach the handle provider at all."""
    collection_info = _FakeCollectionInfo(
        extra_metadata={coordinator_module.KB_STORAGE_METADATA_KEY: "external_vector"}
    )
    coordinator, provider, _metadata_store = _coordinator(collection_info)

    with pytest.raises(ValueError, match="external_vector"):
        await coordinator.open_collection(KBContextRequest(collection="c"))

    assert provider.handles == []


@pytest.mark.asyncio
async def test_a_scoped_read_forwards_caller_identity_and_limit() -> None:
    """Non-default scope must reach the handle, not be defaulted away."""
    coordinator, provider, _metadata_store = _coordinator()

    await coordinator.list_document_records(
        collection="c", user_id=7, is_admin=False, limit=3
    )

    name, args, kwargs = provider.handles[0].calls[0]
    assert name == "list_documents"
    assert kwargs == {"user_id": 7, "is_admin": False, "limit": 3}
    assert args == ()
    assert provider.contexts[0].user_scope.user_id == 7
    assert provider.contexts[0].user_scope.is_admin is False


@pytest.mark.asyncio
async def test_a_scoped_cleanup_forwards_its_target_and_write_access() -> None:
    """Destructive cleanup must carry its target and open a WRITE context."""
    coordinator, provider, _metadata_store = _coordinator()

    await coordinator.cleanup_vectors_for_operation(
        collection="c",
        doc_id="d",
        parse_hash="p",
        chunk_ids=["ch1"],
        model_tag="m",
        user_id=7,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    name, args, kwargs = provider.handles[0].calls[0]
    assert name == "cleanup_embeddings_for_operation"
    assert args == ()
    assert kwargs == {
        "doc_id": "d",
        "parse_hash": "p",
        "chunk_ids": ["ch1"],
        "model_tag": "m",
        "preview_only": False,
        "confirm": True,
    }
    context = provider.contexts[0]
    assert context.access_mode is KBAccessMode.WRITE
    assert context.user_scope.user_id == 7
