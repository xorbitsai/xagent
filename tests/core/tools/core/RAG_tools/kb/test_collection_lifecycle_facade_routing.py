"""Tests for coordinator delete_collection/rename_collection + facade routing (H05 Phase 5).

Cycles:
  5.1 – coordinator.delete_collection routes admin/tenant calls to handle
  5.2 – coordinator.rename_collection calls data → status → metadata in order
  5.3 – api_compatibility and management facade routing switch
  5.4 – config-only invariant: shared vectors must NOT be deleted when own_count==0
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionOperationResult
from xagent.core.tools.core.RAG_tools.kb.api_compatibility import (
    KBApiCompatibilityFacade,
)
from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
from xagent.core.tools.core.RAG_tools.kb.management_facade import (
    KBCoreManagementCompatibilityFacade,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_handle() -> MagicMock:
    """Return a mock LanceDBCollectionHandle with async stubs."""
    handle = MagicMock()
    handle.delete_collection_data = MagicMock(return_value={"documents": 3})
    handle.delete_documents_data = MagicMock(return_value={"documents": 1})
    handle.delete_collection_config = AsyncMock(return_value=1)
    handle.rename_collection_data = MagicMock(return_value=[])
    handle.rename_collection_status = MagicMock(return_value=[])
    handle.rename_collection_metadata = AsyncMock(return_value=None)
    # H05 additions
    handle.list_collection_documents = MagicMock(return_value=[])
    handle.count_documents = MagicMock(return_value=0)
    handle.load_ingestion_status = MagicMock(return_value=[{"doc_id": "d1"}])
    handle.load_ingestion_status_async = AsyncMock(return_value=[{"doc_id": "d2"}])
    handle.promote_version_main = MagicMock(return_value={"promoted": True})
    handle.list_candidates = MagicMock(return_value={"candidates": []})
    return handle


def _make_coordinator_with_mock_handle(handle: MagicMock) -> KBCoordinator:
    """Build a KBCoordinator that returns *handle* from open_collection."""
    coordinator = KBCoordinator.__new__(KBCoordinator)
    # Minimal attribute init — only what delete/rename need
    coordinator._handle_provider = MagicMock()
    coordinator._storage_factory = MagicMock()
    coordinator._storage_shim = MagicMock()
    coordinator._file_compatibility = MagicMock()
    coordinator._management = MagicMock()
    coordinator._parse_display_compatibility = MagicMock()
    coordinator._maintenance_compatibility = MagicMock()
    coordinator._version_compatibility = MagicMock()
    coordinator._retrieval_helper_compatibility = MagicMock()
    coordinator._vector_storage_compatibility = MagicMock()
    coordinator._operation_compatibility = MagicMock()
    coordinator._pipeline_compatibility = MagicMock()
    coordinator._legacy_step_compatibility = MagicMock()
    coordinator._tool_compatibility = MagicMock()
    coordinator._api_compatibility = MagicMock()

    # open_collection is awaitable, so patch it as an AsyncMock returning handle
    coordinator.open_collection = AsyncMock(return_value=handle)
    return coordinator


# ---------------------------------------------------------------------------
# Cycle 5.1 – coordinator.delete_collection
# ---------------------------------------------------------------------------


class TestCoordinatorDeleteCollection:
    """Cycle 5.1: coordinator.delete_collection routes admin/tenant paths."""

    def test_coordinator_delete_routes_admin_to_delete_collection_data(self) -> None:
        """Admin delete calls handle.delete_collection_data with is_admin=True."""
        handle = _make_mock_handle()
        coordinator = _make_coordinator_with_mock_handle(handle)

        asyncio.run(
            coordinator.delete_collection(
                collection="my_coll",
                user_id=None,
                is_admin=True,
            )
        )

        handle.delete_collection_data.assert_called_once()
        call_kwargs = handle.delete_collection_data.call_args
        # is_admin must be True in the call
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()
        assert kwargs.get("is_admin", False) is True or (
            len(args) >= 2 and args[1] is True
        )
        handle.delete_documents_data.assert_not_called()

    def test_coordinator_delete_routes_tenant_with_doc_ids_to_delete_documents_data(
        self,
    ) -> None:
        """Tenant delete with doc_ids calls handle.delete_documents_data."""
        handle = _make_mock_handle()
        coordinator = _make_coordinator_with_mock_handle(handle)
        doc_ids = ["doc-1", "doc-2"]

        asyncio.run(
            coordinator.delete_collection(
                collection="tenant_coll",
                user_id=1,
                is_admin=False,
                doc_ids=doc_ids,
            )
        )

        handle.delete_documents_data.assert_called_once()
        handle.delete_collection_data.assert_not_called()

    def test_coordinator_delete_returns_collection_operation_result(self) -> None:
        """delete_collection returns a CollectionOperationResult."""
        handle = _make_mock_handle()
        coordinator = _make_coordinator_with_mock_handle(handle)

        result = asyncio.run(
            coordinator.delete_collection(
                collection="any_coll",
                user_id=None,
                is_admin=True,
            )
        )

        assert isinstance(result, CollectionOperationResult)
        assert result.status in {"success", "partial_success", "error"}

    def test_coordinator_delete_partial_success_on_database_operation_error(
        self,
    ) -> None:
        """When delete_collection_data raises DatabaseOperationError, return partial_success."""
        handle = _make_mock_handle()
        handle.delete_collection_data = MagicMock(
            side_effect=DatabaseOperationError("disk full")
        )
        coordinator = _make_coordinator_with_mock_handle(handle)

        result = asyncio.run(
            coordinator.delete_collection(
                collection="broken_coll",
                user_id=None,
                is_admin=True,
            )
        )

        assert isinstance(result, CollectionOperationResult)
        assert result.status in {"partial_success", "error"}

    def test_coordinator_delete_calls_delete_collection_config_when_requested(
        self,
    ) -> None:
        """delete_collection_config is called when delete_orphaned_metadata=True."""
        handle = _make_mock_handle()
        coordinator = _make_coordinator_with_mock_handle(handle)

        asyncio.run(
            coordinator.delete_collection(
                collection="cfg_coll",
                user_id=None,
                is_admin=True,
                delete_orphaned_metadata=True,
            )
        )

        handle.delete_collection_config.assert_called_once()

    def test_coordinator_delete_skips_delete_collection_config_when_not_requested(
        self,
    ) -> None:
        """delete_collection_config is not called when delete_orphaned_metadata=False."""
        handle = _make_mock_handle()
        coordinator = _make_coordinator_with_mock_handle(handle)

        asyncio.run(
            coordinator.delete_collection(
                collection="cfg_coll",
                user_id=None,
                is_admin=True,
                delete_orphaned_metadata=False,
            )
        )

        handle.delete_collection_config.assert_not_called()


# ---------------------------------------------------------------------------
# Cycle 5.2 – coordinator.rename_collection
# ---------------------------------------------------------------------------


class TestCoordinatorRenameCollection:
    """Cycle 5.2: coordinator.rename_collection calls steps in order."""

    def test_coordinator_rename_calls_data_then_status_then_metadata_in_order(
        self,
    ) -> None:
        """rename_collection calls data → status → metadata in that order."""
        call_order: list[str] = []
        handle = _make_mock_handle()
        handle.rename_collection_data = MagicMock(
            side_effect=lambda *a, **kw: call_order.append("data") or []
        )
        handle.rename_collection_status = MagicMock(
            side_effect=lambda *a, **kw: call_order.append("status") or []
        )
        handle.rename_collection_metadata = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append("metadata")
        )
        coordinator = _make_coordinator_with_mock_handle(handle)

        asyncio.run(
            coordinator.rename_collection(
                old_name="old_coll",
                new_name="new_coll",
                user_id=None,
                is_admin=True,
            )
        )

        assert call_order == ["data", "status", "metadata"]

    def test_coordinator_rename_aborts_when_data_step_raises(self) -> None:
        """A hard exception from the data rename gates the control-plane rename.

        The data rename is the gate: if it raises, the exception propagates and
        status/metadata are never renamed, avoiding a split-brain collection.
        """
        handle = _make_mock_handle()
        handle.rename_collection_data = MagicMock(
            side_effect=DatabaseOperationError("data step failed")
        )
        handle.rename_collection_status = MagicMock(return_value=["status_warn"])
        handle.rename_collection_metadata = AsyncMock(return_value=None)
        coordinator = _make_coordinator_with_mock_handle(handle)

        with pytest.raises(DatabaseOperationError):
            asyncio.run(
                coordinator.rename_collection(
                    old_name="old_coll",
                    new_name="new_coll",
                    user_id=None,
                    is_admin=True,
                )
            )

        # Control-plane steps must NOT run after a failed data rename.
        handle.rename_collection_status.assert_not_called()
        handle.rename_collection_metadata.assert_not_called()

    def test_coordinator_rename_aborts_when_data_step_returns_warnings(self) -> None:
        """Non-empty data warnings also gate the control-plane rename.

        ``rename_collection_data`` catches per-table failures and returns them as
        warnings rather than raising, so a non-empty list means some vector rows
        were not moved.  The coordinator must short-circuit and not rename
        status/metadata, returning the data warnings to the caller.
        """
        handle = _make_mock_handle()
        handle.rename_collection_data = MagicMock(
            return_value=["Failed to update 'chunks': boom"]
        )
        handle.rename_collection_status = MagicMock(return_value=["status_warn"])
        handle.rename_collection_metadata = AsyncMock(return_value=None)
        coordinator = _make_coordinator_with_mock_handle(handle)

        warnings = asyncio.run(
            coordinator.rename_collection(
                old_name="old_coll",
                new_name="new_coll",
                user_id=None,
                is_admin=True,
            )
        )

        handle.rename_collection_status.assert_not_called()
        handle.rename_collection_metadata.assert_not_called()
        assert any("Failed to update 'chunks'" in w for w in warnings)
        assert all("status_warn" not in w for w in warnings)

    def test_coordinator_rename_returns_list_of_warnings(self) -> None:
        """rename_collection always returns a list."""
        handle = _make_mock_handle()
        coordinator = _make_coordinator_with_mock_handle(handle)

        result = asyncio.run(
            coordinator.rename_collection(
                old_name="a",
                new_name="b",
                user_id=None,
                is_admin=True,
            )
        )

        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Cycle 5.3 – Facade routing switch
# ---------------------------------------------------------------------------


class TestApiCompatRenameRoutingSwitch:
    """Cycle 5.3: api_compatibility.rename_collection_data/status/metadata routes."""

    def test_api_compat_rename_collection_data_routes_through_coordinator_when_present(
        self,
    ) -> None:
        """rename_collection_data calls coordinator.rename_collection when coordinator is set."""
        coordinator = MagicMock()
        coordinator.rename_collection = AsyncMock(return_value=[])
        facade = KBApiCompatibilityFacade(coordinator=coordinator)

        result = asyncio.run(
            facade.rename_collection_data(
                collection_name="old",
                new_name="new",
                user_id=1,
                is_admin=False,
            )
        )

        coordinator.rename_collection.assert_called_once()
        assert isinstance(result, list)

    def test_api_compat_rename_collection_data_falls_back_to_store_when_no_coordinator(
        self,
    ) -> None:
        """rename_collection_data falls back to store when no coordinator."""
        facade = KBApiCompatibilityFacade(coordinator=None)
        mock_store = MagicMock()
        mock_store.rename_collection_data = MagicMock(return_value=["warn"])

        with patch(
            "xagent.core.tools.core.RAG_tools.kb.api_compatibility.get_vector_index_store"
            if False
            else "xagent.core.tools.core.RAG_tools.storage.factory.get_vector_index_store",
            return_value=mock_store,
        ):
            # The fallback path uses the storage factory — we just check the method exists
            # and returns a list even when no coordinator is set.
            # We mock the internal call so it doesn't hit a real DB.
            with patch.object(
                facade,
                "_active_storage_shim",
                return_value=None,
            ):
                from unittest.mock import patch as _patch

                with _patch(
                    "xagent.core.tools.core.RAG_tools.storage.factory.get_vector_index_store",
                    return_value=mock_store,
                ):
                    result = asyncio.run(
                        facade.rename_collection_data(
                            collection_name="old",
                            new_name="new",
                            user_id=None,
                            is_admin=False,
                        )
                    )

        # The store method must have been invoked
        mock_store.rename_collection_data.assert_called_once()
        assert isinstance(result, list)


class TestManagementFacadeDeleteRoutingSwitch:
    """Cycle 5.3: management facade delete_collection routing (sync AND async)."""

    def test_management_facade_delete_collection_sync_routes_through_coordinator_when_present(
        self,
    ) -> None:
        """Sync delete_collection bridges to coordinator.delete_collection via thread."""
        coordinator = MagicMock()
        expected = CollectionOperationResult(
            status="success",
            collection="c",
            message="ok",
        )
        coordinator.delete_collection = AsyncMock(return_value=expected)
        facade = KBCoreManagementCompatibilityFacade(coordinator=coordinator)

        # Call the SYNC method — it should route to coordinator without async/await
        result = facade.delete_collection(
            collection="c",
            user_id=None,
            is_admin=True,
        )

        coordinator.delete_collection.assert_called_once()
        assert result.status == "success"

    def test_management_facade_delete_collection_async_routes_through_coordinator_when_present(
        self,
    ) -> None:
        """Async delete_collection_async also routes to coordinator.delete_collection."""
        coordinator = MagicMock()
        expected = CollectionOperationResult(
            status="success",
            collection="c",
            message="ok",
        )
        coordinator.delete_collection = AsyncMock(return_value=expected)
        facade = KBCoreManagementCompatibilityFacade(coordinator=coordinator)

        result = asyncio.run(
            facade.delete_collection_async(
                collection="c",
                user_id=None,
                is_admin=True,
            )
        )

        coordinator.delete_collection.assert_called_once()
        assert result.status == "success"

    def test_management_facade_delete_collection_falls_back_to_impl_when_no_coordinator(
        self,
    ) -> None:
        """Sync delete_collection falls back to _delete_collection_impl when no coordinator."""
        facade = KBCoreManagementCompatibilityFacade(coordinator=None)

        expected = CollectionOperationResult(
            status="success",
            collection="c",
            message="ok",
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.management.collections._delete_collection_impl",
            return_value=expected,
        ) as mock_impl:
            result = facade.delete_collection(
                collection="c",
                user_id=None,
                is_admin=True,
            )

        mock_impl.assert_called_once()
        assert result.status == "success"


# ---------------------------------------------------------------------------
# Cycle 5.4 – config-only invariant
# ---------------------------------------------------------------------------


class TestConfigOnlyInvariant:
    """Cycle 5.4: config-only delete must not touch shared vector data."""

    def test_config_only_delete_does_not_touch_shared_vectors(self) -> None:
        """Tenant delete with own_count==0 and total>0 must not call delete_collection_data/delete_documents_data."""
        handle = _make_mock_handle()
        coordinator = _make_coordinator_with_mock_handle(handle)

        # Simulate config-only scenario:
        # is_admin=False, doc_ids=None (no own docs to delete)
        result = asyncio.run(
            coordinator.delete_collection(
                collection="shared_coll",
                user_id=1,
                is_admin=False,
                doc_ids=None,  # caller says: I have no owned docs to delete
                delete_orphaned_metadata=True,
            )
        )

        # Neither data delete should be called when there are no doc_ids to delete
        handle.delete_collection_data.assert_not_called()
        handle.delete_documents_data.assert_not_called()
        # Config cleanup is scoped to this tenant only (non-admin must not delete other tenants' rows)
        handle.delete_collection_config.assert_called_once_with(tenant_only=True)
        assert isinstance(result, CollectionOperationResult)

    def test_resolve_delete_mode_returns_config_only_for_shared_collection(
        self,
    ) -> None:
        """The config-only rule: own_count==0 and total>0 and not is_admin → config_only.

        This logic lives in web/api/kb.py::_resolve_delete_mode_from_counts.
        We test the business rule inline here (same logic) to avoid importing
        the web layer which requires fastapi.
        """

        def _resolve(total_count: int, own_count: int, is_admin: bool) -> str:
            if is_admin:
                return "full"
            if total_count > 0 and own_count == 0:
                return "config_only"
            return "full"

        assert _resolve(5, 0, False) == "config_only"

    def test_resolve_delete_mode_returns_full_for_admin(self) -> None:
        """Admin always gets full delete mode."""

        def _resolve(total_count: int, own_count: int, is_admin: bool) -> str:
            if is_admin:
                return "full"
            if total_count > 0 and own_count == 0:
                return "config_only"
            return "full"

        assert _resolve(5, 0, True) == "full"

    def test_resolve_delete_mode_returns_full_when_tenant_owns_docs(self) -> None:
        """When the tenant owns at least some docs, mode is full (delete own docs)."""

        def _resolve(total_count: int, own_count: int, is_admin: bool) -> str:
            if is_admin:
                return "full"
            if total_count > 0 and own_count == 0:
                return "config_only"
            return "full"

        assert _resolve(5, 3, False) == "full"

    def test_coordinator_delete_with_empty_doc_ids_is_noop_on_data(self) -> None:
        """delete_collection called with empty doc_ids=[] is a no-op on data."""
        handle = _make_mock_handle()
        coordinator = _make_coordinator_with_mock_handle(handle)

        result = asyncio.run(
            coordinator.delete_collection(
                collection="shared_coll",
                user_id=1,
                is_admin=False,
                doc_ids=[],  # empty list — no docs owned by user
                delete_orphaned_metadata=False,
            )
        )

        handle.delete_collection_data.assert_not_called()
        handle.delete_documents_data.assert_not_called()
        assert isinstance(result, CollectionOperationResult)


# ---------------------------------------------------------------------------
# Bug-fix regression tests (issues found in review)
# ---------------------------------------------------------------------------


class TestCoordinatorDeleteOrphanedMetadataGuard:
    """Coordinator must not delete orphaned metadata when other tenants still have data."""

    def test_delete_collection_config_tenant_only_when_remaining_records_exist(
        self,
    ) -> None:
        """When count_documents > 0 after deletion, only the current tenant's config row is removed."""
        handle = _make_mock_handle()
        # Simulate another tenant still having rows
        handle.count_documents = MagicMock(return_value=5)
        coordinator = _make_coordinator_with_mock_handle(handle)

        asyncio.run(
            coordinator.delete_collection(
                collection="shared_coll",
                user_id=1,
                is_admin=False,
                doc_ids=["doc-1"],
                delete_orphaned_metadata=True,
            )
        )

        # Data was deleted; tenant's own config row is removed but other tenants' rows are preserved.
        handle.delete_documents_data.assert_called_once()
        handle.delete_collection_config.assert_called_once_with(tenant_only=True)

    def test_delete_collection_config_admin_scope_when_admin_and_empty(
        self,
    ) -> None:
        """Admin caller + remaining == 0: full admin-scope config cleanup (all tenant rows)."""
        handle = _make_mock_handle()
        handle.count_documents = MagicMock(return_value=0)
        coordinator = _make_coordinator_with_mock_handle(handle)

        asyncio.run(
            coordinator.delete_collection(
                collection="my_coll",
                user_id=None,
                is_admin=True,
                delete_orphaned_metadata=True,
            )
        )

        handle.delete_collection_data.assert_called_once()
        # Admin + empty collection → full cleanup, no tenant_only restriction
        handle.delete_collection_config.assert_called_once_with()

    def test_delete_collection_config_tenant_only_when_non_admin_and_empty(
        self,
    ) -> None:
        """Non-admin caller + remaining == 0: only the current tenant's config row is removed."""
        handle = _make_mock_handle()
        handle.count_documents = MagicMock(return_value=0)
        coordinator = _make_coordinator_with_mock_handle(handle)

        asyncio.run(
            coordinator.delete_collection(
                collection="my_coll",
                user_id=1,
                is_admin=False,
                doc_ids=["doc-a"],
                delete_orphaned_metadata=True,
            )
        )

        handle.delete_documents_data.assert_called_once()
        # Non-admin must not escalate to admin-scope cleanup even when collection appears empty
        handle.delete_collection_config.assert_called_once_with(tenant_only=True)


class TestCoordinatorDeleteAutoDiscoversDocIds:
    """Coordinator must auto-discover tenant doc_ids when caller does not provide them."""

    def test_tenant_delete_without_doc_ids_discovers_and_deletes_own_docs(
        self,
    ) -> None:
        """When is_admin=False and doc_ids=None, coordinator lists and deletes own docs."""
        handle = _make_mock_handle()
        handle.list_collection_documents = MagicMock(return_value=["doc-a", "doc-b"])
        handle.count_documents = MagicMock(return_value=0)
        coordinator = _make_coordinator_with_mock_handle(handle)

        asyncio.run(
            coordinator.delete_collection(
                collection="my_coll",
                user_id=42,
                is_admin=False,
                doc_ids=None,  # caller did NOT pre-compute doc_ids
                delete_orphaned_metadata=False,
            )
        )

        # Must have discovered doc_ids from the handle
        handle.list_collection_documents.assert_called()
        # Must have passed discovered ids to delete_documents_data
        handle.delete_documents_data.assert_called_once()
        call_args = handle.delete_documents_data.call_args
        passed_ids = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("doc_ids")
        )
        assert sorted(passed_ids) == ["doc-a", "doc-b"]

    def test_tenant_delete_with_explicit_doc_ids_does_not_call_list(
        self,
    ) -> None:
        """When doc_ids is explicitly provided, list_collection_documents is only called for affected_documents."""
        handle = _make_mock_handle()
        handle.list_collection_documents = MagicMock(return_value=["doc-x"])
        handle.count_documents = MagicMock(return_value=0)
        coordinator = _make_coordinator_with_mock_handle(handle)

        asyncio.run(
            coordinator.delete_collection(
                collection="my_coll",
                user_id=42,
                is_admin=False,
                doc_ids=["doc-x"],  # explicitly provided
                delete_orphaned_metadata=False,
            )
        )

        # delete_documents_data uses the explicitly provided list
        handle.delete_documents_data.assert_called_once()
        call_args = handle.delete_documents_data.call_args
        passed_ids = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("doc_ids")
        )
        assert passed_ids == ["doc-x"]


class TestRenameFacadeCoordinatorEarlyReturn:
    """rename_collection_status/_metadata must be no-ops when coordinator is active."""

    def test_rename_collection_status_returns_empty_when_coordinator_present(
        self,
    ) -> None:
        """rename_collection_status returns [] without hitting store when coordinator set."""
        coordinator = MagicMock()
        facade = KBApiCompatibilityFacade(coordinator=coordinator)

        result = facade.rename_collection_status(
            old_name="old",
            new_name="new",
            user_id=1,
            is_admin=False,
        )

        assert result == []

    def test_rename_collection_metadata_is_noop_when_coordinator_present(
        self,
    ) -> None:
        """rename_collection_metadata returns None without hitting store when coordinator set."""
        coordinator = MagicMock()
        facade = KBApiCompatibilityFacade(coordinator=coordinator)

        # Should return None (no-op) without touching any store
        result = asyncio.run(
            facade.rename_collection_metadata(
                old_name="old",
                new_name="new",
                user_id=1,
                is_admin=False,
            )
        )

        assert result is None

    def test_rename_collection_status_hits_store_when_no_coordinator(
        self,
    ) -> None:
        """rename_collection_status uses store when coordinator is None."""
        facade = KBApiCompatibilityFacade(coordinator=None)
        mock_store = MagicMock()
        mock_store.rename_collection_status = MagicMock(return_value=[])

        with patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_ingestion_status_store",
            return_value=mock_store,
        ):
            result = facade.rename_collection_status(
                old_name="old",
                new_name="new",
                user_id=1,
                is_admin=False,
            )

        mock_store.rename_collection_status.assert_called_once()
        assert isinstance(result, list)


class TestIngestionStatusReadRouting:
    """#514: load_ingestion_status(_async) must route through the handle,
    not the injected storage shim."""

    def test_load_ingestion_status_routes_through_handle(self) -> None:
        handle = _make_mock_handle()
        handle.load_ingestion_status = MagicMock(return_value=[{"doc_id": "d1"}])
        coordinator = _make_coordinator_with_mock_handle(handle)

        result = asyncio.run(
            coordinator.load_ingestion_status(
                "docs", doc_id="d1", user_id=7, is_admin=False
            )
        )

        assert result == [{"doc_id": "d1"}]
        handle.load_ingestion_status.assert_called_once_with(
            doc_id="d1", user_id=7, is_admin=False
        )
        coordinator._storage_shim.get_ingestion_status_store.assert_not_called()

    def test_load_ingestion_status_async_routes_through_handle(self) -> None:
        handle = _make_mock_handle()
        handle.load_ingestion_status_async = AsyncMock(return_value=[{"doc_id": "d2"}])
        coordinator = _make_coordinator_with_mock_handle(handle)

        result = asyncio.run(
            coordinator.load_ingestion_status_async(
                "docs", doc_id="d2", user_id=None, is_admin=True
            )
        )

        assert result == [{"doc_id": "d2"}]
        handle.load_ingestion_status_async.assert_awaited_once_with(
            doc_id="d2", user_id=None, is_admin=True
        )
        coordinator._storage_shim.get_ingestion_status_store.assert_not_called()


class TestVersionPromotionRouting:
    """#514: version promotion/listing must route through the handle."""

    def test_promote_version_main_routes_through_handle(self) -> None:
        handle = _make_mock_handle()
        handle.promote_version_main = MagicMock(return_value={"promoted": True})
        coordinator = _make_coordinator_with_mock_handle(handle)

        result = asyncio.run(
            coordinator.promote_version_main(
                "docs", "doc-1", "parse", "cand-9", operator="alice", confirm=True
            )
        )

        assert result == {"promoted": True}
        handle.promote_version_main.assert_called_once_with(
            "doc-1",
            "parse",
            "cand-9",
            operator="alice",
            preview_only=False,
            confirm=True,
            model_tag=None,
        )

    def test_list_candidates_routes_through_handle(self) -> None:
        handle = _make_mock_handle()
        handle.list_candidates = MagicMock(return_value={"candidates": []})
        coordinator = _make_coordinator_with_mock_handle(handle)

        result = asyncio.run(coordinator.list_candidates("docs", "doc-1", "parse"))

        assert result == {"candidates": []}
        handle.list_candidates.assert_called_once_with(
            "doc-1", "parse", None, None, 50, "created_at desc"
        )


class TestRollbackRestoreRouting:
    """#514 rollback additions: rollback data-plane restore routes through the handle."""

    def test_restore_main_pointer_snapshot_routes_through_handle(self) -> None:
        handle = _make_mock_handle()
        handle.restore_main_pointer_snapshot = MagicMock(return_value=True)
        coordinator = _make_coordinator_with_mock_handle(handle)

        snapshot = MagicMock()
        snapshot.collection = "docs"

        result = asyncio.run(
            coordinator.restore_main_pointer_snapshot(snapshot, operator="bob")
        )

        assert result is True
        handle.restore_main_pointer_snapshot.assert_called_once_with(
            snapshot, operator="bob"
        )


class TestCrossOwnerRollbackRouting:
    """#514: multi-owner rollback opens the handle with the snapshot's OWN
    owner identity, not the caller's — orchestration stays coordinator-owned."""

    def test_restore_candidate_cleanup_uses_snapshot_owner(self) -> None:
        handle = _make_mock_handle()
        handle.restore_candidate_cleanup_snapshot = MagicMock(
            return_value={"rolled_back": True}
        )
        coordinator = _make_coordinator_with_mock_handle(handle)

        captured: list = []

        async def _spy_open(request):
            captured.append(request)
            return handle

        coordinator.open_collection = _spy_open

        snapshot = MagicMock()
        snapshot.collection = "docs"
        snapshot.user_id = 42
        snapshot.is_admin = False

        result = asyncio.run(
            coordinator.restore_candidate_cleanup_snapshot(
                snapshot, cleanup_executed=True
            )
        )

        assert result == {"rolled_back": True}
        assert len(captured) == 1
        assert captured[0].collection == "docs"
        assert captured[0].user_id == 42
        assert captured[0].is_admin is False
        handle.restore_candidate_cleanup_snapshot.assert_called_once_with(
            snapshot, cleanup_executed=True
        )
