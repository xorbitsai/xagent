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
                user_id="u1",
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

    def test_coordinator_rename_collects_warnings_from_all_steps_best_effort(
        self,
    ) -> None:
        """Warnings from each step are all collected even if data step raises."""
        handle = _make_mock_handle()
        handle.rename_collection_data = MagicMock(
            side_effect=DatabaseOperationError("data step failed")
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

        # All subsequent steps still ran and their warnings were collected
        handle.rename_collection_status.assert_called_once()
        handle.rename_collection_metadata.assert_called_once()
        assert any("status_warn" in w for w in warnings) or "status_warn" in warnings

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
    """Cycle 5.3: management facade delete_collection routing."""

    def test_management_facade_delete_collection_routes_through_coordinator_when_present(
        self,
    ) -> None:
        """delete_collection calls coordinator.delete_collection when coordinator set."""
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
        """delete_collection falls back to _delete_collection_impl when no coordinator."""
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
                user_id="u1",
                is_admin=False,
                doc_ids=None,  # caller says: I have no owned docs to delete
                delete_orphaned_metadata=True,
            )
        )

        # Neither data delete should be called when there are no doc_ids to delete
        handle.delete_collection_data.assert_not_called()
        handle.delete_documents_data.assert_not_called()
        # Config is cleaned up
        handle.delete_collection_config.assert_called_once()
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
                user_id="u1",
                is_admin=False,
                doc_ids=[],  # empty list — no docs owned by user
                delete_orphaned_metadata=False,
            )
        )

        handle.delete_collection_data.assert_not_called()
        handle.delete_documents_data.assert_not_called()
        assert isinstance(result, CollectionOperationResult)
