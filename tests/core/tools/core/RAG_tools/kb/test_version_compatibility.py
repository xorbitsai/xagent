"""Tests for the KB version-management compatibility facade."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


class _FakeMainPointerStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str, Optional[str]], dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _key(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str],
    ) -> tuple[str, str, str, Optional[str]]:
        return (collection, doc_id, step_type, model_tag)

    def get_main_pointer(
        self,
        *,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str],
        user_id: Optional[int],
    ) -> Optional[dict[str, Any]]:
        self.calls.append(("get", dict(locals())))
        row = self.rows.get(self._key(collection, doc_id, step_type, model_tag))
        return dict(row) if row is not None else None

    def set_main_pointer(
        self,
        *,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: Optional[str],
        operator: Optional[str],
        user_id: Optional[int],
    ) -> None:
        self.calls.append(("set", dict(locals())))
        self.rows[self._key(collection, doc_id, step_type, model_tag)] = {
            "collection": collection,
            "doc_id": doc_id,
            "step_type": step_type,
            "semantic_id": semantic_id,
            "technical_id": technical_id,
            "model_tag": model_tag,
            "operator": operator,
        }

    def list_main_pointers(
        self,
        *,
        collection: str,
        doc_id: Optional[str],
        user_id: Optional[int],
        limit: int,
    ) -> list[dict[str, Any]]:
        self.calls.append(("list", dict(locals())))
        rows = [
            dict(row)
            for (row_collection, row_doc_id, _, _), row in self.rows.items()
            if row_collection == collection and (doc_id is None or row_doc_id == doc_id)
        ]
        return rows[:limit]

    def delete_main_pointer(
        self,
        *,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str],
        user_id: Optional[int],
    ) -> bool:
        self.calls.append(("delete", dict(locals())))
        return (
            self.rows.pop(self._key(collection, doc_id, step_type, model_tag), None)
            is not None
        )


class _FakeStorageShim:
    def __init__(self, main_pointer_store: _FakeMainPointerStore) -> None:
        self.main_pointer_store = main_pointer_store

    def get_main_pointer_store(self) -> _FakeMainPointerStore:
        return self.main_pointer_store


def _handle_backed_version_coordinator() -> MagicMock:
    """Coordinator mock that answers restore via the real handle logic."""
    from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
        LanceDBCollectionHandle,
    )

    coordinator = MagicMock()
    coordinator.restore_candidate_cleanup_snapshot_sync.side_effect = (
        lambda snapshot, cleanup_executed=False: (
            LanceDBCollectionHandle.restore_candidate_cleanup_snapshot(
                MagicMock(), snapshot, cleanup_executed=cleanup_executed
            )
        )
    )
    return coordinator


def _signature_shape(callable_obj: Any) -> inspect.Signature:
    return inspect.signature(callable_obj)


def test_kb_version_compatibility_public_surface_imports() -> None:
    """Given the KB package, the version facade is publicly importable."""
    import xagent.core.tools.core.RAG_tools.kb as kb
    from xagent.core.tools.core.RAG_tools.kb import (
        KBMainPointerSnapshot,
        KBVersionCandidateCleanupSnapshot,
        KBVersionCandidateRollbackResult,
        KBVersionCompatibilityFacade,
        get_kb_coordinator,
        reset_kb_coordinator_for_tests,
    )

    assert hasattr(kb, "KBVersionCompatibilityFacade")
    assert hasattr(kb, "KBMainPointerSnapshot")
    assert hasattr(kb, "KBVersionCandidateCleanupSnapshot")
    assert hasattr(kb, "KBVersionCandidateRollbackResult")
    reset_kb_coordinator_for_tests()
    coordinator = get_kb_coordinator()
    assert isinstance(coordinator.version_compatibility, KBVersionCompatibilityFacade)
    assert coordinator.version is coordinator.version_compatibility
    assert KBMainPointerSnapshot.__name__ == "KBMainPointerSnapshot"
    assert (
        KBVersionCandidateCleanupSnapshot.__name__
        == "KBVersionCandidateCleanupSnapshot"
    )
    assert (
        KBVersionCandidateRollbackResult.__name__ == "KBVersionCandidateRollbackResult"
    )


def test_version_facade_methods_match_public_helper_signatures() -> None:
    """Given legacy helpers, facade methods preserve public call signatures."""
    from xagent.core.tools.core.RAG_tools.kb import KBVersionCompatibilityFacade

    cascade_cleaner = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner"
    )
    list_candidates = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.list_candidates"
    )
    main_pointer_manager = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager"
    )
    promote_version_main = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.promote_version_main"
    )

    facade = KBVersionCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeMainPointerStore())
    )
    pairs = [
        (facade.list_candidates, list_candidates.list_candidates),
        (facade.promote_version_main, promote_version_main.promote_version_main),
        (facade.get_main_pointer, main_pointer_manager.get_main_pointer),
        (facade.set_main_pointer, main_pointer_manager.set_main_pointer),
        (facade.list_main_pointers, main_pointer_manager.list_main_pointers),
        (facade.delete_main_pointer, main_pointer_manager.delete_main_pointer),
        (facade.cascade_delete, cascade_cleaner.cascade_delete),
        (facade.cleanup_cascade, cascade_cleaner.cleanup_cascade),
        (facade.cleanup_document_cascade, cascade_cleaner.cleanup_document_cascade),
        (facade.cleanup_parse_cascade, cascade_cleaner.cleanup_parse_cascade),
        (facade.cleanup_chunk_cascade, cascade_cleaner.cleanup_chunk_cascade),
        (facade.cleanup_embed_cascade, cascade_cleaner.cleanup_embed_cascade),
    ]

    for facade_method, public_helper in pairs:
        assert _signature_shape(facade_method) == _signature_shape(public_helper)


def test_version_management_package_exports_retained_functions_and_cascade_delete() -> (
    None
):
    """Given package-level imports, retained version helpers remain available."""
    import xagent.core.tools.core.RAG_tools.version_management as version_management

    expected = {
        "list_candidates",
        "promote_version_main",
        "get_main_pointer",
        "set_main_pointer",
        "list_main_pointers",
        "delete_main_pointer",
        "cleanup_cascade",
        "cleanup_document_cascade",
        "cleanup_parse_cascade",
        "cleanup_chunk_cascade",
        "cleanup_embed_cascade",
        "cascade_delete",
    }

    assert expected.issubset(set(version_management.__all__))
    for name in expected:
        assert hasattr(version_management, name)
        assert not inspect.iscoroutinefunction(getattr(version_management, name))


def test_public_version_helpers_remain_sync_and_route_through_facade(
    monkeypatch,
) -> None:
    """Given public version helper calls, they synchronously route through facade."""
    cascade_cleaner = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner"
    )
    list_candidates = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.list_candidates"
    )
    main_pointer_manager = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager"
    )
    promote_version_main = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.promote_version_main"
    )

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class _FakeFacade:
        def list_candidates(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(("list_candidates", args, kwargs))
            return {"candidates": [], "total_count": 0, "returned_count": 0}

        def promote_version_main(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(("promote_version_main", args, kwargs))
            return {"promoted": False, "preview": True}

        def get_main_pointer(self, *args: Any, **kwargs: Any) -> None:
            calls.append(("get_main_pointer", args, kwargs))
            return None

        def cascade_delete(self, *args: Any, **kwargs: Any) -> dict[str, int]:
            calls.append(("cascade_delete", args, kwargs))
            return {"documents": 1}

    fake_facade = _FakeFacade()
    monkeypatch.setattr(
        list_candidates,
        "_get_version_compatibility_facade",
        lambda: fake_facade,
    )
    monkeypatch.setattr(
        promote_version_main,
        "_get_version_compatibility_facade",
        lambda: fake_facade,
    )
    monkeypatch.setattr(
        main_pointer_manager,
        "_get_version_compatibility_facade",
        lambda: fake_facade,
    )
    monkeypatch.setattr(
        cascade_cleaner,
        "_get_version_compatibility_facade",
        lambda: fake_facade,
    )

    assert not inspect.iscoroutinefunction(list_candidates.list_candidates)
    assert not inspect.iscoroutinefunction(promote_version_main.promote_version_main)
    assert not inspect.iscoroutinefunction(main_pointer_manager.get_main_pointer)
    assert not inspect.iscoroutinefunction(cascade_cleaner.cascade_delete)

    assert list_candidates.list_candidates("docs", "doc-1", "parse") == {
        "candidates": [],
        "total_count": 0,
        "returned_count": 0,
    }
    assert promote_version_main.promote_version_main(
        "docs", "doc-1", "parse", "parse_manual_abc", preview_only=True
    ) == {"promoted": False, "preview": True}
    assert main_pointer_manager.get_main_pointer("docs", "doc-1", "parse") is None
    assert cascade_cleaner.cascade_delete(
        target="document", collection="docs", doc_id="doc-1"
    ) == {"documents": 1}

    assert calls == [
        (
            "list_candidates",
            (),
            {
                "collection": "docs",
                "doc_id": "doc-1",
                "step_type": "parse",
                "model_tag": None,
                "state": None,
                "limit": 50,
                "order_by": "created_at desc",
            },
        ),
        (
            "promote_version_main",
            (),
            {
                "collection": "docs",
                "doc_id": "doc-1",
                "step_type": "parse",
                "selected_id": "parse_manual_abc",
                "operator": None,
                "preview_only": True,
                "confirm": False,
                "model_tag": None,
            },
        ),
        (
            "get_main_pointer",
            (),
            {
                "collection": "docs",
                "doc_id": "doc-1",
                "step_type": "parse",
                "model_tag": None,
            },
        ),
        (
            "cascade_delete",
            (),
            {
                "target": "document",
                "collection": "docs",
                "doc_id": "doc-1",
                "user_id": None,
                "is_admin": None,
                "model_tag": None,
                "preview_only": True,
                "confirm": False,
            },
        ),
    ]


def test_version_facade_rebinds_coordinator_storage_for_main_pointers() -> None:
    """Facade delegation rebinds storage factory access to its coordinator shim."""
    from xagent.core.tools.core.RAG_tools.kb import KBVersionCompatibilityFacade
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        bind_storage_shim_for_current_context,
        get_main_pointer_store,
    )

    outer_store = _FakeMainPointerStore()
    inner_store = _FakeMainPointerStore()
    outer_shim = _FakeStorageShim(outer_store)
    inner_shim = _FakeStorageShim(inner_store)
    facade = KBVersionCompatibilityFacade(storage_shim=inner_shim)

    with bind_storage_shim_for_current_context(outer_shim):
        assert get_main_pointer_store() is outer_store
        facade.set_main_pointer(
            "",
            "docs",
            "doc-inner",
            "parse",
            "parse_manual_abc",
            "parse-hash-abc",
            operator="tester",
        )
        assert facade.get_main_pointer("docs", "doc-inner", "parse")[
            "technical_id"
        ] == ("parse-hash-abc")
        assert get_main_pointer_store() is outer_store

    assert outer_store.rows == {}
    assert (
        inner_store.list_main_pointers(
            collection="docs", doc_id="doc-inner", user_id=None, limit=100
        )[0]["semantic_id"]
        == "parse_manual_abc"
    )


def test_main_pointer_snapshot_restore_reverts_mutated_pointer() -> None:
    """Rollback-facing snapshot hooks can restore the previous main pointer."""
    from xagent.core.tools.core.RAG_tools.kb import KBVersionCompatibilityFacade

    store = _FakeMainPointerStore()
    facade = KBVersionCompatibilityFacade(storage_shim=_FakeStorageShim(store))
    facade.set_main_pointer(
        "",
        "docs",
        "doc-1",
        "parse",
        "parse_old",
        "old-hash",
        operator="setup",
    )

    snapshot = facade.capture_main_pointer_snapshot("docs", "doc-1", "parse")
    facade.set_main_pointer(
        "",
        "docs",
        "doc-1",
        "parse",
        "parse_new",
        "new-hash",
        operator="mutator",
    )

    assert (
        facade.get_main_pointer("docs", "doc-1", "parse")["technical_id"] == "new-hash"
    )
    assert facade.restore_main_pointer_snapshot(snapshot, operator="rollback")
    restored = facade.get_main_pointer("docs", "doc-1", "parse")
    assert restored is not None
    assert restored["semantic_id"] == "parse_old"
    assert restored["technical_id"] == "old-hash"


def test_main_pointer_snapshot_restore_succeeds_when_pointer_already_absent() -> None:
    """Restoring an absent pointer succeeds when the desired state is already met."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBMainPointerSnapshot,
        KBVersionCompatibilityFacade,
    )

    store = _FakeMainPointerStore()
    facade = KBVersionCompatibilityFacade(storage_shim=_FakeStorageShim(store))
    snapshot = KBMainPointerSnapshot(
        collection="docs",
        doc_id="doc-1",
        step_type="parse",
        model_tag=None,
        pointer=None,
    )

    assert facade.restore_main_pointer_snapshot(snapshot)
    assert store.calls[-1][0] == "delete"
    assert facade.list_main_pointers("docs") == []


def test_main_pointer_snapshot_restore_returns_false_for_incomplete_pointer() -> None:
    """Incomplete main-pointer snapshots report restore failure without crashing."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBMainPointerSnapshot,
        KBVersionCompatibilityFacade,
    )

    facade = KBVersionCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeMainPointerStore())
    )
    missing_semantic = KBMainPointerSnapshot(
        collection="docs",
        doc_id="doc-1",
        step_type="parse",
        model_tag=None,
        pointer={"technical_id": "parse-hash"},
    )
    missing_technical = KBMainPointerSnapshot(
        collection="docs",
        doc_id="doc-1",
        step_type="parse",
        model_tag=None,
        pointer={"semantic_id": "parse_manual_hash"},
    )

    assert not facade.restore_main_pointer_snapshot(missing_semantic)
    assert not facade.restore_main_pointer_snapshot(missing_technical)
    assert facade.list_main_pointers("docs") == []


def test_candidate_cleanup_snapshot_records_preview_counts(monkeypatch) -> None:
    """Candidate cleanup snapshots record preview counts without deleting rows."""
    from xagent.core.tools.core.RAG_tools.kb import KBVersionCompatibilityFacade

    cascade_cleaner = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner"
    )
    calls: list[dict[str, Any]] = []

    def fake_cleanup_cascade_impl(**kwargs: Any) -> dict[str, int]:
        calls.append(dict(kwargs))
        return {"chunks": 2, "embeddings": 5}

    monkeypatch.setattr(
        cascade_cleaner,
        "_cleanup_cascade_impl",
        fake_cleanup_cascade_impl,
    )

    snapshot = KBVersionCompatibilityFacade().capture_candidate_cleanup_snapshot(
        "docs",
        "doc-1",
        "parse",
        new_parse_hash="new-hash",
        old_parse_hash="old-hash",
        model_tag="text-embedding-3-small",
        user_id=7,
        is_admin=False,
    )

    assert snapshot.cleanup_counts == {"chunks": 2, "embeddings": 5}
    assert snapshot.collection == "docs"
    assert snapshot.doc_id == "doc-1"
    assert snapshot.scope == "parse"
    assert calls == [
        {
            "collection": "docs",
            "doc_id": "doc-1",
            "scope": "parse",
            "new_parse_hash": "new-hash",
            "old_parse_hash": "old-hash",
            "model_tag": "text-embedding-3-small",
            "user_id": 7,
            "is_admin": False,
            "preview_only": True,
            "confirm": False,
        }
    ]


def test_candidate_cleanup_restore_marks_remaining_side_effects() -> None:
    """Executed candidate cleanup is reported as incomplete, not fake-restored."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBVersionCandidateCleanupSnapshot,
        KBVersionCompatibilityFacade,
    )

    snapshot = KBVersionCandidateCleanupSnapshot(
        collection="docs",
        doc_id="doc-1",
        scope="parse",
        cleanup_counts={"chunks": 2, "embeddings": 5},
        new_parse_hash="new-hash",
        old_parse_hash="old-hash",
    )

    result = KBVersionCompatibilityFacade(
        coordinator=_handle_backed_version_coordinator()
    ).restore_candidate_cleanup_snapshot(snapshot, cleanup_executed=True)

    assert result.status == "incomplete"
    assert result.skipped
    assert result.reason == "candidate_cleanup_not_restorable"
    assert not result.restorable
    assert result.side_effects_may_remain
    assert result.cleanup_counts == {"chunks": 2, "embeddings": 5}
    assert result.warnings


def test_candidate_cleanup_restore_does_not_delete_active_artifacts(
    monkeypatch,
) -> None:
    """Rollback-incomplete restore reports state without issuing more cleanup."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBVersionCandidateCleanupSnapshot,
        KBVersionCompatibilityFacade,
    )

    cascade_cleaner = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner"
    )
    cleanup_calls: list[dict[str, Any]] = []

    def fake_cleanup_cascade_impl(**kwargs: Any) -> dict[str, int]:
        cleanup_calls.append(dict(kwargs))
        raise AssertionError("restore must not delete active version artifacts")

    monkeypatch.setattr(
        cascade_cleaner,
        "_cleanup_cascade_impl",
        fake_cleanup_cascade_impl,
    )

    snapshot = KBVersionCandidateCleanupSnapshot(
        collection="docs",
        doc_id="doc-1",
        scope="parse",
        cleanup_counts={"parses": 1, "chunks": 2},
    )
    result = KBVersionCompatibilityFacade(
        coordinator=_handle_backed_version_coordinator()
    ).restore_candidate_cleanup_snapshot(snapshot, cleanup_executed=True)

    assert result.status == "incomplete"
    assert result.side_effects_may_remain
    assert cleanup_calls == []


def test_candidate_cleanup_restore_not_needed_when_cleanup_was_not_executed() -> None:
    """A captured plan does not imply residual side effects before cleanup runs."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBVersionCandidateCleanupSnapshot,
        KBVersionCompatibilityFacade,
    )

    snapshot = KBVersionCandidateCleanupSnapshot(
        collection="docs",
        doc_id="doc-1",
        scope="parse",
        cleanup_counts={"chunks": 2},
    )

    result = KBVersionCompatibilityFacade(
        coordinator=_handle_backed_version_coordinator()
    ).restore_candidate_cleanup_snapshot(snapshot)

    assert result.status == "not_needed"
    assert result.restorable
    assert not result.skipped
    assert not result.side_effects_may_remain


def test_cleanup_count_shape_is_preserved_through_version_facade(monkeypatch) -> None:
    """Cleanup count keys and values pass through facade delegation unchanged."""
    from xagent.core.tools.core.RAG_tools.kb import KBVersionCompatibilityFacade

    cascade_cleaner = importlib.import_module(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner"
    )

    expected = {
        "embeddings": 3,
        "chunks": 2,
        "parses": 1,
        "main_pointers": 1,
        "documents": 1,
        "ingestion_runs": 1,
    }
    calls: list[dict[str, Any]] = []

    def fake_cleanup_document_cascade_impl(**kwargs: Any) -> dict[str, int]:
        calls.append(dict(kwargs))
        return expected

    monkeypatch.setattr(
        cascade_cleaner,
        "_cleanup_document_cascade_impl",
        fake_cleanup_document_cascade_impl,
    )

    result = KBVersionCompatibilityFacade().cleanup_document_cascade(
        "docs",
        "doc-1",
        model_tag="text-embedding-3-small",
        user_id=7,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    assert result == expected
    assert calls == [
        {
            "collection": "docs",
            "doc_id": "doc-1",
            "model_tag": "text-embedding-3-small",
            "user_id": 7,
            "is_admin": False,
            "preview_only": False,
            "confirm": True,
        }
    ]


def test_failed_promotion_does_not_advance_main_pointer() -> None:
    """Failed cleanup during promotion does not call set_main_pointer."""
    from xagent.core.tools.core.RAG_tools.core.exceptions import VersionManagementError
    from xagent.core.tools.core.RAG_tools.core.schemas import StepType
    from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
        LanceDBCollectionHandle,
    )
    from xagent.core.tools.core.RAG_tools.version_management.promote_version_main import (
        promote_version_main,
    )

    mock_list_candidates = MagicMock(
        return_value={
            "candidates": [
                {
                    "technical_id": "new-hash",
                    "semantic_id": "parse_new",
                }
            ]
        }
    )
    mock_cleanup_plan = MagicMock(
        return_value={
            "deleted_counts": {"chunks": 1, "embeddings": 1},
            "notes": [],
            "current_pointer": {
                "technical_id": "old-hash",
                "semantic_id": "parse_old",
            },
            "new_technical_id": "new-hash",
        }
    )
    mock_cleanup = MagicMock(side_effect=RuntimeError("cleanup failed"))
    mock_set_main_pointer = MagicMock()

    with patch.object(LanceDBCollectionHandle, "list_candidates", mock_list_candidates):
        with patch.object(
            LanceDBCollectionHandle,
            "_calculate_cleanup_plan_for_step",
            mock_cleanup_plan,
        ):
            with patch.object(LanceDBCollectionHandle, "cleanup_cascade", mock_cleanup):
                with patch.object(
                    LanceDBCollectionHandle, "set_main_pointer", mock_set_main_pointer
                ):
                    with pytest.raises(VersionManagementError, match="cleanup failed"):
                        promote_version_main(
                            "docs",
                            "doc-1",
                            StepType.PARSE,
                            "parse_new",
                            operator="tester",
                            confirm=True,
                        )

    mock_set_main_pointer.assert_not_called()


def test_facade_cascade_delete_routes_to_store(mocker) -> None:
    """facade.cascade_delete should delegate to get_vector_index_store().cascade_delete."""
    from xagent.core.tools.core.RAG_tools.kb import KBVersionCompatibilityFacade

    fake_store = mocker.Mock()
    fake_store.cascade_delete.return_value = {"documents": 3}
    mocker.patch(
        "xagent.core.tools.core.RAG_tools.storage.factory.get_vector_index_store",
        return_value=fake_store,
    )
    facade = KBVersionCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeMainPointerStore())
    )

    out = facade.cascade_delete(
        target="collection",
        collection="c1",
        user_id=1,
        is_admin=True,
        preview_only=False,
        confirm=True,
    )

    assert out == {"documents": 3}
    fake_store.cascade_delete.assert_called_once()
    kw = fake_store.cascade_delete.call_args.kwargs
    assert kw["target"] == "collection"
    assert kw["collection"] == "c1"
    assert "conn" not in kw
