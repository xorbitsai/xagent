"""#515 - Coordinator failed-ingest rollback entry point.

The coordinator is the single owner of rollback ORCHESTRATION: boundary
ordering DOCUMENT -> FILE -> STATUS -> SNAPSHOT, the SNAPSHOT-after-FILE gate,
side_effects_may_remain / rollback_status inference, and error folding.
Per-plane compensation mechanics arrive as request callbacks or saga steps.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

import pytest

from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
from xagent.core.tools.core.RAG_tools.kb.models import (
    RollbackFailedIngestionRequest,
)
from xagent.core.tools.core.RAG_tools.kb.operation_compatibility import (
    KBOperation,
    PersistencePolicy,
    RollbackStatus,
    SideEffectPlane,
)


def _make_coordinator() -> KBCoordinator:
    # The rollback entry point is pure orchestration (no stores, no handle
    # opens), so a bare instance without __init__ is sufficient and keeps the
    # test offline.
    return KBCoordinator.__new__(KBCoordinator)


def _make_operation() -> KBOperation:
    return KBOperation(
        operation_type="web_page_ingestion",
        collection="demo",
        persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
    )


def _spy_callbacks(
    order: list[str],
    *,
    document_raises: Optional[Exception] = None,
    file_raises: Optional[Exception] = None,
) -> dict[str, Callable[..., Any]]:
    def document_factory(result: object = None) -> Callable[[], None]:
        def _cb() -> None:
            order.append("document")
            if document_raises is not None:
                raise document_raises

        return _cb

    def status_factory(result: object = None) -> Callable[[], None]:
        def _cb() -> None:
            order.append("status")

        return _cb

    def file_cb() -> None:
        order.append("file")
        if file_raises is not None:
            raise file_raises

    def snapshot_cb() -> None:
        order.append("snapshot")

    return {
        "document_compensation": document_factory,
        "file_compensation": file_cb,
        "status_compensation": status_factory,
        "snapshot_compensation": snapshot_cb,
    }


def _make_request(
    callbacks: dict[str, Callable[..., Any]],
    *,
    operation: Optional[KBOperation] = None,
    doc_id: Optional[str] = "doc-1",
    rollback_context: Optional[dict[str, Any]] = None,
) -> RollbackFailedIngestionRequest:
    return RollbackFailedIngestionRequest(
        collection="demo",
        user_id=None,
        is_admin=False,
        operation=operation,
        doc_id=doc_id,
        source="https://example.com/a",
        rollback_context=rollback_context or {},
        **callbacks,
    )


class TestBoundaryOrdering:
    @pytest.mark.parametrize("with_operation", [True, False])
    def test_boundaries_run_document_file_status_snapshot(
        self, with_operation: bool
    ) -> None:
        coordinator = _make_coordinator()
        order: list[str] = []
        operation = _make_operation() if with_operation else None
        request = _make_request(_spy_callbacks(order), operation=operation)

        result = coordinator.rollback_failed_ingestion_sync(request)

        assert order == ["document", "file", "status", "snapshot"]
        assert result.status == "complete"
        assert result.rollback_complete is True
        assert result.first_error is None
        assert result.boundary_errors == {}
        assert result.side_effects_may_remain is False
        assert result.rollback_status is RollbackStatus.COMPLETE


class TestSnapshotGate:
    @pytest.mark.parametrize("with_operation", [True, False])
    def test_snapshot_skipped_when_file_compensation_failed(
        self, with_operation: bool
    ) -> None:
        coordinator = _make_coordinator()
        order: list[str] = []
        operation = _make_operation() if with_operation else None
        request = _make_request(
            _spy_callbacks(order, file_raises=RuntimeError("file boom")),
            operation=operation,
        )

        result = coordinator.rollback_failed_ingestion_sync(request)

        assert "snapshot" not in order
        assert order == ["document", "file", "status"]
        assert result.status == "incomplete"
        assert result.rollback_complete is False
        assert result.first_error == "FILE boundary compensation failed: file boom"
        assert result.boundary_errors["FILE"] == ("file boom",)
        assert result.side_effects_may_remain is True
        assert result.rollback_status is RollbackStatus.INCOMPLETE

    @pytest.mark.parametrize("with_operation", [True, False])
    def test_snapshot_skipped_after_prior_boundary_error(
        self, with_operation: bool
    ) -> None:
        coordinator = _make_coordinator()
        order: list[str] = []
        operation = _make_operation() if with_operation else None
        request = _make_request(
            _spy_callbacks(order, document_raises=RuntimeError("doc boom")),
            operation=operation,
        )

        result = coordinator.rollback_failed_ingestion_sync(request)

        # FILE and STATUS still run after a DOCUMENT failure; SNAPSHOT does not.
        assert order == ["document", "file", "status"]
        assert result.first_error == "DOCUMENT boundary compensation failed: doc boom"
        assert any(
            "Web rollback DOCUMENT compensation failed for https://example.com/a: "
            "doc boom" == w
            for w in result.warnings
        )

    @pytest.mark.parametrize("with_operation", [True, False])
    def test_snapshot_runs_when_no_file_compensation_registered(
        self, with_operation: bool
    ) -> None:
        coordinator = _make_coordinator()
        order: list[str] = []
        callbacks = _spy_callbacks(order)
        callbacks.pop("file_compensation")
        operation = _make_operation() if with_operation else None
        request = _make_request(callbacks, operation=operation)

        result = coordinator.rollback_failed_ingestion_sync(request)

        assert order == ["document", "status", "snapshot"]
        assert result.status == "complete"


class TestInference:
    def test_not_needed_when_nothing_registered(self) -> None:
        coordinator = _make_coordinator()
        request = _make_request({}, operation=None)

        result = coordinator.rollback_failed_ingestion_sync(request)

        assert result.status == "not_needed"
        assert result.rollback_status is RollbackStatus.NOT_NEEDED
        assert result.rollback_complete is True
        assert result.side_effects_may_remain is False

    def test_uncompensated_operation_steps_keep_side_effects_flag(self) -> None:
        coordinator = _make_coordinator()
        operation = _make_operation()
        operation.record_side_effect(
            name="remove_parse_record",
            plane=SideEffectPlane.PARSE,
            payload={"collection": "demo"},
            idempotency_key="parse:demo:doc-1:hash",
        )
        request = _make_request({}, operation=operation)

        result = coordinator.rollback_failed_ingestion_sync(request)

        # No callbacks ran, but the saga still holds uncompensated steps.
        assert result.side_effects_may_remain is True
        assert result.rollback_status is RollbackStatus.INCOMPLETE
        # Regression (#515 review): status/rollback_complete must not
        # contradict side_effects_may_remain by claiming "not_needed"/complete.
        assert result.status == "incomplete"
        assert result.rollback_complete is False

    def test_document_compensation_marks_cascade_planes(self) -> None:
        coordinator = _make_coordinator()
        operation = _make_operation()
        # Bookkeeping steps as the pipeline facade records them (no bodies).
        operation.record_side_effect(
            name="remove_registered_document",
            plane=SideEffectPlane.DOCUMENT,
            payload={"collection": "demo", "doc_id": "doc-1"},
            idempotency_key="document:demo:doc-1",
        )
        operation.record_side_effect(
            name="remove_parse_record",
            plane=SideEffectPlane.PARSE,
            payload={"collection": "demo"},
            idempotency_key="parse:demo:doc-1:hash",
        )
        operation.record_side_effect(
            name="remove_chunk_records",
            plane=SideEffectPlane.CHUNK,
            payload={"collection": "demo"},
            idempotency_key="chunk:demo:doc-1:hash",
        )
        order: list[str] = []
        callbacks = _spy_callbacks(order)
        callbacks.pop("file_compensation")
        callbacks.pop("snapshot_compensation")
        request = _make_request(callbacks, operation=operation)

        result = coordinator.rollback_failed_ingestion_sync(request)

        assert result.status == "complete"
        assert operation.has_uncompensated_side_effects() is False
        assert result.side_effects_may_remain is False

    def test_file_step_dedupes_against_ingest_time_registration(self) -> None:
        """Key fidelity: file:{collection}:{file_id} must match web_ingestion's
        ingest-time registration so the callback runs exactly once."""
        coordinator = _make_coordinator()
        operation = _make_operation()
        calls: list[str] = []

        def file_cb() -> None:
            calls.append("file")

        operation.record_side_effect(
            name="cleanup_web_page_persistence",
            plane=SideEffectPlane.FILE,
            payload={"collection": "demo"},
            idempotency_key="file:demo:file-1",
            compensation=file_cb,
        )
        request = RollbackFailedIngestionRequest(
            collection="demo",
            user_id=None,
            is_admin=False,
            operation=operation,
            source="https://example.com/a",
            file_compensation=file_cb,
            rollback_context={"file_id": "file-1"},
        )

        result = coordinator.rollback_failed_ingestion_sync(request)

        assert calls == ["file"]
        file_steps = [
            step
            for step in operation.compensation_steps
            if step.plane is SideEffectPlane.FILE
        ]
        assert len(file_steps) == 1
        assert result.status == "complete"


class TestCallbacksOnlyAsyncCompensationRejected:
    """Regression (#515 review): an async compensation callback must fold
    into an error, not be silently discarded as a false "complete" success.
    """

    def test_async_document_compensation_folds_into_error(self) -> None:
        coordinator = _make_coordinator()

        async def _async_document_cb() -> None:
            pass

        request = RollbackFailedIngestionRequest(
            collection="demo",
            user_id=None,
            is_admin=False,
            operation=None,
            source="https://example.com/a",
            document_compensation=lambda _result: _async_document_cb,
        )

        result = coordinator.rollback_failed_ingestion_sync(request)

        assert result.status == "incomplete"
        assert result.rollback_complete is False
        assert result.side_effects_may_remain is True
        assert result.first_error is not None
        assert "DOCUMENT" in result.boundary_errors


class TestAsyncTwin:
    def test_async_form_offloads_to_sync(self) -> None:
        coordinator = _make_coordinator()
        order: list[str] = []
        request = _make_request(_spy_callbacks(order), operation=None)

        result = asyncio.run(coordinator.rollback_failed_ingestion(request))

        assert order == ["document", "file", "status", "snapshot"]
        assert result.status == "complete"
        assert result.rollback_complete is True
