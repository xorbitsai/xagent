"""Tests for KB operation rollback compatibility outcomes."""

from __future__ import annotations

from contextvars import Context
from typing import Callable, cast

import pytest

from xagent.core.tools.core.RAG_tools.kb import (
    KBOperationCompatibilityFacade,
    RollbackStatus,
    SideEffectPlane,
)
from xagent.core.tools.core.RAG_tools.kb.operation_compatibility import (
    KBOperation,
    PersistencePolicy,
    finish_ingestion_outcome,
    record_chunk_side_effect,
    record_document_registration_side_effect,
    record_parse_side_effect,
)


def test_operation_compensation_steps_are_idempotent_and_lifo() -> None:
    facade = KBOperationCompatibilityFacade()

    with facade.start_operation(
        operation_type="document_ingestion",
        collection="demo",
    ) as operation:
        operation.record_side_effect(
            name="remove_document",
            plane=SideEffectPlane.DOCUMENT,
            payload={"doc_id": "doc-1"},
            idempotency_key="document:doc-1",
        )
        operation.record_side_effect(
            name="remove_parse",
            plane=SideEffectPlane.PARSE,
            payload={"parse_hash": "parse-1"},
            idempotency_key="parse:parse-1",
        )
        operation.record_side_effect(
            name="remove_document",
            plane=SideEffectPlane.DOCUMENT,
            payload={"doc_id": "doc-1"},
            idempotency_key="document:doc-1",
        )
        operation.finish(
            status="partial",
            rollback_status=RollbackStatus.INCOMPLETE,
            side_effects_may_remain=True,
        )

    outcome = facade.last_outcome

    assert outcome is not None
    assert [step.name for step in outcome.compensation_steps] == [
        "remove_document",
        "remove_parse",
    ]
    assert [step.name for step in outcome.compensation_plan] == [
        "remove_parse",
        "remove_document",
    ]
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True


def test_operation_executes_registered_compensations_lifo_and_marks_complete() -> None:
    facade = KBOperationCompatibilityFacade()
    calls: list[str] = []

    with facade.start_operation(
        operation_type="document_ingestion",
        collection="demo",
    ) as operation:
        operation.record_side_effect(
            name="remove_document",
            plane=SideEffectPlane.DOCUMENT,
            idempotency_key="document:doc-1",
            compensation=lambda: calls.append("document"),
        )
        operation.record_side_effect(
            name="remove_parse",
            plane=SideEffectPlane.PARSE,
            idempotency_key="parse:parse-1",
            compensation=lambda: calls.append("parse"),
        )

        assert operation.execute_compensations() == ()
        operation.finish(status="error")

    outcome = facade.last_outcome

    assert calls == ["parse", "document"]
    assert outcome is not None
    assert outcome.rollback_status is RollbackStatus.COMPLETE
    assert outcome.side_effects_may_remain is False


def test_operation_partial_compensation_leaves_uncovered_steps_incomplete() -> None:
    facade = KBOperationCompatibilityFacade()
    calls: list[str] = []

    with facade.start_operation(
        operation_type="web_page_ingestion",
        collection="demo",
    ) as operation:
        operation.record_side_effect(
            name="cleanup_web_page_persistence",
            plane=SideEffectPlane.FILE,
            idempotency_key="file:page-1",
            compensation=lambda: calls.append("file"),
        )
        operation.record_side_effect(
            name="remove_registered_document",
            plane=SideEffectPlane.DOCUMENT,
            idempotency_key="document:doc-1",
        )

        assert operation.execute_compensations(planes={SideEffectPlane.FILE}) == ()
        operation.finish(status="error", side_effects_may_remain=False)

    outcome = facade.last_outcome

    assert calls == ["file"]
    assert outcome is not None
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert [step.plane for step in outcome.compensation_steps] == [
        SideEffectPlane.FILE,
        SideEffectPlane.DOCUMENT,
    ]


def test_failed_compensation_remains_retryable_until_it_succeeds() -> None:
    facade = KBOperationCompatibilityFacade()
    attempts = 0

    def compensation() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first failure")

    with facade.start_operation(
        operation_type="document_ingestion",
        collection="demo",
    ) as operation:
        operation.record_side_effect(
            name="remove_document",
            plane=SideEffectPlane.DOCUMENT,
            idempotency_key="document:doc-1",
            compensation=compensation,
        )

        first_errors = operation.execute_compensations()
        second_errors = operation.execute_compensations()
        operation.finish(status="error", side_effects_may_remain=bool(second_errors))

    outcome = facade.last_outcome

    assert len(first_errors) == 1
    assert second_errors == ()
    assert attempts == 2
    assert outcome is not None
    assert outcome.rollback_status is RollbackStatus.COMPLETE
    assert outcome.side_effects_may_remain is False
    assert "first failure" in outcome.warnings[0]


def test_async_compensation_is_rejected_without_marking_complete(recwarn) -> None:
    facade = KBOperationCompatibilityFacade()
    calls: list[str] = []

    async def compensation() -> None:
        calls.append("compensated")

    with facade.start_operation(
        operation_type="document_ingestion",
        collection="demo",
    ) as operation:
        operation.record_side_effect(
            name="remove_document",
            plane=SideEffectPlane.DOCUMENT,
            idempotency_key="document:doc-1",
            compensation=cast(Callable[[], None], compensation),
        )

        first_errors = operation.execute_compensations()
        second_errors = operation.execute_compensations()
        operation.finish(status="error")

    outcome = facade.last_outcome

    assert calls == []
    assert len(first_errors) == 1
    assert len(second_errors) == 1
    assert isinstance(first_errors[0], TypeError)
    assert isinstance(second_errors[0], TypeError)
    assert outcome is not None
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert len(outcome.warnings) == 2
    assert "Async compensation callback is not supported" in outcome.warnings[0]
    assert not any("was never awaited" in str(item.message) for item in recwarn)


def test_system_exit_from_compensation_propagates_and_records_outcome() -> None:
    facade = KBOperationCompatibilityFacade()

    def compensation() -> None:
        raise SystemExit("stop")

    with pytest.raises(SystemExit):
        with facade.start_operation(
            operation_type="document_ingestion",
            collection="demo",
        ) as operation:
            operation.record_side_effect(
                name="remove_document",
                plane=SideEffectPlane.DOCUMENT,
                idempotency_key="document:doc-1",
                compensation=compensation,
            )
            operation.execute_compensations()

    outcome = facade.last_outcome
    assert outcome is not None
    assert outcome.status == "error"
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert outcome.warnings == ("SystemExit: stop",)


def test_last_outcome_is_isolated_by_execution_context() -> None:
    facade = KBOperationCompatibilityFacade()
    initial_current_context_outcome = facade.last_outcome

    def run_operation(collection: str):
        with facade.start_operation(
            operation_type="document_ingestion",
            collection=collection,
        ):
            pass

        outcome = facade.last_outcome
        assert outcome is not None
        return outcome

    context_a = Context()
    context_b = Context()

    outcome_a = context_a.run(run_operation, "collection-a")
    outcome_b = context_b.run(run_operation, "collection-b")

    assert outcome_a.collection == "collection-a"
    assert outcome_b.collection == "collection-b"
    assert context_a.run(lambda: facade.last_outcome) is outcome_a
    assert context_b.run(lambda: facade.last_outcome) is outcome_b
    assert facade.last_outcome is initial_current_context_outcome


class _OperationCancelled(BaseException):
    pass


def test_operation_base_exception_records_error_outcome() -> None:
    facade = KBOperationCompatibilityFacade()

    with pytest.raises(_OperationCancelled):
        with facade.start_operation(
            operation_type="document_ingestion",
            collection="demo",
        ) as operation:
            operation.record_side_effect(
                name="remove_document",
                plane=SideEffectPlane.DOCUMENT,
                payload={"doc_id": "doc-1"},
                idempotency_key="document:doc-1",
            )
            raise _OperationCancelled("cancelled")

    outcome = facade.last_outcome
    assert outcome is not None
    assert outcome.status == "error"
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert outcome.warnings == ("_OperationCancelled: cancelled",)


def test_operation_exception_warning_includes_exception_type() -> None:
    facade = KBOperationCompatibilityFacade()

    with pytest.raises(KeyError):
        with facade.start_operation(
            operation_type="document_ingestion",
            collection="demo",
        ):
            raise KeyError("doc_id")

    outcome = facade.last_outcome
    assert outcome is not None
    assert outcome.status == "error"
    assert outcome.rollback_status is RollbackStatus.NOT_NEEDED
    assert outcome.warnings == ("KeyError: 'doc_id'",)


def test_snapshot_plane_mark_and_uncompensated_tracking() -> None:
    """SNAPSHOT plane participates in mark_compensated_steps and uncompensated_steps."""
    facade = KBOperationCompatibilityFacade()

    with facade.start_operation(
        operation_type="web_page_ingestion",
        collection="demo",
    ) as operation:
        operation.record_side_effect(
            name="cleanup_backup_file",
            plane=SideEffectPlane.SNAPSHOT,
            payload={"backup_path": "/tmp/backup"},
            idempotency_key="snapshot:/tmp/backup",
        )
        assert SideEffectPlane.SNAPSHOT.value == "snapshot"
        uncompensated = operation.uncompensated_steps()
        assert len(uncompensated) == 1
        assert uncompensated[0].plane is SideEffectPlane.SNAPSHOT

        operation.mark_compensated_steps(planes={SideEffectPlane.SNAPSHOT})
        assert operation.has_uncompensated_side_effects() is False


def _engine_test_operation() -> KBOperation:
    return KBOperation(
        operation_type="document_ingestion",
        collection="demo",
        persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
    )


def test_engine_recording_helpers_use_canonical_keys_and_payloads() -> None:
    operation = _engine_test_operation()

    record_document_registration_side_effect(
        operation,
        collection="demo",
        doc_id="doc-1",
        created=True,
        source_path="/tmp/a.md",
        file_id="file-1",
        user_id=7,
    )
    record_parse_side_effect(
        operation, collection="demo", doc_id="doc-1", parse_hash="hash-1"
    )
    record_chunk_side_effect(
        operation,
        collection="demo",
        doc_id="doc-1",
        parse_hash="hash-1",
        chunk_count=3,
    )

    by_key = {step.idempotency_key: step for step in operation.compensation_steps}
    assert set(by_key) == {
        "document:demo:doc-1",
        "status:demo:doc-1",
        "parse:demo:doc-1:hash-1",
        "chunk:demo:doc-1:hash-1",
    }
    assert by_key["document:demo:doc-1"].name == "remove_registered_document"
    assert by_key["document:demo:doc-1"].plane is SideEffectPlane.DOCUMENT
    assert by_key["document:demo:doc-1"].payload == {
        "collection": "demo",
        "doc_id": "doc-1",
        "created": True,
        "source_path": "/tmp/a.md",
        "file_id": "file-1",
    }
    assert by_key["status:demo:doc-1"].name == "clear_ingestion_status"
    assert by_key["status:demo:doc-1"].plane is SideEffectPlane.STATUS
    assert by_key["status:demo:doc-1"].payload == {
        "collection": "demo",
        "doc_id": "doc-1",
        "user_id": 7,
    }
    assert by_key["parse:demo:doc-1:hash-1"].plane is SideEffectPlane.PARSE
    assert by_key["chunk:demo:doc-1:hash-1"].payload["chunk_count"] == 3


def test_engine_recording_helpers_are_none_safe() -> None:
    # operation=None is a no-op for every recording helper.
    record_document_registration_side_effect(
        None,
        collection="demo",
        doc_id="doc-1",
        created=True,
        source_path="/tmp/a.md",
        file_id=None,
        user_id=None,
    )
    record_parse_side_effect(None, collection="demo", doc_id="d", parse_hash="h")
    record_chunk_side_effect(
        None, collection="demo", doc_id="d", parse_hash="h", chunk_count=1
    )


def test_finish_ingestion_outcome_preserves_recorded_side_effect_semantics() -> None:
    """#515 risk 6.3 pin: with the default flag, recorded-but-COMPENSATED side
    effects still report side_effects_may_remain=True on failure (the
    historical has_side_effects() facade semantics, not the engine default)."""
    operation = _engine_test_operation()
    operation.record_side_effect(
        name="remove_registered_document",
        plane=SideEffectPlane.DOCUMENT,
        payload={"collection": "demo"},
        idempotency_key="document:demo:doc-1",
        compensation=lambda: None,
    )
    errors = operation.execute_compensations(
        step_names={"remove_registered_document"},
        planes={SideEffectPlane.DOCUMENT},
    )
    assert errors == ()
    assert operation.has_uncompensated_side_effects() is False

    outcome = finish_ingestion_outcome(operation, status="error", message="failed")

    assert outcome is not None
    assert outcome.side_effects_may_remain is True
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE


def test_finish_ingestion_outcome_success_clears_flag() -> None:
    operation = _engine_test_operation()
    record_parse_side_effect(operation, collection="demo", doc_id="d", parse_hash="h")

    outcome = finish_ingestion_outcome(operation, status="success", message="ok")

    assert outcome is not None
    assert outcome.side_effects_may_remain is False
    assert outcome.rollback_status is RollbackStatus.NOT_NEEDED
