"""Coordinator-owned KB operation outcome compatibility facade."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Optional, cast
from uuid import uuid4


class RollbackStatus(StrEnum):
    """Explicit rollback state for a KB compatibility operation."""

    NOT_NEEDED = "not_needed"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    SKIPPED_BY_POLICY = "skipped_by_policy"


class PersistencePolicy(StrEnum):
    """Batch operation policy for successful child side effects."""

    PRESERVE_SUCCESSFUL_CHILDREN = "preserve_successful_children"
    ROLLBACK_ALL_CHILDREN = "rollback_all_children"


class SideEffectPlane(StrEnum):
    """Storage plane touched by a compensatable KB operation step."""

    COLLECTION = "collection"
    DOCUMENT = "document"
    PARSE = "parse"
    CHUNK = "chunk"
    EMBEDDING = "embedding"
    STATUS = "status"
    FILE = "file"
    SNAPSHOT = "snapshot"
    WEB_PAGE = "web_page"


@dataclass(frozen=True)
class CompensationStep:
    """Structured description of a side effect and its compensation boundary."""

    name: str
    plane: SideEffectPlane
    payload: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class KBOperationOutcome:
    """Internal outcome model for rollback-aware KB compatibility operations."""

    operation_id: str
    operation_type: str
    collection: str
    status: str
    rollback_status: RollbackStatus
    persistence_policy: PersistencePolicy
    compensation_steps: tuple[CompensationStep, ...] = ()
    child_outcomes: tuple["KBOperationOutcome", ...] = ()
    warnings: tuple[str, ...] = ()
    side_effects_may_remain: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def compensation_plan(self) -> tuple[CompensationStep, ...]:
        """Return compensation steps in LIFO execution order."""
        return tuple(reversed(self.compensation_steps))


def _close_awaitable_if_possible(value: Any) -> None:
    """Close coroutine-like objects that cannot be awaited by a sync caller."""
    close = getattr(value, "close", None)
    if callable(close):
        close()


class KBOperation:
    """Mutable operation builder stored only in the current execution context."""

    def __init__(
        self,
        *,
        operation_type: str,
        collection: str,
        persistence_policy: PersistencePolicy,
        operation_id: Optional[str] = None,
        parent_operation_id: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.operation_id = operation_id or str(uuid4())
        self.operation_type = operation_type
        self.collection = collection
        self.persistence_policy = persistence_policy
        self.parent_operation_id = parent_operation_id
        self.details: dict[str, Any] = dict(details or {})
        self.compensation_steps: list[CompensationStep] = []
        self.child_outcomes: list[KBOperationOutcome] = []
        self.warnings: list[str] = []
        self.side_effects_may_remain = False
        self._idempotency_keys: set[str] = set()
        self._compensation_callbacks: dict[str, Callable[[], None]] = {}
        self._completed_compensation_keys: set[str] = set()
        self._compensation_attempted = False
        self._outcome: KBOperationOutcome | None = None

    @property
    def outcome(self) -> KBOperationOutcome | None:
        """Return the finalized outcome when available."""
        return self._outcome

    def has_side_effects(self) -> bool:
        """Return whether this operation or any child recorded side effects."""
        return bool(self.compensation_steps) or any(
            child.side_effects_may_remain or child.compensation_steps
            for child in self.child_outcomes
        )

    def uncompensated_steps(self) -> tuple[CompensationStep, ...]:
        """Return own side effects not covered by successful compensation."""
        return tuple(
            step
            for step in self.compensation_steps
            if (
                step.idempotency_key is None
                or step.idempotency_key not in self._completed_compensation_keys
            )
        )

    def has_uncompensated_side_effects(self) -> bool:
        """Return whether any recorded side effect still lacks compensation."""
        if self.uncompensated_steps():
            return True
        return any(
            child.side_effects_may_remain
            or child.rollback_status is RollbackStatus.INCOMPLETE
            for child in self.child_outcomes
        )

    @property
    def compensation_attempted(self) -> bool:
        """Return whether this operation attempted executable compensation."""
        return self._compensation_attempted

    def update_details(self, **details: Any) -> None:
        """Merge operation metadata captured during execution."""
        self.details.update(details)

    def record_side_effect(
        self,
        *,
        name: str,
        plane: SideEffectPlane,
        payload: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        compensation: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register an idempotent compensation boundary for one side effect."""
        step_payload = dict(payload or {})
        dedupe_key = idempotency_key or f"{plane.value}:{name}:{step_payload!r}"
        if dedupe_key in self._idempotency_keys:
            if compensation is not None:
                self._compensation_callbacks.setdefault(dedupe_key, compensation)
            return

        self._idempotency_keys.add(dedupe_key)
        if compensation is not None:
            self._compensation_callbacks[dedupe_key] = compensation
        self.compensation_steps.append(
            CompensationStep(
                name=name,
                plane=plane,
                payload=step_payload,
                idempotency_key=dedupe_key,
            )
        )

    def add_child_outcome(self, outcome: KBOperationOutcome) -> None:
        """Attach a finalized child operation outcome."""
        self.child_outcomes.append(outcome)

    def mark_compensated_steps(
        self,
        *,
        step_names: Optional[set[str]] = None,
        planes: Optional[set[SideEffectPlane]] = None,
    ) -> int:
        """Mark steps covered by a broader successful compensation callback."""
        if step_names is None and planes is None:
            return 0

        completed = 0
        for step in self.compensation_steps:
            if step_names is not None and step.name not in step_names:
                continue
            if planes is not None and step.plane not in planes:
                continue
            if step.idempotency_key is None:
                continue
            if step.idempotency_key in self._completed_compensation_keys:
                continue

            self._completed_compensation_keys.add(step.idempotency_key)
            completed += 1

        if completed:
            self.side_effects_may_remain = self.has_uncompensated_side_effects()
        return completed

    def execute_compensations(
        self,
        *,
        step_names: Optional[set[str]] = None,
        planes: Optional[set[SideEffectPlane]] = None,
    ) -> tuple[BaseException, ...]:
        """Execute registered compensation callbacks in LIFO order.

        Callbacks are kept outside the immutable outcome so public result shapes
        stay serializable. Successful callbacks are not re-run, while failed
        callbacks remain retryable because the idempotency key stays registered.
        """
        errors: list[BaseException] = []
        attempted = False
        for step in reversed(self.compensation_steps):
            if step_names is not None and step.name not in step_names:
                continue
            if planes is not None and step.plane not in planes:
                continue
            if step.idempotency_key is None:
                continue
            if step.idempotency_key in self._completed_compensation_keys:
                continue

            callback = self._compensation_callbacks.get(step.idempotency_key)
            if callback is None:
                continue

            self._compensation_attempted = True
            attempted = True
            try:
                result = cast(Callable[[], Any], callback)()
                if inspect.isawaitable(result):
                    _close_awaitable_if_possible(result)
                    raise TypeError(
                        "Async compensation callback is not supported in synchronous "
                        f"execute_compensations for step {step.name}"
                    )
            except Exception as exc:  # noqa: BLE001 - preserve retryability
                errors.append(exc)
                self.warnings.append(f"{step.name}: {_format_exception_warning(exc)}")
            else:
                self._completed_compensation_keys.add(step.idempotency_key)

        if attempted:
            self.side_effects_may_remain = bool(errors)
        return tuple(errors)

    def finish(
        self,
        *,
        status: str,
        rollback_status: RollbackStatus | None = None,
        side_effects_may_remain: Optional[bool] = None,
        warnings: Optional[tuple[str, ...]] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> KBOperationOutcome:
        """Finalize and return the immutable operation outcome."""
        if details:
            self.details.update(details)
        if warnings:
            self.warnings.extend(warnings)

        inferred_side_effects_may_remain = self._infer_side_effects_may_remain(status)
        if side_effects_may_remain is None:
            side_effects_may_remain = inferred_side_effects_may_remain
        elif status != "success":
            side_effects_may_remain = (
                side_effects_may_remain or inferred_side_effects_may_remain
            )

        if rollback_status is None:
            rollback_status = self.infer_rollback_status(
                status,
                side_effects_may_remain=side_effects_may_remain,
            )

        self.side_effects_may_remain = side_effects_may_remain
        self._outcome = KBOperationOutcome(
            operation_id=self.operation_id,
            operation_type=self.operation_type,
            collection=self.collection,
            status=status,
            rollback_status=rollback_status,
            persistence_policy=self.persistence_policy,
            compensation_steps=tuple(self.compensation_steps),
            child_outcomes=tuple(self.child_outcomes),
            warnings=tuple(self.warnings),
            side_effects_may_remain=side_effects_may_remain,
            details=dict(self.details),
        )
        return self._outcome

    def _infer_side_effects_may_remain(self, status: str) -> bool:
        if status == "success":
            return False
        return self.has_uncompensated_side_effects()

    def infer_rollback_status(
        self,
        status: str,
        *,
        side_effects_may_remain: bool,
    ) -> RollbackStatus:
        """Infer rollback status from current compensation and child state."""
        if status == "success":
            return RollbackStatus.NOT_NEEDED

        child_statuses = {child.status for child in self.child_outcomes}
        has_successful_child = "success" in child_statuses
        has_failed_child = any(
            child_status != "success" for child_status in child_statuses
        )
        if (
            self.persistence_policy is PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN
            and has_successful_child
            and has_failed_child
        ):
            return RollbackStatus.SKIPPED_BY_POLICY

        if side_effects_may_remain or self.has_uncompensated_side_effects():
            return RollbackStatus.INCOMPLETE
        if self._compensation_attempted and self.has_side_effects():
            return RollbackStatus.COMPLETE
        if any(
            child.rollback_status is RollbackStatus.COMPLETE
            for child in self.child_outcomes
        ):
            return RollbackStatus.COMPLETE
        if self.has_side_effects():
            return RollbackStatus.INCOMPLETE
        return RollbackStatus.NOT_NEEDED


_CURRENT_OPERATION: ContextVar[KBOperation | None] = ContextVar(
    "xagent_kb_current_operation",
    default=None,
)

_LAST_OUTCOME: ContextVar[KBOperationOutcome | None] = ContextVar(
    "xagent_kb_last_outcome",
    default=None,
)


def _format_exception_warning(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


# --- Consolidated per-plane registration helpers (#515, spec §3.2) ---
#
# Registration METADATA only: these helpers record compensation boundaries
# and idempotency keys. They define NO compensation bodies - per the #515
# §3.1.5 guardrail, PARSE/CHUNK/EMBEDDING/COLLECTION piggyback on the
# DOCUMENT compensation cascade at rollback time.


def step_metadata(completed_steps: Sequence[Any], name: str) -> dict[str, Any] | None:
    """Return one completed pipeline step's metadata dict (duck-typed)."""
    for step in completed_steps or ():
        if getattr(step, "name", None) == name:
            return dict(getattr(step, "metadata", None) or {})
    return None


def record_document_registration_side_effect(
    operation: KBOperation | None,
    *,
    collection: str,
    doc_id: str,
    created: Any,
    source_path: str,
    file_id: Optional[str],
    user_id: Optional[int],
) -> None:
    """Record the DOCUMENT + STATUS bookkeeping pair for one registered doc."""
    if operation is None:
        return
    operation.record_side_effect(
        name="remove_registered_document",
        plane=SideEffectPlane.DOCUMENT,
        payload={
            "collection": collection,
            "doc_id": doc_id,
            "created": created,
            "source_path": source_path,
            "file_id": file_id,
        },
        idempotency_key=f"document:{collection}:{doc_id}",
    )
    operation.record_side_effect(
        name="clear_ingestion_status",
        plane=SideEffectPlane.STATUS,
        payload={"collection": collection, "doc_id": doc_id, "user_id": user_id},
        idempotency_key=f"status:{collection}:{doc_id}",
    )


def record_parse_side_effect(
    operation: KBOperation | None,
    *,
    collection: str,
    doc_id: str,
    parse_hash: str,
) -> None:
    if operation is None:
        return
    operation.record_side_effect(
        name="remove_parse_record",
        plane=SideEffectPlane.PARSE,
        payload={
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
        },
        idempotency_key=f"parse:{collection}:{doc_id}:{parse_hash}",
    )


def record_chunk_side_effect(
    operation: KBOperation | None,
    *,
    collection: str,
    doc_id: str,
    parse_hash: str,
    chunk_count: int,
) -> None:
    if operation is None:
        return
    operation.record_side_effect(
        name="remove_chunk_records",
        plane=SideEffectPlane.CHUNK,
        payload={
            "collection": collection,
            "doc_id": doc_id,
            "parse_hash": parse_hash,
            "chunk_count": chunk_count,
        },
        idempotency_key=f"chunk:{collection}:{doc_id}:{parse_hash}",
    )


def record_document_ingestion_side_effects(
    operation: KBOperation | None,
    *,
    collection: str,
    doc_id: Optional[str],
    parse_hash: Optional[str],
    completed_steps: Sequence[Any],
    chunk_count: int,
    vector_count: int,
    failed_step: Optional[str],
    source_path: str,
    file_id: Optional[str],
    user_id: Optional[int],
    is_admin: Optional[bool],
) -> None:
    """Record every ingestion plane's bookkeeping from completed steps."""
    if operation is None or operation.outcome is not None:
        return

    operation.update_details(
        source_path=source_path,
        file_id=file_id,
        doc_id=doc_id,
        parse_hash=parse_hash,
        failed_step=failed_step,
        user_id=user_id,
        is_admin=is_admin,
    )

    initialize_step = step_metadata(completed_steps, "initialize_collection")
    if initialize_step is not None:
        operation.record_side_effect(
            name="restore_collection_initialization",
            plane=SideEffectPlane.COLLECTION,
            payload={
                "collection": collection,
                "embedding_model_id": initialize_step.get("embedding_model_id"),
            },
            idempotency_key=f"collection:{collection}:initialize",
        )

    register_step = step_metadata(completed_steps, "register_document")
    if doc_id and register_step is not None:
        record_document_registration_side_effect(
            operation,
            collection=collection,
            doc_id=doc_id,
            created=register_step.get("created"),
            source_path=source_path,
            file_id=file_id,
            user_id=user_id,
        )

    parse_step = step_metadata(completed_steps, "parse_document")
    if doc_id and parse_hash and parse_step is not None:
        if parse_step.get("written") is not False:
            record_parse_side_effect(
                operation,
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
            )

    chunk_step = step_metadata(completed_steps, "chunk_document")
    if doc_id and parse_hash and chunk_step is not None:
        if chunk_count > 0 and chunk_step.get("created") is not False:
            record_chunk_side_effect(
                operation,
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                chunk_count=chunk_count,
            )

    write_step = step_metadata(completed_steps, "write_vectors_to_db")
    if doc_id and vector_count > 0 and write_step is not None:
        operation.record_side_effect(
            name="remove_embedding_vectors",
            plane=SideEffectPlane.EMBEDDING,
            payload={
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "vector_count": vector_count,
            },
            idempotency_key=f"embedding:{collection}:{doc_id}:{parse_hash}",
        )


def should_finish_document_ingestion_operation(
    operation: KBOperation | None,
) -> bool:
    if operation is None:
        return False
    if operation.operation_type == "document_ingestion":
        return True
    return (
        operation.operation_type == "web_page_ingestion"
        and "url" not in operation.details
    )


def finish_ingestion_outcome(
    operation: KBOperation | None,
    *,
    status: str,
    message: Optional[str],
    treat_recorded_side_effects_as_remaining: bool = True,
) -> KBOperationOutcome | None:
    """Finish an ingestion operation with facade-compatible inference.

    ``treat_recorded_side_effects_as_remaining=True`` preserves the historical
    facade semantics (has_side_effects()-based) instead of the engine default
    (has_uncompensated_side_effects()) - the over-cautious direction. The
    precision switch is tracked in #795 (spec §6.3).
    """
    if operation is None or operation.outcome is not None:
        return None
    if status != "success":
        if treat_recorded_side_effects_as_remaining:
            side_effects_may_remain = operation.has_side_effects()
        else:
            side_effects_may_remain = operation.has_uncompensated_side_effects()
    else:
        side_effects_may_remain = False
    operation.finish(
        status=status,
        rollback_status=operation.infer_rollback_status(
            status,
            side_effects_may_remain=side_effects_may_remain,
        ),
        side_effects_may_remain=side_effects_may_remain,
        details={"message": message},
    )
    return operation.outcome


def finish_web_ingestion_outcome(
    operation: KBOperation | None,
    *,
    status: str,
    documents_created: int,
    pages_crawled: int,
    pages_failed: int,
    failed_urls: Mapping[str, str],
    message: str,
) -> KBOperationOutcome | None:
    if operation is None:
        return None
    if operation.outcome is not None:
        return operation.outcome

    child_side_effects_may_remain = any(
        child.side_effects_may_remain for child in operation.child_outcomes
    )
    own_side_effects_may_remain = bool(operation.uncompensated_steps())
    if status == "success":
        side_effects_may_remain = False
    else:
        side_effects_may_remain = (
            child_side_effects_may_remain or own_side_effects_may_remain
        )
    return operation.finish(
        status=status,
        rollback_status=operation.infer_rollback_status(
            status,
            side_effects_may_remain=side_effects_may_remain,
        ),
        side_effects_may_remain=side_effects_may_remain,
        details={
            "documents_created": documents_created,
            "pages_crawled": pages_crawled,
            "pages_failed": pages_failed,
            "failed_urls": dict(failed_urls),
            "message": message,
        },
    )


def finish_owned_operation(
    operation: KBOperation | None,
    owns_operation: bool,
    *,
    status: str,
) -> None:
    if operation is None or not owns_operation or operation.outcome is not None:
        return
    operation.finish(
        status=status,
        rollback_status=RollbackStatus.NOT_NEEDED,
        side_effects_may_remain=False,
    )


class KBOperationCompatibilityFacade:
    """Compatibility facade for rollback-aware coordinator operations.

    The model is intentionally internal: public pipeline/API schemas stay stable
    while compatibility facades can record operation outcomes and child side
    effects for future handle-level compensation.
    """

    @property
    def last_outcome(self) -> KBOperationOutcome | None:
        """Return the most recently finalized operation outcome."""
        return _LAST_OUTCOME.get()

    def current_operation(self) -> KBOperation | None:
        """Return the operation active in the current context, if any."""
        return _CURRENT_OPERATION.get()

    @contextmanager
    def start_operation(
        self,
        *,
        operation_type: str,
        collection: str,
        persistence_policy: PersistencePolicy = (
            PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN
        ),
        details: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[KBOperation]:
        """Start a root or nested operation and attach nested outcomes to parents."""
        parent_operation = _CURRENT_OPERATION.get()
        operation = KBOperation(
            operation_type=operation_type,
            collection=collection,
            persistence_policy=persistence_policy,
            parent_operation_id=(
                parent_operation.operation_id if parent_operation is not None else None
            ),
            details=details,
        )
        token = _CURRENT_OPERATION.set(operation)

        try:
            yield operation
        except BaseException as exc:  # noqa: BLE001 - record outcome before propagating
            if operation.outcome is None:
                operation.finish(
                    status="error",
                    side_effects_may_remain=operation.has_side_effects(),
                    warnings=(_format_exception_warning(exc),),
                )
            raise
        finally:
            if operation.outcome is None:
                operation.finish(status="success")

            _CURRENT_OPERATION.reset(token)
            outcome = operation.outcome
            if outcome is not None:
                if parent_operation is not None:
                    parent_operation.add_child_outcome(outcome)
                _LAST_OUTCOME.set(outcome)

    @contextmanager
    def start_child_operation(
        self,
        *,
        operation_type: str,
        collection: str,
        persistence_policy: PersistencePolicy = (
            PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN
        ),
        details: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[KBOperation]:
        """Start an operation intended to be attached to the current parent."""
        with self.start_operation(
            operation_type=operation_type,
            collection=collection,
            persistence_policy=persistence_policy,
            details=details,
        ) as operation:
            yield operation
