"""Durable ``Idempotency-Key`` records for the ``/v1`` create and append calls.

One row per ``(scope, operation, key)``. The row is reserved before the call
runs (``in_progress``), is bound to the accepted task inside the same
transaction that creates or claims the turn, and is completed with the response
that was returned, so a retry on any process, or after a restart, replays that
response instead of starting a second run.

``request_fingerprint`` is a SHA-256 digest of the canonical request body, never
the body itself: the body can carry connector secrets and the row outlives the
request.
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .database import Base

IDEMPOTENCY_STATE_IN_PROGRESS = "in_progress"
IDEMPOTENCY_STATE_COMPLETED = "completed"


class ApiIdempotencyKey(Base):  # type: ignore
    __tablename__ = "api_idempotency_keys"
    __table_args__ = (
        UniqueConstraint(
            "scope_kind",
            "scope_id",
            "operation",
            "idempotency_key",
            name="uq_api_idempotency_scope_operation_key",
        ),
        Index("ix_api_idempotency_keys_expires_at", "expires_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Whose key it is: ``agent`` / ``workforce`` (a runtime key's bound
    # resource) or ``user`` (a personal key's holder). Rotating a key within the
    # same scope keeps its retries deduplicated.
    scope_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    scope_id: Mapped[int] = mapped_column(Integer, nullable=False)
    # ``create_task`` or ``append_message:<task_id>``.
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    # Until this instant a retry of an ``in_progress`` row is answered 409;
    # afterwards the reserving request is presumed dead.
    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # After this instant the row is dead and its key may be reused.
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # SET NULL: a deleted task leaves the key unreplayable, not reusable.
    task_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True
    )
    response_body: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    response_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
