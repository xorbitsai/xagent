"""Authoritative conversation/execution facts for version-two tasks."""

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from .database import Base


class TaskExecutionEvent(Base):  # type: ignore
    __tablename__ = "task_execution_events"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_task_execution_events_event_id"),
        UniqueConstraint(
            "task_id", "sequence", name="uq_task_execution_events_sequence"
        ),
        UniqueConstraint(
            "task_id",
            "scope_id",
            "idempotency_key",
            name="uq_task_execution_events_idempotency",
        ),
        CheckConstraint("sequence > 0", name="ck_task_execution_events_sequence"),
        CheckConstraint(
            "payload_version > 0", name="ck_task_execution_events_payload_version"
        ),
        CheckConstraint(
            "scope_id <> '' AND idempotency_key <> '' AND kind <> ''",
            name="ck_task_execution_events_identity",
        ),
        Index(
            "ix_task_execution_events_scope_cursor", "task_id", "scope_id", "sequence"
        ),
    )

    id = Column(Integer, primary_key=True)
    event_id = Column(String(36), nullable=False)
    task_id = Column(
        Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    sequence = Column(BigInteger, nullable=False)
    # "root" for the manager; child producers supply their stable scope id.
    scope_id = Column(String(255), nullable=False)
    run_id = Column(String(64), nullable=True)
    turn_id = Column(String(64), nullable=True)
    # One assistant response may declare multiple tool calls. Keep the batch
    # identity separate from each retryable tool attempt's identity.
    assistant_message_id = Column(String(255), nullable=True)
    tool_attempt_id = Column(String(255), nullable=True)
    idempotency_key = Column(String(255), nullable=False)
    kind = Column(String(64), nullable=False)
    payload_version = Column(Integer, nullable=False)
    payload = Column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)
    occurred_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
