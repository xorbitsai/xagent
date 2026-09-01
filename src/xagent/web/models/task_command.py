"""Durable control-plane commands for task execution."""

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .database import Base


class TaskExecutionCommand(Base):  # type: ignore
    """One idempotent task command, claimed by at most one worker at a time."""

    __tablename__ = "task_execution_commands"
    __table_args__ = (
        UniqueConstraint("task_id", "command_id", name="uq_task_command_identity"),
        Index("ix_task_commands_status_created", "status", "created_at"),
        Index("ix_task_commands_task_order", "task_id", "id"),
    )

    id = Column(Integer, primary_key=True)
    task_id = Column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
    )
    actor_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Immutable acceptance-time pseudonym. ``actor_user_id`` remains the live
    # relational join and may be NULLed when the account is deleted.
    actor_subject = Column(String(64), nullable=True)
    # Immutable task-owner correlation captured at command acceptance. Legacy
    # rows stay NULL because a current numeric owner cannot prove historical
    # ownership after account deletion and SQLite id reuse.
    task_owner_user_id = Column(Integer, nullable=True)
    task_owner_subject = Column(String(64), nullable=True)
    command_id = Column(String(64), nullable=False)
    kind = Column(String(32), nullable=False)
    payload = Column(JSON, nullable=False)

    # The run/worker observed when the command was accepted. Commands aimed at
    # a live run stay with its lease owner; once that lease expires another
    # worker may recover them from the durable inbox.
    target_run_id = Column(String(64), nullable=True)
    # Legacy rows created before this snapshot existed remain NULL: treating
    # an unknown historical version as a real version 0 would create a false
    # run-correlation tuple during recovery.
    target_state_version = Column(Integer, nullable=True)
    target_runner_id = Column(String(255), nullable=True)

    status = Column(
        String(32), nullable=False, default="pending", server_default="pending"
    )
    claimed_by = Column(String(255), nullable=True)
    claim_expires_at = Column(DateTime(timezone=True), nullable=True)
    attempt_count = Column(Integer, nullable=False, default=0, server_default="0")
    failure_count = Column(Integer, nullable=False, default=0, server_default="0")
    defer_count = Column(Integer, nullable=False, default=0, server_default="0")
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
