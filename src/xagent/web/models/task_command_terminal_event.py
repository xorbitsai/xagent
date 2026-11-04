"""Durable client-facing outcomes for terminal task commands."""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import false, func, true

from .database import Base


class TaskCommandTerminalEvent(Base):  # type: ignore
    """One append-only terminal outcome, replayable by every web worker."""

    __tablename__ = "task_command_terminal_events"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_task_command_terminal_event_id"),
        UniqueConstraint(
            "task_command_id",
            "outcome_version",
            name="uq_task_command_terminal_outcome_version",
        ),
        Index("ix_task_command_terminal_events_task_cursor", "task_id", "id"),
    )

    id = Column(Integer, primary_key=True)
    event_id = Column(String(36), nullable=False)
    task_command_id = Column(
        Integer,
        ForeignKey("task_execution_commands.id", ondelete="CASCADE"),
        nullable=False,
    )
    task_id = Column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
    )
    task_run_id = Column(String(64), nullable=True)
    task_state_version = Column(Integer, nullable=True)
    command_id = Column(String(64), nullable=False)
    command_kind = Column(String(32), nullable=False)
    # Historical numeric hints and stable opaque subjects are deliberately not
    # relational joins. Numeric ids never drive identity or authorization
    # because SQLite may reuse them after account deletion.
    actor_user_id = Column(Integer, nullable=True)
    actor_subject = Column(String(64), nullable=True)
    task_owner_user_id = Column(Integer, nullable=False)
    task_owner_subject = Column(String(64), nullable=True)
    outcome_version = Column(Integer, nullable=False)
    outcome = Column(String(32), nullable=False)
    message_code = Column(String(64), nullable=True)
    resend_safe = Column(Boolean, nullable=False, default=False, server_default=false())
    include_command_identity = Column(
        Boolean,
        nullable=False,
        default=True,
        server_default=true(),
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
