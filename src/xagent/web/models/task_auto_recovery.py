"""Durable bookkeeping for automatic recovery of interrupted tasks.

``task_auto_recovery`` holds one row per task describing the most recent
interruption of its run and where automatic recovery stands. It is kept apart
from ``tasks`` so that no existing writer of ``tasks`` has to maintain
scheduling columns: the row is fenced by ``paused_state_version`` instead. A
task without a row simply does not take part in automatic recovery.

The counters and ``episode_started_at`` are scoped to ``run_id``: a writer
that reuses the row for a new run must reset ``no_progress_resumes``,
``total_resumes`` and ``episode_started_at`` together (and clear
``last_command_id``), so a non-zero counter always belongs to the run named
on the row.

``task_recovery_events`` is its append-only history, used for timelines,
audit and test assertions. Both tables cascade with their task.

``reason`` holds an ``xagent.core.agent.interruption.InterruptionReason``
value; ``state`` and ``event`` hold :class:`TaskAutoRecoveryState` and
:class:`TaskRecoveryEventType` values. The columns stay plain strings so
adding a value needs no schema change.
"""

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .database import Base

# Upper bound on ``TaskAutoRecovery.last_error``; writers truncate to it.
TASK_AUTO_RECOVERY_LAST_ERROR_MAX_CHARS = 2048


class TaskAutoRecoveryState(str, enum.Enum):
    """Where automatic recovery of the row's run stands."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"
    DISPATCHED = "dispatched"
    EXHAUSTED = "exhausted"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    INELIGIBLE = "ineligible"
    DISABLED = "disabled"
    STALE = "stale"
    DISPATCH_FAILED = "dispatch_failed"


class TaskRecoveryEventType(str, enum.Enum):
    """Kinds of entries in ``task_recovery_events``."""

    INTERRUPTED = "interrupted"
    SCHEDULED = "scheduled"
    AUTO_RESUMED = "auto_resumed"
    EXHAUSTED = "exhausted"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    INELIGIBLE = "ineligible"
    SKIPPED_USER_INTENT = "skipped_user_intent"
    DISPATCH_FAILED = "dispatch_failed"
    STALE = "stale"
    # A scheduled or dispatched resume was dropped because
    # XAGENT_TASK_AUTO_RESUME_ENABLED is off; the row rests in ``manual``.
    SKIPPED_DISABLED = "skipped_disabled"


class TaskAutoRecovery(Base):  # type: ignore
    __tablename__ = "task_auto_recovery"
    __table_args__ = (
        # The sweeper's scan: scheduled rows whose next attempt is due.
        Index("ix_task_auto_recovery_due", "state", "next_attempt_at"),
    )

    task_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tasks.id", ondelete="CASCADE"), primary_key=True
    )
    # The run this row describes.
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # Most recent interruption reason (an InterruptionReason value).
    reason: Mapped[str] = mapped_column(String(32), nullable=False)
    # A TaskAutoRecoveryState value.
    state: Mapped[str] = mapped_column(String(24), nullable=False)
    # e.g. ``ineligible:source_sdk``.
    state_detail: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # ``tasks.state_version`` after the settling write; for a PAUSED row it
    # fences dispatch (a FAILED row's value is recorded but never read).
    paused_state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # Start of the staleness window.
    interrupted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Start of the current run of interruptions without progress.
    episode_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Automatic resumes dispatched since the last observed progress.
    no_progress_resumes: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0", default=0
    )
    # Automatic resumes dispatched for this run, regardless of progress.
    total_resumes: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0", default=0
    )
    # Fingerprint of the recoverable checkpoint at interruption time.
    progress_marker: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # When ``state == 'scheduled'``: the time the attempt falls due. For
    # ``dispatched``: when the dispatch is next checked; NULL once confirmed.
    next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # ``command_id`` of the most recently dispatched RESUME command.
    last_command_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Operator-only diagnostic, at most TASK_AUTO_RECOVERY_LAST_ERROR_MAX_CHARS
    # (writers truncate); never sent to clients.
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class TaskRecoveryEvent(Base):  # type: ignore
    __tablename__ = "task_recovery_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # A TaskRecoveryEventType value.
    event: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    attempt: Mapped[int | None] = mapped_column(Integer, nullable=True)
    next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Client-safe fields only.
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
