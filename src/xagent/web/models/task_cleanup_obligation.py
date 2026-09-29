"""External cleanup a task deletion still owes after its rows are gone (#2587).

A task deletion commits its rows first and releases external resources (the
workspace directory, runtime-extension state) afterwards, because the rows are
what the resources are found from and a database rollback cannot restore a
directory that was already removed. That leaves a window where the rows are
gone and the resource is not. This table is the durable record of that window:
one row per resource, written in the same transaction as the row deletion and
deleted once the resource is released.

It deliberately has no foreign key to ``tasks`` or ``users``. The whole point
of a row is to outlive both, and its ``locator`` is self-sufficient for the
same reason: nothing about the task can be read back once the obligation is
the only thing left of it.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .database import Base


class TaskCleanupObligation(Base):  # type: ignore
    __tablename__ = "task_cleanup_obligations"
    __table_args__ = (
        # Deliberately no uniqueness on (task_id, kind, key): SQLite hands a
        # deleted task's id to the next task, so an old obligation and a new
        # one can name the same id, and each deletion must keep its own.
        # The retry driver's scan: due pending rows, oldest due first.
        Index(
            "ix_task_cleanup_obligations_status_due",
            "status",
            "next_attempt_at",
        ),
        # Every outcome is fenced on (id, attempts). Without AUTOINCREMENT,
        # SQLite hands a discharged obligation's id to the next insert, which
        # starts at attempts=0 -- and a late outcome for the old row would
        # then match the new one. PostgreSQL sequences never reuse an id.
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    owner_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resource_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Distinguishes several resources of one kind -- the extension name for
    #: ``runtime_extension``; empty for the single workspace a task owns.
    resource_key: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default=""
    )
    locator: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0", default=0
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
