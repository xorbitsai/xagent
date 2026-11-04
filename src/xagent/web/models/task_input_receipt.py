"""Immutable input acceptance identity, independent of task retention."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .database import Base


class TaskInputReceipt(Base):  # type: ignore
    __tablename__ = "task_input_receipts"

    # SHA-256 of the canonical, source-scoped input identity, not its contents.
    identity_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Deletion keeps the identity as a tombstone rather than permitting replay.
    task_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    command_db_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("task_execution_commands.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
