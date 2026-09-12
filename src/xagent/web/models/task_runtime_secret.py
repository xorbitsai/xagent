"""Encrypted, run-scoped connector values, never part of task serialization."""

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .database import Base


class TaskRuntimeSecret(Base):  # type: ignore
    __tablename__ = "task_runtime_secrets"
    __table_args__ = (
        UniqueConstraint("task_id", "turn_id", name="uq_task_runtime_secret_turn"),
    )

    id = Column(Integer, primary_key=True)
    task_id = Column(
        Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    turn_id = Column(String(64), nullable=False)
    run_id = Column(String(64), nullable=True)
    owner_subject = Column(String(64), nullable=False)
    ciphertext = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
