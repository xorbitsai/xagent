"""Persistent execution budgets and immutable command admission scopes."""

from sqlalchemy import Column, ForeignKey, Index, Integer, String

from .database import Base


class TaskAdmissionBucket(Base):  # type: ignore
    __tablename__ = "task_admission_buckets"

    key = Column(String(255), primary_key=True)
    capacity = Column(Integer, nullable=False)
    max_pending = Column(Integer, nullable=False)


class TaskAdmissionTicket(Base):  # type: ignore
    __tablename__ = "task_admission_tickets"
    __table_args__ = (
        Index("ix_task_admission_bucket_command", "bucket_key", "command_id"),
    )

    command_id = Column(
        Integer,
        ForeignKey("task_execution_commands.id", ondelete="CASCADE"),
        primary_key=True,
    )
    bucket_key = Column(
        String(255), ForeignKey("task_admission_buckets.key"), nullable=False
    )
    task_id = Column(
        Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    runner_id = Column(String(255), nullable=True)
    owner_attempt_id = Column(String(64), nullable=True)
