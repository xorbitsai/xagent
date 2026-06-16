from __future__ import annotations

import enum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    true,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class TriggerType(str, enum.Enum):
    WEBHOOK = "webhook"
    SCHEDULED = "scheduled"


class TriggerRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentTrigger(Base):  # type: ignore
    """Reusable automatic entry point for an agent."""

    __tablename__ = "agent_triggers"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type = Column(String(32), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True, server_default=true())
    config = Column(JSON, nullable=False, default=dict)
    prompt_template = Column(Text, nullable=True)

    webhook_token = Column(String(128), nullable=True, unique=True, index=True)
    secret_hash = Column(String(64), nullable=True)

    next_run_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    last_error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User", back_populates="agent_triggers")
    agent = relationship("Agent", back_populates="triggers")
    runs = relationship(
        "TriggerRun",
        back_populates="trigger",
        cascade="all, delete-orphan",
        order_by="TriggerRun.id.desc()",
    )

    def __repr__(self) -> str:
        return (
            f"<AgentTrigger(id={self.id}, agent_id={self.agent_id}, "
            f"type='{self.type}', enabled={self.enabled})>"
        )


class TriggerRun(Base):  # type: ignore
    """One accepted trigger event and its generated task, if any."""

    __tablename__ = "trigger_runs"

    id = Column(Integer, primary_key=True, index=True)
    trigger_id = Column(
        Integer,
        ForeignKey("agent_triggers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_id = Column(
        Integer, ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    background_job_id = Column(
        String(36),
        ForeignKey("background_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status = Column(
        String(32),
        nullable=False,
        default=TriggerRunStatus.PENDING.value,
        index=True,
    )
    source_event_id = Column(String(255), nullable=True, index=True)
    payload_snapshot: Any = Column(JSON, nullable=True)
    idempotency_key = Column(String(255), nullable=False, unique=True, index=True)
    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    trigger = relationship("AgentTrigger", back_populates="runs")
    task = relationship("Task")
    background_job = relationship("BackgroundJob")

    def __repr__(self) -> str:
        return (
            f"<TriggerRun(id={self.id}, trigger_id={self.trigger_id}, "
            f"status='{self.status}', task_id={self.task_id})>"
        )
