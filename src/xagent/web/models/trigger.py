from __future__ import annotations

import enum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
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
    GMAIL = "gmail"


class TriggerRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TriggerProvisioningStatus(str, enum.Enum):
    """Provider-side resource provisioning state for a trigger."""

    PENDING = "pending"
    ACTIVE = "active"
    FAILED = "failed"


class TriggerAuditOutcome(str, enum.Enum):
    """Durable outcome recorded for callback and payload-access activity."""

    ACCEPTED = "accepted"
    REJECTED_SIGNATURE = "rejected_signature"
    REJECTED_RESOURCE = "rejected_resource"
    REJECTED_DISABLED = "rejected_disabled"
    UNKNOWN_PROVIDER = "unknown_provider"
    UNKNOWN_CALLBACK = "unknown_callback"
    NOT_FOUND = "not_found"
    EXECUTION_FAILURE = "execution_failure"
    PAYLOAD_READ = "payload_read"
    RATE_LIMITED = "rate_limited"


class AgentTrigger(Base):  # type: ignore
    """Reusable automatic entry point for an agent or a workforce.

    Exactly one of ``agent_id`` / ``workforce_id`` is set (enforced by the
    service layer). Workforce triggers fire through ``create_workforce_run``
    and are never bound to the workforce's generated manager agent: a plain
    Task without a workforce run silently loses all delegation ability.
    """

    __tablename__ = "agent_triggers"
    __table_args__ = (
        # Exactly one owner. Mirrors the CHECK the migration adds to upgraded
        # databases so fresh installs (schema built from these models via
        # Base.metadata.create_all) enforce the same invariant. Keep the name
        # in sync with the migration so downgrade can drop it.
        CheckConstraint(
            "(agent_id IS NULL) <> (workforce_id IS NULL)",
            name="ck_agent_triggers_single_owner",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    workforce_id = Column(
        Integer,
        ForeignKey("workforces.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    type = Column(String(32), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True, server_default=true())
    config = Column(JSON, nullable=False, default=dict)
    prompt_template = Column(Text, nullable=True)

    webhook_token = Column(String(128), nullable=True, unique=True, index=True)
    secret_hash = Column(String(64), nullable=True)

    provider = Column(String(64), nullable=True, index=True)
    callback_id = Column(String(128), nullable=True, unique=True, index=True)
    resource_id = Column(String(255), nullable=True, index=True)
    secret_encrypted = Column(Text, nullable=True)
    provisioning_status = Column(String(32), nullable=True)
    provisioning_error = Column(Text, nullable=True)

    next_run_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    last_error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User", back_populates="agent_triggers")
    agent = relationship("Agent", back_populates="triggers")
    workforce = relationship("Workforce", back_populates="triggers")
    runs = relationship(
        "TriggerRun",
        back_populates="trigger",
        cascade="all, delete-orphan",
        order_by="TriggerRun.id.desc()",
    )
    # Default (non-delete) cascade nullifies audit trigger_id on trigger
    # deletion, preserving audit history even when SQLite runs without
    # foreign-key enforcement.
    audits = relationship("TriggerAudit", back_populates="trigger")

    def __repr__(self) -> str:
        return (
            f"<AgentTrigger(id={self.id}, agent_id={self.agent_id}, "
            f"workforce_id={self.workforce_id}, "
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


class TriggerAudit(Base):  # type: ignore
    """Durable audit record for trigger callback and payload access activity.

    Rows outlive the trigger they describe: trigger_id is nullable and set to
    NULL when the trigger row is deleted.
    """

    __tablename__ = "trigger_audits"

    id = Column(Integer, primary_key=True, index=True)
    trigger_id = Column(
        Integer,
        ForeignKey("agent_triggers.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    provider = Column(String(64), nullable=True, index=True)
    callback_id = Column(String(128), nullable=True, index=True)
    outcome = Column(String(64), nullable=False, index=True)
    detail: Any = Column(JSON, nullable=True)
    remote_ip = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    trigger = relationship("AgentTrigger", back_populates="audits")

    def __repr__(self) -> str:
        return (
            f"<TriggerAudit(id={self.id}, trigger_id={self.trigger_id}, "
            f"provider='{self.provider}', outcome='{self.outcome}')>"
        )
