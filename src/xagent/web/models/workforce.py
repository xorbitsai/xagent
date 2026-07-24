from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class Workforce(Base):  # type: ignore[no-any-unimported]
    __tablename__ = "workforces"
    __table_args__ = (
        UniqueConstraint(
            "scope_type", "scope_id", "name", name="uq_workforce_scope_name"
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    owner_user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scope_type = Column(String(50), nullable=False, default="user", index=True)
    scope_id = Column(String(200), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    manager_agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    status = Column(String(20), nullable=False, default="draft", index=True)
    canvas_layout = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    owner = relationship("User", foreign_keys=[owner_user_id])
    manager_agent = relationship("Agent", foreign_keys=[manager_agent_id])
    workers = relationship(
        "WorkforceAgent",
        back_populates="workforce",
        cascade="all, delete-orphan",
    )
    runs = relationship(
        "WorkforceRun",
        back_populates="workforce",
        cascade="all, delete-orphan",
    )
    builder_messages = relationship(
        "WorkforceBuilderMessage",
        back_populates="workforce",
        cascade="all, delete-orphan",
    )
    triggers = relationship(
        "AgentTrigger",
        back_populates="workforce",
        cascade="all, delete-orphan",
    )


class WorkforceAgent(Base):  # type: ignore[no-any-unimported]
    __tablename__ = "workforce_agents"
    __table_args__ = (
        UniqueConstraint("workforce_id", "agent_id", name="uq_workforce_agent"),
    )

    id = Column(Integer, primary_key=True, index=True)
    workforce_id = Column(
        Integer,
        ForeignKey("workforces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    alias = Column(String(200), nullable=True)
    assignment_instructions = Column(Text, nullable=False)
    source_type = Column(String(20), nullable=False, default="existing")
    template_id = Column(String(200), nullable=True)
    enabled = Column(Boolean, nullable=False, default=True)
    sort_order = Column(Integer, nullable=False, default=0)
    canvas_position = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    workforce = relationship("Workforce", back_populates="workers")
    agent = relationship("Agent", foreign_keys=[agent_id])


class WorkforceRun(Base):  # type: ignore[no-any-unimported]
    __tablename__ = "workforce_runs"
    __table_args__ = (
        # Unique index (not constraint) so SQLite can gain it without a table
        # rebuild; NULL keys are exempt on both SQLite and PostgreSQL, so only
        # callers that opt into idempotency pay the dedup guarantee.
        Index(
            "uq_workforce_run_idempotency",
            "workforce_id",
            "idempotency_key",
            unique=True,
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    workforce_id = Column(
        Integer,
        ForeignKey("workforces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_id = Column(
        Integer, ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, unique=True
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status = Column(String(20), nullable=False, default="pending", index=True)
    is_preview = Column(Boolean, nullable=False, default=False, server_default="0")
    # Caller-supplied dedup token for external channels (webhook retries,
    # network-retried API calls). Unique per workforce when set; see
    # ``uq_workforce_run_idempotency``.
    idempotency_key = Column(String(128), nullable=True)
    snapshot = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    workforce = relationship("Workforce", back_populates="runs")
    task = relationship("Task")
    user = relationship("User", foreign_keys=[user_id])


class WorkforceBuilderMessage(Base):  # type: ignore[no-any-unimported]
    __tablename__ = "workforce_builder_messages"

    id = Column(Integer, primary_key=True, index=True)
    workforce_id = Column(
        Integer,
        ForeignKey("workforces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    proposed_patch = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False, default="message")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    workforce = relationship("Workforce", back_populates="builder_messages")
    user = relationship("User", foreign_keys=[user_id])
