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
    # passive_deletes=True on all four: their child FK is a real
    # ON DELETE CASCADE (workforce_agents.workforce_id, workforce_runs.
    # workforce_id, workforce_builder_messages.workforce_id, agent_triggers.
    # workforce_id) and this project enables SQLite foreign-key enforcement
    # per connection (db/sqlite.py's PRAGMA foreign_keys=ON), so deleting a
    # Workforce can rely on the database to cascade-delete these rows
    # instead of the ORM loading and deleting every child row in Python one
    # at a time.
    workers = relationship(
        "WorkforceAgent",
        back_populates="workforce",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    runs = relationship(
        "WorkforceRun",
        back_populates="workforce",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    builder_messages = relationship(
        "WorkforceBuilderMessage",
        back_populates="workforce",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    triggers = relationship(
        "AgentTrigger",
        back_populates="workforce",
        cascade="all, delete-orphan",
        passive_deletes=True,
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
    # Nullable: ephemeral preview runs (test-before-save in the workforce
    # builder) have a manager + inline worker configs but no persisted
    # Workforce row to point at.
    workforce_id = Column(
        Integer,
        ForeignKey("workforces.id", ondelete="CASCADE"),
        nullable=True,
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
    # Bumped on every sync_workforce_run_status call (workforce_runtime.py)
    # that actually changes this row, i.e. once per turn of an active
    # conversation. created_at alone can't tell a genuinely-abandoned preview
    # run from one that's mid-conversation but has simply been open a long
    # time -- the preview-run reaper keys staleness off this column instead
    # (PR review round 8, F-NEW-1: reaping off created_at could permanently
    # cancel an actively-used preview session).
    #
    # The precise "only on a real transition" semantics above come from
    # sync_workforce_run_status's own `changed` check, which sets this
    # column explicitly -- NOT from the onupdate=func.now() below, which is
    # a blunter instrument: it fires on *any* UPDATE statement that touches
    # this row, transition or not. That only lines up with the comment above
    # today because sync_workforce_run_status is the only code path that
    # updates WorkforceRun rows; it would stop lining up the moment another
    # write path is added without the same guard.
    last_activity_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

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
