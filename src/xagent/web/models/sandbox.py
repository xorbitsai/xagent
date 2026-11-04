"""
Sandbox database models.
"""

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

from .database import Base


class SandboxInfo(Base):  # type: ignore[no-any-unimported]
    """Database model for sandbox information."""

    __tablename__ = "sandbox_info"
    __table_args__ = (
        UniqueConstraint("name", "sandbox_type", name="uix_name_sandbox_type"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    sandbox_type = Column(
        String(50), nullable=False, index=True
    )  # boxlite, docker, etc.
    name = Column(String(255), nullable=False, index=True)
    state = Column(String(50), nullable=False)

    # Template stored as JSON
    template = Column(
        Text
    )  # JSON string: {"type": "...", "image": "...", "snapshot_id": "..."}

    # Config stored as JSON
    config = Column(
        Text
    )  # JSON string: {"cpus": ..., "memory": ..., "env": {...}, "volumes": [...], ...}

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class SandboxSnapshot(Base):  # type: ignore[no-any-unimported]
    """Database model for persisted sandbox snapshots."""

    __tablename__ = "sandbox_snapshot"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id", "sandbox_type", name="uix_snapshot_id_sandbox_type"
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    sandbox_type = Column(String(50), nullable=False, index=True)
    snapshot_id = Column(String(255), nullable=False, index=True)
    metadata_json = Column("metadata", Text, nullable=False)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class DurableSandboxLifecycle(Base):  # type: ignore[no-any-unimported]
    """Opaque, crash-recoverable ownership record for a physical sandbox.

    This table deliberately contains no raw actor, user, resource-owner, or
    backend connection identity.  ``scope_digest`` is the stable logical
    lookup key; ``lifecycle_token`` makes every physical generation unique;
    and ``owner_token`` plus ``version`` is the CAS fence for destructive
    work.
    """

    __tablename__ = "durable_sandbox_lifecycles"
    __table_args__ = (
        CheckConstraint("length(scope_digest) = 64", name="ck_dsl_scope_digest"),
        CheckConstraint(
            "active_scope_digest IS NULL OR length(active_scope_digest) = 64",
            name="ck_dsl_active_scope_digest",
        ),
        CheckConstraint("length(lifecycle_token) = 64", name="ck_dsl_lifecycle_token"),
        CheckConstraint(
            "length(backend_lifecycle_digest) = 64",
            name="ck_dsl_backend_lifecycle_digest",
        ),
        CheckConstraint("length(owner_token) = 64", name="ck_dsl_owner_token"),
        CheckConstraint(
            "length(create_operation_token) = 64",
            name="ck_dsl_create_operation_token",
        ),
        CheckConstraint(
            "turn_digest IS NULL OR length(turn_digest) = 64",
            name="ck_dsl_turn_digest",
        ),
        CheckConstraint("version >= 1", name="ck_dsl_version"),
        CheckConstraint("delete_attempts >= 0", name="ck_dsl_delete_attempts"),
        CheckConstraint(
            "state IN ('registered', 'ready', 'deleting')",
            name="ck_dsl_state",
        ),
        CheckConstraint(
            "create_phase IN ('not_started', 'may_publish', 'terminal', 'observed')",
            name="ck_dsl_create_phase",
        ),
        CheckConstraint(
            "(create_phase = 'terminal' AND create_terminal_outcome IS NOT NULL AND "
            "create_terminal_outcome IN ('success', 'terminal_absent')) OR "
            "(create_phase != 'terminal' AND create_terminal_outcome IS NULL)",
            name="ck_dsl_create_terminal_outcome",
        ),
        CheckConstraint(
            "((state = 'registered' OR state = 'ready') "
            "AND active_scope_digest IS NOT NULL "
            "AND active_scope_digest = scope_digest) OR "
            "(state = 'deleting' AND active_scope_digest IS NULL)",
            name="ck_dsl_active_generation",
        ),
        CheckConstraint(
            "(state = 'registered' AND ready_at IS NULL AND deleting_at IS NULL "
            "AND delete_claim_expires_at IS NULL) OR "
            "(state = 'ready' AND ready_at IS NOT NULL AND deleting_at IS NULL "
            "AND delete_claim_expires_at IS NULL) OR "
            "(state = 'deleting' AND deleting_at IS NOT NULL "
            "AND delete_claim_expires_at IS NOT NULL)",
            name="ck_dsl_state_shape",
        ),
        UniqueConstraint("active_scope_digest", name="uq_dsl_active_scope_digest"),
        UniqueConstraint("lifecycle_token", name="uq_dsl_lifecycle_token"),
        UniqueConstraint(
            "create_operation_token", name="uq_dsl_create_operation_token"
        ),
        UniqueConstraint(
            "backend_lifecycle_digest", name="uq_dsl_backend_lifecycle_digest"
        ),
        Index(
            "ix_dsl_reclaim",
            "state",
            "eligible_at",
            "retry_at",
            "delete_claim_expires_at",
            "id",
        ),
        Index(
            "ix_dsl_task_attempt",
            "task_id",
            "run_id",
            "lease_attempt_id",
        ),
        Index("ix_dsl_scope_digest", "scope_digest"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    scope_digest = Column(String(64), nullable=False)
    # Cross-dialect active-generation fence.  Active rows repeat their opaque
    # scope digest here; quarantined/deleting rows store NULL.  SQLite and
    # PostgreSQL both allow multiple NULLs under a unique constraint while
    # rejecting two active rows for the same scope.
    active_scope_digest = Column(String(64), nullable=True)
    lifecycle_token = Column(String(64), nullable=False)
    backend_lifecycle_digest = Column(String(64), nullable=False)
    owner_token = Column(String(64), nullable=False)
    create_operation_token = Column(String(64), nullable=False)
    create_phase = Column(
        String(16), nullable=False, default="not_started", server_default="not_started"
    )
    create_terminal_outcome = Column(String(16), nullable=True)
    version = Column(Integer, nullable=False, default=1, server_default="1")
    state = Column(String(16), nullable=False, default="registered")

    # Minimal opaque task-attempt fence.  No FK: the lifecycle must remain
    # discoverable after task deletion until backend cleanup is confirmed.
    task_id = Column(Integer, nullable=False)
    run_id = Column(String(64), nullable=False)
    lease_attempt_id = Column(String(64), nullable=False)
    turn_digest = Column(String(64), nullable=True)

    eligible_at = Column(DateTime(timezone=True), nullable=False)
    owner_lease_expires_at = Column(DateTime(timezone=True), nullable=False)
    delete_claim_expires_at = Column(DateTime(timezone=True), nullable=True)
    retry_at = Column(DateTime(timezone=True), nullable=True)
    registered_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ready_at = Column(DateTime(timezone=True), nullable=True)
    deleting_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    delete_attempts = Column(Integer, nullable=False, default=0, server_default="0")
