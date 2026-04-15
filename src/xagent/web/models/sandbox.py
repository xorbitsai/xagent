"""
Sandbox database models.
"""

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint, func

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
