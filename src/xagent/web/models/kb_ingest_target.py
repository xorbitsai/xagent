import uuid

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class KBIngestTarget(Base):  # type: ignore
    """Latest accepted background-ingest generation for one KB file target."""

    __tablename__ = "kb_ingest_targets"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "collection",
            "target_path",
            name="uq_kb_ingest_targets_user_collection_path",
        ),
        Index("ix_kb_ingest_targets_user_collection", "user_id", "collection"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    collection = Column(String(255), nullable=False)
    target_path = Column(String(2048), nullable=False)
    file_id = Column(String(36), nullable=False)
    latest_generation_id = Column(
        String(36), nullable=False, default=lambda: str(uuid.uuid4())
    )
    latest_job_id = Column(
        String(36), ForeignKey("background_jobs.id", ondelete="SET NULL"), nullable=True
    )
    latest_file_sha256 = Column(String(64), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User")
    latest_job = relationship("BackgroundJob")

    def __repr__(self) -> str:
        return (
            "<KBIngestTarget("
            f"user_id={self.user_id}, collection={self.collection!r}, "
            f"target_path={self.target_path!r}, generation={self.latest_generation_id}"
            ")>"
        )
