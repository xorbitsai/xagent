import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base

DETACHED_REASON_TASK_DELETED = "task_deleted"
DETACHED_REASON_TASK_CREATE_FAILED = "task_create_failed"
DETACHED_REASONS = (
    DETACHED_REASON_TASK_DELETED,
    DETACHED_REASON_TASK_CREATE_FAILED,
)


class UploadedFile(Base):  # type: ignore
    __tablename__ = "uploaded_files"
    __table_args__ = (
        Index(
            "ix_uploaded_files_status_updated_at_id",
            "storage_status",
            "updated_at",
            "id",
        ),
        # Serves the orphan-GC sweep predicate (#973). Declared here as well
        # as in the migration because fresh installations stamp Alembic head
        # BEFORE Base.metadata.create_all() — the migration never runs there,
        # so create_all() must produce the index itself.
        Index(
            "ix_uploaded_files_orphan_gc",
            "upload_source",
            "task_id",
            "created_at",
        ),
        Index(
            "ix_uploaded_files_detached_gc",
            "task_id",
            "storage_status",
            "detached_at",
            "id",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(
        String(36),
        unique=True,
        index=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    task_id = Column(
        Integer, ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True
    )
    # Index is created by migration 20260410_add_index_on_uploaded_files_filename.py
    # to ensure existing databases have the index for URL deduplication queries.
    filename = Column(String(512), nullable=False)
    storage_path = Column(String(2048), nullable=False, unique=True)
    storage_backend = Column(String(64), nullable=True)
    storage_key = Column(String(2048), nullable=True)
    storage_uri = Column(String(4096), nullable=True)
    checksum = Column(String(128), nullable=True)
    etag = Column(String(255), nullable=True)
    workspace_relative_path = Column(String(2048), nullable=True)
    workspace_category = Column(String(64), nullable=True)
    # Provenance marker for uploads created before any task/owner binding
    # exists (currently the task-less public-share path, #973). NULL for the
    # overwhelming majority of rows (task-bound / logged-in draft uploads);
    # only orphan GC keys off this so a coarse "task_id IS NULL" sweep can't
    # reap a logged-in user's un-sent draft attachments.
    upload_source = Column(String(64), nullable=True)
    detached_reason = Column(String(64), nullable=True)
    detached_at = Column(DateTime(timezone=True), nullable=True)
    storage_status = Column(String(32), nullable=False, default="legacy")
    mime_type = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="uploaded_files")
    task = relationship("Task", back_populates="uploaded_files")

    def __repr__(self) -> str:
        return f"<UploadedFile(file_id={self.file_id}, filename='{self.filename}', user_id={self.user_id})>"


def uploaded_file_bind_values(task_id: int) -> dict[Any, Any]:
    """Return one atomic transition from detached/unbound to task-bound."""

    return {
        UploadedFile.task_id: task_id,
        UploadedFile.detached_reason: None,
        UploadedFile.detached_at: None,
    }


def uploaded_file_detach_values(
    *, reason: str, detached_at: datetime
) -> dict[Any, Any]:
    """Return one validated task-detach transition."""

    if reason not in DETACHED_REASONS:
        raise ValueError(f"Unsupported uploaded-file detach reason: {reason}")
    return {
        UploadedFile.task_id: None,
        UploadedFile.detached_reason: reason,
        UploadedFile.detached_at: detached_at,
    }
