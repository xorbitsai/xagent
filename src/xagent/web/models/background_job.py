import enum
import uuid
from typing import Any

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class BackgroundJobStatus(str, enum.Enum):
    PENDING = "pending"
    ENQUEUED = "enqueued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundJobType(str, enum.Enum):
    KB_INGEST_DOCUMENT = "kb.ingest.document"
    KB_INGEST_WEB = "kb.ingest.web"
    TRIGGER_EVENT = "trigger.event"
    TRIGGER_SCAN = "trigger.scan"


class BackgroundJob(Base):  # type: ignore
    """Durable state for Celery-backed background work.

    Redis/Celery owns delivery and short-lived worker coordination. This table is
    the source of truth exposed to users and is safe to recover after Redis loss.
    """

    __tablename__ = "background_jobs"

    id = Column(
        String(36),
        primary_key=True,
        index=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    job_type = Column(String(100), nullable=False, index=True)
    queue = Column(String(64), nullable=False, default="default", index=True)
    status = Column(
        String(32),
        nullable=False,
        default=BackgroundJobStatus.PENDING.value,
        index=True,
    )
    payload = Column(JSON, nullable=False, default=dict)
    progress = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    idempotency_key = Column(String(255), nullable=True, unique=True)
    celery_task_id = Column(String(255), nullable=True, index=True)
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User", back_populates="background_jobs")

    @property
    def is_terminal(self) -> bool:
        return str(self.status) in {
            BackgroundJobStatus.SUCCEEDED.value,
            BackgroundJobStatus.FAILED.value,
            BackgroundJobStatus.CANCELLED.value,
        }

    def __repr__(self) -> str:
        return (
            f"<BackgroundJob(id={self.id}, type={self.job_type}, status={self.status})>"
        )


BackgroundJobPayload = dict[str, Any]
