"""Optional durable startup pacing for an execution-admission bucket."""

from sqlalchemy import Column, Float, ForeignKey, Integer, String

from .database import Base


class TaskAdmissionPacing(Base):  # type: ignore
    __tablename__ = "task_admission_pacing"

    bucket_key = Column(
        String(255),
        ForeignKey("task_admission_buckets.key", ondelete="CASCADE"),
        primary_key=True,
    )
    interval_seconds = Column(Float, nullable=False)
    burst = Column(Integer, nullable=False)
    lane = Column(String(16), nullable=False)
    next_start_at = Column(Float, nullable=False, default=0.0, server_default="0")
