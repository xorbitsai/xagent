"""Keep reclaimed file identity after compensation removes its upload row."""

from sqlalchemy import Column, DateTime, String
from sqlalchemy.sql import func

from .database import Base


class UploadedFileCleanupFence(Base):  # type: ignore
    __tablename__ = "uploaded_file_cleanup_fences"

    # No FK: this identity must survive upload settlement and user cascades.
    file_id = Column(String(36), primary_key=True)
    claimed_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
