import uuid
from pathlib import Path

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ...config import get_uploads_dir
from .database import Base


class UploadedFile(Base):  # type: ignore
    __tablename__ = "uploaded_files"

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
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=True)
    # Index is created by migration 20260410_add_index_on_uploaded_files_filename.py
    # to ensure existing databases have the index for URL deduplication queries.
    filename = Column(String(512), nullable=False)
    storage_path = Column(String(2048), nullable=False, unique=True)
    mime_type = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="uploaded_files")
    task = relationship("Task", back_populates="uploaded_files")

    @property
    def absolute_path(self) -> Path:
        """Get absolute file path.

        Handles both relative and absolute paths in storage_path for backward compatibility.
        - If storage_path is absolute, returns as-is (old data)
        - If storage_path is relative, resolves against UPLOADS_DIR/user_{user_id}

        Returns:
            Absolute Path to the file
        """

        stored = Path(self.storage_path)  # pyright: ignore[reportArgumentType]

        # If already absolute (old data), return as-is
        if stored.is_absolute():
            return stored

        # If relative (new data), resolve to absolute
        user_root = get_uploads_dir() / f"user_{self.user_id}"
        return (user_root / self.storage_path).resolve()

    def __repr__(self) -> str:
        return f"<UploadedFile(file_id={self.file_id}, filename='{self.filename}', user_id={self.user_id})>"
