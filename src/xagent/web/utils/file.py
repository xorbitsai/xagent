"""File path utilities for uploaded files.

This module provides path conversion functions for the uploaded_files table.
The table stores relative paths for portability, with backward compatibility
for existing absolute path records.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from sqlalchemy.orm import Session

from ...config import get_uploads_dir
from ..models.uploaded_file import UploadedFile

if TYPE_CHECKING:
    from _typeshed import StrPath


def to_relative_path(absolute_path: StrPath, user_id: Optional[int] = None) -> str:
    """Convert absolute path to relative path (POSIX format).

    Args:
        absolute_path: The absolute file path to convert
        user_id: Optional user ID. If provided, path is relative to user directory.

    Returns:
        Relative path string with POSIX separators (/)

    Raises:
        ValueError: If path is not within UPLOADS_DIR (caught by caller)
    """
    absolute_path = Path(absolute_path)
    try:
        if user_id:
            base = get_uploads_dir() / f"user_{user_id}"
        else:
            base = get_uploads_dir()
        return absolute_path.relative_to(base).as_posix()
    except ValueError:
        # Path is outside uploads_dir - return as-is for external directories
        return absolute_path.as_posix()


def to_absolute_path(relative_path: StrPath, user_id: Optional[int] = None) -> Path:
    """Convert relative path to absolute path.

    Handles both relative and absolute paths in input:
    - If input is absolute, returns as-is (for backward compatibility)
    - If input is relative, resolves against UPLOADS_DIR

    Args:
        relative_path: The path to convert (can be absolute or relative)
        user_id: Optional user ID. If provided, path is relative to user directory.

    Returns:
        Resolved absolute Path
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path

    if user_id:
        return (get_uploads_dir() / f"user_{user_id}" / path).resolve()
    return (get_uploads_dir() / path).resolve()


def find_file_by_path(
    db: Session, file_path: StrPath, user_id: int
) -> Optional[UploadedFile]:
    """Find file record by path, handles both absolute and relative storage formats.

    Database may contain:
    - Old records: absolute paths like '/root/.../uploads/user_1/web_task_29/output/file.txt'
    - New records: relative paths like 'web_task_29/output/file.txt'

    Args:
        db: Database session
        file_path: Absolute file path to search for
        user_id: User ID for relative path conversion

    Returns:
        UploadedFile record or None
    """
    file_path = Path(file_path)

    # Try exact match first (handles old data with absolute paths)
    record = (
        db.query(UploadedFile)
        .filter(UploadedFile.storage_path == str(file_path))
        .first()
    )

    # If not found and path is absolute, try relative path (handles new data)
    if record is None and file_path.is_absolute():
        try:
            relative = to_relative_path(file_path, user_id)
            record = (
                db.query(UploadedFile)
                .filter(UploadedFile.storage_path == relative)
                .first()
            )
        except ValueError:
            pass

    return record
