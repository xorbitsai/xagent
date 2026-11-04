"""
Legacy file path handling utilities.

This module provides functions for handling legacy file paths (pre-file_id system).
It includes utilities for resolving legacy paths to actual file system locations,
inferring ownership from paths, and maintaining backward compatibility.

The legacy system used relative paths like:
- web_task_235/output/generated_image.jpeg
- task_123/result.pdf

The new system uses UUID file_id references with database records.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from ...config import get_uploads_dir
from ...core.workspace import scoped_user_root

logger = logging.getLogger(__name__)


def is_valid_uuid(file_id: str) -> bool:
    """Check if the string is a valid UUID format."""
    try:
        uuid.UUID(file_id)
        return True
    except ValueError:
        return False


def infer_user_id_from_legacy_path(db: Session, file_path: str) -> Optional[int]:
    """
    Infer user_id from legacy path format (e.g., web_task_235).

    This function attempts to extract task_id from the path and query the database
    to find the corresponding user_id.

    Args:
        db: Database session
        file_path: Legacy file path (e.g., "web_task_235/output/file.jpeg")

    Returns:
        User ID if found, None otherwise
    """
    from ..models.task import Task

    path_parts = file_path.split("/")
    for part in path_parts:
        if part.startswith("web_task_") or part.startswith("task_"):
            try:
                task_id = int(part.replace("web_task_", "", 1).replace("task_", "", 1))
                task = db.query(Task.user_id).filter(Task.id == task_id).first()
                if task:
                    return int(task.user_id)
            except (ValueError, AttributeError):
                continue
    return None


def infer_owner_from_relative_path(
    db: Session, relative_path: str
) -> Optional[Tuple[int, Optional[int]]]:
    """
    Infer owner (user_id) and task_id from a relative path.

    This is a more comprehensive version that handles various path formats.

    Args:
        db: Database session
        relative_path: Relative path (e.g., "user_1/web_task_235/output/file.jpeg")

    Returns:
        Tuple of (user_id, task_id) if found, None otherwise
    """
    from ..models.task import Task

    path_parts = Path(relative_path).parts
    if not path_parts:
        return None

    user_id: Optional[int] = None
    task_id: Optional[int] = None

    first = path_parts[0]
    remaining = path_parts[1:] if len(path_parts) > 1 else []

    # Handle user_* prefix
    if first.startswith("user_"):
        try:
            user_id = int(first.replace("user_", "", 1))
        except ValueError:
            return None
        if remaining:
            task_segment = remaining[0]
            if task_segment.startswith("web_task_"):
                try:
                    task_id = int(task_segment.replace("web_task_", "", 1))
                except ValueError:
                    task_id = None
            elif task_segment.startswith("task_"):
                try:
                    task_id = int(task_segment.replace("task_", "", 1))
                except ValueError:
                    task_id = None
        return user_id, task_id

    # Handle task_* or web_task_* prefix without user_ prefix
    if first.startswith("web_task_"):
        try:
            task_id = int(first.replace("web_task_", "", 1))
        except ValueError:
            return None
    elif first.startswith("task_"):
        try:
            task_id = int(first.replace("task_", "", 1))
        except ValueError:
            return None

    if task_id is not None:
        task_row = db.query(Task).filter(Task.id == task_id).first()
        if task_row and getattr(task_row, "user_id", None) is not None:
            return int(getattr(task_row, "user_id")), task_id

    return None


def resolve_legacy_file_path(file_path: str, user_id: int) -> Optional[Path]:
    """
    Resolve a legacy file path to an actual file system path.

    This function tries to find a file by:
    1. Direct relative path lookup
    2. Filename search within user directory

    Args:
        file_path: Legacy file path (e.g., "web_task_235/output/file.jpeg")
        user_id: User ID to scope the search

    Returns:
        Resolved absolute Path if found, None otherwise
    """
    user_root = scoped_user_root(get_uploads_dir(), user_id)
    if not user_root.exists():
        return None

    # Try direct relative path
    candidate = user_root / file_path
    if candidate.exists() and candidate.is_file():
        return candidate

    # Try to find by filename alone (for simple paths)
    filename = Path(file_path).name
    matches = list(user_root.rglob(filename))
    if matches:
        # Return the most recent match
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    return None


def resolve_legacy_file_path_cross_user(file_path: str) -> Optional[Tuple[Path, int]]:
    """
    Resolve a legacy file path across all user directories.

    Internal compatibility lookup only. The caller must authenticate and verify
    the resolved owner; public download/preview routes must never use this.

    Args:
        file_path: Legacy file path

    Returns:
        Tuple of (resolved_path, user_id) if found, None otherwise
    """
    if not get_uploads_dir().exists():
        return None

    # Try to find the file in any user directory
    for user_dir in get_uploads_dir().iterdir():
        if not user_dir.is_dir() or not user_dir.name.startswith("user_"):
            continue

        candidate = user_dir / file_path
        if candidate.exists() and candidate.is_file():
            try:
                owner_user_id = int(user_dir.name.replace("user_", "", 1))
                return candidate, owner_user_id
            except ValueError:
                continue

    return None
