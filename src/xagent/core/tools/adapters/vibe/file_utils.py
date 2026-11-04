"""
Shared file utilities for workspace tools
"""

import logging
from pathlib import Path
from typing import List, Optional

from ....workspace import TaskWorkspace

logger = logging.getLogger(__name__)


def register_created_files(
    workspace: TaskWorkspace,
    working_directory: str,
    skip_dirs: Optional[List[str]] = None,
) -> None:
    """
    Register files created by code execution in workspace.

    Args:
        workspace: The workspace instance
        working_directory: Directory to scan for new files
        skip_dirs: Directory names to skip (e.g., ['node_modules', '__pycache__'])
    """
    if not workspace:
        return

    work_dir = Path(working_directory)
    if not work_dir.exists():
        return

    skip_dirs = skip_dirs or []

    # Scan current files in workspace
    existing_files = set()
    if workspace.workspace_dir.exists():
        for f in workspace.workspace_dir.rglob("*"):
            if f.is_file() and not any(p.startswith(".") for p in f.parts):
                existing_files.add(f)

    # Scan for new files
    for file_path in work_dir.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip hidden files and specified directories
        if any(part.startswith(".") for part in file_path.parts):
            continue
        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            continue

        try:
            relative_path = str(file_path.relative_to(workspace.workspace_dir))
            if relative_path not in existing_files:
                workspace.register_file(str(file_path))
                logger.info(
                    f"Registered file created by code execution: {relative_path}"
                )
        except Exception as e:
            logger.warning(f"Failed to register file {file_path}: {e}")
