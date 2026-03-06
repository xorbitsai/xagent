"""File storage configuration for xagent web application"""

import logging
import os
from pathlib import Path
from typing import List, Optional

# Get the base directory for web application
WEB_DIR = Path(__file__).parent

# File storage configuration
# Support XAGENT_UPLOADS_DIR environment variable, default to WEB_DIR/uploads for backward compatibility
_UPLOADS_DIR = os.getenv("XAGENT_UPLOADS_DIR")
if _UPLOADS_DIR:
    UPLOADS_DIR = Path(_UPLOADS_DIR)
else:
    UPLOADS_DIR = WEB_DIR / "uploads"

STATIC_DIR = WEB_DIR / "static"

# Ensure uploads directory exists
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# External upload directories (for accessing knowledge base files from other projects)
# Format: comma-separated list of directory paths
# Example: /path/to/FenixAOS/src/fenixaos/web/uploads,/another/path/uploads
_EXTERNAL_UPLOAD_DIRS = os.getenv("XAGENT_EXTERNAL_UPLOAD_DIRS", "")
ALLOWED_EXTERNAL_UPLOAD_DIRS: List[Path] = []

if _EXTERNAL_UPLOAD_DIRS:
    logger = logging.getLogger(__name__)
    for dir_path in _EXTERNAL_UPLOAD_DIRS.split(","):
        dir_path = dir_path.strip()
        if dir_path:
            path = Path(dir_path)
            if path.exists():
                ALLOWED_EXTERNAL_UPLOAD_DIRS.append(path)
                logger.info(f"Added external upload directory: {path}")
            else:
                logger.warning(f"External upload directory does not exist: {path}")

# File storage paths for AI tools
FILE_STORAGE_BASE_DIR = UPLOADS_DIR
FILE_STORAGE_URL_BASE = "/uploads"

# Supported file types
ALLOWED_EXTENSIONS = {
    "general": [
        ".txt",
        ".md",
        ".py",
        ".js",
        ".json",
        ".csv",
        ".doc",
        ".docx",
        ".pdf",
        ".html",
        ".htm",
        ".xlsx",
        ".xls",
        ".pptx",
    ],
    "text": [".txt", ".md", ".html", ".htm"],
    "code": [".py", ".js", ".json", ".html", ".htm"],
    "data": [".csv", ".json", ".xlsx", ".xls"],
    "document": [
        ".doc",
        ".docx",
        ".pdf",
        ".txt",
        ".md",
        ".html",
        ".htm",
        ".xlsx",
        ".xls",
        ".pptx",
    ],
}

# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024


def get_upload_path(
    filename: str,
    task_id: Optional[str] = None,
    folder: Optional[str] = None,
    user_id: Optional[int] = None,
) -> Path:
    """Get the full path for an uploaded file.

    Security: Extracts only the basename from filename to prevent path traversal attacks.
    For example, "../../../etc/passwd" becomes "passwd".
    """
    # SECURITY: Extract only basename to prevent path traversal attacks
    safe_filename = Path(filename).name

    if user_id:
        # Create user-specific directory structure
        user_dir = UPLOADS_DIR / f"user_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)

        if task_id and folder:
            # Create task-specific folder under user directory
            task_dir = user_dir / f"task_{task_id}" / folder
            task_dir.mkdir(parents=True, exist_ok=True)
            return task_dir / safe_filename
        else:
            # User's root directory
            return user_dir / safe_filename
    elif task_id and folder:
        # Create task-specific folder structure (backward compatibility)
        task_dir = UPLOADS_DIR / f"task_{task_id}" / folder
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir / safe_filename
    else:
        # Default behavior
        return UPLOADS_DIR / safe_filename


def get_file_url(
    filename: str,
    task_id: Optional[str] = None,
    folder: Optional[str] = None,
    user_id: Optional[int] = None,
) -> str:
    """Get the URL for accessing an uploaded file.

    Security: Extracts only the basename from filename to prevent path traversal attacks.
    """
    # SECURITY: Extract only basename to prevent path traversal attacks
    safe_filename = Path(filename).name

    if user_id:
        if task_id and folder:
            return f"{FILE_STORAGE_URL_BASE}/{safe_filename}"
        else:
            return f"{FILE_STORAGE_URL_BASE}/user_{user_id}/{safe_filename}"
    elif task_id and folder:
        return f"{FILE_STORAGE_URL_BASE}/task_{task_id}/{folder}/{safe_filename}"
    else:
        return f"{FILE_STORAGE_URL_BASE}/{safe_filename}"


def is_allowed_file(filename: str, task_type: str = "general") -> bool:
    """Check if file is allowed for the given task type"""
    file_ext = Path(filename).suffix.lower()
    extensions = ALLOWED_EXTENSIONS.get(task_type, ALLOWED_EXTENSIONS["general"])
    return file_ext in extensions


def get_file_info(file_path: str) -> dict | None:
    """Get file information"""
    path = Path(file_path)
    if not path.exists():
        return None

    stat = path.stat()
    return {
        "filename": path.name,
        "file_path": str(path),
        "file_size": stat.st_size,
        "modified_time": stat.st_mtime,
        "is_file": path.is_file(),
        "extension": path.suffix.lower(),
    }
