"""File storage configuration for xagent web application

This module now imports configuration functions from the core config module
to ensure consistency across all xagent components. Paths are computed
dynamically to support environment variable changes at runtime.
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from ..config import get_uploads_dir

# File storage paths for AI tools (computed dynamically when needed)
FILE_STORAGE_URL_BASE = "/uploads"

# Binary file extensions that should not be previewed as text
BINARY_EXTENSIONS = {
    # Image files
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
    ".ico",
    # Video files
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    # Audio files
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".aac",
    # Archive files
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}

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
    ]
    + list(BINARY_EXTENSIONS),
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
    "image": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".ico"],
}

# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Word characters (\w = letters/digits/underscore in any language), spaces, and hyphens.
ALLOWED_NAME_PATTERN = re.compile(r"^[\w -]+$")
ALLOWED_NAME_PATTERN_NO_SPACES = re.compile(r"^[\w-]+$")
_CONFUSABLE_SCRIPT_FAMILIES = ("LATIN", "GREEK", "CYRILLIC")


def _get_confusable_script_family(char: str) -> Optional[str]:
    if not char.isalpha():
        return None

    try:
        unicode_name = unicodedata.name(char)
    except ValueError:
        return None

    for script_family in _CONFUSABLE_SCRIPT_FAMILIES:
        if script_family in unicode_name:
            return script_family

    return None


def _has_mixed_confusable_scripts(value: str) -> bool:
    script_families: set[str] = set()
    for char in value:
        script_family = _get_confusable_script_family(char)
        if script_family:
            script_families.add(script_family)

    return len(script_families) > 1


# Maximum length for collection and folder names (reasonable limit for file system and database)
# This prevents path length issues and database field overflow
MAX_COLLECTION_NAME_LENGTH = 100
MAX_COLLECTION_NAME_BYTES = 255
MIN_COLLECTION_NAME_LENGTH = 1


def sanitize_path_component(name: str, component_type: str = "path") -> str:
    """Sanitize a path component to prevent path traversal attacks.

    This function ensures that path components (like collection names, folder names)
    do not contain path separators or other dangerous characters that could lead
    to path traversal vulnerabilities. It also enforces length limits to prevent
    file system and database issues.

    Args:
        name: The path component name to sanitize.
        component_type: Type of component for error messages (e.g., "collection", "folder").

    Returns:
        Sanitized path component.

    Raises:
        ValueError: If the name is invalid (empty, too long, contains path separators,
            or contains invalid characters after sanitization).

    Examples:
        >>> sanitize_path_component("my_collection")
        'my_collection'
        >>> sanitize_path_component("../../../etc")
        Traceback (most recent call last):
        ...
        ValueError: Invalid collection name: contains path separators or invalid characters
        >>> sanitize_path_component("a" * 200)
        Traceback (most recent call last):
        ...
        ValueError: Invalid collection name: exceeds maximum length of 100 characters
    """
    if not name or not name.strip():
        raise ValueError(f"Invalid {component_type} name: cannot be empty")

    # Remove leading/trailing whitespace
    name = name.strip()

    normalized_name = unicodedata.normalize("NFKC", name)
    if normalized_name != name:
        raise ValueError(
            f"Invalid {component_type} name: contains invalid characters. "
            f"Only letters, numbers, spaces, underscores, and hyphens are allowed."
        )
    name = normalized_name

    # Extract only the basename to prevent path traversal
    # This handles cases like "../../../etc" -> "etc"
    safe_name = Path(name).name

    # Additional validation: ensure no path separators remain
    if "/" in safe_name or "\\" in safe_name:
        raise ValueError(
            f"Invalid {component_type} name: contains path separators or invalid characters"
        )

    # Validate length limits
    if len(safe_name) < MIN_COLLECTION_NAME_LENGTH:
        raise ValueError(
            f"Invalid {component_type} name: too short (minimum {MIN_COLLECTION_NAME_LENGTH} character)"
        )
    if len(safe_name) > MAX_COLLECTION_NAME_LENGTH:
        raise ValueError(
            f"Invalid {component_type} name: exceeds maximum length of {MAX_COLLECTION_NAME_LENGTH} characters"
        )
    if len(safe_name.encode("utf-8")) > MAX_COLLECTION_NAME_BYTES:
        raise ValueError(
            f"Invalid {component_type} name: exceeds maximum byte length of {MAX_COLLECTION_NAME_BYTES}"
        )

    # Collections allow internal spaces, but task folders remain stricter to
    # preserve existing upload path semantics and security expectations.
    allowed_pattern = (
        ALLOWED_NAME_PATTERN
        if component_type == "collection"
        else ALLOWED_NAME_PATTERN_NO_SPACES
    )

    # Validate against allowed character pattern
    # This ensures only safe characters are used
    if not allowed_pattern.match(safe_name):
        allowed_chars = (
            "Only letters, numbers, spaces, underscores, and hyphens are allowed."
            if component_type == "collection"
            else "Only letters, numbers, underscores, and hyphens are allowed."
        )
        raise ValueError(
            f"Invalid {component_type} name: contains invalid characters. "
            f"{allowed_chars}"
        )

    if _has_mixed_confusable_scripts(safe_name):
        raise ValueError(
            f"Invalid {component_type} name: contains mixed-script confusable characters"
        )

    # Ensure the sanitized name matches the original (after stripping)
    # This prevents silent truncation of valid names
    if safe_name != name:
        raise ValueError(
            f"Invalid {component_type} name: contains path separators or invalid characters"
        )

    return safe_name


def get_upload_path(
    filename: str,
    task_id: Optional[str] = None,
    folder: Optional[str] = None,
    user_id: Optional[int] = None,
    collection: Optional[str] = None,
    create_if_not_exists: bool = True,
    collection_is_sanitized: bool = False,
) -> Path:
    """Get the full path for an uploaded file.

    Security: Extracts only the basename from filename to prevent path traversal attacks.
    For example, "../../../etc/passwd" becomes "passwd".

    Args:
        filename: Name of the file
        task_id: Optional task ID
        folder: Optional folder name
        user_id: Optional user ID
        collection: Optional collection name
        create_if_not_exists: If True, create directories if they don't exist.
            Set to False when you only need the path without creating directories
            (e.g., for checking if a directory exists before renaming).
        collection_is_sanitized: If True, treat `collection` as already sanitized by
            `sanitize_path_component(collection, "collection")` and skip sanitization.

    Returns:
        Path object for the file location
    """
    # Get uploads directory dynamically
    uploads_dir = get_uploads_dir()

    # SECURITY: Extract only basename to prevent path traversal attacks
    safe_filename = Path(filename).name

    if user_id:
        # Create user-specific directory structure
        user_dir = uploads_dir / f"user_{user_id}"

        if collection:
            # SECURITY: Sanitize collection name to prevent path traversal attacks
            safe_collection = (
                collection
                if collection_is_sanitized
                else sanitize_path_component(collection, "collection")
            )
            # Create collection-specific directory under user directory
            collection_dir = user_dir / safe_collection
            if create_if_not_exists:
                collection_dir.mkdir(parents=True, exist_ok=True)
            return collection_dir / safe_filename

        if create_if_not_exists:
            user_dir.mkdir(parents=True, exist_ok=True)

        if task_id and folder:
            # SECURITY: Sanitize folder name to prevent path traversal attacks
            safe_folder = sanitize_path_component(folder, "folder")
            # Create task-specific folder under user directory
            task_dir = user_dir / f"task_{task_id}" / safe_folder
            if create_if_not_exists:
                task_dir.mkdir(parents=True, exist_ok=True)
            return task_dir / safe_filename
        else:
            # User's root directory
            return user_dir / safe_filename
    elif task_id and folder:
        # SECURITY: Sanitize folder name to prevent path traversal attacks
        safe_folder = sanitize_path_component(folder, "folder")
        # Create task-specific folder structure (backward compatibility)
        task_dir = uploads_dir / f"task_{task_id}" / safe_folder
        if create_if_not_exists:
            task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir / safe_filename
    else:
        # Default behavior
        return uploads_dir / safe_filename


def get_file_url(
    filename: str,
    task_id: Optional[str] = None,
    folder: Optional[str] = None,
    user_id: Optional[int] = None,
    collection: Optional[str] = None,
) -> str:
    """Get the URL for accessing an uploaded file.

    Security: Extracts only the basename from filename to prevent path traversal attacks.
    """
    # SECURITY: Extract only basename to prevent path traversal attacks
    safe_filename = Path(filename).name

    if user_id:
        if collection:
            # SECURITY: Sanitize collection name to prevent path traversal and URL injection
            safe_collection = sanitize_path_component(collection, "collection")
            encoded_collection = quote(safe_collection, safe="")
            encoded_filename = quote(safe_filename, safe="")
            return f"{FILE_STORAGE_URL_BASE}/user_{user_id}/{encoded_collection}/{encoded_filename}"
        if task_id and folder:
            encoded_filename = quote(safe_filename, safe="")
            return f"{FILE_STORAGE_URL_BASE}/{encoded_filename}"
        else:
            encoded_filename = quote(safe_filename, safe="")
            return f"{FILE_STORAGE_URL_BASE}/user_{user_id}/{encoded_filename}"
    elif task_id and folder:
        # SECURITY: Sanitize folder name to prevent path traversal and URL injection
        safe_folder = sanitize_path_component(folder, "folder")
        encoded_folder = quote(safe_folder, safe="")
        encoded_filename = quote(safe_filename, safe="")
        return f"{FILE_STORAGE_URL_BASE}/task_{task_id}/{encoded_folder}/{encoded_filename}"
    else:
        encoded_filename = quote(safe_filename, safe="")
        return f"{FILE_STORAGE_URL_BASE}/{encoded_filename}"


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
