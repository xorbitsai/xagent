"""File utility functions for RAG tools.


This module provides common file operations including file type detection,
content reading, and file validation.
"""

from pathlib import Path

from ..core.exceptions import DocumentValidationError


def check_file_type(source_path: str) -> str:
    """Check file type from file extension.

    Args:
        source_path: Path to the file.

    Returns:
        Checked file type string.


    Raises:
        DocumentValidationError: If file type cannot be determined.
    """
    path = Path(source_path)
    suffix = path.suffix.lower()

    # Map common file extensions to types
    extension_map = {
        ".txt": "txt",
        ".md": "md",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf",
        ".doc": "doc",
        ".docx": "docx",
        ".csv": "csv",
        ".xlsx": "xlsx",
        ".ppt": "ppt",
        ".pptx": "pptx",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".gif": "image",
        ".bmp": "image",
        ".tiff": "image",
        ".tif": "image",
    }

    file_type = extension_map.get(suffix)
    if not file_type:
        raise DocumentValidationError(f"Unsupported file type: {suffix}")

    return file_type


def validate_file_path(file_path: str) -> bool:
    """Validate if file path exists and is accessible.

    Args:
        file_path: Path to the file to validate.

    Returns:
        True if file exists and is accessible.

    Raises:
        DocumentValidationError: If file path is invalid.
    """
    if not file_path:
        raise DocumentValidationError("File path cannot be empty")

    path = Path(file_path)
    if not path.exists():
        raise DocumentValidationError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise DocumentValidationError(f"Path is not a file: {file_path}")

    return True
