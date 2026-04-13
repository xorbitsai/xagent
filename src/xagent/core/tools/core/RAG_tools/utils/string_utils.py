"""
Utility functions for string manipulation and escaping.
"""

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Pattern for sanitizing document IDs and filenames
# Only allows: letters, numbers, underscore, hyphen
_DOC_ID_SANITIZE_PATTERN: re.Pattern[str] = re.compile(r"[^A-Za-z0-9_-]")


def escape_lancedb_string(input_string: Any) -> str:
    """
    Safely escapes a string for use in LanceDB 'where' clauses.

    This function prevents injection attacks by properly escaping
    special characters that could alter the query's intent.

    Args:
        input_string: The string to be escaped.

    Returns:
        The escaped string, safe for use in LanceDB 'where' clauses.
    """
    if not isinstance(input_string, str):
        return str(input_string)
    # Escape single quotes by doubling them, and escape backslashes
    return input_string.replace("\\", "\\\\").replace("'", "''")


def build_lancedb_filter_expression(
    filters: Dict[str, Any],
    *,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    skip_user_filter: bool = False,
) -> str:
    """
    Builds a safe LanceDB filter expression from a dictionary of filters.

    This function now uses the abstract filter layer internally for better
    backend compatibility, while maintaining the same interface for
    backward compatibility.

    Args:
        filters: A dictionary where keys are column names and values are the filter values.
        user_id: Optional user ID for multi-tenancy filtering.
        is_admin: Whether the user has admin privileges.
        skip_user_filter: If True, bypasses user permission filter.

    Returns:
        A string representing the safely constructed LanceDB filter expression.
    """
    from ..storage.contracts import (
        FilterCondition,
        FilterExpression,
        FilterOperator,
    )
    from ..storage.factory import get_vector_index_store

    # Convert to FilterCondition list
    conditions: List[FilterCondition] = []
    for key, value in filters.items():
        conditions.append(
            FilterCondition(field=key, operator=FilterOperator.EQ, value=value)
        )

    # Use abstract filter builder
    vector_store = get_vector_index_store()

    # Combine conditions with AND (tuple convention)
    # Type: FilterExpression can be FilterCondition or tuple of FilterConditions
    if len(conditions) == 1:
        filter_expr: FilterExpression = conditions[0]
    else:
        filter_expr = tuple(conditions)

    # Get backend-specific syntax
    backend_filter = vector_store.build_filter_expression(
        filters=filter_expr,
        user_id=user_id if not skip_user_filter else None,
        is_admin=is_admin or skip_user_filter,
    )

    return backend_filter or ""


def sanitize_for_doc_id(text: str, max_length: int = 64) -> str:
    """
    Sanitize text for safe use in document IDs.

    This function ensures that the resulting string is safe for use in:
    - File system paths
    - Database identifiers
    - URL parameters
    - API endpoints

    Rules:
    - Only allows: letters (A-Z, a-z), numbers (0-9), underscore (_), hyphen (-)
    - Replaces all other characters with underscore
    - Removes leading/trailing underscores
    - Collapses multiple consecutive underscores
    - Limits length to max_length (default: 64)
    - If result is empty, returns a hash-based fallback

    Args:
        text: The text to sanitize (e.g., filename, user input).
        max_length: Maximum length of the sanitized string (default: 64).

    Returns:
        A sanitized string safe for use in document IDs.

    Examples:
        >>> sanitize_for_doc_id("report 2024.pdf")
        'report_2024_pdf'
        >>> sanitize_for_doc_id("../../etc/passwd")
        'etc_passwd'
        >>> sanitize_for_doc_id("doc@2024#test")
        '2024'
        >>> sanitize_for_doc_id("")
        'a1b2c3d4'  # hash-based fallback
    """
    if not text:
        # Return a short hash-based identifier if input is empty
        return hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest()[:8]

    # Replace non-allowed characters with underscore
    sanitized = _DOC_ID_SANITIZE_PATTERN.sub("_", text)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Collapse multiple consecutive underscores into a single one
    sanitized = re.sub(r"_+", "_", sanitized)

    # If result is empty after sanitization, use hash fallback
    if not sanitized:
        sanitized = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # Limit length, ensuring we don't cut in the middle of a word
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("_")
        # If we cut off everything, use hash fallback
        if not sanitized:
            sanitized = hashlib.sha256(text.encode("utf-8")).hexdigest()[:max_length]

    return sanitized


def generate_doc_id_from_filename(
    source_path: str, include_extension: bool = False
) -> str:
    """
    Generate a user-friendly document ID from a file path.

    This function extracts the filename from a path, sanitizes it, and optionally
    appends a short UUID for uniqueness. The result is safe for use in document IDs.

    Format: {sanitized_filename}_{short_uuid}[.{extension}]

    Args:
        source_path: Absolute or relative path to the file.
        include_extension: If True, includes file extension in the doc_id.

    Returns:
        A sanitized document ID based on the filename.

    Examples:
        >>> generate_doc_id_from_filename("/path/to/report 2024.pdf")
        'report_2024_a3f2b5c1'
        >>> generate_doc_id_from_filename("/path/to/report 2024.pdf", include_extension=True)
        'report_2024_a3f2b5c1.pdf'
        >>> generate_doc_id_from_filename("../../etc/passwd")
        'etc_passwd_d4e6f7a8'
    """
    path = Path(source_path)
    filename = path.stem  # Extract filename without extension
    extension = path.suffix.lower() if path.suffix else ""

    # Sanitize the filename
    sanitized = sanitize_for_doc_id(filename, max_length=50)

    # If sanitization resulted in empty string, use fallback
    if not sanitized:
        sanitized = "doc"

    # Generate short UUID for uniqueness (8 characters)
    short_uuid = uuid.uuid4().hex[:8]

    # Combine: filename_uuid
    doc_id = f"{sanitized}_{short_uuid}"

    # Optionally add extension
    if include_extension and extension:
        doc_id = f"{doc_id}{extension}"

    return doc_id


def generate_deterministic_doc_id(collection: str, source_path: str) -> str:
    """Generate a deterministic document ID from collection and source path.

    Same (collection, source_path) always yields the same doc_id, so re-uploading
    or double-submitting the same file results in one record (idempotent registration).

    Args:
        collection: LanceDB collection name.
        source_path: Absolute path to the file.

    Returns:
        A deterministic doc_id: {sanitized_stem}_{hash_8chars}.

    Examples:
        >>> generate_deterministic_doc_id("kb1", "/uploads/user_1/report.docx")
        'report_a1b2c3d4'
    """
    path = Path(source_path)
    filename = path.stem
    sanitized = sanitize_for_doc_id(filename, max_length=50)
    if not sanitized:
        sanitized = "doc"
    key = f"{collection}|{source_path}"
    h = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"{sanitized}_{h}"


# Security validation patterns
_COLLECTION_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[\w -]+$")
_DOC_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_.-]+$")


def validate_collection_name(name: str) -> str:
    """
    Validate collection name for safe use in LanceDB queries.

    Allows Unicode word characters (letters, numbers, underscore), spaces,
    and hyphens. This prevents injection attacks while remaining compatible
    with internationalized collection names.

    Args:
        name: Collection name to validate

    Returns:
        The validated collection name

    Raises:
        ValueError: If collection name contains invalid characters
    """
    if not isinstance(name, str):
        raise ValueError(f"Collection name must be a string, got {type(name)}")
    if not name or not name.strip():
        raise ValueError("Collection name cannot be empty")
    name = name.strip()
    if not _COLLECTION_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid collection name '{name}'. "
            "Only letters, numbers, spaces, underscores, and hyphens are allowed."
        )
    return name


def validate_doc_id(doc_id: str) -> str:
    """
    Validate document ID for safe use in LanceDB queries.

    Allows alphanumeric characters, underscores, hyphens, and dots.
    This is more permissive than collection names since doc_ids
    often include file extensions and UUIDs.

    Args:
        doc_id: Document ID to validate

    Returns:
        The validated document ID

    Raises:
        ValueError: If document ID contains invalid characters
    """
    if not isinstance(doc_id, str):
        raise ValueError(f"Document ID must be a string, got {type(doc_id)}")
    if not doc_id:
        raise ValueError("Document ID cannot be empty")
    if not _DOC_ID_PATTERN.match(doc_id):
        raise ValueError(
            f"Invalid document ID '{doc_id}'. "
            "Only letters, numbers, underscores, hyphens, and dots are allowed."
        )
    return doc_id


def build_safe_collection_filter(
    collection: str, user_filter: Optional[str] = None, doc_id: Optional[str] = None
) -> str:
    """
    Build a safe LanceDB filter expression for collection queries.

    Args:
        collection: Collection name (will be validated)
        user_filter: Optional user permission filter
        doc_id: Optional document ID filter (will be validated)

    Returns:
        Safe filter expression string

    Raises:
        ValueError: If collection or doc_id are invalid
    """
    validated_collection = validate_collection_name(collection)

    filter_parts = [f"collection == '{escape_lancedb_string(validated_collection)}'"]

    if doc_id is not None:
        validated_doc_id = validate_doc_id(doc_id)
        filter_parts.append(f"doc_id == '{escape_lancedb_string(validated_doc_id)}'")

    base_filter = " AND ".join(filter_parts)

    if user_filter:
        return f"({base_filter}) AND ({user_filter})"

    return base_filter
