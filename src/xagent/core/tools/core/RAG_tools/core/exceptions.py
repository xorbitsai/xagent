"""Core exceptions for RAG tools.

This module defines custom exceptions used throughout the RAG tools core layer.
All exceptions follow a consistent hierarchy for proper error handling.
"""

from typing import Optional


class RagCoreException(Exception):
    """Base exception for all RAG core operations.

    This is the root exception class for all RAG-related errors.
    It provides a consistent interface for error handling across
    all RAG tool operations.

    Attributes:
        message: Human-readable error message
        details: Optional additional context or metadata
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional additional context or metadata
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DocumentNotFoundError(RagCoreException):
    """Exception raised when a requested document is not found.

    This exception is raised when attempting to access a document
    that does not exist in the system.
    """

    pass


class DocumentValidationError(RagCoreException):
    """Exception raised when document validation fails.

    This exception is raised when document data fails validation,
    such as invalid file types, corrupted content, or schema violations.
    """

    pass


class DatabaseOperationError(RagCoreException):
    """Exception raised when database operations fail.

    This exception is raised when there are issues with database
    operations like connection failures, schema errors, or constraint violations.
    """

    pass


class ConfigurationError(RagCoreException):
    """Exception raised when configuration is invalid or missing.

    This exception is raised when required configuration parameters
    are missing, invalid, or improperly formatted.
    """

    pass


class EmbeddingAdapterError(ConfigurationError):
    """Exception raised when embedding adapter resolution or usage fails."""

    pass


class HashComputationError(RagCoreException):
    """Exception raised when content hash computation fails.

    This exception is raised when there's an error computing
    content hashes, such as file read errors or algorithm failures.
    """

    pass


class VectorValidationError(RagCoreException):
    """Exception raised when vector validation fails.

    This exception is raised when vector data is invalid, such as
    wrong dimensions, invalid format, or type mismatches.

    """

    pass


class VersionManagementError(RagCoreException):
    """Exception raised when version management operations fail.

    This exception is raised when there are issues with version
    management operations like listing candidates or promoting versions.
    """

    pass


class MainPointerError(RagCoreException):
    """Exception raised when main pointer operations fail.

    This exception is raised when there are issues with main pointer
    management, such as pointer not found or update failures.
    """

    pass


class CascadeCleanupError(RagCoreException):
    """Exception raised when cascade cleanup operations fail.

    This exception is raised when there are issues during cascade
    cleanup operations, such as partial cleanup failures.
    """

    pass


class ProgressPersistenceError(RagCoreException):
    """Exception raised when progress persistence operations fail.

    This exception is raised when saving or loading task progress fails,
    typically due to IO errors or storage issues.
    """

    pass
