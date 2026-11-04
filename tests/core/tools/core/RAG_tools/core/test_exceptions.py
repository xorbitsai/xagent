"""Tests for core exceptions."""

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentNotFoundError,
    DocumentValidationError,
    HashComputationError,
    RagCoreException,
    VectorValidationError,
)


class TestRagCoreException:
    """Test RagCoreException base class."""

    def test_exception_creation_without_details(self):
        """Test exception creation without details."""
        exception = RagCoreException("Test error")

        assert str(exception) == "Test error"
        assert exception.message == "Test error"
        assert exception.details == {}

    def test_exception_creation_with_details(self):
        """Test exception creation with details."""
        details = {"file_path": "/tmp/test.txt", "error_code": 500}
        exception = RagCoreException("Test error", details=details)

        assert (
            str(exception)
            == "Test error | Details: {'file_path': '/tmp/test.txt', 'error_code': 500}"
        )
        assert exception.message == "Test error"
        assert exception.details == details

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from RagCoreException."""
        exceptions = [
            DocumentNotFoundError("Document not found"),
            DocumentValidationError("Validation failed"),
            DatabaseOperationError("Database error"),
            ConfigurationError("Configuration error"),
            HashComputationError("Hash computation failed"),
            VectorValidationError("Vector validation failed"),
        ]

        for exception in exceptions:
            assert isinstance(exception, RagCoreException)
            assert isinstance(exception, Exception)


class TestSpecificExceptions:
    """Test specific exception classes."""

    def test_document_not_found_error(self):
        """Test DocumentNotFoundError."""
        exception = DocumentNotFoundError("Document not found")
        assert str(exception) == "Document not found"
        assert isinstance(exception, RagCoreException)

    def test_document_validation_error(self):
        """Test DocumentValidationError."""
        exception = DocumentValidationError("Validation failed")
        assert str(exception) == "Validation failed"
        assert isinstance(exception, RagCoreException)

    def test_database_operation_error(self):
        """Test DatabaseOperationError."""
        exception = DatabaseOperationError("Database error")
        assert str(exception) == "Database error"
        assert isinstance(exception, RagCoreException)

    def test_configuration_error(self):
        """Test ConfigurationError."""
        exception = ConfigurationError("Configuration error")
        assert str(exception) == "Configuration error"
        assert isinstance(exception, RagCoreException)

    def test_hash_computation_error(self):
        """Test HashComputationError."""
        exception = HashComputationError("Hash computation failed")
        assert str(exception) == "Hash computation failed"
        assert isinstance(exception, RagCoreException)

    def test_vector_validation_error(self):
        """Test VectorValidationError."""
        exception = VectorValidationError("Vector validation failed")
        assert str(exception) == "Vector validation failed"
        assert isinstance(exception, RagCoreException)


class TestExceptionHierarchy:
    """Test exception hierarchy and inheritance."""

    def test_exception_hierarchy(self):
        """Test that exceptions form correct hierarchy."""

        # Specific exceptions should be instances of base
        specific_exceptions = [
            DocumentNotFoundError("Document not found"),
            DocumentValidationError("Validation failed"),
            DatabaseOperationError("Database error"),
            ConfigurationError("Configuration error"),
            HashComputationError("Hash computation failed"),
            VectorValidationError("Vector validation failed"),
        ]

        for exception in specific_exceptions:
            assert isinstance(exception, RagCoreException)
            assert issubclass(type(exception), RagCoreException)
