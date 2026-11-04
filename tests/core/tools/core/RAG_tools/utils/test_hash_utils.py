"""Tests for hash utility functions."""

import hashlib

import pytest

from xagent.core.tools.core.RAG_tools.utils.hash_utils import (
    compute_content_hash,
    compute_file_hash,
    validate_hash_format,
)


class TestComputeContentHash:
    """Test compute_content_hash function."""

    def test_sha256_hash_computation(self):
        """Test SHA256 hash computation."""
        content = b"Hello, World!"
        expected_hash = hashlib.sha256(content).hexdigest()

        result = compute_content_hash(content, "sha256")
        assert result == expected_hash
        assert len(result) == 64  # SHA256 produces 64 character hex string

    def test_md5_hash_computation(self):
        """Test MD5 hash computation."""
        content = b"test content"
        expected_hash = hashlib.md5(content).hexdigest()

        result = compute_content_hash(content, "md5")
        assert result == expected_hash
        assert len(result) == 32  # MD5 produces 32 character hex string

    def test_different_content_different_hash(self):
        """Test that different content produces different hashes."""
        hash1 = compute_content_hash(b"content1")
        hash2 = compute_content_hash(b"content2")

        assert hash1 != hash2

    def test_same_content_same_hash(self):
        """Test that same content produces same hash."""
        content = b"identical content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)

        assert hash1 == hash2

    def test_empty_content_hash(self):
        """Test hash computation for empty content."""
        empty_hash = compute_content_hash(b"")
        expected = hashlib.sha256(b"").hexdigest()

        assert empty_hash == expected

    def test_unsupported_algorithm_raises_error(self):
        """Test that unsupported algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            compute_content_hash(b"content", "unsupported_algo")


class TestComputeFileHash:
    """Test compute_file_hash function."""

    def test_file_hash_computation(self, tmp_path):
        """Test file hash computation."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        result = compute_file_hash(str(test_file))
        expected = hashlib.sha256(test_content.encode("utf-8")).hexdigest()

        assert result == expected

    def test_large_file_hash(self, tmp_path):
        """Test hash computation for large file."""
        test_file = tmp_path / "large.txt"
        large_content = "x" * 1024 * 1024  # 1MB content
        test_file.write_text(large_content)

        result = compute_file_hash(str(test_file))
        expected = hashlib.sha256(large_content.encode("utf-8")).hexdigest()

        assert result == expected

    def test_binary_file_hash(self, tmp_path):
        """Test hash computation for binary file."""
        test_file = tmp_path / "binary.bin"
        binary_content = b"\x00\x01\x02\x03\xff\xfe\xfd"
        test_file.write_bytes(binary_content)

        result = compute_file_hash(str(test_file))
        expected = hashlib.sha256(binary_content).hexdigest()

        assert result == expected

    def test_empty_file_hash(self, tmp_path):
        """Test hash computation for empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        result = compute_file_hash(str(test_file))
        expected = hashlib.sha256(b"").hexdigest()

        assert result == expected

    def test_md5_file_hash(self, tmp_path):
        """Test MD5 file hash computation."""
        test_file = tmp_path / "test.txt"
        test_content = "test content"
        test_file.write_text(test_content)

        result = compute_file_hash(str(test_file), "md5")
        expected = hashlib.md5(test_content.encode("utf-8")).hexdigest()

        assert result == expected

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises IOError."""
        with pytest.raises(IOError, match="Failed to read file for hashing"):
            compute_file_hash("/nonexistent/path/file.txt")

    def test_unsupported_algorithm_raises_error(self, tmp_path):
        """Test that unsupported algorithm raises ValueError."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            compute_file_hash(str(test_file), "unsupported_algo")


class TestValidateHashFormat:
    """Test validate_hash_format function."""

    def test_valid_sha256_hash(self):
        """Test validation of valid SHA256 hash."""
        valid_hash = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
        result = validate_hash_format(valid_hash)
        assert result is True

    def test_valid_md5_hash(self):
        """Test validation of valid MD5 hash."""
        valid_hash = "5d41402abc4b2a76b9719d911017c592"
        result = validate_hash_format(valid_hash)
        assert result is True

    def test_empty_hash_raises_error(self):
        """Test that empty hash raises ValueError."""
        with pytest.raises(ValueError, match="Hash string cannot be empty"):
            validate_hash_format("")

    def test_none_hash_raises_error(self):
        """Test that None hash raises ValueError."""
        with pytest.raises(ValueError, match="Hash string cannot be empty"):
            validate_hash_format(None)

    def test_invalid_hex_characters_raises_error(self):
        """Test that invalid hex characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid hash format"):
            validate_hash_format("zzzzggggyyyyhhhh")

    def test_expected_length_validation(self):
        """Test hash length validation."""
        # Valid SHA256 length
        valid_sha256 = "a" * 64
        assert validate_hash_format(valid_sha256, expected_length=64)

        # Invalid length
        with pytest.raises(ValueError, match="Hash length mismatch"):
            validate_hash_format("a" * 32, expected_length=64)

    def test_no_expected_length(self):
        """Test validation without expected length."""
        # Any valid hex string should pass
        assert validate_hash_format("abc123")
        assert validate_hash_format("ABCDEF1234567890")


class TestHashUtilsIntegration:
    """Integration tests for hash utilities."""

    def test_content_to_file_hash_consistency(self, tmp_path):
        """Test that file hash matches content hash."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        # Hash from file
        file_hash = compute_file_hash(str(test_file))

        # Hash from content
        content_hash = compute_content_hash(test_content.encode("utf-8"))

        assert file_hash == content_hash

    def test_different_algorithms(self, tmp_path):
        """Test different hash algorithms produce different results."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        sha256_hash = compute_file_hash(str(test_file), "sha256")
        md5_hash = compute_file_hash(str(test_file), "md5")

        assert sha256_hash != md5_hash
        assert len(sha256_hash) == 64  # SHA256
        assert len(md5_hash) == 32  # MD5

    def test_hash_format_validation_workflow(self):
        """Test complete workflow of hash computation and validation."""
        content = b"test content"

        # Compute hash
        hash_value = compute_content_hash(content)

        # Validate format
        assert validate_hash_format(hash_value)
        assert validate_hash_format(hash_value, expected_length=64)

        # Test with different content
        different_hash = compute_content_hash(b"different content")
        assert hash_value != different_hash
        assert validate_hash_format(different_hash)
