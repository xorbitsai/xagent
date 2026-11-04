"""Tests for xagent.web.config module, focusing on security functions."""

import pytest

from xagent.web.config import (
    FILE_STORAGE_URL_BASE,
    MAX_COLLECTION_NAME_BYTES,
    MAX_COLLECTION_NAME_LENGTH,
    get_file_url,
    sanitize_path_component,
)


class TestSanitizePathComponent:
    """Test sanitize_path_component function for path traversal prevention."""

    def test_valid_collection_names(self):
        """Test that valid collection names are accepted."""
        valid_names = [
            "my_collection",
            "my collection",
            "collection-123",
            "test123",
            "a",
            "A",
            "collection_name_123",
            "a" * MAX_COLLECTION_NAME_LENGTH,  # Maximum length
        ]

        for name in valid_names:
            result = sanitize_path_component(name, "collection")
            assert result == name
            assert isinstance(result, str)

    def test_path_traversal_attacks(self):
        """Test that path traversal attacks are rejected."""
        malicious_names = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "collection/../other",
            "collection\\..\\other",
            "../collection",
            "collection/",
            "collection\\",
            "/collection",
            "\\collection",
        ]

        for name in malicious_names:
            with pytest.raises(
                ValueError, match="path separators or invalid characters"
            ):
                sanitize_path_component(name, "collection")

    def test_empty_and_whitespace(self):
        """Test that empty and whitespace-only names are rejected."""
        invalid_names = [
            "",
            "   ",
            "\t",
            "\n",
            "\r\n",
        ]

        for name in invalid_names:
            with pytest.raises(ValueError, match="cannot be empty"):
                sanitize_path_component(name, "collection")

    def test_length_limits(self):
        """Test that length limits are enforced."""
        # Test minimum length (already tested by empty string, but test edge case)
        single_char = "a"
        result = sanitize_path_component(single_char, "collection")
        assert result == single_char

        # Test maximum length
        max_length_name = "a" * MAX_COLLECTION_NAME_LENGTH
        result = sanitize_path_component(max_length_name, "collection")
        assert result == max_length_name

        # Test exceeding maximum length
        too_long_name = "a" * (MAX_COLLECTION_NAME_LENGTH + 1)
        with pytest.raises(
            ValueError, match=f"exceeds maximum length of {MAX_COLLECTION_NAME_LENGTH}"
        ):
            sanitize_path_component(too_long_name, "collection")

    def test_utf8_byte_length_limit(self):
        """Test that UTF-8 byte-length limits are enforced for Unicode names."""
        within_limit_name = "知" * (
            MAX_COLLECTION_NAME_BYTES // len("知".encode("utf-8"))
        )
        assert (
            sanitize_path_component(within_limit_name, "collection")
            == within_limit_name
        )

        over_limit_name = "知" * (
            (MAX_COLLECTION_NAME_BYTES // len("知".encode("utf-8"))) + 1
        )
        assert len(over_limit_name) < MAX_COLLECTION_NAME_LENGTH
        with pytest.raises(
            ValueError,
            match=f"exceeds maximum byte length of {MAX_COLLECTION_NAME_BYTES}",
        ):
            sanitize_path_component(over_limit_name, "collection")

    def test_invalid_characters(self):
        """Test that invalid characters are rejected."""
        invalid_names = [
            "collection@name",  # @ symbol
            "collection#name",  # # symbol
            "collection$name",  # $ symbol
            "collection%name",  # % symbol
            "collection&name",  # & symbol
            "collection*name",  # * symbol
            "collection+name",  # + symbol
            "collection=name",  # = symbol
            "collection?name",  # ? symbol
            "collection|name",  # | symbol
            "collection<name",  # < symbol
            "collection>name",  # > symbol
            "collection:name",  # : symbol (Windows path separator)
            "collection;name",  # ; symbol
            "collection'name",  # ' symbol
            'collection"name',  # " symbol
            "collection[name",  # [ symbol
            "collection]name",  # ] symbol
            "collection{name",  # { symbol
            "collection}name",  # } symbol
            "collection,name",  # , symbol
            "collection.name",  # . symbol (but multiple dots might be OK if not path traversal)
            "collection!name",  # ! symbol
            "collection~name",  # ~ symbol
        ]

        for name in invalid_names:
            with pytest.raises(ValueError, match="contains invalid characters"):
                sanitize_path_component(name, "collection")

    def test_valid_special_characters(self):
        """Test that allowed spaces and special characters are accepted."""
        valid_names = [
            "collection name",
            "collection_name",
            "collection-name",
            "collection name-123",
            "collection_name-123",
            "collection-123_name",
            "_collection",
            "-collection",
            "collection_",
            "collection-",
        ]

        for name in valid_names:
            result = sanitize_path_component(name, "collection")
            assert result == name

    def test_whitespace_stripping(self):
        """Test that leading/trailing whitespace is stripped before validation."""
        # Valid name with whitespace should be stripped and accepted
        name_with_whitespace = "  my collection  "
        result = sanitize_path_component(name_with_whitespace, "collection")
        assert result == "my collection"

        # But if after stripping it becomes invalid, it should be rejected
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_path_component("   ", "collection")

    def test_component_type_in_error_messages(self):
        """Test that component_type appears in error messages."""
        with pytest.raises(ValueError, match="Invalid collection name"):
            sanitize_path_component("", "collection")

        with pytest.raises(ValueError, match="Invalid folder name"):
            sanitize_path_component("", "folder")

        with pytest.raises(ValueError, match="Invalid path name"):
            sanitize_path_component("", "path")

    def test_path_traversal_after_basename_extraction(self):
        """Test that path traversal is caught even after basename extraction."""
        # Path like "../../../etc" becomes "etc" after basename extraction
        # But we check if safe_name != name, so it should still be rejected
        malicious = "../../../etc"
        with pytest.raises(ValueError, match="path separators or invalid characters"):
            sanitize_path_component(malicious, "collection")

    def test_unicode_and_special_unicode(self):
        valid_unicode_names = [
            "collection中文",
            "collectioné",
            "collectionñ",
            "示例知识库集合",
            "collection١٢٣",
            "知识库_123-β",
        ]

        for name in valid_unicode_names:
            result = sanitize_path_component(name, "collection")
            assert result == name

        invalid_unicode_names = [
            "collection🚀",
            "collectiοn",  # Greek omicron mixed with Latin
            "cоllection",  # Cyrillic o mixed with Latin
            "collectionα",  # Greek alpha mixed with Latin
        ]
        for name in invalid_unicode_names:
            with pytest.raises(ValueError, match="contains"):
                sanitize_path_component(name, "collection")

    def test_rejects_compatibility_homoglyph_forms(self):
        with pytest.raises(ValueError, match="contains invalid characters"):
            sanitize_path_component("Ａgent", "collection")

    def test_numeric_only_names(self):
        """Test that numeric-only names are accepted."""
        numeric_names = ["123", "0", "999999"]
        for name in numeric_names:
            result = sanitize_path_component(name, "collection")
            assert result == name

    def test_mixed_case_names(self):
        """Test that mixed case names are accepted."""
        mixed_names = ["MyCollection", "myCollection", "MY_COLLECTION", "My-Collection"]
        for name in mixed_names:
            result = sanitize_path_component(name, "collection")
            assert result == name


class TestGetFileUrl:
    def test_get_file_url_encodes_unicode_collection_and_filename(self):
        collection_name = "示例知识库集合"
        filename = "报告.txt"

        url = get_file_url(filename, user_id=7, collection=collection_name)

        assert url == (
            f"{FILE_STORAGE_URL_BASE}/user_7/"
            "%E7%A4%BA%E4%BE%8B%E7%9F%A5%E8%AF%86%E5%BA%93%E9%9B%86%E5%90%88/"
            "%E6%8A%A5%E5%91%8A.txt"
        )
