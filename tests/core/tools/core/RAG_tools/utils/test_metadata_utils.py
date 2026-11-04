"""Tests for metadata serialization utilities."""

import json

import pandas as pd

from xagent.core.tools.core.RAG_tools.utils.metadata_utils import (
    deserialize_metadata,
    serialize_metadata,
)


class TestSerializeMetadata:
    """Test serialize_metadata function."""

    def test_serialize_simple_metadata(self):
        """Test serializing a simple metadata dictionary."""
        metadata = {"page": 1, "section": "intro"}
        result = serialize_metadata(metadata)

        assert result is not None
        assert isinstance(result, str)
        # Parse back to verify correctness
        parsed = json.loads(result)
        assert parsed == metadata
        # Check sorted keys for consistency
        assert result == '{"page": 1, "section": "intro"}'

    def test_serialize_nested_metadata(self):
        """Test serializing nested metadata dictionaries."""
        metadata = {
            "source": "/path/to/file.pdf",
            "position": {
                "page_number": 1,
                "coordinates": {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.6},
            },
            "parser": "deepdoc",
        }
        result = serialize_metadata(metadata)

        assert result is not None
        parsed = json.loads(result)
        assert parsed == metadata

    def test_serialize_none_returns_none(self):
        """Test that serializing None returns None."""
        result = serialize_metadata(None)
        assert result is None

    def test_serialize_empty_dict(self):
        """Test serializing an empty dictionary."""
        metadata = {}
        result = serialize_metadata(metadata)

        assert result is not None
        assert result == "{}"
        parsed = json.loads(result)
        assert parsed == {}

    def test_serialize_unicode_characters(self):
        """Test serializing metadata with Unicode characters."""
        metadata = {"title": "文档标题", "section": "章节一"}
        result = serialize_metadata(metadata)

        assert result is not None
        # Should preserve Unicode characters (ensure_ascii=False)
        assert "文档标题" in result
        assert "章节一" in result
        parsed = json.loads(result)
        assert parsed == metadata

    def test_serialize_sorted_keys(self):
        """Test that serialization sorts keys for consistency."""
        metadata = {"zebra": 1, "alpha": 2, "beta": 3}
        result = serialize_metadata(metadata)

        assert result is not None
        # Keys should be sorted
        assert result.startswith('{"alpha":')
        parsed = json.loads(result)
        assert parsed == metadata

    def test_serialize_complex_types(self):
        """Test serializing metadata with various types."""
        metadata = {
            "string": "text",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }
        result = serialize_metadata(metadata)

        assert result is not None
        parsed = json.loads(result)
        assert parsed == metadata


class TestDeserializeMetadata:
    """Test deserialize_metadata function."""

    def test_deserialize_simple_json(self):
        """Test deserializing a simple JSON string."""
        metadata_json = '{"page": 1, "section": "intro"}'
        result = deserialize_metadata(metadata_json)

        assert result == {"page": 1, "section": "intro"}

    def test_deserialize_nested_json(self):
        """Test deserializing nested JSON string."""
        metadata_json = '{"source": "/path/to/file.pdf", "position": {"page": 1}}'
        result = deserialize_metadata(metadata_json)

        assert result == {"source": "/path/to/file.pdf", "position": {"page": 1}}

    def test_deserialize_none_returns_none(self):
        """Test that deserializing None returns None."""
        result = deserialize_metadata(None)
        assert result is None

    def test_deserialize_pandas_na_returns_none(self):
        """Test that deserializing pandas NA returns None."""
        result = deserialize_metadata(pd.NA)
        assert result is None

    def test_deserialize_empty_string_returns_none(self):
        """Test that deserializing empty string returns None."""
        result = deserialize_metadata("")
        assert result is None

    def test_deserialize_whitespace_string_returns_none(self):
        """Test that deserializing whitespace-only string returns None."""
        result = deserialize_metadata("   ")
        assert result is None

    def test_deserialize_unicode_characters(self):
        """Test deserializing JSON with Unicode characters."""
        metadata_json = '{"title": "文档标题", "section": "章节一"}'
        result = deserialize_metadata(metadata_json)

        assert result == {"title": "文档标题", "section": "章节一"}

    def test_deserialize_invalid_json_returns_none(self):
        """Test that deserializing invalid JSON returns None without raising."""
        invalid_json = "{invalid json}"
        result = deserialize_metadata(invalid_json)

        # Should return None instead of raising exception
        assert result is None

    def test_deserialize_wrong_type_returns_none(self):
        """Test that deserializing non-string type returns None."""
        result = deserialize_metadata(123)
        assert result is None

        result = deserialize_metadata(["not", "a", "string"])
        assert result is None

    def test_deserialize_empty_dict(self):
        """Test deserializing empty JSON object."""
        result = deserialize_metadata("{}")
        assert result == {}


class TestMetadataUtilsRoundTrip:
    """Test round-trip serialization and deserialization."""

    def test_round_trip_simple_metadata(self):
        """Test that serialize → deserialize preserves metadata."""
        original = {"page": 1, "section": "intro", "source": "/path/to/file.pdf"}

        serialized = serialize_metadata(original)
        deserialized = deserialize_metadata(serialized)

        assert deserialized == original

    def test_round_trip_nested_metadata(self):
        """Test round-trip with nested structures."""
        original = {
            "source": "/path/to/file.pdf",
            "position": {
                "page_number": 1,
                "coordinates": {"x0": 0.1, "y0": 0.2},
            },
            "parser": {"name": "deepdoc", "version": "1.0.0"},
        }

        serialized = serialize_metadata(original)
        deserialized = deserialize_metadata(serialized)

        assert deserialized == original

    def test_round_trip_with_none_values(self):
        """Test round-trip with None values in metadata."""
        original = {"page": 1, "section": None, "anchor": "anchor1"}

        serialized = serialize_metadata(original)
        deserialized = deserialize_metadata(serialized)

        assert deserialized == original

    def test_round_trip_unicode(self):
        """Test round-trip with Unicode characters."""
        original = {
            "title": "文档标题",
            "content": "这是内容",
            "mixed": "Mixed: 文档 with English",
        }

        serialized = serialize_metadata(original)
        deserialized = deserialize_metadata(serialized)

        assert deserialized == original

    def test_round_trip_complex_types(self):
        """Test round-trip with various data types."""
        original = {
            "string": "text",
            "int": 42,
            "float": 3.14,
            "bool_true": True,
            "bool_false": False,
            "list": [1, 2, 3, "mixed"],
            "dict": {"nested": "value"},
        }

        serialized = serialize_metadata(original)
        deserialized = deserialize_metadata(serialized)

        assert deserialized == original

    def test_round_trip_none_metadata(self):
        """Test round-trip with None metadata."""
        original = None

        serialized = serialize_metadata(original)
        assert serialized is None

        deserialized = deserialize_metadata(serialized)
        assert deserialized is None
