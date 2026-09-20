"""
Unit tests for output filter module.
"""

from xagent.config import (
    get_tool_max_field_count,
    get_tool_max_output_length,
    get_tool_max_recursion_depth,
)
from xagent.core.tools.adapters.vibe.output_filter import (
    CIRCULAR_REFERENCE_MESSAGE,
    DEFAULT_TRUNCATION_MESSAGE,
    NESTED_TOO_DEEP_MESSAGE,
    TRUNCATED_DICT_TEMPLATE,
    TRUNCATED_FIELDS_MESSAGE,
    TRUNCATED_ITEMS_MESSAGE,
    TRUNCATED_ITEMS_TEMPLATE,
    OutputValueFilter,
)

# Get default values from config module
DEFAULT_MAX_OUTPUT_LENGTH = get_tool_max_output_length()
DEFAULT_MAX_FIELDS = get_tool_max_field_count()
DEFAULT_MAX_RECURSION = get_tool_max_recursion_depth()


def _create_filter(
    max_chars: int = DEFAULT_MAX_OUTPUT_LENGTH,
    max_fields: int = DEFAULT_MAX_FIELDS,
    max_recursion: int = DEFAULT_MAX_RECURSION,
) -> OutputValueFilter:
    """Helper to create filter with default values."""
    return OutputValueFilter(max_chars, max_fields, max_recursion)


def test_string_within_limit():
    """Test that strings within limit are not modified."""
    filter = _create_filter(max_chars=100)
    result = filter.filter("a" * 50, "test_tool")
    assert len(result) == 50
    assert result == "a" * 50


def test_string_exceeds_limit():
    """Test that strings exceeding limit are truncated."""
    filter = _create_filter(max_chars=100)
    result = filter.filter("a" * 200, "test_tool")
    assert len(result) == 100 + len(DEFAULT_TRUNCATION_MESSAGE)
    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)
    assert result.startswith("a" * 100)


def test_dict_preservation():
    """Test that dict structure is preserved."""
    filter = _create_filter(max_chars=50)
    data = {"short": "ok", "long": "a" * 100, "nested": {"value": "b" * 100}}
    result = filter.filter(data, "test_tool")
    assert result["short"] == "ok"
    # After truncation, length is max_chars + truncation message length
    assert len(result["long"]) == 50 + len(DEFAULT_TRUNCATION_MESSAGE)
    assert len(result["nested"]["value"]) == 50 + len(DEFAULT_TRUNCATION_MESSAGE)


def test_dict_with_non_string_values():
    """Test that dict with non-string values is handled correctly."""
    filter = _create_filter(max_chars=50)
    data = {
        "number": 42,
        "boolean": True,
        "none": None,
        "list": [1, 2, 3],
        "long_string": "a" * 100,
    }
    result = filter.filter(data, "test_tool")
    # Primitive types are preserved (better design)
    assert result["number"] == 42
    assert result["boolean"] is True
    assert result["none"] is None
    # List elements are also preserved
    assert result["list"] == [1, 2, 3]
    assert len(result["long_string"]) == 50 + len(DEFAULT_TRUNCATION_MESSAGE)


def test_list_filtering():
    """Test that list items are filtered."""
    filter = _create_filter(max_chars=50)
    data = ["short", "a" * 100, {"key": "b" * 100}]
    result = filter.filter(data, "test_tool")
    assert result[0] == "short"
    # After truncation, length is max_chars + truncation message length
    assert len(result[1]) == 50 + len(DEFAULT_TRUNCATION_MESSAGE)
    assert len(result[2]["key"]) == 50 + len(DEFAULT_TRUNCATION_MESSAGE)


def test_none_passthrough():
    """Test that None values pass through."""
    filter = _create_filter()
    assert filter.filter(None, "test_tool") is None


def test_empty_string():
    """Test that empty strings pass through."""
    filter = _create_filter(max_chars=100)
    result = filter.filter("", "test_tool")
    assert result == ""


def test_exact_limit():
    """Test that strings at exact limit are not modified."""
    filter = _create_filter(max_chars=100)
    result = filter.filter("a" * 100, "test_tool")
    assert len(result) == 100
    assert not result.endswith("[TRUNCATED]")


def test_default_limit():
    """Test default limit is 50K characters."""
    filter = _create_filter()
    assert filter.max_chars == DEFAULT_MAX_OUTPUT_LENGTH
    assert DEFAULT_MAX_OUTPUT_LENGTH == 50 * 1024


def test_unicode_string():
    """Test that unicode strings are handled correctly."""
    filter = _create_filter(max_chars=20)
    result = filter.filter("你好世界" * 10, "test_tool")
    # Each Chinese character is counted as 1 character
    assert len(result) <= 20 + len(DEFAULT_TRUNCATION_MESSAGE)


def test_nested_structures():
    """Test deeply nested structures."""
    filter = _create_filter(max_chars=10)
    data = {"level1": {"level2": {"level3": {"value": "a" * 100}}}}
    result = filter.filter(data, "test_tool")
    # After truncation, length is max_chars + truncation message length
    assert len(result["level1"]["level2"]["level3"]["value"]) == 10 + len(
        DEFAULT_TRUNCATION_MESSAGE
    )


def test_list_of_dicts():
    """Test list of dictionaries."""
    filter = _create_filter(max_chars=20)
    data = [
        {"name": "short", "value": "ok"},
        {"name": "long", "value": "a" * 100},
    ]
    result = filter.filter(data, "test_tool")
    assert result[0]["name"] == "short"
    assert result[0]["value"] == "ok"
    # After truncation, length is max_chars + truncation message length
    assert len(result[1]["value"]) == 20 + len(DEFAULT_TRUNCATION_MESSAGE)


def test_number_conversion():
    """Test that numbers are preserved (not converted to strings)."""
    filter = _create_filter(max_chars=5)
    result = filter.filter(1234567890, "test_tool")
    # Numbers are preserved as-is (no conversion)
    assert result == 1234567890
    assert isinstance(result, int)


def test_boolean_conversion():
    """Test that booleans are preserved (not converted to strings)."""
    filter = _create_filter(max_chars=10)
    result = filter.filter(True, "test_tool")
    # Booleans are preserved as-is
    assert result is True


def test_zero_max_chars():
    """Test edge case of zero max_chars."""
    filter = _create_filter(max_chars=0)
    result = filter.filter("a" * 100, "test_tool")
    assert result == DEFAULT_TRUNCATION_MESSAGE


def test_small_limit():
    """Test very small limit."""
    filter = _create_filter(max_chars=5)
    result = filter.filter("a" * 100, "test_tool")
    assert result.startswith("a" * 5)
    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)


def test_env_variable_default():
    """Test that environment variable is used for config values."""
    import os

    from xagent.config import (
        MIN_TOOL_MAX_OUTPUT_LENGTH,
        TOOL_MAX_OUTPUT_LENGTH,
        get_tool_max_output_length,
    )

    # Save original value
    original_value = os.getenv(TOOL_MAX_OUTPUT_LENGTH)

    try:
        # Test with valid env var
        os.environ[TOOL_MAX_OUTPUT_LENGTH] = "100000"
        assert get_tool_max_output_length() == 100000

        for too_small in ("18", "0", "-1"):
            os.environ[TOOL_MAX_OUTPUT_LENGTH] = too_small
            assert get_tool_max_output_length() == MIN_TOOL_MAX_OUTPUT_LENGTH

        # Test with invalid env var (should fallback to default)
        os.environ[TOOL_MAX_OUTPUT_LENGTH] = "invalid"
        result = get_tool_max_output_length()
        assert result == 50 * 1024  # Fallback to default

        # Test without env var (should use default)
        os.environ.pop(TOOL_MAX_OUTPUT_LENGTH, None)
        assert get_tool_max_output_length() == 50 * 1024
    finally:
        # Restore original value
        if original_value is None:
            os.environ.pop(TOOL_MAX_OUTPUT_LENGTH, None)
        else:
            os.environ[TOOL_MAX_OUTPUT_LENGTH] = original_value


def test_filter_uses_default_when_none():
    """Test that filter uses default value from config."""
    from xagent.config import get_tool_max_output_length

    filter = _create_filter()  # Uses defaults from config
    assert filter.max_chars == get_tool_max_output_length()
    assert filter.max_chars == 50 * 1024


def test_bytes_handling():
    """Test that bytes are decoded correctly."""
    filter = _create_filter(max_chars=11)
    data = b"hello world, this is a test"
    result = filter.filter(data, "test_tool")
    # Bytes should be decoded to string and truncated
    assert isinstance(result, str)
    # Decoded string is 27 chars, max_chars=11, so truncated to 11 + message
    assert len(result) == 11 + len(DEFAULT_TRUNCATION_MESSAGE)
    assert result.startswith("hello world")
    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)


def test_tuple_handling():
    """Test that tuples are handled correctly."""
    filter = _create_filter(max_chars=50)
    data = ("a" * 100, "b" * 100, "c" * 100)
    result = filter.filter(data, "test_tool")
    # Tuple should be processed and returned as tuple (not truncated)
    assert isinstance(result, tuple)
    assert len(result) == 3
    # Each element should be truncated (truncation message adds to length)
    assert all("a" in s and "TRUNCATED" in s for s in result)


def test_tuple_truncated_to_list():
    """Test that truncated tuples are converted to list."""
    filter = _create_filter(max_chars=10, max_fields=2)
    data = tuple("item" + str(i) for i in range(10))  # 10 items
    result = filter.filter(data, "test_tool")
    # Truncated tuple should return as list
    assert isinstance(result, list)
    assert len(result) == 3  # 2 items + 1 truncation message


def test_set_handling():
    """Test that sets are handled correctly."""
    filter = _create_filter(max_fields=3)
    data = {10, 20, 30, 40, 50}
    result = filter.filter(data, "test_tool")
    # Set should be converted to sorted list with truncation
    assert isinstance(result, list)
    # First 3 items (sorted) + truncation message
    assert len(result) == 4
    assert TRUNCATED_ITEMS_MESSAGE in result[3]
    assert result[0] in {10, 20, 30, 40, 50}


def test_nested_tuple():
    """Test nested tuple with strings."""
    filter = _create_filter(max_chars=20)
    data = {"key": ("a" * 100, "b" * 100)}
    result = filter.filter(data, "test_tool")
    assert isinstance(result, dict)
    assert isinstance(result["key"], tuple)
    # Strings in tuple should be truncated
    assert all("TRUNCATED" in s for s in result["key"])


def test_max_recursion_depth():
    """Test that deeply nested structures are handled correctly."""
    filter = _create_filter(max_recursion=3)

    # Create a structure with depth 5 (exceeds max_recursion=3)
    data = {"l1": {"l2": {"l3": {"l4": {"l5": "deep"}}}}}
    result = filter.filter(data, "test_tool")

    # Should truncate at depth limit
    assert result["l1"]["l2"]["l3"]["l4"] == NESTED_TOO_DEEP_MESSAGE
    # l5 should not be processed
    assert "l5" not in str(result)


def test_circular_reference_detection():
    """Test that circular references are detected and handled."""
    filter = _create_filter()

    # Create a circular reference
    data: dict = {}
    data["key"] = "value"
    data["self"] = data  # Circular reference (type: ignore)

    result = filter.filter(data, "test_tool")

    # Should detect circular reference and return special message
    assert result["self"] == CIRCULAR_REFERENCE_MESSAGE
    assert result["key"] == "value"


def test_circular_reference_in_list():
    """Test circular reference in list."""
    filter = _create_filter()

    # Create a circular reference in list
    data: list = [1, 2, 3]
    data.append(data)  # Circular reference (type: ignore)

    result = filter.filter(data, "test_tool")

    # Should detect circular reference
    assert CIRCULAR_REFERENCE_MESSAGE in str(result)
    # First elements should be preserved
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_max_recursion_with_large_max():
    """Test that larger max_recursion allows deeper nesting."""
    filter = _create_filter(max_recursion=10)

    # Create a structure with depth 5
    data = {"l1": {"l2": {"l3": {"l4": {"l5": "deep"}}}}}
    result = filter.filter(data, "test_tool")

    # Should process fully since max_recursion=10 > depth=5
    assert result["l1"]["l2"]["l3"]["l4"]["l5"] == "deep"


def test_max_recursion_at_exact_limit():
    """Test that depth exactly at limit is processed correctly."""
    filter = _create_filter(max_recursion=3)

    # Create a structure with depth exactly 3
    data = {"l1": {"l2": {"l3": "exact_limit"}}}
    result = filter.filter(data, "test_tool")

    # Should process fully since depth=3 == max_recursion=3
    # (check happens at depth > max_recursion, so depth=3 is allowed)
    assert result["l1"]["l2"]["l3"] == "exact_limit"


def test_max_recursion_at_limit_plus_one():
    """Test that depth at limit+1 triggers truncation."""
    filter = _create_filter(max_recursion=3)

    # Create a structure with depth exactly 4 (limit+1)
    data = {"l1": {"l2": {"l3": {"l4": "too_deep"}}}}
    result = filter.filter(data, "test_tool")

    # Should truncate at l4 since depth=4 > max_recursion=3
    assert result["l1"]["l2"]["l3"]["l4"] == NESTED_TOO_DEEP_MESSAGE


def test_dict_max_fields_truncation():
    """Test that dict with too many fields uses TRUNCATED_FIELDS_MESSAGE."""
    filter = _create_filter(max_fields=3)
    data = {"field1": "a", "field2": "b", "field3": "c", "field4": "d", "field5": "e"}
    result = filter.filter(data, "test_tool")
    # Should have 4 items: 3 fields + 1 truncation message
    assert len(result) == 4
    assert "field1" in result
    assert "field2" in result
    assert "field3" in result
    expected_key = TRUNCATED_DICT_TEMPLATE.format(count=2)
    assert expected_key in result
    assert result[expected_key] == TRUNCATED_FIELDS_MESSAGE


def test_list_max_fields_truncation():
    """Test that list with too many items uses TRUNCATED_ITEMS_TEMPLATE."""
    filter = _create_filter(max_fields=3)
    data = ["item1", "item2", "item3", "item4", "item5"]
    result = filter.filter(data, "test_tool")
    # Should have 4 items: 3 items + 1 truncation message
    assert len(result) == 4
    assert result[0] == "item1"
    assert result[1] == "item2"
    assert result[2] == "item3"
    assert TRUNCATED_ITEMS_TEMPLATE.format(count=2) == result[3]


def test_tuple_max_fields_truncation():
    """Test that tuple with too many items is converted to list with truncation."""
    filter = _create_filter(max_fields=2)
    data = ("item1", "item2", "item3", "item4")
    result = filter.filter(data, "test_tool")
    # Truncated tuple should return as list
    assert isinstance(result, list)
    assert len(result) == 3  # 2 items + 1 truncation message
    assert result[0] == "item1"
    assert result[1] == "item2"
    assert result[2] == TRUNCATED_ITEMS_TEMPLATE.format(count=2)
