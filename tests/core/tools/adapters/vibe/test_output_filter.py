"""
Unit tests for output filter module.
"""

import json

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

    from xagent.config import TOOL_MAX_OUTPUT_LENGTH, get_tool_max_output_length

    # Save original value
    original_value = os.getenv(TOOL_MAX_OUTPUT_LENGTH)

    try:
        # Test with valid env var
        os.environ[TOOL_MAX_OUTPUT_LENGTH] = "100000"
        assert get_tool_max_output_length() == 100000

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


def _large_json_payload(num_rows: int = 500) -> str:
    """A realistic large JSON payload: a status envelope wrapping a data list."""
    data = json.dumps(
        {
            "status": "success",
            "cursor": "opaque-pagination-cursor-abc123",
            "total_count": num_rows,
            "data": [
                {
                    "id": f"ad_{i:05d}",
                    "name": f"Ad campaign number {i}",
                    "impressions": i * 137,
                    "clicks": i * 3,
                    "spend": round(i * 0.42, 2),
                    "notes": "performance summary " * 5,
                }
                for i in range(num_rows)
            ],
        },
        ensure_ascii=False,
    )
    return data


def test_json_truncation_keeps_result_parseable():
    """A large JSON string exceeding max_chars is trimmed, not char-sliced."""
    payload = _large_json_payload()
    filter = _create_filter(max_chars=2000)

    result = filter.filter(payload, "test_tool")

    assert isinstance(result, str)
    assert len(result) <= 2000
    parsed = json.loads(result)  # must not raise - still valid JSON
    assert parsed["status"] == "success"


def test_json_truncation_drops_trailing_items_and_preserves_metadata():
    """Trimming removes whole trailing rows from the largest list field and
    keeps unrelated fields (pagination cursor, status) untouched."""
    payload = _large_json_payload()
    original = json.loads(payload)
    filter = _create_filter(max_chars=3000)

    result = filter.filter(payload, "test_tool")
    parsed = json.loads(result)

    assert parsed["status"] == original["status"]
    assert parsed["cursor"] == original["cursor"]
    assert parsed["total_count"] == original["total_count"]
    assert len(parsed["data"]) < len(original["data"])
    # Kept rows are untouched, real dicts (not raw sliced fragments)
    for row in parsed["data"][:-1]:
        assert isinstance(row, dict)
        assert "id" in row
    # Last entry communicates how many rows were dropped
    assert TRUNCATED_ITEMS_MESSAGE in parsed["data"][-1]


def test_json_truncation_falls_back_for_invalid_json():
    """Non-JSON strings still use the raw character-slice fallback."""
    filter = _create_filter(max_chars=50)
    result = filter.filter("not json " * 50, "test_tool")

    assert result.startswith("not json ")
    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)


def test_json_truncation_falls_back_when_no_list_present():
    """JSON with no list field to trim falls back to raw character slicing."""
    payload = json.dumps({"message": "x" * 500, "status": "success"})
    filter = _create_filter(max_chars=50)

    result = filter.filter(payload, "test_tool")

    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)


def test_json_truncation_does_not_drop_items_when_compacting_alone_fits():
    """A pretty-printed (indented) JSON string can exceed max_chars purely from
    whitespace; if compacting it (keeping every item) already fits, no items
    should be dropped and no truncation marker should be added."""
    data = {"status": "ok", "data": list(range(100))}
    raw = json.dumps(data, indent=4)
    compact_full = json.dumps(data, ensure_ascii=False)
    filter = _create_filter(max_chars=len(compact_full) + 5)

    result = filter.filter(raw, "test_tool")
    parsed = json.loads(result)

    assert len(parsed["data"]) == 100
    assert parsed["data"] == list(range(100))


def test_json_truncation_root_level_list():
    """A bare JSON array (no wrapping object) is trimmed in place."""
    payload = json.dumps([{"id": i, "value": "x" * 50} for i in range(200)])
    filter = _create_filter(max_chars=1000)

    result = filter.filter(payload, "test_tool")
    parsed = json.loads(result)

    assert isinstance(parsed, list)
    assert len(parsed) < 200
    assert TRUNCATED_ITEMS_MESSAGE in parsed[-1]


def _deeply_nested_list(depth: int) -> str:
    """A JSON array nested `depth` levels deep: [[[...["leaf"]...]]]."""
    obj: list = ["leaf"]
    for _ in range(depth):
        obj = [obj]
    return json.dumps(obj)


def test_json_truncation_falls_back_gracefully_on_deep_nesting():
    """Nesting deep enough to risk a RecursionError while walking/re-serializing
    the parsed structure must fall back to raw slicing, not crash the filter."""
    payload = _deeply_nested_list(depth=3000)
    filter = _create_filter(max_chars=10)

    result = filter.filter(payload, "test_tool")

    # Must not raise, and must still fall back to the safe raw-slice path.
    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)


def test_find_largest_list_does_not_reserialize_subtrees():
    """`_find_largest_list` must size candidate lists without calling
    `json.dumps` on them - re-serializing whole subtrees at every nesting
    level is what made the previous implementation quadratic (or worse) for
    JSON with several levels of nesting, which real hierarchical API
    responses can have. Asserting the call count directly (rather than
    comparing wall-clock timings across input sizes) keeps this deterministic
    instead of depending on CI machine speed/noise."""
    from unittest import mock

    nested = {"a": [1, 2], "b": {"c": list(range(50)), "d": [{"e": [1]}]}}

    with mock.patch("json.dumps") as mocked_dumps:
        result = _create_filter()._find_largest_list(nested)

    mocked_dumps.assert_not_called()
    assert result == list(range(50))


def test_json_truncation_ignores_single_item_envelope_wrapper():
    """A one-item envelope wrapper around the real data list must not be
    picked as the trim target - by bottom-up size it always looks "largest"
    since its estimated size is the wrapped list's size plus more, so without
    a >= 2 items floor the whole real payload gets discarded instead of the
    wrapper. Reproduces the exact shape called out in review: a single-page
    "results" wrapper around a real "items" array."""
    data = {
        "results": [{"items": [{"id": i, "name": f"row-{i}"} for i in range(100)]}],
        "status": "ok",
    }
    payload = json.dumps(data)
    filter = _create_filter(max_chars=300)

    result = filter.filter(payload, "test_tool")
    parsed = json.loads(result)

    assert parsed["status"] == "ok"
    kept = parsed["results"][0]["items"]
    assert len(kept) > 0, "wrapper should not swallow all real data"
    assert kept[0] == {"id": 0, "name": "row-0"}


def test_json_truncation_falls_back_when_only_marker_would_fit():
    """If the chosen list's items are individually too large for even one to
    fit, dropping all of them (keeping only the truncation marker) discards
    100% of the real content - strictly less real information than the old
    raw-slice fallback would keep. Must fall back instead of returning a
    valid-but-empty JSON envelope."""
    data = {"data": ["x" * 2000, "y" * 2000, "z" * 2000]}
    payload = json.dumps(data)
    filter = _create_filter(max_chars=100)

    result = filter.filter(payload, "test_tool")

    # Falls back to the raw-slice path (real, if malformed, content) rather
    # than a marker-only, content-free JSON list.
    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)
    assert "xxx" in result


def test_json_truncation_falls_back_when_non_list_field_dominates():
    """When a huge non-list field (not the list itself) is what pushes the
    payload over budget, trimming the list to zero items still won't fit -
    must fall back to raw slicing (this specific "best_serialized stays None"
    branch had no direct test coverage before)."""
    data = {"message": "x" * 5000, "data": [1, 2, 3]}
    payload = json.dumps(data)
    filter = _create_filter(max_chars=300)

    result = filter.filter(payload, "test_tool")

    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)
    assert result.startswith('{"message"')


def test_json_truncation_single_item_list_is_never_the_trim_target():
    """A list with exactly one item can never usefully be trimmed (it would
    have to go straight to marker-only), so it must never be selected as the
    largest list - even when it's the only list in the payload."""
    data = {"data": ["x" * 5000]}
    payload = json.dumps(data)
    filter = _create_filter(max_chars=200)

    result = filter.filter(payload, "test_tool")

    # No list qualifies as a trim target, so this falls back to raw slicing.
    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)


def test_json_truncation_preserves_non_ascii_content():
    """ensure_ascii=False must be honored end to end: non-ASCII characters in
    kept items stay as literal UTF-8 text, not \\uXXXX escapes."""
    data = {"data": [{"name": f"名前-{i}", "note": "説明" * 5} for i in range(200)]}
    payload = json.dumps(data, ensure_ascii=False)
    filter = _create_filter(max_chars=2000)

    result = filter.filter(payload, "test_tool")
    parsed = json.loads(result)

    assert len(parsed["data"]) < 200
    assert parsed["data"][0]["name"] == "名前-0"
    assert "\\u" not in result


def test_json_truncation_exact_boundary_is_not_truncated():
    """A JSON string whose length is exactly max_chars must be returned as-is
    (no truncation, no marker) - the filter.filter entry point already
    guards `len(value) <= self.max_chars`, but pin it through the JSON path
    too since it re-serializes rather than returning the input verbatim."""
    payload = json.dumps({"data": [1, 2, 3]})
    filter = _create_filter(max_chars=len(payload))

    result = filter.filter(payload, "test_tool")

    assert result == payload


def test_json_truncation_falls_back_on_out_of_range_float():
    """A payload containing an out-of-range number (json.loads happily parses
    `1e400` to float('inf')) must not be re-serialized with the non-RFC8259
    `Infinity` token; allow_nan=False turns that into a ValueError that's
    caught, falling back to raw slicing instead. Uses a bounded fixture large
    enough for max_chars to still land past the structured (not just raw)
    path, so a mutant that dropped allow_nan=False and let Infinity leak
    through the structured serializer would actually be caught here."""
    data = {"data": list(range(100)) + [1e400]}
    payload = json.dumps(data, indent=2)
    filter = _create_filter(max_chars=500, max_fields=200)

    result = filter.filter(payload, "test_tool")

    assert "Infinity" not in result
    assert "NaN" not in result


def test_json_truncation_ignores_multi_item_envelope_wrapper():
    """Selecting the trim target by item count (not total size) means an
    outer wrapper with 2+ elements, each wrapping its own real nested list,
    no longer "wins" over either nested list just by containing more bytes.
    Reproduces a common paginated-envelope shape: two result pages, each
    with its own 100-row "items" array."""
    data = {
        "results": [
            {"items": [{"id": i, "name": f"row-{i}"} for i in range(100)]},
            {"items": [{"id": i, "name": f"row-{i}"} for i in range(100, 200)]},
        ],
        "status": "ok",
    }
    payload = json.dumps(data)
    filter = _create_filter(max_chars=3100)

    result = filter.filter(payload, "test_tool")
    parsed = json.loads(result)

    assert len(result) <= 3100
    assert parsed["status"] == "ok"
    assert len(parsed["results"]) == 2
    # Neither page should be entirely discarded, and at least one page must
    # actually have been trimmed (not just re-compacted) - a regression that
    # left both pages fully intact, or that dropped a page wholesale to a
    # bare marker, must fail this.
    kept_counts = []
    for page in parsed["results"]:
        assert isinstance(page, dict), "a whole page must not be dropped wholesale"
        assert 0 < len(page["items"]) <= 100
        kept_counts.append(len(page["items"]))
    assert min(kept_counts) < 100, "expected real trimming, not just compaction"


def test_json_truncation_prefers_primary_records_list_over_nested_tag_arrays():
    """A regression found via review: excluding every list that contains a
    qualifying nested list (an earlier attempt at fixing the envelope-wrapper
    bug) incorrectly excludes an ordinary records list whenever each record
    carries its own small nested array (e.g. per-record "tags") - the exact
    opposite failure, where the *real* data-bearing list gets skipped in
    favor of one lone record's tiny tags array. Item-count-based selection
    (this list has far more items than any single "tags" list) avoids both
    failure modes without needing to track ancestor/descendant relationships
    at all."""
    data = {"data": [{"id": i, "tags": ["a", "b", "c"]} for i in range(180)]}
    payload = json.dumps(data)
    filter = _create_filter(max_chars=2000, max_fields=1000)

    result = filter.filter(payload, "test_tool")
    parsed = json.loads(result)

    assert len(result) <= 2000
    assert 0 < len(parsed["data"]) < 180
    for record in parsed["data"][:-1]:
        assert isinstance(record, dict)
        assert record["tags"] == ["a", "b", "c"]


def test_json_truncation_enforces_max_fields_like_native_objects():
    """A JSON string that needs truncating (exceeds max_chars) must respect
    max_fields the same way the native dict/list path does - previously the
    JSON-aware path only consulted max_chars, so a string pre-serialized by
    a tool could return far more items than the same data returned as a
    native object would."""
    data = {
        "status": "ok",
        "data": [f"item-number-{i:06d}-padding-padding" for i in range(3000)],
    }
    payload = json.dumps(data)
    filter = _create_filter(max_chars=60000, max_fields=1000)

    result_from_string = filter.filter(payload, "test_tool")
    result_from_native = filter.filter(data, "test_tool")

    assert len(json.loads(result_from_string)["data"]) == len(
        result_from_native["data"]
    )


def test_json_truncation_survives_unrelated_nan_elsewhere_in_payload():
    """A non-finite float (NaN/Infinity) that ends up trimmed away by the
    binary search must not abort the whole JSON-aware attempt - only a
    candidate that would actually *keep* the offending value should be
    treated as not fitting."""
    data = {"data": [float("nan") if i == 35 else i for i in range(40)]}
    payload = json.dumps(data)
    filter = _create_filter(max_chars=100)

    result = filter.filter(payload, "test_tool")
    parsed = json.loads(result)

    assert "nan" not in [str(x).lower() for x in parsed["data"]]
    assert len(parsed["data"]) > 1  # a real trim happened, not just the marker


def test_json_truncation_compacts_pretty_printed_payload_with_no_list():
    """A pretty-printed (indented) JSON object with no list at all must still
    get the cheap "just compact it" win when compacting alone fits - falling
    straight through to a corrupting raw slice here would defeat the point
    of this whole feature for exactly the payloads that don't happen to
    contain a list."""
    obj = {f"field_{i}": f"value_{i}" for i in range(20)}
    pretty = json.dumps(obj, indent=2)
    compact = json.dumps(obj)
    filter = _create_filter(max_chars=(len(pretty) + len(compact)) // 2)

    result = filter.filter(pretty, "test_tool")
    parsed = json.loads(result)

    assert parsed == obj


def test_json_truncation_uses_tight_separators():
    """The "does it already fit" and binary-search re-serializations must use
    minimal separators (no space after ','/':' ), not just "no indentation" -
    otherwise wasted separator bytes can force dropping items that a truly
    compact encoding wouldn't have needed to drop."""
    data = {"data": [{"v": i} for i in range(50)]}
    pretty = json.dumps(data, indent=2)
    minimal_full = json.dumps(data, separators=(",", ":"))
    filter = _create_filter(max_chars=len(minimal_full))

    result = filter.filter(pretty, "test_tool")
    parsed = json.loads(result)

    assert len(parsed["data"]) == 50, "true-minimal encoding should fit everything"


def test_json_truncation_skips_pathologically_large_input():
    """Above the structured-truncation size safety valve, the JSON-aware path
    must bail out before ever parsing the input, rather than pay the
    repeated re-serialization cost a large structure would incur. Asserting
    that `json.loads` is never called (with the threshold patched down, so
    the test doesn't need a real multi-megabyte payload) is deterministic,
    unlike a wall-clock timing assertion."""
    from unittest import mock

    data = {"data": list(range(50))}
    payload = json.dumps(data)
    filter = _create_filter(max_chars=20)
    assert 10 < len(payload)  # sanity: bigger than max_chars and the patched
    # threshold below, so _try_json_truncate is actually entered.

    original_limit = OutputValueFilter._MAX_STRUCTURED_TRUNCATE_INPUT_CHARS
    OutputValueFilter._MAX_STRUCTURED_TRUNCATE_INPUT_CHARS = 10
    try:
        with mock.patch("json.loads") as mocked_loads:
            result = filter.filter(payload, "test_tool")
    finally:
        OutputValueFilter._MAX_STRUCTURED_TRUNCATE_INPUT_CHARS = original_limit

    mocked_loads.assert_not_called()
    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)


def test_json_truncation_reports_true_count_after_field_cap_and_char_trim():
    """A regression found via review: when max_fields caps a list first (say
    to 10 items + a marker) and max_chars *then* trims that same
    already-capped list further (down to, say, 7 items), the final marker
    must report how many of the ORIGINAL items are missing in total (50 - 7
    = 43), not how many were dropped relative to the already-capped
    intermediate list (10 - 7 = 3) - the latter drastically understates real
    data loss."""
    total_items = 50
    data = {"data": [f"item-{i:04d}" for i in range(total_items)]}
    payload = json.dumps(data)
    # Chosen so the field-capped form (10 items + marker) doesn't fit, but a
    # further char-budget trim does.
    filter = _create_filter(max_chars=140, max_fields=10)

    result = filter.filter(payload, "test_tool")
    parsed = json.loads(result)

    kept = [x for x in parsed["data"] if not x.startswith("...")]
    marker = next(x for x in parsed["data"] if x.startswith("..."))
    assert 0 < len(kept) < 10  # char budget trimmed below the max_fields cap
    assert TRUNCATED_ITEMS_TEMPLATE.format(count=total_items - len(kept)) == marker


def test_json_truncation_survives_oversized_integer():
    """A JSON integer literal long enough to hit Python's int-to-str
    conversion digit limit makes json.loads raise a plain ValueError (not a
    json.JSONDecodeError). This must be caught and fall back to raw slicing
    like any other unparsable/unrepresentable input, not propagate and
    crash the caller."""
    payload = '{"n":' + "1" * 4301 + "}"
    filter = _create_filter(max_chars=10)

    result = filter.filter(payload, "test_tool")  # must not raise

    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)
