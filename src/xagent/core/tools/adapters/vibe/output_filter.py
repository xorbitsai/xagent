"""
Tool Output Value Filtering Module

Provides multi-layered output limiting for all tools to prevent excessive output.

This module implements a three-pronged approach to control tool output size:
1. Per-string length limit: Truncates individual string values
2. Field count limit: Limits the number of items in dicts/lists
3. Recursion depth limit: Prevents excessively deep nesting

Rather than calculating total output size (which would be expensive), these
limits work together to provide reasonable protection while maintaining good
performance. For token safety, the combination of these limits is sufficient
for most real-world scenarios.
"""

import json
import logging
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "OutputValueFilter",
    "DEFAULT_TRUNCATION_MESSAGE",
    "NESTED_TOO_DEEP_MESSAGE",
    "CIRCULAR_REFERENCE_MESSAGE",
    "TRUNCATED_FIELDS_MESSAGE",
    "TRUNCATED_ITEMS_MESSAGE",
    "TRUNCATED_DICT_TEMPLATE",
    "TRUNCATED_ITEMS_TEMPLATE",
]


# Message constants for output filtering
DEFAULT_TRUNCATION_MESSAGE = "\n\n[OUTPUT TRUNCATED: exceeded maximum length]"
NESTED_TOO_DEEP_MESSAGE = "[... nested too deep ...]"
CIRCULAR_REFERENCE_MESSAGE = "[... circular reference ...]"
TRUNCATED_FIELDS_MESSAGE = "[truncated]"
TRUNCATED_ITEMS_MESSAGE = " [truncated]"
TRUNCATED_DICT_TEMPLATE = "... and {count} more keys"
TRUNCATED_ITEMS_TEMPLATE = "... and {count} more items" + TRUNCATED_ITEMS_MESSAGE


class OutputValueFilter:
    """Filter and truncate tool return values using multi-layered limits.

    This class applies three types of limits to control tool output size:
    - Per-string length (max_chars): Limits individual string values
    - Field/item count (max_fields): Limits collection cardinality
    - Recursion depth (max_recursion): Prevents deep nesting

    Note: This does NOT enforce a hard total output size limit. The combination
    of these limits provides practical protection for token usage without the
    performance cost of calculating total serialized size.
    """

    def __init__(self, max_chars: int, max_fields: int, max_recursion: int):
        """
        Initialize output filter.

        Args:
            max_chars: Maximum length per string value (not total output).
            max_fields: Maximum number of fields/items in dicts/lists.
            max_recursion: Maximum nesting depth for recursive structures.
        """
        self.max_chars = max_chars
        self.max_fields = max_fields
        self.max_recursion = max_recursion

    def filter(self, value: Any, tool_name: str = "unknown") -> Any:
        """
        Filter return value based on character limit.

        Args:
            value: Return value to filter
            tool_name: Name of the tool (for logging)

        Returns:
            Filtered value (may be truncated)
        """
        return self._filter_with_depth(value, tool_name, depth=0, memo_set=None)

    def _filter_with_depth(
        self, value: Any, tool_name: str, depth: int, memo_set: set | None
    ) -> Any:
        """
        Internal filter method with recursion depth and circular reference tracking.

        Args:
            value: Return value to filter
            tool_name: Name of the tool (for logging)
            depth: Current recursion depth
            memo_set: Set of object ids to detect circular references

        Returns:
            Filtered value (may be truncated)
        """
        # Check recursion depth limit
        if depth > self.max_recursion:
            logger.warning(
                f"Tool '{tool_name}' output nested too deep (>{self.max_recursion} levels). "
                f"Truncating to prevent stack overflow."
            )
            return NESTED_TOO_DEEP_MESSAGE

        # Initialize memo_set for circular reference detection
        if memo_set is None:
            memo_set = set()

        if value is None:
            return None

        # Check for circular references (only for container types)
        if isinstance(value, (dict, list)):
            value_id = id(value)
            if value_id in memo_set:
                logger.warning(
                    f"Tool '{tool_name}' output contains circular reference. "
                    f"Breaking the cycle to prevent infinite recursion."
                )
                return CIRCULAR_REFERENCE_MESSAGE
            memo_set = memo_set | {value_id}

        # Handle strings
        if isinstance(value, str):
            return self._filter_string(value, tool_name)

        # Handle dicts - recursively filter each string value
        elif isinstance(value, dict):
            dist_result = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= self.max_fields:
                    dist_result[
                        TRUNCATED_DICT_TEMPLATE.format(count=len(value) - i)
                    ] = TRUNCATED_FIELDS_MESSAGE
                    break
                dist_result[k] = self._filter_with_depth(
                    v, tool_name, depth + 1, memo_set
                )
            return dist_result

        # Handle lists - recursively filter each element
        elif isinstance(value, list):
            list_result = []
            for i, item in enumerate(value):
                if i >= self.max_fields:
                    list_result.append(
                        TRUNCATED_ITEMS_TEMPLATE.format(count=len(value) - i)
                    )
                    break
                list_result.append(
                    self._filter_with_depth(item, tool_name, depth + 1, memo_set)
                )
            return list_result

        # Handle tuples - recursively filter each element, convert to list if truncated
        elif isinstance(value, tuple):
            tuple_result = []
            for i, item in enumerate(value):
                if i >= self.max_fields:
                    tuple_result.append(
                        TRUNCATED_ITEMS_TEMPLATE.format(count=len(value) - i)
                    )
                    break
                tuple_result.append(
                    self._filter_with_depth(item, tool_name, depth + 1, memo_set)
                )
            # Return as list if truncated, otherwise as tuple
            if len(tuple_result) < len(value):
                return tuple_result  # Truncated, return as list
            return tuple(tuple_result)  # Not truncated, return as tuple

        # Handle sets - convert to sorted list for deterministic filtering
        elif isinstance(value, set):
            # Sort for deterministic order when truncating
            sorted_items = sorted(value, key=lambda x: str(x))
            set_result = []
            for i, item in enumerate(sorted_items):
                if i >= self.max_fields:
                    set_result.append(
                        TRUNCATED_ITEMS_TEMPLATE.format(count=len(value) - i)
                    )
                    break
                set_result.append(
                    self._filter_with_depth(item, tool_name, depth + 1, memo_set)
                )
            # Return as list since sets can't be reconstructed after filtering
            return set_result

        # Handle bytes - decode to string
        elif isinstance(value, bytes):
            str_value = value.decode("utf-8", errors="replace")
            return self._filter_string(str_value, tool_name)

        # Handle Pydantic models
        elif hasattr(value, "model_dump"):
            filtered_dict = self._filter_with_depth(
                value.model_dump(), tool_name, depth + 1, memo_set
            )
            try:
                return value.__class__(**filtered_dict)
            except (ValidationError, TypeError, ValueError) as e:
                # ValidationError: truncated value violates constraints (e.g., min_length)
                # TypeError: unexpected constructor arguments
                # ValueError: invalid value for constructor
                logger.warning(
                    f"Failed to reconstruct Pydantic model {value.__class__.__name__} "
                    f"after filtering (value may be truncated): {e}. "
                    f"Returning filtered dict instead."
                )
                return filtered_dict

        # Handle primitives (bool, int, float, etc.) - return as-is
        elif isinstance(value, (bool, int, float)):
            return value

        # Handle other types by converting to string (as last resort)
        else:
            str_value = str(value)
            return self._filter_string(str_value, tool_name)

    def _filter_string(self, value: str, tool_name: str) -> str:
        """
        Filter a string value.

        Args:
            value: String value to filter
            tool_name: Name of the tool (for logging)

        Returns:
            Filtered string value
        """
        if len(value) <= self.max_chars:
            return value

        structured = self._try_json_truncate(value, tool_name)
        if structured is not None:
            return structured

        truncated = value[: self.max_chars]
        result = truncated + DEFAULT_TRUNCATION_MESSAGE
        logger.info(
            f"Tool '{tool_name}' output truncated: "
            f"{len(value)} -> {len(result)} characters"
        )
        return result

    def _try_json_truncate(self, value: str, tool_name: str) -> str | None:
        """
        Truncate an oversized JSON string by dropping trailing items from its
        largest list, keeping the result valid JSON instead of slicing raw
        characters (which would corrupt the JSON syntax mid-token).

        Returns None (falling back to raw character slicing) when the value
        isn't parseable JSON, contains no list to trim, or can't be brought
        under max_chars by trimming alone.
        """
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError, RecursionError):
            return None

        if not isinstance(parsed, (dict, list)):
            return None

        target_list = self._find_largest_list(parsed)
        if not target_list:
            return None

        original_items = list(target_list)
        total = len(original_items)
        if total == 0:
            return None

        # A compact re-serialization (no pretty-print whitespace) of the
        # untouched list may already fit, even though the original (possibly
        # indented) string didn't - check that before dropping anything.
        full_serialized = json.dumps(parsed, ensure_ascii=False)
        if len(full_serialized) <= self.max_chars:
            logger.info(
                f"Tool '{tool_name}' output JSON-compacted (no items dropped): "
                f"{len(value)} -> {len(full_serialized)} characters"
            )
            return full_serialized

        # Binary search the largest prefix of the list (plus a marker for the
        # dropped remainder) whose re-serialized JSON fits within max_chars.
        # Every candidate here always carries the marker (remaining >= 1), so
        # length is monotonic in the prefix size - each additional kept item
        # only adds characters. (The candidate with the whole list and no
        # marker was already ruled out above, since dropping a roughly
        # constant-size marker for the last couple of items can *shrink* the
        # output, which would otherwise break that monotonicity.)
        lo, hi = 0, total - 1
        best_serialized: str | None = None
        best_kept = 0
        try:
            while lo <= hi:
                mid = (lo + hi) // 2
                remaining = total - mid
                candidate = original_items[:mid] + [
                    TRUNCATED_ITEMS_TEMPLATE.format(count=remaining)
                ]
                target_list[:] = candidate
                serialized = json.dumps(parsed, ensure_ascii=False)
                if len(serialized) <= self.max_chars:
                    best_serialized = serialized
                    best_kept = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
        finally:
            target_list[:] = original_items

        if best_serialized is None:
            return None

        logger.info(
            f"Tool '{tool_name}' output JSON-truncated: kept {best_kept}/{total} "
            f"list items ({len(value)} -> {len(best_serialized)} characters)"
        )
        return best_serialized

    @staticmethod
    def _find_largest_list(obj: Any) -> list | None:
        """Find the largest (by serialized size) list nested anywhere in obj."""
        best: list | None = None
        best_size = -1

        def consider(candidate: list) -> None:
            nonlocal best, best_size
            try:
                size = len(json.dumps(candidate, ensure_ascii=False))
            except (TypeError, ValueError):
                return
            if size > best_size:
                best_size = size
                best = candidate

        def walk(node: Any) -> None:
            if isinstance(node, list):
                consider(node)
                for item in node:
                    walk(item)
            elif isinstance(node, dict):
                for item in node.values():
                    walk(item)

        walk(obj)
        return best
