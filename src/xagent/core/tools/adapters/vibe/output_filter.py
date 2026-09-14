"""
Tool Output Value Filtering Module

Provides multi-layered output limiting for all tools to prevent excessive output.

This module implements a three-pronged approach to control tool output size:
1. Per-string length limit: Truncates individual string values. When a
   too-long string is itself JSON, this first tries structure-aware
   trimming (drop items from its largest data-bearing list, enforcing the
   same field-count cap as #2) so the result stays valid JSON, falling
   back to a raw character slice only when that isn't achievable.
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

    # Above this raw string length, skip the JSON-aware truncation attempt
    # entirely and go straight to the O(1) raw character slice. Structure-aware
    # trimming re-serializes the whole parsed structure repeatedly (see
    # _try_json_truncate), which is worth it for the moderately-oversized
    # payloads this feature targets but would cost real CPU time (seconds, for
    # multi-million-item lists) on pathologically large input.
    _MAX_STRUCTURED_TRUNCATE_INPUT_CHARS = 10_000_000

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
        Truncate an oversized JSON string without corrupting its syntax, by
        (in order): capping list/dict cardinality the same way
        _filter_with_depth caps native structures, trying a fully compact
        re-serialization, and - only if that's still too long - binary
        searching how much of the largest genuinely data-bearing list can be
        kept.

        Returns None (falling back to raw character slicing) when the value
        isn't parseable JSON, is too large to be worth the repeated
        re-serialization this involves, contains no list worth trimming, or
        can't be brought under max_chars by trimming alone (e.g. a large
        non-list field dominates the payload) - this is a best-effort
        improvement over raw slicing, never a guaranteed one.
        """
        if len(value) > self._MAX_STRUCTURED_TRUNCATE_INPUT_CHARS:
            return None

        try:
            parsed = json.loads(value)
            if not isinstance(parsed, (dict, list)):
                return None

            # Cap cardinality the same way _filter_with_depth caps native
            # lists/dicts, so a tool that happens to pre-serialize its result
            # to a JSON string doesn't bypass max_fields entirely.
            parsed = self._cap_max_fields(parsed)

            # A fully compact re-serialization (tightest separators, no
            # pretty-print whitespace) may already fit. Try this first,
            # regardless of whether there's a list to trim - it's the
            # cheapest possible win, and it's the only chance a payload with
            # no list (or with a stray NaN/Infinity that doesn't actually
            # need to survive) gets to end up as valid JSON instead of
            # falling straight through to a raw, corrupting slice.
            compact = self._safe_compact_dumps(parsed)
            if compact is not None and len(compact) <= self.max_chars:
                logger.info(
                    f"Tool '{tool_name}' output JSON-compacted (no items "
                    f"dropped): {len(value)} -> {len(compact)} characters"
                )
                return compact

            target_list = self._find_largest_list(parsed)
            if not target_list:
                return None

            original_items = list(target_list)
            total = len(original_items)

            # Binary search the largest prefix of the list (plus a marker for
            # the dropped remainder) whose re-serialized JSON fits within
            # max_chars. Every candidate here always carries the marker
            # (remaining >= 1), so length is monotonic in the prefix size -
            # each additional kept item only adds characters. (The candidate
            # with the whole list and no marker was already ruled out above,
            # since dropping a roughly constant-size marker for the last
            # couple of items can *shrink* the output, which would otherwise
            # break that monotonicity.) A candidate that still contains a
            # non-finite float makes _safe_compact_dumps return None, which
            # is treated the same as "too long" - once a kept prefix reaches
            # that item, every longer prefix keeps it too, so this stays
            # monotonic as well.
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
                    serialized = self._safe_compact_dumps(parsed)
                    if serialized is not None and len(serialized) <= self.max_chars:
                        best_serialized = serialized
                        best_kept = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
            finally:
                target_list[:] = original_items

            # Keeping zero real items (marker only) discards every real
            # element of the chosen list - strictly less real content than
            # the old raw-slice fallback would keep, so it isn't a useful
            # trim; fall back instead of returning an effectively-empty list.
            if best_serialized is None or best_kept == 0:
                return None

            logger.info(
                f"Tool '{tool_name}' output JSON-truncated: kept "
                f"{best_kept}/{total} list items ({len(value)} -> "
                f"{len(best_serialized)} characters)"
            )
            return best_serialized
        except json.JSONDecodeError:
            # The routine, expected case: a large non-JSON string (markdown,
            # HTML, logs, ...). Not worth INFO-level noise on top of the
            # truncation log already emitted by the raw-slice fallback.
            logger.debug(
                f"Tool '{tool_name}' output is not JSON; "
                f"falling back to raw character slicing."
            )
            return None
        except (TypeError, RecursionError) as e:
            logger.warning(
                f"Tool '{tool_name}' JSON-aware truncation failed unexpectedly "
                f"({e!r}); falling back to raw character slicing."
            )
            return None

    @staticmethod
    def _safe_compact_dumps(obj: Any) -> str | None:
        """json.dumps with the tightest valid separators, or None if obj
        contains a non-finite float (NaN/Infinity/-Infinity).

        json.loads happily parses those tokens (Python's non-standard
        extension - common in pandas/numpy-derived tool output), but they
        aren't valid per RFC 8259, so allow_nan=False is used to catch and
        reject re-emitting them rather than let them leak into supposedly
        valid JSON output. Returning None instead of raising lets callers
        treat "would need a non-finite float to represent this candidate" as
        just another reason a candidate doesn't fit.
        """
        try:
            return json.dumps(
                obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        except ValueError:
            return None

    def _cap_max_fields(self, node: Any, depth: int = 0) -> Any:
        """Recursively cap every list/dict in a JSON-parsed structure to
        max_fields items, mirroring _filter_with_depth's cardinality cap on
        native lists/dicts - so the same data returns the same item count
        whether a tool hands it back as a native object or as a pre-
        serialized JSON string.
        """
        if depth > self.max_recursion:
            return node
        if isinstance(node, list):
            if len(node) > self.max_fields:
                kept = [
                    self._cap_max_fields(item, depth + 1)
                    for item in node[: self.max_fields]
                ]
                kept.append(
                    TRUNCATED_ITEMS_TEMPLATE.format(count=len(node) - self.max_fields)
                )
                return kept
            return [self._cap_max_fields(item, depth + 1) for item in node]
        if isinstance(node, dict):
            if len(node) > self.max_fields:
                kept_dict = {
                    k: self._cap_max_fields(v, depth + 1)
                    for k, v in list(node.items())[: self.max_fields]
                }
                kept_dict[
                    TRUNCATED_DICT_TEMPLATE.format(count=len(node) - self.max_fields)
                ] = TRUNCATED_FIELDS_MESSAGE
                return kept_dict
            return {k: self._cap_max_fields(v, depth + 1) for k, v in node.items()}
        return node

    @staticmethod
    def _find_largest_list(obj: Any) -> list | None:
        """Find the largest (by estimated serialized size) *leaf* list nested
        anywhere in obj - a list with >= 2 items that does not itself
        contain another >= 2 item list anywhere inside it - in a single
        bottom-up pass.

        Size is estimated instead of calling json.dumps on every candidate
        list, which would make the walk quadratic (or worse) for structures
        with several nesting levels, since each level would re-serialize
        everything below it.

        Restricting candidates to leaves (not just requiring >= 2 items) is
        what actually avoids the ancestor bias: since this is a bottom-up
        sum, an ancestor list's estimated size always exceeds any list
        nested inside it (it's that list's size plus more), no matter how
        many items the ancestor itself has. A two-page envelope like
        {"results": [{"items": [...100 rows...]}, {"items": [...100 more
        rows...]}]} has 2 items at the outer "results" level, satisfying a
        plain ">= 2" floor, and would still "win" by size over either
        "items" list if ancestors were left eligible - discarding one whole
        real page instead of trimming rows. Excluding any list that contains
        a qualifying descendant list (at any depth) forces selection down to
        the actual data-bearing leaf list, regardless of how many wrapper
        levels or wrapper elements surround it.
        """
        best: list | None = None
        best_size = -1

        def walk(node: Any) -> tuple[int, bool]:
            """Returns (estimated size, whether this subtree is or contains
            a list with >= 2 items)."""
            nonlocal best, best_size
            if isinstance(node, list):
                size = 2 + max(0, len(node) - 1)  # brackets + separators
                contains_qualifying = False
                for item in node:
                    item_size, item_has_qualifying = walk(item)
                    size += item_size
                    contains_qualifying = contains_qualifying or item_has_qualifying
                is_qualifying = len(node) >= 2
                if is_qualifying and not contains_qualifying and size > best_size:
                    best_size = size
                    best = node
                return size, is_qualifying or contains_qualifying
            if isinstance(node, dict):
                size = 2 + max(0, len(node) - 1)  # braces + separators
                contains_qualifying = False
                for key, val in node.items():
                    val_size, val_has_qualifying = walk(val)
                    size += len(str(key)) + 3 + val_size  # "key":
                    contains_qualifying = contains_qualifying or val_has_qualifying
                return size, contains_qualifying
            if isinstance(node, str):
                return len(node) + 2, False  # quotes
            if isinstance(node, bool):
                return (4 if node else 5), False
            if node is None:
                return 4, False
            return len(str(node)), False

        walk(obj)
        return best
