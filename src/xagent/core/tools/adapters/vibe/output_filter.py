"""
Tool Output Value Filtering Module

Provides multi-layered output limiting for all tools to prevent excessive output.

This module implements a three-pronged approach to control tool output size:
1. Per-string length limit: Truncates individual string values. When a
   too-long string is itself JSON, this first tries structure-aware
   trimming (rank candidate lists by how much dropping trailing items
   from each would help, and try them in that order until one actually
   brings the payload under budget, enforcing the same field-count cap
   as #2) so the result stays valid JSON, falling back to a raw
   character slice only when no candidate can be trimmed to fit.
2. Field count limit: Limits the number of items in dicts/lists
3. Recursion depth limit: Prevents excessively deep nesting

Rather than calculating total output size (which would be expensive), these
limits work together to provide reasonable protection while maintaining good
performance. For token safety, the combination of these limits is sufficient
for most real-world scenarios.
"""

import json
import logging
import math
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

    def __init__(
        self,
        max_chars: int,
        max_fields: int,
        max_recursion: int,
        max_structured_truncate_input_chars: int = 10_000_000,
    ):
        """
        Initialize output filter.

        Args:
            max_chars: Maximum length per string value (not total output).
            max_fields: Maximum number of fields/items in dicts/lists.
            max_recursion: Maximum nesting depth for recursive structures.
            max_structured_truncate_input_chars: Above this raw string length,
                skip the JSON-aware truncation attempt entirely and go
                straight to the O(1) raw character slice. Structure-aware
                trimming re-serializes the whole parsed structure repeatedly
                (see _try_json_truncate), which is worth it for the
                moderately-oversized payloads this feature targets but would
                cost real CPU time (seconds, for multi-million-item lists) on
                pathologically large input.
        """
        self.max_chars = max_chars
        self.max_fields = max_fields
        self.max_recursion = max_recursion
        self.max_structured_truncate_input_chars = max_structured_truncate_input_chars

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
            return self._filter_string(value, tool_name, depth)

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
            return self._filter_string(str_value, tool_name, depth)

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
            return self._filter_string(str_value, tool_name, depth)

    def _filter_string(self, value: str, tool_name: str, depth: int = 0) -> str:
        """
        Filter a string value.

        Args:
            value: String value to filter
            tool_name: Name of the tool (for logging)
            depth: Recursion depth this string was found at in the outer
                structure. If the string is itself JSON, any nested
                lists/dicts inside it continue counting from here, so
                content nested past max_recursion gets redacted the same
                way it would if the outer structure had returned it as a
                native dict/list instead of a pre-serialized string.

        Returns:
            Filtered string value
        """
        if len(value) <= self.max_chars:
            return value

        structured = self._try_json_truncate(value, tool_name, depth)
        if structured is not None:
            return structured

        truncated = value[: self.max_chars]
        result = truncated + DEFAULT_TRUNCATION_MESSAGE
        logger.info(
            f"Tool '{tool_name}' output truncated: "
            f"{len(value)} -> {len(result)} characters"
        )
        return result

    def _try_json_truncate(
        self, value: str, tool_name: str, depth: int = 0
    ) -> str | None:
        """
        Truncate an oversized JSON string without corrupting its syntax, by
        (in order): capping list/dict cardinality the same way
        _filter_with_depth caps native structures, trying a fully compact
        re-serialization, and - only if that's still too long - binary
        searching how much of the best candidate list can be kept, trying
        the next-best candidate if the top one can't usefully be trimmed.

        Returns None (falling back to raw character slicing) when the value
        isn't parseable JSON, is too large to be worth the repeated
        re-serialization this involves, contains no list worth trimming, or
        no candidate list can be brought under max_chars by trimming alone
        (e.g. a large non-list field dominates the payload) - this is a
        best-effort improvement over raw slicing, never a guaranteed one.

        Args:
            depth: Recursion depth this JSON string was found at in the
                outer structure (see _filter_string) - content nested past
                max_recursion inside it gets redacted the same way it would
                if the outer structure had returned it as a native dict/list.
        """
        if len(value) > self.max_structured_truncate_input_chars:
            return None

        try:
            parsed = json.loads(value)
            if not isinstance(parsed, (dict, list)):
                return None

            # Cap cardinality the same way _filter_with_depth caps native
            # lists/dicts (so a tool that happens to pre-serialize its result
            # to a JSON string doesn't bypass max_fields entirely), redact
            # anything past max_recursion the same way (continuing from the
            # caller's depth, so nesting already consumed by the outer
            # structure counts), and sanitize non-finite floats to null so a
            # NaN/Infinity anywhere in the payload - not just inside whatever
            # list ends up trimmed - can't force every serialization attempt
            # below to fail. Track each capped list's true original length
            # (keyed by the new list's id()) so that if max_chars *also* ends
            # up trimming this same list further below, its marker can
            # report how much was dropped in total instead of understating
            # it relative to the already-capped length; also track whether
            # anything was actually capped, for accurate logging.
            field_cap_true_lengths: dict[int, int] = {}
            capped_anything = [False]
            parsed = self._cap_max_fields(
                parsed, field_cap_true_lengths, capped_anything, depth
            )

            # A fully compact re-serialization (tightest separators, no
            # pretty-print whitespace) may already fit. Try this first,
            # regardless of whether there's a list to trim - it's the
            # cheapest possible win, and it's the only chance a payload with
            # no list gets to end up as valid JSON instead of falling
            # straight through to a raw, corrupting slice.
            compact = self._safe_compact_dumps(parsed)
            if compact is not None and len(compact) <= self.max_chars:
                if capped_anything[0]:
                    logger.info(
                        f"Tool '{tool_name}' output JSON-compacted (structure "
                        f"modified by field-count/depth capping or non-finite "
                        f"value sanitization): {len(value)} -> {len(compact)} "
                        f"characters"
                    )
                else:
                    logger.info(
                        f"Tool '{tool_name}' output JSON-compacted (no items "
                        f"dropped): {len(value)} -> {len(compact)} characters"
                    )
                return compact

            # Try candidates in order (most items first, ties by estimated
            # size) until one can actually be trimmed to fit - a small list
            # ranking first doesn't mean it can absorb enough of the
            # overflow, and giving up after only the top candidate would
            # miss a valid trim that a different (lower-ranked) list could
            # have provided.
            for target_list in self._find_trim_candidates(parsed):
                restore_state = list(target_list)  # current contents
                # If field-capping already truncated this exact list, its
                # last element is a synthetic marker string, not a real item
                # - it isn't eligible to be "kept" below, and the search
                # needs the list's true pre-capping length (not its
                # already-capped length) to report accurate remaining counts.
                true_length = field_cap_true_lengths.get(id(target_list))
                original_items = (
                    restore_state[:-1] if true_length is not None else restore_state
                )
                total = len(original_items)
                true_total = true_length if true_length is not None else total

                # Binary search the largest prefix of the list (plus a
                # marker for the dropped remainder) whose re-serialized JSON
                # fits within max_chars. Every candidate here always carries
                # the marker (remaining >= 1), so length is monotonic in the
                # prefix size - each additional kept item only adds
                # characters. (The candidate with the whole list and no
                # marker was already ruled out above, since dropping a
                # roughly constant-size marker for the last couple of items
                # can *shrink* the output, which would otherwise break that
                # monotonicity.) _safe_compact_dumps returning None here is
                # just a defensive fallback at this point - _cap_max_fields
                # already sanitized every non-finite float in the whole
                # structure before this loop ever runs, so no candidate
                # should actually trigger it in practice.
                lo, hi = 0, total - 1
                best_serialized: str | None = None
                best_kept = 0
                try:
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        remaining = true_total - mid
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
                    target_list[:] = restore_state

                # Keeping zero real items (marker only) discards every real
                # element of this list - strictly less real content than the
                # old raw-slice fallback would keep, so it isn't a useful
                # trim; try the next candidate instead of settling for an
                # effectively-empty list.
                if best_serialized is None or best_kept == 0:
                    continue

                logger.info(
                    f"Tool '{tool_name}' output JSON-truncated: kept "
                    f"{best_kept}/{true_total} list items ({len(value)} -> "
                    f"{len(best_serialized)} characters)"
                )
                return best_serialized

            return None
        except json.JSONDecodeError:
            # The routine, expected case: a large non-JSON string (markdown,
            # HTML, logs, ...). Not worth INFO-level noise on top of the
            # truncation log already emitted by the raw-slice fallback.
            logger.debug(
                f"Tool '{tool_name}' output is not JSON; "
                f"falling back to raw character slicing."
            )
            return None
        except (TypeError, ValueError, RecursionError) as e:
            # ValueError also covers json.loads raising on an oversized
            # integer literal (Python's int-to-str conversion limit, PEP
            # 3151-era safeguard) - a plain ValueError, not a
            # json.JSONDecodeError, so it isn't caught by the branch above.
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

    def _cap_max_fields(
        self,
        node: Any,
        true_lengths: dict[int, int],
        capped_anything: list[bool],
        depth: int = 0,
    ) -> Any:
        """Recursively cap every list/dict in a JSON-parsed structure to
        max_fields items and redact anything past max_recursion, mirroring
        _filter_with_depth's cardinality cap and depth redaction on native
        lists/dicts - so the same data behaves the same way whether a tool
        hands it back as a native object or as a pre-serialized JSON string.
        Also sanitizes non-finite floats (NaN/Infinity/-Infinity) to null,
        since json.loads accepts those non-standard tokens but re-emitting
        them isn't valid JSON.

        Records each capped list's true (pre-capping) length in
        true_lengths, keyed by id() of the new, already-capped list. This
        lets a later max_chars-driven trim of the *same* list (see
        _try_json_truncate) report how much was dropped in total, instead of
        understating it relative to the already-capped length. Sets
        capped_anything[0] = True if this call (or any nested call) actually
        dropped items via max_fields, so callers can log accurately.
        """
        if depth > self.max_recursion:
            capped_anything[0] = True
            return NESTED_TOO_DEEP_MESSAGE
        if isinstance(node, list):
            if len(node) > self.max_fields:
                kept = [
                    self._cap_max_fields(item, true_lengths, capped_anything, depth + 1)
                    for item in node[: self.max_fields]
                ]
                true_lengths[id(kept)] = len(node)
                kept.append(
                    TRUNCATED_ITEMS_TEMPLATE.format(count=len(node) - self.max_fields)
                )
                capped_anything[0] = True
                return kept
            return [
                self._cap_max_fields(item, true_lengths, capped_anything, depth + 1)
                for item in node
            ]
        if isinstance(node, dict):
            if len(node) > self.max_fields:
                kept_dict = {
                    k: self._cap_max_fields(v, true_lengths, capped_anything, depth + 1)
                    for k, v in list(node.items())[: self.max_fields]
                }
                kept_dict[
                    TRUNCATED_DICT_TEMPLATE.format(count=len(node) - self.max_fields)
                ] = TRUNCATED_FIELDS_MESSAGE
                capped_anything[0] = True
                return kept_dict
            return {
                k: self._cap_max_fields(v, true_lengths, capped_anything, depth + 1)
                for k, v in node.items()
            }
        if isinstance(node, float) and not math.isfinite(node):
            capped_anything[0] = True
            return None
        return node

    # Cap on how many ranked candidates _try_json_truncate will try binary
    # search on before giving up. This is a bounded heuristic, not a full
    # fix: it only raises the number of similarly-ranked-but-untrimmable
    # candidates needed to reproduce the "gives up even though a lower-
    # ranked candidate could have been trimmed" failure mode, it doesn't
    # eliminate it (a structure with more than this many individually-
    # insufficient qualifying lists can still fall back to a raw slice
    # unnecessarily). Kept low - covering a handful of sibling candidates,
    # which is the realistic shape this was written for (e.g. two or three
    # top-level lists in an envelope) - because each extra candidate tried
    # costs a full binary search that re-serializes the whole document per
    # step; measured ~5-7x worst-case slowdown at the previous value (10)
    # for a payload with many similarly-ranked, individually-insufficient
    # lists, on top of _try_json_truncate already running synchronously on
    # the async tool-call path (see output_filter_wrapper.py).
    _MAX_TRIM_CANDIDATES_TO_TRY = 3

    @classmethod
    def _find_trim_candidates(cls, obj: Any) -> list[list]:
        """Find every list (with >= 2 items) worth considering as a trim
        target, ranked most-worth-trimming first: most items wins, ties
        broken by estimated serialized size. Returns at most
        _MAX_TRIM_CANDIDATES_TO_TRY candidates.

        Item count, not total serialized size, is the primary ranking
        because it's what makes ranking safe regardless of nesting: a list
        with few items (an envelope wrapper, e.g. {"results": [{"items":
        [...100 rows...]}]}) can only be trimmed by discarding a whole
        element at a time - there's no graceful partial reduction - so it
        should lose to any list with more items, including one nested inside
        it. A pure total-size comparison gets this backwards, since an
        ancestor's size is always its descendants' size plus more, so it
        would "win" by size regardless of how few items it has. Item count
        also correctly handles the common shape where a large records list's
        *individual* records each carry their own small nested list (e.g.
        per-record "tags"): a 180-item records list beats each record's
        3-item "tags" list on item count, so the actual data-bearing list is
        ranked first either way - no ancestor/descendant relationship needs
        to be tracked at all, unlike a pure-size comparison (which would have
        to explicitly exclude ancestors to avoid ranking the wrapper first,
        and then risks incorrectly excluding this records list too, since it
        also "contains" >= 2 item lists nested inside its records).

        Ranking by item count first can still pick a list that can't absorb
        enough of the overflow (many small items don't add up to much), so
        the caller tries candidates in ranked order and moves on to the next
        one if the top-ranked list can't be usefully trimmed, rather than
        giving up as soon as the single best-ranked candidate falls short.

        Size is estimated instead of calling json.dumps on every candidate
        list, which would make the walk quadratic (or worse) for structures
        with several nesting levels, since each level would re-serialize
        everything below it.
        """
        candidates: list[tuple[tuple[int, int], list]] = []

        def walk(node: Any) -> int:
            """Returns the estimated serialized size of node."""
            if isinstance(node, list):
                size = 2 + max(0, len(node) - 1)  # brackets + separators
                for item in node:
                    size += walk(item)
                if len(node) >= 2:
                    candidates.append(((len(node), size), node))
                return size
            if isinstance(node, dict):
                size = 2 + max(0, len(node) - 1)  # braces + separators
                for dict_key, val in node.items():
                    size += len(str(dict_key)) + 3 + walk(val)  # "key":
                return size
            if isinstance(node, str):
                return len(node) + 2  # quotes
            if isinstance(node, bool):
                return 4 if node else 5
            if node is None:
                return 4
            return len(str(node))

        walk(obj)
        candidates.sort(key=lambda c: c[0], reverse=True)
        return [node for _, node in candidates[: cls._MAX_TRIM_CANDIDATES_TO_TRY]]
