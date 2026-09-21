"""Derived filter columns for the LanceDB memory store (#822).

Promotes the two fields scope searches filter on — the owner ``user_id`` and the
execution-scope memory dimensions — out of the JSON ``metadata`` string into
real, top-level columns, so they can be pushed into a LanceDB ``where`` prefilter
instead of the Python post-filtering that collapses recall under shared
collections (see #822).

The ``metadata`` JSON stays authoritative: these columns are derived
projections, computed from a note's metadata on write and back-filled from it on
migration. Nothing reconstructs a note from them.

``scope_dims`` is a ``list<string>`` column holding one ``"key=value"`` element
per dimension (e.g. ``["agent=x", "tenant=acme"]``). Dimension membership is
tested with DataFusion's ``array_contains``, which is exact per-element string
equality — so values may contain any character (``=``, ``/``, ``_`` …) with no
escaping, and there is no substring-collision surface (``tenant=a`` never matches
``tenant=ab``).
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Optional

from ..execution_scope import MEMORY_DIMENSION_METADATA_PREFIX

# Real column names the memory dimensions are promoted into.
USER_ID_COLUMN = "user_id"
SCOPE_DIMS_COLUMN = "scope_dims"

# Reserved, namespaced top-level filter key (#822). When truthy, the search must
# return only notes carrying NO scope dimensions — the pushdown form of
# ``strict_memory_isolation`` for a dimension-less scope. The ``__`` prefix keeps
# it from colliding with a caller's real equality-filter key. It is consumed by
# the store's filter layer (an ``array_length`` term on the vector path, a Python
# check on the text fallback) and never reaches equality matching.
SCOPE_EXCLUSIVE_FILTER_KEY = "__scope_exclusive__"

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
# Largest integer magnitude an IEEE-754 double still round-trips one-to-one.
# Above it the spacing between representable doubles reaches 2, so ``2**53``
# itself is what both "9007199254740992.0" and "9007199254740993.0" decode to
# and a float that large no longer names one owner. See ``strict_user_id``.
_EXACT_FLOAT_INT_MAX = 2**53 - 1


def scope_dim_element(dim_key: str, value: Any) -> str:
    """The ``"key=value"`` list element stored/matched for one dimension."""
    return f"{dim_key}={value}"


def encode_scope_dims(metadata: Mapping[str, Any]) -> list[str]:
    """The scope-dimension list for a note: one ``"key=value"`` per stamp.

    Reads the ``execution_scope_<key>`` entries from ``metadata``, sorted by key
    (order-independent). Returns ``[]`` when the note carries no dimensions.
    """
    return [
        scope_dim_element(key[len(MEMORY_DIMENSION_METADATA_PREFIX) :], metadata[key])
        for key in sorted(metadata)
        if key.startswith(MEMORY_DIMENSION_METADATA_PREFIX)
    ]


def coerce_user_id(value: Any) -> Optional[int]:
    """Best-effort integer owner id for the column; ``None`` when absent/bad.

    Deliberately total: the query path needs an unreadable ``user_id`` to come
    back as ``None`` so ``build_scope_where`` drops the pushdown term and leaves
    the value to the Python post-filter, which compares the authoritative
    metadata. Never raise from here. Code that rewrites a table, where the
    derived column becomes the only owner a row keeps, must use
    ``strict_user_id``/``strict_scope_columns`` instead.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def strict_user_id(value: Any) -> Optional[int]:
    """The one signed-int64 owner a persisted ``user_id`` denotes.

    ``None`` only when no owner was persisted (key absent or JSON ``null``).
    Anything else that does not denote exactly one in-range integer raises
    ``ValueError``: a rewrite has no authoritative value to fall back on, so
    ``coerce_user_id``'s ``None`` would silently turn an owned note into an
    unowned one (JSON ``NaN``, ``"nan"``) and its truncation would silently hand
    one note to a different owner (``1.5`` -> ``1``, ``true`` -> ``1``).
    Accepts every spelling ``coerce_user_id`` resolves today -- an ``int``, an
    integral finite ``float``, a base-10 numeric string -- unchanged, except
    for a float too large to name one owner.

    That float bound exists because the JSON decoder has already rounded by the
    time the value arrives: ``json.loads('{"user_id": 9007199254740993.0}')``
    yields ``9007199254740992.0``, which ``is_integer()`` accepts and ``int()``
    turns into an owner the stored metadata never named. So a float owner must
    satisfy ``abs(value) <= 2**53 - 1``, the range where every integer is
    exactly representable and no larger integer rounds into it; ``2**53`` is
    rejected as the value both of the texts above collapse to. An id past that
    range is still accepted when it is spelled exactly -- a JSON integer token
    decodes to a Python ``int`` of any size, and a numeric string is parsed in
    base 10 -- so only the lossy spelling is refused, never the owner.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("legacy user_id must be a whole number, not a boolean")
    if isinstance(value, int):
        user_id = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("legacy user_id must be a finite whole number")
        if abs(value) > _EXACT_FLOAT_INT_MAX:
            raise ValueError(
                "legacy user_id float must be an exactly representable integer"
            )
        user_id = int(value)
    elif isinstance(value, str):
        try:
            user_id = int(value, 10)
        except ValueError as exc:
            raise ValueError("legacy user_id must spell a whole number") from exc
    else:
        raise ValueError("legacy user_id must be a number or a numeric string")
    if not _INT64_MIN <= user_id <= _INT64_MAX:
        raise ValueError("legacy user_id must fit signed int64")
    return user_id


def _sql_string_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def user_id_where_term(value: Any) -> Optional[str]:
    """Equality predicate on the ``user_id`` column, or ``None`` if unparsable."""
    user_id = coerce_user_id(value)
    if user_id is None:
        return None
    return f"{USER_ID_COLUMN} = {user_id}"


def scope_dim_where_term(dim_key: str, value: Any) -> str:
    """An ``array_contains`` predicate matching notes carrying this dimension.

    Exact per-element string equality, so a note carrying a superset of the
    queried dimensions still matches while a different value or a prefix does
    not — no escaping or collision reasoning required.
    """
    element = _sql_string_literal(scope_dim_element(dim_key, value))
    return f"array_contains({SCOPE_DIMS_COLUMN}, {element})"


def build_scope_where(
    filters: Optional[Mapping[str, Any]],
) -> tuple[Optional[str], dict[str, Any]]:
    """Split ``filters`` into a pushable ``where`` clause and residual filters.

    Extracts ``user_id`` and the ``execution_scope_*`` dimensions from the nested
    ``filters["metadata"]`` into column predicates (``user_id`` equality +
    ``array_contains`` per dimension), AND-ed together and applied as a prefilter
    so the ANN returns ``k`` already-scoped neighbours. Everything the clause
    cannot express — category, arbitrary metadata keys, an unparsable
    ``user_id`` — is returned as residual filters for the Python post-filter.

    Returns ``(where_sql or None, residual_filters)``.
    """
    if not filters:
        return None, {}
    residual: dict[str, Any] = dict(filters)
    clauses: list[str] = []
    metadata = filters.get("metadata")
    if isinstance(metadata, Mapping):
        residual_metadata = dict(metadata)
        if "user_id" in residual_metadata:
            term = user_id_where_term(residual_metadata["user_id"])
            if term is not None:
                clauses.append(term)
                residual_metadata.pop("user_id")
        for key in list(residual_metadata):
            if key.startswith(MEMORY_DIMENSION_METADATA_PREFIX):
                dim = key[len(MEMORY_DIMENSION_METADATA_PREFIX) :]
                clauses.append(scope_dim_where_term(dim, residual_metadata[key]))
                residual_metadata.pop(key)
        if residual_metadata:
            residual["metadata"] = residual_metadata
        else:
            residual.pop("metadata", None)
    # Strict dimension-less exclusion: only notes with an empty dimension list.
    # Popped from residual so it never reaches equality matching. DataFusion
    # returns 0 (not NULL) for an empty list, so `= 0` matches exactly the
    # dimension-less notes (covered by the strict-exclusion integration test).
    # Every write/migration path (encode_scope_dims / derive_scope_columns /
    # _derive_scope_arrays; _backfill_missing_columns never touches scope_dims)
    # produces `[]`, never NULL — but since `array_length(NULL) IS NULL` would
    # silently drop a NULL row from a strict search, guard the invariant with an
    # explicit `IS NULL` branch so a future write path cannot break isolation.
    if residual.pop(SCOPE_EXCLUSIVE_FILTER_KEY, None):
        clauses.append(
            f"(array_length({SCOPE_DIMS_COLUMN}) = 0 OR {SCOPE_DIMS_COLUMN} IS NULL)"
        )
    where_sql = " AND ".join(clauses) if clauses else None
    return where_sql, residual


def _parsed_metadata(metadata_json: Optional[str]) -> dict[str, Any]:
    """The metadata object a stored JSON string holds; ``{}`` when it holds none."""
    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def derive_scope_columns(
    metadata_json: Optional[str],
) -> tuple[Optional[int], list[str]]:
    """Derive ``(user_id, scope_dims)`` from a stored metadata JSON string.

    Used to back-fill the columns for existing rows during migration. Malformed
    or non-object metadata yields ``(None, [])`` rather than raising, so one bad
    row cannot abort a migration.
    """
    metadata = _parsed_metadata(metadata_json)
    return coerce_user_id(metadata.get("user_id")), encode_scope_dims(metadata)


def strict_scope_columns(
    metadata_json: Optional[str],
) -> tuple[Optional[int], list[str]]:
    """``derive_scope_columns`` for paths that stage a rewrite of the table.

    Identical for every row whose ``user_id`` is readable, but raises
    ``ValueError`` instead of dropping or truncating one that is not, so a
    caller can classify the row as invalid legacy data and leave the stored
    table alone. Unreadable metadata still yields ``(None, [])``: a string that
    holds no JSON object carries no owner to lose.
    """
    metadata = _parsed_metadata(metadata_json)
    return strict_user_id(metadata.get("user_id")), encode_scope_dims(metadata)
