"""Unit tests for the derived scope-column encoding (#822, slice 001/002)."""

from __future__ import annotations

import json

import pytest

from xagent.core.execution_scope import MEMORY_DIMENSION_METADATA_PREFIX
from xagent.core.memory.scope_columns import (
    SCOPE_DIMS_COLUMN,
    SCOPE_EXCLUSIVE_FILTER_KEY,
    USER_ID_COLUMN,
    build_scope_where,
    coerce_user_id,
    derive_scope_columns,
    encode_scope_dims,
    scope_dim_where_term,
    strict_scope_columns,
    strict_user_id,
    user_id_where_term,
)

P = MEMORY_DIMENSION_METADATA_PREFIX


def test_encode_empty_when_no_dimensions():
    assert encode_scope_dims({}) == []
    assert encode_scope_dims({"user_id": 7, "content": "x"}) == []


def test_encode_single_and_multiple_dimensions():
    assert encode_scope_dims({f"{P}tenant": "acme"}) == ["tenant=acme"]
    # Sorted by key (order-independent).
    assert encode_scope_dims({f"{P}tenant": "acme", f"{P}agent": "x"}) == [
        "agent=x",
        "tenant=acme",
    ]


def test_encoding_is_order_independent():
    a = encode_scope_dims({f"{P}tenant": "acme", f"{P}agent": "x"})
    b = encode_scope_dims({f"{P}agent": "x", f"{P}tenant": "acme"})
    assert a == b


def test_encode_preserves_raw_values_without_escaping():
    # array_contains is exact per-element equality, so no delimiter escaping.
    assert encode_scope_dims({f"{P}path": "/a=b/c"}) == ["path=/a=b/c"]
    assert encode_scope_dims({f"{P}t": "a_b"}) == ["t=a_b"]


def test_coerce_user_id():
    assert coerce_user_id(7) == 7
    assert coerce_user_id("7") == 7
    assert coerce_user_id(None) is None
    assert coerce_user_id("not-an-int") is None


def test_derive_scope_columns_from_json():
    uid, dims = derive_scope_columns(
        '{"user_id": 7, "' + P + 'tenant": "acme", "content": "x"}'
    )
    assert uid == 7
    assert dims == ["tenant=acme"]


def test_derive_scope_columns_tolerates_bad_input():
    assert derive_scope_columns(None) == (None, [])
    assert derive_scope_columns("") == (None, [])
    assert derive_scope_columns("not json") == (None, [])
    assert derive_scope_columns("[1, 2, 3]") == (None, [])


# --- where-clause construction (slice 002) ---------------------------------


def test_user_id_where_term():
    assert user_id_where_term(7) == f"{USER_ID_COLUMN} = 7"
    assert user_id_where_term("7") == f"{USER_ID_COLUMN} = 7"
    assert user_id_where_term(None) is None
    assert user_id_where_term("bad") is None


def test_scope_dim_where_term_is_exact_array_contains():
    assert (
        scope_dim_where_term("tenant", "acme")
        == f"array_contains({SCOPE_DIMS_COLUMN}, 'tenant=acme')"
    )
    # Values with SQL-special quotes are doubled; no LIKE wildcards to escape.
    assert scope_dim_where_term("t", "a'b") == (
        f"array_contains({SCOPE_DIMS_COLUMN}, 't=a''b')"
    )
    # Underscore/equals are stored/matched literally (no wildcard semantics).
    assert scope_dim_where_term("t", "a_b") == (
        f"array_contains({SCOPE_DIMS_COLUMN}, 't=a_b')"
    )


def test_build_scope_where_pushes_user_and_dims():
    where_sql, residual = build_scope_where(
        {"metadata": {"user_id": 1, f"{P}tenant": "acme"}}
    )
    assert where_sql is not None
    assert f"{USER_ID_COLUMN} = 1" in where_sql
    assert f"array_contains({SCOPE_DIMS_COLUMN}, 'tenant=acme')" in where_sql
    assert " AND " in where_sql
    assert residual == {}


def test_build_scope_where_keeps_residual_filters():
    where_sql, residual = build_scope_where(
        {
            "category": "work",
            "metadata": {"user_id": 1, f"{P}tenant": "acme", "custom": "v"},
        }
    )
    assert where_sql is not None
    assert residual == {"category": "work", "metadata": {"custom": "v"}}


def test_build_scope_where_unscoped_and_empty():
    assert build_scope_where(None) == (None, {})
    assert build_scope_where({}) == (None, {})
    where_sql, residual = build_scope_where({"metadata": {"user_id": 5}})
    assert where_sql == f"{USER_ID_COLUMN} = 5"
    assert residual == {}


def test_build_scope_where_unparsable_user_id_falls_back_to_python():
    where_sql, residual = build_scope_where({"metadata": {"user_id": "abc"}})
    assert where_sql is None
    assert residual == {"metadata": {"user_id": "abc"}}


def test_build_scope_where_exclusive_directive():
    where_sql, residual = build_scope_where(
        {"metadata": {"user_id": 1}, SCOPE_EXCLUSIVE_FILTER_KEY: True}
    )
    assert where_sql == (
        f"{USER_ID_COLUMN} = 1 AND "
        f"(array_length({SCOPE_DIMS_COLUMN}) = 0 "
        f"OR {SCOPE_DIMS_COLUMN} IS NULL)"
    )
    # The reserved directive is consumed, never left for equality matching.
    assert residual == {}


# --- strict owner resolution for table-rewriting paths ----------------------


def test_strict_user_id_resolves_the_exact_spellings():
    assert strict_user_id(None) is None
    assert strict_user_id(5) == 5
    assert strict_user_id(5.0) == 5
    assert strict_user_id("5") == 5
    assert strict_user_id(-7) == -7
    assert strict_user_id(2**63 - 1) == 2**63 - 1


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        1.5,
        True,
        False,
        "nan",
        "1.5",
        "abc",
        [1],
        {"user_id": 1},
        2**63,
        -(2**63) - 1,
    ],
)
def test_strict_user_id_rejects_every_value_that_names_no_single_owner(value):
    with pytest.raises(ValueError):
        strict_user_id(value)


def test_strict_user_id_float_boundary_at_two_to_the_fifty_third():
    # Below 2**53 every integer is exactly representable and no larger integer
    # rounds into the range, so a float there denotes exactly one owner.
    assert strict_user_id(float(2**53 - 1)) == 2**53 - 1
    assert strict_user_id(float(-(2**53) + 1)) == -(2**53) + 1
    # json.loads has already rounded by the time the value is validated, and
    # 2**53 is what BOTH of these texts decode to -- so the float names no
    # single owner and neither spelling may be staged as one.
    assert json.loads('{"user_id": 9007199254740993.0}')["user_id"] == float(2**53)
    for metadata in (
        '{"user_id": 9007199254740992.0}',
        '{"user_id": 9007199254740993.0}',
    ):
        with pytest.raises(ValueError, match="exactly representable"):
            strict_scope_columns(metadata)


def test_strict_user_id_keeps_exact_spellings_of_large_owners():
    # A JSON integer token is decoded without going through a float, so an id
    # past the float-exact range still resolves to itself rather than blocking.
    big = 2**53 + 1
    assert strict_scope_columns('{"user_id": 9007199254740993}') == (big, [])
    assert strict_scope_columns('{"user_id": "9007199254740993"}') == (big, [])


def test_strict_scope_columns_matches_derive_for_readable_rows():
    metadata = '{"user_id": 7, "' + P + 'tenant": "acme", "content": "x"}'
    assert (
        strict_scope_columns(metadata)
        == derive_scope_columns(metadata)
        == (
            7,
            ["tenant=acme"],
        )
    )
    # Metadata holding no JSON object carries no owner to lose, so it stays
    # tolerated rather than blocking a rewrite.
    for bad in (None, "", "not json", "[1, 2, 3]"):
        assert strict_scope_columns(bad) == (None, [])
