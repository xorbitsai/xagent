"""Unit matrix for the shared prompt-context value filter."""

import pytest

from xagent.web.services.prompt_context import (
    build_prompt_context_value,
    filter_prompt_context,
)

DECLARATIONS = [
    {"name": "account", "max_length": 8},
    {"name": "tier", "max_length": 8},
    {"name": "exact", "max_length": 5},
]


_FIVE_LONG = [{"name": f"f{i}", "max_length": 1024} for i in range(5)]


@pytest.mark.parametrize(
    ("values", "declarations", "rejected"),
    [
        pytest.param({"account": "acme", "tier": 3}, DECLARATIONS, (), id="all_kept"),
        pytest.param({}, DECLARATIONS, (), id="empty"),
        pytest.param(
            {"other": "x", "account": "acme"}, DECLARATIONS, ("other",), id="undeclared"
        ),
        pytest.param(
            {"exact": "abcdef", "account": "acme"},
            DECLARATIONS,
            ("exact",),
            id="too_long",
        ),
        pytest.param(
            {"other": "x", "account": "acme", "exact": "abcdef"},
            DECLARATIONS,
            ("other", "exact"),
            id="mixed",
        ),
        pytest.param(
            {"account": "acme", "tier": 3},
            [{"name": "account"}],
            ("account", "tier"),
            id="malformed_declarations",
        ),
        # Every field fits its declaration, but together they exceed the
        # rendered-size cap, so the value carries none of them.
        pytest.param(
            {f"f{i}": "x" * 1000 for i in range(5)},
            _FIVE_LONG,
            tuple(f"f{i}" for i in range(5)),
            id="over_total_cap",
        ),
    ],
)
def test_unsigned_rejected_names_are_every_input_key_not_kept(
    values, declarations, rejected
) -> None:
    assert filter_prompt_context(values, declarations, signed=False)[1] == rejected


def test_unsigned_keeps_declared_primitive_values() -> None:
    assert build_prompt_context_value(
        {"account": "acme", "tier": 3, "exact": "abcde"},
        DECLARATIONS,
        signed=False,
    ) == {"data": {"account": "acme", "tier": 3, "exact": "abcde"}, "signed": False}


def test_unsigned_drops_undeclared_keys() -> None:
    assert build_prompt_context_value(
        {"account": "acme", "other": "x"}, DECLARATIONS, signed=False
    ) == {"data": {"account": "acme"}, "signed": False}


def test_unsigned_drops_over_length_values() -> None:
    assert build_prompt_context_value(
        {"account": "acme", "exact": "abcdef"}, DECLARATIONS, signed=False
    ) == {"data": {"account": "acme"}, "signed": False}


def test_unsigned_measures_non_string_primitives_by_str_length() -> None:
    assert (
        build_prompt_context_value({"tier": 123456789}, DECLARATIONS, signed=False)
        is None
    )


def test_unsigned_drops_none_and_containers() -> None:
    assert (
        build_prompt_context_value(
            {"account": None, "tier": [1], "exact": {"a": 1}},
            DECLARATIONS,
            signed=False,
        )
        is None
    )


def test_signed_keeps_undeclared_and_ignores_length_limits() -> None:
    values = {"undeclared": "x" * 4000, "none": None, "list": [1]}
    assert build_prompt_context_value(values, [], signed=True) == {
        "data": {"undeclared": "x" * 4000},
        "signed": True,
    }


def test_signed_keeps_only_primitive_values() -> None:
    assert build_prompt_context_value(
        {"a": "x", "b": True, "c": 1.5, "d": None, "e": [1]},
        None,
        signed=True,
    ) == {"data": {"a": "x", "b": True, "c": 1.5}, "signed": True}


def test_empty_and_non_mapping_values_return_none() -> None:
    assert build_prompt_context_value({}, DECLARATIONS, signed=False) is None
    assert build_prompt_context_value(None, DECLARATIONS, signed=False) is None
    assert build_prompt_context_value(["a"], DECLARATIONS, signed=False) is None


def test_malformed_declarations_declare_nothing() -> None:
    values = {"account": "acme"}
    for declarations in (
        None,
        "account",
        [{"name": "account"}],
        [{"name": "account", "max_length": "8"}],
        [{"name": "account", "max_length": 0}],
        [{"name": "account", "max_length": True}],
        [{"name": "account", "max_length": 8.0}],
        [{"max_length": 8}],
        [None],
        [None, {"name": "account", "max_length": 8}],
        [{"name": "account", "max_length": 8}, {"name": "account", "max_length": 9}],
        [{"name": f"f{i}", "max_length": 8} for i in range(32)]
        + [{"name": "account", "max_length": 8}],
    ):
        assert build_prompt_context_value(values, declarations, signed=False) is None, (
            declarations
        )


def test_invalid_declared_name_admits_nothing() -> None:
    assert (
        build_prompt_context_value(
            {"bad name": "a"}, [{"name": "bad name", "max_length": 4}], signed=False
        )
        is None
    )


def test_unsigned_drops_integer_too_large_to_stringify() -> None:
    assert build_prompt_context_value(
        {"big": 10**5000, "account": "acme"},
        [{"name": "big", "max_length": 1024}, {"name": "account", "max_length": 8}],
        signed=False,
    ) == {"data": {"account": "acme"}, "signed": False}


def test_over_total_cap_returns_none() -> None:
    assert build_prompt_context_value({"a": "x" * 4096}, [], signed=True) is None
    assert build_prompt_context_value({"a": "x" * 4000}, [], signed=True) == {
        "data": {"a": "x" * 4000},
        "signed": True,
    }


def test_nan_returns_none() -> None:
    assert build_prompt_context_value({"a": float("nan")}, [], signed=True) is None


_DECLARED_A_B = [{"name": "a", "max_length": 8}, {"name": "b", "max_length": 8}]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "\ud83d"])
@pytest.mark.parametrize("signed", [False, True])
def test_unserializable_value_drops_only_that_field(bad, signed) -> None:
    assert build_prompt_context_value(
        {"a": "ok", "b": bad}, _DECLARED_A_B, signed=signed
    ) == {"data": {"a": "ok"}, "signed": signed}


def test_signed_nan_drops_only_that_field() -> None:
    assert build_prompt_context_value(
        {"a": "ok", "b": float("nan")}, _DECLARED_A_B, signed=True
    ) == {"data": {"a": "ok"}, "signed": True}


@pytest.mark.parametrize("value", ["text", 1, 0, -2.5, True, False])
def test_signed_keeps_each_primitive_kind(value: str | int | float | bool) -> None:
    assert build_prompt_context_value({"a": value}, [], signed=True) == {
        "data": {"a": value},
        "signed": True,
    }


_DESCRIBED = [
    {"name": "account", "max_length": 8, "description": "Customer account slug"},
    {"name": "tier", "max_length": 8, "description": "Customer membership level"},
    {"name": "unused", "max_length": 8, "description": "Not passed"},
]


def test_descriptions_carry_only_for_kept_fields() -> None:
    assert build_prompt_context_value(
        {"account": "acme", "other": "x"}, _DESCRIBED, signed=False
    ) == {
        "data": {"account": "acme"},
        "signed": False,
        "descriptions": {"account": "Customer account slug"},
    }


def test_descriptions_omitted_when_declaration_has_none() -> None:
    assert build_prompt_context_value(
        {"account": "acme"}, DECLARATIONS, signed=False
    ) == {"data": {"account": "acme"}, "signed": False}


def test_signed_descriptions_cover_declared_kept_names_only() -> None:
    assert build_prompt_context_value(
        {"account": "acme", "tier": 3, "extra": "x"}, _DESCRIBED, signed=True
    ) == {
        "data": {"account": "acme", "tier": 3, "extra": "x"},
        "signed": True,
        "descriptions": {
            "account": "Customer account slug",
            "tier": "Customer membership level",
        },
    }
