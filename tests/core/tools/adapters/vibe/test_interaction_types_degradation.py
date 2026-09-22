"""The per-field form of the options rule (xagent#2529).

``_is_answerable`` (``react.py``) applies "a pick-from-a-list control with
no options cannot be answered" to a whole list, and only to decide whether
to append a free-text field beside it. These two helpers apply the same
rule to one interaction so a publisher can replace the dead control itself:
``lacks_required_options`` is the predicate, ``degrade_to_text_input`` the
replacement, and ``degrade_options_less_pickers`` applies them to a list
and reports which fields it touched.
"""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.tools.adapters.vibe.interaction_types import (
    TYPES_REQUIRING_OPTIONS,
    degrade_options_less_pickers,
    degrade_to_text_input,
    lacks_required_options,
)

PICKER_TYPES = sorted(TYPES_REQUIRING_OPTIONS)


@pytest.mark.parametrize("picker_type", PICKER_TYPES)
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"options": []},
        {"options": None},
        {"options": "auto"},
        {"options": {"a": "b"}},
    ],
    ids=["key-omitted", "empty-list", "none", "string", "dict"],
)
def test_a_picker_without_a_usable_options_list_lacks_them(
    picker_type: str, extra: dict[str, Any]
) -> None:
    """Anything the render surface cannot iterate counts as no options: a
    missing key, an empty list, and the non-list shapes
    ``_normalize_ask_user_interactions`` warns about and leaves verbatim."""

    interaction = {"type": picker_type, "field": "choice", "label": "Which?", **extra}
    assert lacks_required_options(interaction)


@pytest.mark.parametrize("picker_type", PICKER_TYPES)
def test_a_picker_with_one_option_has_what_it_needs(picker_type: str) -> None:
    interaction = {
        "type": picker_type,
        "field": "choice",
        "label": "Which?",
        "options": [{"label": "Paris", "value": "paris"}],
    }
    assert not lacks_required_options(interaction)


@pytest.mark.parametrize(
    "interaction_type",
    ["text_input", "number_input", "confirm", "file_upload", "connect_apps"],
)
@pytest.mark.parametrize(
    "extra", [{}, {"options": []}], ids=["key-omitted", "empty-list"]
)
def test_a_type_that_does_not_need_options_never_lacks_them(
    interaction_type: str, extra: dict[str, Any]
) -> None:
    interaction = {"type": interaction_type, "field": "f", "label": "L", **extra}
    assert not lacks_required_options(interaction)


@pytest.mark.parametrize(
    "interaction_type", [["select_one"], None, 7], ids=["list", "none", "int"]
)
def test_a_type_that_is_not_a_string_is_not_a_picker(interaction_type: Any) -> None:
    """The model is free to send ``type`` as a list. ``TYPES_REQUIRING_OPTIONS``
    is a frozenset, and a membership test on an unhashable value raises
    ``TypeError`` -- the same trap ``_is_answerable`` guards against."""

    interaction = {"type": interaction_type, "field": "f", "options": []}
    assert not lacks_required_options(interaction)


def test_a_missing_type_is_not_a_picker() -> None:
    assert not lacks_required_options({"field": "f", "label": "L", "options": []})


def test_degradation_keeps_field_label_and_placeholder() -> None:
    """The three keys that carry the question. ``field`` is what the answer
    is filed under, ``label`` is the question text and the run has no other
    copy of it, ``placeholder`` is the model's hint about the expected
    shape."""

    degraded = degrade_to_text_input(
        {
            "type": "select_one",
            "field": "client_id",
            "label": "Client",
            "placeholder": "Search clients",
            "options": [],
        }
    )
    assert degraded == {
        "type": "text_input",
        "field": "client_id",
        "label": "Client",
        "placeholder": "Search clients",
    }


def test_degradation_drops_the_keys_the_new_type_does_not_use() -> None:
    """``options`` is what was missing; ``default_value`` named one of those
    missing options; the rest are other types' knobs. The write side's
    validator (``validate_v1_write_payload``) refuses a ``text_input`` with a
    non-empty ``options``; an emptied list would pass, but the key goes so
    the degraded field carries nothing its type does not use."""

    degraded = degrade_to_text_input(
        {
            "type": "select_multiple",
            "field": "staff_ids",
            "label": "Staff",
            "options": [],
            "default_value": "s1",
            "min": 1,
            "max": 3,
            "multiple": True,
            "accept": [".csv"],
        }
    )
    assert degraded == {
        "type": "text_input",
        "field": "staff_ids",
        "label": "Staff",
        "multiline": True,
    }


@pytest.mark.parametrize(
    ("picker_type", "multiline"),
    [("select_one", False), ("select_multiple", True), ("action_cards", False)],
)
def test_only_a_degraded_select_multiple_is_multiline(
    picker_type: str, multiline: bool
) -> None:
    """A ``select_multiple`` asked for several values; a one-line box would
    invite one. The other two asked for a single choice."""

    degraded = degrade_to_text_input({"type": picker_type, "field": "f", "label": "L"})
    assert degraded.get("multiline", False) is multiline


@pytest.mark.parametrize(
    "placeholder", ["", "   ", None, 7], ids=["empty", "blank", "none", "int"]
)
def test_degradation_omits_a_placeholder_that_is_not_a_non_blank_string(
    placeholder: Any,
) -> None:
    degraded = degrade_to_text_input(
        {"type": "action_cards", "field": "f", "label": "L", "placeholder": placeholder}
    )
    assert "placeholder" not in degraded


def test_degradation_omits_a_label_that_is_not_a_string() -> None:
    """The frontend falls back to ``field`` when ``label`` is absent
    (``interaction.label || interaction.field``); a non-string label is
    handed on as absent rather than as a value that fallback cannot read."""

    degraded = degrade_to_text_input({"type": "select_one", "field": "f", "label": 7})
    assert degraded == {"type": "text_input", "field": "f"}


def test_degradation_returns_a_new_dict_and_leaves_the_input_alone() -> None:
    original = {"type": "select_one", "field": "f", "label": "L", "options": []}
    snapshot = dict(original)
    degraded = degrade_to_text_input(original)
    assert degraded is not original
    assert original == snapshot


def test_the_list_form_degrades_in_place_and_names_what_it_touched() -> None:
    """Untouched entries come back as the same objects (a caller's other
    references to them stay valid); degraded ones are new dicts; the names
    come back in list order, under the field name each entry carries at the
    time -- which is why the caller runs this after deduplication."""

    text = {"type": "text_input", "field": "note", "label": "Note"}
    picker = {"type": "select_one", "field": "choice_2", "label": "Which?"}
    kept = {
        "type": "select_one",
        "field": "city",
        "label": "City",
        "options": [{"label": "Paris", "value": "paris"}],
    }
    published, degraded_fields = degrade_options_less_pickers([text, picker, kept])

    assert degraded_fields == ["choice_2"]
    assert published[0] is text
    assert published[1] == {
        "type": "text_input",
        "field": "choice_2",
        "label": "Which?",
    }
    assert published[2] is kept


def test_the_list_form_reports_nothing_for_a_clean_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clean = [{"type": "text_input", "field": "note", "label": "Note"}]
    with caplog.at_level("WARNING"):
        published, degraded_fields = degrade_options_less_pickers(clean)
    assert published == clean
    assert degraded_fields == []
    assert not caplog.records


def test_the_list_form_warns_with_counts_and_never_the_field_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same payload discipline as ``_normalize_ask_user_interactions``'s own
    warnings: a model-controlled field name never reaches the log line."""

    with caplog.at_level("WARNING"):
        degrade_options_less_pickers(
            [
                {"type": "select_one", "field": "secret_field_name", "label": "L"},
                {"type": "text_input", "field": "note", "label": "Note"},
            ]
        )
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "1 options-less selection field(s)" in record.getMessage()
    assert "out of 2" in record.getMessage()
    assert "secret_field_name" not in record.getMessage()
    assert record.degraded == 1
    assert record.total == 2
