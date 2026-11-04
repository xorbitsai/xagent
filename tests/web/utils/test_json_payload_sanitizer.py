"""Contract tests for ``sanitize_json_payload`` (#1248).

The sanitizer is the write-side half of the ``jsonb`` migration: PostgreSQL's
``jsonb`` rejects the NUL escape and unpaired UTF-16 surrogates at INSERT
time, so every payload headed for a trace table has to shed those code
points first. The payload shapes pinned here mirror the seed data of
``tests/web/api/test_monitor_postgresql.py`` -- the read-side test that
documents why these exact code points are hazardous.

Two properties matter beyond simple replacement:

- a *valid* non-BMP character (an emoji in LLM output) must survive
  untouched -- dropping it would corrupt ordinary payloads; and
- an unchanged payload must come back as the *same object*, because the
  sanitizer runs on every trace write and almost every payload is clean --
  copying each one would be pure overhead on the hot path.
"""

from __future__ import annotations

import json
import math

from xagent.web.utils.json_payload_sanitizer import (
    REPLACEMENT_CHARACTER,
    sanitize_json_payload,
)

# Built with ``chr`` because an editor will happily turn an escape sequence
# into the character it names, and a lone surrogate is not encodable as
# UTF-8 (same convention as tests/web/api/test_monitor_postgresql.py).
NUL = chr(0x0000)
LONE_HIGH_SURROGATE = chr(0xD800)
LONE_LOW_SURROGATE = chr(0xDC00)
NON_BMP_CHAR = chr(0x1F600)
BACKSLASH = chr(92)
LITERAL_ESCAPE_TEXT = BACKSLASH + "u0000"


class TestOffendingCodePoints:
    def test_nul_is_replaced(self) -> None:
        assert sanitize_json_payload(f"a{NUL}b") == f"a{REPLACEMENT_CHARACTER}b"

    def test_lone_high_surrogate_is_replaced(self) -> None:
        result = sanitize_json_payload(f"x{LONE_HIGH_SURROGATE}y")
        assert result == f"x{REPLACEMENT_CHARACTER}y"

    def test_lone_low_surrogate_is_replaced(self) -> None:
        result = sanitize_json_payload(f"x{LONE_LOW_SURROGATE}y")
        assert result == f"x{REPLACEMENT_CHARACTER}y"

    def test_adjacent_surrogates_are_both_replaced(self) -> None:
        # A high+low pair of raw code points in a Python str is not an
        # astral character -- it cannot be encoded as UTF-8 -- so it is
        # replaced wholesale rather than treated as a valid pair.
        result = sanitize_json_payload(LONE_HIGH_SURROGATE + LONE_LOW_SURROGATE)
        assert result == REPLACEMENT_CHARACTER * 2

    def test_sanitized_output_is_utf8_encodable(self) -> None:
        payload = {
            "message": f"{NUL}{LONE_HIGH_SURROGATE}{LONE_LOW_SURROGATE}",
            "nested": [f"tail{LONE_LOW_SURROGATE}"],
        }
        cleaned = sanitize_json_payload(payload)
        # The exact property jsonb requires: the payload decodes to native
        # text. This raises UnicodeEncodeError on any surviving surrogate.
        json.dumps(cleaned, ensure_ascii=False).encode("utf-8")


class TestBenignPayloadsSurvive:
    def test_non_bmp_character_is_preserved(self) -> None:
        payload = f"emoji {NON_BMP_CHAR} stays"
        assert sanitize_json_payload(payload) is payload

    def test_literal_escape_text_is_preserved(self) -> None:
        # Text that merely *looks* like an escape: a backslash followed by
        # "u0000" is six ordinary characters, not a NUL.
        assert sanitize_json_payload(LITERAL_ESCAPE_TEXT) is LITERAL_ESCAPE_TEXT

    def test_clean_payload_returns_the_same_object(self) -> None:
        payload = {
            "model_name": "gpt-4o",
            "attempt": 1,
            "nested": {"items": ["a", "b"], "ok": True, "score": 1.5},
            "none": None,
        }
        assert sanitize_json_payload(payload) is payload

    def test_non_string_scalars_pass_through(self) -> None:
        for scalar in (1, 1.5, True, None):
            assert sanitize_json_payload(scalar) is scalar


class TestStructureTraversal:
    def test_nested_containers_are_sanitized(self) -> None:
        payload = {
            "outer": [{"inner": f"bad{NUL}"}, "clean"],
            "clean": "ok",
        }
        cleaned = sanitize_json_payload(payload)
        assert cleaned == {
            "outer": [{"inner": f"bad{REPLACEMENT_CHARACTER}"}, "clean"],
            "clean": "ok",
        }

    def test_dict_keys_are_sanitized(self) -> None:
        payload = {f"key{LONE_HIGH_SURROGATE}": "value"}
        cleaned = sanitize_json_payload(payload)
        assert cleaned == {f"key{REPLACEMENT_CHARACTER}": "value"}

    def test_tuple_stays_a_tuple(self) -> None:
        cleaned = sanitize_json_payload((f"a{NUL}", "b"))
        assert cleaned == (f"a{REPLACEMENT_CHARACTER}", "b")
        assert isinstance(cleaned, tuple)

    def test_clean_siblings_keep_identity_inside_changed_container(self) -> None:
        clean_branch = {"deep": ["untouched"]}
        payload = {"clean": clean_branch, "dirty": f"x{NUL}"}
        cleaned = sanitize_json_payload(payload)
        assert cleaned is not payload
        assert cleaned["clean"] is clean_branch


class TestFloatsJsonbWouldRetype:
    """jsonb re-renders numbers in plain notation, so a float written as
    ``1e+16`` reads back as an int. The checkpoint blob path re-hashes what
    it reads and compares against the write-time hash, so the two forms have
    to be reconciled before the hash is taken -- see the module docstring.
    """

    def test_float_at_the_threshold_becomes_an_int(self) -> None:
        cleaned = sanitize_json_payload(1e16)
        assert cleaned == 10000000000000000
        assert isinstance(cleaned, int)

    def test_large_float_becomes_an_int(self) -> None:
        assert sanitize_json_payload(1.5e20) == 150000000000000000000
        assert isinstance(sanitize_json_payload(1.5e20), int)

    def test_negative_large_float_becomes_an_int(self) -> None:
        assert sanitize_json_payload(-1e18) == -1000000000000000000
        assert isinstance(sanitize_json_payload(-1e18), int)

    def test_float_below_the_threshold_is_untouched(self) -> None:
        # repr writes this one in plain notation, so json and jsonb agree.
        value = 1e15
        assert sanitize_json_payload(value) is value

    def test_ordinary_floats_are_untouched(self) -> None:
        for value in (0.1, 3.14, 2.0, 1e-10, 0.0):
            assert sanitize_json_payload(value) is value

    def test_negative_zero_is_normalized_to_positive_zero(self) -> None:
        """PostgreSQL numeric has no signed zero, so jsonb renders -0.0 as
        0.0 while json.dumps writes "-0.0". Without this the write-time
        hash of a payload carrying -0.0 would not match what comes back."""
        cleaned = sanitize_json_payload(-0.0)
        assert json.dumps(cleaned) == "0.0"
        assert math.copysign(1.0, cleaned) > 0

    def test_negative_zero_is_normalized_when_nested(self) -> None:
        cleaned = sanitize_json_payload({"outer": [{"n": -0.0}]})
        assert json.dumps(cleaned, sort_keys=True) == '{"outer": [{"n": 0.0}]}'

    def test_positive_zero_keeps_its_identity(self) -> None:
        """The negative-zero branch must not turn every zero into a copy --
        +0.0 already matches what jsonb hands back."""
        value = 0.0
        assert sanitize_json_payload(value) is value

    def test_bools_are_not_treated_as_numbers(self) -> None:
        # bool subclasses int, not float, so the float branch never sees
        # one; this pins the outcome rather than the mechanism.
        for value in (True, False):
            assert sanitize_json_payload(value) is value

    def test_normalization_reaches_nested_values(self) -> None:
        cleaned = sanitize_json_payload({"outer": [{"n": 1e16}]})
        assert cleaned == {"outer": [{"n": 10000000000000000}]}
        assert isinstance(cleaned["outer"][0]["n"], int)

    def test_canonical_form_is_stable_under_a_json_round_trip(self) -> None:
        """The property the blob hash depends on: re-serializing what the
        database returns must reproduce the text that was stored."""
        cleaned = sanitize_json_payload({"n": 1e16})
        stored = json.dumps(cleaned, sort_keys=True)
        # json.loads models what psycopg2 does with the jsonb column's
        # plain-notation output.
        assert json.dumps(json.loads(stored), sort_keys=True) == stored


class TestCleanPathAllocatesNothing:
    """The hot-path contract the docstring states: a clean payload is not
    merely returned unchanged, it is never copied on the way. Pinned because
    the obvious implementation (build a copy, then decide whether to return
    it) satisfies every other test in this file while doing the work anyway.
    """

    def test_clean_nested_payload_keeps_every_container_identity(self) -> None:
        inner_list = ["a", "b"]
        inner_dict = {"items": inner_list, "score": 1.5}
        payload = {"model_name": "gpt-4o", "nested": inner_dict}

        cleaned = sanitize_json_payload(payload)

        assert cleaned is payload
        assert cleaned["nested"] is inner_dict
        assert cleaned["nested"]["items"] is inner_list

    def test_unchanged_siblings_are_reused_not_rebuilt(self) -> None:
        """Even when a copy is unavoidable, the branches that needed no edit
        are carried over by reference rather than rebuilt."""
        before = {"deep": ["untouched"]}
        after = {"also": "clean"}
        payload = {"before": before, "dirty": f"x{NUL}", "after": after}

        cleaned = sanitize_json_payload(payload)

        assert cleaned is not payload
        # The prefix backfilled at the first change and the suffix walked
        # afterwards must both be the original objects.
        assert cleaned["before"] is before
        assert cleaned["after"] is after

    def test_dict_key_order_is_preserved_across_a_copy(self) -> None:
        payload = {"first": 1, "dirty": f"x{NUL}", "last": 3}

        cleaned = sanitize_json_payload(payload)

        assert list(cleaned) == ["first", "dirty", "last"]

    def test_list_order_is_preserved_across_a_copy(self) -> None:
        payload = ["head", f"mid{NUL}", "tail"]

        cleaned = sanitize_json_payload(payload)

        assert cleaned == ["head", f"mid{REPLACEMENT_CHARACTER}", "tail"]
