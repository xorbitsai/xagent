"""Direct coverage for ``serialize_trace_data``.

This is the module-level function ``WebSocketTraceHandler._serialize_data``
now delegates to (see ``ws_trace_handlers.py``). The whole point of lifting
it out of the class was to let a second caller -- the v1 SSE
content-projection layer -- reuse the exact same pass without going through
``WebSocketTraceHandler``. These tests call it directly, the way that
second caller will, instead of only exercising it indirectly through the
handler class (as ``tests/core/agent/test_react_clarification_draft.py``
already does).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from xagent.web.api.ws_trace_handlers import serialize_trace_data


def test_recursively_serializes_nested_structures_into_json_safe_data() -> None:
    """Nested dicts/lists/datetimes/bytes all come out JSON-encodable."""

    payload = {
        "when": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "nested": {
            "items": [1, "two", {"three": 3.0}],
            "raw": b"hello",
        },
        "tuple_becomes_list": (1, 2, 3),
    }

    result = serialize_trace_data(payload)

    # No exception on json.dumps means every leaf is now JSON-safe.
    json.dumps(result)
    assert (
        result["when"]
        == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    )
    assert result["nested"]["items"] == [1, "two", {"three": 3.0}]
    assert result["nested"]["raw"] == "hello"
    assert result["tuple_becomes_list"] == [1, 2, 3]


def test_serializes_model_dump_and_to_dict_objects() -> None:
    """Pydantic-style ``model_dump()`` and plain ``to_dict()`` objects
    are unwrapped into their dict form, then recursively serialized."""

    class ModelDumpLike:
        def model_dump(self) -> dict:
            return {
                "kind": "model_dump",
                "created": datetime(2024, 1, 1, tzinfo=timezone.utc),
            }

    class ToDictLike:
        def to_dict(self) -> dict:
            return {"kind": "to_dict"}

    result = serialize_trace_data({"a": ModelDumpLike(), "b": ToDictLike()})

    json.dumps(result)
    assert result["a"]["kind"] == "model_dump"
    assert result["b"] == {"kind": "to_dict"}


def test_strips_control_characters_but_keeps_newline_tab_and_cr() -> None:
    dirty = "keep\nthis\tand\rthis" + "\x00" + "\x01" + "drop-that"

    result = serialize_trace_data({"text": dirty})

    assert result["text"] == "keep\nthis\tand\rthisdrop-that"


def test_serialize_trace_data_strips_control_chars_same_as_the_old_per_char_loop() -> (
    None
):
    """``clean_string`` (``ws_trace_handlers.serialize_trace_data``'s
    nested helper) moved from a per-character loop to ``str.translate``
    against a module-level table purely for speed -- this pins that the
    output is unchanged for every relevant class of input: every
    control codepoint 0-31 except tab/newline/CR (all removed, same as
    before), tab/newline/CR themselves (all kept), DEL and the C1
    control range 128-159 (both kept, same as before -- the old loop's
    ``ord(char) >= 32`` check let both through untouched), and ordinary
    multi-byte content (CJK, an astral-plane emoji) untouched.
    """

    def old_clean_string(value: str) -> str:
        # Reference implementation: the per-character loop this module
        # used before the ``str.translate`` change, inlined here so this
        # test pins behavior against the pre-change semantics rather
        # than against the (also changed) production code. The old code
        # also had a redundant NUL-stripping ``replace`` call right
        # before this filter; omitted here since it is a no-op given the
        # filter below already drops every codepoint < 32, NUL included.
        return "".join(char for char in value if ord(char) >= 32 or char in "\n\r\t")

    control_chars = "".join(chr(c) for c in range(32))
    del_and_c1 = "".join(chr(c) for c in range(127, 160))
    sample = (
        control_chars
        + "\t\n\r"
        + del_and_c1
        + "中文测试"
        + "\U0001f600\U0001f680"  # astral-plane emoji
        + "plain ascii text"
    )

    expected = old_clean_string(sample)
    actual = serialize_trace_data({"value": sample})["value"]
    assert actual == expected
    # Pin the two concrete claims separately so a regression in either
    # direction fails on its own assertion, not just the aggregate diff.
    assert "\x00" not in actual
    for c in "\t\n\r":
        assert actual.count(c) == sample.count(c)
    assert del_and_c1 in actual
    assert "中文测试" in actual
    assert "\U0001f600\U0001f680" in actual


def test_falls_back_to_serialization_error_stub_instead_of_raising() -> None:
    """An object the serializer has no unwrap path for (no ``model_dump``,
    ``to_dict``, or ``dict``, and not JSON-native) survives the recursive
    pass unchanged, then fails ``json.dumps`` -- at which point
    ``serialize_trace_data`` must degrade to the ``_serialization_error``
    stub rather than propagating the ``TypeError``."""

    class Unserializable:
        pass

    result = serialize_trace_data({"bad": Unserializable()})

    # Must not raise, and must not silently produce non-JSON-safe output.
    json.dumps(result)
    assert result["_serialization_error"] == "Failed to serialize dict"
    assert result["_original_type"] == "dict"
