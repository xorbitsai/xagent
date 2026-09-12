"""Database trace cleaning preserves its wire data and JSON validation."""

import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from xagent.web.services import trace_handlers
from xagent.web.services.trace_handlers import DatabaseTraceHandler
from xagent.web.utils.json_payload_sanitizer import sanitize_json_payload


@pytest.mark.parametrize(
    "value",
    [
        "",
        "ordinary ASCII output" * 1000,
        "中文、emoji 😀🚀\n" * 1000,
        "".join(chr(code) for code in range(256)),
        "a\x00\x01b\t\n\rc\x0b\x0c\x1fd",
        "surrogates \ud800 \udfff and literal \\u0000",
    ],
)
def test_string_cleaning_matches_previous_character_filter(value):
    expected = "".join(char for char in value if ord(char) >= 32 or char in "\n\r\t")
    assert DatabaseTraceHandler(task_id=1)._serialize_data_for_json(value) == expected


def test_nested_values_and_payload_size_are_preserved(monkeypatch):
    observe = Mock()
    monkeypatch.setattr(trace_handlers, "observe_value", observe)
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)

    class Model:
        def model_dump(self):
            return {"text": "model\x00 text"}

    class Record:
        def to_dict(self):
            return {"text": "record\x01 text"}

    payload = {
        "values": [
            "文本\x00",
            b"bytes\x01\t",
            b"\xff",
            when,
            when.replace(tzinfo=None),
        ],
        "tuple": (Model(), Record(), None, True, 3.5),
        "key\x00": "keys remain untouched at this layer",
    }
    result = DatabaseTraceHandler(task_id=1)._serialize_data_for_json(payload)
    assert result == {
        "values": ["文本", "bytes\t", "<bytes: 1>", when.timestamp(), when.timestamp()],
        "tuple": [{"text": "model text"}, {"text": "record text"}, None, True, 3.5],
        "key\x00": "keys remain untouched at this layer",
    }
    assert payload["values"][0] == "文本\x00"
    observe.assert_called_once_with(
        "xagent.trace.payload.size", len(json.dumps(result)), unit="By"
    )


def test_invalid_json_still_uses_error_payload(monkeypatch):
    observe = Mock()
    monkeypatch.setattr(trace_handlers, "observe_value", observe)
    result = DatabaseTraceHandler(task_id=1)._serialize_data_for_json({"bad": object()})
    assert result["_serialization_error"] == "Failed to serialize dict"
    assert result["_original_type"] == "dict"
    json.dumps(result)
    observe.assert_not_called()


def test_jsonb_normalization_remains_at_storage_boundary():
    result = DatabaseTraceHandler(task_id=1)._serialize_data_for_json(
        {"bad\x00key": "a\ud800\x00b", "large_float": 1e16}
    )
    assert result == {"bad\x00key": "a\ud800b", "large_float": 1e16}
    normalized = sanitize_json_payload(result)
    assert normalized == {"bad�key": "a�b", "large_float": 10000000000000000}
    assert isinstance(normalized["large_float"], int)
