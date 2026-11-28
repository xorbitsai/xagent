from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.agent.pattern.react.tool_call_observation import (
    observe_tool_call_batch,
)
from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.base import ChatRequestIdentity


def _tool(name: str, **schema: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "parameters": schema}}


def _call(name: str, arguments: Any) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": arguments}}


def _llm(provider: str = "zhipu", model: str = "glm-5") -> SimpleNamespace:
    return SimpleNamespace(
        request_identity=ChatRequestIdentity(provider, model, "test", "zhipu")
    )


READ = _tool(
    "read_file",
    type="object",
    properties={"path": {"type": "string"}},
    required=["path"],
)
LIST = _tool("list_files", type="object", properties={}, additionalProperties=False)


def test_large_valid_batch_and_blank_parameterless_calls() -> None:
    calls = [
        _call("read_file", json.dumps({"path": f"attachment-{i}.txt"}))
        for i in range(9)
    ]
    result = observe_tool_call_batch(_llm(), {"tool_calls": calls}, [READ, LIST])[
        "tool_call_observation"
    ]
    assert result["call_count"] == 9
    assert result["distinct_tool_count"] == result["called_offered_tool_count"] == 1
    assert result["offered_tool_count"] == 2
    assert result["findings"] == []

    calls.append(_call("list_files", "  "))
    result = observe_tool_call_batch(_llm(), {"tool_calls": calls}, [READ, LIST])[
        "tool_call_observation"
    ]
    assert result["called_offered_tool_count"] == 2
    assert result["findings"] == []


def test_observation_distinguishes_argument_failures_and_redacts_values() -> None:
    calls = [
        _call("read_file", '{"path":"sensitive-value'),
        _call("read_file", '["sensitive-value"]'),
        _call("read_file", '{"path":123}'),
        _call("read_file", ""),
        _call("sensitive-tool-name", "{}"),
        _call("", "{}"),
    ]
    result = observe_tool_call_batch(_llm(), {"tool_calls": calls}, [READ, LIST])[
        "tool_call_observation"
    ]
    assert result["findings"] == [
        {"call_index": 0, "reason": "invalid_json"},
        {"call_index": 1, "reason": "arguments_not_object"},
        {"call_index": 2, "reason": "schema_mismatch"},
        {"call_index": 3, "reason": "schema_mismatch"},
        {"call_index": 4, "reason": "tool_not_offered"},
        {"call_index": 5, "reason": "missing_tool_name"},
    ]
    assert "sensitive" not in json.dumps(result)


@pytest.mark.parametrize(
    ("provider", "model", "profile"),
    [
        ("openrouter", "z-ai/glm-5", "glm-5"),
        ("zhipuai_coding_plan", "glm-5", "glm-5"),
        ("openrouter", "moonshotai/kimi-k2.5", "kimi-k2.5"),
    ],
)
def test_configured_provider_aliases_survive_retry_wrapper(
    provider: str, model: str, profile: str
) -> None:
    llm = create_base_llm(
        ChatModelConfig(
            id="saved-model",
            model_name=model,
            model_provider=provider,
            api_key="test-key",
        )
    )
    observation = observe_tool_call_batch(
        llm, {"tool_calls": [_call("list_files", "")]}, [LIST]
    )
    result = observation["tool_call_observation"]
    assert result["profile"] == profile
    assert result["identity"] == {
        "provider": provider.replace("_", "-"),
        "model_name": model,
        "configuration_id": "saved-model",
        "protocol": "openai_chat_completions",
    }


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("zhipu", "glm-5.2"),
        ("zhipu", "glm-5-custom"),
        ("openrouter", "glm-5"),
        ("openai", "glm-5"),
    ],
)
def test_other_targets_have_no_observation(provider: str, model: str) -> None:
    assert (
        observe_tool_call_batch(
            _llm(provider, model), {"tool_calls": [_call("read_file", "{")]}, [READ]
        )
        == {}
    )


@pytest.mark.parametrize(
    "markers", [{"finish_reason": "length"}, {"stream_aborted": "stalled"}]
)
def test_incomplete_responses_skip_observation(markers: dict[str, str]) -> None:
    assert (
        observe_tool_call_batch(
            _llm(), {"tool_calls": [_call("read_file", "{")], **markers}, [READ]
        )
        == {}
    )


def test_schema_references_are_local_only() -> None:
    calls = [_call("read_file", '{"path":123}')]
    local = _tool(
        "read_file",
        **{"$defs": {"read": READ["function"]["parameters"]}, "$ref": "#/$defs/read"},
    )
    external = _tool("read_file", **{"$ref": "https://example.invalid/secret.json"})
    for tool, expected in [
        (local, "schema_mismatch"),
        (external, "schema_unavailable"),
    ]:
        result = observe_tool_call_batch(_llm(), {"tool_calls": calls}, [tool])[
            "tool_call_observation"
        ]
        assert result["findings"] == [{"call_index": 0, "reason": expected}]
