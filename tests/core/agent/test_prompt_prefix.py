from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime
from xagent.core.agent.prompt_prefix import (
    OBSERVE_PROMPT_PREFIX_KEY,
    PROMPT_PREFIX_KEY,
    PromptPrefixObserver,
)
from xagent.core.agent.trace import normalize_llm_trace_payload


def observe(observer, messages, tools=None, **kwargs):
    return observer.observe(
        messages,
        tools,
        scope=kwargs.get("scope", ("task", "model")),
        call_kind=kwargs.get("call_kind", "main"),
    )


def test_append_only_then_in_place_history_edit():
    observer = PromptPrefixObserver()
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
    ]
    assert observe(observer, messages) == {"change": "initial"}
    messages.append({"role": "assistant", "content": "working"})
    result = observe(observer, messages)
    assert result["change"] == "append_only"
    assert result["common_prefix_messages"] == 2
    assert result["first_changed_message_index"] is None
    messages[0]["content"] = "new rules"
    result = observe(observer, messages)
    assert result["change"] == "messages_changed"
    assert result["first_changed_message_index"] == 0
    assert result["common_prefix_messages"] == 0


def test_removed_messages_and_identical_requests():
    observer = PromptPrefixObserver()
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    observe(observer, messages)
    assert observe(observer, copy.deepcopy(messages))["change"] == "identical"
    result = observe(observer, messages[:1])
    assert result["change"] == "messages_removed"
    assert result["common_prefix_messages"] == 1


def test_tool_schema_and_order_changes_do_not_mutate_payloads():
    observer = PromptPrefixObserver()
    tools = [{"function": {"name": "one"}}, {"function": {"name": "two"}}]
    original = copy.deepcopy(tools)
    observe(observer, [], tools)
    assert observe(observer, [], copy.deepcopy(tools))["tools_changed"] is False
    assert observe(observer, [], list(reversed(tools)))["change"] == "tools_changed"
    tools[0]["function"]["description"] = "changed"
    assert observe(observer, [], tools)["tools_changed"] is True
    assert "description" not in original[0]["function"]
    assert tools[0]["function"]["name"] == "one"


def test_auxiliary_call_does_not_overwrite_main_baseline():
    observer = PromptPrefixObserver()
    original = [{"role": "user", "content": "main"}]
    observe(observer, original)
    observe(
        observer, [{"role": "user", "content": "summarize"}], call_kind="compaction"
    )
    assert observe(observer, original)["change"] == "identical"
    assert observe(observer, original, scope=("different-task", "model")) == {
        "change": "scope_changed"
    }


def test_non_json_observation_does_not_break_calls_or_reuse_old_baseline():
    observer = PromptPrefixObserver()
    observe(observer, [])
    assert observe(observer, [{"content": object()}]) == {"change": "unavailable"}
    assert observe(observer, []) == {"change": "initial"}


def test_snapshot_and_telemetry_do_not_retain_content():
    observer = PromptPrefixObserver()
    secret = "private invoice contents"
    result = observe(observer, [{"role": "user", "content": secret}])
    assert secret not in repr(observer._previous)
    assert secret not in json.dumps(result)
    assert all(
        isinstance(digest, bytes) and len(digest) == 32
        for digest in observer._previous["main"][2]
    )


class Trace:
    def __init__(self):
        self.events: list[dict[str, Any]] = []

    async def trace_event(self, event_type, **kwargs):
        self.events.append(kwargs["data"])


@pytest.mark.asyncio
async def test_runtime_opt_in_pairs_start_and_end_observation():
    trace = Trace()
    runtime = PatternRuntime(tracer=trace)
    context = ExecutionContext(metadata={OBSERVE_PROMPT_PREFIX_KEY: True})
    metadata: dict[str, Any] = {}
    await runtime.on_llm_start(context=context, messages=[], metadata=metadata)
    await runtime.on_llm_end(
        context=context,
        response={"usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        metadata=metadata,
    )
    assert trace.events[0][PROMPT_PREFIX_KEY] == {"change": "initial"}
    assert trace.events[1][PROMPT_PREFIX_KEY] == trace.events[0][PROMPT_PREFIX_KEY]
    metadata = {}
    await runtime.on_llm_start(context=context, messages=[], metadata=metadata)
    assert metadata[PROMPT_PREFIX_KEY]["change"] == "identical"


@pytest.mark.asyncio
async def test_runtime_observation_is_off_by_default():
    runtime = PatternRuntime()
    await runtime.on_llm_start(
        context=ExecutionContext(), messages=[{"content": object()}]
    )
    assert runtime._prompt_prefix_observer is None


@pytest.mark.asyncio
async def test_absent_context_metadata_does_not_enable_or_break_observation():
    from types import SimpleNamespace

    runtime = PatternRuntime()
    await runtime.on_llm_start(context=SimpleNamespace(metadata=None), messages=[])
    assert runtime._prompt_prefix_observer is None


def test_prefix_metrics_survive_trace_normalization():
    observation = {"change": "messages_changed", "first_changed_message_index": 0}
    result = normalize_llm_trace_payload(
        {PROMPT_PREFIX_KEY: observation, "response": "large" * 10000}
    )
    assert result[PROMPT_PREFIX_KEY] == observation
