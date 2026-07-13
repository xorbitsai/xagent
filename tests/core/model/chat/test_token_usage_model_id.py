"""Token-usage details carry model_id, disambiguating identically-named models."""

from typing import Any

import pytest

from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.router import RouterLLM
from xagent.core.model.chat.token_context import (
    TokenContextManager,
    add_token_usage,
    aggregate_token_usage_by_model,
)


class _UsageReportingLLM(BaseLLM):
    def __init__(self, model_name: str, input_tokens: int, output_tokens: int) -> None:
        self._reported_model_name = model_name
        self._model_id = f"router:{model_name}"
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    @property
    def abilities(self) -> list[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        return self._reported_model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        del messages, temperature, max_tokens, tools, tool_choice
        del response_format, thinking, output_config, kwargs
        add_token_usage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model=self.model_name,
            model_id=self.model_id,
            call_type="chat",
        )
        return "ok"


def test_create_base_llm_stamps_model_id():
    """create_base_llm stamps ChatModelConfig.id so adapters report it."""
    config = ChatModelConfig(
        id="deepseek-plat-abc",
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        api_key="test-api-key",
    )
    llm = create_base_llm(config)
    # The retry wrapper delegates attribute access to the inner LLM.
    assert llm._inner.model_id == "deepseek-plat-abc"
    assert llm.model_id == "deepseek-plat-abc"


def test_add_token_usage_records_model_id():
    with TokenContextManager() as mgr:
        add_token_usage(
            input_tokens=10,
            output_tokens=5,
            model="deepseek-v4-flash",
            model_id="platform-ds",
            call_type="chat",
        )
        details = mgr.get_usage().details

    by_type = {d["type"]: d for d in details}
    assert by_type["input"]["model_id"] == "platform-ds"
    assert by_type["input"]["model"] == "deepseek-v4-flash"
    assert by_type["output"]["model_id"] == "platform-ds"


def test_model_id_defaults_empty_when_absent():
    with TokenContextManager() as mgr:
        add_token_usage(input_tokens=3, model="m", call_type="chat")
        details = mgr.get_usage().details

    assert details[0]["model_id"] == ""
    assert details[0]["cached_tokens"] == 0


def test_add_token_usage_records_cached_tokens():
    with TokenContextManager() as mgr:
        add_token_usage(
            input_tokens=100,
            model="m",
            model_id="platform/x",
            call_type="chat",
            cached_input_tokens=40,
        )
        details = mgr.get_usage().details

    inp = next(d for d in details if d["type"] == "input")
    assert inp["tokens"] == 100
    assert inp["cached_tokens"] == 40


def test_cached_input_tokens_helper():
    from xagent.core.model.chat.basic.openai import _cached_input_tokens

    class DeepSeekUsage:
        prompt_cache_hit_tokens = 30

    class Details:
        cached_tokens = 25

    class OpenAIUsage:
        prompt_tokens_details = Details()

    class NoCache:
        pass

    assert _cached_input_tokens(DeepSeekUsage()) == 30  # deepseek field
    assert _cached_input_tokens(OpenAIUsage()) == 25  # openai/dashscope field
    assert _cached_input_tokens(NoCache()) == 0
    assert _cached_input_tokens(None) == 0


def test_aggregate_token_usage_by_model_prefers_model_id_and_sorts_by_total():
    details = [
        {
            "type": "input",
            "tokens": 100,
            "model": "shared-name",
            "model_id": "main-model",
        },
        {
            "type": "output",
            "tokens": 25,
            "model": "shared-name",
            "model_id": "main-model",
        },
        {
            "type": "input",
            "tokens": 20,
            "model": "shared-name",
            "model_id": "compact-model",
        },
        {
            "type": "output",
            "tokens": 5,
            "model": "shared-name",
            "model_id": "compact-model",
        },
    ]

    assert aggregate_token_usage_by_model(details) == [
        {
            "model_id": "main-model",
            "model_name": "shared-name",
            "input_tokens": 100,
            "output_tokens": 25,
        },
        {
            "model_id": "compact-model",
            "model_name": "shared-name",
            "input_tokens": 20,
            "output_tokens": 5,
        },
    ]


def test_aggregate_token_usage_by_model_keeps_legacy_unattributed_tokens():
    details = [
        {"type": "input", "tokens": "12"},
        {"type": "output", "tokens": 3},
        {"type": "other", "tokens": 99},
        "invalid",
    ]

    assert aggregate_token_usage_by_model(details) == [
        {
            "model_id": "",
            "model_name": "",
            "input_tokens": 12,
            "output_tokens": 3,
        }
    ]


def test_aggregate_token_usage_by_model_merges_unique_legacy_name_group():
    details = [
        {"type": "input", "tokens": 100, "model": "gpt-4o"},
        {"type": "output", "tokens": 20, "model": "gpt-4o"},
        {
            "type": "input",
            "tokens": 50,
            "model": "gpt-4o",
            "model_id": "openai:gpt-4o",
        },
        {
            "type": "output",
            "tokens": 10,
            "model": "gpt-4o",
            "model_id": "openai:gpt-4o",
        },
    ]

    assert aggregate_token_usage_by_model(details) == [
        {
            "model_id": "openai:gpt-4o",
            "model_name": "gpt-4o",
            "input_tokens": 150,
            "output_tokens": 30,
        }
    ]


def test_aggregate_token_usage_by_model_keeps_ambiguous_legacy_name_group():
    details = [
        {"type": "input", "tokens": 30, "model": "shared-name"},
        {
            "type": "input",
            "tokens": 20,
            "model": "shared-name",
            "model_id": "main-model",
        },
        {
            "type": "input",
            "tokens": 10,
            "model": "shared-name",
            "model_id": "compact-model",
        },
    ]

    assert aggregate_token_usage_by_model(details) == [
        {
            "model_id": "",
            "model_name": "shared-name",
            "input_tokens": 30,
            "output_tokens": 0,
        },
        {
            "model_id": "main-model",
            "model_name": "shared-name",
            "input_tokens": 20,
            "output_tokens": 0,
        },
        {
            "model_id": "compact-model",
            "model_name": "shared-name",
            "input_tokens": 10,
            "output_tokens": 0,
        },
    ]


@pytest.mark.asyncio
async def test_openrouter_auto_usage_is_attributed_to_each_selected_model():
    selected_models = iter(["deepseek/deepseek-v4-flash", "anthropic/claude-opus-4.8"])
    token_counts = {
        "deepseek/deepseek-v4-flash": (100, 50),
        "anthropic/claude-opus-4.8": (20, 10),
    }

    def resolve(model_name: str) -> BaseLLM:
        return _UsageReportingLLM(model_name, *token_counts[model_name])

    router = RouterLLM(model_name="auto", downstream_resolver=resolve)
    router.context_window = 128_000

    async def select_model(_prompt: str) -> str:
        return next(selected_models)

    router._select_model = select_model  # type: ignore[method-assign]

    with TokenContextManager() as manager:
        await router.chat([{"role": "user", "content": "first"}])
        await router.chat([{"role": "user", "content": "second"}])
        model_usage = aggregate_token_usage_by_model(manager.get_usage().details)

    assert model_usage == [
        {
            "model_id": "router:deepseek/deepseek-v4-flash",
            "model_name": "deepseek/deepseek-v4-flash",
            "input_tokens": 100,
            "output_tokens": 50,
        },
        {
            "model_id": "router:anthropic/claude-opus-4.8",
            "model_name": "anthropic/claude-opus-4.8",
            "input_tokens": 20,
            "output_tokens": 10,
        },
    ]
