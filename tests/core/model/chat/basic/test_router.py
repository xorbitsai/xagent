"""Tests for RouterLLM provider compatibility retries."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.router import RouterLLM
from xagent.core.model.chat.types import ChunkType, StreamChunk

_OPENROUTER_TOOL_CHOICE_ERROR = (
    "OpenAI API error (404): Error code: 404 - {'error': {'message': "
    "\"No endpoints found that support the provided 'tool_choice' value.\"}}"
)
_THINKING_TOOL_CHOICE_ERROR = (
    "OpenAI bad request (400): Thinking mode does not support this tool_choice"
)


def _tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Answer the user",
            "parameters": {"type": "object", "properties": {}},
        },
    }


class _ToolChoiceRetryLLM(BaseLLM):
    def __init__(self) -> None:
        self.chat_tool_choices: list[str | dict[str, Any] | None] = []
        self.stream_tool_choices: list[str | dict[str, Any] | None] = []

    @property
    def abilities(self) -> list[str]:
        return ["chat", "tool_calling"]

    @property
    def model_name(self) -> str:
        return "z-ai/glm-5.2"

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
    ) -> str | dict[str, Any]:
        del messages, temperature, max_tokens, tools, response_format
        del thinking, output_config, kwargs
        self.chat_tool_choices.append(tool_choice)
        if tool_choice == "required":
            raise RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR)
        return "ok"

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        del messages, temperature, max_tokens, tools, response_format
        del thinking, output_config, kwargs
        self.stream_tool_choices.append(tool_choice)
        if tool_choice == "required":
            raise RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR)
        yield StreamChunk(type=ChunkType.TOKEN, content="ok", delta="ok")


class _ScriptedChatLLM(BaseLLM):
    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        self.tool_choices: list[str | dict[str, Any] | None] = []
        self.thinking_values: list[dict[str, Any] | None] = []

    @property
    def abilities(self) -> list[str]:
        return ["chat", "tool_calling"]

    @property
    def model_name(self) -> str:
        return "z-ai/glm-5.2"

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
    ) -> str | dict[str, Any]:
        del messages, temperature, max_tokens, tools, response_format
        del output_config, kwargs
        self.tool_choices.append(tool_choice)
        self.thinking_values.append(thinking)
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        return "ok"


async def _select_glm(_prompt: str) -> str:
    return "z-ai/glm-5.2"


@pytest.mark.asyncio
async def test_prepare_for_call_reuses_route_and_exposes_profile_context_window(
    monkeypatch,
):
    downstream = _ScriptedChatLLM([])
    downstream._model_id = "configured-openrouter-model"
    router = RouterLLM(
        timeout=42.0,
        downstream_resolver=lambda _model_id: downstream,
    )
    selected: list[str] = []

    async def select_model(prompt: str) -> str:
        selected.append(prompt)
        return "deepseek/deepseek-v4-flash"

    monkeypatch.setattr(router, "_select_model", select_model)
    monkeypatch.setattr(
        router,
        "_profile_context_window",
        lambda _model_id: 1_048_576,
    )

    prepared = await router.prepare_for_call(
        [{"role": "user", "content": "make a podcast"}]
    )

    assert prepared.model_name == "deepseek/deepseek-v4-flash"
    assert prepared.model_id == "configured-openrouter-model"
    assert prepared.timeout == 42.0
    assert prepared.context_window == 1_048_576
    assert await prepared.chat([{"role": "user", "content": "continue"}]) == "ok"
    assert selected == ["make a podcast"]


@pytest.mark.asyncio
async def test_prepare_for_call_handles_missing_router_context_window(monkeypatch):
    downstream = _ScriptedChatLLM([])
    router = RouterLLM(downstream_resolver=lambda _model_id: downstream)

    async def select_model(_prompt: str) -> str:
        return "deepseek/deepseek-v4-flash"

    monkeypatch.delattr(BaseLLM, "context_window")
    monkeypatch.setattr(router, "_select_model", select_model)
    monkeypatch.setattr(
        router,
        "_profile_context_window",
        lambda _model_id: 1_048_576,
    )

    prepared = await router.prepare_for_call(
        [{"role": "user", "content": "make a podcast"}]
    )

    assert prepared.context_window == 1_048_576


def test_profile_context_window_returns_none_when_catalog_lookup_fails(
    monkeypatch, caplog
):
    def fail_service_lookup():
        raise RuntimeError("profile catalog unavailable")

    monkeypatch.setattr(
        "xagent.core.model.chat.basic.router._get_service",
        fail_service_lookup,
    )

    assert RouterLLM._profile_context_window("test/model") is None
    assert "Could not resolve xrouter context window for test/model" in caplog.text


@pytest.mark.asyncio
async def test_router_chat_relaxes_required_tool_choice_on_openrouter_endpoint_error(
    monkeypatch,
):
    llm = _ToolChoiceRetryLLM()
    router = RouterLLM(downstream_resolver=lambda _model_id: llm)

    async def select_model(_prompt: str) -> str:
        return "z-ai/glm-5.2"

    monkeypatch.setattr(router, "_select_model", select_model)

    result = await router.chat(
        [{"role": "user", "content": "score?"}],
        tools=[_tool_schema()],
        tool_choice="required",
    )

    assert result == "ok"
    assert llm.chat_tool_choices == ["required", "auto"]


@pytest.mark.asyncio
async def test_router_stream_relaxes_required_tool_choice_before_first_chunk(
    monkeypatch,
):
    llm = _ToolChoiceRetryLLM()
    router = RouterLLM(downstream_resolver=lambda _model_id: llm)

    async def select_model(_prompt: str) -> str:
        return "z-ai/glm-5.2"

    monkeypatch.setattr(router, "_select_model", select_model)

    chunks = [
        chunk
        async for chunk in router.stream_chat(
            [{"role": "user", "content": "score?"}],
            tools=[_tool_schema()],
            tool_choice="required",
        )
    ]

    assert [chunk.delta for chunk in chunks] == ["ok"]
    assert llm.stream_tool_choices == ["required", "auto"]


@pytest.mark.asyncio
async def test_router_chat_does_not_relax_auto_tool_choice_on_openrouter_error(
    monkeypatch,
):
    llm = _ScriptedChatLLM([_OPENROUTER_TOOL_CHOICE_ERROR])
    router = RouterLLM(downstream_resolver=lambda _model_id: llm)
    monkeypatch.setattr(router, "_select_model", _select_glm)

    with pytest.raises(RuntimeError, match="No endpoints found"):
        await router.chat(
            [{"role": "user", "content": "score?"}],
            tools=[_tool_schema()],
            tool_choice="auto",
        )

    assert llm.tool_choices == ["auto"]


@pytest.mark.asyncio
async def test_router_chat_propagates_non_matching_errors_without_retry(monkeypatch):
    llm = _ScriptedChatLLM(["different provider error"])
    router = RouterLLM(downstream_resolver=lambda _model_id: llm)
    monkeypatch.setattr(router, "_select_model", _select_glm)

    with pytest.raises(RuntimeError, match="different provider error"):
        await router.chat(
            [{"role": "user", "content": "score?"}],
            tools=[_tool_schema()],
            tool_choice="required",
        )

    assert llm.tool_choices == ["required"]


@pytest.mark.asyncio
async def test_router_chat_does_not_retry_same_action_twice(monkeypatch):
    llm = _ScriptedChatLLM([_THINKING_TOOL_CHOICE_ERROR, _THINKING_TOOL_CHOICE_ERROR])
    router = RouterLLM(downstream_resolver=lambda _model_id: llm)
    monkeypatch.setattr(router, "_select_model", _select_glm)

    with pytest.raises(RuntimeError, match="Thinking mode does not support"):
        await router.chat(
            [{"role": "user", "content": "score?"}],
            tools=[_tool_schema()],
            tool_choice="required",
            thinking={"type": "disabled", "enable": False},
        )

    assert llm.tool_choices == ["required", "required"]
    assert llm.thinking_values == [
        {"type": "disabled", "enable": False},
        {"type": "disabled", "enable": False},
    ]


@pytest.mark.asyncio
async def test_router_chat_can_chain_thinking_and_tool_choice_retries(monkeypatch):
    llm = _ScriptedChatLLM([_THINKING_TOOL_CHOICE_ERROR, _OPENROUTER_TOOL_CHOICE_ERROR])
    router = RouterLLM(downstream_resolver=lambda _model_id: llm)
    monkeypatch.setattr(router, "_select_model", _select_glm)

    result = await router.chat(
        [{"role": "user", "content": "score?"}],
        tools=[_tool_schema()],
        tool_choice="required",
        thinking={"type": "enabled", "enable": True},
    )

    assert result == "ok"
    assert llm.tool_choices == ["required", "required", "auto"]
    assert llm.thinking_values == [
        {"type": "enabled", "enable": True},
        {"type": "disabled", "enable": False},
        {"type": "disabled", "enable": False},
    ]
