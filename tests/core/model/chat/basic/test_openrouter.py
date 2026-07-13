"""Test cases for OpenRouter LLM provider behavior."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from xagent.core.model.chat.basic import openrouter as openrouter_module
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.openrouter import OpenRouterLLM
from xagent.core.model.chat.error import retry_on
from xagent.core.model.chat.exceptions import LLMRetryableError
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.core.retry.strategy import FixedDelay
from xagent.core.retry.wrapper import create_retry_wrapper


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("deepseek/deepseek-v4-flash", True),
        ("openrouter/deepseek/deepseek-v4-flash", True),
        ("anthropic/claude-sonnet-4.6", False),
    ],
)
def test_openrouter_uses_deepseek_tool_protocol_only_for_deepseek_models(
    model_name, expected
):
    llm = OpenRouterLLM(model_name=model_name, api_key="test-key")

    assert llm._uses_deepseek_tool_protocol is expected


@pytest.mark.asyncio
async def test_openrouter_deepseek_marks_serialized_tool_protocol_retryable(
    mocker,
):
    message = SimpleNamespace(
        content="<｜｜DSML｜｜tool_calls>",
        tool_calls=None,
        reasoning_content=None,
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-invalid-protocol"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    with pytest.raises(
        LLMRetryableError,
        match="serialized_tool_call_content",
    ):
        await llm.chat(
            [{"role": "user", "content": "Use a tool"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )


@pytest.mark.asyncio
async def test_openrouter_deepseek_protocol_error_uses_shared_chat_retry(mocker):
    invalid_message = SimpleNamespace(
        content="<｜｜DSML｜｜tool_calls>",
        tool_calls=None,
        reasoning_content=None,
    )
    invalid_response = SimpleNamespace(
        choices=[SimpleNamespace(message=invalid_message)],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-invalid-protocol"},
    )
    tool_call = SimpleNamespace(
        id="call_route",
        type="function",
        function=SimpleNamespace(
            name="select_execution_pattern",
            arguments="{}",
        ),
    )
    valid_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tool_call]))
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-valid-protocol"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        invalid_response,
        valid_response,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    inner = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    llm = create_retry_wrapper(
        inner,
        BaseLLM,  # type: ignore[type-abstract]
        retry_methods={"chat"},
        strategy=FixedDelay(delay_ms=0),
        max_retries=2,
        retry_on=retry_on,
    )

    result = await llm.chat(
        [{"role": "user", "content": "Route this request"}],
        tools=_single_tool_schema(),
        tool_choice="required",
    )

    assert result["tool_calls"][0]["function"]["name"] == ("select_execution_pattern")
    assert mock_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_openrouter_deepseek_protocol_error_uses_shared_stream_retry(mocker):
    attempts = 0

    async def invalid_stream():
        yield StreamChunk(
            type=ChunkType.TOKEN,
            delta="Let me route this. <｜｜DSML｜｜tool_calls>",
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="stop")

    async def valid_stream():
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_route",
                    "type": "function",
                    "function": {
                        "name": "select_execution_pattern",
                        "arguments": "{}",
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="tool_calls")

    def fake_stream(**kwargs):
        nonlocal attempts
        del kwargs
        attempts += 1
        return invalid_stream() if attempts == 1 else valid_stream()

    inner = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    mocker.patch.object(
        inner,
        "_stream_chat_with_prefix_retry",
        side_effect=fake_stream,
    )
    llm = create_retry_wrapper(
        inner,
        BaseLLM,  # type: ignore[type-abstract]
        retry_methods={"stream_chat"},
        strategy=FixedDelay(delay_ms=0),
        max_retries=2,
        retry_on=retry_on,
    )

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Route this request"}],
            tools=_single_tool_schema(),
            tool_choice="required",
        )
    ]

    assert attempts == 2
    assert not any(chunk.is_protocol_error() for chunk in chunks)
    assert any(
        chunk.is_tool_call()
        and chunk.tool_calls[0]["function"]["name"] == "select_execution_pattern"
        for chunk in chunks
    )


def _deepseek_function_prefix_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        "Error code: 400 - {'error': {'message': 'Provider returned error'}}",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        ),
        body={
            "error": {
                "message": "Provider returned error",
                "code": 400,
                "metadata": {
                    "provider_name": "DeepSeek",
                    "raw": (
                        '{"error":{"message":'
                        '"Function call should not be used with prefix"}}'
                    ),
                },
            }
        },
    )


def _unrelated_bad_request() -> openai.BadRequestError:
    return openai.BadRequestError(
        "Error code: 400 - {'error': {'message': 'Unrelated invalid request'}}",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        ),
        body={"error": {"message": "Unrelated invalid request", "code": 400}},
    )


def _tool_call_history() -> list[dict]:
    return [
        {"role": "user", "content": "Generate music"},
        {
            "role": "assistant",
            "content": "I will generate the music first.",
            "tool_calls": [
                {
                    "id": "call_music",
                    "type": "function",
                    "function": {
                        "name": "generate_music",
                        "arguments": '{"prompt":"intro"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_music",
            "content": '{"success":true}',
        },
    ]


def _tool_call_history_with_trailing_progress() -> list[dict]:
    messages = _tool_call_history()
    messages[1]["content"] = ""
    messages.append(
        {
            "role": "assistant",
            "content": "Still working on the generated audio.",
        }
    )
    return messages


def _single_tool_schema(name: str = "select_execution_pattern") -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.parametrize(
    "tools",
    [
        [None],
        ["invalid"],
        [{}],
        [{"function": None}],
    ],
)
def test_openrouter_deepseek_preserves_required_for_malformed_single_tool(
    tools,
):
    assert (
        openrouter_module._force_single_required_deepseek_tool(tools, "required")
        == "required"
    )


@pytest.mark.asyncio
async def test_openrouter_official_provider_pinning_disabled_by_default(
    mock_chat_completion, mocker, monkeypatch
):
    """OpenRouter provider pinning is opt-in to preserve fallback behavior."""

    monkeypatch.delenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", raising=False)
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    await llm.chat([{"role": "user", "content": "Hello"}])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "extra_body" not in call_kwargs


@pytest.mark.asyncio
async def test_openrouter_deepseek_uses_official_provider(
    mock_chat_completion, mocker, monkeypatch
):
    """OpenRouter DeepSeek slugs should avoid third-party host fallbacks."""

    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "true")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    await llm.chat([{"role": "user", "content": "Hello"}])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"]["provider"] == {
        "only": ["deepseek"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_openrouter_official_provider_mapping_covers_auto_router_authors(
    monkeypatch,
):
    """Auto-selected official slugs should pin to official OpenRouter providers."""

    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "true")
    cases = {
        "anthropic/claude-sonnet-4.6": ["anthropic"],
        "deepseek/deepseek-v4-flash": ["deepseek"],
        "google/gemini-3-flash-preview": ["google-ai-studio", "google-vertex"],
        "minimax/minimax-m3": ["minimax"],
        "openai/gpt-5.5": ["openai"],
        "z-ai/glm-5.2": ["z-ai"],
    }

    for model_name, expected_providers in cases.items():
        llm = OpenRouterLLM(
            model_name=model_name,
            api_key="test-key",
        )

        extra_body = llm._prepare_extra_body({})

        assert extra_body["provider"]["only"] == expected_providers
        assert extra_body["provider"]["allow_fallbacks"] is False
        assert extra_body["provider"]["require_parameters"] is True


@pytest.mark.asyncio
async def test_openrouter_provider_override_is_preserved(
    mock_chat_completion, mocker, monkeypatch
):
    """Explicit provider routing should win over automatic official pinning."""

    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "true")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    await llm.chat(
        [{"role": "user", "content": "Hello"}],
        extra_body={"provider": {"only": ["deepinfra"]}, "trace_id": "manual"},
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {
        "provider": {"only": ["deepinfra"]},
        "trace_id": "manual",
    }


@pytest.mark.asyncio
async def test_openrouter_deepseek_names_the_only_required_tool(mocker, monkeypatch):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    tool_call = SimpleNamespace(
        id="call_route",
        type="function",
        function=SimpleNamespace(
            name="select_execution_pattern",
            arguments="{}",
        ),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[tool_call],
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-route"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    await llm.chat(
        [{"role": "user", "content": "Route this request"}],
        tools=_single_tool_schema(),
        tool_choice="required",
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "select_execution_pattern"},
    }


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_names_the_only_required_tool(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")

    async def empty_stream():
        if False:
            yield None

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = empty_stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Route this request"}],
            tools=_single_tool_schema(),
            tool_choice="required",
        )
    ]

    assert chunks == []
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "select_execution_pattern"},
    }


@pytest.mark.asyncio
async def test_openrouter_deepseek_retries_function_call_without_assistant_prefix(
    mock_chat_completion, mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _deepseek_function_prefix_error(),
        mock_chat_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    messages = _tool_call_history()
    strip_spy = mocker.spy(openrouter_module, "_strip_assistant_tool_call_prefixes")

    result = await llm.chat(messages)

    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 2
    assert strip_spy.call_count == 1
    first_messages = mock_client.chat.completions.create.call_args_list[0].kwargs[
        "messages"
    ]
    retry_messages = mock_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert first_messages[1]["content"] == "I will generate the music first."
    assert retry_messages[1]["content"] == ""
    assert retry_messages[1]["tool_calls"] == messages[1]["tool_calls"]
    assert messages[1]["content"] == "I will generate the music first."


@pytest.mark.asyncio
async def test_openrouter_deepseek_retries_without_trailing_assistant_progress(
    mock_chat_completion, mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _deepseek_function_prefix_error(),
        mock_chat_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    messages = _tool_call_history_with_trailing_progress()

    result = await llm.chat(messages)

    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 2
    retry_messages = mock_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert retry_messages[-1]["role"] == "tool"
    assert all(
        message.get("content") != "Still working on the generated audio."
        for message in retry_messages
    )
    assert messages[-1]["content"] == "Still working on the generated audio."


@pytest.mark.asyncio
async def test_openrouter_deepseek_propagates_sanitized_retry_failure(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _deepseek_function_prefix_error(),
        _deepseek_function_prefix_error(),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    with pytest.raises(RuntimeError, match="Function call should not be used"):
        await llm.chat(_tool_call_history())

    assert mock_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_openrouter_deepseek_does_not_retry_whitespace_only_prefix(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = _deepseek_function_prefix_error()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    messages = _tool_call_history()
    messages[1]["content"] = "   "

    with pytest.raises(RuntimeError, match="Function call should not be used"):
        await llm.chat(messages)

    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_retries_prefix_error_before_first_chunk(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")

    async def empty_stream():
        if False:
            yield None

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _deepseek_function_prefix_error(),
        empty_stream(),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    chunks = [chunk async for chunk in llm.stream_chat(_tool_call_history())]

    assert chunks == []
    assert mock_client.chat.completions.create.await_count == 2
    retry_messages = mock_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert retry_messages[1]["content"] == ""


@pytest.mark.asyncio
async def test_openrouter_does_not_retry_unrelated_bad_request(mocker, monkeypatch):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = _unrelated_bad_request()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    with pytest.raises(RuntimeError, match="Unrelated invalid request"):
        await llm.chat(_tool_call_history())

    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_openrouter_non_deepseek_does_not_retry_function_prefix_error(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = _deepseek_function_prefix_error()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="openai/gpt-5.5",
        api_key="test-key",
    )

    with pytest.raises(RuntimeError, match="Function call should not be used"):
        await llm.chat(_tool_call_history())

    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_openrouter_stream_deepseek_uses_official_provider(mocker, monkeypatch):
    """Streaming calls should carry the same OpenRouter provider routing."""

    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "true")

    async def empty_stream():
        if False:
            yield None

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = empty_stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    _ = [
        chunk async for chunk in llm.stream_chat([{"role": "user", "content": "Hello"}])
    ]

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"]["provider"] == {
        "only": ["deepseek"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thinking",
    [
        {"type": "disabled", "enable": False},
        {"type": "omit"},
    ],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek/deepseek-v4-flash",
        "openrouter/deepseek/deepseek-v4-flash",
    ],
)
async def test_openrouter_deepseek_stream_uses_disabled_thinking_payload(
    mocker, monkeypatch, thinking, model_name
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")

    async def empty_stream():
        if False:
            yield None

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = empty_stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name=model_name,
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    _ = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Hello"}],
            tool_choice="required",
            thinking=thinking,
        )
    ]

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }
    assert "enable_thinking" not in call_kwargs["extra_body"]
    assert call_kwargs["tool_choice"] == "required"


def test_openrouter_reasoning_hook_enables_reasoning_payload(monkeypatch):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={"trace_id": "abc", "enable_thinking": False},
        thinking={"type": "enabled"},
        tools=None,
        response_format=None,
        output_config=None,
        is_streaming=True,
    )

    assert extra_body == {
        "trace_id": "abc",
        "reasoning": {"enabled": True},
        "thinking": {"type": "enabled"},
    }


@pytest.mark.asyncio
async def test_structured_output_retry_disables_openrouter_reasoning(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")

    first_message = SimpleNamespace(
        content="not json",
        tool_calls=None,
        reasoning_content="reasoning here",
    )
    second_message = SimpleNamespace(
        content='{"status": "ok"}',
        tool_calls=None,
        reasoning_content=None,
    )
    first_response = SimpleNamespace(
        choices=[SimpleNamespace(message=first_message)],
        usage=None,
        model_dump=lambda: {"id": "openrouter-first"},
    )
    second_response = SimpleNamespace(
        choices=[SimpleNamespace(message=second_message)],
        usage=None,
        model_dump=lambda: {"id": "openrouter-second"},
    )

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [first_response, second_response]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    result = await llm.chat(
        [{"role": "user", "content": "Return JSON"}],
        response_format={"type": "json_object"},
        thinking={"type": "enabled"},
    )

    assert result["type"] == "text"
    assert result["content"] == '{"status": "ok"}'
    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call["extra_body"]["reasoning"] == {"enabled": False}
    assert second_call["extra_body"]["thinking"] == {"type": "disabled"}
