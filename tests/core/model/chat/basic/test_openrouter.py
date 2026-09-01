"""Test cases for OpenRouter LLM provider behavior."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from xagent.core.model.chat.basic import openrouter as openrouter_module
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.deepseek_tool_protocol import (
    DEEPSEEK_PROVIDER_STATE_NAMESPACE,
    DEEPSEEK_REASONING_CONTENT_STATE_KEY,
)
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.model.chat.basic.openrouter import OpenRouterLLM
from xagent.core.model.chat.error import retry_on
from xagent.core.model.chat.exceptions import (
    LLMRetryableError,
    LLMToolProtocolError,
)
from xagent.core.model.chat.tool_protocol import (
    TOOL_PROTOCOL_ERROR_KEY,
    get_tool_protocol_error,
)
from xagent.core.model.chat.types import (
    PROVIDER_STATE_METADATA_KEY,
    ChunkType,
    StreamChunk,
)
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


@pytest.mark.asyncio
async def test_openrouter_does_not_replay_unavailable_tool_call(mocker):
    unavailable_tool_call = SimpleNamespace(
        id="call_unavailable",
        type="function",
        function=SimpleNamespace(
            name="calculator",
            arguments='{"expression":"2+2"}',
        ),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[unavailable_tool_call],
                    reasoning_content=None,
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-unavailable-tool"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
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
        max_retries=10,
        retry_on=retry_on,
    )

    result = await llm.chat(
        [{"role": "user", "content": "Finish this task"}],
        tools=_single_tool_schema(),
        tool_choice="required",
    )

    protocol_error = get_tool_protocol_error(result)
    assert protocol_error is not None
    assert protocol_error["code"] == "unavailable_tool_call"
    assert mock_client.chat.completions.create.await_count == 1


def test_retry_filter_defers_unavailable_tool_call_to_agent_pattern() -> None:
    error = LLMToolProtocolError(
        provider="deepseek",
        code="unavailable_tool_call",
        message="DeepSeek returned an unavailable tool call.",
    )

    assert retry_on(error) is False


def test_retry_filter_defers_malformed_tool_arguments_to_agent_pattern() -> None:
    error = LLMToolProtocolError(
        provider="deepseek",
        code="malformed_tool_arguments",
        message="DeepSeek returned malformed arguments for 'final_answer'.",
    )

    assert retry_on(error) is False


def test_deepseek_protocol_error_preserves_argument_diagnostics() -> None:
    details = {
        "original_arguments_preview": '{"answer":',
        "original_arguments_length": 10,
        "repair_status": "skipped_incomplete",
    }

    error = openrouter_module._deepseek_tool_protocol_retry_error(
        {
            TOOL_PROTOCOL_ERROR_KEY: {
                "provider": "deepseek",
                "code": "malformed_tool_arguments",
                "message": "Malformed arguments.",
                "details": details,
            }
        }
    )

    assert isinstance(error, LLMToolProtocolError)
    assert error.details == details


@pytest.mark.asyncio
async def test_openrouter_stream_defers_unavailable_tool_call_without_replay(mocker):
    attempts = 0

    async def unavailable_stream():
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_unavailable",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="tool_calls")

    def fake_stream(**kwargs):
        nonlocal attempts
        del kwargs
        attempts += 1
        return unavailable_stream()

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
        max_retries=10,
        retry_on=retry_on,
    )

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Finish this task"}],
            tools=_single_tool_schema(),
            tool_choice="required",
        )
    ]

    assert attempts == 1
    protocol_errors = [chunk for chunk in chunks if chunk.is_protocol_error()]
    assert len(protocol_errors) == 1
    assert protocol_errors[0].protocol_error["code"] == "unavailable_tool_call"


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
    assert "provider" not in call_kwargs["extra_body"]
    assert call_kwargs["extra_body"]["thinking"] == {"type": "disabled"}


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

        assert extra_body["provider"] == {
            "only": expected_providers,
            "allow_fallbacks": False,
        }


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
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
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


@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek/deepseek-v4-flash",
        "openrouter/deepseek/deepseek-v4-flash",
    ],
)
@pytest.mark.parametrize("is_streaming", [True, False])
@pytest.mark.parametrize("response_format", [None, {"type": "json_object"}])
def test_openrouter_deepseek_defaults_to_disabled_thinking(
    monkeypatch, model_name, is_streaming, response_format
):
    """The undeclared-ability half of a declared/undeclared contrast pair.

    This model record never declares ``thinking_mode``, so every request
    shape here keeps the disabled default. See
    ``test_openrouter_deepseek_declared_thinking_ability_leaves_default_open``
    for the same shapes with the ability declared.
    """
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name=model_name,
        api_key="test-key",
        abilities=["chat", "tool_calling"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={"trace_id": "abc"},
        thinking=None,
        tools=None,
        response_format=response_format,
        output_config=None,
        is_streaming=is_streaming,
    )

    assert extra_body == {
        "trace_id": "abc",
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }


@pytest.mark.parametrize(
    ("is_streaming", "response_format", "expected_extra"),
    [
        (False, None, {}),
        (
            False,
            {"type": "json_object"},
            {"reasoning": {"enabled": False}, "thinking": {"type": "disabled"}},
        ),
        (True, None, {}),
        (
            True,
            {"type": "json_object"},
            {"reasoning": {"enabled": False}, "thinking": {"type": "disabled"}},
        ),
    ],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek/deepseek-v4-flash",
        "openrouter/deepseek/deepseek-v4-flash",
    ],
)
def test_openrouter_deepseek_declared_thinking_ability_leaves_default_open(
    monkeypatch, model_name, is_streaming, response_format, expected_extra
):
    """The declared-ability half of a declared/undeclared contrast pair.

    A model record that declares ``thinking_mode`` gets nothing sent for an
    unspecified request, except when the request asks for structured
    output -- streaming or not -- which stays disabled: that branch is only
    reached when the caller specified no thinking configuration at all, and
    the declared ability says the operator wants reasoning, not that they
    want it mixed into a JSON body. See
    ``test_openrouter_deepseek_defaults_to_disabled_thinking`` for the same
    shapes without the ability declared.
    """
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name=model_name,
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={"trace_id": "abc"},
        thinking=None,
        tools=None,
        response_format=response_format,
        output_config=None,
        is_streaming=is_streaming,
    )

    assert extra_body == {"trace_id": "abc", **expected_extra}


def test_openrouter_deepseek_structured_streaming_disables_thinking(monkeypatch):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={},
        thinking=None,
        tools=None,
        response_format={"type": "json_object"},
        output_config=None,
        is_streaming=True,
    )

    assert extra_body == {
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }


@pytest.mark.parametrize("response_format", [None, {"type": "json_object"}])
def test_openrouter_non_deepseek_defaults_leave_thinking_unset(
    monkeypatch, response_format
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name="openai/gpt-5",
        api_key="test-key",
        abilities=["chat", "tool_calling"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={"trace_id": "abc"},
        thinking=None,
        tools=None,
        response_format=response_format,
        output_config=None,
        is_streaming=True,
    )

    assert extra_body == {"trace_id": "abc"}


@pytest.mark.parametrize(
    ("is_streaming", "expected_extra_body"),
    [
        (True, {"reasoning": {"enabled": False}, "thinking": {"type": "disabled"}}),
        (False, {}),
    ],
)
def test_openrouter_non_deepseek_thinking_structured_output_fork(
    monkeypatch, is_streaming, expected_extra_body
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name="openai/gpt-5",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={},
        thinking=None,
        tools=None,
        response_format={"type": "json_object"},
        output_config=None,
        is_streaming=is_streaming,
    )

    assert extra_body == expected_extra_body


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


@pytest.mark.asyncio
async def test_openrouter_deepseek_default_thinking_triggers_structured_degrade_resend(
    mocker, monkeypatch
):
    """The structured-output degrade resend above is triggered by the
    response's own reasoning_content, not by what this call requested -- so
    it also fires when the caller passes no ``thinking`` at all. The
    mandatory reasoning-disable for structured output does not depend on
    whether this call's model record declares the thinking ability either,
    so the first request already carries the disable payload; the degrade
    resend exists because this endpoint reasoned anyway, ignoring that
    payload, and it repeats the same disable ask on the resend.
    """
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
        model_dump=lambda: {"id": "openrouter-default-first"},
    )
    second_response = SimpleNamespace(
        choices=[SimpleNamespace(message=second_message)],
        usage=None,
        model_dump=lambda: {"id": "openrouter-default-second"},
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
    )

    assert result["content"] == '{"status": "ok"}'
    assert mock_client.chat.completions.create.await_count == 2
    disabled = {
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }
    # The structured-output rule already asked for no reasoning on the way
    # out; this endpoint reasoned anyway. The degrade resend is the second
    # line of defence for exactly that, and it repeats the same ask.
    assert (
        mock_client.chat.completions.create.call_args_list[0].kwargs["extra_body"]
        == disabled
    )
    assert (
        mock_client.chat.completions.create.call_args_list[1].kwargs["extra_body"]
        == disabled
    )


# ==========================================================================
# Provider-compatibility retries owned by the client (direct-slug entrypoints)
# ==========================================================================

_MANDATORY_REASONING_ERROR = (
    "Reasoning is mandatory for this endpoint and cannot be disabled."
)
_THINKING_TOOL_CHOICE_ERROR = "Thinking mode does not support this tool_choice"
_OPENROUTER_TOOL_CHOICE_ERROR = (
    "No endpoints found that support the provided 'tool_choice' value."
)


def _two_tool_schema() -> list[dict]:
    return [
        _single_tool_schema("answer")[0],
        _single_tool_schema("skip")[0],
    ]


@pytest.mark.asyncio
async def test_openrouter_direct_relaxes_tool_choice_on_endpoint_404(
    mock_chat_completion, mocker
):
    """A direct (non-auto) OpenRouter slug retries a rejected tool_choice itself."""
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR),
        mock_chat_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    result = await llm.chat(
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
    )

    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 2
    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call["tool_choice"] == "auto"


def _bad_request_error(message: str) -> openai.BadRequestError:
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {{'message': '{message}'}}}}",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        ),
        body={"error": {"message": message, "code": 400}},
    )


def _bad_request_error_with_metadata(
    message: str, metadata: dict
) -> openai.BadRequestError:
    """Build a real SDK ``BadRequestError`` carrying an ``error.metadata`` body.

    Used to reproduce OpenRouter's provider-level 400 shape (a
    ``provider_name`` and sometimes a ``raw`` echo of the provider's own
    response nested under ``metadata``), which ``_bad_request_error`` above
    does not model.
    """
    payload = {"message": message, "code": 400, "metadata": metadata}
    return openai.BadRequestError(
        f"Error code: 400 - {payload!r}",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        ),
        body={"error": payload},
    )


@pytest.mark.asyncio
async def test_openrouter_direct_relaxes_tool_choice_after_wrapped_resend_failure(
    mock_chat_completion, mocker
):
    """The compat retry recovers when the response_format resend also 400s.

    ``OpenAILLM.chat`` converts every ``openai.BadRequestError`` into a
    ``RuntimeError``, including a failure of its response_format pop-and-retry
    resend, so the compat loop sees the wrapped form here. The loop's catch
    tuple still includes the bare SDK exception as defense in depth for any
    future base-client path that leaks one unwrapped.
    """
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _bad_request_error(
            "the model does not support response_format for this request"
        ),
        _bad_request_error(_OPENROUTER_TOOL_CHOICE_ERROR),
        mock_chat_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    result = await llm.chat(
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
        response_format={"type": "json_object"},
    )

    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 3
    final_call = mock_client.chat.completions.create.call_args_list[2].kwargs
    assert final_call["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_openrouter_vision_relaxes_tool_choice_after_wrapped_resend_failure(
    mock_chat_completion, mocker
):
    """The vision path recovers when the response_format resend also 400s.

    ``vision_chat`` is the entrypoint with a live response_format producer in
    this repository, so this recovery is reachable in production. A failed
    resend arrives here wrapped as ``RuntimeError`` by the base client; the
    compat loop's catch tuple keeps the bare SDK exception as defense in
    depth. The value asserted below reaches the caller from the THIRD
    upstream call, which succeeds on its first attempt after the compat loop
    relaxed ``tool_choice`` -- it does not come through a successful resend.
    So this pins that ``vision_chat``'s processed result survives the
    OpenRouter compat loop; the successful-resend return itself is pinned
    directly by ``test_vision_chat_response_format_retry_success_returns_result``
    in ``tests/core/model/chat/basic/test_openai.py``.
    """
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _bad_request_error(
            "the model does not support response_format for this request"
        ),
        _bad_request_error(_OPENROUTER_TOOL_CHOICE_ERROR),
        mock_chat_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="z-ai/glm-5.2",
        api_key="test-key",
        abilities=["chat", "tool_calling", "vision"],
    )

    result = await llm.vision_chat(
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
        response_format={"type": "json_object"},
    )

    assert result["type"] == "text"
    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 3
    final_call = mock_client.chat.completions.create.call_args_list[2].kwargs
    assert final_call["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_openrouter_direct_relaxes_tool_choice_on_provider_level_400(
    mock_chat_completion, mocker
):
    """Z.AI's provider-level 400 is recognized through the structured body.

    Regression test for xorbitsai/xagent#1960: Z.AI rejects a strict ``tool_choice``
    with ``{'error': {'message': 'Tool choice must be auto', 'metadata':
    {'provider_name': 'Z.AI'}}}``. That message spells ``tool choice`` with
    a space (not the ``tool_choice`` token the old flattened-string match
    looked for) and never contains ``no endpoints found``, so the
    pre-existing string match never fired for this shape -- the whole reason
    #1960 was filed. The structured path reads ``error.message`` directly and
    normalizes the spacing, so it must fire here.
    """
    zai_error = _bad_request_error_with_metadata(
        "Tool choice must be auto", metadata={"provider_name": "Z.AI"}
    )
    assert openrouter_module._should_retry_with_relaxed_tool_choice(
        zai_error, tools=_two_tool_schema(), tool_choice="required"
    )

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        zai_error,
        mock_chat_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    result = await llm.chat(
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
    )

    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 2
    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call["tool_choice"] == "auto"


@pytest.mark.parametrize(
    "provider_message",
    [
        "This model's maximum context length is 4096 tokens",
        "The tool choice you made is not available on this plan",
    ],
    ids=["unrelated-subject", "names-tool-choice-but-is-not-a-must-be-auto-rejection"],
)
def test_relaxed_tool_choice_ignores_unrelated_400_with_provider_name(provider_message):
    """A provider-level 400 that is not a "must be auto" rejection stays False.

    ``metadata.provider_name`` only says that some provider endpoint
    answered; it never says that endpoint rejected ``tool_choice``. Both
    halves of that distinction need pinning: the first message is about an
    entirely different subject, while the second one does name the tool
    choice and still is not the "must be auto" rejection this relax retry
    exists to answer. Resending with ``tool_choice="auto"`` in either case
    would repeat a request the provider gave no reason to expect would fare
    any better.
    """
    unrelated_error = _bad_request_error_with_metadata(
        provider_message,
        metadata={"provider_name": "Z.AI"},
    )

    assert not openrouter_module._should_retry_with_relaxed_tool_choice(
        unrelated_error, tools=_two_tool_schema(), tool_choice="required"
    )


def test_relaxed_tool_choice_does_not_match_on_provider_raw_echo():
    """A route-level phrase echoed only in ``metadata.raw`` must not trigger relax.

    ``_openai_error_details`` folds ``metadata.raw`` (up to 4000 characters
    of provider-controlled text) into the flattened ``str(exc)`` used by the
    string-matching fallback. Here the real ``error.message`` is unrelated
    ("Upstream provider error"), but a provider echoed a route-level 404's
    wording inside ``metadata.raw``. The structured path reads only
    ``error.message`` and must ignore ``raw`` entirely, so this must stay
    False even though the flattened string contains both trigger tokens.
    """
    raw_echo_error = _bad_request_error_with_metadata(
        "Upstream provider error",
        metadata={
            "provider_name": "SomeProvider",
            "raw": _OPENROUTER_TOOL_CHOICE_ERROR,
        },
    )

    # Sanity check on the premise: the flattened string this test guards
    # against really does contain both tokens the old code searched for.
    assert "no endpoints found" in str(raw_echo_error).lower()
    assert "tool_choice" in str(raw_echo_error).lower()

    assert not openrouter_module._should_retry_with_relaxed_tool_choice(
        raw_echo_error, tools=_two_tool_schema(), tool_choice="required"
    )


def _bad_request_error_with_non_dict_body(message: str) -> openai.BadRequestError:
    """Build an SDK ``BadRequestError`` whose error body is not a JSON object.

    A provider can answer a 400 with a bare JSON array or string instead of
    the ``{"error": {...}}`` object OpenRouter itself sends. The structured
    read has to decline such a body instead of indexing into it.
    """
    return openai.BadRequestError(
        f"Error code: 400 - {message}",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        ),
        body=[message],
    )


@pytest.mark.parametrize(
    "build_error",
    [
        lambda: RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR),
        lambda: _bad_request_error_with_non_dict_body(_OPENROUTER_TOOL_CHOICE_ERROR),
    ],
    ids=["no-sdk-error-in-cause-chain", "sdk-error-with-non-dict-body"],
)
def test_relaxed_tool_choice_falls_back_to_string_match_without_structured_body(
    build_error,
):
    """An error with no readable structured body still relies on string matching.

    Two shapes reach the predicate without a usable structured body: a plain
    ``RuntimeError`` with no ``openai.BadRequestError`` anywhere in its cause
    chain (a test double, or any future non-SDK transport), and a real SDK
    error whose body is not a JSON object. The structured extractor returns
    ``(None, None)`` for both without raising, and the predicate must then
    fall back to exactly the old flattened-string check rather than treating
    "no structured body" as "do not retry".
    """
    assert openrouter_module._should_retry_with_relaxed_tool_choice(
        build_error(), tools=_two_tool_schema(), tool_choice="required"
    )


@pytest.mark.asyncio
async def test_openrouter_compat_loop_catches_bare_sdk_bad_request(mocker):
    """A bare ``openai.BadRequestError`` reaching the compat loop is handled.

    The base client wraps provider 4xx failures into ``RuntimeError`` on
    every known path, so this pin drives the loop directly: the inner call
    raises the bare SDK error once, and the loop must treat it as a compat
    adjustment opportunity rather than let it escape. This is the only test
    that fails when ``openai.BadRequestError`` is removed from
    ``_COMPAT_RETRYABLE_ERRORS``. The streaming compat loop in
    ``_run_stream_chat_with_compat_retry`` consumes the same tuple but has
    no equivalent bare-error stub coverage yet.
    """
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    prefix_retry_mock = mocker.patch.object(
        llm,
        "_chat_with_prefix_retry",
        side_effect=[
            _bad_request_error(_OPENROUTER_TOOL_CHOICE_ERROR),
            {"type": "text", "content": "Hello World", "tool_calls": None},
        ],
    )

    result = await llm.chat(
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
    )

    assert result["content"] == "Hello World"
    assert prefix_retry_mock.call_count == 2
    assert prefix_retry_mock.call_args_list[1].kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_openrouter_direct_does_not_repeat_mandatory_reasoning_retry(mocker):
    """Each compat action fires at most once per call, even across a 3-error run."""
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_MANDATORY_REASONING_ERROR),
        RuntimeError(_THINKING_TOOL_CHOICE_ERROR),
        RuntimeError(_MANDATORY_REASONING_ERROR),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    with pytest.raises(RuntimeError, match="Reasoning is mandatory"):
        await llm.chat(
            [{"role": "user", "content": "score?"}],
            tools=_two_tool_schema(),
            tool_choice="required",
            thinking={"type": "disabled", "enable": False},
        )

    assert mock_client.chat.completions.create.await_count == 3
    thinking_values = [
        call.kwargs["extra_body"].get("thinking")
        for call in mock_client.chat.completions.create.call_args_list
    ]
    assert thinking_values == [
        {"type": "disabled"},
        {"type": "enabled"},
        {"type": "disabled"},
    ]


@pytest.mark.asyncio
async def test_openrouter_direct_chains_thinking_and_tool_choice_retries(mocker):
    """A thinking-conflict 400 followed by a tool_choice 404 chains two adjustments."""
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_THINKING_TOOL_CHOICE_ERROR),
        RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok", tool_calls=None, reasoning_content=None
                    )
                )
            ],
            usage=None,
            model_dump=lambda: {"id": "openrouter-chain"},
        ),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    result = await llm.chat(
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
        thinking={"type": "enabled", "enable": True},
    )

    assert result["content"] == "ok"
    tool_choices = [
        call.kwargs["tool_choice"]
        for call in mock_client.chat.completions.create.call_args_list
    ]
    assert tool_choices == ["required", "required", "auto"]


@pytest.mark.asyncio
async def test_openrouter_stream_relaxes_tool_choice_before_first_chunk(mocker):
    """Streaming retries the same compat rules while nothing has been yielded yet."""
    calls: list[dict] = []

    async def rejects_tool_choice():
        if False:
            yield None
        raise RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR)

    async def succeeds():
        yield StreamChunk(type=ChunkType.TOKEN, content="ok", delta="ok")

    def fake_stream(*_args, **kwargs):
        calls.append(kwargs)
        return rejects_tool_choice() if len(calls) == 1 else succeeds()

    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    mocker.patch.object(llm, "_stream_chat_inner", side_effect=fake_stream)

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            tools=_two_tool_schema(),
            tool_choice="required",
        )
    ]

    assert [chunk.delta for chunk in chunks] == ["ok"]
    assert len(calls) == 2
    assert [call["tool_choice"] for call in calls] == ["required", "auto"]


@pytest.mark.asyncio
async def test_openrouter_stream_error_after_first_chunk_not_retried(mocker):
    """Once a chunk has actually reached the caller, a later error is never retried."""
    attempts = 0

    async def yields_then_fails():
        nonlocal attempts
        attempts += 1
        yield StreamChunk(type=ChunkType.TOKEN, content="partial", delta="partial")
        raise RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR)

    def fake_stream(*_args, **kwargs):
        del kwargs
        return yields_then_fails()

    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    mocker.patch.object(llm, "_stream_chat_inner", side_effect=fake_stream)

    received = []
    with pytest.raises(RuntimeError, match="No endpoints found"):
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            tools=_two_tool_schema(),
            tool_choice="required",
        ):
            received.append(chunk)

    assert [chunk.delta for chunk in received] == ["partial"]
    assert attempts == 1


@pytest.mark.asyncio
async def test_openrouter_compat_retry_does_not_catch_llm_retryable_error(mocker):
    """A retryable protocol error is re-raised even when its text matches a rule.

    The error message deliberately matches the relaxed-tool_choice predicate
    and the call carries tools with tool_choice="required": without the
    LLMRetryableError guard the compat retry would swallow the error and
    replay the request, so the single-call assertion pins the guard itself.
    """
    protocol_error = LLMToolProtocolError(
        provider="deepseek",
        code="malformed_tool_arguments",
        message=("No endpoints found that support the provided 'tool_choice' value."),
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    prefix_retry_mock = mocker.patch.object(
        llm, "_chat_with_prefix_retry", side_effect=protocol_error
    )

    with pytest.raises(LLMToolProtocolError, match="No endpoints found"):
        await llm.chat(
            [{"role": "user", "content": "score?"}],
            tools=_two_tool_schema(),
            tool_choice="required",
        )

    assert prefix_retry_mock.call_count == 1


@pytest.mark.asyncio
async def test_openrouter_compat_retry_skips_retryable_error_hidden_in_cause(mocker):
    """Ordering contract: retry_on() wins over compat-rule text matching.

    A plain RuntimeError whose ``__cause__`` is an ``openai.RateLimitError``
    is not an ``LLMRetryableError`` instance, so the existing
    ``except LLMRetryableError: raise`` guard does not catch it — only
    ``retry_on()``'s ``__cause__`` inspection recognizes it as retryable. The
    error text is deliberately built to also match the relaxed-tool_choice
    compat rule ("no endpoints found" / "tool_choice"), so without checking
    retry_on() first the compat loop would treat this retryable error as an
    OpenRouter compatibility quirk and replay the request instead of leaving
    it for the shared LLM retry wrapper.
    """
    cause = openai.RateLimitError("rate limited", response=mocker.Mock(), body=None)
    try:
        raise RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR) from cause
    except RuntimeError as exc:
        wrapped_rate_limit_error = exc

    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    prefix_retry_mock = mocker.patch.object(
        llm, "_chat_with_prefix_retry", side_effect=wrapped_rate_limit_error
    )

    with pytest.raises(RuntimeError, match="No endpoints found"):
        await llm.chat(
            [{"role": "user", "content": "score?"}],
            tools=_two_tool_schema(),
            tool_choice="required",
        )

    assert prefix_retry_mock.call_count == 1


@pytest.mark.asyncio
async def test_openrouter_direct_thinking_retry_changes_rendered_extra_body(mocker):
    """A thinking-rule retry (rules 1/2) must change what is actually sent."""
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_MANDATORY_REASONING_ERROR),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok", tool_calls=None, reasoning_content=None
                    )
                )
            ],
            usage=None,
            model_dump=lambda: {"id": "openrouter-thinking"},
        ),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    await llm.chat(
        [{"role": "user", "content": "score?"}],
        thinking={"type": "disabled", "enable": False},
    )

    extra_bodies = [
        call.kwargs["extra_body"]
        for call in mock_client.chat.completions.create.call_args_list
    ]
    assert extra_bodies[0] != extra_bodies[1]
    assert extra_bodies[0]["thinking"] == {"type": "disabled"}
    assert extra_bodies[1]["thinking"] == {"type": "enabled"}


_DISABLE_THINKING_MATCHING_EXTRA_BODY = {
    "reasoning": {"enabled": False},
    "thinking": {"type": "disabled"},
}


@pytest.mark.asyncio
async def test_openrouter_direct_thinking_rule_skipped_when_render_unchanged(
    mocker, monkeypatch
):
    """A disable-thinking match that renders no real change is a no-op.

    The caller already sends ``extra_body`` with thinking disabled, so
    replaying with ``_DISABLE_DOWNSTREAM_THINKING`` would produce a
    byte-identical request. The no-op check inside ``_next_compat_adjustment``
    only catches this when ``render()`` starts from the caller's actual
    extra_body instead of a synthetic empty one, so with the fix the rule is
    skipped and the error surfaces after exactly one call.
    """
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = RuntimeError(
        _THINKING_TOOL_CHOICE_ERROR
    )
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    with pytest.raises(RuntimeError, match="Thinking mode does not support"):
        await llm.chat(
            [{"role": "user", "content": "score?"}],
            tool_choice="required",
            thinking=None,
            extra_body=dict(_DISABLE_THINKING_MATCHING_EXTRA_BODY),
        )

    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_openrouter_stream_thinking_rule_skipped_when_render_unchanged(
    mocker, monkeypatch
):
    """Streaming counterpart of the no-op disable-thinking skip above."""
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    calls = 0

    async def rejects():
        if False:
            yield None
        raise RuntimeError(_THINKING_TOOL_CHOICE_ERROR)

    def fake_inner(*_args, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        return rejects()

    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    mocker.patch.object(llm, "_stream_chat_inner", side_effect=fake_inner)

    with pytest.raises(RuntimeError, match="Thinking mode does not support"):
        async for _chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            tool_choice="required",
            thinking=None,
            extra_body=dict(_DISABLE_THINKING_MATCHING_EXTRA_BODY),
        ):
            pass

    assert calls == 1


@pytest.mark.asyncio
async def test_openrouter_vision_chat_retries_mandatory_reasoning(mocker):
    """vision_chat shares the same compat retry as chat, with no prefix retry.

    The message carries a real multimodal content list (text + image_url) so
    this exercises the actual vision dispatch, not just a plain-text payload
    that happens to go through vision_chat. The spies pin both directions of
    the fork: OpenAILLM.vision_chat must be the method that actually issues
    the request, and OpenRouterLLM.chat / _chat_with_prefix_retry (the
    DeepSeek prefix-retry path) must never run for a vision call. Rewiring
    vision_chat to super().chat(...) turns this red via the vision_chat spy
    dropping to zero (super() bypasses the OpenRouterLLM.chat spy), while
    rewiring it to self.chat(...) is caught by the chat and prefix-retry
    spies — together the spies discriminate both dispatch mistakes.
    """
    success_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello World", tool_calls=None, reasoning_content=None
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-vision"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_MANDATORY_REASONING_ERROR),
        success_response,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="z-ai/glm-5.2",
        api_key="test-key",
        abilities=["chat", "tool_calling", "vision"],
    )
    vision_chat_spy = mocker.spy(OpenAILLM, "vision_chat")
    chat_spy = mocker.spy(OpenRouterLLM, "chat")
    prefix_retry_spy = mocker.spy(llm, "_chat_with_prefix_retry")

    result = await llm.vision_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                ],
            }
        ],
        thinking={"type": "disabled", "enable": False},
    )

    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 2
    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call["extra_body"]["thinking"] == {"type": "enabled"}
    assert vision_chat_spy.call_count == 2
    assert chat_spy.call_count == 0
    assert prefix_retry_spy.call_count == 0
    for call in mock_client.chat.completions.create.call_args_list:
        assert "sanitized_out" not in call.kwargs


@pytest.mark.asyncio
async def test_openrouter_deepseek_no_op_thinking_retry_falls_through_to_next_rule(
    mocker,
):
    """A no-op disable-thinking match is skipped so a real fix (enable) still fires."""
    combined_error = (
        "Reasoning is mandatory for this endpoint and cannot be disabled, "
        "and thinking conflicts with tool_choice here."
    )
    success_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ok", tool_calls=None, reasoning_content=None
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-noop"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(combined_error),
        success_response,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    # deepseek's default (thinking=None) already renders as disabled, so the
    # "disable thinking" rule (2) would be a no-op against this error; the
    # aggregator must fall through to rule 1 (enable thinking) instead of
    # wasting the retry budget replaying an unchanged request.
    result = await llm.chat(
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
    )

    assert result["content"] == "ok"
    assert mock_client.chat.completions.create.await_count == 2
    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call["extra_body"]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_openrouter_deepseek_declared_thinking_makes_disable_retry_a_real_fix(
    mocker, monkeypatch
):
    """Mirror of the no-op test above: once the model record declares
    ``thinking_mode``, DeepSeek's unspecified default no longer renders as
    disabled, so the "disable thinking" rule (2) is a real change against
    this same combined error and gets to fire instead of being skipped as a
    no-op.

    The endpoint modelled here is the one the error text describes: it
    refuses every request that does not ask for reasoning explicitly. So the
    disable attempt this fallthrough now spends is rejected in turn, and
    recovery costs three upstream requests -- the unspecified first attempt,
    the disable attempt, and the enable attempt that finally succeeds.
    """
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    combined_error = (
        "Reasoning is mandatory for this endpoint and cannot be disabled, "
        "and thinking conflicts with tool_choice here."
    )
    success_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ok", tool_calls=None, reasoning_content=None
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-declared-noop"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(combined_error),
        RuntimeError(combined_error),
        success_response,
    ]
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
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
    )

    assert result["content"] == "ok"
    calls = mock_client.chat.completions.create.call_args_list
    assert mock_client.chat.completions.create.await_count == 3
    assert "extra_body" not in calls[0].kwargs
    assert calls[1].kwargs["extra_body"] == {
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }
    assert calls[2].kwargs["extra_body"] == {
        "reasoning": {"enabled": True},
        "thinking": {"type": "enabled"},
    }


# ==========================================================================
# Client-layer equivalents of the RouterLLM-level retry tests that used to
# live in tests/core/model/test_router_provider_config.py (moved here because
# RouterLLM no longer retries on its own; see router.py).
# ==========================================================================


@pytest.mark.asyncio
async def test_openrouter_direct_disables_thinking_for_tool_choice_conflict(mocker):
    """chat: a thinking/tool_choice 400 retries once with thinking disabled.

    Starting from an already-enabled thinking value keeps the disable
    adjustment a real change (not a no-op), isolating this single rule.
    """
    calls: list[dict] = []

    async def fake_inner(messages, **kwargs):
        del messages
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(_THINKING_TOOL_CHOICE_ERROR)
        return "ok"

    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    mocker.patch.object(llm, "_chat_with_prefix_retry", side_effect=fake_inner)

    result = await llm.chat(
        [{"role": "user", "content": "score?"}],
        tool_choice="required",
        thinking={"type": "enabled", "enable": True},
    )

    assert result == "ok"
    assert len(calls) == 2
    assert calls[0]["tool_choice"] == "required"
    assert calls[0]["thinking"] == {"type": "enabled", "enable": True}
    assert calls[1]["tool_choice"] == "required"
    assert calls[1]["thinking"] == openrouter_module._DISABLE_DOWNSTREAM_THINKING


@pytest.mark.asyncio
async def test_openrouter_stream_disables_thinking_for_tool_choice_conflict(mocker):
    """stream_chat: same rule as above, exercised on the streaming entrypoint."""
    calls: list[dict] = []

    async def rejects(**_kwargs):
        if False:
            yield None
        raise RuntimeError(_THINKING_TOOL_CHOICE_ERROR)

    async def succeeds(**_kwargs):
        yield StreamChunk(type=ChunkType.TOKEN, content="ok", delta="ok")

    def fake_inner(messages, **kwargs):
        del messages
        calls.append(kwargs)
        return rejects() if len(calls) == 1 else succeeds()

    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    mocker.patch.object(llm, "_stream_chat_inner", side_effect=fake_inner)

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            tool_choice="required",
            thinking={"type": "enabled", "enable": True},
        )
    ]

    assert [chunk.delta for chunk in chunks] == ["ok"]
    assert len(calls) == 2
    assert calls[0]["tool_choice"] == "required"
    assert calls[0]["thinking"] == {"type": "enabled", "enable": True}
    assert calls[1]["tool_choice"] == "required"
    assert calls[1]["thinking"] == openrouter_module._DISABLE_DOWNSTREAM_THINKING


@pytest.mark.asyncio
async def test_openrouter_stream_disables_thinking_when_unspecified_non_deepseek(
    mocker,
):
    """stream_chat: an unspecified (None) thinking value still gets a real
    disable retry for a non-DeepSeek model.

    DeepSeek's None default already renders as disabled (see the no-op test
    below), so this covers the case where starting from None is a genuine
    adjustment: any non-DeepSeek slug, where an unset thinking value renders
    as an empty extra_body rather than an explicit disabled one.
    """
    calls: list[dict] = []

    async def rejects(**_kwargs):
        if False:
            yield None
        raise RuntimeError(_THINKING_TOOL_CHOICE_ERROR)

    async def succeeds(**_kwargs):
        yield StreamChunk(type=ChunkType.TOKEN, content="ok", delta="ok")

    def fake_inner(messages, **kwargs):
        del messages
        calls.append(kwargs)
        return rejects() if len(calls) == 1 else succeeds()

    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    mocker.patch.object(llm, "_stream_chat_inner", side_effect=fake_inner)

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            tool_choice="required",
        )
    ]

    assert [chunk.delta for chunk in chunks] == ["ok"]
    assert len(calls) == 2
    assert calls[0]["thinking"] is None
    assert calls[1]["tool_choice"] == "required"
    assert calls[1]["thinking"] == openrouter_module._DISABLE_DOWNSTREAM_THINKING


@pytest.mark.asyncio
async def test_openrouter_stream_thinking_retry_changes_rendered_extra_body(mocker):
    """stream_chat: same rule as the chat-path test, exercised on streaming.

    A mandatory-reasoning 400 (rule 3) retries once with thinking enabled,
    and the retried request's ``extra_body`` must actually reflect that
    change, not just the ``thinking`` kwarg passed to the inner call.

    The retry's success is mocked as an empty stream (the same convention
    ``test_openrouter_deepseek_stream_retries_prefix_error_before_first_chunk``
    uses): the raw OpenAI SDK chunk shape this mock would otherwise need to
    produce is unrelated to what this test is pinning, which is the request
    actually sent on retry.
    """

    async def empty_stream():
        if False:
            yield None

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_MANDATORY_REASONING_ERROR),
        empty_stream(),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            thinking={"type": "disabled", "enable": False},
        )
    ]

    assert chunks == []
    assert mock_client.chat.completions.create.await_count == 2
    extra_bodies = [
        call.kwargs["extra_body"]
        for call in mock_client.chat.completions.create.call_args_list
    ]
    assert extra_bodies[0] != extra_bodies[1]
    assert extra_bodies[0]["thinking"] == {"type": "disabled"}
    assert extra_bodies[1]["thinking"] == {"type": "enabled"}
    for call in mock_client.chat.completions.create.call_args_list:
        assert "sanitized_out" not in call.kwargs


@pytest.mark.asyncio
async def test_openrouter_stream_does_not_repeat_mandatory_reasoning_retry(mocker):
    """stream_chat: each compat action fires at most once per call, even
    across a 3-error run, mirroring the chat-path budget guarantee.
    """
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_MANDATORY_REASONING_ERROR),
        RuntimeError(_THINKING_TOOL_CHOICE_ERROR),
        RuntimeError(_MANDATORY_REASONING_ERROR),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    with pytest.raises(RuntimeError, match="Reasoning is mandatory"):
        async for _chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            tools=_two_tool_schema(),
            tool_choice="required",
            thinking={"type": "disabled", "enable": False},
        ):
            pass

    assert mock_client.chat.completions.create.await_count == 3
    thinking_values = [
        call.kwargs["extra_body"].get("thinking")
        for call in mock_client.chat.completions.create.call_args_list
    ]
    assert thinking_values == [
        {"type": "disabled"},
        {"type": "enabled"},
        {"type": "disabled"},
    ]


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_no_op_thinking_default_propagates(mocker):
    """stream_chat: DeepSeek's unspecified (None) thinking already renders
    disabled, so a thinking/tool_choice 400 is a no-op for rule 2; with no
    other rule matching this text, the error surfaces after exactly one call
    instead of wasting a retry replaying an unchanged request.
    """
    calls = 0

    async def rejects():
        if False:
            yield None
        raise RuntimeError(_THINKING_TOOL_CHOICE_ERROR)

    def fake_inner(*_args, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        return rejects()

    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")
    mocker.patch.object(llm, "_stream_chat_inner", side_effect=fake_inner)

    with pytest.raises(RuntimeError, match="Thinking mode does not support"):
        async for _chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            tool_choice="required",
        ):
            pass

    assert calls == 1


@pytest.mark.asyncio
async def test_openrouter_deepseek_declared_thinking_stream_spends_disable_retry(
    mocker, monkeypatch
):
    """Mirror of the no-op test above: once the model record declares
    ``thinking_mode``, DeepSeek's unspecified default no longer renders as
    disabled, so the disable-thinking retry (rule 2) is a real change and
    actually spends its budget before the request fails for good.
    """
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    seen: list = []

    async def rejects():
        if False:
            yield None
        raise RuntimeError(_THINKING_TOOL_CHOICE_ERROR)

    def fake_inner(*_args, **kwargs):
        seen.append(kwargs.get("thinking"))
        return rejects()

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )
    mocker.patch.object(llm, "_stream_chat_inner", side_effect=fake_inner)

    with pytest.raises(RuntimeError, match="Thinking mode does not support"):
        async for _chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            tool_choice="required",
        ):
            pass

    assert seen == [None, openrouter_module._DISABLE_DOWNSTREAM_THINKING]


# ---------------------------------------------------------------------------
# DeepSeek reasoning-content replay on OpenRouter (issue #1537).
#
# These cover the shared capture/replay mechanism now wired into
# OpenRouterLLM via ``deepseek_tool_protocol``'s shared functions. The
# direct-DeepSeek side of that same mechanism keeps its own coverage in
# test_deepseek.py, and test_openai.py holds the negative case pinning that
# a plain OpenAI-compatible client captures nothing. Whether a deepseek slug
# is asked to disable reasoning when the caller says nothing is decided by
# the model record's declared abilities; the request-payload tests around
# ``test_openrouter_deepseek_defaults_to_disabled_thinking`` pin both rows
# of that contrast.
# ---------------------------------------------------------------------------


def _deepseek_provider_state(reasoning_content: object) -> dict:
    return {
        DEEPSEEK_PROVIDER_STATE_NAMESPACE: {
            DEEPSEEK_REASONING_CONTENT_STATE_KEY: reasoning_content
        }
    }


def _openrouter_tool_call_response(
    *,
    reasoning_content: object = "unset",
    raw_message_extra: dict | None = None,
    tool_name: str = "search",
) -> SimpleNamespace:
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name=tool_name, arguments="{}"),
    )
    message_kwargs: dict = {"content": None, "tool_calls": [tool_call]}
    if reasoning_content != "unset":
        message_kwargs["reasoning_content"] = reasoning_content
    message = SimpleNamespace(**message_kwargs)
    raw_message = dict(raw_message_extra or {})
    raw = {
        "id": "openrouter-deepseek",
        "choices": [{"message": raw_message}],
    }
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
        model_dump=lambda: raw,
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_captures_reasoning_provider_state(mocker):
    """A deepseek-slug tool-call response captures reasoning_content."""
    response = _openrouter_tool_call_response(
        reasoning_content="Use the search tool first"
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    result = await llm.chat(
        [{"role": "user", "content": "Search xagent"}],
        tools=_single_tool_schema("search"),
        thinking={"type": "enabled"},
    )

    assert result[PROVIDER_STATE_METADATA_KEY] == _deepseek_provider_state(
        "Use the search tool first"
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_captures_empty_reasoning_content(mocker):
    """An explicit empty-string reasoning_content is still captured.

    An empty string is a value the provider sent, not a missing field, so
    it must round-trip like any other captured content.
    """
    response = _openrouter_tool_call_response(reasoning_content="")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    result = await llm.chat(
        [{"role": "user", "content": "Search xagent"}],
        tools=_single_tool_schema("search"),
        thinking={"type": "enabled"},
    )

    assert result[PROVIDER_STATE_METADATA_KEY] == _deepseek_provider_state("")


@pytest.mark.asyncio
async def test_openrouter_deepseek_captures_reasoning_alias_from_raw(mocker):
    """The ``reasoning`` alias is captured from the raw response body
    when the SDK message object never surfaced ``reasoning_content`` at all.

    Capture checks both spellings on purpose: no live OpenRouter response
    was available to confirm which one a deepseek slug actually sends, so
    the raw body is consulted as a fallback for whichever spelling the
    transport layer did not already normalize.
    """
    response = _openrouter_tool_call_response(
        reasoning_content="unset",
        raw_message_extra={"reasoning": "alt-spelling-thinking"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    result = await llm.chat(
        [{"role": "user", "content": "Search xagent"}],
        tools=_single_tool_schema("search"),
        thinking={"type": "enabled"},
    )

    # The base class's own message-level mirrors never fired (the SDK
    # message object had no ``reasoning_content`` attribute at all) -- only
    # the provider-state capture, which additionally consults raw, sees it.
    assert "reasoning_content" not in result
    assert result[PROVIDER_STATE_METADATA_KEY] == _deepseek_provider_state(
        "alt-spelling-thinking"
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_declared_thinking_captures_and_replays_without_request_thinking(
    mocker,
):
    """The path PR-2 opens: a model record that declares ``thinking_mode``
    gets reasoning captured and replayed across a tool-call chain even when
    neither request in the chain passes ``thinking`` at all.
    """
    first = _openrouter_tool_call_response(
        reasoning_content="Use the search tool first"
    )
    second = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="done", tool_calls=None, reasoning_content=None
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-second-turn"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [first, second]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    round1 = await llm.chat(
        [{"role": "user", "content": "Search xagent"}],
        tools=_single_tool_schema("search"),
    )
    assert (
        "extra_body" not in mock_client.chat.completions.create.call_args_list[0].kwargs
    )
    assert round1[PROVIDER_STATE_METADATA_KEY] == _deepseek_provider_state(
        "Use the search tool first"
    )

    messages = [
        {"role": "user", "content": "Search xagent"},
        {
            "role": "assistant",
            "content": "",
            PROVIDER_STATE_METADATA_KEY: round1[PROVIDER_STATE_METADATA_KEY],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]
    await llm.chat(messages, tools=_single_tool_schema("search"))
    sent = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert sent[1]["reasoning_content"] == "Use the search tool first"
    assert PROVIDER_STATE_METADATA_KEY not in sent[1]


@pytest.mark.asyncio
async def test_openrouter_deepseek_replays_reasoning_content(
    mocker, mock_chat_completion
):
    """Captured provider state is translated back to ``reasoning_content``
    on the next request, and the internal marker never reaches the wire.
    """
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")
    messages = [
        {"role": "user", "content": "Search xagent"},
        {
            "role": "assistant",
            "content": "",
            PROVIDER_STATE_METADATA_KEY: _deepseek_provider_state("prior thought"),
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    await llm.chat(messages)

    call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert call_messages[1]["reasoning_content"] == "prior thought"
    assert PROVIDER_STATE_METADATA_KEY not in call_messages[1]


@pytest.mark.asyncio
async def test_openrouter_deepseek_replays_empty_reasoning_fallback(
    mocker, mock_chat_completion
):
    """An assistant tool-call message with no captured state gets the
    empty-string fallback so the history stays structurally valid.
    """
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")
    messages = [
        {"role": "user", "content": "Search xagent"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    await llm.chat(messages)

    call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert call_messages[1]["reasoning_content"] == ""


@pytest.mark.asyncio
async def test_openrouter_deepseek_replays_when_thinking_disabled(
    mocker, mock_chat_completion
):
    """Replay happens regardless of this call's own thinking setting.

    What the provider requires back is the reasoning content it produced
    earlier in this tool chain, so whether thinking is requested again on
    the current call cannot gate the replay.
    """
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")
    messages = [
        {"role": "user", "content": "Search xagent"},
        {
            "role": "assistant",
            "content": "",
            PROVIDER_STATE_METADATA_KEY: _deepseek_provider_state("prior thought"),
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    await llm.chat(messages, thinking={"type": "disabled"})

    call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert call_messages[1]["reasoning_content"] == "prior thought"


def _reasoning_delta_chunk(
    *,
    reasoning_content: object = None,
    reasoning: object = "unset",
    finish_reason: object = None,
) -> SimpleNamespace:
    delta_kwargs: dict = {
        "content": None,
        "tool_calls": None,
        "reasoning_content": reasoning_content,
    }
    if reasoning != "unset":
        delta_kwargs["reasoning"] = reasoning
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(**delta_kwargs),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "reasoning-delta"},
    )


def _tool_call_delta_chunk(
    *,
    call_id: object,
    index: int,
    name: object = None,
    arguments: str = "",
    finish_reason: object = None,
) -> SimpleNamespace:
    tool_call = SimpleNamespace(
        id=call_id,
        index=index,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[tool_call],
                    reasoning_content=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "tool-call-delta"},
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_captures_reasoning_provider_state(mocker):
    """Streamed reasoning content accumulates onto the tool-call chunk's
    raw payload as captured provider state.
    """

    async def stream():
        yield _reasoning_delta_chunk(reasoning_content="Think first.")
        yield _tool_call_delta_chunk(
            call_id="call_1",
            index=0,
            name="search",
            arguments='{"query":"xagent"}',
            finish_reason="tool_calls",
        )

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=_single_tool_schema("search"),
        )
    ]

    tool_chunks = [chunk for chunk in chunks if chunk.is_tool_call()]
    assert tool_chunks, "expected at least one tool-call chunk"
    assert tool_chunks[-1].raw[PROVIDER_STATE_METADATA_KEY] == (
        _deepseek_provider_state("Think first.")
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_captures_reasoning_alias_delta(mocker):
    """A streamed delta using the ``reasoning`` alias is captured too.

    A delta that carries the alias and never ``reasoning_content`` must
    still be recognized -- this is the branch ``_delta_reasoning_content``'s
    override exists for, since the base class's own delta check only ever
    recognizes ``reasoning_content``.
    """

    async def stream():
        yield _reasoning_delta_chunk(reasoning_content=None, reasoning="Alt thinking.")
        yield _tool_call_delta_chunk(
            call_id="call_1",
            index=0,
            name="search",
            arguments='{"query":"xagent"}',
            finish_reason="tool_calls",
        )

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=_single_tool_schema("search"),
        )
    ]

    tool_chunks = [chunk for chunk in chunks if chunk.is_tool_call()]
    assert tool_chunks, "expected at least one tool-call chunk"
    assert tool_chunks[-1].raw[PROVIDER_STATE_METADATA_KEY] == (
        _deepseek_provider_state("Alt thinking.")
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_provider_state_survives_tool_truncation(
    mocker,
):
    """Withholding a partial-looking argument tail preserves captured state.

    When the protocol adapter holds back a tail that looks like the start of
    a serialized tool-call marker (``_safe_streaming_tool_chunk``'s
    ``dataclasses.replace``), the truncated chunk it yields in that moment
    must still carry the captured provider state -- ``replace()`` only
    rewrites ``tool_calls``, never ``raw``.
    """

    async def stream():
        yield _reasoning_delta_chunk(reasoning_content="Think first.")
        # The tail ``<dsml`` looks like the start of DeepSeek's serialized
        # tool-call marker (see ``_PARTIAL_MARKER_TARGET``), so the adapter
        # withholds it from this chunk instead of passing it through.
        yield _tool_call_delta_chunk(
            call_id="call_1", index=0, name="search", arguments='{"query":"xa<dsml'
        )
        yield _tool_call_delta_chunk(
            call_id=None, index=0, arguments='ing"}', finish_reason="tool_calls"
        )

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=_single_tool_schema("search"),
        )
    ]

    tool_chunks = [chunk for chunk in chunks if chunk.is_tool_call()]
    assert tool_chunks, "expected at least one tool-call chunk"
    assert all(
        chunk.raw[PROVIDER_STATE_METADATA_KEY]
        == _deepseek_provider_state("Think first.")
        for chunk in tool_chunks
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_replays_reasoning_content(mocker):
    """``stream_chat``'s outbound request replays captured reasoning too.

    ``stream_chat`` runs through the same ``_build_request_messages`` /
    ``_prepare_messages_for_request`` path as ``chat``, so captured provider
    state on a prior assistant tool-call message must be translated back to
    ``reasoning_content`` on the request the streaming entry point actually
    sends, not only on the non-streaming one.
    """

    async def stream():
        yield _tool_call_delta_chunk(
            call_id="call_2",
            index=0,
            name="search",
            arguments='{"query":"xagent"}',
            finish_reason="tool_calls",
        )

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")
    messages = [
        {"role": "user", "content": "Search xagent"},
        {
            "role": "assistant",
            "content": "",
            PROVIDER_STATE_METADATA_KEY: _deepseek_provider_state("prior thought"),
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            messages, tools=_single_tool_schema("search")
        )
    ]

    assert chunks, "expected at least one streamed chunk"
    call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert call_messages[1]["reasoning_content"] == "prior thought"
    assert PROVIDER_STATE_METADATA_KEY not in call_messages[1]


@pytest.mark.asyncio
async def test_openrouter_deepseek_declared_thinking_stream_captures_and_replays_without_request_thinking(
    mocker,
):
    """Streaming counterpart of the non-streaming capture-then-replay test.

    Each half of this round trip is already pinned on its own -- streamed
    capture, streamed replay, and the declared-ability default. What is only
    covered here is the composition: one streamed tool-call turn captures
    reasoning, and the next streamed turn sends it back, with neither call
    passing ``thinking`` at all.
    """

    async def first_stream():
        yield _reasoning_delta_chunk(reasoning_content="Use the search tool first")
        yield _tool_call_delta_chunk(
            call_id="call_1",
            index=0,
            name="search",
            arguments='{"query":"xagent"}',
            finish_reason="tool_calls",
        )

    async def second_stream():
        yield _tool_call_delta_chunk(
            call_id="call_2",
            index=0,
            name="search",
            arguments='{"query":"xagent"}',
            finish_reason="tool_calls",
        )

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        first_stream(),
        second_stream(),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    round1 = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=_single_tool_schema("search"),
        )
    ]
    assert (
        "extra_body" not in mock_client.chat.completions.create.call_args_list[0].kwargs
    )
    tool_chunks = [chunk for chunk in round1 if chunk.is_tool_call()]
    assert tool_chunks, "expected at least one tool-call chunk"
    captured = tool_chunks[-1].raw[PROVIDER_STATE_METADATA_KEY]
    assert captured == _deepseek_provider_state("Use the search tool first")

    messages = [
        {"role": "user", "content": "Search xagent"},
        {
            "role": "assistant",
            "content": "",
            PROVIDER_STATE_METADATA_KEY: captured,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]
    async for _chunk in llm.stream_chat(messages, tools=_single_tool_schema("search")):
        pass

    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert "extra_body" not in second_call
    sent = second_call["messages"]
    assert sent[1]["reasoning_content"] == "Use the search tool first"
    assert PROVIDER_STATE_METADATA_KEY not in sent[1]


@pytest.mark.asyncio
async def test_openrouter_deepseek_default_first_turn_request_is_byte_identical(
    mocker, mock_chat_completion
):
    """The default first-turn request is unchanged by reasoning replay.

    With thinking left unset (``thinking=None``, which still renders
    disabled) and no prior assistant tool-call message in the history, the
    ``_prepare_messages_for_request`` override has nothing to rewrite --
    ``restore_deepseek_reasoning_content`` only touches assistant messages
    that already carry ``tool_calls``. So every message on the wire is
    exactly what this client sent before replay existed, and ``extra_body``
    renders the same disabled payload, because
    ``_prepare_provider_reasoning_extra_body`` is not modified.

    This is the request-level counterpart of the ``extra_body`` check in
    ``test_openrouter_deepseek_defaults_to_disabled_thinking``: that one
    pins the rendered reasoning payload, this one pins the whole request.
    """
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    await llm.chat([{"role": "user", "content": "Hello"}])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert call_kwargs["extra_body"] == {
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }


@pytest.mark.asyncio
async def test_openrouter_non_deepseek_does_not_capture_or_replay_reasoning(mocker):
    """A non-deepseek slug is untouched by any of the three hooks.

    No capture, no replay, and an upstream-supplied internal marker is still
    stripped by the shared base-class sanitization rather than by anything
    added here. The response used below deliberately *does* carry
    reasoning_content on a tool call, so the model-name gate is actually
    exercised instead of passing vacuously on a response with nothing to
    capture.
    """
    response = _openrouter_tool_call_response(
        reasoning_content="should never be captured for this model"
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="anthropic/claude-sonnet-4.6", api_key="test-key")
    messages = [
        {
            "role": "assistant",
            "content": "",
            PROVIDER_STATE_METADATA_KEY: _deepseek_provider_state("stale"),
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    result = await llm.chat(
        messages, tools=_single_tool_schema("search"), thinking={"type": "enabled"}
    )

    call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert PROVIDER_STATE_METADATA_KEY not in call_messages[0]
    assert "reasoning_content" not in call_messages[0]
    assert PROVIDER_STATE_METADATA_KEY not in result


@pytest.mark.asyncio
async def test_openrouter_deepseek_round_trips_reasoning_across_two_calls(
    mocker, mock_chat_completion
):
    """A full two-call chain captures then replays the same content.

    Reasoning content is captured from the first response and replayed
    verbatim on the assistant history sent in the second call -- the
    end-to-end behavior the provider's 400 demands.
    """
    first_response = _openrouter_tool_call_response(
        reasoning_content="Search first, then answer."
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = first_response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    first_result = await llm.chat(
        [{"role": "user", "content": "Search xagent"}],
        tools=_single_tool_schema("search"),
        thinking={"type": "enabled"},
    )

    second_messages = [
        {"role": "user", "content": "Search xagent"},
        {
            "role": "assistant",
            "content": "",
            PROVIDER_STATE_METADATA_KEY: first_result[PROVIDER_STATE_METADATA_KEY],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "found it"},
    ]
    mock_client.chat.completions.create.return_value = mock_chat_completion

    await llm.chat(second_messages, tools=_single_tool_schema("search"))

    second_call_messages = mock_client.chat.completions.create.call_args.kwargs[
        "messages"
    ]
    assert second_call_messages[1]["reasoning_content"] == "Search first, then answer."
    assert PROVIDER_STATE_METADATA_KEY not in second_call_messages[1]


@pytest.mark.asyncio
async def test_openrouter_deepseek_replays_reasoning_after_mandatory_reasoning_retry(
    mocker, mock_chat_completion
):
    """Reasoning captured by the enable-thinking retry replays later.

    When the compat retry turns thinking on to recover a mandatory-reasoning
    400, the content captured from that recovered call must replay correctly
    on a later request. Recovering the first call was already possible; the
    second call in the same tool chain used to fail anyway, and this pins
    that it no longer does.
    """
    first_response = _openrouter_tool_call_response(
        reasoning_content="Reasoned after enabling thinking."
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(
            "OpenRouter bad request (400): reasoning is mandatory for this "
            "model and cannot be disabled"
        ),
        first_response,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    result = await llm.chat(
        [{"role": "user", "content": "Search xagent"}],
        tools=_single_tool_schema("search"),
    )

    assert mock_client.chat.completions.create.await_count == 2
    assert result[PROVIDER_STATE_METADATA_KEY] == _deepseek_provider_state(
        "Reasoned after enabling thinking."
    )

    mock_client.chat.completions.create.side_effect = None
    mock_client.chat.completions.create.return_value = mock_chat_completion

    second_messages = [
        {"role": "user", "content": "Search xagent"},
        {
            "role": "assistant",
            "content": "",
            PROVIDER_STATE_METADATA_KEY: result[PROVIDER_STATE_METADATA_KEY],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "found it"},
    ]
    await llm.chat(second_messages, tools=_single_tool_schema("search"))

    second_call_messages = mock_client.chat.completions.create.call_args.kwargs[
        "messages"
    ]
    assert (
        second_call_messages[1]["reasoning_content"]
        == "Reasoned after enabling thinking."
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_prefix_retry_preserves_provider_state(
    mocker, mock_chat_completion
):
    """The prefix-stripping retry keeps the provider-state marker.

    ``_strip_assistant_tool_call_prefixes`` rebuilds messages through a
    ``dict(message)`` shallow copy, which must carry the provider-state
    marker along so the retried request still replays reasoning correctly.
    """
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(openrouter_module._DEEPSEEK_FUNCTION_PREFIX_ERROR),
        mock_chat_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")
    messages = [
        {"role": "user", "content": "Search xagent"},
        {
            "role": "assistant",
            "content": "stray prefix text",
            PROVIDER_STATE_METADATA_KEY: _deepseek_provider_state("prior thought"),
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    await llm.chat(messages, tools=_single_tool_schema("search"))

    assert mock_client.chat.completions.create.await_count == 2
    retried_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert retried_messages[1]["reasoning_content"] == "prior thought"
    assert retried_messages[1]["content"] == ""
    assert PROVIDER_STATE_METADATA_KEY not in retried_messages[1]


@pytest.mark.asyncio
async def test_openrouter_deepseek_vision_captures_reasoning_provider_state(mocker):
    """``vision_chat`` inherits reasoning capture with no vision-specific code.

    Nothing here overrides ``vision_chat`` itself; capture works because the
    base class calls the same shared hooks from both entrypoints.
    """
    response = _openrouter_tool_call_response(
        reasoning_content="Looking at the image first."
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
        abilities=["chat", "tool_calling", "vision"],
    )

    result = await llm.vision_chat(
        [{"role": "user", "content": [{"type": "text", "text": "what is this?"}]}],
        tools=_single_tool_schema("search"),
        thinking={"type": "enabled"},
    )

    assert result[PROVIDER_STATE_METADATA_KEY] == _deepseek_provider_state(
        "Looking at the image first."
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_vision_replays_reasoning_content(
    mocker, mock_chat_completion
):
    """``vision_chat`` replays captured reasoning exactly as ``chat`` does.

    Both entrypoints read the same namespace and key, because this is one
    shared mechanism rather than two parallel implementations.
    """
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "vision"],
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "what is this?"}]},
        {
            "role": "assistant",
            "content": "",
            PROVIDER_STATE_METADATA_KEY: _deepseek_provider_state("prior thought"),
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    # Replay does not look at whether thinking is enabled for this call.
    await llm.vision_chat(messages, thinking={"type": "disabled"})

    call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert call_messages[1]["reasoning_content"] == "prior thought"
    assert PROVIDER_STATE_METADATA_KEY not in call_messages[1]


@pytest.mark.asyncio
async def test_openrouter_deepseek_warns_when_capture_misses_all_known_spellings(
    mocker, caplog
):
    """The capture-miss WARNING names the keys it saw, never their content.

    It fires when thinking was requested and the response is a tool call but
    neither known reasoning spelling was captured. Naming the unrecognized
    keys it did observe is the warning's entire diagnostic value, since that
    is what identifies a renamed provider field; the values behind those
    keys must never reach a log line.
    """
    import logging

    response = _openrouter_tool_call_response(
        reasoning_content="unset",
        raw_message_extra={
            "reasoning_details": [{"type": "reasoning.text", "text": "SECRET-THOUGHT"}]
        },
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        await llm.chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=_single_tool_schema("search"),
            thinking={"type": "enabled"},
        )

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if "no reasoning content was captured" in record.message
    ]
    assert warning_messages
    assert any("reasoning_details" in message for message in warning_messages)
    for record in caplog.records:
        assert "SECRET-THOUGHT" not in record.getMessage()
        assert "Search xagent" not in record.getMessage()


@pytest.mark.asyncio
async def test_openrouter_deepseek_no_warning_when_thinking_not_requested(
    mocker, caplog
):
    """The capture-miss WARNING stays silent for a model record that never
    declared ``thinking_mode``.

    This model record's requests still go out with an explicit disable
    payload, so an empty reasoning capture here is the expected outcome
    rather than a sign that the provider renamed its reasoning field, and
    warning about it would turn the signal into noise.
    """
    import logging

    response = _openrouter_tool_call_response(reasoning_content="unset")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        await llm.chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=_single_tool_schema("search"),
        )

    assert not any(
        "no reasoning content was captured" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_warns_when_declared_thinking_capture_misses(
    mocker, caplog
):
    """The gate this change fixes: a model record that declares
    ``thinking_mode`` gets nothing sent on an unspecified request, so a
    tool-call response with no captured reasoning is a real capture miss
    and must warn, even though the caller never asked for thinking either.
    """
    import logging

    response = _openrouter_tool_call_response(
        reasoning_content="unset",
        raw_message_extra={
            "reasoning_details": [{"type": "reasoning.text", "text": "SECRET-THOUGHT"}]
        },
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        await llm.chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=_single_tool_schema("search"),
        )

    messages = [
        record.getMessage()
        for record in caplog.records
        if "no reasoning content was captured" in record.message
    ]
    assert len(messages) == 1
    assert "reasoning_details" in messages[0]
    for record in caplog.records:
        assert "SECRET-THOUGHT" not in record.getMessage()
        assert "Search xagent" not in record.getMessage()


@pytest.mark.asyncio
async def test_openrouter_deepseek_no_capture_warning_after_thinking_disabled_retry(
    mocker, caplog
):
    """The capture-miss WARNING judges each attempt by that attempt's own
    thinking configuration.

    ``chat`` retries once with thinking disabled when a structured-output
    request came back as non-JSON while thinking was on. That retry really
    does ask for no thinking, so an empty reasoning capture on its response
    is the expected outcome -- not the silent-failure mode this warning
    exists to report. Judging the retry by the first attempt's "thinking
    enabled" would print a warning that sends a reader looking for a
    renamed provider field that is not there.
    """
    import logging

    non_json_message = SimpleNamespace(
        content="Here is the answer, but it is not JSON.",
        tool_calls=None,
        reasoning_content="Thinking about the schema first.",
    )
    thinking_on_response = SimpleNamespace(
        choices=[SimpleNamespace(message=non_json_message)],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-non-json"},
    )
    thinking_off_response = _openrouter_tool_call_response(reasoning_content="unset")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        thinking_on_response,
        thinking_off_response,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        result = await llm.chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=_single_tool_schema("search"),
            response_format={"type": "json_object"},
            thinking={"type": "enabled"},
        )

    # The retry really did go out with thinking off -- otherwise this test
    # would be asserting silence on a request that never changed.
    assert mock_client.chat.completions.create.await_count == 2
    retry_extra_body = mock_client.chat.completions.create.call_args.kwargs[
        "extra_body"
    ]
    assert retry_extra_body["thinking"] == {"type": "disabled"}
    assert retry_extra_body["reasoning"] == {"enabled": False}
    assert result["type"] == "tool_call"

    assert not [
        record.message
        for record in caplog.records
        if "no reasoning content was captured" in record.message
    ]


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_warns_when_capture_misses_all_known_spellings(
    mocker, caplog
):
    """Streaming counterpart of
    ``test_openrouter_deepseek_warns_when_capture_misses_all_known_spellings``:
    the non-streaming WARNING in ``_response_provider_state`` has no
    streaming equivalent before this test -- ``_check_stream_reasoning_
    capture`` closes that gap. Same gate, checked over the whole stream
    instead of one response body: thinking requested, the stream ends with
    a tool call, but no delta ever carried a recognized reasoning field.
    """
    import logging

    async def stream():
        chunk = _tool_call_delta_chunk(
            call_id="call_1",
            index=0,
            name="search",
            arguments='{"query":"xagent"}',
            finish_reason="tool_calls",
        )
        chunk.choices[0].delta.reasoning_details = [
            {"type": "reasoning.text", "text": "SECRET-THOUGHT"}
        ]
        yield chunk

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        chunks = [
            chunk
            async for chunk in llm.stream_chat(
                [{"role": "user", "content": "Search xagent"}],
                tools=_single_tool_schema("search"),
                thinking={"type": "enabled"},
            )
        ]

    assert chunks
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if "no reasoning content was captured" in record.message
    ]
    assert warning_messages
    assert any("reasoning_details" in message for message in warning_messages)
    for record in caplog.records:
        assert "SECRET-THOUGHT" not in record.getMessage()
        assert "Search xagent" not in record.getMessage()


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_warns_when_declared_thinking_capture_misses(
    mocker, caplog
):
    """Streaming counterpart of
    ``test_openrouter_deepseek_warns_when_declared_thinking_capture_misses``:
    a model record that declares ``thinking_mode`` gets nothing sent on an
    unspecified streaming request either, so a stream that ends with a tool
    call and no captured reasoning is a real capture miss and must warn.
    """
    import logging

    async def stream():
        chunk = _tool_call_delta_chunk(
            call_id="call_1",
            index=0,
            name="search",
            arguments='{"query":"xagent"}',
            finish_reason="tool_calls",
        )
        chunk.choices[0].delta.reasoning_details = [
            {"type": "reasoning.text", "text": "SECRET-THOUGHT"}
        ]
        yield chunk

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        chunks = [
            chunk
            async for chunk in llm.stream_chat(
                [{"role": "user", "content": "Search xagent"}],
                tools=_single_tool_schema("search"),
            )
        ]

    assert chunks
    messages = [
        record.getMessage()
        for record in caplog.records
        if "no reasoning content was captured" in record.message
    ]
    assert len(messages) == 1
    assert "reasoning_details" in messages[0]
    for record in caplog.records:
        assert "SECRET-THOUGHT" not in record.getMessage()
        assert "Search xagent" not in record.getMessage()


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_no_warning_when_thinking_not_requested(
    mocker, caplog
):
    """The streaming capture-miss WARNING stays silent for a model record
    that never declared ``thinking_mode``.

    Same rule as
    ``test_openrouter_deepseek_no_warning_when_thinking_not_requested``: this
    model record's requests still go out with an explicit disable payload,
    so an empty capture here is the expected outcome, not a sign of a
    renamed provider field.
    """
    import logging

    async def stream():
        yield _tool_call_delta_chunk(
            call_id="call_1",
            index=0,
            name="search",
            arguments='{"query":"xagent"}',
            finish_reason="tool_calls",
        )

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="deepseek/deepseek-v4-flash", api_key="test-key")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        chunks = [
            chunk
            async for chunk in llm.stream_chat(
                [{"role": "user", "content": "Search xagent"}],
                tools=_single_tool_schema("search"),
            )
        ]

    assert chunks
    assert not any(
        "no reasoning content was captured" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_openrouter_non_deepseek_stream_no_capture_warning(mocker, caplog):
    """The streaming capture sentinel is gated on deepseek-authored slugs:
    a non-deepseek model streaming a tool call with thinking requested and
    no reasoning field is that provider's normal shape, not spelling drift.
    """
    import logging

    async def stream():
        yield _tool_call_delta_chunk(
            call_id="call_1",
            index=0,
            name="search",
            arguments='{"query":"xagent"}',
            finish_reason="tool_calls",
        )

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="openai/gpt-5.6-sol", api_key="test-key")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        chunks = [
            chunk
            async for chunk in llm.stream_chat(
                [{"role": "user", "content": "Search xagent"}],
                tools=_single_tool_schema("search"),
                thinking={"type": "enabled"},
            )
        ]

    assert chunks
    assert not any(
        "no reasoning content was captured" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_openrouter_deepseek_structured_non_stream_stays_disabled_and_silent(
    mocker, caplog
):
    """Non-streaming half of the structured-output rule, and of the sentinel
    agreement the streaming test below pins.

    A structured request disables thinking whatever the transport, and the
    non-streaming capture sentinel must agree, staying silent about a
    capture it knows this request could not have produced. The model record
    declares the ability specifically so this test can tell apart the two
    things the sentinel could be looking at: if it ignored this call's own
    ``response_format`` and reused the declared-ability default-open
    answer, it would treat the missing capture as a real miss and warn.
    """
    import logging

    response = _openrouter_tool_call_response(
        reasoning_content="unset",
        raw_message_extra={
            "reasoning_details": [{"type": "reasoning.text", "text": "SECRET-THOUGHT"}]
        },
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
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        await llm.chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=_single_tool_schema("search"),
            response_format={"type": "json_object"},
        )

    assert mock_client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }
    assert not [
        record.message
        for record in caplog.records
        if "no reasoning content was captured" in record.message
    ]
    for record in caplog.records:
        assert "SECRET-THOUGHT" not in record.getMessage()


@pytest.mark.asyncio
async def test_openrouter_deepseek_structured_stream_stays_disabled_and_silent(
    mocker, caplog
):
    """A structured streaming request still disables thinking unconditionally
    even for a model record that declares ``thinking_mode`` -- and the
    streaming capture sentinel must agree, staying silent about a capture it
    knows this request could not have produced.

    The model record here declares the ability specifically so this test can
    tell apart the two things the streaming sentinel could be looking at: if
    it forgot to look at this call's own ``response_format`` and instead
    reused the declared-ability default-open answer, it would wrongly treat
    the missing capture as a real miss and warn.
    """
    import logging

    async def stream():
        chunk = _tool_call_delta_chunk(
            call_id="call_1",
            index=0,
            name="search",
            arguments='{"query":"xagent"}',
            finish_reason="tool_calls",
        )
        chunk.choices[0].delta.reasoning_details = [
            {"type": "reasoning.text", "text": "SECRET-THOUGHT"}
        ]
        yield chunk

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        chunks = [
            chunk
            async for chunk in llm.stream_chat(
                [{"role": "user", "content": "Search xagent"}],
                tools=_single_tool_schema("search"),
                response_format={"type": "json_object"},
            )
        ]

    assert chunks
    assert mock_client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }
    assert not [
        record.message
        for record in caplog.records
        if "no reasoning content was captured" in record.message
    ]


@pytest.mark.asyncio
async def test_openrouter_deepseek_response_format_resend_stays_silent(mocker, caplog):
    """The capture sentinel must judge the resent request, not the caller's.

    When the endpoint rejects ``response_format`` with a 400, ``OpenAILLM.chat``
    drops it and resends -- but reuses the extra_body it already built while
    ``response_format`` was still set, so the resend still goes out with
    thinking disabled. The sentinel must recognize that: if it instead looked
    at the now-cleared ``response_format`` local, it would conclude the resend
    left thinking open and wrongly warn about a missing capture on this
    tool-call response.
    """
    import logging

    response = _openrouter_tool_call_response(
        reasoning_content="unset",
        raw_message_extra={
            "reasoning_details": [{"type": "reasoning.text", "text": "SECRET-THOUGHT"}]
        },
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _bad_request_error(
            "the model does not support response_format for this request"
        ),
        response,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.model.chat.basic.openrouter"
    ):
        await llm.chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=_single_tool_schema("search"),
            response_format={"type": "json_object"},
        )

    assert mock_client.chat.completions.create.await_count == 2
    for call in mock_client.chat.completions.create.call_args_list:
        assert call.kwargs["extra_body"] == {
            "reasoning": {"enabled": False},
            "thinking": {"type": "disabled"},
        }
    assert not [
        record.message
        for record in caplog.records
        if "no reasoning content was captured" in record.message
    ]
    for record in caplog.records:
        assert "SECRET-THOUGHT" not in record.getMessage()
