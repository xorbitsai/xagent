from types import SimpleNamespace

import httpx
import openai
import pytest

from xagent.core.model.chat.basic.dashscope import DashScopeLLM
from xagent.core.model.chat.types import ChunkType


def _tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }


def _thinking_tool_choice_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        "Error code: 400 - {'error': {'message': '<400> "
        "InternalError.Algo.InvalidParameter: The tool_choice parameter does "
        "not support being set to required or object in thinking mode'}}",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "POST",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            ),
        ),
        body={
            "error": {
                "message": "<400> InternalError.Algo.InvalidParameter: "
                "The tool_choice parameter does not support being set to "
                "required or object in thinking mode",
                "type": "invalid_request_error",
                "code": "invalid_parameter_error",
            }
        },
    )


def _generic_bad_request_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        "Error code: 400 - {'error': {'message': 'Unrelated invalid request'}}",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "POST",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            ),
        ),
        body={
            "error": {
                "message": "Unrelated invalid request",
                "type": "invalid_request_error",
                "code": "invalid_parameter_error",
            }
        },
    )


async def _stream_token():
    yield SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))],
        usage=None,
    )


@pytest.fixture
def dashscope_llm() -> DashScopeLLM:
    return DashScopeLLM(
        model_name="qwen3.5-plus",
        api_key="test-key",
        abilities=["thinking_mode"],
    )


@pytest.mark.asyncio
async def test_strict_tool_call_disables_thinking_for_chat(
    dashscope_llm, mock_tool_call_completion, mocker
):
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_tool_call_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    await dashscope_llm.chat(
        [{"role": "user", "content": "Use the weather tool."}],
        tools=[_tool_schema()],
        tool_choice="required",
        thinking={"type": "enabled", "enable": True},
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == "required"
    assert call_kwargs["extra_body"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_strict_tool_call_disables_thinking_for_stream(dashscope_llm, mocker):
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = _stream_token()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    chunks = [
        chunk
        async for chunk in dashscope_llm.stream_chat(
            [{"role": "user", "content": "Use the weather tool."}],
            tools=[_tool_schema()],
            tool_choice="required",
            thinking={"type": "enabled", "enable": True},
        )
    ]

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert [chunk.type for chunk in chunks] == [ChunkType.TOKEN]
    assert call_kwargs["tool_choice"] == "required"
    assert call_kwargs["extra_body"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_thinking_tool_choice_error_falls_back_to_auto(
    dashscope_llm, mock_tool_call_completion, mocker
):
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _thinking_tool_choice_error(),
        mock_tool_call_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    result = await dashscope_llm.chat(
        [{"role": "user", "content": "Use the weather tool."}],
        tools=[_tool_schema()],
        tool_choice="required",
    )

    first_call = mock_client.chat.completions.create.call_args_list[0].kwargs
    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert result["type"] == "tool_call"
    assert first_call["tool_choice"] == "required"
    assert first_call["extra_body"] == {"enable_thinking": False}
    assert second_call["tool_choice"] == "auto"
    assert second_call["extra_body"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_non_thinking_tool_choice_bad_request_does_not_fallback(
    dashscope_llm, mocker
):
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = _generic_bad_request_error()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    with pytest.raises(RuntimeError, match="Unrelated invalid request"):
        await dashscope_llm.chat(
            [{"role": "user", "content": "Use the weather tool."}],
            tools=[_tool_schema()],
            tool_choice="required",
        )

    assert mock_client.chat.completions.create.call_count == 1
