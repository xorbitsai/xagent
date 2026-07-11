from types import SimpleNamespace

import pytest

from xagent.core.model.chat.basic.deepseek import (
    DEEPSEEK_DEFAULT_BASE_URL,
    DEEPSEEK_PROVIDER_STATE_NAMESPACE,
    DEEPSEEK_REASONING_CONTENT_STATE_KEY,
    DeepSeekLLM,
)
from xagent.core.model.chat.basic.openai import PROVIDER_STATE_METADATA_KEY, OpenAILLM
from xagent.core.model.chat.tool_protocol import get_tool_protocol_error
from xagent.core.model.chat.types import ChunkType


class TestDeepSeekLLM:
    @pytest.fixture
    def llm(self):
        return DeepSeekLLM(api_key="test-api-key")

    def test_defaults(self, llm):
        assert llm.model_name == "deepseek-v4-flash"
        assert llm.base_url == DEEPSEEK_DEFAULT_BASE_URL
        assert llm.api_key == "test-api-key"
        assert llm.abilities == ["chat", "tool_calling", "thinking_mode"]

    def test_api_key_env_fallback(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek-key")

        llm = DeepSeekLLM(api_key=None, base_url=None)

        assert llm.api_key == "env-deepseek-key"
        assert llm.base_url == DEEPSEEK_DEFAULT_BASE_URL

    def test_api_key_falls_back_to_openai_key_after_deepseek_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-compatible-key")

        llm = DeepSeekLLM(api_key=None)

        assert llm.api_key == "openai-compatible-key"

    def test_blank_api_key_falls_back_to_deepseek_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-compatible-key")

        assert DeepSeekLLM(api_key="").api_key == "env-deepseek-key"
        assert DeepSeekLLM(api_key="   ").api_key == "env-deepseek-key"

    def test_api_key_ignores_placeholders_before_openai_fallback(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")
        monkeypatch.setenv("OPENAI_API_KEY", "your-openai-api-key")

        llm = DeepSeekLLM(api_key="your-deepseek-api-key")

        assert llm.api_key == ""

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported DeepSeek model"):
            DeepSeekLLM(model_name="deepseek-chat", api_key="test-api-key")
        with pytest.raises(ValueError, match="Unsupported DeepSeek model"):
            DeepSeekLLM(model_name="deepseek-reasoner", api_key="test-api-key")
        with pytest.raises(ValueError, match="Unsupported DeepSeek model"):
            DeepSeekLLM(model_name="not-a-deepseek-model", api_key="test-api-key")

    def test_structured_output_capabilities(self, llm):
        assert llm.supports_json_schema_response_format is False
        assert llm.supports_json_object_response_format is True

    def test_deepseek_is_not_openai_subclass(self):
        assert not issubclass(DeepSeekLLM, OpenAILLM)

    @pytest.mark.asyncio
    async def test_chat_rejects_serialized_tool_protocol_content(self, llm, mocker):
        message = SimpleNamespace(
            content="<｜｜DSML｜｜tool_calls>",
            tool_calls=None,
            reasoning_content=None,
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=None,
            model_dump=lambda: {"id": "deepseek-invalid-protocol"},
        )
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = response
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        result = await llm.chat(
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

        error = get_tool_protocol_error(result)
        assert error is not None
        assert error["code"] == "serialized_tool_call_content"

    def test_provider_reasoning_hook_disables_deepseek_thinking(self, llm):
        assert llm._prepare_provider_reasoning_extra_body(
            extra_body={"trace_id": "abc"},
            thinking={"type": "disabled"},
            tools=None,
            response_format=None,
            output_config=None,
            is_streaming=False,
        ) == {
            "trace_id": "abc",
            "thinking": {"type": "disabled"},
        }

    @pytest.mark.asyncio
    async def test_explicit_thinking_enabled_uses_deepseek_extra_body(
        self, llm, mock_chat_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        await llm.chat(
            [{"role": "user", "content": "Hello"}],
            thinking={"type": "enabled"},
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
        assert "enable_thinking" not in call_kwargs["extra_body"]

    @pytest.mark.asyncio
    async def test_chat_disables_thinking_by_default(
        self, llm, mock_chat_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        await llm.chat([{"role": "user", "content": "Hello"}])

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_stream_chat_disables_thinking_by_default(self, llm, mocker):
        async def stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content=None,
                            content="Hello",
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
                model_dump=lambda: {"id": "content"},
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content=None,
                            content=None,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
                model_dump=lambda: {"id": "end"},
            )

        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = stream()
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        chunks = [
            chunk
            async for chunk in llm.stream_chat([{"role": "user", "content": "Hello"}])
        ]

        assert [chunk.delta for chunk in chunks if chunk.type == ChunkType.TOKEN] == [
            "Hello"
        ]
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_tool_calls_disable_thinking_by_default(
        self, llm, mock_tool_call_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_tool_call_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        tools = [
            {
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
        ]

        result = await llm.chat([{"role": "user", "content": "Weather?"}], tools=tools)

        assert result["type"] == "tool_call"
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_response_format_disables_thinking_by_default(
        self, llm, mock_json_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_json_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        await llm.chat(
            [{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_json_schema_response_format_uses_deepseek_json_object(
        self, llm, mock_json_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_json_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        await llm.chat(
            [{"role": "user", "content": "Return JSON"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "action_decision",
                    "strict": True,
                    "schema": {"type": "object"},
                },
            },
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_output_config_disables_thinking_by_default(
        self, llm, mock_json_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_json_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        await llm.chat(
            [{"role": "user", "content": "Return schema output"}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {"title": "Result", "type": "object"},
                }
            },
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_reasoning_effort_is_forwarded(
        self, llm, mock_chat_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        await llm.chat(
            [{"role": "user", "content": "Hello"}],
            reasoning_effort="max",
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "max"
        assert "reasoning_effort" not in call_kwargs.get("extra_body", {})

    @pytest.mark.asyncio
    async def test_env_reasoning_effort_is_forwarded_as_top_level_param(
        self, llm, mock_chat_completion, mocker, monkeypatch
    ):
        monkeypatch.setenv("DEEPSEEK_REASONING_EFFORT", "high")
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        await llm.chat([{"role": "user", "content": "Hello"}])

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"
        assert "reasoning_effort" not in call_kwargs.get("extra_body", {})

    @pytest.mark.asyncio
    async def test_caller_extra_body_is_merged_with_deepseek_extra_body(
        self, llm, mock_chat_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        await llm.chat(
            [{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
            extra_body={"trace_id": "abc123"},
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "trace_id": "abc123",
            "thinking": {"type": "disabled"},
        }

    @pytest.mark.asyncio
    async def test_structured_output_retry_disables_thinking_with_deepseek_payload(
        self, llm, mocker
    ):
        first_message = SimpleNamespace(
            content="not json",
            tool_calls=None,
            reasoning_content="Need to reason first",
        )
        second_message = SimpleNamespace(
            content='{"status":"ok"}',
            tool_calls=None,
            reasoning_content=None,
        )
        first_response = SimpleNamespace(
            choices=[SimpleNamespace(message=first_message)],
            usage=None,
            model_dump=lambda: {"id": "deepseek-first"},
        )
        second_response = SimpleNamespace(
            choices=[SimpleNamespace(message=second_message)],
            usage=None,
            model_dump=lambda: {"id": "deepseek-second"},
        )

        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        result = await llm.chat(
            [{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
            thinking={"type": "enabled"},
            reasoning_effort="max",
        )

        assert result["type"] == "text"
        assert result["content"] == '{"status":"ok"}'
        second_call_kwargs = mock_client.chat.completions.create.call_args_list[
            1
        ].kwargs
        assert second_call_kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        assert second_call_kwargs["reasoning_effort"] == "max"
        assert "reasoning_effort" not in second_call_kwargs["extra_body"]
        assert "enable_thinking" not in second_call_kwargs["extra_body"]

    @pytest.mark.asyncio
    async def test_reasoning_content_is_preserved_for_text_response(self, llm, mocker):
        message = SimpleNamespace(
            content="Hello from DeepSeek",
            tool_calls=None,
            reasoning_content="Detailed reasoning",
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=None,
            model_dump=lambda: {"id": "deepseek-text"},
        )

        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = response
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        result = await llm.chat(
            [{"role": "user", "content": "Hello"}],
            thinking={"type": "enabled"},
        )

        assert result["type"] == "text"
        assert result["content"] == "Hello from DeepSeek"
        assert result["reasoning_content"] == "Detailed reasoning"
        assert result["reasoning"] == "Detailed reasoning"

    @pytest.mark.asyncio
    async def test_reasoning_content_is_preserved_for_tool_calls(self, llm, mocker):
        tool_call = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="search", arguments='{"query":"xagent"}'),
        )
        message = SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
            reasoning_content="Use the search tool first",
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=None,
            model_dump=lambda: {"id": "deepseek-tool"},
        )

        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = response
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        result = await llm.chat(
            [{"role": "user", "content": "Search xagent"}], tools=[{}]
        )

        assert result["type"] == "tool_call"
        assert result["reasoning_content"] == "Use the search tool first"
        assert result["reasoning"] == "Use the search tool first"
        assert result[PROVIDER_STATE_METADATA_KEY] == {
            DEEPSEEK_PROVIDER_STATE_NAMESPACE: {
                DEEPSEEK_REASONING_CONTENT_STATE_KEY: "Use the search tool first"
            }
        }

    @pytest.mark.asyncio
    async def test_empty_reasoning_content_is_preserved_for_tool_calls(
        self, llm, mocker
    ):
        tool_call = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="search", arguments='{"query":"xagent"}'),
        )
        message = SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
            reasoning_content="",
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=None,
            model_dump=lambda: {"id": "deepseek-tool"},
        )

        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = response
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        result = await llm.chat(
            [{"role": "user", "content": "Search xagent"}],
            tools=[{}],
            thinking={"type": "enabled"},
        )

        assert result["type"] == "tool_call"
        assert result["reasoning_content"] == ""
        assert result["reasoning"] == ""
        assert result[PROVIDER_STATE_METADATA_KEY] == {
            DEEPSEEK_PROVIDER_STATE_NAMESPACE: {
                DEEPSEEK_REASONING_CONTENT_STATE_KEY: ""
            }
        }

    @pytest.mark.asyncio
    async def test_assistant_tool_call_replays_empty_reasoning_content(
        self, llm, mock_chat_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )
        messages = [
            {"role": "user", "content": "Search xagent"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"xagent"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Tool search returned: {}",
            },
        ]

        await llm.chat(messages, tools=[{}])

        call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert call_messages[1]["reasoning_content"] == ""

    @pytest.mark.asyncio
    async def test_thinking_enabled_preserves_existing_empty_reasoning_content(
        self, llm, mock_chat_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"xagent"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Tool search returned: {}",
            },
        ]

        await llm.chat(messages, tools=[{}], thinking={"type": "enabled"})

        call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert call_messages[0]["reasoning_content"] == ""

    @pytest.mark.asyncio
    async def test_converts_provider_state_to_reasoning_content(
        self, llm, mock_chat_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                PROVIDER_STATE_METADATA_KEY: {
                    DEEPSEEK_PROVIDER_STATE_NAMESPACE: {
                        DEEPSEEK_REASONING_CONTENT_STATE_KEY: ""
                    }
                },
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"xagent"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Tool search returned: {}",
            },
        ]

        await llm.chat(messages, tools=[{}])

        call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert call_messages[0]["reasoning_content"] == ""
        assert PROVIDER_STATE_METADATA_KEY not in call_messages[0]

    @pytest.mark.asyncio
    async def test_converts_real_provider_state_to_reasoning_content(
        self, llm, mock_chat_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                PROVIDER_STATE_METADATA_KEY: {
                    DEEPSEEK_PROVIDER_STATE_NAMESPACE: {
                        DEEPSEEK_REASONING_CONTENT_STATE_KEY: (
                            "I should inspect available files first."
                        )
                    }
                },
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"xagent"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Tool search returned: {}",
            },
        ]

        await llm.chat(messages)

        call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert (
            call_messages[0]["reasoning_content"]
            == "I should inspect available files first."
        )
        assert PROVIDER_STATE_METADATA_KEY not in call_messages[0]

    @pytest.mark.asyncio
    async def test_thinking_disabled_still_replays_empty_reasoning_content(
        self, llm, mock_tool_call_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_tool_call_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"xagent"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Tool search returned: {}",
            },
        ]

        await llm.chat(messages, tools=[{}], thinking={"type": "disabled"})

        call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert call_messages[0]["reasoning_content"] == ""

    @pytest.mark.asyncio
    async def test_final_answer_call_replays_empty_reasoning_content_without_tools(
        self, llm, mock_chat_completion, mocker
    ):
        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )
        messages = [
            {"role": "user", "content": "Hi, what can you do?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "list_all_user_files",
                            "arguments": '{"limit":10}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Tool list_all_user_files returned: {}",
            },
        ]

        await llm.chat(messages)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert call_kwargs["messages"][1]["reasoning_content"] == ""

    @pytest.mark.asyncio
    async def test_stream_reasoning_content_is_accumulated_without_token_output(
        self, llm, mocker
    ):
        async def stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content="Think first.",
                            content=None,
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
                model_dump=lambda: {"id": "reasoning"},
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content=None,
                            content="Final answer",
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
                model_dump=lambda: {"id": "content"},
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content=None,
                            content=None,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
                model_dump=lambda: {"id": "end"},
            )

        mock_client = mocker.AsyncMock()
        mock_client.chat.completions.create.return_value = stream()
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )

        chunks = [
            chunk
            async for chunk in llm.stream_chat(
                [{"role": "user", "content": "Hello"}],
                thinking={"type": "enabled"},
            )
        ]

        token_chunks = [chunk for chunk in chunks if chunk.type == ChunkType.TOKEN]
        assert [chunk.delta for chunk in token_chunks] == ["Final answer"]
        assert token_chunks[0].raw["reasoning_content"] == "Think first."

        end_chunk = next(chunk for chunk in chunks if chunk.type == ChunkType.END)
        assert end_chunk.raw["reasoning_content"] == "Think first."

    @pytest.mark.asyncio
    async def test_list_available_models_returns_curated_v4_models(self):
        models = await DeepSeekLLM.list_available_models("test-api-key")

        assert [model["id"] for model in models] == [
            "deepseek-v4-flash",
            "deepseek-v4-pro",
        ]
        assert all(model["owned_by"] == "deepseek" for model in models)
        assert all(
            model["abilities"] == ["chat", "tool_calling", "thinking_mode"]
            for model in models
        )
