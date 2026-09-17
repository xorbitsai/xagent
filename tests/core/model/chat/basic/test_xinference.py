import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.model.chat.basic.xinference import XinferenceLLM, _create_async_client
from xagent.core.model.chat.error import retry_on
from xagent.core.model.chat.exceptions import LLMTimeoutError
from xagent.core.model.chat.timeout_config import TimeoutConfig
from xagent.core.model.chat.types import ChunkType


class TestXinferenceLLM:
    @pytest.mark.asyncio
    async def test_stream_chat_does_not_block_event_loop_before_first_token(
        self,
    ) -> None:
        release_stream = asyncio.Event()

        class BlockingModelHandle:
            async def chat(self, **kwargs: object):
                async def generate():
                    await release_stream.wait()
                    yield {
                        "choices": [
                            {
                                "delta": {"content": "ok"},
                                "finish_reason": None,
                            }
                        ]
                    }

                return generate()

        llm = XinferenceLLM(
            model_name="qwen3.8",
            timeout_config=TimeoutConfig(
                first_token_timeout=1,
                token_interval_timeout=1,
            ),
        )
        llm._client = MagicMock()
        llm._model_handle = BlockingModelHandle()

        heartbeat_completed = asyncio.Event()

        async def heartbeat() -> None:
            await asyncio.sleep(0)
            heartbeat_completed.set()

        heartbeat_task = asyncio.create_task(heartbeat())

        async def release_later() -> None:
            await asyncio.wait_for(heartbeat_completed.wait(), timeout=0.5)
            release_stream.set()

        release_task = asyncio.create_task(release_later())
        try:
            chunks = [
                chunk
                async for chunk in llm.stream_chat(
                    messages=[{"role": "user", "content": "hello"}]
                )
            ]
        finally:
            release_stream.set()
            await release_task

        await heartbeat_task
        assert heartbeat_completed.is_set()
        assert [chunk.delta for chunk in chunks] == ["ok"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("separate_usage_chunk", [False, True])
    async def test_stream_chat_requests_and_surfaces_usage(
        self, separate_usage_chunk: bool
    ) -> None:
        captured_generate_config: dict[str, object] = {}
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "prompt_tokens_details": {"cached_tokens": 60},
        }

        class ModelHandle:
            async def chat(self, **kwargs: object):
                generate_config = kwargs["generate_config"]
                assert isinstance(generate_config, dict)
                captured_generate_config.update(generate_config)

                async def generate():
                    yield {
                        "choices": [
                            {
                                "delta": {"content": ""},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": usage,
                    }
                    if separate_usage_chunk:
                        yield {
                            "choices": [],
                            "usage": usage,
                        }

                return generate()

        llm = XinferenceLLM(model_name="qwen3.8")
        llm._client = MagicMock()
        llm._model_handle = ModelHandle()

        with patch(
            "xagent.core.model.chat.basic.xinference.add_token_usage"
        ) as add_usage:
            chunks = [
                chunk
                async for chunk in llm.stream_chat(
                    messages=[{"role": "user", "content": "hello"}]
                )
            ]

        assert captured_generate_config["stream_options"] == {"include_usage": True}
        assert [chunk.type for chunk in chunks] == [ChunkType.END, ChunkType.USAGE]
        assert chunks[-1].usage == usage
        add_usage.assert_called_once_with(
            input_tokens=100,
            output_tokens=5,
            model="qwen3.8",
            model_id="",
            call_type="stream_chat",
            cached_input_tokens=60,
        )

    @pytest.mark.asyncio
    async def test_stream_chat_enforces_timeout_while_waiting_for_first_token(
        self,
    ) -> None:
        release_stream = asyncio.Event()

        class BlockingModelHandle:
            async def chat(self, **kwargs: object):
                async def generate():
                    await release_stream.wait()
                    yield {
                        "choices": [
                            {
                                "delta": {"content": "late"},
                                "finish_reason": None,
                            }
                        ]
                    }

                return generate()

        llm = XinferenceLLM(
            model_name="qwen3.8",
            timeout_config=TimeoutConfig(
                first_token_timeout=0.02,
                token_interval_timeout=0.02,
            ),
        )
        llm._client = MagicMock()
        llm._model_handle = BlockingModelHandle()

        try:
            with pytest.raises(LLMTimeoutError, match="First token timeout"):
                async for _ in llm.stream_chat(
                    messages=[{"role": "user", "content": "hello"}]
                ):
                    pass
        finally:
            release_stream.set()

    @pytest.mark.asyncio
    async def test_chat_timeout_is_retryable(self) -> None:
        class BlockingModelHandle:
            async def chat(self, **kwargs: object):
                await asyncio.Event().wait()

        llm = XinferenceLLM(model_name="qwen3.8", timeout=0.02)
        llm._client = MagicMock()
        llm._model_handle = BlockingModelHandle()

        with pytest.raises(LLMTimeoutError, match="Xinference chat timeout") as exc:
            await llm.chat(messages=[{"role": "user", "content": "hello"}])

        assert retry_on(exc.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blocking_phase", ["get_model", "chat"])
    async def test_stream_first_token_timeout_bounds_request_initiation(
        self, blocking_phase: str
    ) -> None:
        class BlockingClient:
            async def get_model(self, model_uid: str):
                await asyncio.Event().wait()

        class BlockingModelHandle:
            async def chat(self, **kwargs: object):
                await asyncio.Event().wait()

        llm = XinferenceLLM(
            model_name="qwen3.8",
            timeout_config=TimeoutConfig(
                first_token_timeout=0.02,
                token_interval_timeout=1,
            ),
        )
        if blocking_phase == "get_model":
            llm._client = BlockingClient()
        else:
            llm._client = MagicMock()
            llm._model_handle = BlockingModelHandle()

        with pytest.raises(LLMTimeoutError, match="First token timeout"):
            async for _ in llm.stream_chat(
                messages=[{"role": "user", "content": "hello"}]
            ):
                pass

    @pytest.mark.asyncio
    async def test_stream_chat_enforces_token_interval_timeout_and_closes_stream(
        self,
    ) -> None:
        class TrackingStream:
            def __init__(self) -> None:
                self._first = True
                self.closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._first:
                    self._first = False
                    return {
                        "choices": [
                            {
                                "delta": {"content": "first"},
                                "finish_reason": None,
                            }
                        ]
                    }
                await asyncio.Event().wait()

            async def aclose(self) -> None:
                self.closed = True

        stream = TrackingStream()

        class ModelHandle:
            async def chat(self, **kwargs: object):
                return stream

        llm = XinferenceLLM(
            model_name="qwen3.8",
            timeout_config=TimeoutConfig(
                first_token_timeout=1,
                token_interval_timeout=0.02,
            ),
        )
        llm._client = MagicMock()
        llm._model_handle = ModelHandle()

        chunks = []
        with pytest.raises(LLMTimeoutError, match="Token interval timeout"):
            async for chunk in llm.stream_chat(
                messages=[{"role": "user", "content": "hello"}]
            ):
                chunks.append(chunk)

        assert [chunk.delta for chunk in chunks] == ["first"]
        assert stream.closed

    @pytest.mark.asyncio
    async def test_stream_chat_keeps_first_deadline_until_payload(self) -> None:
        class DelayedPayloadStream:
            def __init__(self) -> None:
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._index += 1
                if self._index == 1:
                    return {
                        "choices": [
                            {
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ]
                    }
                if self._index == 2:
                    await asyncio.sleep(0.05)
                    return {
                        "choices": [
                            {
                                "delta": {"content": "first payload"},
                                "finish_reason": None,
                            }
                        ]
                    }
                raise StopAsyncIteration

            async def aclose(self) -> None:
                pass

        class ModelHandle:
            async def chat(self, **kwargs: object):
                return DelayedPayloadStream()

        llm = XinferenceLLM(
            model_name="qwen3.8",
            timeout_config=TimeoutConfig(
                first_token_timeout=0.2,
                token_interval_timeout=0.02,
            ),
        )
        llm._client = MagicMock()
        llm._model_handle = ModelHandle()

        chunks = [
            chunk
            async for chunk in llm.stream_chat(
                messages=[{"role": "user", "content": "hello"}]
            )
        ]

        assert [chunk.delta for chunk in chunks] == ["first payload"]

    @pytest.mark.asyncio
    async def test_stream_chat_preserves_transport_timeout_cause(self) -> None:
        transport_error = asyncio.TimeoutError("SDK read timed out")

        class TransportTimeoutStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise transport_error

            async def aclose(self) -> None:
                pass

        class ModelHandle:
            async def chat(self, **kwargs: object):
                return TransportTimeoutStream()

        llm = XinferenceLLM(
            model_name="qwen3.8",
            timeout_config=TimeoutConfig(
                first_token_timeout=1,
                token_interval_timeout=1,
            ),
        )
        llm._client = MagicMock()
        llm._model_handle = ModelHandle()

        with pytest.raises(LLMTimeoutError, match="transport timeout") as exc:
            async for _ in llm.stream_chat(
                messages=[{"role": "user", "content": "hello"}]
            ):
                pass

        assert exc.value.__cause__ is transport_error

    @pytest.mark.asyncio
    async def test_stream_chat_outer_close_closes_inner_iterator(self) -> None:
        class TrackingStream:
            def __init__(self) -> None:
                self.closed = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                return {
                    "choices": [
                        {
                            "delta": {"content": "first"},
                            "finish_reason": None,
                        }
                    ]
                }

            async def aclose(self) -> None:
                self.closed += 1

        stream = TrackingStream()

        class ModelHandle:
            async def chat(self, **kwargs: object):
                return stream

        llm = XinferenceLLM(model_name="qwen3.8")
        llm._client = MagicMock()
        llm._model_handle = ModelHandle()
        outer = llm.stream_chat(messages=[{"role": "user", "content": "hello"}])

        chunk = await outer.__anext__()
        await outer.aclose()

        assert chunk.delta == "first"
        assert stream.closed == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("api_key", "expected_authorization"),
        [(None, None), ("secret", "Bearer secret")],
    )
    async def test_async_client_skips_sync_auth_probe(
        self, api_key: str | None, expected_authorization: str | None
    ) -> None:
        with patch(
            "requests.Session.get",
            side_effect=AssertionError("synchronous auth probe must not run"),
        ):
            client = _create_async_client("http://localhost:9997", api_key)

        try:
            assert client._headers.get("Authorization") == expected_authorization
        finally:
            await client.close()

    def test_parse_stream_chunk_accumulates_tool_arguments_by_index(self) -> None:
        llm = XinferenceLLM(model_name="qwen3.5")
        accumulated_tool_calls: dict[str, dict] = {}

        first_chunk = llm._parse_stream_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "auto_decision",
                                        "arguments": "{",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            accumulated_tool_calls,
        )
        second_chunk = llm._parse_stream_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '"action":"react"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            accumulated_tool_calls,
        )
        final_chunk = llm._parse_stream_chunk(
            {
                "choices": [
                    {
                        "delta": {"content": ""},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            accumulated_tool_calls,
        )

        assert first_chunk is not None
        assert first_chunk.type == ChunkType.TOOL_CALL
        assert second_chunk is not None
        assert second_chunk.tool_calls[0]["function"]["arguments"] == (
            '{"action":"react"}'
        )
        assert final_chunk is not None
        assert final_chunk.finish_reason == "tool_calls"
        assert final_chunk.tool_calls == [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "auto_decision",
                    "arguments": '{"action":"react"}',
                },
            }
        ]

    def test_parse_stream_chunk_does_not_merge_mismatched_tool_call_index(
        self,
    ) -> None:
        llm = XinferenceLLM(model_name="qwen3.5")
        accumulated_tool_calls: dict[str, dict] = {}

        llm._parse_stream_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "auto_decision",
                                        "arguments": "{",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            accumulated_tool_calls,
        )

        chunk = llm._parse_stream_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "function": {
                                        "arguments": '"action":"react"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            accumulated_tool_calls,
        )

        assert chunk is not None
        assert chunk.tool_calls == [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "auto_decision",
                    "arguments": "{",
                },
            }
        ]

    def test_parse_stream_chunk_handles_null_tool_call_function(self) -> None:
        llm = XinferenceLLM(model_name="qwen3.5")
        accumulated_tool_calls: dict[str, dict] = {}

        chunk = llm._parse_stream_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": None,
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            accumulated_tool_calls,
        )

        assert chunk is not None
        assert chunk.tool_calls == [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            }
        ]

    @pytest.mark.asyncio
    @patch("xagent.core.model.chat.basic.xinference._create_async_client")
    async def test_list_available_models_handles_dict_response(
        self, mock_client_factory: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(
            return_value={
                "qwen-chat-uid": {
                    "model_name": "Qwen3-8B-Instruct",
                    "model_type": "LLM",
                    "model_ability": ["chat", "vision", "tool_calling"],
                    "model_description": "Qwen chat model",
                },
                "whisper-uid": {
                    "model_name": "whisper-large-v3",
                    "model_type": "audio",
                    "model_ability": ["audio2text"],
                    "model_description": "ASR model",
                },
            }
        )
        mock_client.close = AsyncMock()
        mock_client_factory.return_value = mock_client

        models = await XinferenceLLM.list_available_models(
            base_url="http://localhost:9997", api_key="test-key"
        )

        assert len(models) == 2
        assert models[0] == {
            "id": "Qwen3-8B-Instruct",
            "model_uid": "qwen-chat-uid",
            "model_type": "LLM",
            "model_ability": ["chat", "vision", "tool_calling"],
            "abilities": ["chat", "vision", "tool_calling"],
            "description": "Qwen chat model",
        }
        assert models[1] == {
            "id": "whisper-large-v3",
            "model_uid": "whisper-uid",
            "model_type": "audio",
            "model_ability": ["asr"],
            "abilities": ["asr"],
            "description": "ASR model",
        }
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("xagent.core.model.chat.basic.xinference._create_async_client")
    async def test_list_available_models_preserves_embedding_ability(
        self, mock_client_factory: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(
            return_value={
                "embedding-uid": {
                    "model_name": "Qwen3-Embedding-8B",
                    "model_type": "embedding",
                    "model_ability": ["embedding"],
                    "model_description": "Embedding model",
                }
            }
        )
        mock_client.close = AsyncMock()
        mock_client_factory.return_value = mock_client

        models = await XinferenceLLM.list_available_models(
            base_url="http://localhost:9997", api_key="test-key"
        )

        assert models == [
            {
                "id": "Qwen3-Embedding-8B",
                "model_uid": "embedding-uid",
                "model_type": "embedding",
                "model_ability": ["embedding"],
                "abilities": ["embedding"],
                "description": "Embedding model",
            }
        ]

    @pytest.mark.asyncio
    @patch("xagent.core.model.chat.basic.xinference._create_async_client")
    async def test_list_available_models_handles_legacy_list_response(
        self, mock_client_factory: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(
            return_value=[
                {
                    "id": "legacy-chat-uid",
                    "model_name": "legacy-chat",
                    "model_type": "LLM",
                    "model_ability": ["chat"],
                    "model_description": "Legacy chat model",
                }
            ]
        )
        mock_client.close = AsyncMock()
        mock_client_factory.return_value = mock_client

        models = await XinferenceLLM.list_available_models(
            base_url="http://localhost:9997", api_key="test-key"
        )

        assert models == [
            {
                "id": "legacy-chat",
                "model_uid": "legacy-chat-uid",
                "model_type": "LLM",
                "model_ability": ["chat"],
                "abilities": ["chat"],
                "description": "Legacy chat model",
            }
        ]

    @pytest.mark.asyncio
    @patch("xagent.core.model.chat.basic.xinference._create_async_client")
    async def test_list_available_models_remains_responsive_and_closes_client(
        self, mock_client_factory: MagicMock
    ) -> None:
        discovery_started = asyncio.Event()
        release_discovery = asyncio.Event()

        async def list_models():
            discovery_started.set()
            await release_discovery.wait()
            return {}

        mock_client = MagicMock()
        mock_client.list_models = list_models
        mock_client.close = AsyncMock()
        mock_client_factory.return_value = mock_client

        discovery_task = asyncio.create_task(
            XinferenceLLM.list_available_models("http://localhost:9997")
        )
        try:
            # These guards bound a hang, not model-discovery latency.
            await asyncio.wait_for(discovery_started.wait(), timeout=30)
            heartbeat = asyncio.create_task(asyncio.sleep(0))
            await asyncio.wait_for(heartbeat, timeout=30)
            assert not discovery_task.done()
        finally:
            release_discovery.set()
            models = await asyncio.wait_for(discovery_task, timeout=30)

        assert models == []
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "xagent.core.model.chat.basic.xinference.asyncio.sleep", new_callable=AsyncMock
    )
    @patch("xagent.core.model.chat.basic.xinference._create_async_client")
    async def test_list_available_models_closes_client_after_errors(
        self, mock_client_factory: MagicMock, _mock_sleep: AsyncMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(side_effect=RuntimeError("boom"))
        mock_client.close = AsyncMock()
        mock_client_factory.return_value = mock_client

        with pytest.raises(RuntimeError, match="Cannot connect to Xinference"):
            await XinferenceLLM.list_available_models("http://localhost:9997")

        assert mock_client_factory.call_count == 3
        assert mock_client.close.await_count == 3

    @pytest.mark.asyncio
    @patch(
        "xagent.core.model.chat.basic.xinference._MODEL_DISCOVERY_TIMEOUT_SECONDS",
        0.01,
    )
    @patch(
        "xagent.core.model.chat.basic.xinference.asyncio.sleep", new_callable=AsyncMock
    )
    @patch("xagent.core.model.chat.basic.xinference._create_async_client")
    async def test_list_available_models_bounds_and_cleans_up_slow_discovery(
        self, mock_client_factory: MagicMock, _mock_sleep: AsyncMock
    ) -> None:
        async def list_models():
            await asyncio.Event().wait()

        mock_client = MagicMock()
        mock_client.list_models = list_models
        mock_client.close = AsyncMock()
        mock_client_factory.return_value = mock_client

        with pytest.raises(RuntimeError, match="Cannot connect to Xinference") as exc:
            await XinferenceLLM.list_available_models("http://localhost:9997")

        assert isinstance(exc.value.__cause__, LLMTimeoutError)
        assert mock_client_factory.call_count == 3
        assert mock_client.close.await_count == 3


class TestProcessChatResponse:
    """Tests for ``XinferenceLLM._process_chat_response``.

    Reasoning-capable models served via Xinference (``qwen3-thinking``,
    ``deepseek-r1``, etc.) can return a response whose ``content`` is empty
    while ``reasoning_content`` carries the partial answer — most commonly
    when ``max_tokens`` truncates the generation before the final answer is
    produced. The adapter must surface those responses as text instead of
    raising ``Invalid Xinference response``.
    """

    def _make_llm(self) -> XinferenceLLM:
        return XinferenceLLM(model_name="qwen3-thinking")

    def test_plain_text_response_is_returned_as_text(self) -> None:
        llm = self._make_llm()
        result = llm._process_chat_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello there",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
        )

        assert result["type"] == "text"
        assert result["content"] == "Hello there"
        assert "reasoning_content" not in result

    def test_text_response_with_reasoning_content_attaches_reasoning(self) -> None:
        llm = self._make_llm()
        result = llm._process_chat_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "42",
                            "reasoning_content": "Need to compute 6 * 7",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

        assert result["type"] == "text"
        assert result["content"] == "42"
        assert result["reasoning_content"] == "Need to compute 6 * 7"
        assert result["reasoning"] == "Need to compute 6 * 7"

    def test_empty_content_with_reasoning_falls_back_to_reasoning(self) -> None:
        """Reproduces the bug where ``max_tokens`` truncates a reasoning
        model: ``content=""`` but ``reasoning_content`` is populated and
        ``finish_reason="length"``. The adapter must NOT raise; it should
        treat the reasoning text as the response content.
        """
        llm = self._make_llm()
        result = llm._process_chat_response(
            {
                "id": "chat-1",
                "object": "chat.completion",
                "model": "qwen3-thinking",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Here",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 1},
            }
        )

        assert result["type"] == "text"
        assert result["content"] == "Here"
        assert result["reasoning_content"] == "Here"
        assert result["reasoning"] == "Here"

    def test_tool_call_response_attaches_reasoning_when_present(self) -> None:
        llm = self._make_llm()
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "do_thing", "arguments": "{}"},
            }
        ]
        result = llm._process_chat_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": tool_calls,
                            "reasoning_content": "I should call do_thing",
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

        assert result["type"] == "tool_call"
        assert result["tool_calls"] == tool_calls
        assert result["reasoning_content"] == "I should call do_thing"

    def test_empty_response_without_reasoning_still_raises(self) -> None:
        """If the response has neither ``content``, ``tool_calls`` nor
        ``reasoning_content``, the adapter must keep raising so callers
        can surface the underlying provider issue.
        """
        llm = self._make_llm()

        with pytest.raises(RuntimeError, match="Invalid Xinference response"):
            llm._process_chat_response(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

    def test_finish_reason_stop_with_only_reasoning_still_raises(self) -> None:
        """The reasoning-content fallback is scoped to truncated responses
        (``finish_reason="length"``) only.

        If a provider returns ``finish_reason="stop"`` with empty content
        and a populated reasoning trace, the model is claiming to be done
        without producing a final answer -- that is a real model failure
        and the adapter must surface it instead of silently promoting the
        scratchpad to the assistant message.
        """
        llm = self._make_llm()

        with pytest.raises(RuntimeError, match="Invalid Xinference response"):
            llm._process_chat_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "I should answer the user.",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

    def test_whitespace_only_reasoning_with_length_finish_still_raises(
        self,
    ) -> None:
        """A whitespace-only reasoning trace must NOT be treated as a
        usable answer even when ``finish_reason="length"``.

        Otherwise a provider returning ``reasoning_content="   "`` would
        surface a blank string as the assistant message.
        """
        llm = self._make_llm()

        with pytest.raises(RuntimeError, match="Invalid Xinference response"):
            llm._process_chat_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "   \n  ",
                            },
                            "finish_reason": "length",
                        }
                    ]
                }
            )


class TestXinferencePromptCacheUsage:
    """Dict-shaped usage payloads surface cached input tokens."""

    def test_process_chat_response_records_cached_tokens(self) -> None:
        from xagent.core.model.chat.token_context import TokenContextManager

        llm = XinferenceLLM(model_name="qwen3-thinking")

        with TokenContextManager() as manager:
            llm._process_chat_response(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 60},
                    },
                }
            )
            usage = manager.get_usage()

        assert usage.input_tokens == 100
        inp = next(d for d in usage.details if d["type"] == "input")
        assert inp["cached_tokens"] == 60
