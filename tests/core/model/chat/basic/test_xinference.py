from unittest.mock import MagicMock, patch

import pytest

from xagent.core.model.chat.basic.xinference import XinferenceLLM
from xagent.core.model.chat.types import ChunkType


class TestXinferenceLLM:
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
    @patch("xagent.core.model.chat.basic.xinference.XinferenceClient")
    async def test_list_available_models_handles_dict_response(
        self, mock_client_class: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
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
        mock_client_class.return_value = mock_client

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

    @pytest.mark.asyncio
    @patch("xagent.core.model.chat.basic.xinference.XinferenceClient")
    async def test_list_available_models_preserves_embedding_ability(
        self, mock_client_class: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
            "embedding-uid": {
                "model_name": "Qwen3-Embedding-8B",
                "model_type": "embedding",
                "model_ability": ["embedding"],
                "model_description": "Embedding model",
            }
        }
        mock_client_class.return_value = mock_client

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
    @patch("xagent.core.model.chat.basic.xinference.XinferenceClient")
    async def test_list_available_models_handles_legacy_list_response(
        self, mock_client_class: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.list_models.return_value = [
            {
                "id": "legacy-chat-uid",
                "model_name": "legacy-chat",
                "model_type": "LLM",
                "model_ability": ["chat"],
                "model_description": "Legacy chat model",
            }
        ]
        mock_client_class.return_value = mock_client

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
