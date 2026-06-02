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
