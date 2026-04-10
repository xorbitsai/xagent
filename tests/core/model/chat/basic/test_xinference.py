from unittest.mock import MagicMock, patch

import pytest

from xagent.core.model.chat.basic.xinference import XinferenceLLM


class TestXinferenceLLM:
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
