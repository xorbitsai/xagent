"""Tests for provider model listing service."""

from __future__ import annotations

import pytest

from xagent.core.model.asr.elevenlabs import ElevenLabsASR
from xagent.core.model.music.elevenlabs import ElevenLabsMusicModel
from xagent.core.model.sound_effect.elevenlabs import ElevenLabsSoundEffectModel
from xagent.core.model.tts.elevenlabs import ElevenLabsTTS
from xagent.web.services.model_list_service import (
    PROVIDER_FETCHERS,
    fetch_dashscope_embedding_models,
    fetch_elevenlabs_models,
    fetch_elevenlabs_music_models,
    fetch_elevenlabs_sound_effect_models,
    fetch_openai_models,
)


def test_openai_compatible_is_registered_in_provider_fetchers() -> None:
    """Guards the openai-compatible provider against silently 400ing on
    "fetch models" — every provider in _SUPPORTED_PROVIDER_METADATA that can
    list models must also appear in PROVIDER_FETCHERS."""
    assert PROVIDER_FETCHERS.get("openai-compatible") is fetch_openai_models


async def test_fetch_dashscope_embedding_models_returns_curated_text_models() -> None:
    models = await fetch_dashscope_embedding_models(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    assert models == [
        {
            "id": "text-embedding-v4",
            "object": "model",
            "owned_by": "dashscope",
        },
        {
            "id": "text-embedding-v3",
            "object": "model",
            "owned_by": "dashscope",
        },
    ]


async def test_fetch_elevenlabs_models_combines_tts_and_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def list_tts_models(api_key=None, base_url=None):
        return [
            {
                "id": "eleven_v3",
                "object": "model",
                "owned_by": "elevenlabs",
                "abilities": ["tts"],
            }
        ]

    async def list_asr_models(api_key=None, base_url=None):
        return [
            {
                "id": "scribe_v2",
                "object": "model",
                "owned_by": "elevenlabs",
                "abilities": ["asr"],
            }
        ]

    monkeypatch.setattr(
        ElevenLabsTTS,
        "async_list_available_models",
        staticmethod(list_tts_models),
    )
    monkeypatch.setattr(
        ElevenLabsASR,
        "async_list_available_models",
        staticmethod(list_asr_models),
    )

    models = await fetch_elevenlabs_models(
        api_key="test-key",
        base_url="https://api.elevenlabs.io",
    )

    assert models == [
        {
            "id": "eleven_v3",
            "object": "model",
            "owned_by": "elevenlabs",
            "abilities": ["tts"],
        },
        {
            "id": "scribe_v2",
            "object": "model",
            "owned_by": "elevenlabs",
            "abilities": ["asr"],
        },
    ]


async def test_fetch_elevenlabs_music_models_is_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def list_models(api_key=None, base_url=None):
        return [
            {
                "id": "music_v2",
                "category": "music",
                "abilities": ["generate"],
            }
        ]

    monkeypatch.setattr(
        ElevenLabsMusicModel,
        "async_list_available_models",
        classmethod(
            lambda cls, api_key=None, base_url=None: list_models(api_key, base_url)
        ),
    )

    models = await fetch_elevenlabs_music_models("test-key")

    assert models == [
        {
            "id": "music_v2",
            "category": "music",
            "abilities": ["generate"],
        }
    ]


async def test_fetch_elevenlabs_sound_effect_models_is_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def list_models(api_key=None, base_url=None):
        return [
            {
                "id": "eleven_text_to_sound_v2",
                "category": "sound_effect",
                "abilities": ["generate"],
            }
        ]

    monkeypatch.setattr(
        ElevenLabsSoundEffectModel,
        "async_list_available_models",
        classmethod(
            lambda cls, api_key=None, base_url=None: list_models(api_key, base_url)
        ),
    )

    models = await fetch_elevenlabs_sound_effect_models("test-key")

    assert models == [
        {
            "id": "eleven_text_to_sound_v2",
            "category": "sound_effect",
            "abilities": ["generate"],
        }
    ]
