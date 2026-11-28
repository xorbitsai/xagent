"""Tests for provider model listing service."""

from __future__ import annotations

import sys
from inspect import Parameter, signature
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.core.model.asr.elevenlabs import ElevenLabsASR
from xagent.core.model.music.elevenlabs import ElevenLabsMusicModel
from xagent.core.model.sound_effect.elevenlabs import ElevenLabsSoundEffectModel
from xagent.core.model.tts.elevenlabs import ElevenLabsTTS
from xagent.web.services.model_list_service import (
    PROVIDER_FETCHERS,
    fetch_bedrock_models,
    fetch_dashscope_embedding_models,
    fetch_elevenlabs_models,
    fetch_elevenlabs_music_models,
    fetch_elevenlabs_sound_effect_models,
    fetch_models_from_provider,
    fetch_openai_models,
)


def test_openai_compatible_is_registered_in_provider_fetchers() -> None:
    """Guards the openai-compatible provider against silently 400ing on
    "fetch models" — every provider in _SUPPORTED_PROVIDER_METADATA that can
    list models must also appear in PROVIDER_FETCHERS."""
    assert PROVIDER_FETCHERS.get("openai-compatible") is fetch_openai_models


async def test_bedrock_catalog_is_not_an_invocation_admission_gate() -> None:
    assert PROVIDER_FETCHERS.get("bedrock") is fetch_bedrock_models
    assert await fetch_bedrock_models("key", "https://runtime.example") == []


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


@pytest.mark.parametrize("provider", sorted(PROVIDER_FETCHERS))
def test_builtin_fetcher_accepts_strict_mode(provider: str) -> None:
    parameters = signature(PROVIDER_FETCHERS[provider]).parameters
    strict = parameters.get("raise_on_error")
    assert strict is not None
    assert strict.kind == Parameter.KEYWORD_ONLY
    assert strict.default is False


@pytest.mark.parametrize(
    "provider", ["volcengine-ark", "byteplus-ark", "ark", "ark-video"]
)
async def test_ark_catalog_distinguishes_failure_from_empty(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = TimeoutError("offline catalog timeout")
    client = AsyncMock()
    client.models.list.side_effect = failure
    monkeypatch.setattr(
        "xagent.core.model.chat.basic.openai.create_owned_async_sdk_client",
        Mock(return_value=client),
    )
    args = (provider, "test-key", "https://provider.example/v1")

    assert await fetch_models_from_provider(*args) == []
    with pytest.raises(TimeoutError) as exc_info:
        await fetch_models_from_provider(*args, raise_on_error=True)
    assert exc_info.value is failure

    client.models.list.side_effect = None
    client.models.list.return_value = SimpleNamespace(data=[])
    assert await fetch_models_from_provider(*args, raise_on_error=True) == []
    assert client.models.list.await_count == client.close.await_count == 3


async def test_xinference_rerank_catalog_distinguishes_failure_from_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("offline catalog failure")
    client = Mock()
    client.list_models.side_effect = failure
    monkeypatch.setattr(
        "xagent.core.model.public_xinference.PublicXinferenceClient",
        Mock(return_value=client),
    )
    args = ("xinference-rerank", "test-key", "https://provider.example")

    assert await fetch_models_from_provider(*args) == []
    with pytest.raises(RuntimeError) as exc_info:
        await fetch_models_from_provider(*args, raise_on_error=True)
    assert exc_info.value is failure

    client.list_models.side_effect = None
    client.list_models.return_value = {}
    assert await fetch_models_from_provider(*args, raise_on_error=True) == []
    assert client.list_models.call_count == client.close.call_count == 3


@pytest.mark.parametrize("raise_on_error", [False, True])
async def test_xinference_rerank_preserves_client_setup_errors(
    monkeypatch: pytest.MonkeyPatch, raise_on_error: bool
) -> None:
    failure = ImportError("Xinference SDK unavailable")
    monkeypatch.setattr(
        "xagent.core.model.public_xinference.PublicXinferenceClient",
        Mock(side_effect=failure),
    )
    with pytest.raises(ImportError) as exc_info:
        await fetch_models_from_provider(
            "xinference-rerank",
            "test-key",
            "https://provider.example",
            raise_on_error=raise_on_error,
        )
    assert exc_info.value is failure


@pytest.mark.parametrize("raise_on_error", [False, True])
async def test_xinference_rerank_preserves_missing_sdk_error(
    monkeypatch: pytest.MonkeyPatch, raise_on_error: bool
) -> None:
    monkeypatch.setitem(sys.modules, "xagent.core.model.public_xinference", None)
    with pytest.raises(ModuleNotFoundError, match="public_xinference"):
        await fetch_models_from_provider(
            "xinference-rerank",
            "test-key",
            "https://provider.example",
            raise_on_error=raise_on_error,
        )


async def test_curated_catalog_ignores_strict_mode() -> None:
    args = ("dashscope-embedding", "test-key", "https://provider.example")
    default_catalog = await fetch_models_from_provider(*args)
    assert default_catalog
    assert (
        await fetch_models_from_provider(*args, raise_on_error=True) == default_catalog
    )
    assert (
        await fetch_models_from_provider(
            "unregistered-provider", "test-key", raise_on_error=True
        )
        == []
    )


async def test_strict_mode_accepts_legacy_external_fetcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [{"id": "external-model"}]

    async def legacy_fetcher(api_key: str, base_url: str) -> list[dict[str, str]]:
        assert api_key == "test-key"
        assert base_url == "https://provider.example"
        return expected

    monkeypatch.setitem(PROVIDER_FETCHERS, "external-test", legacy_fetcher)
    assert (
        await fetch_models_from_provider(
            "external-test", "test-key", "https://provider.example", raise_on_error=True
        )
        == expected
    )


async def test_strict_mode_forwards_keyword_without_retrying_body_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = TypeError("fetcher failed internally")
    calls: list[bool] = []

    async def strict_fetcher(
        api_key: str, base_url: str, *, raise_on_error: bool = False
    ) -> list[dict[str, str]]:
        calls.append(raise_on_error)
        raise failure

    monkeypatch.setitem(PROVIDER_FETCHERS, "external-test", strict_fetcher)
    with pytest.raises(TypeError) as exc_info:
        await fetch_models_from_provider(
            "external-test", "test-key", "https://provider.example", raise_on_error=True
        )
    assert exc_info.value is failure
    assert calls == [True]
