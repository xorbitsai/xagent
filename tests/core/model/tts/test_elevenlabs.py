"""Unit tests for ElevenLabs TTS model."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.core.model.tts import ElevenLabsTTS, TTSResult
from xagent.core.model.tts.adapter import get_tts_model, get_tts_model_instance
from xagent.core.model.tts.elevenlabs import (
    ELEVENLABS_DEFAULT_BASE_URL,
    ELEVENLABS_DEFAULT_OUTPUT_FORMAT,
    ELEVENLABS_DEFAULT_VOICE_ID,
)


def test_init_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    tts = ElevenLabsTTS()

    assert tts.model == "eleven_v3"
    assert tts.model_name == "eleven_v3"
    assert tts.base_url == ELEVENLABS_DEFAULT_BASE_URL
    assert tts.api_key is None
    assert tts.voice == ELEVENLABS_DEFAULT_VOICE_ID
    assert tts.output_format == ELEVENLABS_DEFAULT_OUTPUT_FORMAT
    assert tts.sample_rate == 44100


async def test_synthesize_uses_sdk_convert_and_joins_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    convert_calls: list[dict[str, object]] = []

    async def convert(**kwargs: object):
        convert_calls.append(kwargs)

        async def gen():
            yield b"fake "
            yield b"audio"

        return gen()

    fake_client = SimpleNamespace(text_to_speech=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key", model="eleven_flash_v2_5")
    result = await tts.synthesize(
        "Hello",
        voice="voice-123",
        language="eng",
        verbose=True,
        voice_settings={"stability": 0.5},
    )

    assert isinstance(result, TTSResult)
    assert result.audio == b"fake audio"
    assert result.format == "mp3"
    assert result.sample_rate == 44100
    assert result.language == "eng"
    assert len(convert_calls) == 1
    call = convert_calls[0]
    assert call["text"] == "Hello"
    assert call["voice_id"] == "voice-123"
    assert call["model_id"] == "eleven_flash_v2_5"
    assert call["output_format"] == "mp3_44100_128"
    assert call["language_code"] == "eng"
    assert call["voice_settings"].stability == 0.5


async def test_synthesize_maps_pcm_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    convert_calls: list[dict[str, object]] = []

    async def convert(**kwargs: object):
        convert_calls.append(kwargs)

        async def gen():
            yield b"pcm-audio"

        return gen()

    fake_client = SimpleNamespace(text_to_speech=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key")
    result = await tts.synthesize(
        "Hello",
        format="pcm",
        sample_rate=16000,
        verbose=True,
    )

    assert isinstance(result, TTSResult)
    assert result.audio == b"pcm-audio"
    assert result.format == "pcm"
    assert result.sample_rate == 16000
    assert result.raw_response == {
        "model": "eleven_v3",
        "voice_id": ELEVENLABS_DEFAULT_VOICE_ID,
        "output_format": "pcm_16000",
    }
    assert convert_calls[0]["output_format"] == "pcm_16000"


async def test_synthesize_maps_pcm_default_to_requested_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    convert_calls: list[dict[str, object]] = []

    async def convert(**kwargs: object):
        convert_calls.append(kwargs)

        async def gen():
            yield b"pcm-audio"

        return gen()

    fake_client = SimpleNamespace(text_to_speech=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key", format="pcm")
    result = await tts.synthesize("Hello", sample_rate=16000, verbose=True)

    assert isinstance(result, TTSResult)
    assert result.format == "pcm"
    assert result.sample_rate == 16000
    assert result.raw_response == {
        "model": "eleven_v3",
        "voice_id": ELEVENLABS_DEFAULT_VOICE_ID,
        "output_format": "pcm_16000",
    }
    assert convert_calls[0]["output_format"] == "pcm_16000"


async def test_synthesize_rejects_mismatched_sample_rate_for_fixed_format() -> None:
    tts = ElevenLabsTTS(api_key="test-key")

    with pytest.raises(ValueError) as exc_info:
        await tts.synthesize("Hello", format="wav", sample_rate=16000)

    assert "Use format='pcm' for custom sample rates" in str(exc_info.value)


async def test_synthesize_rejects_unsupported_provider_options() -> None:
    tts = ElevenLabsTTS(api_key="test-key")

    with pytest.raises(ValueError) as exc_info:
        await tts.synthesize("Hello", reference_audio="ref.wav")

    assert "Unsupported ElevenLabs provider_options keys: reference_audio" in str(
        exc_info.value
    )


async def test_synthesize_redacts_sensitive_sdk_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert(**kwargs: object):
        raise RuntimeError(
            "request failed api_key=sk-elevenlabs-secret123 "
            "Authorization: Bearer bearer-secret456"
        )

    fake_client = SimpleNamespace(text_to_speech=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key")
    with pytest.raises(RuntimeError) as exc_info:
        await tts.synthesize("Hello")

    message = str(exc_info.value)
    assert "sk-elevenlabs-secret123" not in message
    assert "bearer-secret456" not in message
    assert "api_key=***t123" in message
    assert "Authorization: Bearer ***t456" in message


async def test_aclose_closes_cached_clients() -> None:
    class SyncClient:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class AsyncClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    sync_client = SyncClient()
    async_client = AsyncClient()
    tts = ElevenLabsTTS(api_key="test-key")
    tts._client = sync_client
    tts._async_client = async_client

    await tts.aclose()

    assert sync_client.closed is True
    assert async_client.closed is True
    assert tts._client is None
    assert tts._async_client is None


def test_synthesize_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    tts = ElevenLabsTTS(api_key=None)

    with pytest.raises(RuntimeError, match="API key is required"):
        tts._ensure_client()


def test_adapter_routes_elevenlabs() -> None:
    tts = get_tts_model(provider="elevenlabs", model="eleven_flash_v2_5")

    assert isinstance(tts, ElevenLabsTTS)
    assert tts.model == "eleven_flash_v2_5"


def test_get_tts_model_instance_routes_elevenlabs() -> None:
    db_model = SimpleNamespace(
        model_provider="elevenlabs",
        model_name="eleven_v3",
        api_key="test-key",
        base_url=None,
    )

    tts = get_tts_model_instance(db_model)

    assert isinstance(tts, ElevenLabsTTS)
    assert tts.model == "eleven_v3"
    assert tts.api_key == "test-key"


def test_list_available_models_normalizes_sdk_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_models = Mock(
        return_value=SimpleNamespace(
            models=[
                SimpleNamespace(
                    model_id="eleven_v3",
                    name="Eleven v3",
                    description="Expressive TTS",
                    can_do_text_to_speech=True,
                    languages=["eng"],
                ),
                SimpleNamespace(
                    model_id="scribe_v2",
                    name="Scribe v2",
                    can_do_text_to_speech=False,
                ),
            ]
        )
    )
    close = Mock()
    fake_client = SimpleNamespace(models=SimpleNamespace(list=list_models), close=close)
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    models = ElevenLabsTTS.list_available_models(api_key="test-key")

    assert models == [
        {
            "id": "eleven_v3",
            "object": "model",
            "owned_by": "elevenlabs",
            "abilities": ["tts"],
            "name": "Eleven v3",
            "description": "Expressive TTS",
            "languages": ["eng"],
        }
    ]
    list_models.assert_called_once_with()
    close.assert_called_once_with()


async def test_async_list_available_models_uses_async_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_models = AsyncMock(
        return_value=SimpleNamespace(
            models=[
                SimpleNamespace(
                    model_id="eleven_flash_v2_5",
                    name="Eleven Flash v2.5",
                    can_do_text_to_speech=True,
                )
            ]
        )
    )
    aclose = AsyncMock()
    fake_client = SimpleNamespace(
        models=SimpleNamespace(list=list_models),
        aclose=aclose,
    )
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    models = await ElevenLabsTTS.async_list_available_models(api_key="test-key")

    assert models == [
        {
            "id": "eleven_flash_v2_5",
            "object": "model",
            "owned_by": "elevenlabs",
            "abilities": ["tts"],
            "name": "Eleven Flash v2.5",
        }
    ]
    list_models.assert_awaited_once_with()
    aclose.assert_awaited_once_with()


async def test_list_available_voices_normalizes_sdk_voices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_all = AsyncMock(
        return_value=SimpleNamespace(
            voices=[
                SimpleNamespace(
                    voice_id="voice-123",
                    name="Rachel",
                    category="premade",
                    description="Warm narration voice",
                    labels={"accent": "american"},
                    preview_url="https://example.com/preview.mp3",
                    available_for_tiers=["free", "creator"],
                    high_quality_base_model_ids=["eleven_v3"],
                    settings=SimpleNamespace(
                        stability=0.45,
                        similarity_boost=0.8,
                        style=None,
                        speed=1.0,
                        use_speaker_boost=True,
                    ),
                    verified_languages=[
                        SimpleNamespace(
                            language="en",
                            model_id="eleven_v3",
                            accent="US",
                            locale="en-US",
                            preview_url="https://example.com/en-preview.mp3",
                        )
                    ],
                )
            ]
        )
    )
    fake_client = SimpleNamespace(voices=SimpleNamespace(get_all=get_all))
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    voices = await ElevenLabsTTS(api_key="test-key").list_available_voices()

    assert voices == [
        {
            "voice_id": "voice-123",
            "provider": "elevenlabs",
            "name": "Rachel",
            "category": "premade",
            "description": "Warm narration voice",
            "preview_url": "https://example.com/preview.mp3",
            "labels": {"accent": "american"},
            "available_for_tiers": ["free", "creator"],
            "high_quality_base_model_ids": ["eleven_v3"],
            "settings": {
                "stability": 0.45,
                "similarity_boost": 0.8,
                "speed": 1.0,
                "use_speaker_boost": True,
            },
            "verified_languages": [
                {
                    "language": "en",
                    "model_id": "eleven_v3",
                    "accent": "US",
                    "locale": "en-US",
                    "preview_url": "https://example.com/en-preview.mp3",
                }
            ],
        }
    ]
    get_all.assert_awaited_once_with()


def test_elevenlabs_advertises_voice_capabilities() -> None:
    tts = ElevenLabsTTS(api_key="test-key")

    assert tts.provider_name == "elevenlabs"
    assert tts.supports_voice_listing is True
    assert tts.supports_voice_settings is True
    assert tts.supported_voice_settings == [
        "stability",
        "similarity_boost",
        "style",
        "speed",
        "use_speaker_boost",
    ]
    assert "seed" in tts.supported_provider_options


def test_list_available_models_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key is required"):
        ElevenLabsTTS.list_available_models()
