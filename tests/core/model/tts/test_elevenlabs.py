"""Unit tests for ElevenLabs TTS model."""

from __future__ import annotations

import asyncio
from pathlib import Path
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


async def test_synthesize_supports_direct_async_generator_sdk_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert(**kwargs: object):
        yield b"direct "
        yield b"stream"

    fake_client = SimpleNamespace(text_to_speech=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key")

    result = await tts.synthesize("Hello")

    assert result == b"direct stream"


async def test_synthesize_turns_invalid_voice_id_into_retry_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert(**kwargs: object):
        raise RuntimeError(
            "headers: {'x-trace-id': 'trace'}, status_code: 400, "
            "body: {'detail': {'status': 'invalid_uid', "
            "'message': 'An invalid ID has been received: en-male'}}"
        )

    fake_client = SimpleNamespace(text_to_speech=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key")

    with pytest.raises(RuntimeError) as exc_info:
        await tts.synthesize("Hello", voice="en-male")

    error = str(exc_info.value)
    assert "call list_tts_voices" in error
    assert "exact returned voice_id" in error
    assert "omit voice" in error
    assert "headers" not in error


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


async def test_synthesize_clones_reference_audio_and_reuses_voice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"reference-audio")
    original_read_bytes = Path.read_bytes
    read_paths: list[Path] = []

    def read_bytes(path: Path) -> bytes:
        read_paths.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    create = AsyncMock(return_value=SimpleNamespace(voice_id="cloned-voice"))

    async def convert(**kwargs: object):
        async def gen():
            yield b"cloned-audio"

        return gen()

    fake_client = SimpleNamespace(
        voices=SimpleNamespace(
            ivc=SimpleNamespace(create=create),
            delete=AsyncMock(),
        ),
        text_to_speech=SimpleNamespace(convert=convert),
    )
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key")
    first = await tts.synthesize("Hello", reference_audio=str(reference_audio))
    second = await tts.synthesize("Again", reference_audio=str(reference_audio))

    assert first == b"cloned-audio"
    assert second == b"cloned-audio"
    create.assert_awaited_once()
    call = create.await_args.kwargs
    assert call["name"].startswith("xagent-reference-")
    assert call["files"] == [("reference.wav", b"reference-audio", "audio/x-wav")]
    assert read_paths == [reference_audio]


async def test_synthesize_rejects_missing_reference_audio() -> None:
    tts = ElevenLabsTTS(api_key="test-key")

    with pytest.raises(RuntimeError, match="reference audio file does not exist"):
        await tts.synthesize("Hello", reference_audio="missing.wav")


async def test_aclose_deletes_temporary_voice_clones() -> None:
    delete = AsyncMock()
    async_client = SimpleNamespace(voices=SimpleNamespace(delete=delete))
    tts = ElevenLabsTTS(api_key="test-key")
    tts._async_client = async_client
    tts._cloned_voice_ids = {
        "first": "voice-1",
        "same-voice": "voice-1",
        "second": "voice-2",
    }

    await tts.aclose()

    assert delete.await_count == 2
    assert {call.args[0] for call in delete.await_args_list} == {
        "voice-1",
        "voice-2",
    }
    assert tts._cloned_voice_ids == {}


async def test_aclose_waits_for_inflight_temporary_voice_clone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"reference-audio")
    create_started = asyncio.Event()
    allow_create = asyncio.Event()

    async def create(**kwargs: object) -> SimpleNamespace:
        create_started.set()
        await allow_create.wait()
        return SimpleNamespace(voice_id="inflight-voice")

    delete = AsyncMock()
    fake_client = SimpleNamespace(
        voices=SimpleNamespace(
            ivc=SimpleNamespace(create=create),
            delete=delete,
        )
    )
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key")
    clone_task = asyncio.create_task(
        tts._get_or_create_cloned_voice(str(reference_audio))
    )
    await create_started.wait()

    close_task = asyncio.create_task(tts.aclose())
    await asyncio.sleep(0)
    assert close_task.done() is False

    allow_create.set()
    assert await clone_task == "inflight-voice"
    await close_task

    delete.assert_awaited_once_with("inflight-voice")
    assert tts._cloned_voice_ids == {}


async def test_aclose_retries_failed_temporary_voice_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_delete = AsyncMock(side_effect=RuntimeError("temporary network error"))
    first_client = SimpleNamespace(voices=SimpleNamespace(delete=first_delete))
    retry_delete = AsyncMock()
    retry_client = SimpleNamespace(voices=SimpleNamespace(delete=retry_delete))
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: retry_client),
    )

    tts = ElevenLabsTTS(api_key="test-key")
    tts._async_client = first_client
    tts._cloned_voice_ids = {"reference": "voice-to-retry"}

    await tts.aclose()
    assert tts._cloned_voice_ids == {"reference": "voice-to-retry"}

    await tts.aclose()
    retry_delete.assert_awaited_once_with("voice-to-retry")
    assert tts._cloned_voice_ids == {}


async def test_clone_voice_creates_persistent_voice_without_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_audio = tmp_path / "first.mp3"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"first-audio")
    second_audio.write_bytes(b"second-audio")
    create = AsyncMock(
        return_value=SimpleNamespace(
            voice_id="persistent-voice",
            requires_verification=False,
        )
    )
    delete = AsyncMock()
    fake_client = SimpleNamespace(
        voices=SimpleNamespace(
            ivc=SimpleNamespace(create=create),
            delete=delete,
        )
    )
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key")
    result = await tts.clone_voice(
        name="Product narrator",
        reference_audio_files=[str(first_audio), str(second_audio)],
        description="Consistent narration voice",
        labels={"language": "en", "accent": "american"},
        remove_background_noise=True,
    )
    await tts.aclose()

    assert result == {
        "voice_id": "persistent-voice",
        "name": "Product narrator",
        "provider": "elevenlabs",
        "persistent": True,
        "requires_verification": False,
    }
    assert create.await_args.kwargs == {
        "name": "Product narrator",
        "files": [
            ("first.mp3", b"first-audio", "audio/mpeg"),
            ("second.wav", b"second-audio", "audio/x-wav"),
        ],
        "description": "Consistent narration voice",
        "labels": '{"language": "en", "accent": "american"}',
        "remove_background_noise": True,
    }
    delete.assert_not_awaited()


async def test_delete_voice_deletes_persistent_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delete = AsyncMock()
    fake_client = SimpleNamespace(voices=SimpleNamespace(delete=delete))
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key")
    await tts.delete_voice(" persistent-voice ")

    delete.assert_awaited_once_with("persistent-voice")


async def test_delete_voice_rejects_empty_voice_id() -> None:
    tts = ElevenLabsTTS(api_key="test-key")

    with pytest.raises(ValueError, match="voice ID must not be empty"):
        await tts.delete_voice("  ")


async def test_delete_voice_redacts_sensitive_sdk_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delete = AsyncMock(
        side_effect=RuntimeError(
            "request failed api_key=sk-elevenlabs-secret123 "
            "Authorization: Bearer bearer-secret456"
        )
    )
    fake_client = SimpleNamespace(voices=SimpleNamespace(delete=delete))
    monkeypatch.setattr(
        ElevenLabsTTS,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    tts = ElevenLabsTTS(api_key="test-key")
    with pytest.raises(RuntimeError) as exc_info:
        await tts.delete_voice("persistent-voice")

    message = str(exc_info.value)
    assert "sk-elevenlabs-secret123" not in message
    assert "bearer-secret456" not in message
    assert "api_key=***t123" in message
    assert "Authorization: Bearer ***t456" in message


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


@pytest.mark.parametrize("category", ["cloned", "professional", "generated", "premade"])
def test_normalize_voice_response_preserves_category(category: str) -> None:
    voices = ElevenLabsTTS._normalize_voice_response(
        {"voices": [{"voice_id": "voice-123", "category": category}]}
    )

    assert voices[0]["category"] == category


def test_elevenlabs_advertises_voice_capabilities() -> None:
    tts = ElevenLabsTTS(api_key="test-key")

    assert tts.provider_name == "elevenlabs"
    assert tts.supports_voice_cloning is True
    assert tts.supports_persistent_voice_cloning is True
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
