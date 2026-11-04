"""Unit tests for ElevenLabs ASR model."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.core.model.asr import ASRResult, ElevenLabsASR
from xagent.core.model.asr.adapter import get_asr_model, get_asr_model_instance
from xagent.core.model.asr.elevenlabs import (
    ELEVENLABS_DEFAULT_ASR_MODEL,
    ELEVENLABS_DEFAULT_BASE_URL,
)


def test_init_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    asr = ElevenLabsASR()

    assert asr.model == ELEVENLABS_DEFAULT_ASR_MODEL
    assert asr.model_name == ELEVENLABS_DEFAULT_ASR_MODEL
    assert asr.base_url == ELEVENLABS_DEFAULT_BASE_URL
    assert asr.api_key is None
    assert asr.language is None


async def test_transcribe_uses_sdk_convert_and_parses_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    convert_calls: list[dict[str, object]] = []

    async def convert(**kwargs: object) -> object:
        convert_calls.append(kwargs)
        return SimpleNamespace(
            text="Hello world",
            language_code="eng",
            words=[
                SimpleNamespace(
                    text="Hello",
                    start=0.0,
                    end=0.4,
                    speaker_id="speaker_0",
                    logprob=-0.1,
                ),
                SimpleNamespace(text=" ", start=None, end=None, speaker_id=None),
                SimpleNamespace(
                    text="world",
                    start=0.5,
                    end=0.9,
                    speaker_id="speaker_0",
                    logprob=-0.2,
                ),
            ],
        )

    fake_client = SimpleNamespace(speech_to_text=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsASR,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    asr = ElevenLabsASR(api_key="test-key", model="scribe_v2")
    result = await asr.transcribe(
        b"fake audio",
        format="mp3",
        language="eng",
        verbose=True,
        diarize=True,
        hotword=["Xagent"],
        tag_audio_events=False,
    )

    assert isinstance(result, ASRResult)
    assert result.text == "Hello world"
    assert result.language == "eng"
    assert result.segments is not None
    assert [segment.text for segment in result.segments] == ["Hello", "world"]
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 0.4
    assert result.segments[0].speaker == "speaker_0"
    assert result.segments[0].confidence == pytest.approx(0.904837418)
    assert result.segments[1].confidence == pytest.approx(0.818730753)
    assert len(convert_calls) == 1
    call = convert_calls[0]
    assert call["model_id"] == "scribe_v2"
    assert call["language_code"] == "eng"
    assert call["timestamps_granularity"] == "word"
    assert call["diarize"] is True
    assert call["keyterms"] == ["Xagent"]
    assert call["tag_audio_events"] is False
    assert call["file"][0] == "audio.mp3"
    assert call["file"][1] == b"fake audio"
    assert call["file"][2] == "audio/mpeg"


async def test_transcribe_source_url_does_not_send_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    convert = AsyncMock(return_value=SimpleNamespace(text="remote audio", words=[]))
    fake_client = SimpleNamespace(speech_to_text=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsASR,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    asr = ElevenLabsASR(api_key="test-key")
    result = await asr.transcribe("https://example.com/audio.mp3")

    assert result == "remote audio"
    convert.assert_awaited_once()
    call = convert.call_args.kwargs
    assert call["source_url"] == "https://example.com/audio.mp3"
    assert "file" not in call


async def test_transcribe_source_url_kwarg_does_not_send_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav-data")
    convert = AsyncMock(return_value=SimpleNamespace(text="remote audio", words=[]))
    fake_client = SimpleNamespace(speech_to_text=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsASR,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    asr = ElevenLabsASR(api_key="test-key")
    monkeypatch.setattr(
        asr,
        "_build_file_payload",
        lambda audio, format: pytest.fail("source_url should skip file upload"),
    )

    result = await asr.transcribe(
        str(audio_path), source_url="https://example.com/uploaded.wav"
    )

    assert result == "remote audio"
    convert.assert_awaited_once()
    call = convert.call_args.kwargs
    assert call["source_url"] == "https://example.com/uploaded.wav"
    assert "file" not in call


async def test_transcribe_local_path_opens_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav-data")
    observed_file_state: dict[str, object] = {}

    async def convert(**kwargs: object) -> object:
        file_payload = kwargs["file"]
        observed_file_state["filename"] = file_payload[0]
        observed_file_state["content_type"] = file_payload[2]
        observed_file_state["data"] = file_payload[1].read()
        observed_file_state["closed_during_call"] = file_payload[1].closed
        return SimpleNamespace(text="local audio", words=[])

    fake_client = SimpleNamespace(speech_to_text=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsASR,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    asr = ElevenLabsASR(api_key="test-key")
    result = await asr.transcribe(str(audio_path), verbose=False)

    assert result == "local audio"
    assert observed_file_state == {
        "filename": "sample.wav",
        "content_type": "audio/x-wav",
        "data": b"wav-data",
        "closed_during_call": False,
    }


async def test_transcribe_parses_multichannel_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert(**kwargs: object) -> object:
        return SimpleNamespace(
            transcripts=[
                SimpleNamespace(
                    text="agent hello",
                    language_code="en",
                    words=[
                        SimpleNamespace(
                            text="agent",
                            start=0.0,
                            end=0.2,
                            speaker_id="agent",
                        )
                    ],
                ),
                SimpleNamespace(
                    text="customer hi",
                    language_code="en",
                    words=[
                        SimpleNamespace(
                            text="customer",
                            start=0.3,
                            end=0.6,
                            speaker_id="customer",
                        )
                    ],
                ),
            ]
        )

    fake_client = SimpleNamespace(speech_to_text=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsASR,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    asr = ElevenLabsASR(api_key="test-key")
    result = await asr.transcribe(b"fake audio", verbose=True, use_multi_channel=True)

    assert isinstance(result, ASRResult)
    assert result.text == "agent hello\ncustomer hi"
    assert result.language == "en"
    assert result.segments is not None
    assert [segment.speaker for segment in result.segments] == ["agent", "customer"]


async def test_transcribe_rejects_unsupported_provider_options() -> None:
    asr = ElevenLabsASR(api_key="test-key")

    with pytest.raises(ValueError) as exc_info:
        await asr.transcribe(b"fake audio", reference_audio="ref.wav")

    assert "Unsupported ElevenLabs STT provider_options keys: reference_audio" in str(
        exc_info.value
    )


async def test_transcribe_redacts_sensitive_sdk_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert(**kwargs: object) -> object:
        raise RuntimeError(
            "request failed api_key=sk-elevenlabs-secret123 "
            "Authorization: Bearer bearer-secret456"
        )

    fake_client = SimpleNamespace(speech_to_text=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsASR,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    asr = ElevenLabsASR(api_key="test-key")
    with pytest.raises(RuntimeError) as exc_info:
        await asr.transcribe(b"fake audio")

    message = str(exc_info.value)
    assert "sk-elevenlabs-secret123" not in message
    assert "bearer-secret456" not in message
    assert "api_key=***t123" in message
    assert "Authorization: Bearer ***t456" in message


async def test_transcribe_closes_local_file_on_sdk_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav-data")
    file_handle = audio_path.open("rb")

    async def convert(**kwargs: object) -> object:
        raise RuntimeError("request failed")

    fake_client = SimpleNamespace(speech_to_text=SimpleNamespace(convert=convert))
    monkeypatch.setattr(
        ElevenLabsASR,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    asr = ElevenLabsASR(api_key="test-key")
    monkeypatch.setattr(
        asr,
        "_build_file_payload",
        lambda audio, format: ("sample.wav", file_handle, "audio/x-wav"),
    )

    with pytest.raises(RuntimeError, match="ElevenLabs ASR failed"):
        await asr.transcribe(str(audio_path))

    assert file_handle.closed is True


async def test_aclose_closes_cached_clients() -> None:
    class AsyncClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    async_client = AsyncClient()
    asr = ElevenLabsASR(api_key="test-key")
    asr._async_client = async_client

    await asr.aclose()

    assert async_client.closed is True
    assert asr._async_client is None


async def test_transcribe_requires_api_key_before_opening_local_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav-data")
    asr = ElevenLabsASR(api_key=None)
    monkeypatch.setattr(
        asr,
        "_build_file_payload",
        lambda audio, format: pytest.fail("missing API key should skip file open"),
    )

    with pytest.raises(RuntimeError, match="API key is required"):
        await asr.transcribe(str(audio_path))


def test_adapter_routes_elevenlabs() -> None:
    asr = get_asr_model(provider="elevenlabs", model="scribe_v2")

    assert isinstance(asr, ElevenLabsASR)
    assert asr.model == "scribe_v2"


def test_adapter_rejects_none_provider() -> None:
    with pytest.raises(ValueError, match="ASR provider cannot be None"):
        get_asr_model(provider=None)


def test_get_asr_model_instance_routes_elevenlabs() -> None:
    db_model = SimpleNamespace(
        model_provider="elevenlabs",
        model_name="scribe_v2",
        api_key="test-key",
        base_url=None,
    )

    asr = get_asr_model_instance(db_model)

    assert isinstance(asr, ElevenLabsASR)
    assert asr.model == "scribe_v2"
    assert asr.api_key == "test-key"


async def test_async_list_available_models_returns_scribe_models() -> None:
    models = await ElevenLabsASR.async_list_available_models()

    assert models[0]["id"] == "scribe_v2"
    assert models[0]["abilities"] == ["asr"]
    assert models[1]["id"] == "scribe_v1"


def test_elevenlabs_advertises_asr_capabilities() -> None:
    asr = ElevenLabsASR(api_key="test-key")

    assert asr.provider_name == "elevenlabs"
    assert asr.supports_speaker_diarization is True
    assert asr.supports_timestamps is True
    assert asr.abilities == ["asr", "timestamps", "speaker_diarization"]
