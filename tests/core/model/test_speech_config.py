"""Speech defaults survive storage and reach provider requests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base

from xagent.core.model.asr.adapter import get_asr_model_instance
from xagent.core.model.model import SpeechModelConfig
from xagent.core.model.storage.db.adapter import SQLAlchemyModelHub
from xagent.core.model.storage.db.db_models import create_model_table
from xagent.core.model.tts.adapter import get_tts_model_instance
from xagent.web.services.llm_utils import CoreStorage


@pytest.fixture(params=[SQLAlchemyModelHub, CoreStorage], ids=["core", "web"])
def hub(request, monkeypatch):
    monkeypatch.setenv("ENCRYPTION_KEY", "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc=")
    base = declarative_base()
    model = create_model_table(base)
    engine = create_engine("sqlite:///:memory:")
    base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            yield request.param(db, model)
    finally:
        engine.dispose()


@pytest.mark.parametrize("provider", ["xinference", "elevenlabs"])
async def test_stored_tts_defaults_reach_requests_and_can_be_overridden(
    hub, provider, monkeypatch
):
    config = SpeechModelConfig(
        id="configured-tts",
        model_name="configured-model",
        model_provider=provider,
        api_key="test-key",
        abilities=["tts"],
        language="en",
        voice="configured-voice",
        format="pcm",
        sample_rate=16000,
    )
    hub.store(config)
    hub.db.expire_all()
    assert hub.load(config.id).model_dump() == config.model_dump()
    assert hub.list()[config.id].model_dump() == config.model_dump()
    row = hub.db.query(hub.Model).filter_by(model_id=config.id).one()
    tts = get_tts_model_instance(row)
    calls = []

    if provider == "xinference":

        async def speech(**kwargs):
            calls.append(kwargs)
            return b"audio"

        monkeypatch.setattr(
            tts,
            "_ensure_model_handle",
            AsyncMock(return_value=SimpleNamespace(speech=speech)),
        )
    else:

        async def convert(**kwargs):
            calls.append(kwargs)
            yield b"audio"

        monkeypatch.setattr(
            tts,
            "_ensure_async_client",
            lambda: SimpleNamespace(text_to_speech=SimpleNamespace(convert=convert)),
        )

    assert await tts.synthesize("Hello") == b"audio"
    assert (
        await tts.synthesize(
            "Bonjour",
            language="fr",
            voice="override-voice",
            format="pcm",
            sample_rate=24000,
        )
        == b"audio"
    )
    assert tts.model_id == config.id
    if provider == "xinference":
        assert [
            (
                call["language"],
                call["voice"],
                call["output_audio_format"],
                call["sample_rate"],
            )
            for call in calls
        ] == [
            ("en", "configured-voice", "pcm", 16000),
            ("fr", "override-voice", "pcm", 24000),
        ]
    else:
        assert [
            (call["language_code"], call["voice_id"], call["output_format"])
            for call in calls
        ] == [
            ("en", "configured-voice", "pcm_16000"),
            ("fr", "override-voice", "pcm_24000"),
        ]


@pytest.mark.parametrize("provider", ["xinference", "elevenlabs"])
async def test_stored_asr_language_reaches_requests_and_can_be_overridden(
    hub, provider, monkeypatch
):
    config = SpeechModelConfig(
        id="configured-asr",
        model_name="configured-model",
        model_provider=provider,
        api_key="test-key",
        abilities=["asr"],
        language="en",
    )
    hub.store(config)
    hub.db.expire_all()
    row = hub.db.query(hub.Model).filter_by(model_id=config.id).one()
    asr = get_asr_model_instance(row)
    request = AsyncMock(
        return_value={"text": "Hello", "duration": 1, "audio_duration_secs": 1}
    )
    if provider == "xinference":
        monkeypatch.setattr(
            asr,
            "_ensure_model_handle",
            AsyncMock(return_value=SimpleNamespace(transcriptions=request)),
        )
        language_key = "language"
    else:
        monkeypatch.setattr(
            asr,
            "_ensure_async_client",
            lambda: SimpleNamespace(speech_to_text=SimpleNamespace(convert=request)),
        )
        language_key = "language_code"

    assert await asr.transcribe(b"audio", format="wav") == "Hello"
    assert await asr.transcribe(b"audio", format="wav", language="fr") == "Hello"
    assert [call.kwargs[language_key] for call in request.await_args_list] == [
        "en",
        "fr",
    ]
    assert asr.model_id == config.id


@pytest.mark.parametrize("provider", ["xinference", "elevenlabs"])
def test_unset_speech_settings_keep_provider_defaults(hub, provider):
    hub.store(
        SpeechModelConfig(
            id="legacy-speech",
            model_name="configured-model",
            model_provider=provider,
            api_key="test-key",
            abilities=["asr", "tts"],
        )
    )
    hub.db.expire_all()
    row = hub.db.query(hub.Model).filter_by(model_id="legacy-speech").one()
    asr = get_asr_model_instance(row)
    tts = get_tts_model_instance(row)
    assert asr.language is None
    assert tts.language is None
    if provider == "xinference":
        assert (tts.voice, tts.format, tts.sample_rate) == (None, "mp3", 24000)
    else:
        assert (tts.output_format, tts.sample_rate) == ("mp3_44100_128", 44100)
