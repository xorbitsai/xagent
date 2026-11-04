"""Audio providers meter at the provider boundary: one row per billed call.

Covers every ElevenLabs and Xinference audio provider on the success path and
on the failure paths that matter for billing: a call the provider accepted and
charged must leave exactly one row however it ends afterwards, and a call the
provider rejected must leave none.
"""

from __future__ import annotations

import asyncio
import io
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from xagent.core.model import media_metering
from xagent.core.model.asr.adapter import get_asr_model_instance
from xagent.core.model.asr.elevenlabs import ElevenLabsASR
from xagent.core.model.asr.xinference import XinferenceASR
from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.music.elevenlabs import ElevenLabsMusicModel
from xagent.core.model.sound_effect.elevenlabs import ElevenLabsSoundEffectModel
from xagent.core.model.tts.adapter import get_tts_model_instance
from xagent.core.model.tts.elevenlabs import ElevenLabsTTS
from xagent.core.model.tts.xinference import XinferenceTTS

_MP3_FRAME = bytes([0xFF, 0xFB, 0x90, 0x44]) + bytes(413)  # 1152 samples @ 44.1k
_MP3_FRAME_SECONDS = 1152 / 44100


def _wav(seconds: float, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buffer.getvalue()


def _rows(manager: TokenContextManager) -> list[dict]:
    return [d for d in manager.get_usage().details if d.get("type") == "media"]


class _RawStream:
    def __init__(self, chunks, headers, fail_after=None):
        async def iterate():
            for index, chunk in enumerate(chunks):
                if fail_after is not None and index >= fail_after:
                    raise ConnectionError("stream reset")
                yield chunk

        self.data = iterate()
        self.headers = headers


class _RawContext:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self.response

    async def __aexit__(self, *exc):
        return False


def _streaming_section(
    method: str, calls: list, *, chunks=(), headers=None, error=None, fail_after=None
):
    """An SDK section whose raw client streams ``chunks`` (or rejects)."""

    def raw_call(**kwargs):
        calls.append(kwargs)
        if error is not None:
            return _RawContext(error=error)
        return _RawContext(_RawStream(list(chunks), headers or {}, fail_after))

    return SimpleNamespace(
        **{"with_raw_response": SimpleNamespace(**{method: raw_call})}
    )


def _install_client(monkeypatch, cls, client):
    monkeypatch.setattr(
        cls, "_create_async_client", staticmethod(lambda *a, **k: client)
    )


# --- ElevenLabs TTS -------------------------------------------------------------


async def test_elevenlabs_tts_bills_the_aliased_text_and_reported_cost(monkeypatch):
    calls: list = []
    section = _streaming_section(
        "convert", calls, chunks=[b"aud", b"io"], headers={"character-cost": "9"}
    )
    _install_client(monkeypatch, ElevenLabsTTS, SimpleNamespace(text_to_speech=section))
    tts = ElevenLabsTTS(api_key="k", model="eleven_v3")
    tts.model_id = "platform/3"

    with TokenContextManager() as manager:
        audio = await tts.synthesize(
            "Hi XAI", voice="v1", pronunciation_aliases={"XAI": "ex ay eye"}
        )

    assert audio == b"audio"
    [row] = _rows(manager)
    sent = calls[0]["text"]
    assert sent == "Hi ex ay eye"
    assert (row["unit"], row["quantity"]) == ("characters", float(len(sent)))
    assert row["provider_billed_units"] == 9
    assert (row["model_id"], row["model"]) == ("platform/3", "eleven_v3")


async def test_elevenlabs_tts_stream_broken_after_acceptance_still_bills(monkeypatch):
    section = _streaming_section("convert", [], chunks=[b"a", b"b"], fail_after=1)
    _install_client(monkeypatch, ElevenLabsTTS, SimpleNamespace(text_to_speech=section))

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError, match="ElevenLabs TTS failed"):
            await ElevenLabsTTS(api_key="k").synthesize("hello", voice="v1")

    assert [row["quantity"] for row in _rows(manager)] == [5.0]


async def test_elevenlabs_tts_empty_audio_still_bills(monkeypatch):
    section = _streaming_section("convert", [], chunks=[])
    _install_client(monkeypatch, ElevenLabsTTS, SimpleNamespace(text_to_speech=section))

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError, match="no audio"):
            await ElevenLabsTTS(api_key="k").synthesize("hello", voice="v1")

    assert len(_rows(manager)) == 1


async def test_elevenlabs_tts_rejected_request_records_nothing(monkeypatch):
    section = _streaming_section("convert", [], error=RuntimeError("401 unauthorized"))
    _install_client(monkeypatch, ElevenLabsTTS, SimpleNamespace(text_to_speech=section))

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await ElevenLabsTTS(api_key="k").synthesize("hello", voice="v1")

    assert _rows(manager) == []


# --- Xinference TTS -------------------------------------------------------------


def _xinference_tts(handle) -> XinferenceTTS:
    tts = XinferenceTTS(model="cosyvoice", base_url="http://x.test")
    tts.model_id = "platform/4"

    async def ensure():
        return handle

    tts._ensure_model_handle = ensure  # type: ignore[method-assign]
    return tts


async def test_xinference_tts_bills_input_characters():
    async def speech(**kwargs):
        return b"audio"

    with TokenContextManager() as manager:
        await _xinference_tts(SimpleNamespace(speech=speech)).synthesize("hello")

    [row] = _rows(manager)
    assert (row["quantity"], row["model"], row["model_id"]) == (
        5.0,
        "cosyvoice",
        "platform/4",
    )


async def test_xinference_tts_unusable_audio_still_bills():
    async def speech(**kwargs):
        return iter([b"chunk"])  # a stream where bytes were expected

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError, match="Unexpected audio"):
            await _xinference_tts(SimpleNamespace(speech=speech)).synthesize("hello")

    assert len(_rows(manager)) == 1


async def test_xinference_tts_failed_request_records_nothing():
    async def speech(**kwargs):
        raise ConnectionError("refused")

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await _xinference_tts(SimpleNamespace(speech=speech)).synthesize("hello")

    assert _rows(manager) == []


# --- ElevenLabs ASR -------------------------------------------------------------


def _asr_client(convert) -> SimpleNamespace:
    return SimpleNamespace(
        speech_to_text=SimpleNamespace(
            with_raw_response=SimpleNamespace(convert=convert)
        )
    )


@pytest.mark.parametrize("verbose", [False, True])
async def test_elevenlabs_asr_bills_reported_audio_seconds(monkeypatch, verbose):
    async def convert(**kwargs):
        data = SimpleNamespace(text="hi", words=[], audio_duration_secs=12.5)
        return SimpleNamespace(data=data, headers={"character-cost": "21"})

    _install_client(monkeypatch, ElevenLabsASR, _asr_client(convert))
    asr = ElevenLabsASR(api_key="k", model="scribe_v1")
    asr.model_id = "platform/5"

    with TokenContextManager() as manager:
        await asr.transcribe(_wav(1.0), format="wav", verbose=verbose)

    [row] = _rows(manager)
    assert (row["unit"], row["quantity"]) == ("seconds", 12.5)
    assert row["provider_billed_units"] == 21
    assert (row["model_id"], row["model"]) == ("platform/5", "scribe_v1")


async def test_elevenlabs_asr_measures_input_when_no_duration_reported(monkeypatch):
    async def convert(**kwargs):
        return SimpleNamespace(data=SimpleNamespace(text="hi", words=[]), headers={})

    _install_client(monkeypatch, ElevenLabsASR, _asr_client(convert))

    with TokenContextManager() as manager:
        await ElevenLabsASR(api_key="k").transcribe(_wav(3.0), format="wav")

    assert [row["quantity"] for row in _rows(manager)] == [pytest.approx(3.0)]


async def test_elevenlabs_asr_unparsable_2xx_still_bills(monkeypatch):
    error = RuntimeError("could not parse")
    error.status_code = 200  # type: ignore[attr-defined]

    async def convert(**kwargs):
        raise error

    _install_client(monkeypatch, ElevenLabsASR, _asr_client(convert))

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError, match="ElevenLabs ASR failed"):
            await ElevenLabsASR(api_key="k").transcribe(_wav(2.0), format="wav")

    assert [row["quantity"] for row in _rows(manager)] == [pytest.approx(2.0)]


async def test_elevenlabs_asr_rejected_request_records_nothing(monkeypatch):
    error = RuntimeError("invalid key")
    error.status_code = 401  # type: ignore[attr-defined]

    async def convert(**kwargs):
        raise error

    _install_client(monkeypatch, ElevenLabsASR, _asr_client(convert))

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await ElevenLabsASR(api_key="k").transcribe(_wav(2.0), format="wav")

    assert _rows(manager) == []


async def test_elevenlabs_asr_malformed_body_is_billed_before_the_walk(monkeypatch):
    async def convert(**kwargs):
        words = [SimpleNamespace(text="x", start="not-a-number", end=None)]
        data = SimpleNamespace(text="x", words=words, audio_duration_secs=4)
        return SimpleNamespace(data=data, headers={})

    _install_client(monkeypatch, ElevenLabsASR, _asr_client(convert))
    asr = ElevenLabsASR(api_key="k")
    monkeypatch.setattr(
        asr,
        "_normalize_response",
        lambda response: (_ for _ in ()).throw(ValueError("bad")),
    )

    with TokenContextManager() as manager:
        with pytest.raises(ValueError):
            await asr.transcribe(_wav(2.0), format="wav", verbose=True)

    assert [row["quantity"] for row in _rows(manager)] == [4.0]


# --- Xinference ASR -------------------------------------------------------------


def _xinference_asr(handle) -> XinferenceASR:
    asr = XinferenceASR(model="whisper-large-v3", base_url="http://x.test")
    asr.model_id = "platform/6"

    async def ensure():
        return handle

    asr._ensure_model_handle = ensure  # type: ignore[method-assign]
    return asr


async def test_xinference_asr_bills_reported_duration():
    async def transcriptions(**kwargs):
        return {"text": "hi", "duration": 7.5, "segments": []}

    with TokenContextManager() as manager:
        await _xinference_asr(
            SimpleNamespace(transcriptions=transcriptions)
        ).transcribe(_wav(1.0), format="wav", verbose=True)

    [row] = _rows(manager)
    assert (row["quantity"], row["model"], row["model_id"]) == (
        7.5,
        "whisper-large-v3",
        "platform/6",
    )


async def test_xinference_asr_measures_input_for_plain_text_calls():
    async def transcriptions(**kwargs):
        return {"text": "hi"}

    with TokenContextManager() as manager:
        await _xinference_asr(
            SimpleNamespace(transcriptions=transcriptions)
        ).transcribe(_wav(2.0), format="wav")

    assert [row["quantity"] for row in _rows(manager)] == [pytest.approx(2.0)]


async def test_xinference_asr_failed_request_records_nothing():
    async def transcriptions(**kwargs):
        raise ConnectionError("refused")

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await _xinference_asr(
                SimpleNamespace(transcriptions=transcriptions)
            ).transcribe(_wav(1.0), format="wav")

    assert _rows(manager) == []


# --- ElevenLabs music and sound effects ----------------------------------------

_GENERATORS = [
    pytest.param(
        ElevenLabsMusicModel,
        "music",
        "compose",
        "generate_music",
        {"prompt": "lofi"},
        "music_length_seconds",
        30.0,
        id="music",
    ),
    pytest.param(
        ElevenLabsSoundEffectModel,
        "text_to_sound_effects",
        "convert",
        "generate_sound_effect",
        {"text": "door creak"},
        "duration_seconds",
        2.0,
        id="sound_effect",
    ),
]


@pytest.mark.parametrize(
    "cls,section,method,generate,args,length_arg,length", _GENERATORS
)
async def test_generation_bills_the_measured_audio_not_the_request(
    monkeypatch, cls, section, method, generate, args, length_arg, length
):
    # The caller let the model choose the length, so nothing was requested;
    # the returned audio (40 MP3 frames) is what gets measured and billed.
    stream = _streaming_section(
        method,
        [],
        chunks=[_MP3_FRAME * 20, _MP3_FRAME * 20],
        headers={"character-cost": "800"},
    )
    _install_client(monkeypatch, cls, SimpleNamespace(**{section: stream}))
    model = cls(api_key="k")
    model.model_id = "platform/8"

    with TokenContextManager() as manager:
        await getattr(model, generate)(**args)

    [row] = _rows(manager)
    assert row["unit"] == "seconds"
    assert row["quantity"] == pytest.approx(40 * _MP3_FRAME_SECONDS)
    assert row["provider_billed_units"] == 800
    assert row["model_id"] == "platform/8"


@pytest.mark.parametrize(
    "cls,section,method,generate,args,length_arg,length", _GENERATORS
)
async def test_broken_stream_bills_the_requested_length(
    monkeypatch, cls, section, method, generate, args, length_arg, length
):
    stream = _streaming_section(
        method, [], chunks=[_MP3_FRAME, _MP3_FRAME], fail_after=1
    )
    _install_client(monkeypatch, cls, SimpleNamespace(**{section: stream}))

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await getattr(cls(api_key="k"), generate)(**args, **{length_arg: length})

    assert [row["quantity"] for row in _rows(manager)] == [length]


@pytest.mark.parametrize(
    "cls,section,method,generate,args,length_arg,length", _GENERATORS
)
async def test_empty_audio_is_billed_as_unmeasured(
    monkeypatch, cls, section, method, generate, args, length_arg, length
):
    stream = _streaming_section(method, [], chunks=[])
    _install_client(monkeypatch, cls, SimpleNamespace(**{section: stream}))

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError, match="no audio"):
            await getattr(cls(api_key="k"), generate)(**args)

    assert [row["quantity"] for row in _rows(manager)] == [0.0]


@pytest.mark.parametrize(
    "cls,section,method,generate,args,length_arg,length", _GENERATORS
)
async def test_rejected_generation_records_nothing(
    monkeypatch, cls, section, method, generate, args, length_arg, length
):
    stream = _streaming_section(method, [], error=RuntimeError("quota exceeded"))
    _install_client(monkeypatch, cls, SimpleNamespace(**{section: stream}))

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await getattr(cls(api_key="k"), generate)(**args)

    assert _rows(manager) == []


# --- identity from DB rows ----------------------------------------------------


@pytest.mark.parametrize("factory", [get_asr_model_instance, get_tts_model_instance])
@pytest.mark.parametrize("provider", ["xinference", "elevenlabs"])
def test_db_built_speech_models_carry_the_configured_id(factory, provider):
    db_model = SimpleNamespace(
        model_provider=provider,
        model_name="voice-model",
        model_id="platform/11",
        api_key="k",
        base_url="http://x.test",
    )
    assert factory(db_model).model_id == "platform/11"


# --- Ogg/Opus output and input --------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"
_OPUS_2500MS = (_FIXTURES / "tone_2500ms.opus.ogg").read_bytes()


@pytest.mark.parametrize(
    "cls,section,method,generate,args,length_arg,length", _GENERATORS
)
async def test_opus_output_without_a_length_bills_its_measured_seconds(
    monkeypatch, cls, section, method, generate, args, length_arg, length
):
    # Opus is a supported output format; its length sits in the last Ogg page,
    # which a pipe-fed ffprobe cannot reach. ffprobe is made unavailable so the
    # native measurement is what bills the call.
    monkeypatch.setattr(media_metering.shutil, "which", lambda _name: None)
    half = len(_OPUS_2500MS) // 2
    stream = _streaming_section(
        method, [], chunks=[_OPUS_2500MS[:half], _OPUS_2500MS[half:]]
    )
    _install_client(monkeypatch, cls, SimpleNamespace(**{section: stream}))

    with TokenContextManager() as manager:
        await getattr(cls(api_key="k"), generate)(**args, output_format="opus_48000_64")

    assert [row["quantity"] for row in _rows(manager)] == [pytest.approx(2.5)]


async def test_xinference_asr_measures_ogg_voice_input(monkeypatch):
    # Telegram voice notes are Ogg/Opus; a plain-text transcription reports no
    # duration, so the input itself is what gets billed.
    monkeypatch.setattr(media_metering.shutil, "which", lambda _name: None)

    async def transcriptions(**kwargs):
        return {"text": "hi"}

    with TokenContextManager() as manager:
        await _xinference_asr(
            SimpleNamespace(transcriptions=transcriptions)
        ).transcribe(_OPUS_2500MS, format="ogg")

    assert [row["quantity"] for row in _rows(manager)] == [pytest.approx(2.5)]


@pytest.mark.parametrize("reported", [float("nan"), float("inf"), -1, "bad", 0])
async def test_xinference_asr_unusable_reported_duration_measures_input(reported):
    async def transcriptions(**kwargs):
        return {"text": "hi", "duration": reported}

    with TokenContextManager() as manager:
        await _xinference_asr(
            SimpleNamespace(transcriptions=transcriptions)
        ).transcribe(_wav(2.0), format="wav", verbose=True)

    assert [row["quantity"] for row in _rows(manager)] == [pytest.approx(2.0)]


# --- cancellation after the provider accepted the call --------------------------


def _block_measurement(monkeypatch, module) -> asyncio.Event:
    """Make the module's measurement hang until cancelled; return 'entered'."""
    entered = asyncio.Event()

    async def hang(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(module, "measure_audio_seconds_async", hang)
    return entered


async def _cancel_once_entered(coro, entered: asyncio.Event) -> None:
    task = asyncio.create_task(coro)
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "cls,section,method,generate,args,length_arg,length", _GENERATORS
)
async def test_generation_cancelled_while_measuring_bills_once(
    monkeypatch, cls, section, method, generate, args, length_arg, length
):
    calls: list = []
    stream = _streaming_section(method, calls, chunks=[_MP3_FRAME])
    _install_client(monkeypatch, cls, SimpleNamespace(**{section: stream}))
    entered = _block_measurement(monkeypatch, sys.modules[cls.__module__])

    with TokenContextManager() as manager:
        await _cancel_once_entered(
            getattr(cls(api_key="k"), generate)(**args, **{length_arg: length}),
            entered,
        )

    assert len(calls) == 1
    assert [row["quantity"] for row in _rows(manager)] == [length]


async def test_elevenlabs_asr_cancelled_while_measuring_bills_once(monkeypatch):
    calls: list = []

    async def convert(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=SimpleNamespace(text="hi", words=[]), headers={})

    _install_client(monkeypatch, ElevenLabsASR, _asr_client(convert))
    entered = _block_measurement(monkeypatch, sys.modules[ElevenLabsASR.__module__])

    with TokenContextManager() as manager:
        await _cancel_once_entered(
            ElevenLabsASR(api_key="k").transcribe(_wav(1.0), format="wav"), entered
        )

    assert len(calls) == 1
    assert [row["quantity"] for row in _rows(manager)] == [0.0]


async def test_xinference_asr_cancelled_while_measuring_bills_once(monkeypatch):
    calls: list = []

    async def transcriptions(**kwargs):
        calls.append(kwargs)
        return {"text": "hi"}

    entered = _block_measurement(monkeypatch, sys.modules[XinferenceASR.__module__])

    with TokenContextManager() as manager:
        await _cancel_once_entered(
            _xinference_asr(SimpleNamespace(transcriptions=transcriptions)).transcribe(
                _wav(1.0), format="wav"
            ),
            entered,
        )

    assert len(calls) == 1
    assert [row["quantity"] for row in _rows(manager)] == [0.0]
