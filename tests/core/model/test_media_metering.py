"""Shared provider-boundary metering helpers and the billed-units row field."""

from __future__ import annotations

import io
import struct
import wave
from types import SimpleNamespace

import pytest

from xagent.core.model import media_metering
from xagent.core.model.chat.token_context import (
    MediaCallType,
    TokenContextManager,
    TokenUsage,
    aggregate_media_usage_by_model,
)
from xagent.core.model.media_metering import (
    BilledStream,
    accepted_status,
    call_with_headers,
    elevenlabs_billed_units,
    measure_audio_seconds,
    read_billed_stream,
    record_provider_usage,
)

# One MPEG-1 Layer III frame header: 128 kbps, 44.1 kHz, no padding. Each frame
# is 417 bytes and carries 1152 samples.
_MP3_HEADER = bytes([0xFF, 0xFB, 0x90, 0x44])
_MP3_FRAME = _MP3_HEADER + bytes(417 - 4)


def _mp3(frames: int, *, id3: bool = False) -> bytes:
    data = _MP3_FRAME * frames
    if id3:
        # ID3v2 header with a syncsafe size of 20, followed by 20 tag bytes.
        data = b"ID3\x04\x00\x00\x00\x00\x00\x14" + bytes(20) + data
    return data


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


# --- identity and recording --------------------------------------------------


def test_record_uses_stamped_id_and_provider_name():
    provider = SimpleNamespace(model_name="eleven_v3", model_id="platform/42")
    with TokenContextManager() as manager:
        record_provider_usage(provider, MediaCallType.TTS, 11)

    [row] = _rows(manager)
    assert row["model_id"] == "platform/42"
    assert row["model"] == "eleven_v3"
    assert row["unit"] == "characters"
    assert row["quantity"] == 11
    assert "provider_billed_units" not in row


@pytest.mark.parametrize("placeholder", ["default", " None ", "", None, 7])
def test_placeholder_identity_is_written_empty_never_billed(placeholder):
    provider = SimpleNamespace(model_name=placeholder, model_id=placeholder)
    with TokenContextManager() as manager:
        record_provider_usage(provider, MediaCallType.ASR, 3)

    [row] = _rows(manager)
    assert row["model_id"] == ""
    assert row["model"] == ""


def test_falls_back_to_model_attribute_for_xinference_style_providers():
    provider = SimpleNamespace(model="whisper-large-v3", model_id="7")
    with TokenContextManager() as manager:
        record_provider_usage(provider, MediaCallType.ASR, 3)

    [row] = _rows(manager)
    assert (row["model"], row["model_id"]) == ("whisper-large-v3", "7")


def test_unmeasured_call_records_zero_in_the_modality_unit():
    with TokenContextManager() as manager:
        record_provider_usage(SimpleNamespace(), MediaCallType.MUSIC, None)

    [row] = _rows(manager)
    assert (row["quantity"], row["unit"]) == (0.0, "seconds")


def test_recording_never_raises_into_the_media_call(monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("accounting broke")

    monkeypatch.setattr(media_metering, "add_media_usage", boom)
    record_provider_usage(SimpleNamespace(), MediaCallType.VIDEO, 5)


class _ExplodingProvider:
    def __getattr__(self, name):
        raise RuntimeError(f"no {name}")


def test_identity_read_failures_still_write_the_row():
    with TokenContextManager() as manager:
        record_provider_usage(_ExplodingProvider(), MediaCallType.VIDEO, 5)

    [row] = _rows(manager)
    assert (row["quantity"], row["model"], row["model_id"]) == (5.0, "", "")


# --- provider_billed_units on the row ----------------------------------------


@pytest.mark.parametrize(
    ("reported", "stored"),
    [("120", 120), (0, 0), (17, 17), ("bad", None), (-3, None), (True, None)],
)
def test_billed_units_row_field(reported, stored):
    usage = TokenUsage()
    usage.record_media_call("tts", 5, provider_billed_units=reported)

    row = usage.details[-1]
    if stored is None:
        assert "provider_billed_units" not in row
    else:
        assert row["provider_billed_units"] == stored


def test_billed_units_aggregate_only_when_every_row_reports_them():
    complete = TokenUsage()
    complete.record_media_call("tts", 5, model_id="m", provider_billed_units=5)
    complete.record_media_call("tts", 7, model_id="m", provider_billed_units=0)
    [group] = aggregate_media_usage_by_model(complete.details)
    assert group["provider_billed_units"] == 5

    partial = TokenUsage()
    partial.record_media_call("tts", 5, model_id="m", provider_billed_units=5)
    partial.record_media_call("tts", 7, model_id="m")
    [group] = aggregate_media_usage_by_model(partial.details)
    assert "provider_billed_units" not in group


# --- measuring audio ----------------------------------------------------------


def test_measures_mp3_by_frames_and_skips_id3():
    assert measure_audio_seconds(_mp3(100), "mp3_44100_128") == pytest.approx(
        100 * 1152 / 44100
    )
    assert measure_audio_seconds(_mp3(10, id3=True), "auto") == pytest.approx(
        10 * 1152 / 44100
    )


def test_junk_ends_the_mp3_walk_instead_of_inflating_it():
    # Frame-shaped bytes after a non-frame run are not counted: resyncing past
    # junk would let any stray 0xFFE sync pattern add phantom frames.
    junk_then_frames = b"TAG" + bytes(125) + _mp3(5)
    assert measure_audio_seconds(_mp3(10) + junk_then_frames, "mp3") == (
        pytest.approx(10 * 1152 / 44100)
    )


def test_measures_headerless_pcm_and_ulaw_by_length():
    assert measure_audio_seconds(bytes(48000), "pcm_24000") == pytest.approx(1.0)
    assert measure_audio_seconds(bytes(16000), "ulaw_8000") == pytest.approx(2.0)


def test_measures_wav_from_its_header():
    assert measure_audio_seconds(_wav(2.5), "") == pytest.approx(2.5)


def test_unknown_container_without_ffprobe_is_unmeasured(monkeypatch):
    monkeypatch.setattr(media_metering.shutil, "which", lambda _name: None)
    assert measure_audio_seconds(b"OggS" + bytes(200), "ogg") is None
    assert measure_audio_seconds(b"", "mp3") is None
    assert measure_audio_seconds("not bytes", "mp3") is None


def test_unknown_container_uses_ffprobe_when_installed(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv[0], kwargs["input"]))
        return SimpleNamespace(stdout=b"3.25\n")

    monkeypatch.setattr(media_metering.shutil, "which", lambda _name: "/x/ffprobe")
    monkeypatch.setattr(media_metering.subprocess, "run", fake_run)
    payload = b"OggS" + bytes(10)

    assert measure_audio_seconds(payload, "opus_48000_64") == pytest.approx(3.25)
    assert calls == [("/x/ffprobe", payload)]


# --- ElevenLabs transport helpers ----------------------------------------------


class _RawStream:
    def __init__(self, chunks, headers, fail_after=None):
        self.data = self._iterate(chunks, fail_after)
        self.headers = headers

    @staticmethod
    async def _iterate(chunks, fail_after):
        for index, chunk in enumerate(chunks):
            if fail_after is not None and index >= fail_after:
                raise ConnectionError("stream reset")
            yield chunk


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


def _section_with_raw(context):
    return SimpleNamespace(
        with_raw_response=SimpleNamespace(convert=lambda **_kwargs: context)
    )


async def test_raw_stream_marks_acceptance_and_keeps_headers():
    section = _section_with_raw(
        _RawContext(_RawStream([b"ab", b"cd"], {"Character-Cost": "4"}))
    )
    state = BilledStream()

    audio = await read_billed_stream(section, "convert", {}, bytes, state)

    assert audio == b"abcd"
    assert state.accepted is True
    assert elevenlabs_billed_units(state.headers) == "4"


async def test_raw_stream_broken_after_acceptance_stays_accepted():
    section = _section_with_raw(
        _RawContext(_RawStream([b"ab", b"cd"], {}, fail_after=1))
    )
    state = BilledStream()

    with pytest.raises(ConnectionError):
        await read_billed_stream(section, "convert", {}, bytes, state)
    assert state.accepted is True
    assert state.audio == b"ab"


async def test_raw_stream_rejected_request_is_not_accepted():
    section = _section_with_raw(_RawContext(error=RuntimeError("401")))
    state = BilledStream()

    with pytest.raises(RuntimeError):
        await read_billed_stream(section, "convert", {}, bytes, state)
    assert state.accepted is False


async def test_sdk_without_raw_client_falls_back_to_first_chunk():
    async def convert(**_kwargs):
        yield b"x"

    state = BilledStream()
    audio = await read_billed_stream(
        SimpleNamespace(convert=convert), "convert", {}, bytes, state
    )

    assert (audio, state.accepted, state.headers) == (b"x", True, None)


async def test_call_with_headers_prefers_the_raw_client():
    async def raw_convert(**kwargs):
        return SimpleNamespace(data={"echo": kwargs}, headers={"character-cost": "9"})

    async def plain_convert(**kwargs):
        return {"plain": kwargs}

    raw_section = SimpleNamespace(
        convert=plain_convert,
        with_raw_response=SimpleNamespace(convert=raw_convert),
    )
    data, headers = await call_with_headers(raw_section, "convert", a=1)
    assert data == {"echo": {"a": 1}}
    assert elevenlabs_billed_units(headers) == "9"

    data, headers = await call_with_headers(
        SimpleNamespace(convert=plain_convert), "convert", a=1
    )
    assert (data, headers) == ({"plain": {"a": 1}}, None)


def test_cost_header_falls_back_to_the_older_documented_name():
    assert elevenlabs_billed_units({"X-Character-Count": "12"}) == "12"
    assert (
        elevenlabs_billed_units({"character-cost": "3", "x-character-count": "9"})
        == "3"
    )


def test_missing_cost_header_is_absent_not_zero():
    with TokenContextManager() as manager:
        record_provider_usage(
            SimpleNamespace(),
            MediaCallType.TTS,
            3,
            provider_billed_units=elevenlabs_billed_units({"request-id": "r"}),
        )

    [row] = _rows(manager)
    assert "provider_billed_units" not in row


@pytest.mark.parametrize(
    ("status", "accepted"), [(200, True), (299, True), (422, False), (None, False)]
)
def test_accepted_status(status, accepted):
    error = RuntimeError("x")
    error.status_code = status  # type: ignore[attr-defined]
    assert accepted_status(error) is accepted
