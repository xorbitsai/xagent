"""Usage recording at the provider boundary for non-image media.

Video, TTS, ASR, music and sound-effect providers record their own usage, the
same way image providers do (``image/usage.py``). The provider is the only
layer that sees every call: the tools, ``/speech/transcribe`` and the Telegram
voice handler all reach the same provider method, and only the provider holds
what the vendor actually reported -- the produced duration, the response
headers, the text it really sent after rewriting. Metering one layer up meant
one copy per entry point, each reading the request instead of the response.

Rules every producer here follows, matching the image boundary:

1. **One row per billed provider call.** Recorded inside the method that
   makes the call, so a retry wrapper around the provider records once per
   attempt it really made, never once per user request.
2. **Recorded before the response is validated.** Once the vendor has
   accepted the request the charge is real; a body that later turns out to be
   empty or malformed still leaves its row.
3. **The unit is the modality's.** ``quantity`` is always in
   ``MediaCallType.unit``; when it cannot be measured the row records 0 in that
   unit rather than switching units. The vendor's own charge, when it reports
   one, rides along as ``provider_billed_units``.
4. **Identity is the configured row.** ``model_id`` is the configured model id
   stamped on the provider at construction, because saas prices media rows by
   ``model_id`` alone; ``model`` is the provider-facing name. Placeholders are
   written as "" and never billed.

Recording is best-effort: nothing in this module raises into the media call.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, Callable, Mapping, Optional

from ..tools.core.media_usage import is_usable_model_name
from .chat.token_context import MediaCallType, add_media_usage

logger = logging.getLogger(__name__)

# "Argument not supplied", distinct from any value a provider could report.
_ABSENT: Any = object()

# ElevenLabs reports what a request cost in a response header. Its API docs
# name it ``character-cost``; an earlier revision of the same page read
# ``x-character-count``, so both are accepted, in that order.
ELEVENLABS_COST_HEADERS = ("character-cost", "x-character-count")

# ffprobe is a last resort for containers parsed nowhere else; bound its cost.
_FFPROBE_TIMEOUT_SECONDS = 15.0


def billing_identity(value: Any) -> str:
    """A model identity fit to bill, or "" for a placeholder or non-string."""
    return value.strip() if is_usable_model_name(value) else ""


def provider_identity(provider: Any) -> tuple[str, str]:
    """``(model, model_id)`` for a usage row, read from the provider object.

    ``model_id`` is the configured id stamped at construction; ``model`` is the
    first usable provider-facing name. Read defensively: this runs on the
    billing path, which must not raise.
    """

    def _read(name: str) -> Any:
        try:
            return getattr(provider, name, None)
        except Exception:  # noqa: BLE001
            return None

    model = billing_identity(_read("model_name")) or billing_identity(_read("model"))
    return model, billing_identity(_read("model_id"))


def record_provider_usage(
    provider: Any,
    call_type: MediaCallType,
    quantity: Any,
    *,
    output_tokens: Any = 0,
    resolution: str = "",
    provider_billed_units: Any = _ABSENT,
) -> None:
    """Record one billed provider call; never raises.

    ``quantity`` and the provider-reported extras are forwarded raw: the write
    boundary in ``token_context`` is the only sanitiser, and it keeps the row
    when a value is unusable (a bad quantity becomes 0, a bad optional field is
    omitted) instead of losing the call.
    """
    try:
        model, model_id = provider_identity(provider)
        extras: dict[str, Any] = {}
        if provider_billed_units is not _ABSENT:
            extras["provider_billed_units"] = provider_billed_units
        add_media_usage(
            call_type=call_type,
            quantity=0 if quantity is None else quantity,
            model=model,
            model_id=model_id,
            output_tokens=output_tokens,
            resolution=resolution if isinstance(resolution, str) else "",
            **extras,
        )
    except Exception as e:  # noqa: BLE001
        # exc_info: a persisted billing row cannot be repaired afterwards, so a
        # metering bug has to be diagnosable from the log alone.
        logger.warning("Failed to record %s usage: %s", call_type, e, exc_info=True)


def header_value(headers: Any, name: str) -> Any:
    """A response header, case-insensitively, or ``_ABSENT`` when missing."""
    if not isinstance(headers, Mapping):
        return _ABSENT
    try:
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == name:
                return value
    except Exception as e:  # noqa: BLE001
        logger.warning("Reading response header %s failed: %s", name, e)
    return _ABSENT


@dataclass
class BilledStream:
    """What a streamed ElevenLabs call produced, including when it failed.

    ``accepted`` turns True once the provider has accepted the request -- the
    point from which it charges -- so a caller can still record a call whose
    stream broke halfway. ``headers`` holds the response headers when the SDK
    exposed them, which is where ElevenLabs reports what the request cost.
    """

    accepted: bool = False
    headers: Any = None
    chunks: list[bytes] = field(default_factory=list)

    @property
    def audio(self) -> bytes:
        return b"".join(self.chunks)


async def read_billed_stream(
    section: Any,
    method: str,
    kwargs: dict[str, Any],
    coerce: Callable[[Any], bytes],
    state: BilledStream,
) -> bytes:
    """Stream an ElevenLabs audio endpoint into ``state``.

    Uses the SDK's ``with_raw_response`` client when present: it enters only on
    a 2xx response, which marks the request accepted before the first byte is
    read, and exposes the response headers. Older SDKs without it fall back to
    the plain call, where the first chunk is the earliest evidence of
    acceptance. Exceptions propagate; ``state`` keeps what was learned.
    """
    raw_client = getattr(section, "with_raw_response", None)
    raw_method = getattr(raw_client, method, None) if raw_client is not None else None
    if raw_method is not None:
        async with raw_method(**kwargs) as response:
            state.accepted = True
            state.headers = response.headers
            async for chunk in response.data:
                state.chunks.append(coerce(chunk))
        return state.audio

    response = getattr(section, method)(**kwargs)
    # ElevenLabs SDK 2.56+ exposes streaming methods as async generators;
    # older releases returned a coroutine resolving to an async iterator.
    if isawaitable(response):
        response = await response
    async for chunk in response:
        state.accepted = True
        state.chunks.append(coerce(chunk))
    return state.audio


async def call_with_headers(
    section: Any, method: str, **kwargs: Any
) -> tuple[Any, Any]:
    """Call a non-streaming ElevenLabs endpoint, returning ``(data, headers)``.

    ``headers`` is None when the SDK has no ``with_raw_response`` client.
    """
    raw_client = getattr(section, "with_raw_response", None)
    raw_method = getattr(raw_client, method, None) if raw_client is not None else None
    if raw_method is None:
        return await getattr(section, method)(**kwargs), None
    response = await raw_method(**kwargs)
    return response.data, response.headers


def elevenlabs_billed_units(headers: Any) -> Any:
    """ElevenLabs' reported charge for a request, or ``_ABSENT``."""
    for name in ELEVENLABS_COST_HEADERS:
        value = header_value(headers, name)
        if value is not _ABSENT:
            return value
    return _ABSENT


def accepted_status(error: BaseException) -> bool:
    """Whether an SDK error carries a 2xx status: a charged, unparsable reply.

    The ElevenLabs SDK parses JSON bodies inside the call, so a 200 whose body
    does not parse surfaces as an ``ApiError``/``ParsingError`` that still
    carries the 2xx status -- the provider charged for it all the same.
    """
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and 200 <= status < 300


def positive_seconds(value: Any) -> Optional[float]:
    """A finite positive number of seconds, or None when unusable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return seconds


# --- Measuring audio duration from bytes -------------------------------------

# (version, layer) -> samples per frame; version 3 = MPEG-1, 2 = MPEG-2,
# 0 = MPEG-2.5; layer 3 = Layer I, 2 = Layer II, 1 = Layer III.
_MP3_SAMPLES = {
    (3, 3): 384,
    (3, 2): 1152,
    (3, 1): 1152,
    (2, 3): 384,
    (2, 2): 1152,
    (2, 1): 576,
    (0, 3): 384,
    (0, 2): 1152,
    (0, 1): 576,
}
_MP3_SAMPLE_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000)}
_MP3_SAMPLE_RATES[0] = (11025, 12000, 8000)
# kbps by bitrate index, for MPEG-1 and for MPEG-2/2.5, per layer.
_MP3_BITRATES = {
    (3, 3): (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    (3, 2): (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    (3, 1): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    (2, 3): (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
    (2, 2): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
_MP3_BITRATES[(2, 1)] = _MP3_BITRATES[(2, 2)]
for _layer in (3, 2, 1):
    _MP3_BITRATES[(0, _layer)] = _MP3_BITRATES[(2, 3 if _layer == 3 else 2)]


def _mp3_seconds(data: bytes) -> Optional[float]:
    """Duration of an MPEG audio stream by walking its frame headers.

    Exact for both constant and variable bitrate, because it sums the samples
    each frame really carries instead of dividing size by a nominal bitrate.
    Stops at the first byte run that is not a frame, so trailing tags or junk
    end the walk rather than inflating it.
    """
    pos = 0
    if data[:3] == b"ID3" and len(data) >= 10:
        size = 0
        for byte in data[6:10]:
            size = (size << 7) | (byte & 0x7F)
        pos = 10 + size
    samples = 0
    sample_rate = 0
    frames = 0
    end = len(data)
    while pos + 4 <= end:
        header = int.from_bytes(data[pos : pos + 4], "big")
        if header >> 21 != 0x7FF:
            if frames:
                break
            pos += 1  # resync before the first frame only
            continue
        version = (header >> 19) & 0x3
        layer = (header >> 17) & 0x3
        bitrate_index = (header >> 12) & 0xF
        rate_index = (header >> 10) & 0x3
        padding = (header >> 9) & 0x1
        if version == 1 or layer == 0 or bitrate_index in (0, 15) or rate_index == 3:
            if frames:
                break
            pos += 1
            continue
        rate = _MP3_SAMPLE_RATES[version][rate_index]
        bitrate = _MP3_BITRATES[(version, layer)][bitrate_index] * 1000
        frame_samples = _MP3_SAMPLES[(version, layer)]
        if layer == 3:
            length = (12 * bitrate // rate + padding) * 4
        else:
            length = frame_samples // 8 * bitrate // rate + padding
        if length <= 4:
            break
        samples += frame_samples
        sample_rate = rate
        frames += 1
        pos += length
    if not frames or not sample_rate:
        return None
    return samples / sample_rate


def _wav_seconds(data: bytes) -> Optional[float]:
    try:
        with wave.open(io.BytesIO(data)) as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else None
    except (wave.Error, EOFError, ValueError):
        return None


def _ffprobe_seconds(data: bytes) -> Optional[float]:
    """Container-reported duration via ffprobe, when it is installed."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                "-i",
                "pipe:0",
            ],
            input=data,
            capture_output=True,
            timeout=_FFPROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("ffprobe failed to measure audio duration: %s", e)
        return None
    return positive_seconds(completed.stdout.decode(errors="ignore").strip())


def _raw_pcm_rate(audio_format: str) -> tuple[int, int] | None:
    """``(sample_rate, bytes_per_sample)`` for headerless ElevenLabs formats."""
    name, _, rest = audio_format.partition("_")
    rate_text = rest.split("_", 1)[0]
    if not rate_text.isdigit():
        return None
    if name == "pcm":
        return int(rate_text), 2  # 16-bit mono
    if name in {"ulaw", "alaw"}:
        return int(rate_text), 1
    return None


async def measure_audio_seconds_async(
    data: Any, audio_format: str = ""
) -> Optional[float]:
    """:func:`measure_audio_seconds` off the event loop.

    Measuring can shell out to ffprobe, which would otherwise block every other
    request on the loop for as long as the probe runs.
    """
    return await asyncio.to_thread(measure_audio_seconds, data, audio_format)


def measure_audio_seconds(data: Any, audio_format: str = "") -> Optional[float]:
    """Seconds of audio in ``data``, or None when it cannot be measured.

    ``audio_format`` is a hint such as ElevenLabs' ``mp3_44100_128`` or
    ``pcm_24000``, or a plain container name. Headerless PCM/µ-law/A-law is
    measured from its byte length, WAV from its header, MPEG audio by walking
    its frames; anything else (Opus, OGG, WebM, M4A...) goes to ffprobe when it
    is installed. Never raises.
    """
    if not isinstance(data, (bytes, bytearray)) or not data:
        return None
    payload = bytes(data)
    fmt = (audio_format or "").strip().lower()
    try:
        raw = _raw_pcm_rate(fmt)
        if raw is not None:
            rate, width = raw
            return positive_seconds(len(payload) / (rate * width))
        if payload[:4] == b"RIFF" and payload[8:12] == b"WAVE":
            return positive_seconds(_wav_seconds(payload))
        if (
            fmt.startswith("mp3")
            or payload[:3] == b"ID3"
            or (payload[0] == 0xFF and payload[1] & 0xE0 == 0xE0)
        ):
            seconds = positive_seconds(_mp3_seconds(payload))
            if seconds is not None:
                return seconds
        return _ffprobe_seconds(payload)
    except Exception as e:  # noqa: BLE001
        logger.warning("Measuring audio duration failed: %s", e)
        return None
