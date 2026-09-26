"""Media-usage recording for ASR.

ASR is reached from several entry points that do not go through ``audio_tool``
— the Telegram channel and the ``/speech/transcribe`` API both call
``transcribe`` directly — so this module is the single place that knows how to
turn a transcription result into a usage record. Every ASR caller routes here,
including ``audio_tool``: keeping one implementation is what stops the copies
from drifting (an earlier pair differed in whether they logged at all).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Union

from ...tools.core.media_usage import coerce_duration, record_media_seconds
from ..chat.token_context import MediaCallType
from .base import ASRResult

logger = logging.getLogger(__name__)

# Keys providers use for the total length of the submitted audio.
_DURATION_KEYS = ("duration", "audio_duration", "duration_seconds")


def duration_from_raw_response(raw_response: Any) -> Optional[float]:
    """Provider-reported total audio duration, if it exposed one.

    Preferred over segment timings: the end of the last spoken segment
    undercounts a recording with trailing silence, which the provider still
    processed and billed for.
    """
    if not isinstance(raw_response, dict):
        return None
    for key in _DURATION_KEYS:
        seconds = coerce_duration(raw_response.get(key))
        if seconds is not None:
            return seconds
    return None


def duration_from_segments(segments: Optional[Sequence[Any]]) -> Optional[float]:
    """Transcribed duration inferred from the end of the last timed segment.

    Accepts both ``ASRSegment`` objects and the plain dicts ``audio_tool``
    builds from them.
    """
    if not segments:
        return None
    last_end = 0.0
    for segment in segments:
        end = (
            segment.get("end")
            if isinstance(segment, dict)
            else getattr(segment, "end", None)
        )
        seconds = coerce_duration(end)
        if seconds is not None:
            last_end = max(last_end, seconds)
    return last_end if last_end > 0 else None


def resolve_asr_seconds(
    raw_response: Any = None,
    segments: Optional[Sequence[Any]] = None,
) -> Optional[float]:
    """Best available transcribed-audio duration, provider field first."""
    return duration_from_raw_response(raw_response) or duration_from_segments(segments)


def record_asr_seconds(
    seconds: Optional[float],
    *,
    model_name: str = "",
    model_id: str = "",
) -> None:
    """Record one transcription from an already-resolved duration.

    Thin wrapper over the shared :func:`record_media_seconds`: ASR is one of
    several duration-billed modalities, and keeping a second implementation is
    how the two drifted before. The shared helper owns the invariants — unit
    always seconds (a call whose duration is unknown records 0 rather than
    switching units, which would make a (model, unit) price table unusable),
    the unmeasured-call warning, and swallowing errors so accounting never
    breaks a transcription.
    """
    record_media_seconds(
        seconds,
        model=model_name,
        model_id=model_id,
        call_type=MediaCallType.ASR,
    )


def record_asr_usage(
    result: Union[str, ASRResult],
    *,
    model_name: str = "",
    model_id: str = "",
) -> None:
    """Record one transcription from a provider result.

    A non-verbose call returns a bare string with no timing information, so it
    is recorded as unmeasured rather than silently skipped.
    """
    raw_response = None
    segments = None
    if isinstance(result, ASRResult):
        raw_response = result.raw_response
        segments = result.segments
    record_asr_seconds(
        resolve_asr_seconds(raw_response, segments),
        model_name=model_name,
        model_id=model_id,
    )
