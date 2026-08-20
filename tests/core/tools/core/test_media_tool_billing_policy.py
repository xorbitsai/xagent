"""The media tools bill a provider call that happened, even if it then fails.

Music, sound effect, TTS and ASR are otherwise-symmetric tools that had three
different implicit "is this call billable" rules. The policy is now uniform:
usage is recorded as soon as the provider call returns, before the response is
validated, because a call that succeeded at the HTTP level but came back empty
or malformed was still charged for.
"""

from typing import Any, Optional

import pytest

from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.music.base import MusicResult
from xagent.core.tools.core.music_tool import MusicToolCore
from xagent.core.tools.core.sound_effect_tool import SoundEffectToolCore


class _EmptyMusicModel:
    """Provider that returns a well-formed-but-empty result. Still billed."""

    model_name = "music-a"

    async def generate_music(self, **kwargs: Any) -> MusicResult:
        _ = kwargs
        return MusicResult(audio=b"", format="mp3", raw_response={})


class _GarbageMusicModel:
    """Provider that returns the wrong type entirely. Still billed."""

    model_name = "music-b"

    async def generate_music(self, **kwargs: Any) -> Any:
        _ = kwargs
        return {"not": "a MusicResult"}


def _media_entries(manager: TokenContextManager) -> list[dict]:
    return [d for d in manager.get_usage().details if d.get("type") == "media"]


@pytest.mark.asyncio
async def test_music_bills_empty_audio_response() -> None:
    tool = MusicToolCore(models={"music-a": _EmptyMusicModel()})

    with TokenContextManager() as manager:
        result = await tool.generate_music(prompt="p", music_length_seconds=30)
        entries = _media_entries(manager)

    # The tool still reports failure to the caller...
    assert result["success"] is False
    # ...but the provider call happened and was billed, so it is metered.
    assert len(entries) == 1
    assert entries[0]["unit"] == "seconds"
    assert entries[0]["quantity"] == 30.0
    assert entries[0]["call_type"] == "music"


@pytest.mark.asyncio
async def test_music_bills_malformed_response() -> None:
    tool = MusicToolCore(models={"music-b": _GarbageMusicModel()})

    with TokenContextManager() as manager:
        result = await tool.generate_music(prompt="p", music_length_seconds=12)
        entries = _media_entries(manager)

    assert result["success"] is False
    assert len(entries) == 1
    # Unit stays seconds even though nothing usable came back: a (model, unit)
    # price table breaks if the unit varies with response completeness.
    assert entries[0]["unit"] == "seconds"
    assert entries[0]["quantity"] == 12.0


class _EmptySoundEffectModel:
    model_name = "sfx-a"

    async def generate_sound_effect(self, **kwargs: Any) -> Any:
        _ = kwargs
        return {"not": "a SoundEffectResult"}


@pytest.mark.asyncio
async def test_sound_effect_bills_malformed_response() -> None:
    tool = SoundEffectToolCore(models={"sfx-a": _EmptySoundEffectModel()})

    with TokenContextManager() as manager:
        result = await tool.generate_sound_effect(text="p", duration_seconds=4)
        entries = _media_entries(manager)

    assert result["success"] is False
    assert len(entries) == 1
    assert entries[0]["unit"] == "seconds"
    assert entries[0]["quantity"] == 4.0
    assert entries[0]["call_type"] == "sound_effect"


class _RecordingASR:
    """Captures the verbose flag the caller passed."""

    model_name = "asr-a"

    def __init__(self) -> None:
        self.verbose_seen: Optional[bool] = None

    async def transcribe(self, *args: Any, verbose: bool = False, **kwargs: Any) -> Any:
        _ = args, kwargs
        self.verbose_seen = verbose
        if not verbose:
            return "bare string, no timings"
        from xagent.core.model.asr.base import ASRResult, ASRSegment

        return ASRResult(
            text="hi",
            segments=[ASRSegment(text="hi", start=0.0, end=2.5, confidence=1.0)],
            language="eng",
        )


@pytest.mark.asyncio
async def test_asr_usage_is_unmeasured_without_verbose() -> None:
    # Guards the reason /speech/transcribe and the Telegram channel now pass
    # verbose=True: without it the provider returns a bare string with no
    # timings, and every call is metered as an unbillable 0 seconds.
    from xagent.core.model.asr.usage import record_asr_usage

    with TokenContextManager() as manager:
        record_asr_usage("bare string, no timings", model_name="asr-a")
        entries = _media_entries(manager)

    assert entries[0]["unit"] == "seconds"
    assert entries[0]["quantity"] == 0.0


@pytest.mark.asyncio
async def test_asr_usage_meters_duration_with_verbose() -> None:
    from xagent.core.model.asr.base import ASRResult, ASRSegment
    from xagent.core.model.asr.usage import record_asr_usage

    result = ASRResult(
        text="hi",
        segments=[ASRSegment(text="hi", start=0.0, end=2.5, confidence=1.0)],
        language="eng",
    )

    with TokenContextManager() as manager:
        record_asr_usage(result, model_name="asr-a")
        entries = _media_entries(manager)

    assert entries[0]["unit"] == "seconds"
    assert entries[0]["quantity"] == 2.5
    # Name-keyed identity across all three ASR entry points; see PR body.
    assert entries[0]["model"] == "asr-a"
    assert entries[0]["model_id"] == ""
