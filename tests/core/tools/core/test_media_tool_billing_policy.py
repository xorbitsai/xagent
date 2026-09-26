"""The media tools bill a provider call that happened, even if it then fails.

Music, sound effect, TTS and ASR are otherwise-symmetric tools that had three
different implicit "is this call billable" rules. The policy is now uniform:
usage is recorded as soon as the provider call returns, before the response is
validated, because a call that succeeded at the HTTP level but came back empty
or malformed was still charged for.
"""

from pathlib import Path
from typing import Any, Optional

import pytest

from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.music.base import MusicResult
from xagent.core.tools.core.music_tool import MusicToolCore
from xagent.core.tools.core.sound_effect_tool import SoundEffectToolCore

# Sentinel distinguishing "this fake has no model_name attribute at all"
# (Xinference's default model behaves this way) from "its name is empty".
_NO_NAME = object()

# The two ways a provider call can succeed at the HTTP level and still yield
# nothing usable. Both are billed, which is the policy this module guards.
_EMPTY_MUSIC = MusicResult(audio=b"", format="mp3", raw_response={})
_GARBAGE = {"not": "a result object"}
_PLAYABLE_MUSIC = MusicResult(audio=b"x", format="mp3", raw_response={})


def _music_model(result: Any = _GARBAGE, *, name: Any = "music-a") -> Any:
    """A music provider returning ``result``; ``name=_NO_NAME`` exposes none."""

    class _FakeMusicModel:
        async def generate_music(self, **kwargs: Any) -> Any:
            _ = kwargs
            return result

    if name is not _NO_NAME:
        _FakeMusicModel.model_name = name  # type: ignore[attr-defined]
    return _FakeMusicModel()


def _sound_effect_model(*, name: Any = "sfx-a") -> Any:
    """A sound-effect provider returning the wrong type; still billed."""

    class _FakeSoundEffectModel:
        async def generate_sound_effect(self, **kwargs: Any) -> Any:
            _ = kwargs
            return _GARBAGE

    if name is not _NO_NAME:
        _FakeSoundEffectModel.model_name = name  # type: ignore[attr-defined]
    return _FakeSoundEffectModel()


def _media_entries(manager: TokenContextManager) -> list[dict]:
    return [d for d in manager.get_usage().details if d.get("type") == "media"]


@pytest.mark.parametrize(
    ("bad_result", "seconds"),
    [
        pytest.param(_EMPTY_MUSIC, 30.0, id="well-formed-but-empty"),
        pytest.param(_GARBAGE, 12.0, id="wrong-type-entirely"),
    ],
)
@pytest.mark.asyncio
async def test_music_bills_a_call_that_returned_nothing_usable(
    bad_result: Any, seconds: float
) -> None:
    """Both ways a call can succeed at the HTTP level and yield nothing.

    The unit stays seconds in both: a (model, unit) price table breaks if the
    unit varies with how complete the response happened to be.
    """
    tool = MusicToolCore(models={"music-a": _music_model(bad_result)})

    with TokenContextManager() as manager:
        result = await tool.generate_music(prompt="p", music_length_seconds=seconds)
        entries = _media_entries(manager)

    # The tool still reports failure to the caller...
    assert result["success"] is False
    # ...but the provider call happened and was billed, so it is metered.
    assert len(entries) == 1
    assert entries[0]["unit"] == "seconds"
    assert entries[0]["quantity"] == seconds
    assert entries[0]["call_type"] == "music"


@pytest.mark.asyncio
async def test_sound_effect_bills_malformed_response() -> None:
    tool = SoundEffectToolCore(models={"sfx-a": _sound_effect_model()})

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


# --- ASR duration precedence (resolve_asr_seconds) ---------------------------
#
# The producer tests above cover a bare string and one ordered segment. The
# resolver has several more branches — each provider total field, the fallback
# to segment ends, unsorted segments and unusable values — and a regression in
# any of them would silently change the billed quantity while leaving the
# suite green.


@pytest.mark.parametrize("field", ["duration", "audio_duration", "duration_seconds"])
def test_provider_total_duration_wins_over_segments(field: str) -> None:
    """A provider total covers trailing silence the last segment end misses."""
    from xagent.core.model.asr.usage import resolve_asr_seconds

    seconds = resolve_asr_seconds(
        {field: 30.0},
        [{"start": 0.0, "end": 2.5}],
    )
    assert seconds == 30.0


def test_segment_ends_are_used_when_no_provider_total() -> None:
    from xagent.core.model.asr.usage import resolve_asr_seconds

    assert resolve_asr_seconds({}, [{"start": 0.0, "end": 7.5}]) == 7.5


def test_unsorted_segments_take_the_maximum_end() -> None:
    """Segment order is not guaranteed, so the last element is not the end."""
    from xagent.core.model.asr.usage import resolve_asr_seconds

    seconds = resolve_asr_seconds(
        None,
        [{"start": 5.0, "end": 9.0}, {"start": 0.0, "end": 3.0}],
    )
    assert seconds == 9.0


@pytest.mark.parametrize("bad", [None, "abc", float("inf"), float("nan"), -1.0, True])
def test_unusable_provider_total_falls_back_to_segments(bad: Any) -> None:
    """A non-finite/negative/boolean total must not be billed as a duration."""
    from xagent.core.model.asr.usage import resolve_asr_seconds

    seconds = resolve_asr_seconds({"duration": bad}, [{"start": 0.0, "end": 4.0}])
    assert seconds == 4.0


def test_no_usable_timing_reports_none() -> None:
    """None (not 0.0) so the caller can tell "unmeasured" from "zero-length";
    record_media_seconds turns it into a 0-second row plus a warning."""
    from xagent.core.model.asr.usage import resolve_asr_seconds

    assert resolve_asr_seconds({}, []) is None
    assert resolve_asr_seconds(None, None) is None
    assert resolve_asr_seconds({"duration": None}, [{"start": 0.0, "end": 0.0}]) is None


# --- Media identity field shape ---------------------------------------------
#
# Convention: `model` carries the human-readable provider name, `model_id` the
# configured id. Writing the configured id into both (or leaving model_id
# empty) loses the canonical name for display and external consumers. Each
# case below uses a provider name and a configured id that differ, so a
# resolver that returns the id for both fields cannot pass.


@pytest.mark.asyncio
async def test_music_records_provider_name_and_configured_id_separately() -> None:
    """`model` carries the provider's name, `model_id` the configured id.

    Writing the id into both fields loses the canonical name for display and
    for external consumers.
    """
    model = _music_model(_PLAYABLE_MUSIC, name="music-provider-name")
    tool = MusicToolCore(models={"configured-music-id": model})

    with TokenContextManager() as manager:
        await tool.generate_music(prompt="p", music_length_seconds=10)
        entries = _media_entries(manager)

    assert len(entries) == 1
    assert entries[0]["model"] == "music-provider-name"
    assert entries[0]["model_id"] == "configured-music-id"


@pytest.mark.asyncio
async def test_sound_effect_records_provider_name_and_configured_id_separately() -> (
    None
):
    model = _sound_effect_model(name="sfx-provider-name")
    tool = SoundEffectToolCore(models={"configured-sfx-id": model})

    with TokenContextManager() as manager:
        await tool.generate_sound_effect(text="p", duration_seconds=4)
        entries = _media_entries(manager)

    assert len(entries) == 1
    assert entries[0]["model"] == "sfx-provider-name"
    assert entries[0]["model_id"] == "configured-sfx-id"


@pytest.mark.asyncio
async def test_music_falls_back_to_configured_id_not_the_class_name() -> None:
    """A provider with no model_name must still bill under its configured id.

    The resolver's fallback is what decides this, and a Python class name is
    not a billing identity: it is not configurable, not unique across
    providers, and means nothing to a price table.
    """
    model = _music_model(_PLAYABLE_MUSIC, name=_NO_NAME)
    tool = MusicToolCore(models={"configured-music-id": model})

    with TokenContextManager() as manager:
        await tool.generate_music(prompt="p", music_length_seconds=10)
        entries = _media_entries(manager)

    assert len(entries) == 1
    assert entries[0]["model"] == "configured-music-id"
    assert entries[0]["model"] != type(model).__name__
    assert entries[0]["model_id"] == "configured-music-id"


@pytest.mark.asyncio
async def test_sound_effect_falls_back_to_configured_id_not_the_class_name() -> None:
    model = _sound_effect_model(name=_NO_NAME)
    tool = SoundEffectToolCore(models={"configured-sfx-id": model})

    with TokenContextManager() as manager:
        await tool.generate_sound_effect(text="p", duration_seconds=4)
        entries = _media_entries(manager)

    assert len(entries) == 1
    assert entries[0]["model"] == "configured-sfx-id"
    assert entries[0]["model"] != type(model).__name__


# --- Video billing --------------------------------------------------------
#
# video_tool is duration-billed like music/sound effect, but bills
# `duration * n` because the provider reports one duration while generating
# and charging for n videos. These cover the multiplication, the identity
# fields, and the async path that reports no duration.


def _video_model(duration: Any, *, name: str = "video-provider") -> Any:
    from unittest.mock import AsyncMock, Mock

    from xagent.core.model.video.base import BaseVideoModel

    model = Mock(spec=BaseVideoModel)
    model.has_ability = Mock(return_value=True)
    model.abilities = ["generate"]
    model.model_name = name
    result = {"task_id": "t-1", "status": "succeeded", "video_url": "", "ratio": "16:9"}
    if duration is not None:
        result["duration"] = duration
    model.generate_video = AsyncMock(return_value=result)
    return model


@pytest.mark.asyncio
async def test_video_bills_duration_times_count() -> None:
    """The provider reports one duration but generates and bills for n videos."""
    from xagent.core.tools.core.video_tool import VideoGenerationToolCore

    tool = VideoGenerationToolCore(video_models={"cfg-video-id": _video_model(5)})

    with TokenContextManager() as manager:
        await tool.generate_video(prompt="p", n=3)
        entries = _media_entries(manager)

    assert len(entries) == 1
    assert entries[0]["unit"] == "seconds"
    assert entries[0]["quantity"] == 15.0
    assert entries[0]["call_type"] == "video"


@pytest.mark.asyncio
async def test_video_records_provider_name_and_configured_id_separately() -> None:
    """`model` is the provider name, `model_id` the configured id."""
    from xagent.core.tools.core.video_tool import VideoGenerationToolCore

    tool = VideoGenerationToolCore(
        video_models={"cfg-video-id": _video_model(5, name="video-provider")}
    )

    with TokenContextManager() as manager:
        await tool.generate_video(prompt="p")
        entries = _media_entries(manager)

    assert len(entries) == 1
    assert entries[0]["model"] == "video-provider"
    assert entries[0]["model_id"] == "cfg-video-id"


@pytest.mark.asyncio
async def test_video_without_duration_is_recorded_as_unmeasured_seconds() -> None:
    """An async task reports no duration yet. The unit must stay seconds with
    quantity 0 rather than degrading to another unit -- reconciling that row is
    tracked in #1583."""
    from xagent.core.tools.core.video_tool import VideoGenerationToolCore

    tool = VideoGenerationToolCore(video_models={"cfg-video-id": _video_model(None)})

    with TokenContextManager() as manager:
        await tool.generate_video(prompt="p")
        entries = _media_entries(manager)

    assert len(entries) == 1
    assert entries[0]["unit"] == "seconds"
    assert entries[0]["quantity"] == 0.0


# --- ASR records before post-processing -----------------------------------


class _MalformedSegmentASR:
    """Provider call succeeds and is billed; a segment lacks an end time.

    _aggregate_segments raises ValueError on this input, which the tool's
    broad handler turns into success:False. The provider still ran and
    charged, so the row must already be recorded by then.
    """

    model_name = "asr-provider"

    @property
    def abilities(self) -> list:
        return ["asr", "timestamps"]

    async def transcribe(self, audio: Any, **kwargs: Any) -> Any:
        _ = (audio, kwargs)
        from xagent.core.model.asr.base import ASRResult, ASRSegment

        return ASRResult(
            text="hello world",
            raw_response={"duration": 42.0},
            segments=[
                ASRSegment(text="hello", start=0.0, end=1.0, confidence=1.0),
                ASRSegment(text="world", start=1.0, end=None, confidence=1.0),
            ],
        )


@pytest.mark.asyncio
async def test_asr_bills_when_post_processing_raises() -> None:
    """Metering must not depend on post-processing succeeding.

    Recording after _aggregate_segments meant a provider call that succeeded
    and was billed went entirely unmetered whenever segment data was
    malformed -- the exact bug class this module's policy exists to prevent.
    """
    from xagent.core.tools.core.audio_tool import AudioToolCore

    tool = AudioToolCore(asr_models={"asr-provider": _MalformedSegmentASR()})

    with TokenContextManager() as manager:
        result = await tool.transcribe_audio(
            audio_file_path="/tmp/does-not-exist.wav", verbose=False
        )
        entries = _media_entries(manager)

    # The tool still reports failure to the caller...
    assert result["success"] is False
    # ...but the provider call happened and was billed, so it is metered.
    assert len(entries) == 1
    assert entries[0]["unit"] == "seconds"
    assert entries[0]["quantity"] == 42.0
    assert entries[0]["call_type"] == "asr"


# --- verbose=True reaches the provider ------------------------------------
#
# _RecordingASR exists to capture the flag. Without verbose=True the provider
# returns a bare string with no timings and every call meters as an
# unbillable 0 seconds, so the flag reaching the provider is what makes ASR
# billable at all.


@pytest.mark.asyncio
async def test_telegram_passes_verbose_and_bills_a_real_duration(
    tmp_path: Path,
) -> None:
    """Drive the real Telegram method, not a re-implementation of it.

    Asserting on a locally-issued transcribe() call would pass with
    verbose=True deleted from bot.py, which is the only thing worth guarding
    here: without it the provider returns a bare string with no timings and
    every voice message meters as an unbillable 0 seconds.
    """
    from xagent.web.channels.telegram.bot import TelegramBotInstance

    provider = _RecordingASR()
    # __new__ rather than the real constructor: the method only touches a
    # class-level timeout and a staticmethod, so a fully wired bot instance
    # would add setup without adding coverage.
    bot = TelegramBotInstance.__new__(TelegramBotInstance)

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"fake-ogg")

    with TokenContextManager() as manager:
        transcripts = await bot._transcribe_uploaded_voice_files(
            ["file-1"],
            [{"telegram_file_id": "file-1", "path": str(audio), "name": "voice.ogg"}],
            provider,
        )
        entries = _media_entries(manager)

    # bot.py passed verbose through to the provider...
    assert provider.verbose_seen is True
    assert transcripts == {"file-1": "hi"}
    # ...so the row carries a real duration instead of an unbillable zero.
    assert len(entries) == 1
    assert entries[0]["unit"] == "seconds"
    assert entries[0]["quantity"] == 2.5
    # and the identity is the provider name, never the forbidden placeholder.
    assert entries[0]["model"] == "asr-a"
    assert entries[0]["model"] != "default"
