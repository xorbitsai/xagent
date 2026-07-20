"""Tests for audio tool TTS provider options and voice listing."""

from __future__ import annotations

import json
from typing import Any, Optional, Union

import pytest

from xagent.core.model.asr.base import ASRResult, ASRSegment, BaseASR
from xagent.core.model.tts.base import BaseTTS, TTSResult
from xagent.core.tools.adapters.vibe.audio_tool import AudioTool
from xagent.core.tools.core.audio_tool import AudioToolCore
from xagent.core.workspace import TaskWorkspace


class FakeASR(BaseASR):
    async def transcribe(
        self,
        audio: Union[str, bytes],
        language: Optional[str] = None,
        format: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Union[str, ASRResult]:
        _ = audio, format, kwargs
        if not verbose:
            return "Hello world"
        return ASRResult(
            text="Hello world",
            segments=[
                ASRSegment(
                    text="Hello",
                    start=0.1,
                    end=0.5,
                    confidence=0.99,
                ),
                ASRSegment(
                    text="world",
                    start=0.55,
                    end=1.0,
                    confidence=0.98,
                ),
            ],
            language=language or "eng",
        )

    @property
    def abilities(self) -> list[str]:
        return ["asr", "timestamps"]


class FakeTTS(BaseTTS):
    def __init__(
        self,
        *,
        provider_name: str = "fake",
        abilities: Optional[list[str]] = None,
        voices: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._provider_name = provider_name
        self._abilities = abilities or ["tts", "text_to_speech"]
        self._voices = voices or []
        self.calls: list[dict[str, Any]] = []
        self.clone_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.model_name = "fake-tts"

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[bytes, TTSResult]:
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "language": language,
                "format": format,
                "sample_rate": sample_rate,
                **kwargs,
            }
        )
        return TTSResult(
            audio=b"fake-audio",
            format=format or "mp3",
            sample_rate=sample_rate,
            language=language,
        )

    @property
    def abilities(self) -> list[str]:
        return self._abilities

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def supported_voice_settings(self) -> list[str]:
        return ["stability", "style"]

    @property
    def supported_provider_options(self) -> list[str]:
        return ["seed", "apply_text_normalization"]

    async def list_available_voices(self) -> list[dict[str, Any]]:
        return self._voices

    async def clone_voice(
        self,
        *,
        name: str,
        reference_audio_files: list[str],
        description: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        remove_background_noise: bool = False,
    ) -> dict[str, Any]:
        self.clone_calls.append(
            {
                "name": name,
                "reference_audio_files": reference_audio_files,
                "description": description,
                "labels": labels,
                "remove_background_noise": remove_background_noise,
            }
        )
        return {
            "voice_id": "persistent-voice",
            "name": name,
            "provider": self.provider_name,
            "persistent": True,
            "requires_verification": False,
        }

    async def delete_voice(self, voice_id: str) -> None:
        self.delete_calls.append(voice_id)


class ClosableTTS(FakeTTS):
    def __init__(self) -> None:
        super().__init__()
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


def test_audio_path_resolution_uses_workspace_file_id(tmp_path) -> None:
    workspace = TaskWorkspace("task_audio", base_dir=str(tmp_path))
    audio_path = workspace.output_dir / "voice.mp3"
    audio_path.write_bytes(b"audio")
    file_id = workspace.register_file(str(audio_path))
    tool = AudioToolCore(workspace=workspace)

    assert tool._resolve_audio_path(file_id) == str(audio_path.resolve())


def test_audio_path_resolution_does_not_fall_back_to_process_cwd(
    tmp_path, monkeypatch
) -> None:
    workspace = TaskWorkspace("task_audio", base_dir=str(tmp_path / "uploads"))
    process_cwd = tmp_path / "repo"
    misplaced_audio = process_cwd / "output" / "voice.mp3"
    misplaced_audio.parent.mkdir(parents=True)
    misplaced_audio.write_bytes(b"audio")
    monkeypatch.chdir(process_cwd)
    tool = AudioToolCore(workspace=workspace)

    with pytest.raises(FileNotFoundError, match="not found in workspace"):
        tool._resolve_audio_path("output/voice.mp3")


async def test_transcribe_audio_verbose_preserves_provider_timestamps(
    tmp_path,
) -> None:
    workspace = TaskWorkspace("task_audio", base_dir=str(tmp_path))
    audio_path = workspace.output_dir / "voice.mp3"
    audio_path.write_bytes(b"audio")
    file_id = workspace.register_file(str(audio_path))
    tool = AudioToolCore(asr_models={"fake": FakeASR()}, workspace=workspace)

    result = await tool.transcribe_audio(
        audio_file_path=file_id,
        model_id="fake",
        verbose=True,
    )

    assert result["success"] is True
    assert result["segment_count"] == 2
    assert result["segments"] == [
        {
            "text": "Hello",
            "start": 0.1,
            "end": 0.5,
            "speaker": None,
            "confidence": 0.99,
        },
        {
            "text": "world",
            "start": 0.55,
            "end": 1.0,
            "speaker": None,
            "confidence": 0.98,
        },
    ]

    with open(result["transcription_path"], encoding="utf-8") as f:
        transcription = json.load(f)
    assert transcription["segments"] == result["segments"]
    assert transcription["metadata"]["segment_view"] == "raw"
    assert transcription["metadata"]["raw_segment_count"] == 2
    assert transcription["metadata"]["total_segments"] == 2
    assert transcription["metadata"]["segments_merged"] is False


async def test_transcribe_audio_default_returns_processed_segments(tmp_path) -> None:
    workspace = TaskWorkspace("task_audio", base_dir=str(tmp_path))
    audio_path = workspace.output_dir / "voice.mp3"
    audio_path.write_bytes(b"audio")
    file_id = workspace.register_file(str(audio_path))
    tool = AudioToolCore(asr_models={"fake": FakeASR()}, workspace=workspace)

    result = await tool.transcribe_audio(
        audio_file_path=file_id,
        model_id="fake",
    )

    assert result["success"] is True
    assert result["segments"] == [
        {
            "text": "Hello world",
            "start": 0.1,
            "end": 1.0,
            "speaker": None,
            "confidence": pytest.approx(0.985),
        }
    ]

    with open(result["transcription_path"], encoding="utf-8") as f:
        transcription = json.load(f)
    assert transcription["metadata"]["segment_view"] == "processed"
    assert transcription["metadata"]["raw_segment_count"] == 2
    assert transcription["metadata"]["total_segments"] == 1
    assert transcription["metadata"]["segments_merged"] is True


def test_aggregate_segments_uses_sentence_pause_and_speaker_boundaries() -> None:
    tool = AudioToolCore()
    segments = [
        {"text": "Hello", "start": 0.0, "end": 0.2, "speaker": "a"},
        {"text": "world", "start": 0.25, "end": 0.5, "speaker": "a"},
        {"text": ".", "start": 0.5, "end": 0.55, "speaker": "a"},
        {"text": "Next", "start": 0.6, "end": 0.9, "speaker": "a"},
        {"text": "pause", "start": 1.5, "end": 1.8, "speaker": "a"},
        {"text": "Speaker", "start": 1.85, "end": 2.1, "speaker": "b"},
    ]

    aggregated = tool._aggregate_segments(segments)

    assert [segment["text"] for segment in aggregated] == [
        "Hello world.",
        "Next",
        "pause",
        "Speaker",
    ]


@pytest.mark.parametrize(
    "invalid_segment",
    [
        {"text": "Missing start", "end": 1.0},
        {"text": "Null end", "start": 0.6, "end": None},
    ],
)
def test_aggregate_segments_rejects_missing_timestamps(
    invalid_segment: dict[str, Any],
) -> None:
    tool = AudioToolCore()

    with pytest.raises(
        ValueError,
        match="Segment start or end time is missing or null",
    ):
        tool._aggregate_segments(
            [
                {"text": "Valid", "start": 0.0, "end": 0.5},
                invalid_segment,
            ]
        )


async def test_aclose_closes_unique_configured_model_clients() -> None:
    tts = ClosableTTS()
    tool = AudioToolCore(
        tts_models={"fake": tts},
        default_tts_model=tts,
    )

    await tool.aclose()

    assert tts.close_count == 1


async def test_audio_function_tool_teardown_closes_owner_once_per_task() -> None:
    tts = ClosableTTS()
    audio_tool = AudioTool(tts_models={"fake": tts})

    for tool in audio_tool.get_tools():
        await tool.teardown(task_id="task-1")

    assert tts.close_count == 1

    for tool in audio_tool.get_tools():
        await tool.teardown(task_id="task-2")

    assert tts.close_count == 2


async def test_synthesize_speech_passes_structured_tts_options() -> None:
    tts = FakeTTS(abilities=["tts", "voice_cloning", "voice_settings"])
    tool = AudioToolCore(tts_models={"fake": tts})

    result = await tool.synthesize_speech(
        text="Hello",
        voice="voice-1",
        language="en",
        audio_format="wav",
        sample_rate=44100,
        reference_audio="ref.wav",
        voice_settings={"stability": 0.5, "style": 0.2},
        provider_options={"seed": 1234},
        model_id="fake",
    )

    assert result["success"] is True
    assert tts.calls == [
        {
            "text": "Hello",
            "voice": "voice-1",
            "language": "en",
            "format": "wav",
            "sample_rate": 44100,
            "reference_audio": "ref.wav",
            "voice_settings": {"stability": 0.5, "style": 0.2},
            "seed": 1234,
        }
    ]


async def test_synthesize_speech_rejects_reserved_provider_options() -> None:
    tts = FakeTTS()
    tool = AudioToolCore(tts_models={"fake": tts})

    result = await tool.synthesize_speech(
        text="Hello",
        provider_options={"format": "wav"},
        model_id="fake",
    )

    assert result["success"] is False
    assert (
        "provider_options must not include standard TTS parameters" in result["error"]
    )
    assert tts.calls == []


async def test_synthesize_speech_rejects_unsupported_reference_audio() -> None:
    tts = FakeTTS(provider_name="elevenlabs")
    tool = AudioToolCore(tts_models={"elevenlabs": tts})

    result = await tool.synthesize_speech(
        text="Hello",
        reference_audio="ref.wav",
        model_id="elevenlabs",
    )

    assert result["success"] is False
    assert "does not support reference_audio voice cloning" in result["error"]
    assert tts.calls == []


async def test_synthesize_speech_rejects_unsupported_provider_options() -> None:
    tts = FakeTTS(provider_name="elevenlabs")
    tool = AudioToolCore(tts_models={"elevenlabs": tts})

    result = await tool.synthesize_speech(
        text="Hello",
        provider_options={"reference_audio": "ref.wav"},
        model_id="elevenlabs",
    )

    assert result["success"] is False
    assert (
        "Unsupported provider_options keys for provider 'elevenlabs': reference_audio"
        in result["error"]
    )
    assert tts.calls == []


async def test_synthesize_speech_rejects_unsupported_voice_settings() -> None:
    tts = FakeTTS(provider_name="elevenlabs")
    tool = AudioToolCore(tts_models={"elevenlabs": tts})

    result = await tool.synthesize_speech(
        text="Hello",
        voice_settings={"emotion": "happy"},
        model_id="elevenlabs",
    )

    assert result["success"] is False
    assert (
        "Unsupported voice_settings keys for provider 'elevenlabs': emotion"
        in result["error"]
    )
    assert tts.calls == []


def test_list_audio_models_includes_tts_voice_capabilities() -> None:
    tts = FakeTTS(
        provider_name="elevenlabs",
        abilities=["tts", "multiple_voices", "voice_listing", "voice_settings"],
    )
    tool = AudioToolCore(
        tts_models={"eleven_v3": tts},
        model_descriptions={"eleven_v3": "ElevenLabs TTS"},
    )

    result = tool.list_available_models()

    assert result["success"] is True
    assert result["models"] == [
        {
            "type": "tts",
            "model_id": "eleven_v3",
            "provider": "elevenlabs",
            "available": True,
            "description": "ElevenLabs TTS",
            "abilities": ["tts", "multiple_voices", "voice_listing", "voice_settings"],
            "supports_multiple_voices": True,
            "supports_voice_listing": True,
            "supports_voice_settings": True,
            "supports_voice_cloning": False,
            "supports_persistent_voice_cloning": False,
            "supported_voice_settings": ["stability", "style"],
            "supported_provider_options": ["seed", "apply_text_normalization"],
        }
    ]


async def test_list_tts_voices_returns_provider_voices() -> None:
    tts = FakeTTS(
        provider_name="elevenlabs",
        abilities=["tts", "voice_listing"],
        voices=[{"voice_id": "voice-1", "name": "Rachel"}],
    )
    tool = AudioToolCore(tts_models={"eleven_v3": tts})

    result = await tool.list_tts_voices(model_id="eleven_v3")

    assert result == {
        "success": True,
        "supported": True,
        "voices": [{"voice_id": "voice-1", "name": "Rachel"}],
        "count": 1,
        "provider": "elevenlabs",
        "model_used": "eleven_v3",
        "supported_providers": ["elevenlabs"],
    }


async def test_list_tts_voices_filters_metadata_to_configured_provider_models() -> None:
    voice_listing_tts = FakeTTS(
        provider_name="elevenlabs",
        abilities=["tts", "voice_listing"],
        voices=[
            {
                "voice_id": "voice-1",
                "high_quality_base_model_ids": [
                    "eleven_v3",
                    "eleven_flash_v2",
                    "eleven_turbo_v2",
                ],
                "verified_languages": [
                    {"language": "en", "model_id": "eleven_v3"},
                    {"language": "de", "model_id": "eleven_flash_v2"},
                    {"language": "fr", "model_id": "eleven_turbo_v2"},
                    {"language": "es"},
                    {"language": "it", "model_id": None},
                    None,
                    "invalid",
                ],
            },
            {
                "voice_id": "voice-2",
                "high_quality_base_model_ids": ["eleven_turbo_v2"],
                "verified_languages": [
                    {"language": "fr", "model_id": "eleven_turbo_v2"}
                ],
            },
            {
                "voice_id": "voice-3",
                "high_quality_base_model_ids": "eleven_v3",
                "verified_languages": None,
            },
        ],
    )
    tool = AudioToolCore(
        tts_models={
            "eleven_v3": voice_listing_tts,
            "eleven_flash_v2": FakeTTS(provider_name="elevenlabs"),
            "None": FakeTTS(provider_name="elevenlabs"),
            "chat-tts": FakeTTS(provider_name="xinference"),
        }
    )

    result = await tool.list_tts_voices(model_id="eleven_v3")

    assert result["voices"] == [
        {
            "voice_id": "voice-1",
            "high_quality_base_model_ids": ["eleven_v3", "eleven_flash_v2"],
            "verified_languages": [
                {"language": "en", "model_id": "eleven_v3"},
                {"language": "de", "model_id": "eleven_flash_v2"},
            ],
        },
        {"voice_id": "voice-2"},
        {"voice_id": "voice-3"},
    ]


async def test_list_tts_voices_keeps_default_model_metadata() -> None:
    default_tts = FakeTTS(
        provider_name="elevenlabs",
        abilities=["tts", "voice_listing"],
        voices=[
            {
                "voice_id": "voice-1",
                "high_quality_base_model_ids": ["default-model", "other-model"],
                "verified_languages": [
                    {"language": "en", "model_id": "default-model"},
                    {"language": "de", "model_id": "other-model"},
                ],
            }
        ],
    )
    default_tts.model_name = "default-model"
    tool = AudioToolCore(default_tts_model=default_tts)

    result = await tool.list_tts_voices()

    assert result["voices"] == [
        {
            "voice_id": "voice-1",
            "high_quality_base_model_ids": ["default-model"],
            "verified_languages": [{"language": "en", "model_id": "default-model"}],
        }
    ]


def test_filter_voice_model_metadata_handles_empty_configured_set() -> None:
    voices = AudioToolCore._filter_voice_model_metadata(
        [
            {
                "voice_id": "voice-1",
                "high_quality_base_model_ids": ["eleven_v3"],
                "verified_languages": [{"language": "en", "model_id": "eleven_v3"}],
            }
        ],
        configured_model_ids=set(),
    )

    assert voices == [{"voice_id": "voice-1"}]


async def test_list_tts_voices_rejects_model_from_other_provider() -> None:
    tool = AudioToolCore(
        tts_models={
            "voice-model": FakeTTS(
                provider_name="customvoice",
                abilities=["tts", "voice_listing"],
                voices=[{"voice_id": "voice-1"}],
            )
        }
    )

    result = await tool.list_tts_voices(model_id="voice-model")

    assert result["success"] is False
    assert result["supported"] is False
    assert "provider is 'elevenlabs'" in result["error"]


async def test_list_tts_voices_reports_unsupported_provider() -> None:
    tts = FakeTTS(provider_name="xinference", abilities=["tts"])
    tool = AudioToolCore(tts_models={"chat-tts": tts})

    result = await tool.list_tts_voices(model_id="chat-tts")

    assert result["success"] is False
    assert result["supported"] is False
    assert result["provider"] == "xinference"
    assert "provider is 'elevenlabs'" in result["error"]
    assert result["supported_providers"] == []


async def test_list_tts_voices_reports_missing_elevenlabs_model() -> None:
    result = await AudioToolCore().list_tts_voices()

    assert result["success"] is False
    assert result["supported"] is False
    assert result["error"] == "No elevenlabs TTS model is configured"
    assert result["voices"] == []
    assert result["model_used"] == "default"


def test_list_tts_voices_tool_visible_only_for_voice_listing_provider() -> None:
    unsupported_tool = AudioTool(
        tts_models={"chat-tts": FakeTTS(provider_name="xinference", abilities=["tts"])}
    )
    supported_tool = AudioTool(
        tts_models={
            "eleven_v3": FakeTTS(
                provider_name="elevenlabs",
                abilities=["tts", "voice_listing"],
            )
        }
    )

    unsupported_tool_names = {tool.name for tool in unsupported_tool.get_tools()}
    supported_tool_names = {tool.name for tool in supported_tool.get_tools()}

    assert "list_tts_voices" not in unsupported_tool_names
    assert "list_tts_voices" in supported_tool_names


def test_synthesize_speech_schema_exposes_structured_options() -> None:
    tool = AudioTool(tts_models={"fake": FakeTTS()})
    synthesize_tool = next(
        candidate
        for candidate in tool.get_tools()
        if candidate.name == "synthesize_speech"
    )

    schema_properties = synthesize_tool.args_type().model_json_schema()["properties"]

    assert "sample_rate" in schema_properties
    assert "reference_audio" in schema_properties
    assert "voice_settings" in schema_properties
    assert "provider_options" in schema_properties

    assert "Never invent" in synthesize_tool.description
    assert "exact voice_id returned by list_tts_voices" in synthesize_tool.description
    assert '"Claughton": "Cloffton"' in synthesize_tool.description
    assert '"UN": "United Nations"' in synthesize_tool.description
    assert "en-male" not in synthesize_tool.description
    assert "zh-female" not in synthesize_tool.description


def test_transcribe_audio_schema_accepts_workspace_file_path_or_id() -> None:
    tool = AudioTool()
    transcribe_tool = next(
        candidate
        for candidate in tool.get_tools()
        if candidate.name == "transcribe_audio"
    )

    schema = transcribe_tool.args_type().model_json_schema()

    assert "file_path_or_id" in schema["properties"]
    assert "file_path_or_id" in schema["required"]
    assert "audio_file_path" not in schema["properties"]
    assert "exact file_id" in transcribe_tool.description
    assert "only when no workspace file_id" in transcribe_tool.description


async def test_clone_tts_voice_returns_persistent_provider_voice_id() -> None:
    tts = FakeTTS(
        provider_name="elevenlabs",
        abilities=["tts", "voice_cloning", "persistent_voice_cloning"],
    )
    tool = AudioToolCore(tts_models={"eleven_v3": tts})

    result = await tool.clone_tts_voice(
        name="Product narrator",
        reference_audio_files=["first.mp3", "second.wav"],
        provider="elevenlabs",
        description="Narration voice",
        labels={"language": "en"},
        remove_background_noise=True,
        model_id="eleven_v3",
    )

    assert result == {
        "success": True,
        "supported": True,
        "voice_id": "persistent-voice",
        "name": "Product narrator",
        "provider": "elevenlabs",
        "persistent": True,
        "requires_verification": False,
        "model_used": "eleven_v3",
    }
    assert tts.clone_calls == [
        {
            "name": "Product narrator",
            "reference_audio_files": ["first.mp3", "second.wav"],
            "description": "Narration voice",
            "labels": {"language": "en"},
            "remove_background_noise": True,
        }
    ]


async def test_clone_tts_voice_rejects_other_provider_model() -> None:
    elevenlabs = FakeTTS(
        provider_name="elevenlabs",
        abilities=["tts", "persistent_voice_cloning"],
    )
    xinference = FakeTTS(
        provider_name="xinference",
        abilities=["tts", "voice_cloning"],
    )
    tool = AudioToolCore(tts_models={"eleven_v3": elevenlabs, "index-tts": xinference})

    result = await tool.clone_tts_voice(
        name="Wrong provider",
        reference_audio_files=["reference.wav"],
        model_id="index-tts",
    )

    assert result["success"] is False
    assert result["supported"] is False
    assert "provider is 'elevenlabs'" in result["error"]
    assert elevenlabs.clone_calls == []
    assert xinference.clone_calls == []


async def test_clone_tts_voice_reports_missing_elevenlabs_model() -> None:
    result = await AudioToolCore().clone_tts_voice(
        name="Missing provider",
        reference_audio_files=["reference.wav"],
    )

    assert result == {
        "success": False,
        "supported": False,
        "error": "No elevenlabs TTS model is configured",
        "provider": "elevenlabs",
        "model_used": "default",
    }


async def test_clone_tts_voice_selects_provider_not_default_tts() -> None:
    elevenlabs = FakeTTS(
        provider_name="elevenlabs",
        abilities=["tts", "persistent_voice_cloning"],
    )
    xinference = FakeTTS(
        provider_name="xinference",
        abilities=["tts", "voice_cloning"],
    )
    tool = AudioToolCore(
        tts_models={"index-tts": xinference, "eleven_v3": elevenlabs},
        default_tts_model=xinference,
    )

    result = await tool.clone_tts_voice(
        name="ElevenLabs voice",
        reference_audio_files=["reference.wav"],
    )

    assert result["success"] is True
    assert result["provider"] == "elevenlabs"
    assert result["model_used"] == "eleven_v3"
    assert len(elevenlabs.clone_calls) == 1
    assert xinference.clone_calls == []


async def test_delete_tts_voice_deletes_persistent_provider_voice() -> None:
    tts = FakeTTS(
        provider_name="elevenlabs",
        abilities=["tts", "persistent_voice_cloning"],
    )
    tool = AudioToolCore(tts_models={"eleven_v3": tts})

    result = await tool.delete_tts_voice(
        voice_id=" persistent-voice ",
        provider="elevenlabs",
        model_id="eleven_v3",
    )

    assert result == {
        "success": True,
        "supported": True,
        "deleted": True,
        "voice_id": "persistent-voice",
        "provider": "elevenlabs",
        "model_used": "eleven_v3",
    }
    assert tts.delete_calls == ["persistent-voice"]


async def test_delete_tts_voice_rejects_other_provider_model() -> None:
    elevenlabs = FakeTTS(
        provider_name="elevenlabs",
        abilities=["tts", "persistent_voice_cloning"],
    )
    xinference = FakeTTS(
        provider_name="xinference",
        abilities=["tts", "persistent_voice_cloning"],
    )
    tool = AudioToolCore(tts_models={"eleven_v3": elevenlabs, "other": xinference})

    result = await tool.delete_tts_voice(
        voice_id="persistent-voice",
        model_id="other",
    )

    assert result["success"] is False
    assert result["supported"] is False
    assert "provider is 'elevenlabs'" in result["error"]
    assert elevenlabs.delete_calls == []
    assert xinference.delete_calls == []


async def test_delete_tts_voice_rejects_unsupported_model() -> None:
    tts = FakeTTS(provider_name="elevenlabs", abilities=["tts"])
    tool = AudioToolCore(tts_models={"eleven_v3": tts})

    result = await tool.delete_tts_voice(
        voice_id="persistent-voice",
        model_id="eleven_v3",
    )

    assert result == {
        "success": False,
        "supported": False,
        "error": "The configured elevenlabs client does not support persistent voice deletion",
        "provider": "elevenlabs",
        "model_used": "eleven_v3",
    }
    assert tts.delete_calls == []


def test_persistent_voice_tools_expose_provider_enum() -> None:
    elevenlabs_tool = AudioTool(
        tts_models={
            "eleven_v3": FakeTTS(
                provider_name="elevenlabs",
                abilities=["tts", "persistent_voice_cloning"],
            )
        }
    )
    other_provider_tool = AudioTool(
        tts_models={
            "other": FakeTTS(
                provider_name="other",
                abilities=["tts", "persistent_voice_cloning"],
            )
        }
    )

    elevenlabs_tools = {
        candidate.name: candidate for candidate in elevenlabs_tool.get_tools()
    }
    other_tool_names = {candidate.name for candidate in other_provider_tool.get_tools()}

    assert "clone_tts_voice" in elevenlabs_tools
    assert "delete_tts_voice" in elevenlabs_tools
    assert "clone_tts_voice" not in other_tool_names
    assert "delete_tts_voice" not in other_tool_names
    schema = elevenlabs_tools["clone_tts_voice"].args_type().model_json_schema()
    assert set(schema["properties"]) == {
        "name",
        "reference_audio_files",
        "provider",
        "description",
        "labels",
        "remove_background_noise",
        "model_id",
    }
    assert schema["properties"]["provider"]["const"] == "elevenlabs"

    delete_schema = elevenlabs_tools["delete_tts_voice"].args_type().model_json_schema()
    assert set(delete_schema["properties"]) == {"voice_id", "provider", "model_id"}
    assert delete_schema["properties"]["provider"]["const"] == "elevenlabs"


def test_list_tts_voices_tool_exposes_provider_enum() -> None:
    audio_tool = AudioTool(
        tts_models={
            "eleven_v3": FakeTTS(
                provider_name="elevenlabs",
                abilities=["tts", "voice_listing"],
            )
        }
    )
    list_tool = next(
        candidate
        for candidate in audio_tool.get_tools()
        if candidate.name == "list_tts_voices"
    )

    schema = list_tool.args_type().model_json_schema()
    assert schema["properties"]["provider"]["const"] == "elevenlabs"


async def test_synthesize_speech_json_merges_default_and_segment_options() -> None:
    tts = FakeTTS()
    tool = AudioToolCore(tts_models={"fake": tts})

    result = await tool.synthesize_speech_json(
        json_data={
            "segments": [
                {
                    "text": "First line",
                    "voice_settings": {"style": 0.2},
                    "provider_options": {"seed": 1234},
                }
            ],
            "default_voice": "voice-1",
            "default_language": "en",
            "default_voice_settings": {"stability": 0.5},
            "default_provider_options": {"apply_text_normalization": "on"},
            "output_format": "wav",
            "sample_rate": 16000,
        },
        model_id="fake",
    )

    assert result["success"] is True
    assert tts.calls == [
        {
            "text": "First line",
            "voice": "voice-1",
            "language": "en",
            "format": "wav",
            "sample_rate": 16000,
            "voice_settings": {"stability": 0.5, "style": 0.2},
            "apply_text_normalization": "on",
            "seed": 1234,
        }
    ]


async def test_synthesize_speech_json_rejects_non_object_json() -> None:
    tool = AudioToolCore(tts_models={"fake": FakeTTS()})

    result = await tool.synthesize_speech_json(json_data=[])

    assert result == {
        "success": False,
        "error": "JSON data must be an object",
        "results": [],
        "total": 0,
        "successful": 0,
        "failed": 0,
        "errors": ["JSON data must be an object"],
    }
