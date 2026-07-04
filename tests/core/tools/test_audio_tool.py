"""Tests for audio tool TTS provider options and voice listing."""

from __future__ import annotations

from typing import Any, Optional, Union

from xagent.core.model.tts.base import BaseTTS, TTSResult
from xagent.core.tools.adapters.vibe.audio_tool import AudioTool
from xagent.core.tools.core.audio_tool import AudioToolCore


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


class ClosableTTS(FakeTTS):
    def __init__(self) -> None:
        super().__init__()
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


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


async def test_list_tts_voices_reports_dynamic_supported_providers() -> None:
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

    assert result["success"] is True
    assert result["supported_providers"] == ["customvoice"]


async def test_list_tts_voices_reports_unsupported_provider() -> None:
    tts = FakeTTS(provider_name="xinference", abilities=["tts"])
    tool = AudioToolCore(tts_models={"chat-tts": tts})

    result = await tool.list_tts_voices(model_id="chat-tts")

    assert result["success"] is False
    assert result["supported"] is False
    assert result["provider"] == "xinference"
    assert "Currently supported providers:" in result["error"]
    assert result["supported_providers"] == []


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
