"""Tests for the standalone sound effect tool category."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from xagent.core.model.sound_effect import BaseSoundEffectModel, SoundEffectResult
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.sound_effect_tool import (
    create_sound_effect_tools,
    create_sound_effect_tools_from_config,
)
from xagent.core.tools.core.sound_effect_tool import SoundEffectToolCore
from xagent.core.workspace import TaskWorkspace


class FakeSoundEffectModel(BaseSoundEffectModel):
    provider_name = "fake-sound"

    def __init__(self, model_name: str = "sound-v2") -> None:
        self.model_name = model_name
        self.calls: list[dict[str, Any]] = []
        self.close_count = 0

    async def generate_sound_effect(
        self,
        text: str,
        duration_seconds: Optional[float] = None,
        prompt_influence: float = 0.3,
        loop: bool = False,
        output_format: str = "mp3_44100_128",
    ) -> SoundEffectResult:
        self.calls.append(
            {
                "text": text,
                "duration_seconds": duration_seconds,
                "prompt_influence": prompt_influence,
                "loop": loop,
                "output_format": output_format,
            }
        )
        return SoundEffectResult(
            audio=b"fake-sound-effect",
            format="mp3",
            sample_rate=44100,
        )

    async def aclose(self) -> None:
        self.close_count += 1


class FailingSoundEffectModel(FakeSoundEffectModel):
    async def generate_sound_effect(
        self,
        text: str,
        duration_seconds: Optional[float] = None,
        prompt_influence: float = 0.3,
        loop: bool = False,
        output_format: str = "mp3_44100_128",
    ) -> SoundEffectResult:
        raise RuntimeError("sound effect provider failed")


class EmptySoundEffectModel(FakeSoundEffectModel):
    async def generate_sound_effect(
        self,
        text: str,
        duration_seconds: Optional[float] = None,
        prompt_influence: float = 0.3,
        loop: bool = False,
        output_format: str = "mp3_44100_128",
    ) -> SoundEffectResult:
        return SoundEffectResult(audio=b"", format="mp3")


async def test_generate_sound_effect_saves_registered_workspace_file(
    tmp_path: Path,
) -> None:
    model = FakeSoundEffectModel()
    workspace = TaskWorkspace("sound-effect-task", str(tmp_path))
    tool = SoundEffectToolCore(
        models={"configured-sfx": model},
        workspace=workspace,
    )

    result = await tool.generate_sound_effect(
        text="Heavy wooden door slamming",
        duration_seconds=2.5,
        prompt_influence=0.8,
    )

    assert result["success"] is True
    assert result["model_used"] == "configured-sfx"
    assert result["provider_model"] == "sound-v2"
    assert result["provider"] == "fake-sound"
    assert result["file_id"] == result["file_ref"]["file_id"]
    assert Path(result["audio_path"]).read_bytes() == b"fake-sound-effect"
    assert model.calls == [
        {
            "text": (
                "Heavy wooden door slamming. Non-verbal sound effect only; "
                "no intelligible speech, narration, or spoken words."
            ),
            "duration_seconds": 2.5,
            "prompt_influence": 0.8,
            "loop": False,
            "output_format": "mp3_44100_128",
        }
    ]


async def test_generate_sound_effect_selects_configured_model() -> None:
    first = FakeSoundEffectModel("first")
    selected = FakeSoundEffectModel("selected")
    tool = SoundEffectToolCore(models={"first": first, "selected": selected})

    result = await tool.generate_sound_effect(text="Rain", model_id="selected")

    assert result["success"] is True
    assert result["model_used"] == "selected"
    assert first.calls == []
    assert len(selected.calls) == 1


async def test_generate_sound_effect_rejects_unknown_model() -> None:
    tool = SoundEffectToolCore(models={"known": FakeSoundEffectModel()})

    result = await tool.generate_sound_effect(text="Rain", model_id="unknown")

    assert result["success"] is False
    assert "is not configured" in result["error"]


async def test_generate_sound_effect_rejects_empty_description() -> None:
    model = FakeSoundEffectModel()
    tool = SoundEffectToolCore(models={"sound": model})

    result = await tool.generate_sound_effect(text="  ")

    assert result["success"] is False
    assert result["error"] == "Sound effect description must not be empty"
    assert model.calls == []


async def test_generate_sound_effect_returns_provider_failure() -> None:
    tool = SoundEffectToolCore(models={"sound": FailingSoundEffectModel()})

    result = await tool.generate_sound_effect(text="Rain")

    assert result["success"] is False
    assert result["error"] == "sound effect provider failed"


async def test_generate_sound_effect_rejects_empty_audio() -> None:
    tool = SoundEffectToolCore(models={"sound": EmptySoundEffectModel()})

    result = await tool.generate_sound_effect(text="Rain")

    assert result["success"] is False
    assert result["error"] == "Sound effect model returned no audio data"


async def test_generate_sound_effect_accepts_cjk_and_adds_non_speech_constraint() -> (
    None
):
    model = FakeSoundEffectModel()
    tool = SoundEffectToolCore(models={"sound": model})

    result = await tool.generate_sound_effect(text="台风过境的音效")

    assert result["success"] is True
    assert model.calls[0]["text"] == (
        "台风过境的音效. Non-verbal sound effect only; no intelligible speech, "
        "narration, or spoken words."
    )


async def test_teardown_closes_unique_models_once_per_task() -> None:
    model = FakeSoundEffectModel()
    tool = SoundEffectToolCore(models={"sound": model}, default_model=model)

    await tool.teardown(task_id="task-1")
    await tool.teardown(task_id="task-1")

    assert model.close_count == 1


def test_tool_has_independent_category_and_schema(tmp_path: Path) -> None:
    workspace = TaskWorkspace("sound-effect-schema", str(tmp_path))
    [tool] = create_sound_effect_tools(
        models={"sound": FakeSoundEffectModel()},
        workspace=workspace,
    )

    assert tool.metadata.category == ToolCategory.AUDIO
    assert tool.name == "generate_sound_effect"
    assert set(tool.args_type().model_json_schema()["properties"]) == {
        "text",
        "duration_seconds",
        "prompt_influence",
        "loop",
        "output_format",
        "model_id",
    }


async def test_registered_creator_reads_only_sound_effect_config(
    tmp_path: Path,
) -> None:
    model = FakeSoundEffectModel()
    config = ToolConfig(
        {
            "workspace": {
                "task_id": "sound-effect-factory",
                "base_dir": str(tmp_path),
            },
            "sound_effect_models": {"sound": model},
            "sound_effect_model": model,
        }
    )

    tools = await create_sound_effect_tools_from_config(config)

    assert [tool.name for tool in tools] == ["generate_sound_effect"]
    assert tools[0].metadata.category == ToolCategory.AUDIO


async def test_registered_creator_skips_empty_sound_effect_config() -> None:
    tools = await create_sound_effect_tools_from_config(ToolConfig({}))

    assert tools == []
