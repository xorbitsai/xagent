"""Tests for the music generation tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from xagent.core.model.music import BaseMusicModel, MusicResult
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.music_tool import (
    create_music_tools,
    create_music_tools_from_config,
)
from xagent.core.tools.core.music_tool import MusicToolCore
from xagent.core.workspace import TaskWorkspace


class FakeMusicModel(BaseMusicModel):
    provider_name = "fake-music"

    def __init__(self, model_name: str = "music-v2") -> None:
        self.model_name = model_name
        self.calls: list[dict[str, Any]] = []
        self.close_count = 0

    async def generate_music(
        self,
        prompt: str,
        music_length_seconds: Optional[float] = None,
        force_instrumental: bool = False,
        output_format: str = "auto",
    ) -> MusicResult:
        self.calls.append(
            {
                "prompt": prompt,
                "music_length_seconds": music_length_seconds,
                "force_instrumental": force_instrumental,
                "output_format": output_format,
            }
        )
        return MusicResult(audio=b"fake-music", format="mp3", sample_rate=48000)

    async def aclose(self) -> None:
        self.close_count += 1


async def test_generate_music_saves_registered_workspace_file(tmp_path: Path) -> None:
    model = FakeMusicModel()
    workspace = TaskWorkspace("music-task", str(tmp_path))
    tool = MusicToolCore(models={"configured-music": model}, workspace=workspace)

    result = await tool.generate_music(
        prompt="Cinematic ambient score",
        music_length_seconds=20,
        force_instrumental=True,
    )

    assert result["success"] is True
    assert result["model_used"] == "configured-music"
    assert result["provider_model"] == "music-v2"
    assert result["file_id"] == result["file_ref"]["file_id"]
    assert Path(result["audio_path"]).read_bytes() == b"fake-music"
    assert model.calls == [
        {
            "prompt": "Cinematic ambient score",
            "music_length_seconds": 20,
            "force_instrumental": True,
            "output_format": "auto",
        }
    ]


def test_music_tool_uses_audio_category(tmp_path: Path) -> None:
    workspace = TaskWorkspace("music-schema", str(tmp_path))
    [tool] = create_music_tools(models={"music": FakeMusicModel()}, workspace=workspace)

    assert tool.metadata.category == ToolCategory.AUDIO
    assert tool.name == "generate_music"
    assert set(tool.args_type().model_json_schema()["properties"]) == {
        "prompt",
        "music_length_seconds",
        "force_instrumental",
        "output_format",
        "model_id",
    }


async def test_registered_creator_reads_music_config(tmp_path: Path) -> None:
    model = FakeMusicModel()
    config = ToolConfig(
        {
            "workspace": {"task_id": "music-factory", "base_dir": str(tmp_path)},
            "music_models": {"music": model},
            "music_model": model,
        }
    )

    tools = await create_music_tools_from_config(config)

    assert [tool.name for tool in tools] == ["generate_music"]
    assert tools[0].metadata.category == ToolCategory.AUDIO
