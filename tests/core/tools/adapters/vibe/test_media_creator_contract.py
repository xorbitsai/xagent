import pytest

from xagent.core.tools.adapters.vibe.audio_tool import create_audio_tools_from_config
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.image_tool import create_image_tools_from_config
from xagent.core.tools.adapters.vibe.music_tool import create_music_tools_from_config
from xagent.core.tools.adapters.vibe.sound_effect_tool import (
    create_sound_effect_tools_from_config,
)
from xagent.core.tools.adapters.vibe.video_tool import create_video_tools_from_config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "creator,getter",
    [
        (create_image_tools_from_config, "get_image_models"),
        (create_video_tools_from_config, "get_video_models"),
        (create_audio_tools_from_config, "get_asr_models"),
        (create_music_tools_from_config, "get_music_models"),
        (create_sound_effect_tools_from_config, "get_sound_effect_models"),
    ],
)
async def test_media_creators_do_not_swallow_failures(
    creator, getter, tmp_path
) -> None:
    # ToolFactory.create_registered_tools is the single enforcement point for the
    # creator contract; a blanket handler in any creator hides the two exception
    # types the factory promises to re-raise.
    class _Config(ToolConfig):
        def get_workspace_config(self):
            return {"task_id": "creator-contract", "base_dir": str(tmp_path)}

    setattr(_Config, getter, lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        await creator(_Config({}))
