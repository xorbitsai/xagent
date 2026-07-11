"""Vibe adapter for sound effect generation tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ....model.sound_effect import BaseSoundEffectModel
from ....workspace import TaskWorkspace
from ...core.sound_effect_tool import SoundEffectToolCore
from .base import ToolCategory
from .function import FunctionTool

logger = logging.getLogger(__name__)


class SoundEffectFunctionTool(FunctionTool):
    category = ToolCategory.AUDIO

    def __init__(
        self,
        *args: Any,
        owner: Optional[SoundEffectToolCore] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._owner = owner

    async def teardown(self, task_id: Optional[str] = None) -> None:
        if self._owner is not None:
            await self._owner.teardown(task_id=task_id)


def create_sound_effect_tools(
    models: dict[str, BaseSoundEffectModel],
    workspace: TaskWorkspace,
    default_model: Optional[BaseSoundEffectModel] = None,
) -> list[SoundEffectFunctionTool]:
    owner = SoundEffectToolCore(
        models=models,
        workspace=workspace,
        default_model=default_model,
    )
    return [
        SoundEffectFunctionTool(
            owner.generate_sound_effect,
            name="generate_sound_effect",
            description=owner.GENERATE_SOUND_EFFECT_DESCRIPTION.format(
                owner.model_info_text()
            ),
            owner=owner,
        )
    ]


from .factory import ToolFactory, register_tool  # noqa: E402

if TYPE_CHECKING:
    from .config import BaseToolConfig


@register_tool(categories={"audio"})
async def create_sound_effect_tools_from_config(
    config: "BaseToolConfig",
) -> list[Any]:
    models = config.get_sound_effect_models()
    if not models:
        return []
    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if workspace is None:
        return []
    try:
        return create_sound_effect_tools(
            models=models,
            workspace=workspace,
            default_model=config.get_sound_effect_model(),
        )
    except Exception as exc:
        logger.warning("Failed to create sound effect tools: %s", exc)
        return []
