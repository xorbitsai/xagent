"""Vibe adapter for music generation tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ....model.music import BaseMusicModel
from ....workspace import TaskWorkspace
from ...core.music_tool import MusicToolCore
from .base import ToolCategory
from .function import FunctionTool

logger = logging.getLogger(__name__)


class MusicFunctionTool(FunctionTool):
    category = ToolCategory.AUDIO

    def __init__(
        self,
        *args: Any,
        owner: Optional[MusicToolCore] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._owner = owner

    async def teardown(self, task_id: Optional[str] = None) -> None:
        if self._owner is not None:
            await self._owner.teardown(task_id=task_id)


def create_music_tools(
    models: dict[str, BaseMusicModel],
    workspace: TaskWorkspace,
    default_model: Optional[BaseMusicModel] = None,
) -> list[MusicFunctionTool]:
    owner = MusicToolCore(
        models=models,
        workspace=workspace,
        default_model=default_model,
    )
    return [
        MusicFunctionTool(
            owner.generate_music,
            name="generate_music",
            description=owner.GENERATE_MUSIC_DESCRIPTION.format(
                owner.model_info_text()
            ),
            owner=owner,
        )
    ]


from .factory import ToolFactory, register_tool  # noqa: E402

if TYPE_CHECKING:
    from .config import BaseToolConfig


@register_tool(categories={"audio"})
async def create_music_tools_from_config(
    config: "BaseToolConfig",
) -> list[Any]:
    models = config.get_music_models()
    if not models:
        return []
    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if workspace is None:
        return []
    try:
        return create_music_tools(
            models=models,
            workspace=workspace,
            default_model=config.get_music_model(),
        )
    except Exception as exc:
        logger.warning("Failed to create music tools: %s", exc)
        return []
