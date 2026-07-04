"""
Audio processing tool for xagent

This module provides audio processing capabilities including:
- Speech-to-Text (ASR/Automatic Speech Recognition)
- Text-to-Speech (TTS/Speech Synthesis)

Uses pre-configured ASR and TTS models passed from the web layer.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ....model.asr.base import BaseASR
from ....model.tts.base import BaseTTS
from ....workspace import TaskWorkspace
from ...core.audio_tool import AudioToolCore
from .base import ToolCategory
from .function import FunctionTool

logger = logging.getLogger(__name__)


class AudioFunctionTool(FunctionTool):
    """AudioFunctionTool with ToolCategory.AUDIO category."""

    category = ToolCategory.AUDIO

    def __init__(
        self,
        *args: Any,
        owner: Optional[AudioToolCore] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._owner = owner

    async def teardown(self, task_id: Optional[str] = None) -> None:
        if self._owner is not None:
            await self._owner.teardown(task_id=task_id)


class AudioTool(AudioToolCore):
    """
    Audio processing tool that uses pre-configured ASR and TTS models.
    """

    def __init__(
        self,
        asr_models: Optional[Dict[str, BaseASR]] = None,
        tts_models: Optional[Dict[str, BaseTTS]] = None,
        model_descriptions: Optional[Dict[str, str]] = None,
        workspace: Optional[TaskWorkspace] = None,
        default_asr_model: Optional[BaseASR] = None,
        default_tts_model: Optional[BaseTTS] = None,
    ):
        """
        Initialize with pre-configured ASR and TTS models.

        Args:
            asr_models: Dictionary mapping model_id to BaseASR instances
            tts_models: Dictionary mapping model_id to BaseTTS instances
            model_descriptions: Dictionary mapping model_id to description strings
            workspace: Optional workspace for saving generated audio files
            default_asr_model: Default model for speech recognition
            default_tts_model: Default model for speech synthesis
        """
        # Call parent class initialization first
        super().__init__(
            asr_models,
            tts_models,
            model_descriptions,
            workspace,
            default_asr_model,
            default_tts_model,
        )

    def get_tools(self) -> list:
        """Get all tool instances."""
        # Format descriptions with model information
        transcribe_description = self.TRANSCRIBE_AUDIO_DESCRIPTION.format(
            self._asr_model_info_text
        )
        synthesize_description = self.SYNTHESIZE_SPEECH_DESCRIPTION.format(
            self._tts_model_info_text
        )

        # Add batch JSON tool
        json_description = self.SYNTHESIZE_SPEECH_JSON_DESCRIPTION.format(
            self._tts_model_info_text
        )
        tools = [
            AudioFunctionTool(
                self.transcribe_audio,
                name="transcribe_audio",
                description=transcribe_description,
                owner=self,
            ),
            AudioFunctionTool(
                self.synthesize_speech,
                name="synthesize_speech",
                description=synthesize_description,
                owner=self,
            ),
            AudioFunctionTool(
                self.synthesize_speech_json,
                name="synthesize_speech_json",
                description=json_description,
                owner=self,
            ),
            AudioFunctionTool(
                self.list_available_models,
                name="list_audio_models",
                description="List all available audio models (ASR and TTS), including model ID, availability status, and detailed description information (Note: model information is already provided in the transcribe_audio and synthesize_speech tool descriptions)",
                owner=self,
            ),
        ]

        voice_listing_models = list(self._tts_models.values())
        if self._default_tts_model is not None:
            voice_listing_models.append(self._default_tts_model)
        if any(
            getattr(model, "supports_voice_listing", False)
            for model in voice_listing_models
        ):
            list_tts_voices_description = self.LIST_TTS_VOICES_DESCRIPTION.format(
                ", ".join(self._get_voice_listing_supported_providers())
            )
            tools.append(
                AudioFunctionTool(
                    self.list_tts_voices,
                    name="list_tts_voices",
                    description=list_tts_voices_description,
                    owner=self,
                )
            )

        return tools


def create_audio_tool(
    asr_models: Optional[Dict[str, BaseASR]] = None,
    tts_models: Optional[Dict[str, BaseTTS]] = None,
    model_descriptions: Optional[Dict[str, str]] = None,
    workspace: Optional[TaskWorkspace] = None,
    default_asr_model: Optional[BaseASR] = None,
    default_tts_model: Optional[BaseTTS] = None,
) -> list:
    """
    Create audio processing tools with pre-configured models.

    Args:
        asr_models: Dictionary mapping model_id to BaseASR instances
        tts_models: Dictionary mapping model_id to BaseTTS instances
        model_descriptions: Dictionary mapping model_id to description strings
        workspace: Optional workspace for saving generated audio files
        default_asr_model: Default model for speech recognition
        default_tts_model: Default model for speech synthesis

    Returns:
        List of tool instances
    """
    tool_instance = AudioTool(
        asr_models,
        tts_models,
        model_descriptions,
        workspace,
        default_asr_model,
        default_tts_model,
    )
    return tool_instance.get_tools()


# Register tool creator for auto-discovery
# Import at bottom to avoid circular import with factory
from .factory import ToolFactory, register_tool  # noqa: E402

if TYPE_CHECKING:
    from .config import BaseToolConfig


@register_tool(categories={"audio"})
async def create_audio_tools_from_config(config: "BaseToolConfig") -> List[Any]:
    """Create audio processing tools from configuration.

    Internal short-circuit on ``ToolSelectionSpec.includes_category("audio")``
    skips the ASR/TTS-model DB lookup
    (``config.get_asr_models()`` / ``get_tts_models()``) when the spec
    excludes the audio category. Registry-level skip via
    ``categories={"audio"}`` handles the common case; the internal
    check covers the legacy spec=None backward-compat path.
    """
    spec = (
        config.get_tool_selection_spec()
        if hasattr(config, "get_tool_selection_spec")
        else None
    )
    if spec is not None and not spec.includes_category("audio"):
        return []
    asr_models = config.get_asr_models()
    tts_models = config.get_tts_models()

    if not asr_models and not tts_models:
        return []

    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        return create_audio_tool(
            asr_models=asr_models,
            tts_models=tts_models,
            workspace=workspace,
            default_asr_model=config.get_asr_model(),
            default_tts_model=config.get_tts_model(),
        )
    except Exception as e:
        logger.warning(f"Failed to create audio tools: {e}")
        return []
