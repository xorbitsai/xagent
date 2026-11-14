"""What widget voice runs on: the deployment's voice models.

Widget guests never choose a model. The deployment names one live
transcription model and one synthesis model by ``model_id``; they are resolved
for the entity's owner with the same visibility rules as the owner's own
speech requests, so a model an administrator shared and later unshared or
disabled makes widget voice unavailable rather than reachable.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException

from ...config import (
    get_widget_voice_asr_model,
    get_widget_voice_tts_model,
    get_widget_voice_tts_voice,
)
from ...core.model.asr.adapter import get_asr_model_instance
from ...core.model.tts.adapter import get_tts_model_instance
from ..api.speech import _load_speech_config

logger = logging.getLogger(__name__)

VOICE_MODELS_NOT_CONFIGURED = "voice_models_not_configured"
VOICE_MODELS_UNAVAILABLE = "voice_models_unavailable"

# The voice shape TTSStreamRequest accepts; a deployment voice outside it
# could never be sent, so it makes widget voice unavailable.
_VOICE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
# ElevenLabs live transcription commits a segment after this much silence,
# the value the browser call controller uses for the same provider.
_ELEVENLABS_VAD_SILENCE_SECS = 0.5


@dataclass(frozen=True)
class WidgetVoiceModels:
    """The resolved model configurations widget voice runs on for one owner."""

    asr: SimpleNamespace
    tts: SimpleNamespace
    voice: str | None
    vad_silence_threshold_secs: float | None


@dataclass(frozen=True)
class WidgetVoiceResolution:
    """Either the models (``reason`` None) or why widget voice is unavailable."""

    models: WidgetVoiceModels | None
    reason: str | None = None


def _streams(config: SimpleNamespace, factory: Callable[[Any], Any]) -> bool:
    # Building an instance only reads the configuration; provider clients are
    # opened lazily, so there is nothing to close here.
    try:
        return bool(factory(config).supports_streaming)
    except ValueError:
        return False


def resolve_widget_voice_models(owner_user_id: int) -> WidgetVoiceResolution:
    """Resolve the deployment's voice models as visible to ``owner_user_id``.

    Blocking: it reads the database in short sessions of its own, so call it
    from a worker thread. Missing settings are ``voice_models_not_configured``.
    A model that is missing, not visible to the owner, inactive, of the wrong
    ability, or unable to stream is ``voice_models_unavailable`` and logged
    with its ``model_id``. A database failure is raised, never reported as
    unavailable.
    """
    asr_model_id = get_widget_voice_asr_model()
    tts_model_id = get_widget_voice_tts_model()
    if not asr_model_id or not tts_model_id:
        return WidgetVoiceResolution(None, VOICE_MODELS_NOT_CONFIGURED)
    unavailable = WidgetVoiceResolution(None, VOICE_MODELS_UNAVAILABLE)
    voice = get_widget_voice_tts_voice()
    if voice is not None and not _VOICE_PATTERN.fullmatch(voice):
        logger.warning("Widget voice setting has an unusable synthesis voice")
        return unavailable
    try:
        asr = _load_speech_config(owner_user_id, asr_model_id, "asr")
        tts = _load_speech_config(owner_user_id, tts_model_id, "tts")
    except HTTPException as exc:
        logger.warning(
            "Widget voice models %s / %s unavailable for owner %s: %s",
            asr_model_id,
            tts_model_id,
            owner_user_id,
            exc.detail,
        )
        return unavailable
    for config, factory in (
        (asr, get_asr_model_instance),
        (tts, get_tts_model_instance),
    ):
        if not _streams(config, factory):
            logger.warning("Widget voice model %s cannot stream", config.model_id)
            return unavailable
    provider = str(asr.model_provider or "").strip().lower()
    return WidgetVoiceResolution(
        WidgetVoiceModels(
            asr=asr,
            tts=tts,
            voice=voice,
            vad_silence_threshold_secs=(
                _ELEVENLABS_VAD_SILENCE_SECS if provider == "elevenlabs" else None
            ),
        )
    )
