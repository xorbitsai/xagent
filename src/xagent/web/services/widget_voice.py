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
from sqlalchemy.orm import Session

from ...config import (
    get_widget_voice_asr_model,
    get_widget_voice_tts_model,
    get_widget_voice_tts_voice,
)
from ...core.model.asr.adapter import get_asr_model_instance
from ...core.model.tts.adapter import get_tts_model_instance
from ..api.speech import _load_speech_config
from ..models.database import release_db_connection_if_clean
from .db_runtime import run_db_io_cancellation_safe
from .speech_streams import _close_model

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


@dataclass(frozen=True)
class _ConfiguredModels:
    """The deployment's two model configurations, read but not yet asked about."""

    asr: SimpleNamespace
    tts: SimpleNamespace
    voice: str | None


def _read_configured_models(
    owner_user_id: int,
) -> _ConfiguredModels | WidgetVoiceResolution:
    # Blocking: runs in a worker thread, in short sessions of its own.
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
    return _ConfiguredModels(asr=asr, tts=tts, voice=voice)


async def _streams(config: SimpleNamespace, factory: Callable[[Any], Any]) -> bool:
    """Ask the provider whether ``config``'s model streams.

    Some providers only know after this call, so the adapter's
    ``supports_streaming`` property is not read. Only the detached
    configuration is passed in: no database session is held or opened here.
    The probe instance is closed however the call ends.
    """
    try:
        model = factory(config)
    except ValueError:
        logger.warning(
            "Widget voice model %s cannot be built from its configuration",
            config.model_id,
        )
        return False
    try:
        try:
            capabilities = await model.get_streaming_capabilities()
        except Exception as exc:
            logger.warning(
                "Widget voice model %s capability check failed: %s",
                config.model_id,
                type(exc).__name__,
            )
            return False
        if not capabilities.get("streaming"):
            logger.warning("Widget voice model %s cannot stream", config.model_id)
            return False
        return True
    finally:
        await _close_model(model)


def release_request_connection(request_db: Session) -> None:
    """Return ``request_db``'s connection to the pool, or raise ``RuntimeError``.

    The connection cannot be returned when the session has pending ORM
    changes, or its open transaction ran a statement that is not a plain
    SELECT: returning it would discard that work.
    """
    if not release_db_connection_if_clean(request_db):
        raise RuntimeError(
            "Widget voice lookup needs the caller's database connection released"
        )


async def resolve_widget_voice_models(
    owner_user_id: int, *, request_db: Session | None
) -> WidgetVoiceResolution:
    """Resolve the deployment's voice models as visible to ``owner_user_id``.

    ``request_db`` is the caller's request session, or ``None`` when the caller
    has none. Its connection is returned to the pool before anything else,
    because the lookup takes connections from the same pool; a session that
    cannot return it (it has pending ORM changes, or its open transaction ran
    a statement that is not a plain SELECT) raises ``RuntimeError``.

    The model settings are read in a worker thread, in short sessions of its
    own. Then, with no database session held, the provider is asked whether
    each model streams, and each probe instance is closed. Missing settings
    are ``voice_models_not_configured``. A model that is missing, not visible
    to the owner, inactive, of the wrong ability, cannot be built, whose
    capability check raises or times out, or that cannot stream is
    ``voice_models_unavailable`` and logged with its ``model_id``. A database
    failure is raised, never reported as unavailable.
    """
    if request_db is not None:
        release_request_connection(request_db)
    read = await run_db_io_cancellation_safe(
        lambda: _read_configured_models(owner_user_id)
    )
    if isinstance(read, WidgetVoiceResolution):
        return read
    for config, factory in (
        (read.asr, get_asr_model_instance),
        (read.tts, get_tts_model_instance),
    ):
        if not await _streams(config, factory):
            return WidgetVoiceResolution(None, VOICE_MODELS_UNAVAILABLE)
    provider = str(read.asr.model_provider or "").strip().lower()
    return WidgetVoiceResolution(
        WidgetVoiceModels(
            asr=read.asr,
            tts=read.tts,
            voice=read.voice,
            vad_silence_threshold_secs=(
                _ELEVENLABS_VAD_SILENCE_SECS if provider == "elevenlabs" else None
            ),
        )
    )
