"""What widget voice runs on: the deployment's voice models.

Widget guests never choose a model. The deployment names a live transcription
model and a synthesis model by ``model_id``; they are resolved for the
entity's owner with the same visibility rules as the owner's own speech
requests, so a model an administrator shared and later unshared or disabled
makes widget voice unavailable rather than reachable. A request resolves, and
asks the provider about, only the model it runs on; whether widget voice is
offered at all needs both.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

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

# How long the provider has to say whether a model streams. A provider that
# does not answer in time makes widget voice unavailable, so one that hangs
# cannot hold a guest's request open.
_CAPABILITY_CHECK_TIMEOUT_SECONDS = 3.0

# Which of the deployment's models a lookup is about.
_Ability = Literal["asr", "tts"]


@dataclass(frozen=True)
class WidgetVoiceModels:
    """The resolved synthesis configuration widget voice runs on for one owner."""

    tts: SimpleNamespace
    voice: str | None


@dataclass(frozen=True)
class WidgetTranscriptionModel:
    """The resolved transcription configuration for one owner."""

    asr: SimpleNamespace
    vad_silence_threshold_secs: float | None


@dataclass(frozen=True)
class WidgetVoiceAvailability:
    """Whether widget voice is offered (both models usable), and if not, why."""

    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class _ConfiguredModels:
    """The requested models' configurations, read but not yet asked about."""

    configs: dict[_Ability, SimpleNamespace]
    voice: str | None


def _model_id(ability: _Ability) -> str | None:
    if ability == "asr":
        return get_widget_voice_asr_model()
    return get_widget_voice_tts_model()


def _model_instance(ability: _Ability) -> Callable[[Any], Any]:
    return get_asr_model_instance if ability == "asr" else get_tts_model_instance


def _read_configured_models(
    owner_user_id: int, abilities: tuple[_Ability, ...]
) -> _ConfiguredModels | str:
    """The configurations of ``abilities``, or the reason there are none."""
    # Blocking: runs in a worker thread, in short sessions of its own.
    model_ids = {ability: _model_id(ability) for ability in abilities}
    if not all(model_ids.values()):
        return VOICE_MODELS_NOT_CONFIGURED
    voice = None
    if "tts" in abilities:
        voice = get_widget_voice_tts_voice()
        if voice is not None and not _VOICE_PATTERN.fullmatch(voice):
            logger.warning("Widget voice setting has an unusable synthesis voice")
            return VOICE_MODELS_UNAVAILABLE
    configs: dict[_Ability, SimpleNamespace] = {}
    for ability, model_id in model_ids.items():
        assert model_id is not None
        try:
            configs[ability] = _load_speech_config(owner_user_id, model_id, ability)
        except HTTPException as exc:
            logger.warning(
                "Widget voice model %s unavailable for owner %s: %s",
                model_id,
                owner_user_id,
                exc.detail,
            )
            return VOICE_MODELS_UNAVAILABLE
    return _ConfiguredModels(configs=configs, voice=voice)


async def _streams(ability: _Ability, config: SimpleNamespace) -> bool:
    """Ask the provider whether ``config``'s model streams.

    Some providers only know after this call, so the adapter's
    ``supports_streaming`` property is not read. Only the detached
    configuration is passed in: no database session is held or opened here.
    An answer that takes longer than ``_CAPABILITY_CHECK_TIMEOUT_SECONDS``
    counts as a failed check. The probe instance is closed however the call
    ends.
    """
    try:
        model = _model_instance(ability)(config)
    except ValueError:
        logger.warning(
            "Widget voice model %s cannot be built from its configuration",
            config.model_id,
        )
        return False
    try:
        try:
            async with asyncio.timeout(_CAPABILITY_CHECK_TIMEOUT_SECONDS):
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


async def _resolve(
    owner_user_id: int,
    request_db: Session | None,
    abilities: tuple[_Ability, ...],
) -> _ConfiguredModels | str:
    """The usable configurations of ``abilities`` for ``owner_user_id``, or the
    reason one is not.

    ``request_db`` is the caller's request session, or ``None`` when the caller
    has none. Its connection is returned to the pool before anything else,
    because the lookup takes connections from the same pool; a session that
    cannot return it (it has pending ORM changes, or its open transaction ran
    a statement that is not a plain SELECT) raises ``RuntimeError``.

    The settings are read in a worker thread, in short sessions of its own.
    Then, with no database session held, the provider is asked once about
    each requested model, and each probe instance is closed. A missing model
    setting is ``voice_models_not_configured``. A configured synthesis voice
    that is unusable, or a model that is missing, not
    visible to the owner, inactive, of the wrong ability, cannot be built,
    whose capability check raises or times out, or that cannot stream is
    ``voice_models_unavailable`` and logged with its ``model_id``. A database
    failure is raised, never reported as unavailable.
    """
    if request_db is not None:
        release_request_connection(request_db)
    read = await run_db_io_cancellation_safe(
        lambda: _read_configured_models(owner_user_id, abilities)
    )
    if isinstance(read, str):
        return read
    for ability in abilities:
        if not await _streams(ability, read.configs[ability]):
            return VOICE_MODELS_UNAVAILABLE
    return read


async def resolve_widget_voice_models(
    owner_user_id: int, *, request_db: Session | None
) -> WidgetVoiceModels | None:
    """Resolve the deployment's synthesis model as visible to ``owner_user_id``,
    or ``None`` when it is unavailable.

    Only the synthesis model is looked up and asked about; see
    :func:`_resolve` for the connection handling and the failure cases.
    """
    read = await _resolve(owner_user_id, request_db, ("tts",))
    if isinstance(read, str):
        return None
    return WidgetVoiceModels(tts=read.configs["tts"], voice=read.voice)


async def resolve_widget_transcription_model(
    owner_user_id: int, *, request_db: Session | None
) -> WidgetTranscriptionModel | None:
    """Resolve the deployment's transcription model as visible to
    ``owner_user_id``, or ``None`` when it is unavailable.

    Only the transcription model is looked up and asked about; see
    :func:`_resolve` for the connection handling and the failure cases.
    """
    read = await _resolve(owner_user_id, request_db, ("asr",))
    if isinstance(read, str):
        return None
    asr = read.configs["asr"]
    provider = str(asr.model_provider or "").strip().lower()
    return WidgetTranscriptionModel(
        asr=asr,
        vad_silence_threshold_secs=(
            _ELEVENLABS_VAD_SILENCE_SECS if provider == "elevenlabs" else None
        ),
    )


async def resolve_widget_voice_availability(
    owner_user_id: int, *, request_db: Session | None
) -> WidgetVoiceAvailability:
    """Whether widget voice is offered for ``owner_user_id``: both the
    transcription and the synthesis model resolve.

    Each model is asked about once, and the second is not asked about when the
    first already failed. See :func:`_resolve` for the connection handling and
    the failure reasons.
    """
    read = await _resolve(owner_user_id, request_db, ("asr", "tts"))
    if isinstance(read, str):
        return WidgetVoiceAvailability(False, read)
    return WidgetVoiceAvailability(True)
