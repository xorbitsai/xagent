"""Sound effect model factory."""

from __future__ import annotations

from typing import Any

from ..model import SoundEffectModelConfig
from ..providers import canonical_provider_name
from .base import BaseSoundEffectModel
from .elevenlabs import ElevenLabsSoundEffectModel


def create_sound_effect_model(
    model_config: SoundEffectModelConfig,
) -> BaseSoundEffectModel:
    provider = canonical_provider_name(model_config.model_provider)
    if provider != "elevenlabs":
        raise ValueError(f"Unsupported sound effect model provider: {provider}")
    return ElevenLabsSoundEffectModel(
        model_name=model_config.model_name,
        api_key=model_config.api_key,
        base_url=model_config.base_url,
        timeout=model_config.timeout,
        max_retries=model_config.max_retries,
    )


def get_sound_effect_model_instance(db_model: Any) -> BaseSoundEffectModel:
    config = SoundEffectModelConfig(
        id=str(db_model.model_id),
        model_name=str(db_model.model_name),
        model_provider=str(db_model.model_provider),
        api_key=str(db_model.api_key) if db_model.api_key else None,
        base_url=str(db_model.base_url) if db_model.base_url else None,
        timeout=getattr(db_model, "timeout", 180.0) or 180.0,
        max_retries=getattr(db_model, "max_retries", 10) or 10,
    )
    model = create_sound_effect_model(config)
    setattr(model, "model_id", str(db_model.model_id))
    return model
