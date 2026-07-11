"""Adapters for independent music model configurations."""

from __future__ import annotations

from typing import Any

from ..model import MusicModelConfig
from ..providers import canonical_provider_name
from .base import BaseMusicModel
from .elevenlabs import ElevenLabsMusicModel


def create_music_model(config: MusicModelConfig) -> BaseMusicModel:
    provider = canonical_provider_name(config.model_provider)
    if provider != "elevenlabs":
        raise ValueError(f"Unsupported music provider: {config.model_provider}")
    return ElevenLabsMusicModel(
        model_name=config.model_name,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def get_music_model_instance(db_model: Any) -> BaseMusicModel:
    config = MusicModelConfig(
        id=str(db_model.model_id),
        model_name=str(db_model.model_name),
        model_provider=str(db_model.model_provider),
        api_key=db_model.api_key,
        base_url=db_model.base_url,
        timeout=getattr(db_model, "timeout", 180.0) or 180.0,
        max_retries=getattr(db_model, "max_retries", 10) or 10,
    )
    model = create_music_model(config)
    setattr(model, "model_id", str(db_model.model_id))
    return model
