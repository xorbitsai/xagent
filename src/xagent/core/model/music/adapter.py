"""Adapters for independent music model configurations."""

from __future__ import annotations

from typing import Any

from ..model import MusicModelConfig
from .base import BaseMusicModel
from .elevenlabs import ElevenLabsMusicModel


def create_music_model(config: MusicModelConfig) -> BaseMusicModel:
    provider = config.model_provider.lower().strip()
    if provider != "elevenlabs":
        raise ValueError(f"Unsupported music provider: {config.model_provider}")
    return ElevenLabsMusicModel(
        model_name=config.model_name,
        api_key=config.api_key,
        base_url=config.base_url,
    )


def get_music_model_instance(db_model: Any) -> BaseMusicModel:
    config = MusicModelConfig(
        id=str(db_model.model_id),
        model_name=str(db_model.model_name),
        model_provider=str(db_model.model_provider),
        api_key=db_model.api_key,
        base_url=db_model.base_url,
        abilities=list(db_model.abilities or ["generate"]),
        description=db_model.description,
    )
    model = create_music_model(config)
    setattr(model, "model_id", str(db_model.model_id))
    return model
