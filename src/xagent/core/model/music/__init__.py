"""Independent music generation model category."""

from .adapter import create_music_model, get_music_model_instance
from .base import BaseMusicModel, MusicResult
from .elevenlabs import ElevenLabsMusicModel

__all__ = [
    "BaseMusicModel",
    "ElevenLabsMusicModel",
    "MusicResult",
    "create_music_model",
    "get_music_model_instance",
]
