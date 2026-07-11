"""Base interfaces for sound effect generation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SoundEffectResult:
    """Generated sound effect audio and metadata."""

    audio: bytes
    format: str
    sample_rate: Optional[int] = None
    raw_response: Optional[dict[str, Any]] = None


class BaseSoundEffectModel(ABC):
    """Provider-independent sound effect generation interface."""

    provider_name = "unknown"

    @property
    def abilities(self) -> list[str]:
        """Capabilities supported by the model."""
        return ["generate"]

    async def validate_connection(self) -> None:
        """Validate provider credentials without generating billed audio."""
        return None

    @abstractmethod
    async def generate_sound_effect(
        self,
        text: str,
        duration_seconds: Optional[float] = None,
        prompt_influence: float = 0.3,
        loop: bool = False,
        output_format: str = "mp3_44100_128",
    ) -> SoundEffectResult:
        """Generate a sound effect from a text description."""

    async def aclose(self) -> None:
        """Close provider resources when the implementation owns any."""
