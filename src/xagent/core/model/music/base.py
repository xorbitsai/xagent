"""Base abstractions for music generation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MusicResult:
    """Normalized music generation result."""

    audio: bytes
    format: str
    sample_rate: Optional[int] = None
    raw_response: dict[str, Any] = field(default_factory=dict)


class BaseMusicModel(ABC):
    """Provider-independent music generation interface."""

    provider_name: str = "unknown"

    @property
    def abilities(self) -> list[str]:
        return ["generate"]

    async def validate_connection(self) -> None:
        """Validate provider credentials without generating billed music."""
        return None

    async def aclose(self) -> None:
        """Release provider resources."""
        return None

    @abstractmethod
    async def generate_music(
        self,
        prompt: str,
        music_length_seconds: Optional[float] = None,
        force_instrumental: bool = False,
        output_format: str = "auto",
    ) -> MusicResult:
        """Generate music from a natural-language prompt."""
