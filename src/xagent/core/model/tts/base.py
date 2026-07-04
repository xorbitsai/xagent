"""Base classes for Text-to-Speech (TTS) models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class TTSResult:
    """Text-to-Speech synthesis result with metadata."""

    audio: bytes
    """Synthesized audio data"""

    format: str
    """Audio format (e.g., 'mp3', 'wav', 'pcm')"""

    sample_rate: Optional[int] = None
    """Sample rate in Hz"""

    language: Optional[str] = None
    """Language code"""

    raw_response: Optional[dict[str, Any]] = None
    """Raw response from the API for debugging or advanced usage"""


class BaseTTS(ABC):
    """Abstract base class for Text-to-Speech (TTS) models."""

    provider_name = "unknown"

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[bytes, TTSResult]:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            voice: Voice ID or name (e.g., 'zh-android', 'zh-female')
            language: Language code (e.g., 'zh', 'en')
            format: Output audio format (e.g., 'mp3', 'wav', 'pcm')
            sample_rate: Sample rate in Hz (e.g., 22050, 24000, 48000)
            **kwargs: Additional model-specific parameters

        Returns:
            Audio bytes (if verbose=False) or
            TTSResult with detailed information (if verbose=True)

        Raises:
            RuntimeError: If synthesis fails
        """
        pass

    @property
    @abstractmethod
    def abilities(self) -> list[str]:
        """Get the list of abilities supported by this model."""
        pass

    @property
    def supports_multiple_voices(self) -> bool:
        """Check if model supports multiple voices."""
        return "multiple_voices" in self.abilities

    @property
    def supports_voice_cloning(self) -> bool:
        """Check if model supports voice cloning."""
        return "voice_cloning" in self.abilities

    @property
    def supports_voice_listing(self) -> bool:
        """Check if model can list available voices dynamically."""
        return "voice_listing" in self.abilities

    @property
    def supports_voice_settings(self) -> bool:
        """Check if model accepts structured voice settings."""
        return "voice_settings" in self.abilities

    @property
    def supported_voice_settings(self) -> list[str]:
        """Provider-specific voice setting keys accepted by this model."""
        return []

    @property
    def supported_provider_options(self) -> list[str]:
        """Provider-specific synthesis option keys accepted by this model."""
        return []

    async def list_available_voices(self) -> list[dict[str, Any]]:
        """List available voices for providers that support dynamic voice lookup."""
        raise NotImplementedError(
            f"{self.provider_name} TTS does not support dynamic voice listing"
        )
