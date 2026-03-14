from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class ASRResult:
    """Detailed ASR transcription result with metadata."""

    text: str
    """Transcribed text"""

    segments: Optional[list[ASRSegment]] = None
    """List of transcription segments with timing info"""

    language: Optional[str] = None
    """Detected or specified language"""

    raw_response: Optional[dict[str, Any]] = None
    """Raw response from the API for debugging or advanced usage"""

    def __str__(self) -> str:
        """Return the transcribed text."""
        return self.text


@dataclass
class ASRSegment:
    """A segment of transcribed audio with timing and speaker info."""

    text: str
    """Transcribed text for this segment"""

    start: float
    """Start time in seconds"""

    end: float
    """End time in seconds"""

    speaker: Optional[str] = None
    """Speaker identifier if supported by model"""

    confidence: Optional[float] = None
    """Confidence score if supported by model"""

    def __str__(self) -> str:
        """Return segment text with speaker info if available."""
        if self.speaker:
            return f"[{self.speaker}] {self.text}"
        return self.text


class BaseASR(ABC):
    """Abstract base class for Automatic Speech Recognition (ASR) models."""

    @abstractmethod
    async def transcribe(
        self,
        audio: Union[str, bytes],
        language: Optional[str] = None,
        format: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Union[str, ASRResult]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio file path or audio bytes
            language: Language code (e.g., 'zh', 'en', 'yue')
            format: Audio format (e.g., 'wav', 'mp3', 'm4a')
            verbose: If True, return detailed ASRResult with segments and metadata.
                     If False, return only transcribed text string.
            **kwargs: Additional model-specific parameters

        Returns:
            Transcribed text string (if verbose=False) or
            ASRResult with detailed information (if verbose=True)

        Raises:
            RuntimeError: If transcription fails
        """
        pass

    @property
    @abstractmethod
    def abilities(self) -> list[str]:
        """Get the list of abilities supported by this model."""
        pass

    @property
    def supports_speaker_diarization(self) -> bool:
        """Check if model supports speaker identification/diarization."""
        return "speaker_diarization" in self.abilities

    @property
    def supports_timestamps(self) -> bool:
        """Check if model supports timestamp information."""
        return "timestamps" in self.abilities
