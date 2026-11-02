from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union


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


@dataclass(frozen=True)
class ASRStreamEvent:
    """A live transcript update; a final result commits one segment, not a turn."""

    type: Literal["session_started", "partial", "final"]
    text: str = ""
    segment_id: int = 0
    session_id: Optional[str] = None


class ASRStreamSession(ABC):
    """Own one live, mono audio connection and its transcript iterator.

    Closing or cancelling discards pending audio; it does not commit a turn.
    Consumers should use the async context manager to release the connection
    when stopping iteration early. Close the session before the opening
    ``usage_scope`` or ``TokenContextManager`` exits: usage is flushed to that
    scope on close and cannot be delivered after the scope has already exited.
    """

    @abstractmethod
    async def send_audio(self, audio: bytes) -> None: ...

    @abstractmethod
    async def commit(self) -> None: ...

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[ASRStreamEvent]: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    async def __aenter__(self) -> ASRStreamSession:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


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
    def supports_streaming(self) -> bool:
        """Whether this configured model accepts live audio input."""
        return False

    @property
    def supports_vad(self) -> bool:
        """Whether the live ASR service can commit segments using its own VAD."""
        return False

    async def open_stream(
        self,
        *,
        language: Optional[str] = None,
        format: str = "pcm",
        sample_rate: int = 16000,
        vad: bool = True,
        **kwargs: Any,
    ) -> ASRStreamSession:
        """Open a live audio session; PCM means signed 16-bit little-endian mono."""
        raise NotImplementedError("This ASR model does not support live audio")

    @property
    def supports_speaker_diarization(self) -> bool:
        """Check if model supports speaker identification/diarization."""
        return "speaker_diarization" in self.abilities

    @property
    def supports_timestamps(self) -> bool:
        """Check if model supports timestamp information."""
        return "timestamps" in self.abilities
