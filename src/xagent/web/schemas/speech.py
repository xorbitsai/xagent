"""Requests for the authenticated streaming speech transports."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ASRStreamStart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["start"]
    model_id: str = Field(min_length=1, max_length=512)
    language: str | None = Field(default=None, min_length=1, max_length=32)
    format: Literal["pcm", "ulaw"] = "pcm"
    sample_rate: Literal[8000, 16000, 22050, 24000, 44100, 48000] = 16000
    vad: bool = True
    vad_silence_threshold_secs: float | None = Field(default=None, ge=0.3, le=3)
    vad_threshold: float | None = Field(default=None, ge=0.1, le=0.9)


class TTSStreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1, max_length=5000)
    voice: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    language: str | None = Field(default=None, min_length=1, max_length=32)
    format: str | None = Field(default=None, min_length=1, max_length=64)
    sample_rate: int | None = Field(default=None, ge=8000, le=48000)
