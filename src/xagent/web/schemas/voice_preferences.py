"""Account preferences for live voice conversations, separate from writing style."""

from pydantic import BaseModel, ConfigDict, Field


class VoiceConversationPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    asr_model_id: str | None = Field(default=None, min_length=1, max_length=512)
    tts_model_id: str | None = Field(default=None, min_length=1, max_length=512)
    voice: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$")
