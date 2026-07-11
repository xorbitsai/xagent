"""ElevenLabs sound effect generation provider."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from inspect import isawaitable
from typing import Any, Optional

from ...utils.security import redact_sensitive_text
from .base import BaseSoundEffectModel, SoundEffectResult

logger = logging.getLogger(__name__)

ELEVENLABS_DEFAULT_BASE_URL = "https://api.elevenlabs.io"
ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL = "eleven_text_to_sound_v2"
ELEVENLABS_DEFAULT_SOUND_EFFECT_OUTPUT_FORMAT = "mp3_44100_128"
ELEVENLABS_DOCUMENTED_SOUND_EFFECT_MODELS: tuple[dict[str, Any], ...] = (
    {
        "id": ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL,
        "object": "model",
        "owned_by": "elevenlabs",
        "category": "sound_effect",
        "abilities": ["generate"],
        "name": "Text to Sound v2",
        "description": "Sound effects generation from text prompts",
    },
)


def _get_field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _sample_rate_from_output_format(output_format: str) -> Optional[int]:
    for part in output_format.split("_")[1:]:
        if part.isdigit() and len(part) >= 4:
            return int(part)
    return None


def _extension_from_output_format(output_format: str) -> str:
    codec = output_format.split("_", 1)[0].lower()
    return "ulaw" if codec == "mulaw" else codec or "mp3"


class ElevenLabsSoundEffectModel(BaseSoundEffectModel):
    """Generate sound effects through the ElevenLabs API."""

    provider_name = "elevenlabs"

    def __init__(
        self,
        model_name: str = ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.base_url = (
            base_url or os.getenv("ELEVENLABS_BASE_URL") or ELEVENLABS_DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._async_client: Any = None

    @staticmethod
    def _create_async_client(
        api_key: Optional[str], base_url: Optional[str] = None
    ) -> Any:
        try:
            from elevenlabs.client import AsyncElevenLabs
        except ImportError as exc:
            raise RuntimeError(
                "The 'elevenlabs' package is required for sound effect generation"
            ) from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        try:
            return AsyncElevenLabs(**kwargs)
        except TypeError:
            kwargs.pop("base_url", None)
            return AsyncElevenLabs(**kwargs)

    def _ensure_async_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError(
                "ElevenLabs API key is required for sound effect generation"
            )
        if self._async_client is None:
            self._async_client = self._create_async_client(self.api_key, self.base_url)
        return self._async_client

    def _request_options(self) -> Optional[dict[str, Any]]:
        options: dict[str, Any] = {}
        if self.timeout is not None:
            options["timeout_in_seconds"] = int(self.timeout)
        if self.max_retries is not None:
            options["max_retries"] = self.max_retries
        return options or None

    @staticmethod
    def _coerce_audio_bytes(response: Any) -> bytes:
        if isinstance(response, bytes):
            return response
        if isinstance(response, (bytearray, memoryview)):
            return bytes(response)
        data = getattr(response, "data", None)
        if data is not None:
            return ElevenLabsSoundEffectModel._coerce_audio_bytes(data)
        if isinstance(response, Iterable) and not isinstance(response, (str, dict)):
            return b"".join(
                ElevenLabsSoundEffectModel._coerce_audio_bytes(chunk)
                for chunk in response
                if chunk is not None
            )
        raise RuntimeError(
            f"Unexpected ElevenLabs sound effect response: {type(response)}"
        )

    async def aclose(self) -> None:
        client = self._async_client
        self._async_client = None
        if client is None:
            return
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result

    async def validate_connection(self) -> None:
        """Probe account access without requiring a catalog model-name match."""
        try:
            await self._ensure_async_client().models.list(
                request_options=self._request_options()
            )
        except Exception as exc:
            safe_error = redact_sensitive_text(str(exc))
            if "missing_permissions" not in safe_error or "user_read" not in safe_error:
                raise
            logger.warning(
                "ElevenLabs key lacks user_read; generation access cannot be "
                "validated without a billed request: %s",
                safe_error,
            )

    async def generate_sound_effect(
        self,
        text: str,
        duration_seconds: Optional[float] = None,
        prompt_influence: float = 0.3,
        loop: bool = False,
        output_format: str = ELEVENLABS_DEFAULT_SOUND_EFFECT_OUTPUT_FORMAT,
    ) -> SoundEffectResult:
        prompt = text.strip()
        if not prompt:
            raise ValueError("Sound effect text must not be empty")
        if duration_seconds is not None and not 0.5 <= duration_seconds <= 30:
            raise ValueError("duration_seconds must be between 0.5 and 30")
        if not 0 <= prompt_influence <= 1:
            raise ValueError("prompt_influence must be between 0 and 1")
        if loop and self.model_name != ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL:
            raise ValueError("loop is only supported by eleven_text_to_sound_v2")

        request_kwargs: dict[str, Any] = {
            "text": prompt,
            "output_format": output_format,
            "loop": loop,
            "prompt_influence": prompt_influence,
            "model_id": self.model_name,
        }
        request_options = self._request_options()
        if request_options:
            request_kwargs["request_options"] = request_options
        if duration_seconds is not None:
            request_kwargs["duration_seconds"] = duration_seconds

        try:
            response = self._ensure_async_client().text_to_sound_effects.convert(
                **request_kwargs
            )
            if isawaitable(response):
                response = await response
            chunks: list[bytes] = []
            async for chunk in response:
                chunks.append(self._coerce_audio_bytes(chunk))
            audio = b"".join(chunks)
        except Exception as exc:
            redacted_error = redact_sensitive_text(str(exc))
            logger.error(
                "ElevenLabs sound effect generation failed: %s", redacted_error
            )
            raise RuntimeError(
                f"ElevenLabs sound effect generation failed: {redacted_error}"
            ) from exc

        if not audio:
            raise RuntimeError("ElevenLabs sound effect generation returned no audio")

        return SoundEffectResult(
            audio=audio,
            format=_extension_from_output_format(output_format),
            sample_rate=_sample_rate_from_output_format(output_format),
            raw_response={
                "model": self.model_name,
                "output_format": output_format,
                "duration_seconds": duration_seconds,
                "prompt_influence": prompt_influence,
                "loop": loop,
            },
        )

    @classmethod
    async def async_list_available_models(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        resolved_api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not resolved_api_key:
            raise ValueError("ElevenLabs API key is required to list models")

        model = cls(api_key=resolved_api_key, base_url=base_url)
        response: Any = None
        try:
            try:
                response = await model._ensure_async_client().models.list()
            except Exception as exc:
                safe_error = redact_sensitive_text(str(exc))
                if (
                    "missing_permissions" not in safe_error
                    or "user_read" not in safe_error
                ):
                    raise
                logger.warning(
                    "Could not list ElevenLabs sound effect models because the API "
                    "key lacks user_read; using the documented catalog: %s",
                    safe_error,
                )
        finally:
            await model.aclose()

        discovered: list[dict[str, Any]] = []
        for raw_model in _get_field(response, "models") or response or []:
            model_id = _get_field(raw_model, "model_id", "id")
            if not model_id:
                continue

            name = _get_field(raw_model, "name")
            description = _get_field(raw_model, "description")
            searchable = " ".join(
                str(value).lower() for value in (model_id, name, description) if value
            )
            capability = any(
                _get_field(raw_model, field) is True
                for field in (
                    "can_do_text_to_sound_effects",
                    "can_generate_sound_effects",
                    "can_do_sound_generation",
                )
            )
            recognizable = (
                "text_to_sound" in searchable
                or "text-to-sound" in searchable
                or "sound_effect" in searchable
                or "sound effect" in searchable
            )
            if not capability and not recognizable:
                continue

            model_info: dict[str, Any] = {
                "id": str(model_id),
                "object": "model",
                "owned_by": "elevenlabs",
                "category": "sound_effect",
                "abilities": ["generate"],
            }
            if name:
                model_info["name"] = str(name)
            if description:
                model_info["description"] = str(description)
            discovered.append(model_info)

        if discovered:
            return discovered

        # As of July 2026, ElevenLabs documents its sound-effect model but
        # omits it from GET /v1/models, which currently returns only TTS/STS
        # entries. Keep the live discovery path above so newly exposed models
        # (including a future v3) win automatically, and use the provider's
        # documented catalog only while the API returns no SFX entries.
        return [dict(item) for item in ELEVENLABS_DOCUMENTED_SOUND_EFFECT_MODELS]
