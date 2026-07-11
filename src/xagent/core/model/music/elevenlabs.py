"""ElevenLabs Music provider."""

from __future__ import annotations

import logging
import os
import re
from inspect import isawaitable
from typing import Any, Literal, Optional, get_args, get_origin

from ...utils.security import redact_sensitive_text
from .base import BaseMusicModel, MusicResult

logger = logging.getLogger(__name__)

ELEVENLABS_DEFAULT_BASE_URL = "https://api.elevenlabs.io"
ELEVENLABS_DEFAULT_MUSIC_MODEL = "music_v2"
ELEVENLABS_DOCUMENTED_MUSIC_MODELS = ("music_v2", "music_v1")


def _extension_from_output_format(output_format: str) -> str:
    if output_format == "auto":
        return "mp3"
    codec = output_format.split("_", 1)[0].lower()
    return "ulaw" if codec == "mulaw" else codec or "mp3"


def _sample_rate_from_output_format(
    output_format: str, model_name: str
) -> Optional[int]:
    if output_format == "auto":
        return 48000 if model_name == "music_v2" else 44100
    for part in output_format.split("_")[1:]:
        if part.isdigit() and len(part) >= 4:
            return int(part)
    return None


def _literal_string_values(annotation: Any) -> set[str]:
    """Collect string Literal values from an SDK type alias."""
    if get_origin(annotation) is Literal:
        return {value for value in get_args(annotation) if isinstance(value, str)}
    values: set[str] = set()
    for item in get_args(annotation):
        values.update(_literal_string_values(item))
    return values


def _get_field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _music_model_name(model_id: str, name: Optional[str] = None) -> str:
    if name:
        return name
    version = model_id.removeprefix("music_")
    return f"Eleven Music {version}" if version else model_id


def _music_model_sort_key(model_id: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"music_v(\d+)", model_id)
    if match:
        return (1, int(match.group(1)))
    return (0, model_id)


class ElevenLabsMusicModel(BaseMusicModel):
    """Generate music through the ElevenLabs Music API."""

    provider_name = "elevenlabs"

    def __init__(
        self,
        model_name: str = ELEVENLABS_DEFAULT_MUSIC_MODEL,
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
                "The 'elevenlabs' package is required for music generation"
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
            raise RuntimeError("ElevenLabs API key is required for music generation")
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
        """Validate access with the non-billed composition-plan endpoint."""
        request_kwargs: dict[str, Any] = {
            "prompt": "A short instrumental music cue",
            "music_length_ms": 3000,
            "model_id": self.model_name,
        }
        request_options = self._request_options()
        if request_options:
            request_kwargs["request_options"] = request_options
        await self._ensure_async_client().music.composition_plan.create(
            **request_kwargs
        )

    async def generate_music(
        self,
        prompt: str,
        music_length_seconds: Optional[float] = None,
        force_instrumental: bool = False,
        output_format: str = "auto",
    ) -> MusicResult:
        text = prompt.strip()
        if not text:
            raise ValueError("Music prompt must not be empty")
        if len(text) > 4100:
            raise ValueError("Music prompt must not exceed 4100 characters")
        if music_length_seconds is not None and not 3 <= music_length_seconds <= 600:
            raise ValueError("music_length_seconds must be between 3 and 600")

        request_kwargs: dict[str, Any] = {
            "prompt": text,
            "model_id": self.model_name,
            "force_instrumental": force_instrumental,
            "output_format": output_format,
        }
        request_options = self._request_options()
        if request_options:
            request_kwargs["request_options"] = request_options
        if music_length_seconds is not None:
            request_kwargs["music_length_ms"] = int(music_length_seconds * 1000)

        try:
            response = self._ensure_async_client().music.compose(**request_kwargs)
            if isawaitable(response):
                response = await response
            chunks: list[bytes] = []
            async for chunk in response:
                chunks.append(bytes(chunk))
            audio = b"".join(chunks)
        except Exception as exc:
            redacted_error = redact_sensitive_text(str(exc))
            logger.error("ElevenLabs music generation failed: %s", redacted_error)
            raise RuntimeError(
                f"ElevenLabs music generation failed: {redacted_error}"
            ) from exc

        if not audio:
            raise RuntimeError("ElevenLabs music generation returned no audio")

        return MusicResult(
            audio=audio,
            format=_extension_from_output_format(output_format),
            sample_rate=_sample_rate_from_output_format(output_format, self.model_name),
            raw_response={
                "model": self.model_name,
                "output_format": output_format,
                "music_length_seconds": music_length_seconds,
                "force_instrumental": force_instrumental,
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
            raise ValueError("ElevenLabs API key is required to list music models")

        provider_models: dict[str, dict[str, Any]] = {}
        model = cls(api_key=resolved_api_key, base_url=base_url)
        try:
            try:
                response = await model._ensure_async_client().models.list()
                for raw_model in _get_field(response, "models") or response or []:
                    model_id = _get_field(raw_model, "model_id", "id")
                    if not isinstance(model_id, str):
                        continue
                    name = _get_field(raw_model, "name")
                    description = _get_field(raw_model, "description")
                    searchable = " ".join(
                        str(value).lower()
                        for value in (model_id, name, description)
                        if value
                    )
                    if not (
                        model_id.startswith("music_")
                        or "eleven music" in searchable
                        or "music generation" in searchable
                    ):
                        continue
                    provider_models[model_id] = {
                        "id": model_id,
                        "object": "model",
                        "owned_by": "elevenlabs",
                        "category": "music",
                        "abilities": ["generate"],
                        "name": _music_model_name(model_id, name),
                        "description": description
                        or "ElevenLabs prompt-to-music generation model",
                    }
            except Exception as exc:
                safe_error = redact_sensitive_text(str(exc))
                if (
                    "missing_permissions" not in safe_error
                    or "user_read" not in safe_error
                ):
                    raise
                logger.warning(
                    "Could not list ElevenLabs music models because the API key "
                    "lacks user_read; using the SDK catalog: %s",
                    safe_error,
                )
        finally:
            await model.aclose()

        try:
            from elevenlabs.music.types.body_compose_music_v_1_music_post_model_id import (
                BodyComposeMusicV1MusicPostModelId,
            )

            sdk_models = _literal_string_values(BodyComposeMusicV1MusicPostModelId)
        except (ImportError, AttributeError):
            sdk_models = set()

        discovered_ids = set(provider_models) | sdk_models
        ordered = sorted(
            discovered_ids - set(ELEVENLABS_DOCUMENTED_MUSIC_MODELS),
            key=_music_model_sort_key,
            reverse=True,
        )
        ordered.extend(ELEVENLABS_DOCUMENTED_MUSIC_MODELS)
        return [
            provider_models.get(
                model_id,
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "elevenlabs",
                    "category": "music",
                    "abilities": ["generate"],
                    "name": _music_model_name(model_id),
                    "description": "ElevenLabs prompt-to-music generation model",
                },
            )
            for model_id in ordered
        ]
