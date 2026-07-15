"""ElevenLabs TTS provider implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import uuid
from collections.abc import Iterable
from inspect import isawaitable
from pathlib import Path
from typing import Any, Optional, Union

from ...utils.security import redact_sensitive_text
from .base import BaseTTS, TTSResult

logger = logging.getLogger(__name__)

ELEVENLABS_DEFAULT_BASE_URL = "https://api.elevenlabs.io"
ELEVENLABS_DEFAULT_MODEL = "eleven_v3"
ELEVENLABS_DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
_ELEVENLABS_VOICE_SETTING_FIELDS = (
    "stability",
    "similarity_boost",
    "style",
    "speed",
    "use_speaker_boost",
)
_ELEVENLABS_PROVIDER_OPTION_FIELDS = (
    "enable_logging",
    "optimize_streaming_latency",
    "pronunciation_aliases",
    "pronunciation_dictionary_locators",
    "seed",
    "previous_text",
    "next_text",
    "previous_request_ids",
    "next_request_ids",
    "use_pvc_as_ivc",
    "apply_text_normalization",
    "apply_language_text_normalization",
)

_GENERIC_OUTPUT_FORMATS: dict[str, str] = {
    "mp3": ELEVENLABS_DEFAULT_OUTPUT_FORMAT,
    "wav": "wav_44100",
    "pcm": "pcm_24000",
    "ulaw": "ulaw_8000",
    "mulaw": "ulaw_8000",
    "alaw": "alaw_8000",
    "opus": "opus_48000_128",
}


def _get_field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _sample_rate_from_output_format(output_format: str) -> Optional[int]:
    parts = output_format.split("_")
    for part in parts[1:]:
        if part.isdigit() and len(part) >= 4:
            return int(part)
    return None


def _extension_from_output_format(output_format: str) -> str:
    codec = output_format.split("_", 1)[0].lower()
    if codec == "mulaw":
        return "ulaw"
    return codec or "mp3"


class ElevenLabsTTS(BaseTTS):
    """ElevenLabs text-to-speech model client using the official SDK."""

    provider_name = "elevenlabs"

    def __init__(
        self,
        model: str = ELEVENLABS_DEFAULT_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        format: str = ELEVENLABS_DEFAULT_OUTPUT_FORMAT,
        sample_rate: Optional[int] = None,
    ) -> None:
        self.model = model
        self.model_name = model
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.base_url = (
            base_url or os.getenv("ELEVENLABS_BASE_URL") or ELEVENLABS_DEFAULT_BASE_URL
        ).rstrip("/")
        self.voice = voice or ELEVENLABS_DEFAULT_VOICE_ID
        self.language = language
        self.output_format = self._resolve_output_format(format, sample_rate)
        self._validate_sample_rate_request(format, sample_rate, self.output_format)
        self.sample_rate = _sample_rate_from_output_format(self.output_format)
        self._client: Any = None
        self._async_client: Any = None
        self._cloned_voice_ids: dict[str, str] = {}
        self._voice_clone_lock = asyncio.Lock()

    @staticmethod
    def _create_client(api_key: Optional[str], base_url: Optional[str] = None) -> Any:
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise RuntimeError(
                "The 'elevenlabs' package is required for ElevenLabs TTS. "
                "Install project dependencies to enable this provider."
            ) from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        try:
            return ElevenLabs(**kwargs)
        except TypeError:
            # Older SDK releases may not accept base_url.
            kwargs.pop("base_url", None)
            return ElevenLabs(**kwargs)

    @staticmethod
    def _create_async_client(
        api_key: Optional[str], base_url: Optional[str] = None
    ) -> Any:
        try:
            from elevenlabs.client import AsyncElevenLabs
        except ImportError as exc:
            raise RuntimeError(
                "The 'elevenlabs' package is required for ElevenLabs TTS. "
                "Install project dependencies to enable this provider."
            ) from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        try:
            return AsyncElevenLabs(**kwargs)
        except TypeError:
            # Older SDK releases may not accept base_url.
            kwargs.pop("base_url", None)
            return AsyncElevenLabs(**kwargs)

    def _ensure_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key is required for TTS synthesis")
        if self._client is None:
            self._client = self._create_client(self.api_key, self.base_url)
        return self._client

    def _ensure_async_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key is required for TTS synthesis")
        if self._async_client is None:
            self._async_client = self._create_async_client(self.api_key, self.base_url)
        return self._async_client

    @staticmethod
    def _validate_sample_rate_request(
        requested_format: Optional[str],
        sample_rate: Optional[int],
        output_format: str,
    ) -> None:
        if sample_rate is None:
            return

        resolved_codec = output_format.split("_", 1)[0].lower()
        resolved_sample_rate = _sample_rate_from_output_format(output_format)
        if resolved_codec == "pcm" or resolved_sample_rate == sample_rate:
            return

        raise ValueError(
            "ElevenLabs sample_rate must match the resolved output format. "
            "Use format='pcm' for custom sample rates, or pass an ElevenLabs "
            f"output_format that includes {sample_rate}. Requested format "
            f"'{requested_format}' resolves to '{output_format}'."
        )

    def _resolve_output_format(
        self, requested_format: Optional[str], sample_rate: Optional[int]
    ) -> str:
        output_format = (requested_format or ELEVENLABS_DEFAULT_OUTPUT_FORMAT).strip()
        normalized = output_format.lower()
        codec = normalized.split("_", 1)[0]
        if codec == "pcm" and sample_rate:
            return f"pcm_{sample_rate}"
        if "_" in normalized:
            return normalized
        return _GENERIC_OUTPUT_FORMATS.get(normalized, output_format)

    @staticmethod
    def _validate_provider_options(options: dict[str, Any]) -> None:
        unsupported_keys = set(options) - set(_ELEVENLABS_PROVIDER_OPTION_FIELDS)
        if unsupported_keys:
            supported = ", ".join(_ELEVENLABS_PROVIDER_OPTION_FIELDS)
            unsupported = ", ".join(sorted(unsupported_keys))
            raise ValueError(
                "Unsupported ElevenLabs provider_options keys: "
                f"{unsupported}. Supported keys: {supported}."
            )

    @staticmethod
    def _apply_pronunciation_aliases(
        text: str, aliases: Optional[dict[str, str]]
    ) -> str:
        """Apply case-sensitive, non-cascading aliases to ElevenLabs input text."""
        if aliases is None:
            return text
        if not isinstance(aliases, dict):
            raise ValueError(
                "ElevenLabs pronunciation_aliases must be an object mapping "
                "source phrases to spoken aliases"
            )

        normalized_aliases: dict[str, str] = {}
        for source, alias in aliases.items():
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    "ElevenLabs pronunciation_aliases source phrases must be "
                    "non-empty strings"
                )
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(
                    "ElevenLabs pronunciation_aliases spoken aliases must be "
                    "non-empty strings"
                )
            normalized_aliases[source] = alias

        if not normalized_aliases:
            return text

        # Match all source phrases against the original text in one pass so an
        # alias is never processed by a later rule. Prefer longer phrases when
        # rules overlap, which keeps phrase-level corrections deterministic.
        source_phrases = sorted(normalized_aliases, key=len, reverse=True)
        escaped_sources: list[str] = []
        for source in source_phrases:
            escaped = re.escape(source)
            # Guard ASCII word-like edges against partial matches such as
            # replacing "UN" inside "RUN". Do not use Unicode ``\w`` here:
            # it treats CJK characters as word characters and would prevent
            # aliases from matching inside unspaced text.
            if source[0].isascii() and (source[0].isalnum() or source[0] == "_"):
                escaped = rf"(?<![A-Za-z0-9_]){escaped}"
            if source[-1].isascii() and (source[-1].isalnum() or source[-1] == "_"):
                escaped = rf"{escaped}(?![A-Za-z0-9_])"
            escaped_sources.append(escaped)

        pattern = re.compile("|".join(escaped_sources))
        return pattern.sub(
            lambda match: normalized_aliases[match.group(0)],
            text,
        )

    @staticmethod
    async def _close_client(client: Any) -> None:
        if client is None:
            return

        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result

    @staticmethod
    def _close_sync_client(client: Any) -> None:
        if client is None:
            return

        close = getattr(client, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        """Delete temporary voice clones and close cached ElevenLabs SDK clients."""
        async with self._voice_clone_lock:
            client = self._client
            async_client = self._async_client
            if (
                async_client is None
                and self._cloned_voice_ids
                and self.api_key is not None
            ):
                try:
                    async_client = self._create_async_client(
                        self.api_key, self.base_url
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to create ElevenLabs client for temporary voice cleanup: %s",
                        redact_sensitive_text(str(exc)),
                    )
            self._client = None
            self._async_client = None

            if async_client is not None and self._cloned_voice_ids:
                deleted_voice_ids: set[str] = set()
                for voice_id in set(self._cloned_voice_ids.values()):
                    try:
                        await async_client.voices.delete(voice_id)
                        deleted_voice_ids.add(voice_id)
                    except Exception as exc:
                        logger.warning(
                            "Failed to delete temporary ElevenLabs voice clone %s: %s",
                            voice_id,
                            redact_sensitive_text(str(exc)),
                        )
                if deleted_voice_ids:
                    self._cloned_voice_ids = {
                        cache_key: voice_id
                        for cache_key, voice_id in self._cloned_voice_ids.items()
                        if voice_id not in deleted_voice_ids
                    }

            await self._close_client(async_client)
            await self._close_client(client)

    async def __aenter__(self) -> "ElevenLabsTTS":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @staticmethod
    def _coerce_audio_bytes(response: Any) -> bytes:
        if isinstance(response, bytes):
            return response
        if isinstance(response, bytearray):
            return bytes(response)
        if isinstance(response, memoryview):
            return response.tobytes()

        data = getattr(response, "data", None)
        if data is not None:
            return ElevenLabsTTS._coerce_audio_bytes(data)

        read = getattr(response, "read", None)
        if callable(read):
            return ElevenLabsTTS._coerce_audio_bytes(read())

        if isinstance(response, (str, dict)):
            raise RuntimeError(
                f"Unexpected ElevenLabs audio response: {type(response)}"
            )

        if isinstance(response, Iterable):
            chunks: list[bytes] = []
            for chunk in response:
                if isinstance(chunk, bytes):
                    chunks.append(chunk)
                elif isinstance(chunk, bytearray):
                    chunks.append(bytes(chunk))
                elif chunk is None:
                    continue
                else:
                    raise RuntimeError(
                        f"Unexpected ElevenLabs audio chunk type: {type(chunk)}"
                    )
            return b"".join(chunks)

        raise RuntimeError(f"Unexpected ElevenLabs audio response: {type(response)}")

    @staticmethod
    def _coerce_voice_settings(voice_settings: Any) -> Any:
        try:
            from elevenlabs.types import VoiceSettings
        except ImportError as exc:
            raise RuntimeError(
                "The 'elevenlabs' package is required for ElevenLabs voice settings. "
                "Install project dependencies to enable this provider."
            ) from exc

        if isinstance(voice_settings, VoiceSettings):
            return voice_settings

        if not isinstance(voice_settings, dict):
            raise ValueError("ElevenLabs voice_settings must be an object")

        unsupported_keys = set(voice_settings) - set(_ELEVENLABS_VOICE_SETTING_FIELDS)
        if unsupported_keys:
            supported = ", ".join(_ELEVENLABS_VOICE_SETTING_FIELDS)
            unsupported = ", ".join(sorted(unsupported_keys))
            raise ValueError(
                "Unsupported ElevenLabs voice_settings keys: "
                f"{unsupported}. Supported keys: {supported}."
            )

        return VoiceSettings(
            **{k: v for k, v in voice_settings.items() if v is not None}
        )

    @staticmethod
    def _normalize_voice_settings(settings: Any) -> Optional[dict[str, Any]]:
        if settings is None:
            return None
        if isinstance(settings, dict):
            normalized = settings
        elif hasattr(settings, "model_dump"):
            normalized = settings.model_dump(exclude_none=True)
        else:
            normalized = {
                key: _get_field(settings, key)
                for key in _ELEVENLABS_VOICE_SETTING_FIELDS
            }

        return {k: v for k, v in normalized.items() if v is not None} or None

    @staticmethod
    def _normalize_verified_languages(languages: Any) -> Optional[list[dict[str, Any]]]:
        if not languages:
            return None

        normalized_languages: list[dict[str, Any]] = []
        for language in languages:
            normalized = {
                key: _get_field(language, key)
                for key in ("language", "model_id", "accent", "locale", "preview_url")
            }
            filtered = {k: v for k, v in normalized.items() if v is not None}
            if filtered:
                normalized_languages.append(filtered)

        return normalized_languages or None

    @staticmethod
    def _reference_audio_metadata(reference_audio: Any) -> tuple[Path, str, str]:
        if not isinstance(reference_audio, (str, os.PathLike)):
            raise ValueError(
                "ElevenLabs reference_audio must be a local audio file path"
            )

        audio_path = Path(reference_audio).expanduser()
        if not audio_path.is_file():
            raise ValueError(
                f"ElevenLabs reference audio file does not exist: {audio_path}"
            )

        stat = audio_path.stat()
        if stat.st_size == 0:
            raise ValueError("ElevenLabs reference audio file must not be empty")

        mime_type = (
            mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        )
        resolved_path = str(audio_path.resolve())
        cache_key = f"{resolved_path}:{stat.st_size}:{stat.st_mtime_ns}"
        return audio_path, cache_key, mime_type

    @classmethod
    def _reference_audio_file(cls, reference_audio: Any) -> tuple[str, bytes, str]:
        audio_path, cache_key, mime_type = cls._reference_audio_metadata(
            reference_audio
        )
        audio = audio_path.read_bytes()
        if not audio:
            raise ValueError("ElevenLabs reference audio file must not be empty")
        return cache_key, audio, mime_type

    async def _get_or_create_cloned_voice(self, reference_audio: Any) -> str:
        audio_path, cache_key, mime_type = self._reference_audio_metadata(
            reference_audio
        )
        cached_voice_id = self._cloned_voice_ids.get(cache_key)
        if cached_voice_id:
            return cached_voice_id

        async with self._voice_clone_lock:
            cached_voice_id = self._cloned_voice_ids.get(cache_key)
            if cached_voice_id:
                return cached_voice_id

            audio = audio_path.read_bytes()
            if not audio:
                raise ValueError("ElevenLabs reference audio file must not be empty")
            clone_name = f"xagent-{audio_path.stem[:40]}-{uuid.uuid4().hex[:8]}"
            client = self._ensure_async_client()
            response = await client.voices.ivc.create(
                name=clone_name,
                files=[(audio_path.name, audio, mime_type)],
            )
            voice_id = _get_field(response, "voice_id", "id")
            if not voice_id:
                raise RuntimeError("ElevenLabs voice cloning returned no voice ID")

            normalized_voice_id = str(voice_id)
            self._cloned_voice_ids[cache_key] = normalized_voice_id
            return normalized_voice_id

    async def clone_voice(
        self,
        *,
        name: str,
        reference_audio_files: list[str],
        description: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        remove_background_noise: bool = False,
    ) -> dict[str, Any]:
        """Create a persistent ElevenLabs Instant Voice Clone."""
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("ElevenLabs voice clone name must not be empty")
        if not reference_audio_files:
            raise ValueError("At least one reference audio file is required")

        files: list[tuple[str, bytes, str]] = []
        for reference_audio in reference_audio_files:
            _, audio, mime_type = self._reference_audio_file(reference_audio)
            files.append((Path(reference_audio).name, audio, mime_type))

        request_kwargs: dict[str, Any] = {
            "name": normalized_name,
            "files": files,
        }
        if description is not None:
            request_kwargs["description"] = description
        if labels is not None:
            # SDK 2.0 accepted only a serialized labels object, while newer
            # releases also accept a mapping. A JSON string works across both.
            request_kwargs["labels"] = json.dumps(labels)
        if remove_background_noise:
            request_kwargs["remove_background_noise"] = True

        client = self._ensure_async_client()
        try:
            response = await client.voices.ivc.create(**request_kwargs)
        except Exception as exc:
            redacted_error = redact_sensitive_text(str(exc))
            logger.error("ElevenLabs voice cloning failed: %s", redacted_error)
            raise RuntimeError(
                f"ElevenLabs voice cloning failed: {redacted_error}"
            ) from exc

        voice_id = _get_field(response, "voice_id", "id")
        if not voice_id:
            raise RuntimeError("ElevenLabs voice cloning returned no voice ID")

        result: dict[str, Any] = {
            "voice_id": str(voice_id),
            "name": normalized_name,
            "provider": self.provider_name,
            "persistent": True,
        }
        requires_verification = _get_field(response, "requires_verification")
        if requires_verification is not None:
            result["requires_verification"] = bool(requires_verification)
        return result

    async def delete_voice(self, voice_id: str) -> None:
        """Delete a persistent voice from the configured ElevenLabs account."""
        normalized_voice_id = voice_id.strip()
        if not normalized_voice_id:
            raise ValueError("ElevenLabs voice ID must not be empty")

        client = self._ensure_async_client()
        try:
            await client.voices.delete(normalized_voice_id)
        except Exception as exc:
            redacted_error = redact_sensitive_text(str(exc))
            logger.error("ElevenLabs voice deletion failed: %s", redacted_error)
            raise RuntimeError(
                f"ElevenLabs voice deletion failed: {redacted_error}"
            ) from exc

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Union[bytes, TTSResult]:
        reference_audio = kwargs.pop("reference_audio", None)
        voice_id = voice or self.voice
        final_output_format = self._resolve_output_format(
            format or self.output_format, sample_rate or self.sample_rate
        )
        self._validate_sample_rate_request(
            format or self.output_format,
            sample_rate,
            final_output_format,
        )
        final_sample_rate = _sample_rate_from_output_format(final_output_format)
        language_code = kwargs.pop("language_code", None) or language or self.language
        voice_settings = kwargs.pop("voice_settings", None)
        pronunciation_aliases = kwargs.pop("pronunciation_aliases", None)
        synthesis_text = self._apply_pronunciation_aliases(
            text,
            pronunciation_aliases,
        )

        request_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self._validate_provider_options(request_kwargs)
        if language_code:
            request_kwargs["language_code"] = language_code
        if voice_settings is not None:
            request_kwargs["voice_settings"] = self._coerce_voice_settings(
                voice_settings
            )

        client = self._ensure_async_client()
        try:
            if reference_audio:
                voice_id = await self._get_or_create_cloned_voice(reference_audio)
            response = client.text_to_speech.convert(
                text=synthesis_text,
                voice_id=voice_id,
                model_id=self.model,
                output_format=final_output_format,
                **request_kwargs,
            )
            # ElevenLabs SDK 2.56+ exposes ``convert`` as an async generator,
            # while older releases returned a coroutine that resolved to an
            # async iterator. Support both contracts without awaiting a
            # generator object directly.
            if isawaitable(response):
                response = await response
            chunks: list[bytes] = []
            async for chunk in response:
                chunks.append(self._coerce_audio_bytes(chunk))
            audio = b"".join(chunks)
        except Exception as exc:
            error_text = str(exc)
            if (
                "invalid_uid" in error_text
                or "invalid ID has been received" in error_text
            ):
                message = (
                    "ElevenLabs rejected the provided voice ID. ElevenLabs voice IDs "
                    "are account-specific opaque values: call list_tts_voices or "
                    "clone_tts_voice and use an exact returned voice_id, or omit voice "
                    "to use the configured default."
                )
                logger.error("ElevenLabs TTS failed: %s", message)
                raise RuntimeError(message) from exc
            redacted_error = redact_sensitive_text(str(exc))
            logger.error(
                "ElevenLabs TTS failed: %s",
                redacted_error,
            )
            raise RuntimeError(f"ElevenLabs TTS failed: {redacted_error}") from exc

        if not audio:
            raise RuntimeError("ElevenLabs TTS returned no audio data")

        if not verbose:
            return audio

        return TTSResult(
            audio=audio,
            format=_extension_from_output_format(final_output_format),
            sample_rate=final_sample_rate,
            language=language_code,
            raw_response={
                "model": self.model,
                "voice_id": voice_id,
                "output_format": final_output_format,
            },
        )

    @property
    def abilities(self) -> list[str]:
        return [
            "tts",
            "text_to_speech",
            "audio",
            "audio_generation",
            "multilingual",
            "multiple_voices",
            "voice_cloning",
            "persistent_voice_cloning",
            "voice_listing",
            "voice_settings",
            "real_time",
        ]

    @property
    def supported_voice_settings(self) -> list[str]:
        return list(_ELEVENLABS_VOICE_SETTING_FIELDS)

    @property
    def supported_provider_options(self) -> list[str]:
        return list(_ELEVENLABS_PROVIDER_OPTION_FIELDS)

    @staticmethod
    def _normalize_model_response(response: Any) -> list[dict[str, Any]]:
        raw_models = _get_field(response, "models") or response

        models: list[dict[str, Any]] = []
        for raw_model in raw_models:
            can_do_tts = _get_field(raw_model, "can_do_text_to_speech")
            if can_do_tts is False:
                continue

            model_id = _get_field(raw_model, "model_id", "id")
            if not model_id:
                continue

            model_info = {
                "id": str(model_id),
                "object": "model",
                "owned_by": "elevenlabs",
                "abilities": ["tts"],
            }
            name = _get_field(raw_model, "name")
            if name:
                model_info["name"] = str(name)
            description = _get_field(raw_model, "description")
            if description:
                model_info["description"] = str(description)
            languages = _get_field(raw_model, "languages")
            if languages:
                model_info["languages"] = languages
            models.append(model_info)

        return models

    @staticmethod
    def _normalize_voice_response(response: Any) -> list[dict[str, Any]]:
        raw_voices = _get_field(response, "voices") or response

        voices: list[dict[str, Any]] = []
        for raw_voice in raw_voices:
            voice_id = _get_field(raw_voice, "voice_id", "id")
            if not voice_id:
                continue

            voice_info: dict[str, Any] = {
                "voice_id": str(voice_id),
                "provider": "elevenlabs",
            }
            category = _get_field(raw_voice, "category")
            if category:
                voice_info["category"] = str(category).lower()

            for field_name in (
                "name",
                "description",
                "preview_url",
                "labels",
                "available_for_tiers",
                "high_quality_base_model_ids",
            ):
                value = _get_field(raw_voice, field_name)
                if value:
                    voice_info[field_name] = value

            settings = ElevenLabsTTS._normalize_voice_settings(
                _get_field(raw_voice, "settings")
            )
            if settings:
                voice_info["settings"] = settings

            verified_languages = ElevenLabsTTS._normalize_verified_languages(
                _get_field(raw_voice, "verified_languages")
            )
            if verified_languages:
                voice_info["verified_languages"] = verified_languages

            voices.append(voice_info)

        return voices

    async def list_available_voices(self) -> list[dict[str, Any]]:
        client = self._ensure_async_client()
        response = await client.voices.get_all()
        return self._normalize_voice_response(response)

    @staticmethod
    async def async_list_available_models(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        resolved_api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not resolved_api_key:
            raise ValueError("ElevenLabs API key is required to list models")

        client = ElevenLabsTTS._create_async_client(
            resolved_api_key,
            (
                base_url
                or os.getenv("ELEVENLABS_BASE_URL")
                or ELEVENLABS_DEFAULT_BASE_URL
            ),
        )
        try:
            response = await client.models.list()
            return ElevenLabsTTS._normalize_model_response(response)
        finally:
            await ElevenLabsTTS._close_client(client)

    @staticmethod
    def list_available_models(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        resolved_api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not resolved_api_key:
            raise ValueError("ElevenLabs API key is required to list models")

        client = ElevenLabsTTS._create_client(
            resolved_api_key,
            (
                base_url
                or os.getenv("ELEVENLABS_BASE_URL")
                or ELEVENLABS_DEFAULT_BASE_URL
            ),
        )
        try:
            response = client.models.list()
            return ElevenLabsTTS._normalize_model_response(response)
        finally:
            ElevenLabsTTS._close_sync_client(client)
