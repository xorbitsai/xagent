"""ElevenLabs ASR provider implementation."""

from __future__ import annotations

import logging
import math
import mimetypes
import os
from collections.abc import Iterable
from inspect import isawaitable
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from ...utils.security import redact_sensitive_text
from .base import ASRResult, ASRSegment, BaseASR

logger = logging.getLogger(__name__)

ELEVENLABS_DEFAULT_BASE_URL = "https://api.elevenlabs.io"
ELEVENLABS_DEFAULT_ASR_MODEL = "scribe_v2"

_ELEVENLABS_ASR_MODELS: tuple[dict[str, Any], ...] = (
    {
        "id": "scribe_v2",
        "object": "model",
        "owned_by": "elevenlabs",
        "category": "speech",
        "abilities": ["asr"],
        "name": "Scribe v2",
        "description": "ElevenLabs speech-to-text model",
    },
    {
        "id": "scribe_v1",
        "object": "model",
        "owned_by": "elevenlabs",
        "category": "speech",
        "abilities": ["asr"],
        "name": "Scribe v1",
        "description": "ElevenLabs speech-to-text model",
    },
)

_ELEVENLABS_STT_PROVIDER_OPTION_FIELDS = (
    "enable_logging",
    "language_code",
    "tag_audio_events",
    "num_speakers",
    "timestamps_granularity",
    "diarize",
    "diarization_threshold",
    "additional_formats",
    "file_format",
    "cloud_storage_url",
    "source_url",
    "webhook",
    "webhook_id",
    "temperature",
    "seed",
    "use_multi_channel",
    "multichannel_output_style",
    "webhook_metadata",
    "entity_detection",
    "no_verbatim",
    "use_speaker_library",
    "detect_speaker_roles",
    "entity_redaction",
    "entity_redaction_mode",
    "keyterms",
    "request_options",
)

_ELEVENLABS_FILE_FORMATS = {"pcm_s16le_16", "other"}


def _get_field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped: dict[str, Any] = value.model_dump(exclude_none=True)
        return dumped
    if hasattr(value, "dict"):
        legacy_dumped: dict[str, Any] = value.dict(exclude_none=True)
        return legacy_dumped
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and item is not None
        }
    return {"value": str(value)}


def _normalize_keyterms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if item]
    return [str(value)]


def _content_type_for_filename(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


class ElevenLabsASR(BaseASR):
    """ElevenLabs speech-to-text model client using the official SDK."""

    provider_name = "elevenlabs"

    def __init__(
        self,
        model: str = ELEVENLABS_DEFAULT_ASR_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        self.model = model
        self.model_name = model
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.base_url = (
            base_url or os.getenv("ELEVENLABS_BASE_URL") or ELEVENLABS_DEFAULT_BASE_URL
        ).rstrip("/")
        self.language = language
        self._async_client: Any = None

    @staticmethod
    def _create_async_client(
        api_key: Optional[str], base_url: Optional[str] = None
    ) -> Any:
        try:
            from elevenlabs.client import AsyncElevenLabs
        except ImportError as exc:
            raise RuntimeError(
                "The 'elevenlabs' package is required for ElevenLabs ASR. "
                "Install project dependencies to enable this provider."
            ) from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        return AsyncElevenLabs(**kwargs)

    def _ensure_async_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key is required for ASR transcription")
        if self._async_client is None:
            self._async_client = self._create_async_client(self.api_key, self.base_url)
        return self._async_client

    @staticmethod
    async def _close_client(client: Any) -> None:
        if client is None:
            return

        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result

    async def aclose(self) -> None:
        """Close any cached ElevenLabs SDK clients."""
        async_client = self._async_client
        self._async_client = None
        await self._close_client(async_client)

    async def __aenter__(self) -> "ElevenLabsASR":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @staticmethod
    def _validate_provider_options(options: dict[str, Any]) -> None:
        unsupported_keys = set(options) - set(_ELEVENLABS_STT_PROVIDER_OPTION_FIELDS)
        if unsupported_keys:
            supported = ", ".join(_ELEVENLABS_STT_PROVIDER_OPTION_FIELDS)
            unsupported = ", ".join(sorted(unsupported_keys))
            raise ValueError(
                "Unsupported ElevenLabs STT provider_options keys: "
                f"{unsupported}. Supported keys: {supported}."
            )

    def _build_request_kwargs(
        self,
        *,
        language: Optional[str],
        format: Optional[str],
        verbose: bool,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        request_kwargs = {
            key: value for key, value in kwargs.items() if value is not None
        }

        language_code = (
            request_kwargs.pop("language_code", None) or language or self.language
        )
        if language_code:
            request_kwargs["language_code"] = language_code

        hotword = request_kwargs.pop("hotword", None)
        if hotword is not None and "keyterms" not in request_kwargs:
            request_kwargs["keyterms"] = _normalize_keyterms(hotword)

        if verbose:
            request_kwargs.setdefault("timestamps_granularity", "word")

        if format:
            normalized_format = format.lower().strip()
            if (
                normalized_format in _ELEVENLABS_FILE_FORMATS
                and "file_format" not in request_kwargs
            ):
                request_kwargs["file_format"] = normalized_format

        self._validate_provider_options(request_kwargs)
        return request_kwargs

    def _build_file_payload(
        self, audio: Union[str, bytes], format: Optional[str]
    ) -> Any:
        if isinstance(audio, bytes):
            extension = (format or "wav").strip().lstrip(".") or "wav"
            filename = f"audio.{extension}"
            return (filename, audio, _content_type_for_filename(filename))

        audio_path = Path(audio)
        return (
            audio_path.name,
            audio_path.open("rb"),
            _content_type_for_filename(audio_path.name),
        )

    async def transcribe(
        self,
        audio: Union[str, bytes],
        language: Optional[str] = None,
        format: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Union[str, ASRResult]:
        """Transcribe audio to text."""

        request_kwargs = self._build_request_kwargs(
            language=language,
            format=format,
            verbose=verbose,
            kwargs=kwargs,
        )

        client = self._ensure_async_client()
        file_handle: Optional[BinaryIO] = None
        if (
            isinstance(audio, str)
            and audio.startswith(("http://", "https://"))
            and "source_url" not in request_kwargs
            and "cloud_storage_url" not in request_kwargs
        ):
            request_kwargs["source_url"] = audio
        elif (
            "source_url" not in request_kwargs
            and "cloud_storage_url" not in request_kwargs
        ):
            file_payload = self._build_file_payload(audio, format)
            if hasattr(file_payload[1], "close"):
                file_handle = file_payload[1]
            request_kwargs["file"] = file_payload

        try:
            response = await client.speech_to_text.convert(
                model_id=self.model,
                **request_kwargs,
            )
        except Exception as exc:
            redacted_error = redact_sensitive_text(str(exc))
            logger.error("ElevenLabs ASR failed: %s", redacted_error)
            raise RuntimeError(f"ElevenLabs ASR failed: {redacted_error}") from exc
        finally:
            if file_handle is not None:
                file_handle.close()

        text, segments, result_language, raw_response = self._normalize_response(
            response
        )
        if not verbose:
            return text

        return ASRResult(
            text=text,
            segments=segments if segments else None,
            language=result_language,
            raw_response=raw_response,
        )

    @staticmethod
    def _normalize_response(
        response: Any,
    ) -> tuple[str, list[ASRSegment], Optional[str], dict[str, Any]]:
        raw_response = _to_dict(response)
        transcripts = _get_field(response, "transcripts") or []

        if transcripts:
            transcript_texts = [
                str(text)
                for transcript in transcripts
                if (text := _get_field(transcript, "text"))
            ]
            text = "\n".join(transcript_texts)
            language = _get_field(transcripts[0], "language_code")
            segments: list[ASRSegment] = []
            for transcript in transcripts:
                segments.extend(ElevenLabsASR._parse_word_segments(transcript))
            return text, segments, language, raw_response

        text = str(_get_field(response, "text") or "")
        language = _get_field(response, "language_code")
        segments = ElevenLabsASR._parse_word_segments(response)
        return text, segments, language, raw_response

    @staticmethod
    def _word_confidence(word: Any) -> Optional[float]:
        confidence = _get_field(word, "confidence")
        if confidence is not None:
            return float(confidence)

        logprob = _get_field(word, "logprob")
        if logprob is None:
            return None

        probability = math.exp(float(logprob))
        return max(0.0, min(1.0, probability))

    @staticmethod
    def _parse_word_segments(response: Any) -> list[ASRSegment]:
        words = _get_field(response, "words") or []
        if not words:
            return []

        segments: list[ASRSegment] = []
        for word in words:
            word_text = _get_field(word, "text")
            if word_text is None or not str(word_text).strip():
                continue

            start = _get_field(word, "start")
            end = _get_field(word, "end")
            if start is None or end is None:
                continue

            speaker = _get_field(word, "speaker_id")
            confidence = ElevenLabsASR._word_confidence(word)
            segments.append(
                ASRSegment(
                    text=str(word_text),
                    start=float(start),
                    end=float(end),
                    speaker=str(speaker) if speaker is not None else None,
                    confidence=confidence,
                )
            )

        return segments

    @property
    def abilities(self) -> list[str]:
        return [
            "asr",
            "timestamps",
            "speaker_diarization",
        ]

    @staticmethod
    async def async_list_available_models(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        _ = api_key, base_url
        return [dict(model) for model in _ELEVENLABS_ASR_MODELS]
