"""Xinference ASR provider implementation."""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, Union

from ..xinference_base import BaseXinferenceModel
from .base import ASRResult, ASRSegment, BaseASR

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Protocol for xinference ASR model handle."""

    async def transcriptions(self, audio: bytes, **kwargs: Any) -> dict[str, Any]: ...
    def close(self) -> None: ...


class XinferenceASR(BaseASR, BaseXinferenceModel):
    """
    Xinference Automatic Speech Recognition (ASR) model client using the xinference-client SDK.
    Supports speech recognition using Xinference's audio models.
    """

    def __init__(
        self,
        model: str = "whisper-base",
        model_uid: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """
        Initialize Xinference ASR client.

        Args:
            model: Model name (e.g., "whisper-base", "whisper-large-v3")
            model_uid: Unique model UID in Xinference (if model is already launched)
            base_url: Xinference server base URL (e.g., "http://localhost:9997")
            api_key: Optional API key for authentication
            language: Default language code (e.g., 'zh', 'en')
        """
        BaseXinferenceModel.__init__(self, model, model_uid, base_url, api_key)
        self.language = language

    async def transcribe(
        self,
        audio: Union[str, bytes],
        language: Optional[str] = None,
        format: Optional[str] = None,
        verbose: bool = False,
        response_format: Optional[str] = None,
        hotword: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, ASRResult]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio file path or audio bytes
            language: Language code (e.g., 'zh', 'en', 'yue')
            format: Audio format (e.g., 'wav', 'mp3', 'm4a')
            verbose: If True, return detailed ASRResult with segments and metadata
            response_format: 'json' or 'verbose_json' for detailed output
            hotword: Hotwords to improve recognition accuracy (space-separated, e.g., "香港 航空")
            **kwargs: Additional model-specific parameters

        Returns:
            Transcribed text string (if verbose=False) or
            ASRResult with detailed information (if verbose=True)

        Raises:
            RuntimeError: If transcription fails

        Example:
            >>> # Use hotwords to improve specific term recognition
            >>> result = asr.transcribe("audio.mp3", hotword="香港 航空", verbose=True)
        """
        model_handle = await self._ensure_model_handle()

        # Prepare audio input
        if isinstance(audio, bytes):
            # If audio is bytes, we need to write it to a temporary file
            import tempfile

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{format or 'wav'}"
            ) as temp_file:
                temp_file.write(audio)
                audio_path = temp_file.name
        else:
            audio_path = audio

        try:
            # Prepare transcription parameters
            params = {}
            final_language = language or self.language
            if final_language:
                params["language"] = final_language

            # Use verbose_json if verbose is True
            if verbose or response_format:
                params["response_format"] = response_format or "verbose_json"

            # Add hotword parameter if provided
            if hotword:
                params["hotword"] = hotword

            # Add any additional parameters
            params.update(kwargs)

            # Read audio file
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()

            # Call async transcriptions API (not speech - speech is for TTS!)
            result = await model_handle.transcriptions(audio=audio_data, **params)

            # Extract transcription result
            if isinstance(result, dict):
                raw_response = result
                transcription_text: str = result.get("text", "")

                if verbose:
                    # Parse segments if available
                    # Try different segment field names
                    segments_data: Any = (
                        result.get("segments")
                        or result.get("sentences")
                        or result.get("sentence_info")
                        or []
                    )

                    # Also check for word-level timestamps
                    words_data = result.get("words", [])

                    segments = self._parse_segments(segments_data, transcription_text)

                    # If no sentence-level segments but have word-level timestamps, create word segments
                    if not segments and words_data and isinstance(words_data, list):
                        # Create segments from word-level timestamps
                        # Each word segment is a single character/token
                        for i, word_info in enumerate(words_data):
                            if isinstance(word_info, dict):
                                start = word_info.get("start", 0)
                                end = word_info.get("end", 0)

                                # For word-level segments, we don't have the actual text
                                # Just create a placeholder segment
                                segments.append(
                                    ASRSegment(
                                        text=f"[word {i}]",
                                        start=float(start),
                                        end=float(end),
                                        speaker=None,
                                        confidence=None,
                                    )
                                )

                    return ASRResult(
                        text=transcription_text,
                        segments=segments if segments else None,
                        language=result.get("language"),
                        raw_response=raw_response,
                    )
                else:
                    return transcription_text

            elif hasattr(result, "text"):
                alt_transcription_text: str = str(result.text)

                if verbose and hasattr(result, "segments"):
                    segments = self._parse_segments(result.segments)
                    return ASRResult(
                        text=alt_transcription_text,
                        segments=segments,
                        language=getattr(result, "language", None),
                        raw_response=result.__dict__
                        if hasattr(result, "__dict__")
                        else None,
                    )
                else:
                    return alt_transcription_text
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Xinference ASR failed: {e}")
            raise RuntimeError(f"Xinference ASR failed: {str(e)}") from e

        finally:
            # Clean up temporary file if it was created
            if isinstance(audio, bytes):
                import os

                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    def _parse_segments(
        self, segments_data: Any, text: Optional[str] = None
    ) -> list[ASRSegment]:
        """Parse segment data from API response into ASRSegment objects."""
        segments = []

        # Handle sentence-level segments (if available)
        if segments_data and isinstance(segments_data, list):
            for segment in segments_data:
                if isinstance(segment, dict) and "text" in segment:
                    # This is a sentence-level segment
                    seg_text = segment.get("text", "")
                    seg_start = segment.get("start", 0)
                    seg_end = segment.get("end", 0)
                    seg_speaker = segment.get("speaker", segment.get("spk"))
                    seg_confidence = segment.get("confidence")

                    segments.append(
                        ASRSegment(
                            text=seg_text,
                            start=float(seg_start),
                            end=float(seg_end),
                            speaker=str(seg_speaker)
                            if seg_speaker is not None
                            else None,
                            confidence=float(seg_confidence)
                            if seg_confidence is not None
                            else None,
                        )
                    )
                elif hasattr(segment, "text"):
                    segments.append(
                        ASRSegment(
                            text=segment.text,
                            start=float(segment.start)
                            if hasattr(segment, "start")
                            else 0,
                            end=float(segment.end) if hasattr(segment, "end") else 0,
                            speaker=str(segment.speaker)
                            if hasattr(segment, "speaker") and segment.speaker
                            else None,
                            confidence=float(segment.confidence)
                            if hasattr(segment, "confidence") and segment.confidence
                            else None,
                        )
                    )

        return segments

    @property
    def abilities(self) -> list[str]:
        """Get the list of abilities supported by this model."""
        # Xinference ASR models support various features depending on the model
        # Some models like paraformer-zh-spk support speaker diarization
        # Some models support timestamps and detailed output
        # Some models like seaco-paraformer-zh and paraformer-zh-hotword support hotwords
        return [
            "asr",
            "speech_recognition",
            "audio",
            "timestamps",  # Supports timestamp information
            "speaker_diarization",  # Some models support speaker identification
            "verbose_json",  # Supports detailed JSON output
            "hotword",  # Some models support hotword for improved accuracy
        ]

    def __enter__(self) -> "XinferenceASR":
        """Context manager entry."""
        return self

    @staticmethod
    def list_available_models(
        base_url: str, api_key: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Fetch available ASR/audio models from Xinference server.

        Args:
            base_url: Xinference server base URL
            api_key: Optional API key for authentication

        Returns:
            List of available ASR/audio models with their information

        Example:
            >>> models = XinferenceASR.list_available_models(
            ...     base_url="http://localhost:9997"
            ... )
        """
        try:
            from xinference_client import RESTfulClient as XinferenceClient
        except ImportError:
            from xinference.client.restful.restful_client import (
                RESTfulClient as XinferenceClient,  # type: ignore
            )

        client = XinferenceClient(base_url=base_url, api_key=api_key)

        try:
            # Get list of running models
            models_dict = client.list_models()

            # Filter for audio/ASR models
            result = []
            for model_uid, model_info in models_dict.items():
                model_ability = model_info.get("model_ability", [])
                if isinstance(model_ability, str):
                    model_ability = [model_ability]

                # Check if model has audio or ASR ability
                if any(
                    ability in model_ability for ability in ["audio", "asr", "speech"]
                ):
                    result.append(
                        {
                            "id": model_info.get("model_name", model_uid),
                            "model_uid": model_uid,
                            "model_type": model_info.get("model_type", ""),
                            "model_ability": model_ability,
                            "description": model_info.get("model_description", ""),
                        }
                    )

            return result

        except Exception as e:
            logger.error(f"Failed to fetch ASR models from Xinference: {e}")
            return []
