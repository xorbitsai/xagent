"""Xinference TTS provider implementation."""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, Union

from xinference_client import RESTfulClient as XinferenceClient

from .base import BaseTTS, TTSResult

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Protocol for xinference model handle."""

    def speech(self, text: str, **kwargs: Any) -> bytes: ...
    def close(self) -> None: ...


class XinferenceTTS(BaseTTS):
    """
    Xinference Text-to-Speech (TTS) model client using the xinference-client SDK.

    Supports speech synthesis using Xinference's TTS/audio-generation models.
    Models include chat-tts, edge-tts, and other text-to-speech models.
    """

    def __init__(
        self,
        model: str = "chat-tts",
        model_uid: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        format: str = "mp3",
        sample_rate: int = 24000,
    ):
        """
        Initialize Xinference TTS client.

        Args:
            model: Model name (e.g., "chat-tts", "edge-tts")
            model_uid: Unique model UID in Xinference (if model is already launched)
            base_url: Xinference server base URL (e.g., "http://localhost:9997")
            api_key: Optional API key for authentication
            voice: Default voice ID or speaker ID (if supported by the model)
            language: Default language code (e.g., 'zh', 'en')
            format: Output audio format (e.g., "mp3", "wav", "pcm")
            sample_rate: Sample rate in Hz (22050, 24000, 48000)
        """
        self.model = model
        self._model_uid = model_uid or model
        self.base_url = (base_url or "http://localhost:9997").rstrip("/")
        self.api_key = api_key
        self.voice = voice
        self.language = language
        self.format = format
        self.sample_rate = sample_rate

        # Initialize the Xinference client (lazy initialization)
        self._client: Optional[XinferenceClient] = None
        self._model_handle: Optional[ModelProtocol] = None

    def _get_session(self) -> XinferenceClient:  # type: ignore[no-any-unimported]
        """Get or create Xinference client."""
        if self._client is None:
            self._client = XinferenceClient(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._client

    def _ensure_model_handle(self) -> ModelProtocol:
        """Ensure the TTS model handle is initialized."""
        if self._model_handle is None:
            client = self._get_session()
            # Get the model handle (assumes model is already launched on the server)
            self._model_handle = client.get_model(self._model_uid)
        return self._model_handle

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Union[bytes, TTSResult]:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            voice: Voice ID or speaker ID (model-specific)
                   If None, uses the default voice from initialization
            language: Language code (e.g., 'zh', 'en', 'yue')
                      If None, auto-detected or uses default from initialization
            format: Output audio format ('mp3', 'wav', 'pcm')
                    If None, uses the format from initialization
            sample_rate: Sample rate in Hz (22050, 24000, 48000)
                        If None, uses the sample_rate from initialization
            verbose: If True, return TTSResult with metadata. If False, return audio bytes only
            **kwargs: Additional model-specific parameters
                     Common parameters include:
                     - speed: Speech speed (default: 1.0)
                     - volume: Volume control (default: 1.0)

        Returns:
            Audio bytes (if verbose=False) or TTSResult with metadata (if verbose=True)

        Raises:
            RuntimeError: If synthesis fails

        Example:
            >>> # Basic usage
            >>> tts = XinferenceTTS(model="chat-tts")
            >>> audio = tts.synthesize("你好，世界！")
            >>>
            >>> # With voice control
            >>> audio = tts.synthesize("Hello, world!", voice="female")
            >>>
            >>> # With speed control
            >>> audio = tts.synthesize("Test", speed=1.2)
        """
        model_handle = self._ensure_model_handle()

        # Prepare parameters
        final_voice = voice or self.voice
        final_language = language or self.language
        final_format = format or self.format
        final_sample_rate = sample_rate or self.sample_rate

        # Build synthesis parameters
        params: dict[str, Any] = {
            "output_audio_format": final_format,
            "sample_rate": final_sample_rate,
        }

        # Add optional parameters
        if final_voice:
            params["voice"] = final_voice
        if final_language:
            params["language"] = final_language

        # Add any additional parameters
        params.update(kwargs)

        try:
            # Call speech synthesis API
            # Xinference TTS models use speech() method
            audio_data = model_handle.speech(text, **params)

            # Validate response
            if audio_data is None:
                raise RuntimeError("Synthesis returned no audio data")

            if not isinstance(audio_data, bytes):
                raise RuntimeError(f"Unexpected audio data type: {type(audio_data)}")

            if verbose:
                return TTSResult(
                    audio=audio_data,
                    format=final_format,
                    sample_rate=final_sample_rate,
                    language=final_language,
                    raw_response={
                        "model": self.model,
                        "voice": final_voice,
                    },
                )
            else:
                return audio_data

        except Exception as e:
            logger.error(f"Xinference TTS failed: {e}")
            raise RuntimeError(f"Xinference TTS failed: {str(e)}") from e

    @property
    def abilities(self) -> list[str]:
        """Get the list of abilities supported by this model."""
        abilities = [
            "tts",
            "text_to_speech",
            "audio",
            "audio_generation",
            "real_time",  # Supports real-time synthesis
        ]

        # Add multilingual support
        # Most Xinference TTS models support multiple languages
        abilities.append("multilingual")

        # Add voice/speaker support if configured
        if self.voice:
            abilities.append("multiple_voices")

        return abilities

    def close(self) -> None:
        """Close the Xinference client and cleanup resources."""
        if self._model_handle is not None:
            try:
                self._model_handle.close()
            except Exception:
                pass
            self._model_handle = None

        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> "XinferenceTTS":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    @staticmethod
    def list_available_models(
        base_url: str, api_key: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Fetch available TTS/audio-generation models from Xinference server.

        Args:
            base_url: Xinference server base URL
            api_key: Optional API key for authentication

        Returns:
            List of available TTS/audio-generation models with their information

        Example:
            >>> models = XinferenceTTS.list_available_models(
            ...     base_url="http://localhost:9997"
            ... )
            >>> for model in models:
            ...     print(f"{model['id']}: {model['description']}")
        """
        client = XinferenceClient(base_url=base_url, api_key=api_key)

        try:
            # Get list of running models
            models_dict = client.list_models()

            # Filter for audio-generation/TTS models
            result = []
            for model_uid, model_info in models_dict.items():
                model_ability = model_info.get("model_ability", [])
                if isinstance(model_ability, str):
                    model_ability = [model_ability]

                # Check if model has audio-generation or TTS ability
                if any(
                    ability in model_ability
                    for ability in ["audio-generation", "tts", "speech"]
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
            logger.error(f"Failed to fetch TTS models from Xinference: {e}")
            return []

        finally:
            client.close()

    @staticmethod
    def get_sample_rates() -> list[int]:
        """Get supported sample rates.

        Returns:
            List of supported sample rates in Hz
        """
        return [8000, 16000, 22050, 24000, 44100, 48000]

    @staticmethod
    def get_supported_formats() -> list[str]:
        """Get supported audio formats.

        Returns:
            List of supported audio formats
        """
        return ["mp3", "wav", "pcm", "opus"]
