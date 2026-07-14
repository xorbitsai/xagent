"""
Audio processing tool for xagent

This module provides audio processing capabilities including:
- Speech-to-Text (ASR/Automatic Speech Recognition)
- Text-to-Speech (TTS/Speech Synthesis)

Uses pre-configured ASR and TTS models passed from the web layer.
"""

import json
import logging
import uuid
from inspect import isawaitable
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ...file_ref import build_workspace_file_ref
from ...model.asr.base import ASRResult, BaseASR
from ...model.tts.base import BaseTTS, TTSResult
from ...workspace import TaskWorkspace
from .audio_tool_descriptions import (
    CLONE_TTS_VOICE_DESCRIPTION,
    DELETE_TTS_VOICE_DESCRIPTION,
    LIST_TTS_VOICES_DESCRIPTION,
    SYNTHESIZE_SPEECH_DESCRIPTION,
    SYNTHESIZE_SPEECH_JSON_DESCRIPTION,
    TRANSCRIBE_AUDIO_DESCRIPTION,
)

logger = logging.getLogger(__name__)


class AudioToolCore:
    """
    Audio processing tool that uses pre-configured ASR and TTS models.

    Tool descriptions are imported from audio_tool_descriptions.py for better maintainability.
    """

    # Import description templates from separate file
    TRANSCRIBE_AUDIO_DESCRIPTION = TRANSCRIBE_AUDIO_DESCRIPTION
    SYNTHESIZE_SPEECH_DESCRIPTION = SYNTHESIZE_SPEECH_DESCRIPTION
    SYNTHESIZE_SPEECH_JSON_DESCRIPTION = SYNTHESIZE_SPEECH_JSON_DESCRIPTION
    LIST_TTS_VOICES_DESCRIPTION = LIST_TTS_VOICES_DESCRIPTION
    CLONE_TTS_VOICE_DESCRIPTION = CLONE_TTS_VOICE_DESCRIPTION
    DELETE_TTS_VOICE_DESCRIPTION = DELETE_TTS_VOICE_DESCRIPTION

    def __init__(
        self,
        asr_models: Optional[Dict[str, BaseASR]] = None,
        tts_models: Optional[Dict[str, BaseTTS]] = None,
        model_descriptions: Optional[Dict[str, str]] = None,
        workspace: Optional[TaskWorkspace] = None,
        default_asr_model: Optional[BaseASR] = None,
        default_tts_model: Optional[BaseTTS] = None,
    ):
        """
        Initialize with pre-configured ASR and TTS models.

        Args:
            asr_models: Dictionary mapping model_id to BaseASR instances
            tts_models: Dictionary mapping model_id to BaseTTS instances
            model_descriptions: Dictionary mapping model_id to description strings
            workspace: Optional workspace for saving generated audio files
            default_asr_model: Default model for speech recognition
            default_tts_model: Default model for speech synthesis
        """
        self._asr_models = asr_models or {}
        self._tts_models = tts_models or {}
        self._model_descriptions = model_descriptions or {}
        self._workspace = workspace
        self._default_asr_model = default_asr_model
        self._default_tts_model = default_tts_model
        self._last_teardown_task_id: Optional[str] = None
        self._generate_model_info_text()

    @staticmethod
    async def _close_model_client(model: Any) -> None:
        close = getattr(model, "aclose", None) or getattr(model, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result

    async def aclose(self) -> None:
        """Close any configured model clients that expose close hooks."""
        models: list[Any] = [
            self._default_asr_model,
            self._default_tts_model,
            *self._asr_models.values(),
            *self._tts_models.values(),
        ]
        seen_model_ids: set[int] = set()
        for model in models:
            if model is None:
                continue
            model_id = id(model)
            if model_id in seen_model_ids:
                continue
            seen_model_ids.add(model_id)
            await self._close_model_client(model)

    async def teardown(self, task_id: Optional[str] = None) -> None:
        # AgentRunner tears down each tool wrapper for one execution; a new task_id
        # must still close any model clients lazily recreated by a later run.
        if task_id is not None and task_id == self._last_teardown_task_id:
            return

        await self.aclose()
        if task_id is not None:
            self._last_teardown_task_id = task_id

    async def __aenter__(self) -> "AudioToolCore":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _generate_model_info_text(self) -> None:
        """Generate formatted text with available models and descriptions."""
        # Generate ASR model info
        if not self._asr_models:
            self._asr_model_info_text = "No ASR models available"
        else:
            default_asr_id = (
                getattr(self._default_asr_model, "model_name", None)
                if self._default_asr_model
                else None
            )

            default_asr_lines = []
            other_asr_lines = []
            for model_id in self._asr_models.keys():
                description = self._model_descriptions.get(model_id, "")
                is_default = model_id == default_asr_id
                default_marker = " ⭐[DEFAULT]" if is_default else ""

                if description:
                    line = f"- {model_id}: {description}{default_marker}"
                else:
                    line = f"- {model_id}: No description available{default_marker}"

                if is_default:
                    default_asr_lines.append(line)
                else:
                    other_asr_lines.append(line)

            asr_model_lines = default_asr_lines + other_asr_lines
            self._asr_model_info_text = (
                "\n".join(asr_model_lines)
                if asr_model_lines
                else "No ASR models available"
            )

        # Generate TTS model info
        if not self._tts_models:
            self._tts_model_info_text = "No TTS models available"
        else:
            default_tts_id = (
                getattr(self._default_tts_model, "model_name", None)
                if self._default_tts_model
                else None
            )

            default_tts_lines = []
            other_tts_lines = []
            for model_id in self._tts_models.keys():
                description = self._model_descriptions.get(model_id, "")
                is_default = model_id == default_tts_id
                default_marker = " ⭐[DEFAULT]" if is_default else ""

                if description:
                    line = f"- {model_id}: {description}{default_marker}"
                else:
                    line = f"- {model_id}: No description available{default_marker}"

                if is_default:
                    default_tts_lines.append(line)
                else:
                    other_tts_lines.append(line)

            tts_model_lines = default_tts_lines + other_tts_lines
            self._tts_model_info_text = (
                "\n".join(tts_model_lines)
                if tts_model_lines
                else "No TTS models available"
            )

    def _get_model(
        self,
        models: Dict[str, Any],
        default_model: Optional[Any],
        model_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Generic method to get model by ID or default model.

        Args:
            models: Dictionary mapping model_id to model instances
            default_model: Configured default model instance
            model_id: Specific model ID to retrieve

        Returns:
            Model instance or None if not found
        """
        if model_id and model_id in models:
            return models[model_id]

        # Use configured default model
        if default_model:
            return default_model

        # Fallback: return first available model
        if models:
            return next(iter(models.values()))

        return None

    def _get_asr_model(self, model_id: Optional[str] = None) -> Optional[BaseASR]:
        """Get ASR model by ID or default model."""
        return self._get_model(self._asr_models, self._default_asr_model, model_id)

    def _merge_segments(
        self, segments: List[Dict[str, Any]], max_gap: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Merge consecutive segments from the same speaker.

        Args:
            segments: List of segment dictionaries
            max_gap: Maximum time gap (seconds) to merge segments

        Returns:
            List of merged segments with combined text and updated time ranges
        """
        if not segments:
            return []

        merged = []
        current = segments[0].copy()

        for next_seg in segments[1:]:
            # Check if segments should be merged
            gap = next_seg["start"] - current["end"]
            same_speaker = next_seg.get("speaker") == current.get("speaker")

            if same_speaker and gap <= max_gap:
                # Merge segments
                current["text"] += " " + next_seg["text"]
                current["end"] = next_seg["end"]
                # Update confidence to average if both exist
                if (
                    current.get("confidence") is not None
                    and next_seg.get("confidence") is not None
                ):
                    current["confidence"] = (
                        current["confidence"] + next_seg["confidence"]
                    ) / 2
                elif next_seg.get("confidence") is not None:
                    current["confidence"] = next_seg["confidence"]
            else:
                # Don't merge, save current segment
                merged.append(current)
                current = next_seg.copy()

        merged.append(current)
        return merged

    def _get_tts_model(self, model_id: Optional[str] = None) -> Optional[BaseTTS]:
        """Get TTS model by ID or default model."""
        return self._get_model(self._tts_models, self._default_tts_model, model_id)

    @staticmethod
    def _coerce_option_dict(
        value: Optional[Dict[str, Any]],
        field_name: str,
        reserved_keys: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be an object")

        reserved = reserved_keys or set()
        unsupported_keys = set(value) & reserved
        if unsupported_keys:
            unsupported = ", ".join(sorted(unsupported_keys))
            raise ValueError(
                f"{field_name} must not include standard TTS parameters: {unsupported}"
            )

        return {str(k): v for k, v in value.items() if v is not None}

    def _build_tts_provider_kwargs(
        self,
        *,
        tts_model: BaseTTS,
        voice_settings: Optional[Dict[str, Any]] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reserved_keys = {"text", "voice", "language", "format", "sample_rate"}
        options = self._coerce_option_dict(
            provider_options, "provider_options", reserved_keys
        )

        if voice_settings is not None:
            options["voice_settings"] = self._coerce_option_dict(
                voice_settings, "voice_settings"
            )

        if extra_options:
            options.update(
                self._coerce_option_dict(extra_options, "extra_options", reserved_keys)
            )

        self._validate_tts_provider_kwargs(
            tts_model=tts_model,
            voice_settings=options.get("voice_settings"),
            provider_options={
                k: v for k, v in options.items() if k != "voice_settings"
            },
        )
        return options

    def _merge_option_dicts(
        self,
        *,
        default_options: Optional[Dict[str, Any]],
        override_options: Optional[Dict[str, Any]],
        default_field_name: str,
        override_field_name: str,
        reserved_keys: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        merged = self._coerce_option_dict(
            default_options, default_field_name, reserved_keys
        )
        merged.update(
            self._coerce_option_dict(
                override_options, override_field_name, reserved_keys
            )
        )
        return merged

    @staticmethod
    def _get_tts_provider_name(tts_model: BaseTTS) -> str:
        return str(getattr(tts_model, "provider_name", type(tts_model).__name__))

    def _get_voice_listing_supported_providers(self) -> list[str]:
        providers: list[str] = []
        seen: set[str] = set()
        candidates: list[BaseTTS] = []
        if self._default_tts_model is not None:
            candidates.append(self._default_tts_model)
        candidates.extend(self._tts_models.values())

        for candidate in candidates:
            if not getattr(candidate, "supports_voice_listing", False):
                continue
            provider_name = self._get_tts_provider_name(candidate)
            if provider_name not in seen:
                seen.add(provider_name)
                providers.append(provider_name)

        return providers

    def _get_configured_tts_model_ids(self, provider_name: str) -> set[str]:
        """Return model IDs configured for the selected TTS provider."""
        model_ids = {
            model_id
            for model_id, model in self._tts_models.items()
            if self._get_tts_provider_name(model) == provider_name
        }
        if (
            self._default_tts_model is not None
            and self._get_tts_provider_name(self._default_tts_model) == provider_name
        ):
            default_model_id = getattr(self._default_tts_model, "model_name", None)
            if default_model_id:
                model_ids.add(str(default_model_id))
        return model_ids

    @staticmethod
    def _filter_voice_model_metadata(
        voices: list[dict[str, Any]],
        configured_model_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Remove provider model metadata for models not configured by the user."""
        filtered_voices: list[dict[str, Any]] = []
        for voice in voices:
            filtered_voice = dict(voice)

            if "high_quality_base_model_ids" in filtered_voice:
                high_quality_model_ids = filtered_voice["high_quality_base_model_ids"]
                matching_model_ids = [
                    model_id
                    for model_id in (
                        high_quality_model_ids
                        if isinstance(high_quality_model_ids, list)
                        else []
                    )
                    if str(model_id) in configured_model_ids
                ]
                if matching_model_ids:
                    filtered_voice["high_quality_base_model_ids"] = matching_model_ids
                else:
                    filtered_voice.pop("high_quality_base_model_ids")

            if "verified_languages" in filtered_voice:
                verified_languages = filtered_voice["verified_languages"]
                matching_languages = [
                    language
                    for language in (
                        verified_languages
                        if isinstance(verified_languages, list)
                        else []
                    )
                    if isinstance(language, dict)
                    and language.get("model_id") is not None
                    and str(language["model_id"]) in configured_model_ids
                ]
                if matching_languages:
                    filtered_voice["verified_languages"] = matching_languages
                else:
                    filtered_voice.pop("verified_languages")

            filtered_voices.append(filtered_voice)

        return filtered_voices

    def _validate_tts_provider_kwargs(
        self,
        *,
        tts_model: BaseTTS,
        voice_settings: Optional[Dict[str, Any]],
        provider_options: Dict[str, Any],
    ) -> None:
        provider_name = self._get_tts_provider_name(tts_model)

        if voice_settings:
            supported_voice_settings = list(
                getattr(tts_model, "supported_voice_settings", [])
            )
            if (
                not getattr(tts_model, "supports_voice_settings", False)
                and not supported_voice_settings
            ):
                raise ValueError(
                    f"Provider '{provider_name}' does not support voice_settings"
                )
            if supported_voice_settings:
                unsupported_keys = set(voice_settings) - set(supported_voice_settings)
                if unsupported_keys:
                    unsupported = ", ".join(sorted(unsupported_keys))
                    supported = ", ".join(supported_voice_settings)
                    raise ValueError(
                        f"Unsupported voice_settings keys for provider '{provider_name}': "
                        f"{unsupported}. Supported keys: {supported}."
                    )

        if provider_options:
            supported_provider_options = list(
                getattr(tts_model, "supported_provider_options", [])
            )
            if not supported_provider_options:
                unsupported = ", ".join(sorted(provider_options))
                raise ValueError(
                    f"Provider '{provider_name}' does not support provider_options: "
                    f"{unsupported}"
                )
            unsupported_keys = set(provider_options) - set(supported_provider_options)
            if unsupported_keys:
                unsupported = ", ".join(sorted(unsupported_keys))
                supported = ", ".join(supported_provider_options)
                raise ValueError(
                    f"Unsupported provider_options keys for provider '{provider_name}': "
                    f"{unsupported}. Supported keys: {supported}."
                )

    def _validate_tts_reference_audio(
        self,
        *,
        tts_model: BaseTTS,
        reference_audio: Optional[str],
    ) -> None:
        if not reference_audio:
            return
        if getattr(tts_model, "supports_voice_cloning", False):
            return

        provider_name = self._get_tts_provider_name(tts_model)
        raise ValueError(
            f"Provider '{provider_name}' does not support reference_audio voice cloning"
        )

    def _get_tts_model_id(self, tts_model: BaseTTS) -> str:
        for model_id, configured_model in self._tts_models.items():
            if configured_model is tts_model:
                return model_id
        return "default"

    def _get_provider_tts_model(
        self,
        *,
        provider: str,
        capability: str,
        model_id: Optional[str] = None,
    ) -> tuple[Optional[BaseTTS], str]:
        if model_id:
            tts_model = self._tts_models.get(model_id)
            if (
                tts_model is None
                and self._default_tts_model is not None
                and getattr(self._default_tts_model, "model_name", None) == model_id
            ):
                tts_model = self._default_tts_model
            return tts_model, model_id

        candidates: list[BaseTTS] = []
        if self._default_tts_model is not None:
            candidates.append(self._default_tts_model)
        candidates.extend(self._tts_models.values())

        for candidate in candidates:
            if self._get_tts_provider_name(candidate) == provider and getattr(
                candidate, capability, False
            ):
                return candidate, self._get_tts_model_id(candidate)

        return None, "default"

    def _resolve_audio_path(self, audio_input: str) -> str:
        """
        Resolve audio input to appropriate format for audio model.

        Args:
            audio_input: Either a URL string or a local file path

        Returns:
            str: Resolved audio path/URL suitable for the audio model
        """
        # Handle file_id prefix
        if audio_input.startswith("file:") and not audio_input.startswith("file://"):
            audio_input = audio_input[5:].strip()

        # Check if it's a URL (http/https)
        if audio_input.startswith(("http://", "https://")):
            return audio_input

        # Treat as local file path
        if self._workspace:
            try:
                # Use workspace's resolve_path_with_search method for intelligent directory search
                resolved_path = self._workspace.resolve_path_with_search(audio_input)
                logger.info(
                    f"Resolved audio path using workspace search: {audio_input} -> {resolved_path}"
                )
                return str(resolved_path)
            except ValueError as e:
                logger.warning(f"Cannot resolve audio path in workspace: {e}")
                # Fall back to simple path resolution
            except Exception as e:
                logger.warning(f"Error using workspace path resolution: {e}")
                # Fall back to simple path resolution

        # Fallback: simple path resolution
        audio_path = Path(audio_input)

        # If it's a relative path, resolve it relative to current working directory
        if not audio_path.is_absolute():
            audio_path = Path.cwd() / audio_path

        # Convert to absolute path string
        absolute_path = str(audio_path.resolve())

        # Check if file exists
        if not audio_path.exists():
            logger.warning(f"Local audio file not found: {absolute_path}")
        else:
            logger.info(
                f"Resolved audio path using fallback method: {audio_input} -> {absolute_path}"
            )

        return absolute_path

    async def transcribe_audio(
        self,
        audio_file_path: str,
        language: Optional[str] = None,
        model_id: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using ASR.

        Args:
            audio_file_path: Audio file path, file_id, or URL to transcribe
            language: Language code (e.g., 'zh', 'en', 'yue')
            model_id: Specific ASR model to use (optional, uses default if not provided)
            verbose: If True, return detailed result with segments and timing
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary with transcription result containing:
            - success (bool): Whether transcription succeeded
            - file_id (str): File ID for accessing the transcription JSON file
            - transcription_path (str): Path to saved transcription JSON file
            - saved_to_workspace (bool): Whether the transcription was saved
            - segments (list): Detailed segment information (only if verbose=True)
            - language (str): Detected language code
            - model_used (str): The actual model used
            - text_length (int): Length of transcribed text
            - segment_count (int): Number of segments
            - error (str): Error message if success=False

            Note: Complete transcription text is saved in JSON file (use file_id).
            Segments are only in response when verbose=True.
        """
        try:
            # Get the ASR model to use
            asr_model = self._get_asr_model(model_id)

            if not asr_model:
                return {
                    "success": False,
                    "error": "No available ASR models configured",
                    "text": None,
                }

            # Resolve audio path
            audio_path = self._resolve_audio_path(audio_file_path)

            # Transcribe the audio (async)
            result = await asr_model.transcribe(
                audio=audio_path,
                language=language,
                verbose=verbose,
                **kwargs,
            )

            # Determine the actual model used
            actual_model_id = (
                model_id if model_id and model_id in self._asr_models else "default"
            )

            # Handle different result types
            text = None
            segments = None
            language_detected = None

            if isinstance(result, str):
                text = result
            elif isinstance(result, ASRResult):
                text = result.text
                segments = (
                    [
                        {
                            "text": seg.text,
                            "start": seg.start,
                            "end": seg.end,
                            "speaker": seg.speaker,
                            "confidence": seg.confidence,
                        }
                        for seg in result.segments
                    ]
                    if result.segments
                    else None
                )
                language_detected = result.language

            # Merge segments to reduce fragmentation
            if segments:
                merged_segments = self._merge_segments(segments, max_gap=1.0)
                logger.info(
                    f"Merged {len(segments)} segments into {len(merged_segments)} segments"
                )
                segments = merged_segments

            # Save transcription to JSON file if workspace is available
            file_id: Optional[str] = None
            file_ref: Optional[dict[str, Any]] = None
            transcription_path = None

            if text and self._workspace:
                try:
                    # Generate filename for transcription
                    filename = f"transcription_{uuid.uuid4().hex[:8]}.json"

                    # Build structured JSON data
                    transcription_data = {
                        "model": actual_model_id,
                        "language": language_detected,
                        "text": text,
                        "segments": segments,
                        "metadata": {
                            "audio_source": audio_file_path,
                            "verbose_mode": verbose,
                            "total_segments": len(segments) if segments else 0,
                            "segments_merged": True,
                        },
                    }

                    # Register and save file in workspace
                    with self._workspace.auto_register_files():
                        save_path = self._workspace.output_dir / filename

                        # Write transcription to JSON file
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(
                                transcription_data, f, ensure_ascii=False, indent=2
                            )

                        transcription_path = str(save_path)
                        logger.info(f"Saved transcription to: {transcription_path}")

                    # Get file ID from workspace after registration
                    if transcription_path:
                        file_ref = build_workspace_file_ref(
                            workspace=self._workspace,
                            file_path=transcription_path,
                        )
                        file_id = file_ref["file_id"]

                except Exception as e:
                    logger.warning(f"Failed to save transcription to workspace: {e}")
            elif text and not self._workspace:
                logger.warning(
                    "No workspace available, transcription not saved locally"
                )

            return {
                "success": True,
                "file_id": file_id,
                "file_ref": file_ref,
                "transcription_path": transcription_path,
                "segments": segments,
                "language": language_detected,
                "model_used": actual_model_id,
                "saved_to_workspace": transcription_path is not None,
                "text_length": len(text) if text else 0,
                "segment_count": len(segments) if segments else 0,
            }

        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            actual_model_id = (
                model_id if model_id and model_id in self._asr_models else "default"
            )
            return {
                "success": False,
                "error": str(e),
                "file_id": None,
                "transcription_path": None,
                "model_used": actual_model_id,
            }

    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        audio_format: str = "mp3",
        sample_rate: Optional[int] = None,
        reference_audio: Optional[str] = None,
        voice_settings: Optional[Dict[str, Any]] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text using TTS.

        Args:
            text: Input text to synthesize
            voice: Provider-specific voice identifier (optional). Never invent it.
            language: Language code (optional)
            audio_format: Output audio format (default: 'mp3')
            sample_rate: Sample rate in Hz (optional)
            reference_audio: Reference audio path or file ID for voice cloning (optional)
            voice_settings: Provider-specific voice shaping settings
            provider_options: Provider-specific synthesis parameters
            model_id: Specific TTS model to use (optional, uses default if not provided)
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary with synthesis result containing:
            - success (bool): Whether synthesis succeeded
            - audio_path (str): Path to generated audio file
            - file_id (str): File ID for accessing the audio file
            - format (str): Audio format (e.g., 'mp3', 'wav')
            - sample_rate (int): Audio sample rate
            - language (str): Detected/specified language
            - model_used (str): The actual model used for synthesis
            - saved_to_workspace (bool): Whether the audio was saved to workspace
            - error (str): Error message if success=False
        """
        try:
            # Get the TTS model to use
            tts_model = self._get_tts_model(model_id)

            if not tts_model:
                return {
                    "success": False,
                    "error": "No available TTS models configured",
                    "audio_path": None,
                }

            self._validate_tts_reference_audio(
                tts_model=tts_model,
                reference_audio=reference_audio,
            )
            synthesis_kwargs = self._build_tts_provider_kwargs(
                tts_model=tts_model,
                voice_settings=voice_settings,
                provider_options=provider_options,
                extra_options=kwargs,
            )
            if sample_rate is not None:
                synthesis_kwargs["sample_rate"] = sample_rate
            if reference_audio:
                synthesis_kwargs["reference_audio"] = (
                    self._resolve_audio_path(reference_audio)
                    if self._workspace is not None
                    else reference_audio
                )

            # Synthesize the speech (async)
            result = await tts_model.synthesize(
                text=text,
                voice=voice,
                language=language,
                format=audio_format,
                **synthesis_kwargs,
            )

            # Determine the actual model used
            actual_model_id = (
                model_id if model_id and model_id in self._tts_models else "default"
            )

            audio_data: Optional[bytes] = None
            result_audio_format: Optional[str] = None
            result_sample_rate: Optional[int] = None
            language_detected: Optional[str] = None

            # Handle different result types
            if isinstance(result, bytes):
                audio_data = result
                result_audio_format = audio_format
            elif isinstance(result, TTSResult):
                audio_data = result.audio
                result_audio_format = result.format
                result_sample_rate = result.sample_rate
                language_detected = result.language

            # Save audio file to workspace if available
            audio_path = None
            audio_file_id: Optional[str] = None
            file_ref: Optional[dict[str, Any]] = None

            if audio_data and self._workspace:
                try:
                    # Generate filename
                    filename = f"synthesized_speech_{uuid.uuid4().hex[:8]}.{result_audio_format or 'mp3'}"

                    # Register and save audio file in workspace
                    with self._workspace.auto_register_files():
                        save_path = self._workspace.output_dir / filename

                        # Write audio data
                        with open(save_path, "wb") as f:
                            f.write(audio_data)

                        audio_path = str(save_path)
                        logger.info(f"Saved synthesized audio to: {audio_path}")

                    # Get file ID from workspace after registration
                    if audio_path:
                        file_ref = build_workspace_file_ref(
                            workspace=self._workspace,
                            file_path=audio_path,
                        )
                        audio_file_id = file_ref["file_id"]

                except Exception as e:
                    logger.warning(f"Failed to save audio to workspace: {e}")
                    # Continue execution even if save fails
            elif audio_data and not self._workspace:
                logger.warning("No workspace available, audio not saved locally")

            return {
                "success": True,
                "audio_path": audio_path,
                "file_id": audio_file_id,
                "file_ref": file_ref,
                "format": result_audio_format,
                "sample_rate": result_sample_rate,
                "language": language_detected,
                "model_used": actual_model_id,
                "saved_to_workspace": audio_path is not None,
            }

        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            actual_model_id = (
                model_id if model_id and model_id in self._tts_models else "default"
            )
            return {
                "success": False,
                "error": str(e),
                "audio_path": None,
                "model_used": actual_model_id,
            }

    def list_available_models(self) -> Dict[str, Any]:
        """
        List all available audio models (ASR and TTS).

        Returns:
            Dictionary containing:
            - success (bool): Whether operation succeeded
            - asr_models (list): List of ASR model information
            - tts_models (list): List of TTS model information
            - default_asr_model (str): Default ASR model ID (if set)
            - default_tts_model (str): Default TTS model ID (if set)

            Each model info contains: type, model_id, available, description
        """
        try:
            asr_models_info = []
            for model_id in self._asr_models.keys():
                model_info = {
                    "type": "asr",
                    "model_id": model_id,
                    "available": True,
                    "description": self._model_descriptions.get(model_id, ""),
                }
                asr_models_info.append(model_info)

            tts_models_info = []
            for model_id, tts_model in self._tts_models.items():
                provider_name = self._get_tts_provider_name(tts_model)
                model_info = {
                    "type": "tts",
                    "model_id": model_id,
                    "provider": provider_name,
                    "available": True,
                    "description": self._model_descriptions.get(model_id, ""),
                    "abilities": list(getattr(tts_model, "abilities", [])),
                    "supports_multiple_voices": bool(
                        getattr(tts_model, "supports_multiple_voices", False)
                    ),
                    "supports_voice_listing": bool(
                        getattr(tts_model, "supports_voice_listing", False)
                    ),
                    "supports_voice_settings": bool(
                        getattr(tts_model, "supports_voice_settings", False)
                    ),
                    "supports_voice_cloning": bool(
                        getattr(tts_model, "supports_voice_cloning", False)
                    ),
                    "supports_persistent_voice_cloning": bool(
                        getattr(tts_model, "supports_persistent_voice_cloning", False)
                    ),
                    "supported_voice_settings": list(
                        getattr(tts_model, "supported_voice_settings", [])
                    ),
                    "supported_provider_options": list(
                        getattr(tts_model, "supported_provider_options", [])
                    ),
                }
                tts_models_info.append(model_info)

            all_models_info = asr_models_info + tts_models_info

            return {
                "success": True,
                "models": all_models_info,
                "asr_count": len(asr_models_info),
                "tts_count": len(tts_models_info),
                "total_count": len(all_models_info),
            }

        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
            return {
                "success": False,
                "error": str(e),
                "models": [],
                "asr_count": 0,
                "tts_count": 0,
                "total_count": 0,
            }

    async def list_tts_voices(
        self,
        provider: Literal["elevenlabs"] = "elevenlabs",
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List available TTS voices for providers that support dynamic voice lookup.

        Args:
            provider: TTS voice provider. Currently only "elevenlabs".
            model_id: Provider model configuration to use.

        Returns:
            Dictionary containing:
            - success (bool): Whether voice listing succeeded
            - supported (bool): Whether the selected provider supports voice listing
            - voices (list): Normalized voice metadata
            - count (int): Number of voices returned
            - provider (str): Selected provider name
            - supported_providers (list): Providers that currently support this API
        """
        supported_providers = self._get_voice_listing_supported_providers()
        try:
            tts_model, actual_model_id = self._get_provider_tts_model(
                provider=provider,
                capability="supports_voice_listing",
                model_id=model_id,
            )

            if not tts_model:
                return {
                    "success": False,
                    "supported": False,
                    "error": f"No {provider} TTS model is configured",
                    "voices": [],
                    "count": 0,
                    "model_used": actual_model_id,
                    "supported_providers": supported_providers,
                }

            provider_name = self._get_tts_provider_name(tts_model)
            if provider_name != provider:
                return {
                    "success": False,
                    "supported": False,
                    "error": (
                        f"list_tts_voices provider is '{provider}', but model "
                        f"'{actual_model_id}' uses provider '{provider_name}'."
                    ),
                    "voices": [],
                    "count": 0,
                    "provider": provider_name,
                    "model_used": actual_model_id,
                    "supported_providers": supported_providers,
                }
            if not getattr(tts_model, "supports_voice_listing", False):
                return {
                    "success": False,
                    "supported": False,
                    "error": (
                        "Dynamic TTS voice listing is not supported for provider "
                        f"'{provider_name}'. Currently supported providers: "
                        f"{', '.join(supported_providers)}."
                    ),
                    "voices": [],
                    "count": 0,
                    "provider": provider_name,
                    "model_used": actual_model_id,
                    "supported_providers": supported_providers,
                }

            voices = await tts_model.list_available_voices()
            configured_model_ids = self._get_configured_tts_model_ids(provider_name)
            voices = self._filter_voice_model_metadata(voices, configured_model_ids)
            return {
                "success": True,
                "supported": True,
                "voices": voices,
                "count": len(voices),
                "provider": provider_name,
                "model_used": actual_model_id,
                "supported_providers": supported_providers,
            }

        except Exception as e:
            logger.error(f"Failed to list TTS voices: {e}")
            actual_model_id = (
                model_id if model_id and model_id in self._tts_models else "default"
            )
            return {
                "success": False,
                "supported": False,
                "error": str(e),
                "voices": [],
                "count": 0,
                "model_used": actual_model_id,
                "supported_providers": supported_providers,
            }

    async def clone_tts_voice(
        self,
        name: str,
        reference_audio_files: List[str],
        provider: Literal["elevenlabs"] = "elevenlabs",
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        remove_background_noise: bool = False,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a persistent voice clone with a configured provider account."""
        try:
            tts_model, actual_model_id = self._get_provider_tts_model(
                provider=provider,
                capability="supports_persistent_voice_cloning",
                model_id=model_id,
            )
            if not tts_model:
                return {
                    "success": False,
                    "supported": False,
                    "error": f"No {provider} TTS model is configured",
                    "provider": provider,
                    "model_used": actual_model_id,
                }

            provider_name = self._get_tts_provider_name(tts_model)
            if provider_name != provider:
                return {
                    "success": False,
                    "supported": False,
                    "error": (
                        f"clone_tts_voice provider is '{provider}', but model "
                        f"'{actual_model_id}' uses provider '{provider_name}'."
                    ),
                    "provider": provider_name,
                    "model_used": actual_model_id,
                }
            if not getattr(tts_model, "supports_persistent_voice_cloning", False):
                return {
                    "success": False,
                    "supported": False,
                    "error": f"The configured {provider} client does not support persistent voice cloning",
                    "provider": provider_name,
                    "model_used": actual_model_id,
                }

            resolved_audio_files = [
                self._resolve_audio_path(reference_audio)
                if self._workspace is not None
                else reference_audio
                for reference_audio in reference_audio_files
            ]
            clone = await tts_model.clone_voice(
                name=name,
                reference_audio_files=resolved_audio_files,
                description=description,
                labels=labels,
                remove_background_noise=remove_background_noise,
            )
            return {
                "success": True,
                "supported": True,
                **clone,
                "model_used": actual_model_id,
            }
        except Exception as e:
            logger.error(f"Failed to clone TTS voice: {e}")
            actual_model_id = (
                model_id if model_id and model_id in self._tts_models else "default"
            )
            return {
                "success": False,
                "supported": True,
                "error": str(e),
                "provider": provider,
                "model_used": actual_model_id,
            }

    async def delete_tts_voice(
        self,
        voice_id: str,
        provider: Literal["elevenlabs"] = "elevenlabs",
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Permanently delete a voice from a configured provider account."""
        try:
            tts_model, actual_model_id = self._get_provider_tts_model(
                provider=provider,
                capability="supports_persistent_voice_cloning",
                model_id=model_id,
            )
            if not tts_model:
                return {
                    "success": False,
                    "supported": False,
                    "error": f"No {provider} TTS model is configured",
                    "provider": provider,
                    "model_used": actual_model_id,
                }

            provider_name = self._get_tts_provider_name(tts_model)
            if provider_name != provider:
                return {
                    "success": False,
                    "supported": False,
                    "error": (
                        f"delete_tts_voice provider is '{provider}', but model "
                        f"'{actual_model_id}' uses provider '{provider_name}'."
                    ),
                    "provider": provider_name,
                    "model_used": actual_model_id,
                }
            if not getattr(tts_model, "supports_persistent_voice_cloning", False):
                return {
                    "success": False,
                    "supported": False,
                    "error": f"The configured {provider} client does not support persistent voice deletion",
                    "provider": provider_name,
                    "model_used": actual_model_id,
                }

            normalized_voice_id = voice_id.strip()
            await tts_model.delete_voice(normalized_voice_id)
            return {
                "success": True,
                "supported": True,
                "deleted": True,
                "voice_id": normalized_voice_id,
                "provider": provider_name,
                "model_used": actual_model_id,
            }
        except Exception as e:
            logger.error(f"Failed to delete TTS voice: {e}")
            actual_model_id = (
                model_id if model_id and model_id in self._tts_models else "default"
            )
            return {
                "success": False,
                "supported": True,
                "deleted": False,
                "error": str(e),
                "provider": provider,
                "model_used": actual_model_id,
            }

    async def synthesize_speech_json(
        self,
        json_data: Optional[str | Dict[str, Any]] = None,
        file_id: Optional[str] = None,  # Can be file_id, file path, or URL
        segments_field: str = "segments",
        text_field: str = "text",
        voice_field: str = "voice",
        reference_field: str = "reference_audio",
        voice_settings_field: str = "voice_settings",
        provider_options_field: str = "provider_options",
        default_voice: Optional[str] = None,
        default_language: Optional[str] = None,
        default_voice_settings: Optional[Dict[str, Any]] = None,
        default_provider_options: Optional[Dict[str, Any]] = None,
        audio_format: str = "mp3",
        sample_rate: Optional[int] = None,
        model_id: Optional[str] = None,
        batch_size: int = 5,
    ) -> Dict[str, Any]:
        """
        Batch synthesize speech from JSON structure using TTS.

        Supports flexible JSON format with configurable field mapping and voice cloning.

        Args:
            json_data: JSON string or dict containing synthesis configuration
            file_id: File ID, file path, or URL to read JSON data from (alternative to json_data)
            segments_field: Field name containing segments array (default: "segments")
            text_field: Field name containing text within each segment (default: "text")
            voice_field: Field name containing voice within each segment (default: "voice")
            reference_field: Field name containing reference audio ID (default: "reference_audio_id")
            voice_settings_field: Field name containing provider voice settings
            provider_options_field: Field name containing provider synthesis options
            default_voice: Provider-specific voice identifier for segments without one
            default_language: Default language code (auto-detect if None)
            default_voice_settings: Default provider voice settings for all segments
            default_provider_options: Default provider synthesis options for all segments
            audio_format: Output audio format (default: 'mp3')
            sample_rate: Sample rate in Hz (default: model-specific)
            model_id: Specific TTS model to use
            batch_size: Number of syntheses to process in parallel (1-20, default: 5)

        Returns:
            Dictionary with batch synthesis result containing:
            - success (bool): Whether all syntheses succeeded
            - results (list): List of synthesis results, one per segment
            - total (int): Total number of segments processed
            - successful (int): Number of successful syntheses
            - failed (int): Number of failed syntheses
            - errors (list): List of error messages for failed segments
            - saved_to_workspace (bool): Whether audio files were saved to workspace

        JSON Format Example:
            {
                "segments": [
                    {
                        "text": "你好世界",
                        "reference_audio_id": "ref_voice_1"
                    },
                    {
                        "text": "这是一个测试",
                        "reference_audio_id": "ref_voice_2"
                    }
                ],
                "output_format": "mp3",
                "sample_rate": 24000
            }

        Example:
            >>> # Batch synthesis with voice cloning
            >>> data = {
            ...     "segments": [
            ...         {"text": "你好", "reference_audio_id": "ref1"},
            ...         {"text": "世界", "reference_audio_id": "ref2"}
            ...     ]
            ... }
            >>> result = await synthesize_speech_json(json_data=data)
            >>> print(f"Synthesized {result['successful']}/{result['total']} segments")
        """
        # Validate that either json_data or file_id is provided
        if json_data is None and file_id is None:
            return {
                "success": False,
                "error": "Either json_data or file_id must be provided",
                "results": [],
                "total": 0,
                "successful": 0,
                "failed": 0,
                "errors": ["Either json_data or file_id must be provided"],
            }

        # Parse JSON input
        if json_data is not None and isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON: {e}",
                    "results": [],
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "errors": [str(e)],
                }
        elif json_data is not None:
            # json_data is already a dict
            data = json_data

        # Read from file_id if provided (takes precedence over json_data)
        # file_id can be: file_id, file path, or URL
        if file_id is not None:
            try:
                # Check if it's a URL
                if file_id.startswith(("http://", "https://")):
                    # Download JSON from URL
                    import httpx

                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(file_id)
                        response.raise_for_status()
                        json_content = response.text
                        data = json.loads(json_content)
                    logger.info(f"Downloaded JSON data from URL: {file_id}")

                elif self._workspace:
                    # Try to resolve as file_id first
                    file_path = self._workspace.resolve_file_id(file_id)
                    if file_path and file_path.exists():
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        logger.info(f"Read JSON data from file_id: {file_id}")
                    else:
                        # Try as file path
                        try:
                            # Use workspace path resolution with search
                            resolved_path = self._workspace.resolve_path_with_search(
                                file_id
                            )
                            if resolved_path.exists():
                                with open(resolved_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                logger.info(
                                    f"Read JSON data from file path: {resolved_path}"
                                )
                            else:
                                return {
                                    "success": False,
                                    "error": f"File not found: {file_id}",
                                    "results": [],
                                    "total": 0,
                                    "successful": 0,
                                    "failed": 0,
                                    "errors": [f"File not found: {file_id}"],
                                }
                        except ValueError:
                            # resolve_path_with_search failed, try direct path
                            file_path = Path(file_id)
                            if not file_path.is_absolute():
                                file_path = Path.cwd() / file_path
                            if file_path.exists():
                                with open(file_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                logger.info(
                                    f"Read JSON data from direct path: {file_path}"
                                )
                            else:
                                return {
                                    "success": False,
                                    "error": f"File not found: {file_id}",
                                    "results": [],
                                    "total": 0,
                                    "successful": 0,
                                    "failed": 0,
                                    "errors": [f"File not found: {file_id}"],
                                }
                else:
                    # No workspace, try direct file path
                    file_path = Path(file_id)
                    if not file_path.is_absolute():
                        file_path = Path.cwd() / file_path
                    if file_path.exists():
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        logger.info(f"Read JSON data from file path: {file_path}")
                    else:
                        return {
                            "success": False,
                            "error": f"File not found: {file_id}",
                            "results": [],
                            "total": 0,
                            "successful": 0,
                            "failed": 0,
                            "errors": [f"File not found: {file_id}"],
                        }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read file {file_id}: {e}",
                    "results": [],
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "errors": [str(e)],
                }

        if not isinstance(data, dict):
            return {
                "success": False,
                "error": "JSON data must be an object",
                "results": [],
                "total": 0,
                "successful": 0,
                "failed": 0,
                "errors": ["JSON data must be an object"],
            }

        if default_voice_settings is None:
            root_voice_settings = data.get("default_voice_settings")
            if root_voice_settings is not None:
                if not isinstance(root_voice_settings, dict):
                    return {
                        "success": False,
                        "error": "default_voice_settings must be an object",
                        "results": [],
                        "total": 0,
                        "successful": 0,
                        "failed": 0,
                        "errors": ["default_voice_settings must be an object"],
                    }
                default_voice_settings = root_voice_settings

        if default_provider_options is None:
            root_provider_options = data.get("default_provider_options")
            if root_provider_options is not None:
                if not isinstance(root_provider_options, dict):
                    return {
                        "success": False,
                        "error": "default_provider_options must be an object",
                        "results": [],
                        "total": 0,
                        "successful": 0,
                        "failed": 0,
                        "errors": ["default_provider_options must be an object"],
                    }
                default_provider_options = root_provider_options

        if default_voice is None and data.get("default_voice") is not None:
            default_voice = str(data["default_voice"])

        if default_language is None and data.get("default_language") is not None:
            default_language = str(data["default_language"])

        if audio_format == "mp3" and data.get("output_format") is not None:
            audio_format = str(data["output_format"])

        if sample_rate is None and data.get("sample_rate") is not None:
            try:
                sample_rate = int(data["sample_rate"])
            except (TypeError, ValueError):
                return {
                    "success": False,
                    "error": "sample_rate must be an integer",
                    "results": [],
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "errors": ["sample_rate must be an integer"],
                }

        # Extract segments from JSON
        segments = data.get(segments_field, [])
        if not segments:
            return {
                "success": False,
                "error": f"No segments found in field '{segments_field}'",
                "results": [],
                "total": 0,
                "successful": 0,
                "failed": 0,
                "errors": [f"No segments found in field '{segments_field}'"],
            }

        total = len(segments)
        results = []
        errors = []
        successful_count = 0
        failed_count = 0

        # Get TTS model
        tts_model = self._get_tts_model(model_id)
        if not tts_model:
            return {
                "success": False,
                "error": "No available TTS models configured",
                "results": [],
                "total": total,
                "successful": 0,
                "failed": total,
                "errors": ["No TTS models configured"] * total,
            }

        # Process segments in batches with progress tracking
        import asyncio

        from tqdm.asyncio import tqdm as tqdm_async  # type: ignore[import-untyped]

        batches = [segments[i : i + batch_size] for i in range(0, total, batch_size)]

        if len(batches) == 1:
            # Single batch: direct processing
            logger.info(f"Synthesizing single batch of {total} segments")

            for idx, segment in enumerate(segments):
                result = await self._synthesize_single_segment(
                    segment,
                    text_field,
                    voice_field,
                    reference_field,
                    voice_settings_field,
                    provider_options_field,
                    default_voice,
                    default_language,
                    default_voice_settings,
                    default_provider_options,
                    audio_format,
                    sample_rate,
                    tts_model,
                    idx,
                )
                results.append(result)
                if result["success"]:
                    successful_count += 1
                else:
                    failed_count += 1
                    if result.get("error"):
                        errors.append(result["error"])

        else:
            # Multiple batches: parallel processing with progress
            logger.info(
                f"Synthesizing {total} segments in {len(batches)} parallel batches (batch_size={batch_size})"
            )

            async def process_batch(
                batch_texts: List[Dict[str, Any]], batch_index: int
            ) -> List[Dict[str, Any]]:
                """Process a batch of segments"""
                batch_results = []
                for segment in batch_texts:
                    idx = segments.index(segment)  # Get original index
                    result = await self._synthesize_single_segment(
                        segment,
                        text_field,
                        voice_field,
                        reference_field,
                        voice_settings_field,
                        provider_options_field,
                        default_voice,
                        default_language,
                        default_voice_settings,
                        default_provider_options,
                        audio_format,
                        sample_rate,
                        tts_model,
                        idx,
                    )
                    batch_results.append(result)
                return batch_results

            with tqdm_async(
                total=len(batches),
                desc="TTS batches",
                unit="batch",
                colour="green",
            ) as pbar:

                async def process_batch_with_progress(
                    batch: List[Dict[str, Any]], idx: int
                ) -> List[Dict[str, Any]]:
                    result = await process_batch(batch, idx)
                    pbar.update(1)
                    pbar.set_postfix(
                        {"batch": f"{idx + 1}/{len(batches)}", "segments": len(batch)}
                    )
                    return result

                tasks = [
                    process_batch_with_progress(batch, i)
                    for i, batch in enumerate(batches)
                ]

                batch_results = await asyncio.gather(*tasks)

            # Flatten batch results
            for batch_result in batch_results:
                for result in batch_result:
                    results.append(result)
                    if result["success"]:
                        successful_count += 1
                    else:
                        failed_count += 1
                        if result.get("error"):
                            errors.append(result["error"])

        return {
            "success": failed_count == 0,
            "results": results,
            "total": total,
            "successful": successful_count,
            "failed": failed_count,
            "errors": errors if errors else None,
            "saved_to_workspace": self._workspace is not None,
        }

    async def _synthesize_single_segment(
        self,
        segment: Dict[str, Any],
        text_field: str,
        voice_field: str,
        reference_field: str,
        voice_settings_field: str,
        provider_options_field: str,
        default_voice: Optional[str],
        default_language: Optional[str],
        default_voice_settings: Optional[Dict[str, Any]],
        default_provider_options: Optional[Dict[str, Any]],
        audio_format: str,
        sample_rate: Optional[int],
        tts_model: Any,
        index: int,
    ) -> Dict[str, Any]:
        """
        Synthesize speech for a single segment.

        Args:
            segment: Segment dictionary containing synthesis parameters
            text_field: Field name for text content
            voice_field: Field name for voice
            reference_field: Field name for reference audio ID
            voice_settings_field: Field name for provider voice settings
            provider_options_field: Field name for provider synthesis options
            default_voice: Default voice if not specified in segment
            default_language: Default language if not specified
            default_voice_settings: Default provider voice settings
            default_provider_options: Default provider synthesis options
            audio_format: Audio format
            sample_rate: Sample rate
            tts_model: TTS model instance
            index: Segment index for error reporting

        Returns:
            Dictionary with synthesis result
        """
        try:
            # Extract parameters from segment
            text = segment.get(text_field)
            if not text:
                return {
                    "success": False,
                    "error": f"Segment {index}: No text found in field '{text_field}'",
                    "index": index,
                }

            voice = segment.get(voice_field, default_voice)
            language = segment.get("language", default_language)
            voice_settings = self._merge_option_dicts(
                default_options=default_voice_settings,
                override_options=segment.get(voice_settings_field),
                default_field_name="default_voice_settings",
                override_field_name=voice_settings_field,
            )
            provider_options = self._merge_option_dicts(
                default_options=default_provider_options,
                override_options=segment.get(provider_options_field),
                default_field_name="default_provider_options",
                override_field_name=provider_options_field,
                reserved_keys={"text", "voice", "language", "format", "sample_rate"},
            )

            # Validate reference audio field names
            # Check if user provided common alternative field names
            if (
                "reference_audio_id" in segment
                and reference_field != "reference_audio_id"
            ):
                return {
                    "success": False,
                    "error": f"Segment {index}: Found 'reference_audio_id' field but tool expects '{reference_field}'. Please use '{reference_field}' or set reference_field='reference_audio_id' parameter.",
                    "index": index,
                }

            ref_audio_id = segment.get(reference_field)
            self._validate_tts_reference_audio(
                tts_model=tts_model,
                reference_audio=ref_audio_id,
            )

            # Build synthesis parameters
            kwargs = self._build_tts_provider_kwargs(
                tts_model=tts_model,
                voice_settings=voice_settings or None,
                provider_options=provider_options or None,
            )
            kwargs["format"] = audio_format
            if sample_rate:
                kwargs["sample_rate"] = sample_rate

            # Handle reference audio for voice cloning
            if ref_audio_id:
                ref_audio_path = None

                # Try to resolve as file_id first (if workspace available)
                if self._workspace:
                    try:
                        resolved_path = self._workspace.resolve_file_id(ref_audio_id)
                        if resolved_path and resolved_path.exists():
                            ref_audio_path = resolved_path
                    except Exception:
                        pass  # Not a file_id, try as direct path

                # If not found as file_id, try as direct file path
                if not ref_audio_path:
                    direct_path = Path(ref_audio_id)
                    if direct_path.exists():
                        ref_audio_path = direct_path
                    elif direct_path.is_absolute():
                        # Absolute path but doesn't exist
                        logger.warning(
                            f"Reference audio file not found: {ref_audio_id}"
                        )
                    else:
                        # Relative path, try current directory
                        resolved_cwd = Path.cwd() / direct_path
                        if resolved_cwd.exists():
                            ref_audio_path = resolved_cwd
                        else:
                            logger.warning(
                                f"Reference audio file not found: {ref_audio_id}"
                            )

                # Pass reference audio path to TTS model
                if ref_audio_path:
                    kwargs["reference_audio"] = str(ref_audio_path)

            # Synthesize speech
            audio_data = await tts_model.synthesize(
                text=text,
                voice=voice,
                language=language,
                **kwargs,
            )

            # Handle result
            if isinstance(audio_data, bytes):
                audio_binary = audio_data
                # audio_format remains as the function parameter
            else:
                # Assume it's TTSResult
                audio_binary = audio_data.audio
                audio_format = audio_data.format

            # Save to workspace if available
            audio_path = None
            audio_file_id = None
            file_ref: Optional[dict[str, Any]] = None

            if self._workspace:
                try:
                    filename = f"synthesized_speech_{index}_{uuid.uuid4().hex[:8]}.{audio_format or 'mp3'}"

                    with self._workspace.auto_register_files():
                        save_path = self._workspace.output_dir / filename

                        with open(save_path, "wb") as f:
                            f.write(audio_binary)

                        audio_path = str(save_path)
                        logger.info(f"Saved synthesized audio to: {audio_path}")

                    # Get file ID
                    if audio_path:
                        file_ref = build_workspace_file_ref(
                            workspace=self._workspace,
                            file_path=audio_path,
                        )
                        audio_file_id = file_ref["file_id"]

                except Exception as e:
                    logger.warning(f"Failed to save audio to workspace: {e}")

            return {
                "success": True,
                "index": index,
                "text": text,
                "voice": voice,
                "audio_path": audio_path,
                "file_id": audio_file_id,
                "file_ref": file_ref,
                "format": audio_format,
                "saved_to_workspace": audio_path is not None,
            }

        except Exception as e:
            logger.error(f"Segment {index} synthesis failed: {e}")
            return {
                "success": False,
                "error": f"Segment {index}: {str(e)}",
                "index": index,
                "text": text,
                "saved_to_workspace": False,
            }
