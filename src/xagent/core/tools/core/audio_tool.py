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
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...model.asr.base import ASRResult, BaseASR
from ...model.tts.base import BaseTTS, TTSResult
from ...workspace import TaskWorkspace
from .audio_tool_descriptions import (
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
        self._generate_model_info_text()

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
                        file_id = self._workspace.get_file_id_from_path(
                            transcription_path
                        )

                except Exception as e:
                    logger.warning(f"Failed to save transcription to workspace: {e}")
            elif text and not self._workspace:
                logger.warning(
                    "No workspace available, transcription not saved locally"
                )

            return {
                "success": True,
                "file_id": file_id,
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
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text using TTS.

        Args:
            text: Input text to synthesize
            voice: Voice ID or name (optional)
            language: Language code (optional)
            audio_format: Output audio format (default: 'mp3')
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

            # Synthesize the speech (async)
            result = await tts_model.synthesize(
                text=text,
                voice=voice,
                language=language,
                format=audio_format,
                **kwargs,
            )

            # Determine the actual model used
            actual_model_id = (
                model_id if model_id and model_id in self._tts_models else "default"
            )

            audio_data: Optional[bytes] = None
            result_audio_format: Optional[str] = None
            sample_rate: Optional[int] = None
            language_detected: Optional[str] = None

            # Handle different result types
            if isinstance(result, bytes):
                audio_data = result
                result_audio_format = audio_format
            elif isinstance(result, TTSResult):
                audio_data = result.audio
                result_audio_format = result.format
                sample_rate = result.sample_rate
                language_detected = result.language

            # Save audio file to workspace if available
            audio_path = None
            audio_file_id: Optional[str] = None

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
                        audio_file_id = self._workspace.get_file_id_from_path(
                            audio_path
                        )

                except Exception as e:
                    logger.warning(f"Failed to save audio to workspace: {e}")
                    # Continue execution even if save fails
            elif audio_data and not self._workspace:
                logger.warning("No workspace available, audio not saved locally")

            return {
                "success": True,
                "audio_path": audio_path,
                "file_id": audio_file_id,
                "format": result_audio_format,
                "sample_rate": sample_rate,
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
            for model_id in self._tts_models.keys():
                model_info = {
                    "type": "tts",
                    "model_id": model_id,
                    "available": True,
                    "description": self._model_descriptions.get(model_id, ""),
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

    async def synthesize_speech_json(
        self,
        json_data: Optional[str | Dict[str, Any]] = None,
        file_id: Optional[str] = None,  # Can be file_id, file path, or URL
        segments_field: str = "segments",
        text_field: str = "text",
        voice_field: str = "voice",
        reference_field: str = "reference_audio",
        default_voice: Optional[str] = None,
        default_language: Optional[str] = None,
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
            default_voice: Default voice for segments without voice specified
            default_language: Default language code (auto-detect if None)
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
                        "voice": "zh-female",
                        "reference_audio_id": "ref_voice_1"
                    },
                    {
                        "text": "这是一个测试",
                        "voice": "zh-male",
                        "reference_audio_id": "ref_voice_2"
                    }
                ],
                "default_voice": "zh-female",
                "output_format": "mp3",
                "sample_rate": 24000
            }

        Example:
            >>> # Batch synthesis with voice cloning
            >>> data = {
            ...     "segments": [
            ...         {"text": "你好", "voice": "zh-female", "reference_audio_id": "ref1"},
            ...         {"text": "世界", "voice": "zh-male"}
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
                    default_voice,
                    default_language,
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
                        default_voice,
                        default_language,
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
        default_voice: Optional[str],
        default_language: Optional[str],
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
            default_voice: Default voice if not specified in segment
            default_language: Default language if not specified
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

            # Build synthesis parameters
            kwargs: Dict[str, Any] = {"format": audio_format}
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
                        audio_file_id = self._workspace.get_file_id_from_path(
                            audio_path
                        )

                except Exception as e:
                    logger.warning(f"Failed to save audio to workspace: {e}")

            return {
                "success": True,
                "index": index,
                "text": text,
                "voice": voice,
                "audio_path": audio_path,
                "file_id": audio_file_id,
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
