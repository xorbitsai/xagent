"""
Pure Vision Tool Core
Standalone vision capabilities without framework dependencies
"""

import asyncio
import base64
import logging
import math
import mimetypes
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from ...model.chat.basic.base import BaseLLM

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

_MAX_INLINE_VIDEO_BYTES = 64 * 1024 * 1024


class UnderstandMediaResult(BaseModel):
    """Result returned by the unified image and video understanding entrypoint."""

    success: bool
    answer: Optional[str] = None
    media_processed: Optional[int] = None
    images_processed: Optional[int] = None
    videos_processed: Optional[int] = None
    native_videos_processed: Optional[int] = None
    frames_extracted: Optional[int] = None
    model_used: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


# Backwards-compatible name for direct/internal callers. The agent-facing tool is
# ``understand_media`` so image and video understanding do not consume two tool
# slots.
UnderstandImagesResult = UnderstandMediaResult


class DetectObjectsResult(BaseModel):
    """Return model for detect_objects method"""

    success: bool
    detections: List[Dict[str, Any]] = []
    total_detections: int = 0
    image_processed: Optional[str] = None
    confidence_threshold: float = 0.5
    prompt_sent: Optional[str] = None
    marked_image_path: Optional[str] = None
    file_id: Optional[str] = None
    file_ref: Optional[Dict[str, Any]] = None
    box_color: Optional[str] = None
    raw_response: Optional[str] = None
    parsing_method: Optional[str] = None
    error: Optional[str] = None


class VisionCore:
    """
    Core vision functionality using vision-enabled LLM models.
    No framework or workspace dependencies.
    """

    def __init__(self, vision_model: BaseLLM, output_directory: Optional[str] = None):
        """
        Initialize with a vision-enabled LLM model.

        Args:
            vision_model: LLM model with vision capabilities
            output_directory: Optional directory for saving marked images
        """
        self.vision_model = vision_model
        self.output_directory = (
            Path(output_directory) if output_directory else Path("./output")
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _coerce_optional_float(self, value: Any, field_name: str) -> Optional[float]:
        """Normalize optional numeric tool arguments before model calls."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be a number")
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in {"", "none", "null"}:
                return None

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a number") from exc

    def _coerce_optional_int(self, value: Any, field_name: str) -> Optional[int]:
        """Normalize optional integer tool arguments before model calls."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be an integer")
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in {"", "none", "null"}:
                return None

        try:
            parsed_float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer") from exc

        if not parsed_float.is_integer():
            raise ValueError(f"{field_name} must be an integer")

        return int(parsed_float)

    def _convert_image_to_base64(self, image_path: str) -> str:
        """
        Convert image to base64 format for LLM vision chat.

        Args:
            image_path: Path to image file or URL

        Returns:
            Base64 encoded image string with MIME type prefix
        """
        # If it's already a URL, return as-is
        if image_path.startswith(("http://", "https://")):
            return image_path

        # Convert to absolute path if relative
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)

        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"  # Default

        # Read and encode file
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_data = base64.b64encode(image_data).decode("utf-8")
                return f"data:{mime_type};base64,{base64_data}"
        except Exception as e:
            raise RuntimeError(f"Failed to read image file {image_path}: {e}")

    def _convert_video_to_base64(self, video_path: str) -> str:
        """Convert a local video to a provider-neutral Base64 data URL."""
        if video_path.startswith(("http://", "https://", "data:")):
            return video_path

        resolved_path = Path(video_path).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if resolved_path.stat().st_size > _MAX_INLINE_VIDEO_BYTES:
            raise ValueError(
                "Local video is too large for inline native input; "
                "falling back to sampled frames"
            )

        mime_type = mimetypes.guess_type(str(resolved_path))[0] or "video/mp4"
        if not mime_type.startswith("video/"):
            raise ValueError(f"Unsupported video MIME type: {mime_type}")

        try:
            encoded = base64.b64encode(resolved_path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read video file {video_path}: {exc}"
            ) from exc
        return f"data:{mime_type};base64,{encoded}"

    def _validate_images(self, images: Union[str, List[str]]) -> List[str]:
        """
        Validate and normalize image inputs.

        Args:
            images: Single image path/URL or list of image paths/URLs

        Returns:
            List of validated image paths/URLs
        """
        if isinstance(images, str):
            images = [images]

        if not images:
            raise ValueError("At least one image must be provided")

        if len(images) > 10:  # Limit to prevent abuse
            raise ValueError("Maximum 10 images can be analyzed at once")

        return images

    def _validate_media(self, media: Union[str, List[str]]) -> List[str]:
        """Validate and normalize media inputs."""
        if isinstance(media, str):
            media = [media]
        if not media:
            raise ValueError("At least one image or video must be provided")
        if not all(isinstance(item, str) and item.strip() for item in media):
            raise ValueError("all items in media must be non-empty strings")
        if len(media) > 10:
            raise ValueError("Maximum 10 media files can be analyzed at once")
        return media

    def _detect_media_kind(self, media_path: str) -> str:
        """Classify an input without ever treating a video as an image."""
        if media_path.startswith("data:"):
            mime_type = media_path[5:].split(";", 1)[0].lower()
        else:
            type_source = media_path
            if media_path.startswith(("http://", "https://")):
                type_source = urlsplit(media_path).path
            mime_type = (mimetypes.guess_type(type_source)[0] or "").lower()

        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("video/"):
            return "video"

        suffix = Path(urlsplit(media_path).path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            return "image"
        if suffix in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}:
            return "video"
        raise ValueError(
            f"Unsupported media type for {media_path!r}; provide a supported image "
            "or video file"
        )

    def _probe_video_duration(self, video_path: str) -> float:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise RuntimeError(
                "Video understanding requires ffprobe/ffmpeg to sample frames"
            )
        try:
            completed = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            duration = float(completed.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Failed to inspect video {video_path}: {exc}") from exc
        if duration <= 0:
            raise RuntimeError(f"Video has no readable duration: {video_path}")
        return duration

    def _extract_video_frames(
        self,
        video_path: str,
        *,
        start_time: Optional[float],
        end_time: Optional[float],
        max_frames: int,
    ) -> List[tuple[float, str]]:
        """Sample video frames as JPEG data URLs for provider-neutral vision input."""
        if video_path.startswith(("http://", "https://", "data:")):
            raise ValueError(
                "Video URLs and data URLs are not sampled directly; upload the video "
                "or pass a workspace file_id"
            )
        resolved_path = str(Path(video_path).resolve())
        if not Path(resolved_path).is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "Video understanding requires ffprobe/ffmpeg to sample frames"
            )

        duration = self._probe_video_duration(resolved_path)
        start = max(0.0, start_time or 0.0)
        end = min(duration, end_time if end_time is not None else duration)
        if start >= duration:
            raise ValueError(
                f"start_time {start:g}s is outside the {duration:.2f}s video"
            )
        if end <= start:
            raise ValueError("end_time must be greater than start_time")

        # Sample the midpoint of equal time buckets. Container duration can be
        # driven by a longer audio stream, so requesting a frame near exact EOF
        # may legitimately return no video bytes.
        bucket_duration = (end - start) / max_frames
        timestamps = [
            start + bucket_duration * (index + 0.5) for index in range(max_frames)
        ]

        frames: List[tuple[float, str]] = []
        for timestamp in timestamps:
            try:
                completed = subprocess.run(
                    [
                        ffmpeg,
                        "-v",
                        "error",
                        "-ss",
                        f"{timestamp:.3f}",
                        "-i",
                        resolved_path,
                        "-frames:v",
                        "1",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "mjpeg",
                        "pipe:1",
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.warning(
                    "Failed to sample video frame at %.2fs from %s: %s",
                    timestamp,
                    video_path,
                    exc,
                )
                continue
            if not completed.stdout:
                logger.warning(
                    "Video decoder returned an empty frame at %.2fs from %s",
                    timestamp,
                    video_path,
                )
                continue
            encoded = base64.b64encode(completed.stdout).decode("ascii")
            frames.append((timestamp, f"data:image/jpeg;base64,{encoded}"))
        if not frames:
            raise RuntimeError(f"No video frames could be decoded from {video_path}")
        return frames

    def _video_frame_budgets(
        self,
        *,
        image_count: int,
        native_video_count: int,
        video_count: int,
        max_frames: int,
    ) -> List[int]:
        """Share the model's ten-visual-input budget across videos."""
        available = 10 - image_count - native_video_count
        if available < video_count:
            raise ValueError(
                "Too many images and videos to include at least one frame per video"
            )
        budgets = [1] * video_count
        remaining = available - video_count
        while remaining:
            changed = False
            for index, budget in enumerate(budgets):
                if budget >= max_frames:
                    continue
                budgets[index] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
            if not changed:
                break
        return budgets

    def _get_attr_safely(self, obj: Any, attr_name: str) -> Optional[str]:
        """
        Safely get an attribute value from an object, excluding Mock objects.

        Args:
            obj: The object to get the attribute from
            attr_name: Name of the attribute to retrieve

        Returns:
            String value of the attribute, or None if not found or if it's a Mock
        """
        if hasattr(obj, attr_name):
            value = getattr(obj, attr_name)
            if str(value).startswith("<Mock name="):
                return None
            return str(value) if value is not None else None
        return None

    def _model_used(self) -> str:
        """Return the configured model identity, not an internal wrapper class."""
        return (
            self._get_attr_safely(self.vision_model, "model_name")
            or self._get_attr_safely(self.vision_model, "model_id")
            or self.vision_model.__class__.__name__
        )

    async def understand_media(
        self,
        media: Union[str, List[str]],
        question: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_frames: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> UnderstandMediaResult:
        """
        Analyze images, videos, or mixed media and answer a question.

        Args:
            media: Image/video path, provider-supported URL, file id, or a list of them
            question: Question to ask about the images
            start_time: Optional video sampling start in seconds
            end_time: Optional video sampling end in seconds
            max_frames: Maximum sampled frames per video (1-10, default 8)
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with analysis result and metadata
        """
        try:
            temperature = self._coerce_optional_float(temperature, "temperature")
            max_tokens = self._coerce_optional_int(max_tokens, "max_tokens")
            start_time = self._coerce_optional_float(start_time, "start_time")
            end_time = self._coerce_optional_float(end_time, "end_time")
            max_frames = self._coerce_optional_int(max_frames, "max_frames") or 8
            if max_frames < 1 or max_frames > 10:
                raise ValueError("max_frames must be between 1 and 10")
            if start_time is not None and (
                not math.isfinite(start_time) or start_time < 0
            ):
                raise ValueError(
                    "start_time must be a finite number greater than or equal to 0"
                )
            if end_time is not None and (not math.isfinite(end_time) or end_time < 0):
                raise ValueError(
                    "end_time must be a finite number greater than or equal to 0"
                )
            if (
                start_time is not None
                and end_time is not None
                and end_time <= start_time
            ):
                raise ValueError("end_time must be greater than start_time")

            # Validate vision model capability
            if not self.vision_model.has_ability("vision"):
                model_info = f"Model: {self.vision_model.__class__.__name__}"

                model_id = self._get_attr_safely(self.vision_model, "model_id")
                if model_id:
                    model_info += f", ID: {model_id}"

                model_name = self._get_attr_safely(self.vision_model, "model_name")
                if model_name:
                    model_info += f", Name: {model_name}"

                provider = self._get_attr_safely(self.vision_model, "provider")
                if provider:
                    model_info += f", Provider: {provider}"

                return UnderstandMediaResult(
                    success=False,
                    error=f"{model_info} does not support vision capabilities",
                )

            validated_media = self._validate_media(media)
            classified = [
                (media_path, self._detect_media_kind(media_path))
                for media_path in validated_media
            ]
            image_count = sum(kind == "image" for _, kind in classified)
            video_count = sum(kind == "video" for _, kind in classified)
            use_native_video = self.vision_model.supports_native_video_input and (
                image_count == 0 or self.vision_model.supports_native_video_with_images
            )
            if (start_time is not None or end_time is not None) and not (
                self.vision_model.supports_native_video_time_range
            ):
                use_native_video = False

            visual_contents: List[Dict[str, Any]] = []
            warnings: List[str] = []
            native_video_contents: Dict[int, Dict[str, Any]] = {}
            fallback_video_count = video_count
            if use_native_video:
                fallback_video_count = 0
                for index, (media_path, kind) in enumerate(classified):
                    if kind != "video":
                        continue
                    try:
                        video_data = await asyncio.to_thread(
                            self._convert_video_to_base64, media_path
                        )
                        native_video_contents[index] = (
                            self.vision_model.build_native_video_content(
                                video_data,
                                start_time=start_time,
                                end_time=end_time,
                            )
                        )
                    except Exception as exc:
                        fallback_video_count += 1
                        warning = (
                            f"Native video input unavailable for {media_path}: "
                            f"{exc}; using sampled frames"
                        )
                        logger.warning(warning)
                        warnings.append(warning)

            processed_images = 0
            processed_videos = 0
            native_videos = 0
            extracted_frames = 0
            video_budgets = iter(
                self._video_frame_budgets(
                    image_count=image_count,
                    native_video_count=len(native_video_contents),
                    video_count=fallback_video_count,
                    max_frames=max_frames,
                )
                if fallback_video_count
                else []
            )

            for index, (media_path, kind) in enumerate(classified):
                try:
                    if kind == "video":
                        native_content = native_video_contents.get(index)
                        if native_content is not None:
                            visual_contents.append(native_content)
                            processed_videos += 1
                            native_videos += 1
                            continue

                        frame_budget = next(video_budgets)
                        frames = await asyncio.to_thread(
                            self._extract_video_frames,
                            media_path,
                            start_time=start_time,
                            end_time=end_time,
                            max_frames=frame_budget,
                        )
                        for timestamp, frame_data in frames:
                            visual_contents.extend(
                                [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Video {Path(media_path).name}, frame at "
                                            f"{timestamp:.2f} seconds:"
                                        ),
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": frame_data},
                                    },
                                ]
                            )
                        processed_videos += 1
                        extracted_frames += len(frames)
                    else:
                        image_data = (
                            media_path
                            if media_path.startswith(("http://", "https://", "data:"))
                            else self._convert_image_to_base64(media_path)
                        )
                        visual_contents.append(
                            {"type": "image_url", "image_url": {"url": image_data}}
                        )
                        processed_images += 1
                except Exception as e:
                    warning = f"Failed to process {kind} {media_path}: {e}"
                    logger.warning(warning)
                    warnings.append(warning)
                    continue

            if not visual_contents:
                return UnderstandMediaResult(
                    success=False,
                    warnings=warnings,
                    error="No valid images or video frames could be processed",
                )

            content = [{"type": "text", "text": question}]
            content.extend(visual_contents)

            # Create the message for vision chat
            messages = [{"role": "user", "content": content}]

            # Call the vision model
            result = await self.vision_model.vision_chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Process the result
            if isinstance(result, str):
                answer = result
            elif isinstance(result, dict) and result.get("type") == "tool_call":
                answer = f"Model triggered tool call instead of answering: {result.get('tool_calls', [])}"
            else:
                answer = str(result)

            return UnderstandMediaResult(
                success=True,
                answer=answer,
                media_processed=processed_images + processed_videos,
                images_processed=processed_images,
                videos_processed=processed_videos,
                native_videos_processed=native_videos,
                frames_extracted=extracted_frames,
                model_used=self._model_used(),
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Media understanding failed: {e}")
            return UnderstandMediaResult(success=False, error=str(e))

    async def understand_images(
        self,
        images: Union[str, List[str]],
        question: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> UnderstandImagesResult:
        """Backwards-compatible internal image-only entrypoint."""
        return await self.understand_media(
            media=images,
            question=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def describe_images(
        self,
        images: Union[str, List[str]],
        detail_level: str = "normal",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> UnderstandImagesResult:
        """
        Generate descriptions for images.

        Args:
            images: Single image path/URL or list of image paths/URLs
            detail_level: Level of detail ("simple", "normal", "detailed")
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with image descriptions and metadata
        """
        detail_prompts = {
            "simple": "Please provide a brief description of what you see in these images.",
            "normal": "Please describe what you see in these images, including main subjects, actions, and context.",
            "detailed": "Please provide a detailed description of these images, including objects, people, actions, setting, colors, composition, and any notable details.",
        }

        question = detail_prompts.get(detail_level, detail_prompts["normal"])

        result = await self.understand_images(
            images=images,
            question=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result

    async def detect_objects(
        self,
        images: Union[str, List[str]],
        task: str,
        mark_objects: bool = False,
        box_color: str = "red",
        confidence_threshold: float = 0.5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> DetectObjectsResult:
        """
        Detect objects in images with optional marking capability.

        Args:
            images: Single image path/URL or list of image paths/URLs
            task: Natural language description of what to detect
            mark_objects: Whether to create a marked image with bounding boxes
            box_color: Color for bounding boxes if marking
            confidence_threshold: Minimum confidence score for detected objects
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Result with detected objects and optionally marked image path
        """
        try:
            temperature = self._coerce_optional_float(temperature, "temperature")
            max_tokens = self._coerce_optional_int(max_tokens, "max_tokens")

            # Validate vision model capability
            if not self.vision_model.has_ability("vision"):
                model_info = f"Model: {self.vision_model.__class__.__name__}"

                model_id = self._get_attr_safely(self.vision_model, "model_id")
                if model_id:
                    model_info += f", ID: {model_id}"

                model_name = self._get_attr_safely(self.vision_model, "model_name")
                if model_name:
                    model_info += f", Name: {model_name}"

                provider = self._get_attr_safely(self.vision_model, "provider")
                if provider:
                    model_info += f", Provider: {provider}"

                return DetectObjectsResult(
                    success=False,
                    error=f"{model_info} does not support vision capabilities",
                )

            # Validate and normalize images
            validated_images = self._validate_images(images)
            if len(validated_images) > 1:
                logger.warning(
                    "Object detection works best with single images. Using first image only."
                )
                validated_images = validated_images[:1]

            # Convert image to appropriate format
            image_path = validated_images[0]
            try:
                if image_path.startswith(("http://", "https://")):
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": image_path},
                    }
                elif image_path.startswith("data:"):
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": image_path},
                    }
                else:
                    base64_data = self._convert_image_to_base64(image_path)
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": base64_data},
                    }
            except Exception as e:
                return DetectObjectsResult(
                    success=False, error=f"Failed to process image {image_path}: {e}"
                )

            # Prepare the detection prompt
            prompt = f"""
            Task: {task}

            Please analyze this image and detect objects according to the task above.

            For each detected object, provide:
            1. Object class/name
            2. Bounding box coordinates in normalized format [xmin, ymin, xmax, ymax] where:
               - xmin, ymin: top-left corner (0.0 to 1.0)
               - xmax, ymax: bottom-right corner (0.0 to 1.0)
            3. Confidence score (0.0 to 1.0)

            Only include detections with confidence >= {confidence_threshold}.

            Format your response as a JSON object with this structure:
            {{
                "detections": [
                    {{
                        "class": "object_name",
                        "bbox": [xmin, ymin, xmax, ymax],
                        "confidence": confidence_score
                    }}
                ],
                "image_info": {{
                    "width": "estimated_width",
                    "height": "estimated_height"
                }}
            }}
            """

            # Create the message for vision chat
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, image_content],
                }
            ]

            # Call the vision model
            raw_result = await self.vision_model.vision_chat(
                messages=messages,
                temperature=temperature if temperature is not None else 0.1,
                max_tokens=max_tokens if max_tokens is not None else 2000,
                response_format={"type": "json_object"},
            )

            # Parse the result
            detections = []
            parsing_method = "unknown"
            parsing_error = None

            if isinstance(raw_result, str):
                raw_response = raw_result

                try:
                    detections = self._extract_detections_from_text(raw_response)
                    if detections:
                        parsing_method = "regex"
                    else:
                        try:
                            import json

                            parsed_result = json.loads(raw_result)
                            detections = parsed_result.get("detections", [])
                            parsing_method = "json"

                            validated_detections = []
                            for detection in detections:
                                if isinstance(detection, dict):
                                    obj_class = detection.get("class", "unknown")
                                    bbox = detection.get("bbox", [0, 0, 1, 1])
                                    confidence = float(detection.get("confidence", 0.5))

                                    if (
                                        isinstance(bbox, list)
                                        and len(bbox) == 4
                                        and all(
                                            isinstance(coord, (int, float))
                                            for coord in bbox
                                        )
                                        and 0 <= bbox[0] <= 1
                                        and 0 <= bbox[1] <= 1
                                        and 0 <= bbox[2] <= 1
                                        and 0 <= bbox[3] <= 1
                                        and bbox[0] < bbox[2]
                                        and bbox[1] < bbox[3]
                                    ):
                                        validated_detections.append(
                                            {
                                                "class": obj_class,
                                                "bbox": bbox,
                                                "confidence": min(
                                                    max(confidence, 0.0), 1.0
                                                ),
                                            }
                                        )

                            detections = validated_detections

                        except json.JSONDecodeError as e:
                            parsing_error = f"JSON parsing failed: {str(e)}"
                            if not detections:
                                detections = (
                                    self._extract_detections_from_text_fallback(
                                        raw_response
                                    )
                                )
                                parsing_method = "regex_fallback"

                except Exception as e:
                    parsing_error = f"General parsing error: {str(e)}"
                    detections = self._extract_detections_from_text_fallback(
                        raw_response
                    )
                    parsing_method = "simple_text"

            elif isinstance(raw_result, dict):
                raw_response = str(raw_result)
                parsing_method = "dict_response"
                detections = []
            else:
                raw_response = str(raw_result)
                parsing_method = "unknown_type"
                detections = []

            # Base result
            result_data = {
                "success": True,
                "detections": detections,
                "total_detections": len(detections),
                "image_processed": image_path,
                "confidence_threshold": confidence_threshold,
                "prompt_sent": prompt,
                "box_color": box_color if mark_objects else None,
                "raw_response": raw_response,
                "parsing_method": parsing_method,
            }

            if parsing_error:
                result_data["error"] = parsing_error

            # If marking is requested, create marked image
            marked_image_path = None
            if mark_objects:
                if image_path.startswith(("http://", "https://", "data:")):
                    return DetectObjectsResult(
                        success=False,
                        error="Image marking is only supported for local files, not URLs or base64 data",
                        confidence_threshold=confidence_threshold,
                        prompt_sent=prompt,
                    )

                # Convert to absolute path if relative
                resolved_image_path = image_path
                if not os.path.isabs(resolved_image_path):
                    resolved_image_path = os.path.abspath(resolved_image_path)

                if not os.path.exists(resolved_image_path):
                    return DetectObjectsResult(
                        success=False,
                        error=f"Image file not found: {resolved_image_path}",
                        confidence_threshold=confidence_threshold,
                        prompt_sent=prompt,
                    )

                try:
                    marked_image_path = self._draw_bounding_boxes(
                        image_path=resolved_image_path,
                        detections=detections,
                        box_color=box_color,
                    )
                    result_data["marked_image_path"] = marked_image_path
                except Exception as e:
                    logger.error(f"Failed to draw bounding boxes: {e}")
                    return DetectObjectsResult(
                        success=False,
                        error=f"Image marking failed: {e}",
                        confidence_threshold=confidence_threshold,
                        prompt_sent=prompt,
                    )

            return DetectObjectsResult(**result_data)

        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return DetectObjectsResult(success=False, error=str(e))

    def _extract_detections_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract detection information from unstructured text response."""
        detections = []

        patterns = [
            r"(\w+(?:\s+\w+)*)\s*:\s*\[([0-9.]+(?:,\s*[0-9.]+){3})\]\s*\(?confidence:\s*([0-9.]+)\)?",
            r"(\w+(?:\s+\w+)*)\s*at\s*\[([0-9.]+(?:,\s*[0-9.]+){3})\]\s*\(confidence:\s*([0-9.]+)\)",
            r"detected\s+(\w+(?:\s+\w+)*)\s*[,:].*?bbox.*?([0-9.]+(?:,\s*[0-9.]+){3}).*?confidence.*?([0-9.]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                obj_class = match[0].strip()
                bbox_str = match[1].strip()
                confidence_str = match[2].strip()

                try:
                    bbox = [float(x.strip()) for x in bbox_str.split(",")]
                    confidence = float(confidence_str)

                    if (
                        len(bbox) == 4
                        and all(0 <= coord <= 1 for coord in bbox)
                        and bbox[0] < bbox[2]
                        and bbox[1] < bbox[3]
                    ):
                        detections.append(
                            {
                                "class": obj_class,
                                "bbox": bbox,
                                "confidence": min(max(confidence, 0.0), 1.0),
                            }
                        )
                except (ValueError, IndexError):
                    continue

        return detections

    def _extract_detections_from_text_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Aggressive fallback method to extract detection information."""
        detections = []

        patterns = [
            r'"class"\s*:\s*"([^"]+)"[^}]*"bbox"\s*:\s*\[([^\]]+)\][^}]*"confidence"\s*:\s*([0-9.]+)',
            r'class\s*:\s*"([^"]+)"[^}]*bbox\s*:\s*\[([^\]]+)\][^}]*confidence\s*:\s*([0-9.]+)',
            r"([A-Za-z\s]+?)\s*(?:at|located|found)?\s*[\[\(]([0-9.,\s]+)[\]\)][^0-9]*([0-9.]+)",
            r"detected\s+([A-Za-z\s]+?)[\s,:]coordinates?\s*[\[\(]([0-9.,\s]+)[\]\)][^0-9]*([0-9.]+)",
            r"([A-Za-z\s]+?)\s*(?:at|position|location)?\s*[:\-]?\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)",
            r"-\s*([A-Za-z\s]+?):\s*[^\d]*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)[^0-9]*([0-9.]+)?",
            r"([A-Za-z\s]+?)\s*(?:with\s*)?confidence\s*[:\-]?\s*([0-9.]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 3:
                        obj_class = match[0].strip()
                        bbox_str = match[1].strip()
                        confidence = float(match[2])

                        bbox = [
                            float(x.strip())
                            for x in re.split(r"[,\s]+", bbox_str)
                            if x.strip()
                        ]
                        if len(bbox) == 4:
                            bbox = bbox[:4]
                        else:
                            continue

                    elif len(match) == 5:
                        obj_class = match[0].strip()
                        bbox = [
                            float(match[1]),
                            float(match[2]),
                            float(match[3]),
                            float(match[4]),
                        ]
                        confidence = (
                            float(match[5]) if len(match) > 5 and match[5] else 0.8
                        )

                    elif len(match) == 2:
                        obj_class = match[0].strip()
                        confidence = float(match[1])
                        bbox = [0.25, 0.25, 0.75, 0.75]
                    else:
                        continue

                    if any(coord > 1.0 for coord in bbox):
                        bbox = [min(coord / 1000.0, 1.0) for coord in bbox]

                    if (
                        len(bbox) == 4
                        and all(0 <= coord <= 1 for coord in bbox)
                        and bbox[0] < bbox[2]
                        and bbox[1] < bbox[3]
                    ):
                        detections.append(
                            {
                                "class": obj_class,
                                "bbox": bbox,
                                "confidence": min(max(confidence, 0.0), 1.0),
                            }
                        )

                except (ValueError, IndexError, AttributeError):
                    continue

        if not detections:
            object_patterns = [
                r"(?:found|detected|located|identified)\s+([A-Za-z\s]+?)(?:\s*(?:in|at|on)\s+|$)",
                r"(?:there\s+is|are)\s+(?:a|an|some|\d+)\s+([A-Za-z\s]+?)(?:\s*(?:in|at|on)\s+|$)",
                r"([A-Za-z\s]+?)\s*(?:is|are)\s*(?:present|visible|detected)",
            ]

            for pattern in object_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    obj_class = match[0].strip()
                    if obj_class and len(obj_class) > 2:
                        detections.append(
                            {
                                "class": obj_class,
                                "bbox": [0.2, 0.2, 0.8, 0.8],
                                "confidence": 0.7,
                            }
                        )

        return detections

    def _draw_bounding_boxes(
        self, image_path: str, detections: List[Dict[str, Any]], box_color: str = "red"
    ) -> str:
        """Draw bounding boxes on image and return the path to the marked image."""
        if not PIL_AVAILABLE:
            raise RuntimeError(
                "PIL (Pillow) library is required for image marking. Install with: pip install Pillow"
            )

        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                draw = ImageDraw.Draw(img)
                img_width, img_height = img.size

                try:
                    color = box_color.lower()
                    color_map = {
                        "red": (255, 0, 0),
                        "blue": (0, 0, 255),
                        "green": (0, 255, 0),
                        "yellow": (255, 255, 0),
                        "purple": (128, 0, 128),
                        "orange": (255, 165, 0),
                    }
                    rgb_color = color_map.get(color, (255, 0, 0))
                except Exception:
                    rgb_color = (255, 0, 0)

                for detection in detections:
                    if "bbox" in detection and len(detection["bbox"]) == 4:
                        bbox = detection["bbox"]
                        x1 = int(bbox[0] * img_width)
                        y1 = int(bbox[1] * img_height)
                        x2 = int(bbox[2] * img_width)
                        y2 = int(bbox[3] * img_height)

                        draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=3)

                        label = detection.get("class", "Unknown")
                        confidence = detection.get("confidence", 0)
                        label_text = f"{label} ({confidence:.2f})"

                        try:
                            font = ImageFont.load_default()
                        except Exception:
                            font = None

                        text_bbox = draw.textbbox((0, 0), label_text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]

                        label_x = x1
                        label_y = max(0, y1 - text_height - 5)

                        draw.rectangle(
                            [
                                label_x,
                                label_y,
                                label_x + text_width + 4,
                                label_y + text_height + 4,
                            ],
                            fill=rgb_color,
                        )

                        draw.text(
                            (label_x + 2, label_y + 2),
                            label_text,
                            fill="white",
                            font=font,
                        )

                output_filename = (
                    f"marked_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
                )
                output_path = str(self.output_directory / output_filename)

                img.save(output_path, "JPEG", quality=95)

                return output_path

        except Exception as e:
            logger.error(f"Failed to draw bounding boxes: {e}")
            raise RuntimeError(f"Image marking failed: {e}")


# Convenience functions for direct usage
async def understand_media(
    vision_model: BaseLLM,
    media: Union[str, List[str]],
    question: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    max_frames: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    output_directory: Optional[str] = None,
) -> UnderstandMediaResult:
    """Analyze images, videos, or mixed media with a vision model."""
    core = VisionCore(vision_model, output_directory)
    return await core.understand_media(
        media,
        question,
        start_time,
        end_time,
        max_frames,
        temperature,
        max_tokens,
    )


async def understand_images(
    vision_model: BaseLLM,
    images: Union[str, List[str]],
    question: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    output_directory: Optional[str] = None,
) -> UnderstandImagesResult:
    """
    Analyze images and answer questions about their content.

    Args:
        vision_model: The language model with vision capabilities to use for understanding images.
        images: A single image path or list of image paths to understand.
        question: The question to ask about the images.
        temperature: Controls randomness in the model's output. Higher values make output more random.
        max_tokens: Maximum number of tokens to generate in the response.
        output_directory: Directory path where output files should be saved.

    Returns:
        UnderstandImagesResult containing the model's understanding of the images.
    """
    core = VisionCore(vision_model, output_directory)
    return await core.understand_images(images, question, temperature, max_tokens)


async def describe_images(
    vision_model: BaseLLM,
    images: Union[str, List[str]],
    detail_level: str = "normal",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    output_directory: Optional[str] = None,
) -> UnderstandImagesResult:
    """
    Generate descriptions for images.

    Args:
        vision_model: The language model with vision capabilities to use for describing images.
        images: A single image path or list of image paths to describe.
        detail_level: Level of detail for the description. Options include "normal", "high", etc.
        temperature: Controls randomness in the model's output. Higher values make output more random.
        max_tokens: Maximum number of tokens to generate in the response.
        output_directory: Directory path where output files should be saved.

    Returns:
        UnderstandImagesResult containing the descriptions of the images.
    """
    core = VisionCore(vision_model, output_directory)
    return await core.describe_images(images, detail_level, temperature, max_tokens)


async def detect_objects(
    vision_model: BaseLLM,
    images: Union[str, List[str]],
    task: str,
    mark_objects: bool = False,
    box_color: str = "red",
    confidence_threshold: float = 0.5,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    output_directory: Optional[str] = None,
) -> DetectObjectsResult:
    """
    Detect objects in images with optional marking capability.

    Args:
        vision_model: The language model with vision capabilities to use for object detection.
        images: A single image path or list of image paths to analyze for objects.
        task: Description of the object detection task to perform.
        mark_objects: Whether to draw bounding boxes around detected objects.
        box_color: Color of the bounding boxes when mark_objects is True.
        confidence_threshold: Minimum confidence score for detected objects to be included.
        temperature: Controls randomness in the model's output. Higher values make output more random.
        max_tokens: Maximum number of tokens to generate in the response.
        output_directory: Directory path where output files should be saved.

    Returns:
        DetectObjectsResult containing information about detected objects.
    """
    core = VisionCore(vision_model, output_directory)
    return await core.detect_objects(
        images,
        task,
        mark_objects,
        box_color,
        confidence_threshold,
        temperature,
        max_tokens,
    )
