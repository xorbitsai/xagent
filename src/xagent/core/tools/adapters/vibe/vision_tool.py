"""
Vision tool for xagent
Framework wrapper around the pure vision core
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from xagent.config import get_max_upload_size_bytes
from xagent.core.workspace import ExternalUploadSizeLimitError, TaskWorkspace

from ....file_ref import build_workspace_file_ref
from ....model.chat.basic.adapter import attach_chat_retry_wrapper
from ....model.chat.basic.base import BaseLLM
from ...core.vision_tool import (
    DetectObjectsResult,
    UnderstandImagesResult,
    UnderstandMediaResult,
    VisionCore,
)
from .base import ToolCategory
from .function import FunctionTool

logger = logging.getLogger(__name__)


class VisionFunctionTool(FunctionTool):
    """VisionFunctionTool with ToolCategory.VISION category."""

    category = ToolCategory.VISION


class VisionTool:
    """
    Vision tool that uses vision-enabled LLM models to analyze images and detect objects.
    Framework wrapper that handles workspace integration.
    """

    def __init__(
        self,
        vision_model: BaseLLM,
        workspace: Optional[TaskWorkspace] = None,
    ):
        """
        Initialize with a vision-enabled LLM model.

        Args:
            vision_model: LLM model with vision capabilities
            workspace: Optional workspace for resolving local image paths
        """
        self.vision_model = vision_model
        self.workspace = workspace

        # Determine output directory
        output_dir = str(workspace.output_dir) if workspace else "./output"

        # Create core instance
        private_staging_roots = (
            (workspace.external_upload_staging_root(host_private=True),)
            if isinstance(workspace, TaskWorkspace)
            else ()
        )
        self.core = VisionCore(
            vision_model,
            output_directory=output_dir,
            workspace_root=workspace.workspace_dir if workspace else None,
            allowed_local_media_roots=private_staging_roots,
        )

    def _stage_media_path_sync(self, media_path: str) -> Path:
        if self.workspace is None:
            raise ValueError("Local media files require a task workspace")
        source = self.workspace.resolve_path_with_search(
            media_path, detached_file_ids=True
        )
        source = Path(source)
        if not source.is_absolute():
            raise FileNotFoundError(f"File not found: {media_path}")
        return self.workspace.stage_resolved_path_for_external_upload(
            source,
            media_path,
            max_bytes=get_max_upload_size_bytes(),
            host_private=True,
        )

    async def _resolve_media_path(
        self, media_path: str, staged_sources: dict[str, str]
    ) -> str:
        """Resolve an image or video path using the task workspace."""
        # If it's a URL, return as-is
        if media_path.startswith(("http://", "https://", "data:")):
            return media_path

        # The core owns the final read boundary as defense in depth. Keeping
        # an unresolved value here lets it still validate arguments before a
        # local read and preserves callers that mock the media conversion.
        if self.workspace is None:
            return media_path

        task = asyncio.create_task(
            asyncio.to_thread(self._stage_media_path_sync, media_path)
        )
        try:
            safe_path = await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if not task.cancelled() and task.exception() is None:
                completed_path = task.result()
                staged_sources[str(completed_path)] = media_path
            raise
        except ExternalUploadSizeLimitError as exc:
            logger.warning("Vision input exceeded its staging limit: %s", exc)
            raise ValueError("Media exceeds the configured size limit") from None
        except Exception as exc:
            logger.warning("Rejected vision media selector %r: %s", media_path, exc)
            raise ValueError(
                "Media is unavailable or outside the task workspace"
            ) from None
        staged_sources[str(safe_path)] = media_path
        if isinstance(self.workspace, TaskWorkspace):
            self.core.add_allowed_local_media_root(
                self.workspace.external_upload_staging_root(host_private=True)
            )
        if not safe_path.is_file():
            raise FileNotFoundError(f"Media file not found: {media_path}")
        return str(safe_path)

    async def _resolve_media(
        self,
        media: Union[str, List[str]],
        staged_sources: dict[str, str],
        resolution_warnings: Optional[list[str]] = None,
    ) -> str | list[str]:
        """Resolve all media paths using workspace."""
        if isinstance(media, str):
            return await self._resolve_media_path(media, staged_sources)
        if isinstance(media, list):
            resolved: list[str] = []
            failures: list[Exception] = []
            for item in media:
                try:
                    resolved.append(
                        await self._resolve_media_path(item, staged_sources)
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    failures.append(exc)
                    if resolution_warnings is not None:
                        resolution_warnings.append(
                            f"Failed to resolve media {item}: {exc}"
                        )
            if not resolved and failures:
                raise failures[0]
            return resolved
        return media

    def _discard_staged_media(self, staged_sources: dict[str, str]) -> None:
        if self.workspace is None:
            return
        for path in staged_sources:
            try:
                self.workspace.discard_staged_external_upload(path)
            except Exception as exc:
                logger.warning("Failed to discard staged vision input: %s", exc)

    @staticmethod
    def _restore_media_selectors(
        result: UnderstandMediaResult | DetectObjectsResult,
        staged_sources: dict[str, str],
    ) -> None:
        if result.error:
            for staged, original in staged_sources.items():
                result.error = result.error.replace(staged, original)
        if isinstance(result, DetectObjectsResult) and result.image_processed:
            result.image_processed = staged_sources.get(
                result.image_processed, result.image_processed
            )
        if result.warnings:
            restored: list[str] = []
            for warning in result.warnings:
                for staged, original in staged_sources.items():
                    warning = warning.replace(staged, original)
                restored.append(warning)
            result.warnings = restored

    def _normalize_media(
        self,
        media: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        """Normalize and validate image/video input."""
        if media is None:
            raise ValueError("At least one image or video must be provided")

        if isinstance(media, str):
            return media
        if isinstance(media, list):
            if not media:
                raise ValueError("At least one image or video must be provided")
            if all(isinstance(item, str) for item in media):
                return media
            raise ValueError("all items in media must be strings")
        raise TypeError("media must be a string or a list of strings")

    _normalize_images = _normalize_media

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
        """Analyze images, videos, or mixed media with one public tool."""
        staged_sources: dict[str, str] = {}
        resolution_warnings: list[str] = []
        try:
            normalized_media = self._normalize_media(media)
            resolved_media = await self._resolve_media(
                normalized_media, staged_sources, resolution_warnings
            )
            result = await self.core.understand_media(
                resolved_media,
                question,
                start_time,
                end_time,
                max_frames,
                temperature,
                max_tokens,
            )
            result.warnings = [*resolution_warnings, *result.warnings]
            self._restore_media_selectors(result, staged_sources)
            return result
        except Exception as e:
            logger.error("understand_media: error resolving media: %s", e)
            return UnderstandMediaResult(success=False, error=str(e))
        finally:
            self._discard_staged_media(staged_sources)

    async def understand_images(
        self,
        images: Union[str, List[str]],
        question: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> UnderstandImagesResult:
        """
        Analyze images and answer questions about their content.

        Args:
            images: Single image path/URL or list of image paths/URLs
            question: Question to ask about the images
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            UnderstandImagesResult with analysis result and metadata
        """
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
            UnderstandImagesResult with descriptions and metadata
        """
        staged_sources: dict[str, str] = {}
        resolution_warnings: list[str] = []
        try:
            normalized_images = self._normalize_images(images)
            resolved_images = await self._resolve_media(
                normalized_images, staged_sources, resolution_warnings
            )
            result = await self.core.describe_images(
                resolved_images, detail_level, temperature, max_tokens
            )
            result.warnings = [*resolution_warnings, *result.warnings]
            self._restore_media_selectors(result, staged_sources)
            return result
        except Exception as e:
            logger.error(f"describe_images: Error in resolving images: {e}")
            return UnderstandImagesResult(success=False, error=str(e))
        finally:
            self._discard_staged_media(staged_sources)

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
            DetectObjectsResult with detected objects and optionally marked image path
        """
        staged_sources: dict[str, str] = {}
        resolution_warnings: list[str] = []
        try:
            normalized_images = self._normalize_images(images)
            resolved_images = await self._resolve_media(
                normalized_images, staged_sources, resolution_warnings
            )
        except asyncio.CancelledError:
            self._discard_staged_media(staged_sources)
            raise
        except Exception as e:
            logger.error(f"detect_objects: Error in resolving images: {e}")
            self._discard_staged_media(staged_sources)
            return DetectObjectsResult(success=False, error=str(e))
        result: DetectObjectsResult | None = None
        try:
            # Execute within auto_register context when marking
            if mark_objects and self.workspace:
                with self.workspace.auto_register_files():
                    result = await self.core.detect_objects(
                        resolved_images,
                        task,
                        mark_objects,
                        box_color,
                        confidence_threshold,
                        temperature,
                        max_tokens,
                    )
            else:
                result = await self.core.detect_objects(
                    resolved_images,
                    task,
                    mark_objects,
                    box_color,
                    confidence_threshold,
                    temperature,
                    max_tokens,
                )
        finally:
            if result is not None:
                result.warnings = [*resolution_warnings, *result.warnings]
                self._restore_media_selectors(result, staged_sources)
            self._discard_staged_media(staged_sources)

        assert result is not None

        if (
            mark_objects
            and self.workspace
            and result.success
            and result.marked_image_path
        ):
            try:
                file_ref = build_workspace_file_ref(
                    workspace=self.workspace,
                    file_path=result.marked_image_path,
                )
                result = result.model_copy(
                    update={
                        "file_id": file_ref["file_id"],
                        "file_ref": file_ref,
                    }
                )
            except Exception as e:
                logger.warning("Failed to build marked image FileRef: %s", e)

        return result

    def get_tools(self) -> list:
        """Get all tool instances."""
        tools = [
            VisionFunctionTool(
                self.understand_media,
                name="understand_media",
                description="""
Analyze images, videos, or a mixture of both with one AI vision tool.

Use this tool to:
- Identifying objects, people, and scenes
- Reading text in images (OCR capabilities)
- Inspecting SVG markup, exact colors, and the visual design encoded by its source
- Understanding actions and scene changes across a video
- Comparing multiple images or videos
- Answering specific questions about visual content

Do not call this tool merely to inspect a standard uploaded image that is
already attached to the current conversation as visual context; answer from the
attached image directly. Use this tool as the fallback when that image is only
available by file_id, or for video and SVG source inspection. Structured object
localization and marked-image generation belong to detect_objects instead.

Parameters:
- media (required): One image/video path or file_id, or a list of them. Always use
  the exact file_id or filename from uploaded/generated file metadata.
- Local paths must belong to the current task workspace or its authorized uploads;
  absolute host paths, traversal, and symlink escapes are rejected.
- question (required): Question to ask about the media
- start_time/end_time (optional): Video time range in seconds
- max_frames (optional): Maximum sampled frames per video when the model has no
  native video input, 1-10 (default 8)
- temperature (optional): Sampling temperature (0.0 to 2.0)
- max_tokens (optional): Maximum tokens in response

Examples:
- "What is in this image?"
- "What happens in this video from beginning to end?"
- "Does the cat leave the room?"
- "Compare these two images and this video"

Supported images: JPEG, PNG, WebP, GIF, BMP, and SVG. Local/workspace SVGs are
sent directly to the model as bounded XML source so it can inspect exact
viewBox, paths, fill, stroke, gradient, text, and encoded visual structure
without guessing from a raster image. When only raw SVG/XML source is needed,
read_file is also available.
Supported local/workspace videos:
MP4, MOV, M4V, WebM, AVI, MKV. Models with native video support receive the
video directly so motion, timing, and audio-capable model inputs are preserved.
Other vision models use chronologically sampled, timestamped frames.

Never pass a video to detect_objects or encode it as image_url; use this tool.

This tool accepts images and videos only — plus SVG, which is listed above and
belongs here whenever the question is about the design it encodes. It is not a way
to find out what a file is: judge an unfamiliar file by the extension in its
filename, and read other text or code with read_file. A bare file_id carries no
filename: take the name from the same task's get_file_info or from the file
listing that gave you the id, and if neither resolves it, skip the id rather than
probing it here.
                """.strip(),
            ),
            VisionFunctionTool(
                self.detect_objects,
                name="detect_objects",
                description="""
Detect objects in images with optional bounding box annotation.

This unified tool can both detect objects and optionally create marked images with visual annotations. Simply describe what you want to find in natural language.

Parameters:
- images (required): The image parameter name is "images". Provide a single image file path, URL, or a list of multiple image paths/URLs. For best results, use single images. Supports local file paths or file IDs (e.g., MUST use the exact `file_id` or filename from the uploaded files list in the system prompt).
- Local paths must belong to the current task workspace or its authorized uploads; arbitrary host paths, traversal, and symlink escapes are rejected.
- task (required): Describe what you want to detect in plain language. Examples:
  * "Find all people in the image"
  * "Detect workers not wearing safety helmets"
  * "Count the number of cars"
  * "Locate safety violations"
- mark_objects (optional): Whether to draw bounding boxes on the image and save marked version - default: False
- box_color (optional): Color for bounding boxes. Supported: red, blue, green, yellow, purple, orange - default: "red"
- confidence_threshold (optional): Minimum confidence score (0.0 to 1.0) for detections - default: 0.5
- temperature (optional): Sampling temperature (0.0 to 2.0) - default: 0.1 for consistent output
- max_tokens (optional): Maximum tokens in response - default: 2000

Output format:
```json
{
  "success": true,
  "detections": [
    {
      "class": "person",
      "bbox": [0.1, 0.2, 0.8, 0.9],
      "confidence": 0.95
    }
  ],
  "total_detections": 1,
  "image_processed": "image.jpg",
  "confidence_threshold": 0.5,
  "prompt_sent": "Task: Find and count people and vehicles\n\nPlease analyze this image and detect objects according to the task above.",
  "marked_image_path": "/workspace/output/marked_image.jpg",
  "box_color": "red"
}
```

Bounding box format:
- Normalized coordinates [xmin, ymin, xmax, ymax] where all values are between 0.0 and 1.0
- [xmin, ymin]: Top-left corner of the bounding box
- [xmax, ymax]: Bottom-right corner of the bounding box
- To convert to pixel coordinates: multiply by image width/height

Usage examples:
- Detection only: "Find all people in the image" (mark_objects=False)
- Detection with marking: "Find safety violations and mark them" (mark_objects=True)
- Vehicle analysis: "Count cars and mark them in blue" (mark_objects=True, box_color="blue")
- Face detection: "Detect all faces with high confidence" (confidence_threshold=0.8)

When mark_objects=True:
- Returns path to marked image file in workspace output directory
- Each detected object gets a colored box with label and confidence score
- Only works with local image files (not URLs)
- Perfect for creating visual evidence, training data, or presentations

Perfect for:
- Counting and locating specific objects
- Safety compliance checking
- Quality control inspections
- Creating annotated images for documentation
- Security monitoring
- Debugging computer vision systems
                """.strip(),
            ),
        ]

        return tools


def get_default_vision_model() -> Optional[BaseLLM]:
    """
    Get the default vision model from the system.

    Returns:
        The default vision model or None if not available
    """

    # Only use environment variables for vision tool configuration
    # Try OpenAI vision models first
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from xagent.core.model.chat.basic.openai import OpenAILLM

            model_name = os.getenv("OPENAI_VISION_MODEL_NAME")
            base_url = os.getenv("OPENAI_BASE_URL")

            if model_name:
                return attach_chat_retry_wrapper(
                    OpenAILLM(
                        model_name=model_name,
                        api_key=openai_key,
                        base_url=base_url,
                        abilities=["chat", "tool_calling", "vision"],
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to create OpenAI vision model from env: {e}")

    # Try Zhipu vision models
    zhipu_key = os.getenv("ZHIPU_API_KEY")
    if zhipu_key:
        try:
            from xagent.core.model.chat.basic.zhipu import ZhipuLLM

            model_name = os.getenv("ZHIPU_VISION_MODEL_NAME")
            base_url = os.getenv("ZHIPU_BASE_URL")

            if model_name:
                return attach_chat_retry_wrapper(
                    ZhipuLLM(
                        model_name=model_name,
                        api_key=zhipu_key,
                        base_url=base_url,
                        abilities=["chat", "tool_calling", "vision"],
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to create Zhipu vision model from env: {e}")

    # Try Gemini vision models
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from xagent.core.model.chat.basic.gemini import GeminiLLM

            model_name = os.getenv("GEMINI_VISION_MODEL_NAME", "gemini-2.0-flash-exp")
            base_url = os.getenv("GEMINI_BASE_URL")

            return attach_chat_retry_wrapper(
                GeminiLLM(
                    model_name=model_name,
                    api_key=gemini_key,
                    base_url=base_url,
                    abilities=["chat", "tool_calling", "vision"],
                )
            )
        except Exception as e:
            logger.warning(f"Failed to create Gemini vision model from env: {e}")

    # Try Claude vision models
    claude_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if claude_key:
        try:
            from xagent.core.model.chat.basic.claude import ClaudeLLM

            model_name = os.getenv(
                "CLAUDE_VISION_MODEL_NAME", "claude-3-5-sonnet-20241022"
            )
            base_url = os.getenv("CLAUDE_BASE_URL")

            return attach_chat_retry_wrapper(
                ClaudeLLM(
                    model_name=model_name,
                    api_key=claude_key,
                    base_url=base_url,
                    abilities=["chat", "tool_calling", "vision"],
                )
            )
        except Exception as e:
            logger.warning(f"Failed to create Claude vision model from env: {e}")

    logger.warning("No vision model available from environment variables")
    return None


def get_vision_tool(
    vision_model: Optional[BaseLLM] = None,
    workspace: Optional[TaskWorkspace] = None,
) -> list:
    """
    Create vision tools with a vision-enabled LLM model.

    Args:
        vision_model: LLM model with vision capabilities. If None, uses default vision model.
        workspace: Optional workspace for resolving local image paths

    Returns:
        List of tool instances for vision capabilities
    """
    if vision_model is None:
        vision_model = get_default_vision_model()

    if vision_model is None:
        logger.warning("No vision model available for vision tool")
        return []

    tool_instance = VisionTool(vision_model, workspace)
    return tool_instance.get_tools()


# Register tool creator for auto-discovery
# Import at bottom to avoid circular import with factory
from .factory import ToolFactory, register_tool  # noqa: E402

if TYPE_CHECKING:
    from .config import BaseToolConfig


@register_tool(categories={"vision"})
async def create_vision_tools(config: "BaseToolConfig") -> List[Any]:
    """Create vision understanding tools."""
    vision_model = config.get_vision_model()
    if not vision_model:
        return []

    workspace = await ToolFactory.create_workspace_async(config)

    try:
        return get_vision_tool(vision_model=vision_model, workspace=workspace)
    except Exception as e:
        logger.warning(f"Failed to create vision tools: {e}")
        return []
