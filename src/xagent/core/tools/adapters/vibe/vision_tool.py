"""
Vision tool for xagent
Framework wrapper around the pure vision core
"""

import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional, Union

from xagent.core.workspace import TaskWorkspace

from ....model.chat.basic.base import BaseLLM
from ...core.vision_tool import DetectObjectsResult, UnderstandImagesResult, VisionCore
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
        self.core = VisionCore(vision_model, output_directory=output_dir)

    def _resolve_image_path(self, image_path: str) -> str:
        """
        Resolve image path using workspace if available.

        Args:
            image_path: Original image path

        Returns:
            Resolved image path
        """
        # If it's a URL, return as-is
        if image_path.startswith(("http://", "https://", "data:")):
            return image_path

        # Try to resolve using workspace
        if self.workspace:
            try:
                resolved_path = self.workspace.resolve_path_with_search(image_path)
                return str(resolved_path)
            except (ValueError, FileNotFoundError):
                pass

        # Return original path
        return image_path

    def _resolve_images(
        self,
        images: Union[str, List[str]],
    ) -> str | list[str]:
        """Resolve all image paths using workspace."""
        if isinstance(images, str):
            return self._resolve_image_path(images)
        elif isinstance(images, List):
            return [self._resolve_image_path(img) for img in images]
        return images

    def _normalize_images(
        self,
        images: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        """
        Normalize and validate image input.

        Args:
            images: Single image path/URL or list of image paths/URLs

        Returns:
            Normalized image input (single string or list)

        Raises:
            ValueError: If images is None or invalid
        """
        if images is None:
            raise ValueError("At least one image must be provided")

        if isinstance(images, str):
            return images
        elif isinstance(images, list):
            if not images:
                raise ValueError("At least one image must be provided")
            if all(isinstance(x, str) for x in images):
                return images
            else:
                raise ValueError("all items in images must be strings")
        else:
            raise TypeError("images must be a string or a list of strings")

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
        try:
            normalized_images = self._normalize_images(images)
            resolved_images = self._resolve_images(normalized_images)
        except Exception as e:
            logger.error(f"understand_images: Error in resolving images: {e}")
            return UnderstandImagesResult(success=False, error=str(e))
        return await self.core.understand_images(
            resolved_images, question, temperature, max_tokens
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
        try:
            normalized_images = self._normalize_images(images)
            resolved_images = self._resolve_images(normalized_images)
        except Exception as e:
            logger.error(f"describe_images: Error in resolving images: {e}")
            return UnderstandImagesResult(success=False, error=str(e))
        return await self.core.describe_images(
            resolved_images, detail_level, temperature, max_tokens
        )

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
        try:
            normalized_images = self._normalize_images(images)
            resolved_images = self._resolve_images(normalized_images)
        except Exception as e:
            logger.error(f"detect_objects: Error in resolving images: {e}")
            return DetectObjectsResult(success=False, error=str(e))

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

        return result

    def get_tools(self) -> list:
        """Get all tool instances."""
        tools = [
            VisionFunctionTool(
                self.understand_images,
                name="understand_images",
                description="""
Analyze images and answer questions about their content using AI vision capabilities.

This tool can understand and interpret images, including:
- Identifying objects, people, and scenes
- Reading text in images (OCR capabilities)
- Describing actions and activities
- Analyzing image composition and style
- Comparing multiple images
- Answering specific questions about image content

Parameters:
- images (required): Single image path/URL or list of image paths/URLs. Supports:
  * Local file paths or file IDs (e.g., MUST use the exact `file_id` or filename from the uploaded files list in the system prompt)
  * Remote URLs (e.g., "https://example.com/image.jpg")
  * Multiple images as a list
- question (required): Question to ask about the images
- temperature (optional): Sampling temperature (0.0 to 2.0)
- max_tokens (optional): Maximum tokens in response

Examples:
- "What is in this image?"
- "Read the text shown in this image"
- "Compare these two images and tell me the differences"
- "What action is being performed in this image?"
- "Describe the setting and mood of this scene"

Image requirements:
- Formats: JPEG, PNG, WebP, GIF, BMP
- Size: Up to 10MB per image
- Maximum: 10 images per request
- Local files will be automatically resolved in workspace

The tool uses advanced vision AI models to provide detailed, accurate analysis of image content.
                """.strip(),
            ),
            VisionFunctionTool(
                self.describe_images,
                name="describe_images",
                description="""
Generate natural language descriptions of images with configurable detail level.

This tool provides automated image description capabilities, perfect for:
- Generating alt text for accessibility
- Creating image captions
- Documenting visual content
- Automated image analysis workflows

Parameters:
- images (required): The image parameter name is "images". Provide a single image file path, URL, or a list of multiple image paths/URLs. Supports local file paths or file IDs (e.g., MUST use the exact `file_id` or filename from the uploaded files list in the system prompt)
- detail_level (optional): Level of detail ("simple", "normal", "detailed") - default: "normal"
  * "simple": Brief, concise description
  * "normal": Standard description with main elements
  * "detailed": Comprehensive analysis with fine details
- temperature (optional): Sampling temperature (0.0 to 2.0)
- max_tokens (optional): Maximum tokens in response

Use this tool when you need descriptive text about images without asking specific questions.
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
                return OpenAILLM(
                    model_name=model_name,
                    api_key=openai_key,
                    base_url=base_url,
                    abilities=["chat", "tool_calling", "vision"],
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
                return ZhipuLLM(
                    model_name=model_name,
                    api_key=zhipu_key,
                    base_url=base_url,
                    abilities=["chat", "tool_calling", "vision"],
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

            return GeminiLLM(
                model_name=model_name,
                api_key=gemini_key,
                base_url=base_url,
                abilities=["chat", "tool_calling", "vision"],
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

            return ClaudeLLM(
                model_name=model_name,
                api_key=claude_key,
                base_url=base_url,
                abilities=["chat", "tool_calling", "vision"],
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


@register_tool
async def create_vision_tools(config: "BaseToolConfig") -> List[Any]:
    """Create vision understanding tools."""
    vision_model = config.get_vision_model()
    if not vision_model:
        return []

    workspace = ToolFactory._create_workspace(config.get_workspace_config())

    try:
        return get_vision_tool(vision_model=vision_model, workspace=workspace)
    except Exception as e:
        logger.warning(f"Failed to create vision tools: {e}")
        return []
