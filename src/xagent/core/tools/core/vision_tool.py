"""
Pure Vision Tool Core
Standalone vision capabilities without framework dependencies
"""

import base64
import logging
import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ...model.chat.basic.base import BaseLLM

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class UnderstandImagesResult(BaseModel):
    """Return model for understand_images method"""

    success: bool
    answer: Optional[str] = None
    images_processed: Optional[int] = None
    model_used: Optional[str] = None
    error: Optional[str] = None


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
            Dictionary with analysis result and metadata
        """
        try:
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

                return UnderstandImagesResult(
                    success=False,
                    error=f"{model_info} does not support vision capabilities",
                )

            # Validate and normalize images
            validated_images = self._validate_images(images)

            # Convert images to appropriate format
            image_contents: List[Dict[str, Any]] = []
            for img_path in validated_images:
                try:
                    if img_path.startswith(("http://", "https://")):
                        image_contents.append(
                            {"type": "image_url", "image_url": {"url": img_path}}
                        )
                    elif img_path.startswith("data:"):
                        image_contents.append(
                            {"type": "image_url", "image_url": {"url": img_path}}
                        )
                    else:
                        base64_data = self._convert_image_to_base64(img_path)
                        image_contents.append(
                            {"type": "image_url", "image_url": {"url": base64_data}}
                        )
                except Exception as e:
                    logger.warning(f"Failed to process image {img_path}: {e}")
                    continue

            if not image_contents:
                return UnderstandImagesResult(
                    success=False, error="No valid images could be processed"
                )

            # Prepare the message content with images and question
            content = [{"type": "text", "text": question}]
            content.extend(image_contents)

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

            return UnderstandImagesResult(
                success=True,
                answer=answer,
                images_processed=len(image_contents),
                model_used=self.vision_model.__class__.__name__,
            )

        except Exception as e:
            logger.error(f"Image understanding failed: {e}")
            return UnderstandImagesResult(success=False, error=str(e))

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
                temperature=temperature or 0.1,
                max_tokens=max_tokens or 2000,
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
