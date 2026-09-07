import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, List, Optional

import aiohttp

from ..chat.token_context import MediaCallType
from .base import (
    BaseImageModel,
    InvalidImageResponseError,
    invalid_response_from,
)
from .usage import record_image_usage


class DashScopeImageModel(BaseImageModel):
    """
    DashScope image generation model client.
    Supports text-to-image generation using the DashScope API.
    """

    def __init__(
        self,
        model_name: str = "qwen-image",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        timeout: float = 60.0,
        abilities: Optional[List[str]] = None,
        model_id: str = "",
    ):
        self.model_name = model_name
        # See OpenAIImageModel: this provider records its own usage, so the
        # configured id must live here rather than on the retry wrapper.
        self.model_id = model_id
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self._abilities = abilities or ["generate"]

    def _convert_image_to_base64(self, image_input: str) -> str:
        """
        Convert image input to Base64 format suitable for DashScope API.

        Args:
            image_input: Either a URL string or a local file path

        Returns:
            str: Base64-encoded image string in format "data:{mime_type};base64,{base64_data}"

        Raises:
            RuntimeError: If the image cannot be processed or doesn't meet requirements
        """
        # Check if it's a URL (http/https)
        if image_input.startswith(("http://", "https://")):
            # Validate URL doesn't contain Chinese characters
            if any(ord(char) > 127 for char in image_input):
                raise RuntimeError("Image URL cannot contain Chinese characters")
            return image_input

        # Treat as local file path
        image_path = Path(image_input)

        # Check if file exists
        if not image_path.exists():
            raise RuntimeError(f"Image file not found: {image_input}")

        # Check file size (max 10MB)
        file_size = image_path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise RuntimeError(f"Image file too large: {file_size} bytes (max 10MB)")

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            # Default to jpeg if can't determine type
            mime_type = "image/jpeg"

        # Validate supported formats
        supported_formats = [
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/bmp",
            "image/tiff",
            "image/webp",
        ]
        if mime_type.lower() not in supported_formats:
            raise RuntimeError(
                f"Unsupported image format: {mime_type}. Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP"
            )

        # Read and encode file
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_data = base64.b64encode(image_data).decode("utf-8")
                return f"data:{mime_type};base64,{base64_data}"
        except Exception as e:
            raise RuntimeError(f"Failed to read image file: {e}")

    @property
    def abilities(self) -> List[str]:
        """
        Get the list of abilities supported by this DashScope image model.
        Default abilities: ["generate"]

        Returns:
            List[str]: List of supported abilities
        """
        return self._abilities

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024*1024",
        negative_prompt: str = "",
        resolution: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate an image from a text prompt using DashScope API.

        Args:
            prompt: Text prompt for image generation
            size: Image size in format "width*height" (e.g., "1024*1024")
            negative_prompt: Negative prompt for image generation
            resolution: Alternative size specification (e.g., "1920x1080")
            width: Image width in pixels
            height: Image height in pixels
            aspect_ratio: Aspect ratio (e.g., "3:2", "16:9") - DashScope uses size instead
            **kwargs: Additional parameters specific to the model
                     For DashScope: prompt_extend, watermark, etc.

        Returns:
            dict with image generation result containing:
            - image_url: URL of the generated image
            - usage: Image generation usage statistics
            - request_id: Request identifier

        Raises:
            RuntimeError: If the API call fails or model doesn't support generation
        """
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required")

        if not self.has_ability("generate"):
            raise RuntimeError("This model doesn't support image generation")

        # Handle alternative size parameters
        # DashScope uses simple size format like "1024*1024"
        # Priority: resolution > width+height > size
        # Note: aspect_ratio is not supported by DashScope, will be ignored
        if aspect_ratio:
            # DashScope doesn't support aspect_ratio parameter
            # Log a warning but continue with the base size
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"aspect_ratio parameter '{aspect_ratio}' is not supported by DashScope API, using size '{size}' instead"
            )
        elif resolution:
            # resolution format: "1920x1080" -> "1920*1080"
            size = resolution.replace("x", "*")
        elif width and height:
            # width + height format
            size = f"{width}*{height}"

        # Prepare the request payload
        # Set default values for DashScope-specific parameters
        prompt_extend = kwargs.pop("prompt_extend", True)
        watermark = kwargs.pop("watermark", False)

        payload = {
            "model": self.model_name,
            "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
            "parameters": {
                "negative_prompt": negative_prompt,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
                "size": size,
                **kwargs,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            # Make the API call
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    self.base_url, headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"DashScope API error ({response.status}): {error_text}"
                        )

                    response_data = await response.json()

            # Meter before validating the response body. DashScope has already
            # billed this 200, and the `usage` payload sits at the top level —
            # independent of the `output` structure the checks below walk. A
            # billed-but-unparseable response would otherwise go unmetered.
            # Read defensively: this runs before the metering call, in the same
            # try as the structural checks, so a 200 whose body is a JSON array
            # would otherwise raise AttributeError here and lose the row for a
            # call DashScope did charge for.
            usage = (
                response_data.get("usage", {})
                if isinstance(response_data, dict)
                else {}
            )
            record_image_usage(
                {"usage": usage},
                model_name=self.model_name,
                model_id=self.model_id,
                call_type=MediaCallType.GENERATE_IMAGE,
                # `n` genuinely reaches DashScope -- **kwargs is spread into
                # the request `parameters` below, so the provider generates and
                # bills for n images -- but a partially successful 200 bills
                # fewer than it was asked for, and only `usage` knows that. So
                # `n` is the fallback and the reported count wins.
                # Note the parser only returns content[0], so images 2..n are
                # paid for and discarded — a separate defect from metering.
                image_count=kwargs.get("n", 1),
                # `*` -> `x`: the reported pair is `WxH`, so passing DashScope's
                # own `W*H` vocabulary through unchanged would file one physical
                # resolution under two aggregate keys.
                resolution=str(size or "").replace("*", "x")
                if isinstance(size, str)
                else "",
                # Walked inside the helper's swallow, not here: a payload whose
                # `get` raises would otherwise fail this whole call and record
                # nothing, even for a 200 that returned a real image.
                reported_count_from=usage,
                reported_size_from=usage,
            )

            # Extract the image URL from the response
            if "output" not in response_data:
                raise InvalidImageResponseError(
                    "Invalid response format: missing 'output' field"
                )

            output = response_data["output"]
            if "choices" not in output or not output["choices"]:
                raise InvalidImageResponseError(
                    "Invalid response format: missing 'choices' field"
                )

            choice = output["choices"][0]
            if "message" not in choice:
                raise InvalidImageResponseError(
                    "Invalid response format: missing 'message' field"
                )

            message = choice["message"]
            if "content" not in message or not message["content"]:
                raise InvalidImageResponseError(
                    "Invalid response format: missing 'content' field"
                )

            content = message["content"]
            if not isinstance(content, list) or not content:
                raise InvalidImageResponseError(
                    "Invalid response format: content should be a non-empty list"
                )

            image_item = content[0]
            if "image" not in image_item:
                raise InvalidImageResponseError(
                    "Invalid response format: missing 'image' field"
                )

            image_url = image_item["image"]

            # Usage was already recorded above, before validation.
            return {
                "image_url": image_url,
                "usage": usage,
                "task_metric": output.get("task_metric", {}),
                "request_id": response_data.get("request_id"),
                "raw_response": response_data,
            }

        except InvalidImageResponseError:
            # Re-raised unchanged: the blanket handler below would wrap it in a
            # plain RuntimeError and erase the non-retryable classification,
            # re-billing an already-billed and already-metered response.
            raise
        except (TypeError, AttributeError, KeyError, IndexError) as e:
            # Same classification, reached implicitly. The structural checks
            # above validate container shapes but not element types, so a 200
            # whose content is `[null]` or `[123]` fails on `"image" not in
            # image_item` with a TypeError rather than a raise. Enumerating
            # every malformed shape a provider could send is a losing game, so
            # anything that fails walking an already-metered body is classified
            # positionally. The request half of this try raises aiohttp and JSON
            # errors, matched by the clauses below, not these types.
            raise invalid_response_from(e, "Invalid response format") from e
        except aiohttp.ClientError as e:
            raise RuntimeError(
                f"Network error during image generation: {str(e)}"
            ) from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {str(e)}") from e

    async def edit_image(
        self,
        image_url: str | list[str],
        prompt: str,
        negative_prompt: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Edit an image using DashScope image editing API.

        Args:
            image_url: URL (or a list of URLs) of the source image(s) to edit
            prompt: Text prompt describing the desired edits
            negative_prompt: Negative prompt for image generation
            kwargs: Additional parameters specific to the model

        Returns:
            dict: A dict with image editing result containing:

                - image_url: URL of the edited image
                - usage: Image generation usage statistics
                - request_id: Request identifier

        Raises:
            RuntimeError: If the API call fails or image cannot be processed
        """
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required")

        # Get watermark parameter from kwargs
        watermark = kwargs.get("watermark", False)

        image_inputs = image_url if isinstance(image_url, list) else [image_url]
        if not image_inputs:
            raise RuntimeError("At least one input image is required")

        # Convert image input to appropriate format
        processed_images = [
            self._convert_image_to_base64(image_input) for image_input in image_inputs
        ]

        # Prepare the request payload for image editing
        payload = {
            "model": self.model_name,  # Use the configured model name
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            *({"image": image_item} for image_item in processed_images),
                            {"text": prompt},
                        ],
                    }
                ]
            },
            "parameters": {
                "negative_prompt": negative_prompt,
                "watermark": watermark,
                **kwargs,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            # Make the API call
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    self.base_url, headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"DashScope API error ({response.status}): {error_text}"
                        )

                    response_data = await response.json()

            # Meter before validating the response body — see generate_image.
            # Read defensively: this runs before the metering call, in the same
            # try as the structural checks, so a 200 whose body is a JSON array
            # would otherwise raise AttributeError here and lose the row for a
            # call DashScope did charge for.
            usage = (
                response_data.get("usage", {})
                if isinstance(response_data, dict)
                else {}
            )
            record_image_usage(
                {"usage": usage},
                model_name=self.model_name,
                model_id=self.model_id,
                call_type=MediaCallType.EDIT_IMAGE,
                # Same rule as generate_image: prefer what DashScope reports
                # it billed, fall back to the request's `n`/`size`. A non-string
                # size is dropped rather than stringified -- this path reads it
                # straight from caller kwargs with no normalisation, and an int
                # would become a half-resolution key joining no price table.
                image_count=kwargs.get("n", 1),
                resolution=str(kwargs.get("size") or "").replace("*", "x")
                if isinstance(kwargs.get("size"), str)
                else "",
                reported_count_from=usage,
                reported_size_from=usage,
            )

            # Extract the image URL from the response (same structure as generation)
            if "output" not in response_data:
                raise InvalidImageResponseError(
                    "Invalid response format: missing 'output' field"
                )

            output = response_data["output"]
            if "choices" not in output or not output["choices"]:
                raise InvalidImageResponseError(
                    "Invalid response format: missing 'choices' field"
                )

            choice = output["choices"][0]
            if "message" not in choice:
                raise InvalidImageResponseError(
                    "Invalid response format: missing 'message' field"
                )

            message = choice["message"]
            if "content" not in message or not message["content"]:
                raise InvalidImageResponseError(
                    "Invalid response format: missing 'content' field"
                )

            content = message["content"]
            if not isinstance(content, list) or not content:
                raise InvalidImageResponseError(
                    "Invalid response format: content should be a non-empty list"
                )

            image_item = content[0]
            if "image" not in image_item:
                raise InvalidImageResponseError(
                    "Invalid response format: missing 'image' field"
                )

            image_url = image_item["image"]

            # Usage was already recorded above, before validation.
            return {
                "image_url": image_url,
                "usage": usage,
                "task_metric": output.get("task_metric", {}),
                "request_id": response_data.get("request_id"),
                "raw_response": response_data,
            }

        except InvalidImageResponseError:
            # Re-raised unchanged: the blanket handler below would wrap it in a
            # plain RuntimeError and erase the non-retryable classification,
            # re-billing an already-billed and already-metered response.
            raise
        except (TypeError, AttributeError, KeyError, IndexError) as e:
            # Same classification, reached implicitly. The structural checks
            # above validate container shapes but not element types, so a 200
            # whose content is `[null]` or `[123]` fails on `"image" not in
            # image_item` with a TypeError rather than a raise. Enumerating
            # every malformed shape a provider could send is a losing game, so
            # anything that fails walking an already-metered body is classified
            # positionally. The request half of this try raises aiohttp and JSON
            # errors, matched by the clauses below, not these types.
            raise invalid_response_from(e, "Invalid response format") from e
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error during image editing: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Image editing failed: {str(e)}") from e
