"""
Gemini image generation model implementation.

This module provides image generation capabilities using Google's Gemini API.
Supports Gemini models like gemini-2.5-flash-image for text-to-image generation.
"""

import logging
import os
from math import gcd
from typing import Any, Dict, List, Optional

import httpx

from ...utils.security import redact_sensitive_text, redact_url_credentials_for_logging
from .base import BaseImageModel

logger = logging.getLogger(__name__)


def _parse_size_to_gemini_config(size: str, model_name: str) -> Dict[str, str]:
    """
    Parse size string to Gemini image configuration.

    Gemini uses aspectRatio and imageSize instead of direct dimensions.

    Args:
        size: Size string like "1920*1080" or "1920x1080"
        model_name: Model name (e.g., "gemini-3-pro-image-preview-2k")

    Returns:
        Dict with 'aspectRatio' and 'imageSize' keys for Gemini API

    Example:
        >>> _parse_size_to_gemini_config("1920*1080", "gemini-3-pro-image-preview-2k")
        {'aspectRatio': '16:9', 'imageSize': '2K'}
    """
    # Supported aspect ratios by Gemini
    supported_ratios = [
        "1:1",
        "2:3",
        "3:2",
        "3:4",
        "4:3",
        "4:5",
        "5:4",
        "9:16",
        "16:9",
        "21:9",
    ]

    # Determine max resolution based on model name
    model_name_lower = model_name.lower()
    if "2k" in model_name_lower:
        max_resolution = 2048
    elif "4k" in model_name_lower:
        max_resolution = 4096
    else:
        # Default to 2K for unknown models
        max_resolution = 2048

    # Normalize size string
    size_normalized = size.replace("*", "x")

    try:
        if "x" not in size_normalized:
            # Default to square if invalid format
            return {"aspectRatio": "1:1", "imageSize": "1K"}

        width_str, height_str = size_normalized.split("x")
        width = int(width_str)
        height = int(height_str)

        # Calculate aspect ratio
        divisor = gcd(width, height)
        aspect_ratio = f"{width // divisor}:{height // divisor}"

        # Use exact ratio if supported, otherwise find closest
        if aspect_ratio in supported_ratios:
            ratio = aspect_ratio
        else:
            # Find closest supported ratio by comparing decimal values
            target_ratio = width / height
            ratio = min(
                supported_ratios,
                key=lambda r: abs(
                    (lambda x: int(x.split(":")[0]) / int(x.split(":")[1]))(r)
                    - target_ratio
                ),
            )

        # Determine image size based on max dimension and model limit
        max_dimension = max(width, height)

        # First determine what size bucket the request falls into
        if max_dimension <= 1024:
            requested_size = "1K"
        elif max_dimension <= 2048:
            requested_size = "2K"
        elif max_dimension <= 4096:
            requested_size = "4K"
        else:
            requested_size = "4K"

        # Clamp to model's maximum supported resolution
        if max_resolution < 1024:
            image_size = "1K"
        elif max_resolution < 2048:
            image_size = "1K"
        elif max_resolution < 4096:
            # 2K model - cap at 2K
            image_size = "1K" if requested_size == "1K" else "2K"
        else:
            # 4K model - support all sizes
            image_size = requested_size

        return {"aspectRatio": ratio, "imageSize": image_size}

    except (ValueError, IndexError, ZeroDivisionError):
        # Return defaults if parsing fails
        return {"aspectRatio": "1:1", "imageSize": "1K"}


class GeminiImageModel(BaseImageModel):
    """
    Gemini image generation model client.

    Supports text-to-image generation using Google's Gemini models via the generateContent API.
    Compatible with models like:
    - gemini-2.5-flash-image (Imagen 3 powered)
    - gemini-2.0-flash-exp-image-gen
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-image",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        abilities: Optional[List[str]] = None,
    ):
        """
        Initialize Gemini image generation model.

        Args:
            model_name: Model identifier (e.g., "gemini-2.5-flash-image")
            api_key: Google API key for authentication
            base_url: Base URL for API (optional, defaults to official Google API)
            timeout: Request timeout in seconds
            abilities: List of supported abilities (defaults to ["generate"])
        """
        self.model_name = model_name
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        self.base_url = (
            base_url if base_url else "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self.timeout = timeout
        self._abilities = abilities or ["generate"]

        # Gemini image models don't support image editing by default
        if "edit" in self._abilities:
            logger.warning(
                "Gemini image models don't support image editing. Removing 'edit' ability."
            )
            self._abilities = ["generate"]

    @property
    def abilities(self) -> List[str]:
        """
        Get the list of abilities supported by this Gemini image model.

        Returns:
            List[str]: List of supported abilities (typically ["generate"])
        """
        return self._abilities

    def has_ability(self, ability: str) -> bool:
        """
        Check if this image model implementation supports a specific ability.

        Args:
            ability: The ability to check

        Returns:
            bool: True if the ability is supported, False otherwise
        """
        return ability in self.abilities

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024*1024",
        negative_prompt: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate an image from a text prompt using Gemini API.

        Uses the generateContent endpoint with the Gemini model.
        For models like gemini-2.5-flash-image that have Imagen 3 capabilities.

        Args:
            prompt: Text prompt for image generation
            size: Image size in format "width*height" (e.g., "1024*1024")
            negative_prompt: Negative prompt for image generation (passed in request)
            **kwargs: Additional parameters (e.g., temperature, etc.)

        Returns:
            dict with image generation result containing:
            - image_url: Data URL of the generated image (base64 encoded)
            - usage: Image generation usage statistics
            - request_id: Request identifier

        Raises:
            RuntimeError: If the API call fails or model doesn't support generation
        """
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")

        if not self.has_ability("generate"):
            raise RuntimeError("This model doesn't support image generation")

        # Build the API URL using generateContent endpoint
        is_official_api = "googleapis.com" in self.base_url

        if is_official_api:
            api_url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
            headers = {}
        else:
            # For proxy services
            api_url = f"{self.base_url}/models/{self.model_name}:generateContent"
            headers = {"Authorization": f"Bearer {self.api_key}"}

        # Prepare request body following Gemini API format
        # The prompt should be in the contents.parts[0].text field
        request_body: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}

        # Add generation config
        gen_config: dict[str, Any] = {"responseModalities": ["Image"]}

        # Add image configuration based on size
        image_config = _parse_size_to_gemini_config(size, self.model_name)
        if image_config:
            gen_config["imageConfig"] = image_config

        if kwargs:
            if "temperature" in kwargs:
                gen_config["temperature"] = kwargs["temperature"]
        request_body["generationConfig"] = gen_config

        # Log the request
        logger.info(
            "Gemini image generation API URL: %s",
            redact_url_credentials_for_logging(api_url),
        )
        logger.debug(f"Image config: {image_config}")
        logger.debug(f"Request prompt: {prompt[:100]}...")

        try:
            # Make the async HTTP request
            timeout = httpx.Timeout(self.timeout, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    api_url,
                    json=request_body,
                    headers=headers,
                )

                logger.info(
                    f"Gemini image generation response status: {response.status_code}"
                )

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(
                        "Gemini image generation error: %s",
                        redact_sensitive_text(error_text),
                    )
                    response.raise_for_status()

                response_data = response.json()

            # Parse the response
            # Gemini API returns:
            # {
            #   "candidates": [
            #     {
            #       "content": {
            #         "parts": [
            #           {
            #             "inlineData": {
            #               "mimeType": "image/png",
            #               "data": "base64_encoded_image_data"
            #             }
            #           }
            #         ]
            #       },
            #       "finishReason": "STOP",
            #       ...
            #     }
            #   ],
            #   "usageMetadata": {...}
            # }

            candidates = response_data.get("candidates", [])
            if not candidates:
                raise RuntimeError("No candidates in response")

            first_candidate = candidates[0]
            finish_reason = first_candidate.get("finishReason")
            content = first_candidate.get("content", {})
            parts = content.get("parts", [])

            if not parts:
                raise RuntimeError("No parts in response content")

            # Look for inlineData with image (base64 encoded)
            image_url = None
            for part in parts:
                inline_data = part.get("inlineData")
                if inline_data:
                    mime_type = inline_data.get("mimeType", "image/png")
                    base64_data = inline_data.get("data")
                    if base64_data:
                        image_url = f"data:{mime_type};base64,{base64_data}"
                        break

            # If no inlineData, check for Markdown image link in text response
            if not image_url:
                for part in parts:
                    text = part.get("text", "")
                    if text:
                        # Extract image URL from Markdown format: ![Image](url)
                        import re

                        match = re.search(r"!\[.*?\]\((https?://[^\)]+)\)", text)
                        if match:
                            image_url = match.group(1)
                            break

            if not image_url:
                if finish_reason and finish_reason != "STOP":
                    raise RuntimeError(
                        f"Image generation failed with finish reason: {finish_reason}"
                    )
                raise RuntimeError("No image data in response")

            # Extract usage metadata
            usage_metadata = response_data.get("usageMetadata", {})

            # Build token usage info
            token_usage = {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0),
            }

            return {
                "image_url": image_url,
                "usage": token_usage,
                "request_id": response_data.get("requestId"),
                "finish_reason": finish_reason,
                "raw_response": response_data,
            }

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                "Gemini image generation API error "
                f"({e.response.status_code}): {redact_sensitive_text(e.response.text)}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(f"Image generation timeout: {str(e)}") from e
        except httpx.NetworkError as e:
            raise RuntimeError(
                f"Network error during image generation: {str(e)}"
            ) from e
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
        Image editing is not supported by Gemini image models.

        This method is not implemented as Gemini image models don't support image editing.

        Args:
            image_url: URL of the source image to edit
            prompt: Text prompt describing the desired edits
            negative_prompt: Negative prompt for image generation
            **kwargs: Additional parameters

        Raises:
            RuntimeError: Always - image editing is not supported
        """
        raise RuntimeError(
            "Image editing is not supported by Gemini image models. "
            "Please use a different model provider for image editing capabilities."
        )
