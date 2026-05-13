import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse

import aiohttp
from openai import AsyncOpenAI

from .base import BaseImageModel


class OpenAIImageModel(BaseImageModel):
    """
    OpenAI-compatible image generation/editing client using the official OpenAI SDK.
    """

    def __init__(
        self,
        model_name: str = "gpt-image-1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 3600.0,
        abilities: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.timeout = timeout
        self._abilities = abilities or ["generate", "edit"]
        self._client: Optional[AsyncOpenAI] = None

    @property
    def abilities(self) -> List[str]:
        return self._abilities

    def _ensure_client(self) -> None:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url
                if self.base_url != "https://api.openai.com/v1"
                else None,
                api_key=self.api_key,
                timeout=self.timeout,
            )

    def _normalize_size(self, size: str) -> str:
        if "*" in size:
            return size.replace("*", "x")
        return size

    async def _download_url(self, url: str) -> str:
        parsed = urlparse(url)
        extension = Path(parsed.path).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_path = tmp_file.name

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Failed to download image: HTTP {response.status}"
                        )
                    with open(tmp_path, "wb") as output_file:
                        async for chunk in response.content.iter_chunked(8192):
                            output_file.write(chunk)
            return tmp_path
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

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
        Generate an image using OpenAI-compatible API.

        Args:
            prompt: Text prompt for image generation
            size: Image size in format "width*height" (e.g., "1024*1024")
            negative_prompt: Negative prompt (not supported by all providers)
            resolution: Alternative size specification (e.g., "1920x1080")
            width: Image width in pixels
            height: Image height in pixels
            aspect_ratio: Aspect ratio (e.g., "3:2", "16:9")
            **kwargs: Additional parameters (response_format, etc.)

        Returns:
            dict with image generation result
        """
        if not self.has_ability("generate"):
            raise RuntimeError("This model doesn't support image generation")

        # Handle alternative size parameters
        # OpenAI API uses simple size format like "1024x1024"
        # Priority: resolution > width+height > size
        # Note: aspect_ratio is not directly supported, use size instead
        if aspect_ratio:
            # OpenAI doesn't support aspect_ratio parameter directly
            # Log a warning but continue with the base size
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"aspect_ratio parameter '{aspect_ratio}' is not directly supported by OpenAI API, using size '{size}' instead"
            )
        elif resolution:
            # resolution format: "1920x1080" -> "1920x1080" (already in correct format)
            size = resolution.replace("x", "x")  # Normalize to use "x"
        elif width and height:
            # width + height format: convert to "WxH" format
            size = f"{width}x{height}"

        self._ensure_client()
        assert self._client is not None

        response_format = kwargs.pop("response_format", "url")
        images_client: Any = self._client.images
        response = await images_client.generate(
            prompt=prompt,
            model=self.model_name,
            size=self._normalize_size(size),  # pyright: ignore[reportArgumentType]
            response_format=response_format,
            **kwargs,
        )

        image_url = None
        if response.data:
            image_item = response.data[0]
            if getattr(image_item, "url", None):
                image_url = image_item.url
            elif getattr(image_item, "b64_json", None):
                image_url = f"data:image/png;base64,{image_item.b64_json}"

        return {
            "image_url": image_url,
            "usage": getattr(response, "usage", {}) or {},
            "request_id": getattr(response, "id", None),
        }

    async def edit_image(
        self,
        image_url: str | list[str],
        prompt: str,
        negative_prompt: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.has_ability("edit"):
            raise RuntimeError("This model doesn't support image editing")

        self._ensure_client()
        assert self._client is not None

        image_inputs = image_url if isinstance(image_url, list) else [image_url]
        if not image_inputs:
            raise RuntimeError("At least one input image is required")

        temp_paths: list[str] = []
        image_paths: list[str] = []
        for image_input in image_inputs:
            image_path = image_input
            if image_path.startswith(("http://", "https://")):
                temp_path = await self._download_url(image_path)
                temp_paths.append(temp_path)
                image_path = temp_path
            image_paths.append(image_path)

        response_format = kwargs.pop("response_format", "url")
        image_files = []
        try:
            image_files = [open(path, "rb") for path in image_paths]
            images_client: Any = self._client.images
            response = await images_client.edit(
                image=image_files if len(image_files) > 1 else image_files[0],
                prompt=prompt,
                model=self.model_name,
                size=self._normalize_size(kwargs.pop("size", "1024*1024")),  # pyright: ignore[reportArgumentType]
                response_format=response_format,
                **kwargs,
            )
        finally:
            for image_file in image_files:
                try:
                    image_file.close()
                except Exception:
                    pass
            for temp_path in temp_paths:
                Path(temp_path).unlink(missing_ok=True)

        response_image_url: str | None = None
        if response.data:
            image_item = response.data[0]
            if getattr(image_item, "url", None):
                response_image_url = image_item.url
            elif getattr(image_item, "b64_json", None):
                response_image_url = f"data:image/png;base64,{image_item.b64_json}"

        return {
            "image_url": response_image_url,
            "usage": getattr(response, "usage", {}) or {},
            "request_id": getattr(response, "id", None),
        }
