from typing import Any, List

import aiohttp

from ...model import ImageModelConfig
from ...retry import create_retry_wrapper
from .base import BaseImageModel
from .dashscope import DashScopeImageModel
from .gemini import GeminiImageModel
from .openai import OpenAIImageModel
from .xinference import XinferenceImageModel


def get_image_model_instance(db_model: Any) -> BaseImageModel:
    """
    Create a BaseImageModel instance from a database model record.

    Args:
        db_model: Database model instance with fields: model_name, model_provider,
                  api_key, base_url, abilities, timeout, max_retries

    Returns:
        BaseImageModel instance with retry wrapper

    Raises:
        ValueError: If provider is not supported or required fields are missing
    """
    provider = str(db_model.model_provider).lower()
    model_name = str(db_model.model_name)
    api_key = str(db_model.api_key) if db_model.api_key else None
    base_url = str(db_model.base_url) if db_model.base_url else None
    abilities = list(db_model.abilities) if db_model.abilities else ["generate"]
    timeout = getattr(db_model, "timeout", 300.0) or 300.0
    max_retries = getattr(db_model, "max_retries", 3) or 3

    # Create ImageModelConfig
    config = ImageModelConfig(
        id=f"{model_name}-{provider}",
        model_name=model_name,
        model_provider=provider,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        abilities=abilities,
        max_retries=max_retries,
    )

    return create_image_model(config)


def retry_on(e: Exception) -> bool:
    ERRORS = aiohttp.ServerTimeoutError

    if isinstance(e, aiohttp.ClientResponseError):
        return e.status == 429 or 500 <= e.status < 600  # 429 and 5xx
    return isinstance(e, ERRORS)


def create_image_model(model_config: ImageModelConfig) -> BaseImageModel:
    """
    Creates a custom BaseImageModel instance from an ImageModelConfig.
    """
    if not isinstance(model_config, ImageModelConfig):
        raise TypeError(f"Invalid model type: {type(model_config).__name__}")

    provider = model_config.model_provider.lower()

    llm: BaseImageModel

    if provider == "gemini":
        llm = GeminiImageModel(
            model_name=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            timeout=model_config.timeout,
            abilities=model_config.abilities,
        )
    elif provider == "dashscope":
        llm = DashScopeImageModel(
            model_name=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url
            or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            timeout=model_config.timeout,
            abilities=model_config.abilities,
        )
    elif provider == "openai":
        llm = OpenAIImageModel(
            model_name=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            timeout=model_config.timeout,
            abilities=model_config.abilities,
        )
    elif provider == "xinference":
        llm = XinferenceImageModel(
            model_name=model_config.model_name,
            base_url=model_config.base_url,
            api_key=model_config.api_key,
            timeout=model_config.timeout,
            abilities=model_config.abilities,
        )
    else:
        raise ValueError(f"Unsupported image model provider: {provider}")

    return create_retry_wrapper(
        llm,
        BaseImageModel,  # type: ignore[type-abstract]
        retry_methods={"generate_image", "edit_image"},
        max_retries=model_config.max_retries,
    )


class ImageModelAdapter(BaseImageModel):
    """Adapter that makes the new image interface compatible with existing ImageModelConfig configs."""

    def __init__(self, model_config: ImageModelConfig):
        self.model_config = model_config
        self._image_model = create_image_model(model_config)

    @property
    def abilities(self) -> List[str]:
        """
        Get the list of abilities supported by the underlying image model.

        Returns:
            List[str]: List of supported abilities
        """
        return self._image_model.abilities

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024*1024",
        negative_prompt: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text prompt for image generation
            size: Image size in format "width*height" (e.g., "1024*1024")
            negative_prompt: Negative prompt for image generation
            **kwargs: Additional parameters specific to the model

        Returns:
            dict with image generation result containing:
            - image_url: URL of the generated image
            - usage: Image generation usage statistics
            - request_id: Request identifier
        """

        return await self._image_model.generate_image(
            prompt=prompt, size=size, negative_prompt=negative_prompt
        )

    async def edit_image(
        self,
        image_url: str | list[str],
        prompt: str,
        negative_prompt: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Edit an image using a text prompt.

        Args:
            image_url: URL of the source image to edit (or list of URLs)
            prompt: Text prompt describing the desired edits
            negative_prompt: Negative prompt for image generation
            **kwargs: Additional parameters specific to the model

        Returns:
            dict with image editing result containing:
            - image_url: URL of the edited image
            - usage: Image generation usage statistics
            - request_id: Request identifier
        """
        # Merge config_data with kwargs, kwargs takes precedence

        return await self._image_model.edit_image(
            image_url=image_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
