from typing import Any, List

import aiohttp

from ...model import ImageModelConfig
from ...retry import create_retry_wrapper
from .base import BaseImageModel, InvalidImageResponseError, default_image_abilities
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
    abilities = list(
        db_model.abilities or default_image_abilities(provider, model_name)
    )
    timeout = getattr(db_model, "timeout", 300.0) or 300.0
    max_retries = getattr(db_model, "max_retries", 3) or 3
    # The row's own model_id, not a name+provider composite. `id` is what
    # create_image_model hands the provider as its billing identity, and the
    # aggregator groups on `model_id or model` -- a composite of the two
    # non-unique halves still collapses two same-name configurations of one
    # provider into a single billing group. Falls back to the composite only
    # when the row carries no model_id, so an identity is always recorded.
    configured_id = str(getattr(db_model, "model_id", "") or "").strip()

    # Create ImageModelConfig
    config = ImageModelConfig(
        id=configured_id or f"{model_name}-{provider}",
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
    """Whether an image-model failure is worth another provider call.

    Transport-level predicate, unused by ``create_image_model`` -- see
    ``retry_image_call`` for the policy that path installs.
    """
    ERRORS = aiohttp.ServerTimeoutError

    if isinstance(e, aiohttp.ClientResponseError):
        return e.status == 429 or 500 <= e.status < 600  # 429 and 5xx
    return isinstance(e, ERRORS)


def retry_image_call(e: Exception) -> bool:
    """Retry anything except an already-billed response that cannot improve.

    Deliberately as permissive as ``create_retry_wrapper``'s own
    ``lambda _: True`` default, minus one case. Gemini and DashScope flatten
    timeouts, network errors and 5xx into plain ``RuntimeError``, so a predicate
    narrow enough to name only transient types would stop retrying genuine
    transient failures -- a separate problem from this one, and not fixed by
    guessing from a message string.

    The excluded case is ``InvalidImageResponseError``: a 200 the provider
    already billed, whose body carries no usable image. Its metering row is
    already written (recorded before validation precisely because the charge is
    real), and the outcome does not change on a second attempt -- a
    safety-blocked prompt is refused just as deterministically on attempt ten.
    Retrying it multiplied both the provider bill and the recorded quantity by
    the retry count.

    Every retryable failure here is a genuinely separate billed call, so the
    per-attempt accounting the providers do stays correct.
    """
    return not isinstance(e, InvalidImageResponseError)


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
            model_id=model_config.id,
        )
    elif provider == "dashscope":
        llm = DashScopeImageModel(
            model_name=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url
            or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            timeout=model_config.timeout,
            abilities=model_config.abilities,
            model_id=model_config.id,
        )
    elif provider == "openai":
        llm = OpenAIImageModel(
            model_name=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            timeout=model_config.timeout,
            abilities=model_config.abilities,
            model_id=model_config.id,
        )
    elif provider == "xinference":
        llm = XinferenceImageModel(
            model_name=model_config.model_name,
            base_url=model_config.base_url,
            api_key=model_config.api_key,
            timeout=model_config.timeout,
            abilities=model_config.abilities,
            model_id=model_config.id,
        )
    else:
        raise ValueError(f"Unsupported image model provider: {provider}")

    # A retry predicate is passed explicitly: create_retry_wrapper defaults to
    # `lambda _: True`, which retried every exception -- including the
    # already-billed invalid-200 responses that providers meter before
    # validation, so one billed call became up to max_retries charges and
    # max_retries billing rows.
    return create_retry_wrapper(
        llm,
        BaseImageModel,  # type: ignore[type-abstract]
        retry_methods={"generate_image", "edit_image"},
        max_retries=model_config.max_retries,
        retry_on=retry_image_call,
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
