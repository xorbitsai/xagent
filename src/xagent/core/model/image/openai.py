import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse

import aiohttp
from openai import APIResponseValidationError, AsyncOpenAI

from ..chat.token_context import MediaCallType
from .base import (
    BaseImageModel,
    InvalidImageResponseError,
    call_billed_endpoint_async,
    image_url_from_item,
    invalid_response_from,
    resolve_requested_size,
)
from .usage import record_image_usage, record_unusable_response


def _openai_image_url(response: Any) -> Optional[str]:
    """The image URL from an OpenAI images response, or None when absent.

    Raises on a shape the SDK is not supposed to produce -- a truthy but
    non-indexable ``data`` -- rather than guessing. Callers meter first and then
    classify that as an already-billed invalid response.
    """
    if not response.data:
        return None
    return image_url_from_item(response.data[0])


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
        model_id: str = "",
    ):
        self.model_name = model_name
        # Configured id, not the provider-facing name. Kept on the provider
        # itself because the provider is what records usage: setting it on the
        # retry wrapper instead left the recording layer with only model_name,
        # and the aggregator groups on `model_id or model`, so two configured
        # models sharing a name collapsed into one billing group.
        self.model_id = model_id
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

    @property
    def _is_gpt_image(self) -> bool:
        """Whether this row points at the gpt-image family.

        Matched as a substring rather than a prefix because proxies routinely
        re-expose the model under a namespaced name such as
        ``openai.gpt-image-1``, and those serve the same request shape.
        """
        return "gpt-image" in self.model_name.lower()

    @property
    def supports_transparent_background(self) -> bool:
        """Only gpt-image-* accepts ``background``; DALL-E has no alpha at all."""
        return self._is_gpt_image

    def _transparency_kwargs(self) -> dict[str, Any]:
        """Request fields that make gpt-image-* return an alpha channel."""
        if not self.supports_transparent_background:
            raise RuntimeError(
                f"Model {self.model_name} cannot return a transparent "
                "background: only the gpt-image family accepts it."
            )
        # output_format matters as much as background: the default may be webp
        # or jpeg, and jpeg would silently flatten the alpha we just asked for.
        return {"background": "transparent", "output_format": "png"}

    def _apply_response_format(
        self, request_kwargs: dict[str, Any], response_format: Optional[str]
    ) -> None:
        """Set ``response_format`` only where the endpoint tolerates it.

        gpt-image-* rejects the field outright and always answers with
        ``b64_json``; every other OpenAI-compatible image endpoint needs it, and
        defaults to a URL as before.
        """
        if self._is_gpt_image:
            return
        request_kwargs["response_format"] = response_format or "url"

    def _ensure_client(self) -> None:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url
                if self.base_url != "https://api.openai.com/v1"
                else None,
                api_key=self.api_key,
                timeout=self.timeout,
                # The SDK retries twice by default, beneath this module's
                # billing boundary and invisible to it: each of those is a
                # separate charge that the boundary never sees and never
                # records. One retry layer, and it is the one that meters.
                max_retries=0,
            )

    def _normalize_size(self, size: Optional[str]) -> str:
        # Tolerates None because callers forward an unset size explicitly rather
        # than omitting it: image_tool always puts "size" in its params dict, and
        # a parameter default only covers a missing key, never an explicit None.
        if not size:
            return "1024x1024"
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
        transparent_background: bool = False,
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
            transparent_background: Ask for an alpha channel instead of an
                opaque background. Only gpt-image-* supports it; anything else
                raises rather than returning an opaque image as if it had worked.
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

        response_format = kwargs.pop("response_format", None)
        normalized_size = self._normalize_size(size)
        request_kwargs: dict[str, Any] = dict(kwargs)
        self._apply_response_format(request_kwargs, response_format)
        if transparent_background:
            request_kwargs.update(self._transparency_kwargs())

        # Captured for usage accounting: n reaches this provider only via
        # kwargs, and billing must reflect how many images were requested.
        image_count = request_kwargs.get("n", 1)
        images_client: Any = self._client.images
        # Wrapped for the same reason as the other providers: the SDK decodes
        # and validates the body itself, so a 200 it has already been billed for
        # but cannot parse surfaces from this call rather than from a body walk.
        # APIResponseValidationError is that case; status and connection errors
        # are separate types and stay retryable.
        try:
            response = await call_billed_endpoint_async(
                lambda: images_client.generate(
                    prompt=prompt,
                    model=self.model_name,
                    size=normalized_size,  # pyright: ignore[reportArgumentType]
                    **request_kwargs,
                ),
                "OpenAI returned an unusable body",
                also_billed=(APIResponseValidationError,),
            )
        except InvalidImageResponseError:
            # Metered before re-raising, for the same reason the success path
            # meters before walking the body: OpenAI charged for this response.
            # Classifying it without recording would stop the double-billing but
            # leave the one real charge invisible -- the exact failure this
            # boundary exists to prevent, in the other direction.
            #
            # CancelledError is deliberately NOT caught here, unlike the
            # DashScope decode boundary. That one wraps `response.json()` on a
            # 200 already in hand, so the charge is certain. This wraps the
            # whole SDK coroutine, and a cancellation during DNS, connect,
            # upload or the response-header wait happens *before* OpenAI accepts
            # the request -- recording there would invent a charge that was
            # never made. A cancellation after dispatch is an unknown outcome,
            # which needs idempotency and reconciliation (#2513), not a guess.
            record_unusable_response(
                model_name=self.model_name,
                model_id=self.model_id,
                call_type=MediaCallType.GENERATE_IMAGE,
                image_count=image_count,
                resolution=normalized_size,
            )
            raise

        # Metered before the response body is walked, as in the other
        # providers. The typed SDK is supposed to make `data` a list or None,
        # but that is a guarantee from another package: a truthy non-indexable
        # `data` raised a TypeError here, before the metering call, and the
        # retry policy treats a bare TypeError as retryable -- so one billed
        # call became max_retries charges with no row recorded.
        result = {
            "image_url": None,
            "usage": getattr(response, "usage", {}) or {},
            "request_id": getattr(response, "id", None),
        }
        record_image_usage(
            result,
            model_name=self.model_name,
            model_id=self.model_id,
            call_type=MediaCallType.GENERATE_IMAGE,
            image_count=image_count,
            resolution=str(normalized_size or ""),
        )

        try:
            result["image_url"] = _openai_image_url(response)
        except (TypeError, AttributeError, KeyError, IndexError) as e:
            raise invalid_response_from(e, "Invalid response format") from e
        return result

    async def edit_image(
        self,
        image_url: str | list[str],
        prompt: str,
        negative_prompt: str = "",
        transparent_background: bool = False,
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

        response_format = kwargs.pop("response_format", None)
        # Popped, not read: these are request-shaping parameters the tool layer
        # sends for edits too, and forwarding them into the SDK call would be an
        # unexpected field. Same precedence generate_image applies.
        size = self._normalize_size(
            resolve_requested_size(
                kwargs.pop("size", None),
                resolution=kwargs.pop("resolution", None),
                width=kwargs.pop("width", None),
                height=kwargs.pop("height", None),
            )
        )
        kwargs.pop("aspect_ratio", None)
        request_kwargs: dict[str, Any] = dict(kwargs)
        self._apply_response_format(request_kwargs, response_format)
        if transparent_background:
            request_kwargs.update(self._transparency_kwargs())

        # Mirrors generate_image: n reaches this provider only via kwargs, and
        # the provider generates and bills the full count.
        image_count = request_kwargs.get("n", 1)

        image_files = []
        try:
            for path in image_paths:
                image_files.append(open(path, "rb"))
            images_client: Any = self._client.images
            try:
                response = await call_billed_endpoint_async(
                    lambda: images_client.edit(
                        image=image_files if len(image_files) > 1 else image_files[0],
                        prompt=prompt,
                        model=self.model_name,
                        size=size,  # pyright: ignore[reportArgumentType]
                        **request_kwargs,
                    ),
                    "OpenAI returned an unusable body",
                    also_billed=(APIResponseValidationError,),
                )
            except InvalidImageResponseError:
                # See generate_image, including why CancelledError is not
                # caught here.
                record_unusable_response(
                    model_name=self.model_name,
                    model_id=self.model_id,
                    call_type=MediaCallType.EDIT_IMAGE,
                    image_count=image_count,
                    resolution=size,
                )
                raise
        finally:
            for image_file in image_files:
                try:
                    image_file.close()
                except Exception:
                    pass
            for temp_path in temp_paths:
                Path(temp_path).unlink(missing_ok=True)

        # See generate_image: metered before the body walk, and a body-walk
        # failure classified as an invalid response rather than retried.
        result = {
            "image_url": None,
            "usage": getattr(response, "usage", {}) or {},
            "request_id": getattr(response, "id", None),
        }
        record_image_usage(
            result,
            model_name=self.model_name,
            model_id=self.model_id,
            call_type=MediaCallType.EDIT_IMAGE,
            image_count=image_count,
            resolution=str(size or ""),
        )

        try:
            result["image_url"] = _openai_image_url(response)
        except (TypeError, AttributeError, KeyError, IndexError) as e:
            raise invalid_response_from(e, "Invalid response format") from e
        return result
