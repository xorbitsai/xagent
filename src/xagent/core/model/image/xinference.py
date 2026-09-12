"""Xinference Image provider implementation."""

import logging
from typing import Any, List, Optional

from xinference_client import RESTfulClient as XinferenceClient

from ..chat.token_context import MediaCallType
from .base import (
    BaseImageModel,
    InvalidImageResponseError,
    invalid_response_from,
)
from .usage import record_image_usage

logger = logging.getLogger(__name__)


def _xinference_usage(result: Any) -> Any:
    """Usage payload from an xinference result, without raising.

    Read before the metering call, so it must not be the thing that loses the
    row: the client hands back raw server JSON, which may be any shape.
    """
    try:
        if isinstance(result, dict):
            return result.get("usage") or {}
        return getattr(result, "usage", {}) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("Reading xinference usage failed: %s", e)
        return {}


def _xinference_image_url(result: Any) -> Optional[str]:
    """The image URL from an xinference result, or None when absent.

    Raises on a malformed body rather than guessing; callers classify that as
    an already-billed invalid response.
    """
    if hasattr(result, "data") and result.data:
        image_item = result.data[0]
        if hasattr(image_item, "url"):
            return str(image_item.url)
        if hasattr(image_item, "b64_json"):
            return f"data:image/png;base64,{image_item.b64_json}"
        return None
    if isinstance(result, dict):
        data = result.get("data", [])
        if data:
            image_item = data[0]
            b64 = image_item.get("b64_json")
            if b64:
                return f"data:image/png;base64,{b64}"
            url = image_item.get("url")
            return str(url) if url else None
    return None


class XinferenceImageModel(BaseImageModel):
    """
    Xinference image generation/editing client using the xinference-client SDK.
    Supports text-to-image generation and image editing capabilities.
    """

    def __init__(
        self,
        model_name: str = "stable-diffusion-2-1",
        model_uid: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 3600.0,
        abilities: Optional[List[str]] = None,
        model_id: str = "",
    ):
        """
        Initialize Xinference image model client.

        Args:
            model_name: Name of the model (e.g., "stable-diffusion-2-1")
            model_uid: Unique model UID in Xinference (if model is already launched)
            base_url: Xinference server base URL (e.g., "http://localhost:9997")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            abilities: List of model abilities (generate, edit, etc.)
            model_id: Configured model id used for usage attribution
        """
        self.model_name = model_name
        # See OpenAIImageModel: the recording layer lives on the provider, so
        # the configured id has to reach it here rather than on the wrapper.
        self.model_id = model_id
        self._model_uid = model_uid or model_name
        self.base_url = (base_url or "http://localhost:9997").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # Use explicitly configured abilities
        if abilities:
            self._abilities = abilities
        else:
            self._abilities = ["generate"]

        # Initialize the Xinference client (lazy initialization)
        self._client: Optional[XinferenceClient] = None
        self._model_handle: Optional[Any] = None

    @property
    def abilities(self) -> List[str]:
        """Get the list of abilities supported by this image model implementation."""
        return self._abilities

    def _ensure_client(self) -> None:
        """Ensure the Xinference client and model handle are initialized."""
        if self._client is None:
            self._client = XinferenceClient(
                base_url=self.base_url, api_key=self.api_key
            )

        if self._model_handle is None:
            # Get the model handle (assumes model is already launched on the server)
            self._model_handle = self._client.get_model(self._model_uid)

    def _normalize_size(self, size: str) -> str:
        """
        Normalize size format to Xinference format.
        Xinference uses "width*height" format (e.g., "1024*1024").

        Args:
            size: Size string in any format

        Returns:
            Normalized size string
        """
        # Replace "x" with "*" if present
        if "x" in size.lower():
            return size.lower().replace("x", "*")
        return size

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
        Generate an image from a text prompt.

        Args:
            prompt: Text prompt for image generation
            size: Image size in format "width*height" (e.g., "1024*1024")
            negative_prompt: Negative prompt for image generation
            resolution: Alternative size specification (e.g., "1920x1080")
            width: Image width in pixels
            height: Image height in pixels
            aspect_ratio: Aspect ratio (e.g., "3:2", "16:9")
            **kwargs: Additional parameters specific to the model
                      - n: Number of images to generate (default: 1)
                      - response_format: Format of response ("url" or "b64_json")

        Returns:
            dict with image generation result containing:
            - image_url: URL of the generated image (or base64 data URL)
            - usage: Image generation usage statistics
            - request_id: Request identifier

        Raises:
            RuntimeError: If the model doesn't support generation or API call fails
        """
        if not self.has_ability("generate"):
            raise RuntimeError("This model doesn't support image generation")

        # Handle alternative size parameters
        # Xinference uses "width*height" format (e.g., "1024*1024")
        # Priority: resolution > width+height > size
        # Note: aspect_ratio is not directly supported, use size instead
        if aspect_ratio:
            # Xinference doesn't support aspect_ratio parameter directly
            # Log a warning but continue with the base size
            logger.warning(
                f"aspect_ratio parameter '{aspect_ratio}' is not directly supported by Xinference API, using size '{size}' instead"
            )
        elif resolution:
            # resolution format: "1920x1080" -> "1920*1080"
            size = resolution.replace("x", "*")
        elif width and height:
            # width + height format: convert to "W*H" format
            size = f"{width}*{height}"

        self._ensure_client()
        assert self._model_handle is not None

        # Normalize size format
        normalized_size = self._normalize_size(size)

        # Extract parameters
        n = kwargs.pop("n", 1)
        response_format = kwargs.pop("response_format", "url")

        try:
            # Call text_to_image
            result = self._model_handle.text_to_image(
                prompt=prompt,
                n=n,
                size=normalized_size,
                response_format=response_format,
                **kwargs,
            )

            # Process result
            # Metered before the body is walked, matching gemini/dashscope. The
            # xinference client returns `response.json()` verbatim -- raw server
            # JSON despite its type annotation -- so a malformed but billed 200
            # raises while walking it. Recording afterwards lost the row for a
            # call that was charged, and the plain RuntimeError the blanket
            # handler produced was retryable, so every attempt was billed and
            # none recorded.
            out = {
                "image_url": None,
                "usage": _xinference_usage(result),
                "request_id": getattr(result, "id", None),
            }
            record_image_usage(
                out,
                model_name=self.model_name,
                model_id=self.model_id,
                call_type=MediaCallType.GENERATE_IMAGE,
                image_count=n,
                resolution=str(normalized_size or ""),
            )

            try:
                out["image_url"] = _xinference_image_url(result)
            except (TypeError, AttributeError, KeyError, IndexError) as e:
                # Classified positionally, as in the other providers: a failure
                # walking an already-metered body is an invalid response, and
                # retrying it only buys another charge.
                raise invalid_response_from(e, "Invalid response format") from e
            return out

        except InvalidImageResponseError:
            # Re-raised unchanged: the handler below would wrap it in a plain
            # RuntimeError and erase the non-retryable classification,
            # re-billing an already-billed and already-metered response.
            raise
        except Exception as e:
            logger.error(f"Xinference image generation failed: {e}")
            raise RuntimeError(f"Xinference image generation failed: {str(e)}") from e

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
            - image_url: URL of the edited image (or base64 data URL)
            - usage: Image generation usage statistics
            - request_id: Request identifier

        Raises:
            RuntimeError: If the model doesn't support editing or API call fails
        """
        if not self.has_ability("edit"):
            raise RuntimeError("This model doesn't support image editing")

        self._ensure_client()
        assert self._model_handle is not None

        # Normalize image_url to list
        image_inputs = image_url if isinstance(image_url, list) else [image_url]
        if not image_inputs:
            raise RuntimeError("At least one input image is required")

        # Extract parameters
        n = kwargs.pop("n", 1)
        size = self._normalize_size(kwargs.pop("size", "1024*1024"))
        response_format = kwargs.pop("response_format", "url")

        try:
            # For image editing, Xinference may have different methods
            # Try image_to_image or inpainting depending on the model
            if hasattr(self._model_handle, "image_to_image"):
                # image_to_image typically takes: image, prompt, size, negative_prompt, n
                result = self._model_handle.image_to_image(
                    image=image_inputs[0],  # Use first image as base
                    prompt=prompt,
                    size=size,
                    negative_prompt=negative_prompt,
                    n=n,
                    response_format=response_format,
                    **kwargs,
                )
            elif hasattr(self._model_handle, "inpainting"):
                # Use inpainting for edits. n/size are forwarded so the billed
                # image_count below matches what was actually requested —
                # omitting them made the provider default to one image while
                # usage was still recorded as n.
                result = self._model_handle.inpainting(
                    image=image_inputs[0],
                    prompt=prompt,
                    size=size,
                    negative_prompt=negative_prompt,
                    n=n,
                    response_format=response_format,
                    **kwargs,
                )
            else:
                raise RuntimeError(
                    "Image editing is not supported by this Xinference model"
                )

            # Process result
            # See generate_image: metered before the body walk, and body-walk
            # failures classified as invalid responses rather than retried.
            out = {
                "image_url": None,
                "usage": _xinference_usage(result),
                "request_id": getattr(result, "id", None),
            }
            record_image_usage(
                out,
                model_name=self.model_name,
                model_id=self.model_id,
                call_type=MediaCallType.EDIT_IMAGE,
                image_count=n,
                resolution=str(size or ""),
            )

            try:
                out["image_url"] = _xinference_image_url(result)
            except (TypeError, AttributeError, KeyError, IndexError) as e:
                raise invalid_response_from(e, "Invalid response format") from e
            return out

        except InvalidImageResponseError:
            # Re-raised unchanged: the handler below would wrap it in a plain
            # RuntimeError and erase the non-retryable classification,
            # re-billing an already-billed and already-metered response.
            raise
        except Exception as e:
            logger.error(f"Xinference image editing failed: {e}")
            raise RuntimeError(f"Xinference image editing failed: {str(e)}") from e

    async def close(self) -> None:
        """Close the Xinference client and cleanup resources."""
        if self._model_handle is not None:
            try:
                self._model_handle.close()
            except Exception:
                pass
            self._model_handle = None

        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    async def __aenter__(self) -> "XinferenceImageModel":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @staticmethod
    def list_available_models(
        base_url: str, api_key: Optional[str] = None
    ) -> List[dict[str, Any]]:
        """Fetch available image models from Xinference server.

        Args:
            base_url: Xinference server base URL
            api_key: Optional API key for authentication

        Returns:
            List of available image models with their information

        Example:
            >>> models = XinferenceImageModel.list_available_models(
            ...     base_url="http://localhost:9997"
            ... )
        """
        client = XinferenceClient(base_url=base_url, api_key=api_key)

        try:
            # Get list of running models
            # list_models returns Dict[str, Dict[str, Any]] where key is model_uid
            models_dict = client.list_models()

            # Filter for image models
            result = []
            for model_uid, model_info in models_dict.items():
                model_type = model_info.get("model_type", "")
                model_ability = model_info.get("model_ability", [])

                # Check if it's an image model
                if "image" in model_type or any(
                    "image" in str(ability).lower() for ability in model_ability
                ):
                    result.append(
                        {
                            "id": model_info.get("model_name", model_uid),
                            "model_uid": model_uid,
                            "model_type": model_type,
                            "model_ability": model_ability,
                            "description": model_info.get("model_description", ""),
                        }
                    )

            return result

        except Exception as e:
            logger.error(f"Failed to fetch image models from Xinference: {e}")
            return []

        finally:
            client.close()
