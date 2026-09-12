from abc import ABC, abstractmethod
from typing import Any, List

# NOT a claim that these providers edit across their lineup -- xinference defaults
# to stable-diffusion-2-1 and raises unless the backend exposes image_to_image, and
# openai's advertised DALL-E 3 cannot serve images.edit. It is the default both web
# call sites already applied to their NULL rows, kept so this change stays about
# agreement between the two paths rather than about widening or narrowing access.
_EDIT_CAPABLE_PROVIDERS = ("openai", "xinference")

# Per provider, so a marker added for one cannot silently move the other's answer:
# "3-pro" is Gemini vocabulary and has no meaning in a dashscope name.
_NAME_MARKERS_BY_PROVIDER = {
    "dashscope": ("edit",),
    "gemini": ("edit", "3-pro"),
}


def default_image_abilities(provider: str, model_name: str) -> List[str]:
    """Abilities for an image model whose row declares none.

    The single answer for an unconfigured row, so that the two paths building a
    model from one -- get_image_model_instance and model_service.get_image_models
    -- cannot disagree about what it can do. A declared non-empty abilities list is
    authoritative and short-circuits before this function, or an operator's
    deliberate generate-only choice gets overridden.

    A marker match trusts the name over the endpoint, so a model named for editing
    but served by one that cannot edit advertises the ability and fails at call
    time. Declaring abilities explicitly overrides that.
    """
    normalized = provider.strip().lower()
    if normalized in _EDIT_CAPABLE_PROVIDERS:
        return ["generate", "edit"]
    markers = _NAME_MARKERS_BY_PROVIDER.get(normalized, ())
    lowered = model_name.lower()
    if any(marker in lowered for marker in markers):
        return ["generate", "edit"]
    return ["generate"]


class InvalidImageResponseError(RuntimeError):
    """A provider returned a billed 200 whose body carries no usable image.

    Subclasses RuntimeError so existing `except RuntimeError` callers and the
    documented `Raises: RuntimeError` contract of every provider method are
    unchanged; the distinct type exists so the retry policy can tell it apart.

    Retrying this is strictly harmful: the provider already billed the response
    and the metering row was already written (recorded before validation
    precisely because the charge is real), so each retry buys another charge and
    another billing row for a request whose outcome will not change -- a
    safety-blocked prompt is refused just as deterministically on attempt ten.
    Transport and status failures stay plain/typed errors so they remain
    retryable, where per-attempt accounting is correct because each attempt
    really was a separate billed call.
    """


def invalid_response_from(
    error: BaseException, context: str
) -> InvalidImageResponseError:
    """Reclassify a body-walking failure as an already-billed invalid response.

    Explicitly raising the typed error at every structural check is not enough:
    walking a malformed body also fails *implicitly*. ``content[0]`` on a list
    of nulls, ``candidates[0].get(...)`` on a list of strings, or a ``parts``
    entry that is not a dict raise ``TypeError``/``AttributeError``/
    ``KeyError``/``IndexError`` -- and those land in the blanket handler, get
    rewrapped as a plain ``RuntimeError``, and are retried, re-billing and
    re-recording a call whose body will be exactly as malformed next time.

    Enumerating every shape a provider could send is a losing game, so the
    classification is positional instead: anything that fails while walking a
    200 body, after usage was already recorded, is an invalid response.
    """
    return InvalidImageResponseError(f"{context}: {type(error).__name__}: {error}")


class BaseImageModel(ABC):
    """
    Abstract base class for image generation models.
    """

    @property
    @abstractmethod
    def abilities(self) -> List[str]:
        """
        Get the list of abilities supported by this image model implementation.
        Possible abilities: ["generate", "edit"]

        Returns:
            List[str]: List of supported abilities
        """
        pass

    def has_ability(self, ability: str) -> bool:
        """
        Check if this image model implementation supports a specific ability.

        Args:
            ability: The ability to check

        Returns:
            bool: True if the ability is supported, False otherwise
        """
        return ability in self.abilities

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
