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
