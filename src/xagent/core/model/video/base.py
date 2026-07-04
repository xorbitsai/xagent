from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseVideoModel(ABC):
    """Abstract base class for video generation models."""

    @property
    @abstractmethod
    def abilities(self) -> List[str]:
        """Return abilities supported by this video model."""
        pass

    def has_ability(self, ability: str) -> bool:
        return ability in self.abilities

    @abstractmethod
    async def create_video_task(
        self,
        content: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create an asynchronous video generation task."""
        pass

    @abstractmethod
    async def get_video_task(self, task_id: str) -> dict[str, Any]:
        """Retrieve a video generation task."""
        pass

    @abstractmethod
    async def generate_video(
        self,
        prompt: str = "",
        content: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create and optionally wait for a generated video."""
        pass
