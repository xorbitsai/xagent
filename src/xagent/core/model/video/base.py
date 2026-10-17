from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BilledVideoTaskError(RuntimeError):
    """A failure after the provider accepted a billable video generation.

    Raised once the provider holds a task it may charge for -- an Ark task id
    exists, or a 200 response body has started to be read -- and that task's
    usage row is already written. The retry wrapper must not retry it: a retry
    submits a *new* generation while the first may still finish and be billed,
    so each attempt multiplies the cost for an outcome the caller cannot use.

    Subclasses RuntimeError so callers catching the provider's documented
    ``RuntimeError`` are unaffected; the distinct type exists for the retry
    predicate.
    """


class BilledVideoTaskTimeout(BilledVideoTaskError, TimeoutError):
    """A billable video task outlived the wait deadline.

    Still a ``TimeoutError`` for callers that handle timeouts; still never
    retried, because the timed-out task keeps running and may be billed.
    """


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
