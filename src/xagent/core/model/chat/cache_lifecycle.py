"""Best-effort, execution-scoped cache hints for capable model providers."""

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)
_HINT_TIMEOUT = 0.5


@dataclass
class AgentCacheSession:
    id: str = field(default_factory=lambda: uuid4().hex)
    revision: int = 0
    phase: str = "active"
    _providers: dict[tuple[str, str], Callable[[dict[str, Any]], Awaitable[None]]] = (
        field(default_factory=dict, repr=False)
    )

    def register(
        self,
        key: tuple[str, str],
        notify: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> dict[str, Any] | None:
        if self.phase == "finished":
            return None
        self._providers[key] = notify
        return {"id": self.id, "revision": self.revision}

    def create_task(self, call: Awaitable[Any]) -> asyncio.Future[Any]:
        token = agent_cache_session.set(self)
        try:
            return asyncio.ensure_future(call)
        finally:
            agent_cache_session.reset(token)

    async def transition(self, phase: str) -> None:
        if self.phase == "finished" or self.phase == phase:
            return
        self.phase = phase
        self.revision += 1
        payload = {"id": self.id, "revision": self.revision, "phase": phase}
        callbacks = tuple(self._providers.values())
        if phase == "finished":
            self._providers.clear()

        async def notify(
            callback: Callable[[dict[str, Any]], Awaitable[None]],
        ) -> None:
            try:
                await asyncio.wait_for(callback(payload), _HINT_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 - hints must not fail execution
                logger.debug("Agent cache hint skipped (%s)", type(exc).__name__)

        await asyncio.gather(*(notify(callback) for callback in callbacks))


agent_cache_session: ContextVar[AgentCacheSession | None] = ContextVar(
    "agent_cache_session", default=None
)
