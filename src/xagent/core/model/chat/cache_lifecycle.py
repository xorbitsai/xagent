"""Best-effort, execution-scoped cache hints for capable model providers."""

import asyncio
import logging
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)
_HINT_TIMEOUT = 0.5
_HINT_COOLDOWN = 30.0
_HEARTBEAT_INTERVAL = 60.0  # Refresh well before Xinference's 300-second expiry.
_FINISH_RETRY_DELAY = 0.1
_ProviderKey = tuple[str, str]
_Notify = Callable[[dict[str, Any]], Awaitable[None]]
_background_hints: set[asyncio.Task[None]] = set()
_warned_providers: OrderedDict[_ProviderKey, None] = OrderedDict()


@dataclass
class AgentCacheSession:
    id: str = field(default_factory=lambda: uuid4().hex)
    revision: int = 0
    phase: str = "active"
    _providers: dict[_ProviderKey, _Notify] = field(default_factory=dict, repr=False)
    _finished_providers: tuple[tuple[_ProviderKey, _Notify], ...] = field(
        default=(), init=False, repr=False
    )
    _publisher: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _wake: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _retry_after: dict[_ProviderKey, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def register(self, key: _ProviderKey, notify: _Notify) -> dict[str, Any] | None:
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
        """Update local state immediately; never await HTTP on the work path."""
        if self.phase == "finished" or self.phase == phase:
            return
        self.phase = phase
        self.revision += 1
        if phase == "finished":
            self._finished_providers = tuple(self._providers.items())
            self._providers.clear()
        self._wake.set()
        if self._publisher is None and (self._providers or self._finished_providers):
            self._publisher = asyncio.create_task(self._publish())
            # Keep detached notifications alive until their bounded cleanup.
            _background_hints.add(self._publisher)
            self._publisher.add_done_callback(_background_hints.discard)

    async def _notify(
        self,
        key: _ProviderKey,
        callback: _Notify,
        payload: dict[str, Any],
        *,
        force: bool = False,
    ) -> bool:
        """Return whether a transient failure is eligible for a terminal retry."""
        now = asyncio.get_running_loop().time()
        if not force and self._retry_after.get(key, 0) > now:
            return False
        try:
            await asyncio.wait_for(callback(payload), _HINT_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - hints must not fail execution
            self._retry_after[key] = now + _HINT_COOLDOWN
            status = getattr(exc, "status", None)
            if key not in _warned_providers:
                _warned_providers[key] = None
                if len(_warned_providers) > 256:
                    _warned_providers.popitem(last=False)
                # Do not log URLs, payloads, credentials or exception messages.
                logger.warning(
                    "Agent cache hints unavailable (HTTP status=%s, error=%s); "
                    "execution continues",
                    status if status is not None else "unavailable",
                    type(exc).__name__,
                )
            return not isinstance(status, int) or status in (408, 429) or status >= 500
        self._retry_after.pop(key, None)
        return False

    async def _publish(self) -> None:
        """Coalesce transitions into one sender and refresh long tool waits."""
        try:
            while True:
                self._wake.clear()
                payload = {
                    "id": self.id,
                    "revision": self.revision,
                    "phase": self.phase,
                }
                finished = self.phase == "finished"
                callbacks = (
                    self._finished_providers
                    if finished
                    else tuple(self._providers.items())
                )
                retries = await asyncio.gather(
                    *(
                        self._notify(key, callback, payload, force=finished)
                        for key, callback in callbacks
                    )
                )
                if finished:
                    retry_callbacks = [
                        item for item, retry in zip(callbacks, retries) if retry
                    ]
                    if retry_callbacks:
                        await asyncio.sleep(_FINISH_RETRY_DELAY)
                        await asyncio.gather(
                            *(
                                self._notify(key, callback, payload, force=True)
                                for key, callback in retry_callbacks
                            )
                        )
                    return
                if self._wake.is_set():
                    continue
                if self.phase != "waiting":
                    return
                try:
                    await asyncio.wait_for(self._wake.wait(), _HEARTBEAT_INTERVAL)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._publisher = None
            if self.phase == "finished":
                self._finished_providers = ()
                self._retry_after.clear()


agent_cache_session: ContextVar[AgentCacheSession | None] = ContextVar(
    "agent_cache_session", default=None
)
