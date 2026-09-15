"""Async Chrome consumer for the durable sandbox lifecycle substrate."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Protocol, TypeVar

from ...config import (
    get_chrome_lifecycle_sweep_batch_size,
    get_chrome_lifecycle_sweep_interval_seconds,
    get_chrome_session_ttl_seconds,
)
from ...core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    CHROME_BACKEND_OPERATION_TIMEOUT_SECONDS,
)
from ..models.database import get_session_local
from .db_runtime import run_db_io_cancellation_safe
from .durable_sandbox_lifecycle import (
    DurableLifecycleConflict,
    DurableSandboxLifecycleService,
    LifecycleFence,
    RegisterLifecycle,
    digest_turn_identifier,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

# Every Chrome backend operation is bounded below this duration.  The owner
# lease is deliberately longer, so a sweeper cannot take over a generation
# while the prior owner may still be inside a supported backend call.
CHROME_OWNER_GRACE = timedelta(seconds=240)
CHROME_DELETE_CLAIM_TTL = timedelta(seconds=240)
CHROME_DELETE_RETRY_BASE = timedelta(seconds=2)
CHROME_DELETE_RETRY_MAX = timedelta(minutes=5)

if CHROME_OWNER_GRACE.total_seconds() <= CHROME_BACKEND_OPERATION_TIMEOUT_SECONDS:
    raise RuntimeError("Chrome owner grace must exceed the backend operation timeout")


class DurableChromeBackend(Protocol):
    async def delete_durable_sandbox_strict(self, lifecycle_id: str) -> None: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChromeLifecycleLease:
    """One process-local handle to an owner/version-fenced durable row."""

    def __init__(
        self,
        coordinator: ChromeLifecycleCoordinator,
        fence: LifecycleFence,
    ) -> None:
        self._coordinator = coordinator
        self._fence = fence
        self._lock = asyncio.Lock()

    @property
    def fence(self) -> LifecycleFence:
        return self._fence

    @property
    def backend_lifecycle_digest(self) -> str:
        return self._fence.backend_lifecycle_digest

    async def mark_ready(self) -> None:
        async with self._lock:
            now = self._coordinator.now()
            updated = await self._coordinator._db(
                lambda: self._coordinator.service.mark_ready(
                    self._fence,
                    now=now,
                    owner_lease_expires_at=now + CHROME_OWNER_GRACE,
                )
            )
            if updated is None:
                raise DurableLifecycleConflict("Chrome lifecycle ready CAS failed")
            self._fence = updated

    async def renew(self) -> None:
        """Revalidate the exact task attempt and extend this owner's fence."""

        async with self._lock:
            if self._fence.state != "ready":
                raise DurableLifecycleConflict("Chrome lifecycle is not ready")
            now = self._coordinator.now()
            updated = await self._coordinator._db(
                lambda: self._coordinator.service.renew(
                    self._fence,
                    now=now,
                    owner_lease_expires_at=now + CHROME_OWNER_GRACE,
                )
            )
            if updated is None:
                raise DurableLifecycleConflict(
                    "Chrome lifecycle ownership or task attempt was lost"
                )
            self._fence = updated

    async def delete(self) -> None:
        """Commit the tombstone, strictly delete, then settle or back off."""

        async with self._lock:
            await self._mark_deleting_locked()
            await self._coordinator._delete_claim(self._fence)

    async def defer_unknown_create(self) -> None:
        """Tombstone an uncertain create and quarantine it before deletion.

        A cancelled backend may have delegated blocking creation to a thread.
        That thread can publish after the cancelled coroutine has returned.  An
        immediate not-found delete would then settle the row too early, so the
        sweeper must retry only after every supported create deadline elapsed.
        """

        async with self._lock:
            await self._mark_deleting_locked()
            updated = await self._coordinator._backoff(
                self._fence,
                retry_at=self._coordinator.now() + CHROME_OWNER_GRACE,
            )
            if updated is None:
                raise DurableLifecycleConflict(
                    "Chrome lifecycle uncertain-create backoff CAS failed"
                )
            self._fence = updated

    async def _mark_deleting_locked(self) -> None:
        fence = self._fence
        if fence.state == "deleting":
            return
        now = self._coordinator.now()
        claimed = await self._coordinator._db(
            lambda: self._coordinator.service.mark_deleting(
                fence,
                now=now,
                claim_ttl=CHROME_DELETE_CLAIM_TTL,
            )
        )
        if claimed is None:
            raise DurableLifecycleConflict("Chrome lifecycle delete CAS failed")
        self._fence = claimed


class ChromeLifecycleCoordinator:
    """Commit-bounded durable lifecycle orchestration for one web worker."""

    def __init__(
        self,
        service: DurableSandboxLifecycleService,
        backend: DurableChromeBackend,
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.service = service
        self.backend = backend
        self.now = now

    @classmethod
    def production(cls, backend: DurableChromeBackend) -> ChromeLifecycleCoordinator:
        return cls(DurableSandboxLifecycleService(get_session_local()), backend)

    async def _db(self, operation: Callable[[], _T]) -> _T:
        return await run_db_io_cancellation_safe(operation)

    async def register(
        self,
        *,
        scope_digest: str,
        task_id: int,
        run_id: str,
        lease_attempt_id: str,
        turn_id: str,
    ) -> ChromeLifecycleLease:
        now = self.now()
        request = RegisterLifecycle(
            scope_digest=scope_digest,
            task_id=task_id,
            run_id=run_id,
            lease_attempt_id=lease_attempt_id,
            turn_digest=digest_turn_identifier(turn_id),
            eligible_at=now + timedelta(seconds=get_chrome_session_ttl_seconds()),
            owner_lease_expires_at=now + CHROME_OWNER_GRACE,
        )
        fence = await self._db(lambda: self.service.register(request))
        return ChromeLifecycleLease(self, fence)

    async def _delete_claim(self, fence: LifecycleFence) -> None:
        try:
            await asyncio.wait_for(
                self.backend.delete_durable_sandbox_strict(
                    fence.backend_lifecycle_digest
                ),
                timeout=CHROME_BACKEND_OPERATION_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            await self._backoff(fence)
            raise
        except Exception:
            await self._backoff(fence)
            raise

        settled = await self._db(lambda: self.service.settle_delete(fence))
        if not settled:
            raise DurableLifecycleConflict("Chrome lifecycle settle CAS failed")

    async def _backoff(
        self,
        fence: LifecycleFence,
        *,
        retry_at: datetime | None = None,
    ) -> LifecycleFence | None:
        now = self.now()
        if retry_at is None:
            exponent = max(0, min(fence.delete_attempts - 1, 8))
            delay = min(
                CHROME_DELETE_RETRY_BASE * (2**exponent),
                CHROME_DELETE_RETRY_MAX,
            )
            retry_at = now + delay
        updated = await self._db(
            lambda: self.service.backoff_delete(
                fence,
                now=now,
                retry_at=retry_at,
            )
        )
        if updated is None:
            logger.warning(
                "Chrome lifecycle delete backoff CAS lost for backend %s",
                fence.backend_lifecycle_digest,
            )
        return updated

    async def sweep_once(self) -> int:
        now = self.now()
        candidates = await self._db(
            lambda: self.service.list_reclaimable(
                now=now,
                limit=get_chrome_lifecycle_sweep_batch_size(),
            )
        )
        reclaimed = 0
        for candidate in candidates:
            try:
                claim = await self._db(lambda: self._claim_candidate(candidate))
                if claim is None:
                    continue
                await self._delete_claim(claim)
                reclaimed += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Durable Chrome recovery failed for backend %s",
                    candidate.backend_lifecycle_digest,
                    exc_info=True,
                )
        return reclaimed

    def _claim_candidate(self, candidate: LifecycleFence) -> LifecycleFence | None:
        if candidate.state == "deleting":
            return self.service.reclaim_delete(
                candidate,
                now=self.now(),
                claim_ttl=CHROME_DELETE_CLAIM_TTL,
            )
        return self.service.claim_for_delete(
            candidate,
            now=self.now(),
            claim_ttl=CHROME_DELETE_CLAIM_TTL,
        )

    async def run_sweep_loop(self) -> None:
        interval = get_chrome_lifecycle_sweep_interval_seconds()
        while True:
            await asyncio.sleep(interval)
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Durable Chrome recovery pass failed", exc_info=True)
