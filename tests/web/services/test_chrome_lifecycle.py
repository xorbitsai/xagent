from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.web.services.chrome_lifecycle import (
    CHROME_DELETE_CLAIM_TTL,
    CHROME_OWNER_GRACE,
    ChromeLifecycleCoordinator,
    ChromeLifecycleLease,
)
from xagent.web.services.durable_sandbox_lifecycle import LifecycleFence

NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)


def _fence(*, state: str = "ready", version: int = 2) -> LifecycleFence:
    return LifecycleFence(
        id=1,
        scope_digest="a" * 64,
        lifecycle_token="b" * 64,
        backend_lifecycle_digest="c" * 64,
        owner_token="d" * 64,
        version=version,
        state=state,
        task_id=7,
        run_id="run-1",
        lease_attempt_id="attempt-1",
        turn_digest="e" * 64,
        eligible_at=NOW,
        owner_lease_expires_at=NOW,
        delete_claim_expires_at=(NOW if state == "deleting" else None),
        retry_at=None,
        delete_attempts=(1 if state == "deleting" else 0),
    )


@pytest.mark.asyncio
async def test_register_persists_only_a_digest_of_the_raw_turn():
    service = MagicMock()
    service.register.return_value = _fence(state="registered", version=1)
    coordinator = ChromeLifecycleCoordinator(service, MagicMock(), now=lambda: NOW)

    await coordinator.register(
        scope_digest="a" * 64,
        task_id=7,
        run_id="run-1",
        lease_attempt_id="attempt-1",
        turn_id="raw-turn-secret",
    )

    request = service.register.call_args.args[0]
    assert request.turn_digest != "raw-turn-secret"
    assert len(request.turn_digest) == 64
    assert "raw-turn-secret" not in repr(request)


@pytest.mark.asyncio
async def test_owner_close_commits_tombstone_before_strict_backend_delete():
    events: list[str] = []
    service = MagicMock()
    deleting = replace(
        _fence(),
        state="deleting",
        version=3,
        owner_token="f" * 64,
        delete_claim_expires_at=NOW + CHROME_DELETE_CLAIM_TTL,
        delete_attempts=1,
    )

    def mark_deleting(*args, **kwargs):
        events.append("tombstone")
        return deleting

    service.mark_deleting.side_effect = mark_deleting
    service.settle_delete.side_effect = lambda fence: events.append("settle") or True
    backend = MagicMock()

    async def delete(_lifecycle_id):
        events.append("backend-delete")

    backend.delete_durable_sandbox_strict = AsyncMock(side_effect=delete)
    coordinator = ChromeLifecycleCoordinator(service, backend, now=lambda: NOW)
    lease = ChromeLifecycleLease(coordinator, _fence())

    await lease.delete()

    assert events == ["tombstone", "backend-delete", "settle"]
    service.mark_deleting.assert_called_once()
    backend.delete_durable_sandbox_strict.assert_awaited_once_with("c" * 64)


@pytest.mark.asyncio
async def test_transient_delete_failure_keeps_retryable_tombstone():
    service = MagicMock()
    deleting = _fence(state="deleting", version=3)
    service.backoff_delete.return_value = replace(
        deleting,
        version=4,
        retry_at=NOW + timedelta(seconds=2),
    )
    backend = MagicMock()
    backend.delete_durable_sandbox_strict = AsyncMock(
        side_effect=RuntimeError("backend unavailable")
    )
    coordinator = ChromeLifecycleCoordinator(service, backend, now=lambda: NOW)

    with pytest.raises(RuntimeError, match="backend unavailable"):
        await coordinator._delete_claim(deleting)

    service.settle_delete.assert_not_called()
    service.backoff_delete.assert_called_once()
    assert service.backoff_delete.call_args.kwargs["retry_at"] > NOW


@pytest.mark.asyncio
async def test_unknown_create_is_tombstoned_and_quarantined_without_backend_delete():
    service = MagicMock()
    deleting = replace(
        _fence(state="registered", version=1),
        state="deleting",
        version=2,
        delete_claim_expires_at=NOW + CHROME_DELETE_CLAIM_TTL,
        delete_attempts=1,
    )
    deferred = replace(
        deleting,
        version=3,
        delete_claim_expires_at=NOW,
        retry_at=NOW + CHROME_OWNER_GRACE,
    )
    service.mark_deleting.return_value = deleting
    service.backoff_delete.return_value = deferred
    backend = MagicMock()
    backend.delete_durable_sandbox_strict = AsyncMock()
    coordinator = ChromeLifecycleCoordinator(service, backend, now=lambda: NOW)
    lease = ChromeLifecycleLease(coordinator, _fence(state="registered", version=1))

    await lease.defer_unknown_create()

    assert lease.fence == deferred
    service.mark_deleting.assert_called_once()
    assert service.backoff_delete.call_args.kwargs["retry_at"] == (
        NOW + CHROME_OWNER_GRACE
    )
    backend.delete_durable_sandbox_strict.assert_not_awaited()


@pytest.mark.asyncio
async def test_renew_fails_closed_when_exact_attempt_or_owner_cas_is_lost():
    service = MagicMock()
    service.renew.return_value = None
    backend = MagicMock()
    backend.delete_durable_sandbox_strict = AsyncMock()
    coordinator = ChromeLifecycleCoordinator(service, backend, now=lambda: NOW)
    lease = ChromeLifecycleLease(coordinator, _fence())

    with pytest.raises(RuntimeError, match="ownership or task attempt was lost"):
        await lease.renew()

    backend.delete_durable_sandbox_strict.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_owner_cannot_delete_a_successor_generation():
    service = MagicMock()
    service.mark_deleting.return_value = None
    backend = MagicMock()
    backend.delete_durable_sandbox_strict = AsyncMock()
    coordinator = ChromeLifecycleCoordinator(service, backend, now=lambda: NOW)
    lease = ChromeLifecycleLease(coordinator, _fence())

    with pytest.raises(RuntimeError, match="delete CAS failed"):
        await lease.delete()

    backend.delete_durable_sandbox_strict.assert_not_awaited()


@pytest.mark.asyncio
async def test_ready_and_renew_extend_owner_beyond_backend_deadline():
    service = MagicMock()
    registered = _fence(state="registered", version=1)
    ready = replace(
        registered,
        state="ready",
        version=2,
        owner_lease_expires_at=NOW + CHROME_OWNER_GRACE,
    )
    renewed = replace(
        ready,
        version=3,
        owner_lease_expires_at=NOW + CHROME_OWNER_GRACE,
    )
    service.mark_ready.return_value = ready
    service.renew.return_value = renewed
    coordinator = ChromeLifecycleCoordinator(service, MagicMock(), now=lambda: NOW)
    lease = ChromeLifecycleLease(coordinator, registered)

    await lease.mark_ready()
    await lease.renew()

    assert lease.fence == renewed
    assert service.mark_ready.call_args.kwargs["owner_lease_expires_at"] == (
        NOW + CHROME_OWNER_GRACE
    )
    assert service.renew.call_args.kwargs["owner_lease_expires_at"] == (
        NOW + CHROME_OWNER_GRACE
    )
