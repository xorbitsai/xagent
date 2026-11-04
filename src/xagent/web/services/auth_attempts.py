"""Atomic, bounded authentication budgets shared by deployed API workers."""

import math
import secrets
import time
from _thread import LockType
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import cast

from fastapi import HTTPException
from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

from ...config import get_redis_url, requires_secure_secrets

# Check every bucket before charging any. Per-IP admission bounds key churn;
# Redis expires buckets without an attacker-exhaustible global key index.
_RESERVE = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
local window = tonumber(ARGV[1])
local retry = 0
for i, key in ipairs(KEYS) do
  redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
  if redis.call('ZCARD', key) >= tonumber(ARGV[i + 2]) then
    local first = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    retry = math.max(retry, 1, math.ceil((tonumber(first[2]) + window - now) / 1000))
  end
end
if retry > 0 then return retry end
for _, key in ipairs(KEYS) do
  redis.call('ZADD', key, now, ARGV[2])
  redis.call('PEXPIRE', key, window)
end
return 0
"""


def _storage_failure() -> None:
    if requires_secure_secrets():
        raise HTTPException(
            503, "Authentication rate limiting is unavailable"
        ) from None


@lru_cache(maxsize=2)
def _client(url: str) -> Redis:
    return Redis.from_url(
        url,
        socket_connect_timeout=2,
        socket_timeout=2,
        retry=Retry(NoBackoff(), 0),
    )


def _backend() -> Redis | None:
    url = get_redis_url()
    if url:
        return _client(url)
    if requires_secure_secrets():
        raise HTTPException(503, "Authentication rate limiting is unavailable")
    return None


def validate_auth_attempt_storage() -> None:
    """Fail production startup before serving a nonfunctional login API."""
    if not requires_secure_secrets():
        return
    try:
        client = _backend()
        assert client is not None
        client.ping()
    except (HTTPException, RedisError, ValueError):
        raise RuntimeError(
            "Authentication rate limiting requires a reachable XAGENT_REDIS_URL"
        ) from None


def _key(namespace: str, key: str) -> str:
    # This client targets standalone Redis or a managed proxy, not the
    # RedisCluster discovery protocol. Hash tags still colocate script keys.
    return f"xagent:auth:{{{namespace}}}:attempt:{key}"


def reserve_shared_attempts(
    namespace: str,
    budgets: tuple[tuple[str, int], ...],
    *,
    window: int,
    member: str | None = None,
) -> int | None:
    """Return retry seconds, zero on success, None for local development."""
    try:
        client = _backend()
        if client is None:
            return None
        return cast(
            int,
            client.eval(
                _RESERVE,
                len(budgets),
                *(_key(namespace, key) for key, _ in budgets),
                str(window * 1000),
                member or secrets.token_hex(16),
                *(str(limit) for _, limit in budgets),
            ),
        )
    except (RedisError, ValueError):
        _storage_failure()
        return None


def clear_shared_attempt(namespace: str, key: str) -> bool:
    try:
        client = _backend()
        if client is None:
            return False
        client.delete(_key(namespace, key))
    except (RedisError, ValueError):
        _storage_failure()
        return False
    return True


def refund_shared_attempt(namespace: str, key: str, member: str) -> None:
    """Remove only this request's reservation, never another caller's failures."""
    try:
        client = _backend()
        if client is not None:
            client.zrem(_key(namespace, key), member)
    except (RedisError, ValueError):
        _storage_failure()


@dataclass
class AttemptBucket:
    limit: int
    attempts: deque[tuple[str, float]] = field(default_factory=deque)


def reserve_local_attempts(
    store: dict[str, AttemptBucket],
    lock: LockType,
    budgets: tuple[tuple[str, int], ...],
    *,
    window: int,
    max_keys: int,
    now: float | None = None,
    member: str | None = None,
) -> int:
    """Atomic local admission; evict unlocked buckets, never active lockouts."""
    current = time.monotonic() if now is None else now
    with lock:
        for key, bucket in list(store.items()):
            while bucket.attempts and bucket.attempts[0][1] <= current - window:
                bucket.attempts.popleft()
            if not bucket.attempts:
                store.pop(key)
        retry = 0
        for key, limit in budgets:
            existing = store.get(key)
            if existing and len(existing.attempts) >= limit:
                retry = max(
                    retry, 1, math.ceil(existing.attempts[0][1] + window - current)
                )
        if retry:
            return retry
        requested = {key for key, _ in budgets}
        missing = len(requested - store.keys())
        for key, bucket in list(store.items()):
            if len(store) + missing <= max_keys:
                break
            if key not in requested and len(bucket.attempts) < bucket.limit:
                store.pop(key)
        if len(store) + missing > max_keys:
            raise HTTPException(
                503, "Authentication rate limit capacity is unavailable"
            )
        reservation = member or secrets.token_hex(16)
        for key, limit in budgets:
            store.setdefault(key, AttemptBucket(limit)).attempts.append(
                (reservation, current)
            )
        return 0


def refund_local_attempt(
    store: dict[str, AttemptBucket], lock: LockType, key: str, member: str
) -> None:
    with lock:
        bucket = store.get(key)
        if bucket:
            bucket.attempts = deque(
                item for item in bucket.attempts if item[0] != member
            )
            if not bucket.attempts:
                store.pop(key)
