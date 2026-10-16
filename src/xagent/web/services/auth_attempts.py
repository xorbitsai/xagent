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

from ...config import get_environment, get_redis_url, get_require_secure_secrets

# Check every bucket before charging any. Per-IP admission bounds key churn;
# Redis expires buckets without an attacker-exhaustible global key index.
_RESERVE = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
local window = tonumber(ARGV[1])
for i, key in ipairs(KEYS) do
  redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
  if redis.call('ZCARD', key) >= tonumber(ARGV[i + 2]) then
    local first = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    return math.max(1, math.ceil((tonumber(first[2]) + window - now) / 1000))
  end
end
for _, key in ipairs(KEYS) do
  redis.call('ZADD', key, now, ARGV[2])
  redis.call('PEXPIRE', key, window)
end
return 0
"""


def _requires_shared() -> bool:
    return get_environment() != "development" or get_require_secure_secrets()


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
    if _requires_shared():
        raise HTTPException(503, "Authentication rate limiting is unavailable")
    return None


def validate_auth_attempt_storage() -> None:
    """Fail production startup before serving a nonfunctional login API."""
    if not _requires_shared():
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
    namespace: str, budgets: tuple[tuple[str, int], ...], *, window: int
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
                secrets.token_hex(16),
                *(str(limit) for _, limit in budgets),
            ),
        )
    except (RedisError, ValueError):
        if not _requires_shared():
            return None
        raise HTTPException(
            503, "Authentication rate limiting is unavailable"
        ) from None


def reserve_shared_attempt(
    namespace: str, key: str, *, limit: int, window: int, max_keys: int
) -> bool | None:
    """Compatibility wrapper; Redis no longer has a global key cap."""
    retry = reserve_shared_attempts(namespace, ((key, limit),), window=window)
    return None if retry is None else retry == 0


def clear_shared_attempt(namespace: str, key: str) -> bool:
    try:
        client = _backend()
        if client is None:
            return False
        client.delete(_key(namespace, key))
    except (RedisError, ValueError):
        if not _requires_shared():
            return False
        raise HTTPException(
            503, "Authentication rate limiting is unavailable"
        ) from None
    return True


@dataclass
class AttemptBucket:
    limit: int
    attempts: deque[float] = field(default_factory=deque)


def reserve_local_attempts(
    store: dict[str, AttemptBucket],
    lock: LockType,
    budgets: tuple[tuple[str, int], ...],
    *,
    window: int,
    max_keys: int,
    now: float | None = None,
) -> int:
    """Atomic local admission; evict unlocked buckets, never active lockouts."""
    current = time.monotonic() if now is None else now
    with lock:
        for key, bucket in list(store.items()):
            while bucket.attempts and bucket.attempts[0] <= current - window:
                bucket.attempts.popleft()
            if not bucket.attempts:
                store.pop(key)
        for key, limit in budgets:
            existing = store.get(key)
            if existing and len(existing.attempts) >= limit:
                return max(1, math.ceil(existing.attempts[0] + window - current))
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
        for key, limit in budgets:
            store.setdefault(key, AttemptBucket(limit)).attempts.append(current)
        return 0
