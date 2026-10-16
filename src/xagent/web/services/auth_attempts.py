"""Atomic authentication budgets shared by every deployed API worker."""

from functools import lru_cache

from fastapi import HTTPException
from redis import Redis
from redis.exceptions import RedisError

from ...config import get_environment, get_redis_url, get_require_secure_secrets

_RESERVE = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
local window = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[1]) then return 0 end
if not redis.call('ZSCORE', KEYS[2], KEYS[1]) and
   redis.call('ZCARD', KEYS[2]) >= tonumber(ARGV[3]) then return 0 end
redis.call('ZADD', KEYS[1], now, ARGV[4])
redis.call('PEXPIRE', KEYS[1], window)
redis.call('ZADD', KEYS[2], now + window, KEYS[1])
redis.call('PEXPIRE', KEYS[2], window)
return 1
"""

_CLEAR = """
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[2], KEYS[1])
return 1
"""


@lru_cache(maxsize=2)
def _client(url: str) -> Redis:
    return Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)


def _backend() -> Redis | None:
    url = get_redis_url()
    if url:
        return _client(url)
    if get_environment() != "development" or get_require_secure_secrets():
        raise HTTPException(503, "Authentication rate limiting is unavailable")
    return None


def _keys(namespace: str, key: str) -> tuple[str, str]:
    # The shared hash tag also permits atomic scripts with Redis Cluster.
    prefix = f"xagent:auth:{{{namespace}}}"
    return f"{prefix}:attempt:{key}", f"{prefix}:active"


def reserve_shared_attempt(
    namespace: str, key: str, *, limit: int, window: int, max_keys: int
) -> bool | None:
    """Return None only for development's explicitly local fallback."""
    import secrets

    client = _backend()
    if client is None:
        return None
    try:
        return bool(
            client.eval(
                _RESERVE,
                2,
                *_keys(namespace, key),
                str(limit),
                str(window * 1000),
                str(max_keys),
                secrets.token_hex(16),
            )
        )
    except RedisError:
        # Never fall back to an empty local bucket during a Redis outage.
        raise HTTPException(
            503, "Authentication rate limiting is unavailable"
        ) from None


def clear_shared_attempt(namespace: str, key: str) -> bool:
    client = _backend()
    if client is None:
        return False
    try:
        client.eval(_CLEAR, 2, *_keys(namespace, key))
    except RedisError:
        raise HTTPException(
            503, "Authentication rate limiting is unavailable"
        ) from None
    return True
