"""Optional short-TTL cache for high-frequency web/API read paths."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from importlib import import_module
from typing import Any, Protocol

from ...config import (
    get_hot_path_cache_enabled,
    get_hot_path_cache_ttl_seconds,
    get_hot_path_task_cache_ttl_seconds,
    get_redis_url,
)

logger = logging.getLogger(__name__)

_KEY_PREFIX = "xagent:hot:"


class CacheBackend(Protocol):
    def get_json(self, key: str) -> Any | None: ...

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None: ...

    def delete(self, *keys: str) -> None: ...

    def delete_prefix(self, prefix: str) -> None: ...


class NoOpCache:
    def get_json(self, key: str) -> Any | None:
        return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        return None

    def delete(self, *keys: str) -> None:
        return None

    def delete_prefix(self, prefix: str) -> None:
        return None


class InMemoryTTLCache:
    """Small process-local cache used for tests and explicit local fallback."""

    def __init__(self, maxsize: int = 4096) -> None:
        self._maxsize = maxsize
        self._items: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()

    def get_json(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            value, expires_at = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._items[key] = (value, time.time() + ttl_seconds)
            self._items.move_to_end(key)
            while len(self._items) > self._maxsize:
                self._items.popitem(last=False)

    def delete(self, *keys: str) -> None:
        with self._lock:
            for key in keys:
                self._items.pop(key, None)

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in list(self._items):
                if key.startswith(prefix):
                    self._items.pop(key, None)


class RedisJsonCache:
    def __init__(self, redis_url: str) -> None:
        redis = import_module("redis")

        self._client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
            health_check_interval=30,
        )

    def get_json(self, key: str) -> Any | None:
        try:
            raw = self._client.get(_redis_key(key))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hot-path cache read failed for %s: %s", key, exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self.delete(key)
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        try:
            self._client.setex(_redis_key(key), ttl_seconds, json.dumps(value))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hot-path cache write failed for %s: %s", key, exc)

    def delete(self, *keys: str) -> None:
        if not keys:
            return
        try:
            self._client.delete(*[_redis_key(key) for key in keys])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hot-path cache delete failed for %s: %s", keys, exc)

    def delete_prefix(self, prefix: str) -> None:
        try:
            namespaced = _redis_key(prefix)
            batch = []
            for key in self._client.scan_iter(f"{namespaced}*"):
                batch.append(key)
                if len(batch) >= 500:
                    self._client.delete(*batch)
                    batch = []
            if batch:
                self._client.delete(*batch)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hot-path cache prefix delete failed for %s: %s", prefix, exc)


_backend: CacheBackend | None = None
_backend_lock = threading.Lock()


def _redis_key(key: str) -> str:
    return f"{_KEY_PREFIX}{key}"


def cache_version_token(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def get_cache_backend() -> CacheBackend:
    global _backend
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is not None:
            return _backend
        if not get_hot_path_cache_enabled():
            _backend = NoOpCache()
            return _backend
        redis_url = get_redis_url()
        if not redis_url:
            _backend = NoOpCache()
            return _backend
        try:
            _backend = RedisJsonCache(redis_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hot-path Redis cache disabled: %s", exc)
            _backend = NoOpCache()
        return _backend


def set_cache_backend_for_testing(backend: CacheBackend | None) -> None:
    global _backend
    with _backend_lock:
        _backend = backend


def cache_get(key: str) -> Any | None:
    return get_cache_backend().get_json(key)


def cache_set(key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
    get_cache_backend().set_json(
        key,
        value,
        ttl_seconds if ttl_seconds is not None else get_hot_path_cache_ttl_seconds(),
    )


def cache_delete(*keys: str) -> None:
    get_cache_backend().delete(*keys)


def cache_delete_prefix(prefix: str) -> None:
    get_cache_backend().delete_prefix(prefix)


def task_snapshot_key(task_id: int) -> str:
    return f"task:snapshot:{task_id}"


def task_steps_key(task_id: int) -> str:
    return f"task:steps:{task_id}"


def web_task_detail_key(task_id: int) -> str:
    return f"task:web:detail:{task_id}"


def web_task_status_key(task_id: int) -> str:
    return f"task:web:status:{task_id}"


def web_task_history_key(task_id: int) -> str:
    return f"task:web:history:{task_id}"


def agent_list_key(
    user_id: int, team_id: int | None = None, is_team_admin: bool = False
) -> str:
    if team_id is not None:
        # user_id is part of the key: owned_agent_clause still matches a user's
        # own legacy (team_id IS NULL) agents, so the result set is user-specific
        # and must never be served from a key shared across team members.
        return f"agent:list:team:{team_id}:{user_id}:{'a' if is_team_admin else 'm'}"
    return f"agent:list:{user_id}"


def agent_detail_key(
    user_id: int,
    agent_id: int,
    team_id: int | None = None,
    is_team_admin: bool = False,
) -> str:
    if team_id is not None:
        # user_id in the key: see agent_list_key -- a user's own legacy
        # (team_id IS NULL) agents are visible only to them, so the detail
        # cache must be per-user even within a team.
        return (
            f"agent:detail:team:{team_id}:{user_id}:"
            f"{'a' if is_team_admin else 'm'}:{agent_id}"
        )
    return f"agent:detail:{user_id}:{agent_id}"


def model_list_key(
    user_id: int,
    *,
    skip: int,
    limit: int,
    model_provider: str | None,
    category: str | None,
    is_active: bool | None,
) -> str:
    provider = model_provider or "-"
    cat = category or "-"
    active = "none" if is_active is None else str(is_active).lower()
    return f"model:list:{user_id}:{skip}:{limit}:{provider}:{cat}:{active}"


def user_default_models_key(user_id: int) -> str:
    return f"model:defaults:{user_id}"


def user_default_model_key(user_id: int, config_type: str) -> str:
    return f"model:default:{user_id}:{config_type}"


def default_model_key(user_id: int, config_type: str) -> str:
    return f"model:default-view:{user_id}:{config_type}"


def invalidate_task_cache(task_id: int) -> None:
    cache_delete(
        task_snapshot_key(task_id),
        task_steps_key(task_id),
        web_task_detail_key(task_id),
        web_task_status_key(task_id),
        web_task_history_key(task_id),
    )


def invalidate_agent_cache(
    user_id: int, agent_id: int | None = None, team_id: int | None = None
) -> None:
    if team_id is not None:
        # Team keys are per-member (see agent_list_key): a write by one member
        # changes what every teammate may see (e.g. flipping visibility to
        # admins-only), so nuke the whole team namespace rather than just the
        # caller's variants. ponytail: prefix-nuke over-invalidates other agents'
        # detail entries; enumerate members if write churn ever becomes hot.
        cache_delete_prefix(f"agent:list:team:{team_id}:")
        if agent_id is not None:
            cache_delete_prefix(f"agent:detail:team:{team_id}:")
        return
    cache_delete(agent_list_key(user_id))
    if agent_id is not None:
        cache_delete(agent_detail_key(user_id, agent_id))


def invalidate_model_cache(user_id: int | None = None) -> None:
    if user_id is None:
        cache_delete_prefix("model:")
        return
    cache_delete_prefix(f"model:list:{user_id}:")
    cache_delete(user_default_models_key(user_id))
    cache_delete_prefix(f"model:default:{user_id}:")
    cache_delete_prefix(f"model:default-view:{user_id}:")


def task_cache_ttl_seconds() -> int:
    return get_hot_path_task_cache_ttl_seconds()
