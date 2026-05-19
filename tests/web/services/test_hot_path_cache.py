from __future__ import annotations

from xagent.web.services import hot_path_cache
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    RedisJsonCache,
    cache_delete_prefix,
    cache_get,
    cache_set,
    set_cache_backend_for_testing,
)


def teardown_function() -> None:
    set_cache_backend_for_testing(None)


def test_in_memory_cache_respects_ttl(monkeypatch) -> None:
    now = 1000.0
    monkeypatch.setattr(hot_path_cache.time, "time", lambda: now)
    set_cache_backend_for_testing(InMemoryTTLCache())

    cache_set("sample", {"value": 1}, ttl_seconds=5)
    assert cache_get("sample") == {"value": 1}

    now = 1006.0
    assert cache_get("sample") is None


def test_delete_prefix_only_removes_matching_keys() -> None:
    set_cache_backend_for_testing(InMemoryTTLCache())

    cache_set("model:list:1", {"hit": True}, ttl_seconds=30)
    cache_set("model:defaults:1", {"hit": True}, ttl_seconds=30)
    cache_set("agent:list:1", {"hit": True}, ttl_seconds=30)

    cache_delete_prefix("model:")

    assert cache_get("model:list:1") is None
    assert cache_get("model:defaults:1") is None
    assert cache_get("agent:list:1") == {"hit": True}


def test_redis_delete_prefix_deletes_in_batches(monkeypatch) -> None:
    class FakeRedisClient:
        def __init__(self) -> None:
            self.deleted_batches: list[list[str]] = []

        def scan_iter(self, pattern: str):
            assert pattern == "xagent:hot:model:*"
            for index in range(1001):
                yield f"xagent:hot:model:{index}"

        def delete(self, *keys: str) -> None:
            self.deleted_batches.append(list(keys))

    fake_client = FakeRedisClient()

    class FakeRedis:
        @staticmethod
        def from_url(*args, **kwargs):
            return fake_client

    class FakeRedisModule:
        Redis = FakeRedis

    monkeypatch.setattr(
        hot_path_cache,
        "import_module",
        lambda name: FakeRedisModule,
    )

    cache = RedisJsonCache("redis://example")
    cache.delete_prefix("model:")

    assert [len(batch) for batch in fake_client.deleted_batches] == [500, 500, 1]
