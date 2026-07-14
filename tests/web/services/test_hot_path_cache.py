from __future__ import annotations

from xagent.web.services import hot_path_cache
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    RedisJsonCache,
    agent_detail_key,
    agent_list_key,
    cache_delete_prefix,
    cache_get,
    cache_set,
    invalidate_agent_cache,
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


def test_team_agent_write_invalidates_every_member_cache() -> None:
    # Team keys are per-member; a write by member A must also drop member B's
    # cached list/detail so a visibility change can't linger.
    set_cache_backend_for_testing(InMemoryTTLCache())
    team_id = 100
    a_list = agent_list_key(1, team_id, False)
    b_list = agent_list_key(2, team_id, False)
    b_detail = agent_detail_key(2, 5, team_id, False)
    cache_set(a_list, {"hit": True}, ttl_seconds=30)
    cache_set(b_list, {"hit": True}, ttl_seconds=30)
    cache_set(b_detail, {"hit": True}, ttl_seconds=30)

    invalidate_agent_cache(1, agent_id=5, team_id=team_id)

    assert cache_get(a_list) is None
    assert cache_get(b_list) is None
    assert cache_get(b_detail) is None


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
