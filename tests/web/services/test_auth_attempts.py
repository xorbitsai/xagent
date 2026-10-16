import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from redis import Redis
from redis.exceptions import ConnectionError

from xagent.web.services import auth_attempts


def test_production_without_shared_storage_fails_closed(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
    with pytest.raises(HTTPException) as error:
        auth_attempts.reserve_shared_attempt(
            "login", "identity", limit=5, window=900, max_keys=10
        )
    assert error.value.status_code == 503


def test_storage_outage_does_not_fall_back_to_local_memory(monkeypatch):
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://test.invalid/0")
    client = Mock()
    client.eval.side_effect = ConnectionError("private infrastructure details")
    monkeypatch.setattr(auth_attempts, "_client", lambda _: client)
    with pytest.raises(HTTPException) as error:
        auth_attempts.reserve_shared_attempt(
            "login", "identity", limit=5, window=900, max_keys=10
        )
    assert error.value.status_code == 503
    assert "private" not in error.value.detail


def test_development_without_redis_uses_the_bounded_local_fallback(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "development")
    monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
    monkeypatch.delenv("XAGENT_REQUIRE_SECURE_SECRETS", raising=False)
    assert (
        auth_attempts.reserve_shared_attempt(
            "login", "identity", limit=5, window=900, max_keys=10
        )
        is None
    )


@pytest.mark.skipif(
    not os.getenv("XAGENT_TEST_REDIS_URL"), reason="Requires test Redis"
)
def test_redis_atomic_admission_and_capacity_across_clients(monkeypatch):
    url = os.environ["XAGENT_TEST_REDIS_URL"]
    monkeypatch.setenv("XAGENT_REDIS_URL", url)
    namespace = f"test-{uuid.uuid4().hex}"
    clients = [Redis.from_url(url), Redis.from_url(url)]
    # Separate clients emulate independent API workers; never flush shared Redis.
    monkeypatch.setattr(auth_attempts, "_client", lambda _: clients[0])

    def reserve(key):
        return auth_attempts.reserve_shared_attempt(
            namespace, key, limit=5, window=30, max_keys=2
        )

    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            assert sum(pool.map(lambda _: reserve("victim"), range(24))) == 5
        monkeypatch.setattr(auth_attempts, "_client", lambda _: clients[1])
        assert not reserve("victim")
        assert reserve("other")
        assert not reserve("flood")
        assert not reserve("victim")
        auth_attempts.clear_shared_attempt(namespace, "other")
        assert reserve("replacement")
        assert not reserve("victim")
    finally:
        for key in ("victim", "other", "flood", "replacement"):
            auth_attempts.clear_shared_attempt(namespace, key)
        for client in clients:
            client.close()
