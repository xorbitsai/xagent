import os
import time
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
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
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


@pytest.mark.parametrize(
    "environment,raises", [("production", True), ("development", False)]
)
@pytest.mark.parametrize(
    "failure", [ConnectionError("private host"), ValueError("private URL")]
)
def test_backend_creation_errors_are_sanitized_or_local(
    monkeypatch, environment, raises, failure
):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", environment)
    monkeypatch.delenv("XAGENT_REQUIRE_SECURE_SECRETS", raising=False)
    monkeypatch.setenv("XAGENT_REDIS_URL", "invalid://private")

    def fail(_):
        raise failure

    monkeypatch.setattr(auth_attempts, "_client", fail)
    if raises:
        with pytest.raises(HTTPException) as error:
            auth_attempts.reserve_shared_attempts(
                "login", (("identity", 5),), window=900
            )
        assert error.value.status_code == 503
        assert "private" not in error.value.detail
    else:
        assert (
            auth_attempts.reserve_shared_attempts(
                "login", (("identity", 5),), window=900
            )
            is None
        )


@pytest.mark.parametrize("url", [None, "invalid://host", "redis://unavailable"])
def test_production_startup_fails_before_admission(monkeypatch, url):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    if url:
        monkeypatch.setenv("XAGENT_REDIS_URL", url)
    else:
        monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
    client = Mock()
    client.ping.side_effect = ConnectionError("private host")
    monkeypatch.setattr(auth_attempts, "_client", lambda _: client)
    with pytest.raises(RuntimeError, match="reachable XAGENT_REDIS_URL"):
        auth_attempts.validate_auth_attempt_storage()


def test_redis_client_disables_automatic_retry(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(auth_attempts.Redis, "from_url", constructor)
    auth_attempts._client.cache_clear()
    auth_attempts._client("redis://example.invalid")
    retry = constructor.call_args.kwargs["retry"]
    assert retry.get_retries() == 0
    auth_attempts._client.cache_clear()


@pytest.mark.skipif(
    not os.getenv("XAGENT_TEST_REDIS_URL"), reason="Requires test Redis"
)
def test_redis_rejection_is_atomic_and_expiry_restores_budget(monkeypatch):
    url = os.environ["XAGENT_TEST_REDIS_URL"]
    monkeypatch.setenv("XAGENT_REDIS_URL", url)
    namespace = f"test-{uuid.uuid4().hex}"
    with Redis.from_url(url) as client:
        monkeypatch.setattr(auth_attempts, "_client", lambda _: client)
        try:
            assert (
                auth_attempts.reserve_shared_attempts(
                    namespace, (("blocked", 1),), window=1
                )
                == 0
            )
            assert 0 < client.pttl(auth_attempts._key(namespace, "blocked")) <= 1000
            assert (
                auth_attempts.reserve_shared_attempts(
                    namespace, (("new-ip", 100), ("blocked", 1)), window=1
                )
                == 1
            )
            assert not client.exists(auth_attempts._key(namespace, "new-ip"))
            time.sleep(1.1)
            assert (
                auth_attempts.reserve_shared_attempts(
                    namespace, (("new-ip", 100), ("blocked", 1)), window=1
                )
                == 0
            )
            assert not client.exists(f"xagent:auth:{{{namespace}}}:active")
        finally:
            client.delete(
                *(auth_attempts._key(namespace, key) for key in ("new-ip", "blocked"))
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
        # There is no global key-cap denial for an unrelated identity.
        assert reserve("flood")
        assert not reserve("victim")
        auth_attempts.clear_shared_attempt(namespace, "other")
        assert reserve("replacement")
        assert not reserve("victim")
    finally:
        for key in ("victim", "other", "flood", "replacement"):
            auth_attempts.clear_shared_attempt(namespace, key)
        for client in clients:
            client.close()
