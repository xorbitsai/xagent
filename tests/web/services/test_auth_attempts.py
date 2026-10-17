import asyncio
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from redis import Redis
from redis.exceptions import ConnectionError

from xagent.web.services import auth_attempts


def test_local_retry_after_is_maximum_and_refund_only_removes_own_member():
    store, lock = {}, threading.Lock()
    auth_attempts.reserve_local_attempts(
        store, lock, (("old", 1),), window=900, max_keys=10, now=0, member="failed"
    )
    auth_attempts.reserve_local_attempts(
        store,
        lock,
        (("recent", 1),),
        window=900,
        max_keys=10,
        now=800,
        member="other-failed",
    )
    assert (
        auth_attempts.reserve_local_attempts(
            store, lock, (("old", 1), ("recent", 1)), window=900, max_keys=10, now=810
        )
        == 890
    )
    auth_attempts.reserve_local_attempts(
        store, lock, (("ip", 3),), window=900, max_keys=10, now=810, member="failed"
    )
    auth_attempts.reserve_local_attempts(
        store, lock, (("ip", 3),), window=900, max_keys=10, now=810, member="successful"
    )
    auth_attempts.refund_local_attempt(store, lock, "ip", "successful")
    assert list(store["ip"].attempts) == [("failed", 810)]


def test_real_malformed_redis_url_is_sanitized(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("XAGENT_REDIS_URL", "invalid://private-host")
    auth_attempts._client.cache_clear()
    with pytest.raises(HTTPException) as error:
        auth_attempts.reserve_shared_attempts("login", (("id", 5),), window=900)
    assert error.value.status_code == 503
    assert "private" not in error.value.detail


@pytest.mark.skipif(
    not os.getenv("XAGENT_TEST_REDIS_URL"), reason="Requires test Redis"
)
def test_redis_max_retry_and_exact_member_refund(monkeypatch):
    url = os.environ["XAGENT_TEST_REDIS_URL"]
    monkeypatch.setenv("XAGENT_REDIS_URL", url)
    namespace = f"test-{uuid.uuid4().hex}"
    with Redis.from_url(url) as client:
        monkeypatch.setattr(auth_attempts, "_client", lambda _: client)
        keys = [auth_attempts._key(namespace, key) for key in ("old", "recent", "ip")]
        try:
            seconds, microseconds = client.time()
            now = seconds * 1000 + microseconds // 1000
            client.zadd(keys[0], {"failed": now - 800_000})
            client.zadd(keys[1], {"other-failed": now - 10_000})
            client.pexpire(keys[0], 900_000)
            client.pexpire(keys[1], 900_000)
            retry = auth_attempts.reserve_shared_attempts(
                namespace, (("old", 1), ("recent", 1)), window=900
            )
            assert 800 <= retry <= 890
            assert (
                auth_attempts.reserve_shared_attempts(
                    namespace, (("ip", 3),), window=900, member="failed"
                )
                == 0
            )
            assert (
                auth_attempts.reserve_shared_attempts(
                    namespace, (("ip", 3),), window=900, member="successful"
                )
                == 0
            )
            auth_attempts.refund_shared_attempt(namespace, "ip", "successful")
            assert client.zcard(keys[2]) == 1
            assert client.zscore(keys[2], "failed") is not None
        finally:
            client.delete(*keys)


def test_production_without_shared_storage_fails_closed(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
    with pytest.raises(HTTPException) as error:
        auth_attempts.reserve_shared_attempts("login", (("identity", 5),), window=900)
    assert error.value.status_code == 503


def test_storage_outage_does_not_fall_back_to_local_memory(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://test.invalid/0")
    client = Mock()
    client.eval.side_effect = ConnectionError("private infrastructure details")
    monkeypatch.setattr(auth_attempts, "_client", lambda _: client)
    with pytest.raises(HTTPException) as error:
        auth_attempts.reserve_shared_attempts("login", (("identity", 5),), window=900)
    assert error.value.status_code == 503
    assert "private" not in error.value.detail


def test_development_without_redis_uses_the_bounded_local_fallback(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "development")
    monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
    monkeypatch.delenv("XAGENT_REQUIRE_SECURE_SECRETS", raising=False)
    assert (
        auth_attempts.reserve_shared_attempts("login", (("identity", 5),), window=900)
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


def test_web_startup_checks_rate_storage_before_database(monkeypatch):
    from xagent.web import app as web_app

    database = Mock()
    monkeypatch.setattr(web_app, "validate_auth_config", Mock())
    monkeypatch.setattr(web_app, "_initialize_database_and_admit_runtime", database)
    validator = Mock(side_effect=RuntimeError("rate storage unavailable"))
    monkeypatch.setattr(auth_attempts, "validate_auth_attempt_storage", validator)
    with pytest.raises(RuntimeError, match="rate storage unavailable"):
        asyncio.run(web_app.startup_event())
    validator.assert_called_once()
    database.assert_not_called()


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
        return (
            auth_attempts.reserve_shared_attempts(namespace, ((key, 5),), window=30)
            == 0
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
