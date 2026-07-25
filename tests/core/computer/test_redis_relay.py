from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
import redis

from xagent.core.computer.redis_relay import RedisBrowserRelayRegistry
from xagent.core.computer.relay import (
    BROWSER_RELAY_PROTOCOL_VERSION,
    BrowserRelayAuthenticationError,
    BrowserRelayConnection,
    BrowserRelayHello,
    BrowserRelayInUseError,
    BrowserRelayResponse,
    BrowserRelayStatusMessage,
    BrowserRelayUnavailableError,
)


@pytest.fixture(scope="module")
def relay_redis_url() -> Iterator[str]:
    url = os.getenv("XAGENT_TEST_REDIS_URL", "redis://127.0.0.1:6379/0")
    client = redis.Redis.from_url(
        url,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
    )
    try:
        client.ping()
    except redis.RedisError:
        client.close()
        pytest.skip(
            "Redis is required for distributed relay tests; configure "
            "XAGENT_TEST_REDIS_URL to enable them"
        )
    try:
        yield url
    finally:
        client.close()


@pytest.fixture(scope="module")
def relay_namespace_prefix(relay_redis_url: str) -> Iterator[str]:
    prefix = f"xagent:test:browser-relay:{uuid4().hex}"
    yield prefix
    client = redis.Redis.from_url(relay_redis_url)
    keys = list(client.scan_iter(f"{prefix}:*"))
    if keys:
        client.delete(*keys)
    client.close()


@pytest.mark.asyncio
async def test_pairing_and_revocation_work_across_registry_instances(
    relay_redis_url: str,
    relay_namespace_prefix: str,
) -> None:
    namespace = f"{relay_namespace_prefix}:{uuid4().hex}"
    first = RedisBrowserRelayRegistry(relay_redis_url, namespace=namespace)
    second = RedisBrowserRelayRegistry(relay_redis_url, namespace=namespace)
    try:
        pairing = await first.create_pairing(7)
        authentication = await second.authenticate(
            BrowserRelayHello(
                type="hello",
                protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
                client_id="chrome-1",
                client_name="Chrome",
                pairing_token=pairing.pairing_token,
            )
        )

        assert authentication.user_id == 7
        assert authentication.session_token
        with pytest.raises(BrowserRelayAuthenticationError, match="already used"):
            await first.authenticate(
                BrowserRelayHello(
                    type="hello",
                    protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
                    client_id="chrome-1",
                    pairing_token=pairing.pairing_token,
                )
            )

        replacement_pairing = await first.create_pairing(7)
        replacement = await second.authenticate(
            BrowserRelayHello(
                type="hello",
                protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
                client_id="chrome-2",
                pairing_token=replacement_pairing.pairing_token,
            )
        )
        assert replacement.session_token
        with pytest.raises(BrowserRelayAuthenticationError, match="expired"):
            await first.authenticate(
                BrowserRelayHello(
                    type="hello",
                    protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
                    client_id="chrome-1",
                    session_token=authentication.session_token,
                )
            )

        await first.revoke_user(7)
        with pytest.raises(BrowserRelayAuthenticationError, match="expired"):
            await second.authenticate(
                BrowserRelayHello(
                    type="hello",
                    protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
                    client_id="chrome-2",
                    session_token=replacement.session_token,
                )
            )

        async def send(_message: dict) -> None:
            return None

        connection = BrowserRelayConnection(
            user_id=7,
            client_id="chrome-2",
            client_name="Chrome",
            send=send,
            authorization_id=replacement.session_id,
        )
        with pytest.raises(BrowserRelayAuthenticationError, match="revoked"):
            await second.register(connection)
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_command_and_task_claim_route_across_registry_instances(
    relay_redis_url: str,
    relay_namespace_prefix: str,
) -> None:
    namespace = f"{relay_namespace_prefix}:{uuid4().hex}"
    gateway = RedisBrowserRelayRegistry(relay_redis_url, namespace=namespace)
    worker_a = RedisBrowserRelayRegistry(relay_redis_url, namespace=namespace)
    worker_b = RedisBrowserRelayRegistry(relay_redis_url, namespace=namespace)
    connection: BrowserRelayConnection

    async def send(message: dict) -> None:
        if message["command"] == "hang":
            return
        await connection.resolve(
            BrowserRelayResponse(
                type="response",
                protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
                request_id=message["request_id"],
                success=True,
                result={
                    "observation": {
                        "transient_marker": "browser-content-must-not-persist"
                    }
                },
            )
        )

    connection = BrowserRelayConnection(
        user_id=11,
        client_id="chrome-11",
        client_name="Chrome",
        send=send,
    )
    try:
        await gateway.register(connection)
        await gateway.update_connection_status(
            connection,
            BrowserRelayStatusMessage(
                type="status",
                protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
                attached=True,
                tab_id=3,
                title="Shared relay",
                url="https://example.com",
            ),
        )

        assert (await worker_a.status(11))["attached"] is True
        proxy = await worker_a.acquire(user_id=11, owner_task_id="task-a")
        with pytest.raises(BrowserRelayInUseError, match="another task"):
            await worker_b.acquire(user_id=11, owner_task_id="task-b")

        result = await proxy.request("observe", {"frame_id": "frame-1"})

        assert result == {
            "observation": {"transient_marker": "browser-content-must-not-persist"}
        }
        state_client = redis.Redis.from_url(
            relay_redis_url,
            decode_responses=True,
        )
        try:
            stored_state: list[object] = []
            for key in state_client.scan_iter(f"{namespace}:*"):
                key_type = state_client.type(key)
                if key_type == "string":
                    stored_state.append(state_client.get(key))
                elif key_type == "set":
                    stored_state.append(state_client.smembers(key))
            assert "browser-content-must-not-persist" not in repr(stored_state)
            assert "https://example.com" not in repr(stored_state)
            assert "Shared relay" not in repr(stored_state)
        finally:
            state_client.close()
        await worker_a.release(user_id=11, owner_task_id="task-a")
        worker_b_proxy = await worker_b.acquire(
            user_id=11,
            owner_task_id="task-b",
        )
        with pytest.raises(BrowserRelayInUseError, match="no longer controlled"):
            await proxy.request("observe", {"frame_id": "stale"})
        with pytest.raises(BrowserRelayUnavailableError, match="in time"):
            await worker_b_proxy.request("hang", {}, timeout=0.1)
    finally:
        await gateway.unregister(connection)
        await gateway.aclose()
        await worker_a.aclose()
        await worker_b.aclose()


@pytest.mark.asyncio
async def test_new_gateway_connection_replaces_old_gateway(
    relay_redis_url: str,
    relay_namespace_prefix: str,
) -> None:
    namespace = f"{relay_namespace_prefix}:{uuid4().hex}"
    first = RedisBrowserRelayRegistry(relay_redis_url, namespace=namespace)
    second = RedisBrowserRelayRegistry(relay_redis_url, namespace=namespace)

    async def send(_message: dict) -> None:
        return None

    old_connection = BrowserRelayConnection(
        user_id=21,
        client_id="old",
        client_name="Old Chrome",
        send=send,
    )
    new_connection = BrowserRelayConnection(
        user_id=21,
        client_id="new",
        client_name="New Chrome",
        send=send,
    )
    try:
        await first.register(old_connection)
        await second.register(new_connection)
        for _ in range(100):
            if old_connection.is_closed:
                break
            await asyncio.sleep(0.01)

        assert old_connection.is_closed is True
        status = await first.status(21)
        assert status["client_id"] == "new"

        await first.update_connection_status(
            old_connection,
            BrowserRelayStatusMessage(
                type="status",
                protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
                attached=True,
                tab_id=99,
            ),
        )
        assert (await second.status(21))["client_id"] == "new"
    finally:
        await first.unregister(old_connection)
        await second.unregister(new_connection)
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_same_registry_replacement_discards_old_gateway(
    relay_redis_url: str,
    relay_namespace_prefix: str,
) -> None:
    namespace = f"{relay_namespace_prefix}:{uuid4().hex}"
    registry = RedisBrowserRelayRegistry(relay_redis_url, namespace=namespace)

    async def send(_message: dict) -> None:
        return None

    old_connection = BrowserRelayConnection(
        user_id=22,
        client_id="old",
        client_name="Old Chrome",
        send=send,
    )
    new_connection = BrowserRelayConnection(
        user_id=22,
        client_id="new",
        client_name="New Chrome",
        send=send,
    )
    try:
        await registry.register(old_connection)
        old_state = registry._local_gateways[22]

        await registry.register(new_connection)

        assert old_connection.is_closed is True
        assert old_state.listener_task is not None
        assert old_state.listener_task.done()
        assert registry._local_gateways[22].connection is new_connection
        assert (await registry.status(22))["client_id"] == "new"
    finally:
        await registry.unregister(old_connection)
        await registry.unregister(new_connection)
        await registry.aclose()
