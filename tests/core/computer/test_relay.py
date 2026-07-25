from __future__ import annotations

import asyncio

import pytest

from xagent.core.computer.redis_relay import RedisBrowserRelayRegistry
from xagent.core.computer.relay import (
    BROWSER_RELAY_PROTOCOL_VERSION,
    BrowserRelayAuthenticationError,
    BrowserRelayConnection,
    BrowserRelayHello,
    BrowserRelayInUseError,
    BrowserRelayRegistry,
    BrowserRelayResponse,
    BrowserRelayStatusMessage,
    get_browser_relay_registry,
    reset_browser_relay_registry,
)


@pytest.mark.asyncio
async def test_pairing_is_single_use_and_issues_reconnect_token() -> None:
    registry = BrowserRelayRegistry()
    pairing = await registry.create_pairing(7)
    hello = BrowserRelayHello(
        type="hello",
        protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
        client_id="chrome-1",
        client_name="My Chrome",
        pairing_token=pairing.pairing_token,
    )

    authentication = await registry.authenticate(hello)

    assert authentication.user_id == 7
    assert authentication.paired is True
    assert authentication.session_token
    assert pairing.pairing_token not in repr(registry.__dict__)
    with pytest.raises(BrowserRelayAuthenticationError, match="already used"):
        await registry.authenticate(hello)

    reconnect = await registry.authenticate(
        BrowserRelayHello(
            type="hello",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            client_id="chrome-1",
            client_name="Renamed by untrusted client",
            session_token=authentication.session_token,
        )
    )
    assert reconnect.user_id == 7
    assert reconnect.client_name == "My Chrome"
    assert reconnect.session_token is None

    replacement_pairing = await registry.create_pairing(7)
    await registry.authenticate(
        BrowserRelayHello(
            type="hello",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            client_id="chrome-2",
            pairing_token=replacement_pairing.pairing_token,
        )
    )
    with pytest.raises(BrowserRelayAuthenticationError, match="expired"):
        await registry.authenticate(
            BrowserRelayHello(
                type="hello",
                protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
                client_id="chrome-1",
                session_token=authentication.session_token,
            )
        )


@pytest.mark.asyncio
async def test_revoked_session_cannot_register_after_authentication() -> None:
    registry = BrowserRelayRegistry()
    pairing = await registry.create_pairing(7)
    authentication = await registry.authenticate(
        BrowserRelayHello(
            type="hello",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            client_id="chrome-1",
            pairing_token=pairing.pairing_token,
        )
    )

    async def send(_message: dict) -> None:
        return None

    connection = BrowserRelayConnection(
        user_id=7,
        client_id="chrome-1",
        client_name="Chrome",
        send=send,
        authorization_id=authentication.session_id,
    )
    await registry.revoke_user(7)

    with pytest.raises(BrowserRelayAuthenticationError, match="revoked"):
        await registry.register(connection)


@pytest.mark.asyncio
async def test_connection_routes_response_to_waiting_request() -> None:
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    connection = BrowserRelayConnection(
        user_id=2,
        client_id="chrome",
        client_name="Chrome",
        send=send,
    )
    connection.update_status(
        BrowserRelayStatusMessage(
            type="status",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            attached=True,
            tab_id=19,
            title="Example",
            url="https://example.com",
        )
    )

    request = asyncio.create_task(connection.request("observe", {"frame_id": "f1"}))
    await asyncio.sleep(0)
    await connection.resolve(
        BrowserRelayResponse(
            type="response",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            request_id=sent[0]["request_id"],
            success=True,
            result={"observation": {"title": "Example"}},
        )
    )

    assert await request == {"observation": {"title": "Example"}}
    assert sent[0]["command"] == "observe"


@pytest.mark.asyncio
async def test_registry_enforces_one_task_owner_per_user() -> None:
    async def send(_message: dict) -> None:
        return None

    registry = BrowserRelayRegistry()
    connection = BrowserRelayConnection(
        user_id=4,
        client_id="chrome",
        client_name="Chrome",
        send=send,
    )
    connection.attached = True
    await registry.register(connection)

    assert (await registry.acquire(user_id=4, owner_task_id="task-a")) is connection
    with pytest.raises(BrowserRelayInUseError, match="another task"):
        await registry.acquire(user_id=4, owner_task_id="task-b")

    await registry.release(user_id=4, owner_task_id="task-a")
    assert (await registry.acquire(user_id=4, owner_task_id="task-b")) is connection


def test_hello_requires_exactly_one_credential() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        BrowserRelayHello(
            type="hello",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            client_id="chrome",
        )


def test_registry_factory_selects_memory_or_redis(monkeypatch) -> None:
    try:
        monkeypatch.setenv("XAGENT_BROWSER_RELAY_BACKEND", "auto")
        monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
        reset_browser_relay_registry()
        assert isinstance(get_browser_relay_registry(), BrowserRelayRegistry)

        monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6379/0")
        reset_browser_relay_registry()
        assert isinstance(
            get_browser_relay_registry(),
            RedisBrowserRelayRegistry,
        )

        monkeypatch.setenv("XAGENT_BROWSER_RELAY_BACKEND", "redis")
        monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
        reset_browser_relay_registry()
        with pytest.raises(RuntimeError, match="XAGENT_REDIS_URL"):
            get_browser_relay_registry()
    finally:
        reset_browser_relay_registry()
