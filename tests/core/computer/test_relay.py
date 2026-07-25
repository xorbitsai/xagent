from __future__ import annotations

import asyncio

import pytest

from xagent.core.computer.redis_relay import RedisBrowserRelayRegistry
from xagent.core.computer.relay import (
    BROWSER_RELAY_PROTOCOL_VERSION,
    BrowserRelayAuthenticationError,
    BrowserRelayConnection,
    BrowserRelayError,
    BrowserRelayHello,
    BrowserRelayInUseError,
    BrowserRelayProtocolError,
    BrowserRelayRegistry,
    BrowserRelayResponse,
    BrowserRelayStatusMessage,
    BrowserRelayUnavailableError,
    DesktopRelayStatusMessage,
    build_computer_target_readiness,
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
async def test_connection_transport_failure_is_retriable_unavailability() -> None:
    async def send(_message: dict) -> None:
        raise ConnectionError("socket closed")

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
        )
    )

    with pytest.raises(BrowserRelayUnavailableError, match="disconnected"):
        await connection.request("observe", {"frame_id": "f1"})


@pytest.mark.asyncio
async def test_failed_response_uses_latest_status_to_classify_unavailability() -> None:
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
        )
    )
    request = asyncio.create_task(connection.request("observe", {"frame_id": "f1"}))
    await asyncio.sleep(0)

    connection.update_status(
        BrowserRelayStatusMessage(
            type="status",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            attached=False,
        )
    )
    await connection.resolve(
        BrowserRelayResponse(
            type="response",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            request_id=sent[0]["request_id"],
            success=False,
            error="No tab is attached.",
        )
    )

    with pytest.raises(BrowserRelayUnavailableError, match="No browser tab"):
        await request


@pytest.mark.asyncio
async def test_failed_response_stays_an_ordinary_error_when_target_is_ready() -> None:
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
        )
    )
    request = asyncio.create_task(connection.request("act", {}))
    await asyncio.sleep(0)
    await connection.resolve(
        BrowserRelayResponse(
            type="response",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            request_id=sent[0]["request_id"],
            success=False,
            error="The navigation policy blocked this command.",
        )
    )

    with pytest.raises(BrowserRelayError, match="navigation policy") as exc_info:
        await request
    assert type(exc_info.value) is BrowserRelayError


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


@pytest.mark.asyncio
async def test_desktop_relay_has_separate_credentials_and_target_contract() -> None:
    browser_registry = BrowserRelayRegistry()
    desktop_registry = BrowserRelayRegistry(target_kind="desktop")
    browser_pairing = await browser_registry.create_pairing(7)
    desktop_pairing = await desktop_registry.create_pairing(7)

    with pytest.raises(BrowserRelayAuthenticationError, match="invalid"):
        await desktop_registry.authenticate(
            BrowserRelayHello(
                type="hello",
                protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
                client_id="mac-1",
                pairing_token=browser_pairing.pairing_token,
            )
        )

    authentication = await desktop_registry.authenticate(
        BrowserRelayHello(
            type="hello",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            client_id="mac-1",
            pairing_token=desktop_pairing.pairing_token,
        )
    )

    async def send(_message: dict) -> None:
        return None

    browser_connection = BrowserRelayConnection(
        user_id=7,
        client_id="chrome",
        client_name="Chrome",
        send=send,
    )
    with pytest.raises(BrowserRelayProtocolError, match="cannot register"):
        await desktop_registry.register(browser_connection)

    desktop_connection = BrowserRelayConnection(
        user_id=7,
        client_id="mac-1",
        client_name="Mac",
        send=send,
        authorization_id=authentication.session_id,
        target_kind="desktop",
    )
    await desktop_registry.register(desktop_connection)
    await desktop_registry.update_connection_status(
        desktop_connection,
        DesktopRelayStatusMessage(
            type="status",
            protocol_version=BROWSER_RELAY_PROTOCOL_VERSION,
            attached=True,
            window_id=9,
            title="Document",
            application="Editor",
            permissions={"screen_recording": True, "accessibility": True},
        ),
    )

    status = await desktop_registry.status(7)
    assert status["target_kind"] == "desktop"
    assert status["window_id"] == 9
    assert status["permissions"]["accessibility"] is True


def test_readiness_normalizes_browser_and_desktop_recovery_issues() -> None:
    browser = build_computer_target_readiness(
        {"connected": True, "attached": False},
        target_kind="browser",
    )
    assert browser.runtime_kind == "extension_relay"
    assert browser.ready is False
    assert [issue.code for issue in browser.issues] == ["not_attached"]

    desktop = build_computer_target_readiness(
        {
            "connected": True,
            "attached": True,
            "permissions": {
                "screen_recording": False,
                "accessibility": True,
            },
            "paused": True,
        },
        target_kind="desktop",
    )
    assert desktop.runtime_kind == "desktop_relay"
    assert desktop.ready is False
    assert [issue.code for issue in desktop.issues] == [
        "paused",
        "screen_recording_permission_missing",
    ]

    disconnected = build_computer_target_readiness(
        {"connected": False, "attached": False},
        target_kind="desktop",
    )
    assert [issue.code for issue in disconnected.issues] == ["disconnected"]


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
