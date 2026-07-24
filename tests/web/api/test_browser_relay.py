from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.core.computer.relay import reset_browser_relay_registry
from xagent.web.api import browser_relay as browser_relay_api
from xagent.web.auth_dependencies import get_current_user


@pytest.fixture
def relay_client(monkeypatch):
    reset_browser_relay_registry()
    monkeypatch.setattr(browser_relay_api, "_user_exists", lambda _user_id: True)
    app = FastAPI()
    app.include_router(browser_relay_api.browser_relay_router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=12)
    with TestClient(app) as client:
        yield client
    reset_browser_relay_registry()


def test_pairing_websocket_and_reconnect_flow(relay_client: TestClient) -> None:
    pairing_response = relay_client.post("/api/browser-relay/pairings")
    assert pairing_response.status_code == 200
    pairing = pairing_response.json()
    assert pairing["websocket_url"] == "ws://testserver/ws/browser-relay"
    assert pairing_response.headers["cache-control"] == "no-store"

    with relay_client.websocket_connect("/ws/browser-relay") as websocket:
        websocket.send_json(
            {
                "type": "hello",
                "protocol_version": 1,
                "client_id": "chrome-12",
                "client_name": "Chrome",
                "pairing_token": pairing["pairing_token"],
            }
        )
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["paired"] is True
        session_token = ready["session_token"]

        websocket.send_json(
            {
                "type": "status",
                "protocol_version": 1,
                "attached": True,
                "tab_id": 7,
                "title": "Signed in",
                "url": "https://example.com",
            }
        )
        websocket.send_json({"type": "ping", "protocol_version": 1})
        assert websocket.receive_json()["type"] == "pong"

        status = relay_client.get("/api/browser-relay/status")
        assert status.json()["connected"] is True
        assert status.json()["attached"] is True
        assert status.json()["title"] == "Signed in"

    with relay_client.websocket_connect("/ws/browser-relay") as websocket:
        websocket.send_json(
            {
                "type": "hello",
                "protocol_version": 1,
                "client_id": "chrome-12",
                "client_name": "Chrome",
                "session_token": session_token,
            }
        )
        ready = websocket.receive_json()
        assert ready == {
            "type": "ready",
            "protocol_version": 1,
            "paired": False,
        }


def test_pairing_token_cannot_be_reused(relay_client: TestClient) -> None:
    pairing = relay_client.post("/api/browser-relay/pairings").json()
    hello = {
        "type": "hello",
        "protocol_version": 1,
        "client_id": "chrome-12",
        "client_name": "Chrome",
        "pairing_token": pairing["pairing_token"],
    }
    with relay_client.websocket_connect("/ws/browser-relay") as websocket:
        websocket.send_json(hello)
        assert websocket.receive_json()["type"] == "ready"

    with relay_client.websocket_connect("/ws/browser-relay") as websocket:
        websocket.send_json(hello)
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert "already used" in error["error"]
