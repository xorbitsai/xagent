from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.core.computer.desktop_relay import reset_desktop_relay_registry
from xagent.web.api import desktop_relay as desktop_relay_api
from xagent.web.auth_dependencies import get_current_user


@pytest.fixture
def relay_client(monkeypatch):
    monkeypatch.setenv("XAGENT_BROWSER_RELAY_BACKEND", "memory")
    reset_desktop_relay_registry()
    monkeypatch.setattr(desktop_relay_api, "_user_exists", lambda _user_id: True)
    app = FastAPI()
    app.include_router(desktop_relay_api.desktop_relay_router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=12)
    with TestClient(app) as client:
        yield client
    reset_desktop_relay_registry()


def test_desktop_pairing_status_and_reconnect_flow(
    relay_client: TestClient,
) -> None:
    pairing_response = relay_client.post("/api/desktop-relay/pairings")
    assert pairing_response.status_code == 200
    pairing = pairing_response.json()
    assert pairing["websocket_url"] == "ws://testserver/ws/desktop-relay"
    assert pairing_response.headers["cache-control"] == "no-store"

    with relay_client.websocket_connect("/ws/desktop-relay") as websocket:
        websocket.send_json(
            {
                "type": "hello",
                "protocol_version": 1,
                "client_id": "mac-12",
                "client_name": "Work Mac",
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
                "window_id": 7,
                "title": "Signed in",
                "application": "Example",
                "bounds": {
                    "x": 10,
                    "y": 20,
                    "width": 900,
                    "height": 600,
                },
                "permissions": {
                    "screen_recording": True,
                    "accessibility": True,
                },
                "paused": True,
                "emergency_stopped": False,
            }
        )
        websocket.send_json({"type": "ping", "protocol_version": 1})
        assert websocket.receive_json()["type"] == "pong"

        status = relay_client.get("/api/desktop-relay/status").json()
        assert status["connected"] is True
        assert status["attached"] is True
        assert status["application"] == "Example"
        assert status["permissions"]["screen_recording"] is True
        assert status["paused"] is True

    with relay_client.websocket_connect("/ws/desktop-relay") as websocket:
        websocket.send_json(
            {
                "type": "hello",
                "protocol_version": 1,
                "client_id": "mac-12",
                "client_name": "Work Mac",
                "session_token": session_token,
            }
        )
        assert websocket.receive_json() == {
            "type": "ready",
            "protocol_version": 1,
            "paired": False,
        }


def test_desktop_relay_rejects_browser_status_shape(
    relay_client: TestClient,
) -> None:
    pairing = relay_client.post("/api/desktop-relay/pairings").json()
    with relay_client.websocket_connect("/ws/desktop-relay") as websocket:
        websocket.send_json(
            {
                "type": "hello",
                "protocol_version": 1,
                "client_id": "mac-12",
                "client_name": "Work Mac",
                "pairing_token": pairing["pairing_token"],
            }
        )
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json(
            {
                "type": "status",
                "protocol_version": 1,
                "attached": True,
                "tab_id": 7,
                "url": "https://example.com",
            }
        )
        error = websocket.receive_json()
        assert error["type"] == "error"
