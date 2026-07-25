from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api import computer as computer_api
from xagent.web.auth_dependencies import get_current_user


class _StatusRegistry:
    def __init__(self, status: dict) -> None:
        self._status = status

    async def status(self, _user_id: int) -> dict:
        return dict(self._status)


def test_computer_readiness_returns_both_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        computer_api,
        "get_browser_relay_registry",
        lambda: _StatusRegistry(
            {
                "connected": True,
                "attached": True,
                "client_name": "Chrome",
                "title": "Inbox",
            }
        ),
    )
    monkeypatch.setattr(
        computer_api,
        "get_desktop_relay_registry",
        lambda: _StatusRegistry(
            {
                "connected": True,
                "attached": True,
                "client_name": "Work Mac",
                "application": "Editor",
                "permissions": {
                    "screen_recording": True,
                    "accessibility": False,
                },
            }
        ),
    )
    app = FastAPI()
    app.include_router(computer_api.computer_router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=12)

    with TestClient(app) as client:
        response = client.get("/api/computer/readiness")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    targets = response.json()["targets"]
    assert targets["extension_relay"]["ready"] is True
    assert targets["extension_relay"]["title"] == "Inbox"
    assert targets["desktop_relay"]["ready"] is False
    assert [issue["code"] for issue in targets["desktop_relay"]["issues"]] == [
        "accessibility_permission_missing"
    ]
