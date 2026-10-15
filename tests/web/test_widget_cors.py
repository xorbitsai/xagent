from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from xagent.web.app import PublicWidgetCORSMiddleware


def _client() -> TestClient:
    application = FastAPI()

    @application.post("/api/widget/embed-ticket")
    async def widget_ticket() -> dict[str, bool]:
        return {"ok": True}

    @application.post("/api/private")
    async def private_api() -> dict[str, bool]:
        return {"ok": True}

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["https://frontend.example"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(PublicWidgetCORSMiddleware)
    return TestClient(application)


def test_widget_preflight_allows_foreign_origin_without_credentials() -> None:
    response = _client().options(
        "/api/widget/embed-ticket",
        headers={
            "Origin": "https://customer.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://customer.example"
    )
    assert "access-control-allow-credentials" not in response.headers


def test_widget_response_allows_foreign_origin_without_credentials() -> None:
    response = _client().post(
        "/api/widget/embed-ticket",
        headers={"Origin": "https://customer.example"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://customer.example"
    )
    assert "access-control-allow-credentials" not in response.headers


def test_non_widget_preflight_keeps_credentialed_allowlist() -> None:
    response = _client().options(
        "/api/private",
        headers={
            "Origin": "https://customer.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400
