"""Tests for KeenableWebSearchCore. All HTTP goes through httpx.MockTransport."""

import json
import logging
from typing import Any, Callable

import httpx
import pytest

from xagent.core.tools.core import keenable_web_search as mod
from xagent.core.tools.core.keenable_web_search import KeenableWebSearchCore

API_KEY = "test-keenable-key-0000"
QUERY = "acme quarterly revenue forecast"

BODY = {
    "query": QUERY,
    "mode": "pro",
    "results": [
        {
            "title": "T1",
            "url": "https://a.example/1",
            "description": "",
            "snippet": "S1 " + "x" * 600,
        },
        {"title": "T2", "url": "https://a.example/2", "description": "D2"},
        {"title": "no url", "snippet": "dropped"},
    ],
}


@pytest.fixture
def transport(monkeypatch):
    """Route the core's httpx.AsyncClient through a MockTransport."""
    requests: list[httpx.Request] = []

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> list:
        def wrapped(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return handler(request)

        real = httpx.AsyncClient

        def factory(**kwargs: Any) -> httpx.AsyncClient:
            return real(transport=httpx.MockTransport(wrapped), **kwargs)

        monkeypatch.setattr(mod.httpx, "AsyncClient", factory)
        return requests

    return install


@pytest.fixture
def usage(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(mod, "record_web_search_usage", calls.append)
    return calls


@pytest.mark.asyncio
async def test_keyless_request_goes_to_public_path_and_maps_results(
    transport, usage, monkeypatch
):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    requests = transport(lambda r: httpx.Response(200, json=BODY))

    results = await KeenableWebSearchCore().search(QUERY, max_results=3)

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.keenable.ai/v1/search/public"
    assert request.headers["X-Keenable-Title"] == "xagent"
    assert "X-API-Key" not in request.headers
    assert QUERY not in str(request.url)
    assert json.loads(request.content) == {"query": QUERY, "max_results": 3}
    assert results == [
        {
            "title": "T1",
            "link": "https://a.example/1",
            "snippet": ("S1 " + "x" * 600)[:500],
        },
        {"title": "T2", "link": "https://a.example/2", "snippet": "D2"},
    ]
    assert usage == ["keenable"]


@pytest.mark.asyncio
async def test_keyed_request_uses_key_path_and_header(transport, usage, monkeypatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    requests = transport(lambda r: httpx.Response(200, json={"results": []}))

    results = await KeenableWebSearchCore(api_key=API_KEY).search(QUERY, max_results=99)

    request = requests[0]
    assert str(request.url) == "https://api.keenable.ai/v1/search"
    assert request.headers["X-API-Key"] == API_KEY
    assert request.headers["X-Keenable-Title"] == "xagent"
    assert json.loads(request.content)["max_results"] == 50
    assert results == []


@pytest.mark.asyncio
async def test_key_falls_back_to_environment(transport, usage, monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", API_KEY)
    requests = transport(lambda r: httpx.Response(200, json={"results": []}))

    await KeenableWebSearchCore().search(QUERY)

    assert requests[0].headers["X-API-Key"] == API_KEY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,keyed,needle",
    [
        (401, True, "auth failed (401)"),
        (429, False, "1,000 requests per hour"),
        (429, True, "retry later"),
        (500, False, "HTTP error (500)"),
    ],
)
async def test_errors_name_the_status_and_record_no_usage(
    transport, usage, monkeypatch, caplog, status, keyed, needle
):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    transport(lambda r: httpx.Response(status, json={"code": "x"}))

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as excinfo:
            await KeenableWebSearchCore(api_key=API_KEY if keyed else None).search(
                QUERY
            )

    assert needle in str(excinfo.value)
    assert API_KEY not in str(excinfo.value)
    assert QUERY not in caplog.text
    assert usage == []


@pytest.mark.asyncio
async def test_network_error_does_not_leak_the_query(transport, usage, monkeypatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url} for {QUERY}")

    transport(boom)

    with pytest.raises(ValueError) as excinfo:
        await KeenableWebSearchCore().search(QUERY)

    assert "ConnectError" in str(excinfo.value)
    assert QUERY not in str(excinfo.value)
    assert usage == []
