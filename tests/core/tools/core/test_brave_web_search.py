"""Tests for BraveWebSearchCore. All HTTP goes through httpx.MockTransport."""

import json
import logging
import traceback
from typing import Any, Callable

import httpx
import pytest

from xagent.core.tools.core import brave_web_search as mod
from xagent.core.tools.core.brave_web_search import BraveWebSearchCore

API_KEY = "test-brave-key-0000"
QUERY = "jane.doe@example.com quarterly forecast"


@pytest.fixture
def transport(monkeypatch):
    """Route the core's httpx.AsyncClient through a MockTransport."""
    state: dict[str, Any] = {"requests": [], "handler": None}

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> list:
        def wrapped(request: httpx.Request) -> httpx.Response:
            state["requests"].append(request)
            return handler(request)

        real = httpx.AsyncClient

        def factory(**kwargs: Any) -> httpx.AsyncClient:
            return real(transport=httpx.MockTransport(wrapped), **kwargs)

        monkeypatch.setattr(mod.httpx, "AsyncClient", factory)
        return state["requests"]

    return install


WEB_BODY = {
    "web": {
        "results": [
            {"title": "T1", "url": "https://a.example/1", "description": "D1"},
            {"title": "T2", "url": "https://a.example/2", "description": "D2"},
        ]
    }
}
CTX_BODY = {
    "grounding": {
        "generic": [
            {
                "url": "https://c.example/1",
                "title": "C1",
                "snippets": ["first " + "x" * 600, "second"],
            },
            {"url": "https://c.example/2", "title": "C2", "snippets": []},
        ]
    },
    "sources": {},
}


@pytest.mark.asyncio
async def test_web_endpoint_request_and_mapping(transport, monkeypatch):
    monkeypatch.delenv("XAGENT_BRAVE_SEARCH_ENDPOINT", raising=False)
    requests = transport(lambda r: httpx.Response(200, json=WEB_BODY))

    results = await BraveWebSearchCore(api_key=API_KEY).search(QUERY, max_results=50)

    assert results == [
        {"title": "T1", "link": "https://a.example/1", "snippet": "D1"},
        {"title": "T2", "link": "https://a.example/2", "snippet": "D2"},
    ]
    (req,) = requests
    assert req.method == "GET"
    assert req.url.host == "api.search.brave.com"
    assert req.url.path == "/res/v1/web/search"
    assert req.headers["X-Subscription-Token"] == API_KEY
    assert req.url.params["q"] == QUERY
    assert req.url.params["count"] == "20"
    assert req.url.params["text_decorations"] == "false"


@pytest.mark.asyncio
async def test_web_endpoint_include_content_fetches_pages(transport, monkeypatch):
    transport(lambda r: httpx.Response(200, json=WEB_BODY))
    fetched: list[str] = []

    class FakeFetcher:
        def __init__(self, proxy_url=None):
            pass

        async def fetch_text(self, url: str) -> str:
            fetched.append(url)
            return f"page:{url}"

    monkeypatch.setattr(mod, "WebContentFetcher", FakeFetcher)
    results = await BraveWebSearchCore(api_key=API_KEY).search(
        QUERY, include_content=True
    )
    assert [r["content"] for r in results] == [
        "page:https://a.example/1",
        "page:https://a.example/2",
    ]
    assert fetched == ["https://a.example/1", "https://a.example/2"]


@pytest.mark.asyncio
async def test_llm_context_endpoint_request_and_mapping(transport, monkeypatch):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", "llm_context")
    requests = transport(lambda r: httpx.Response(200, json=CTX_BODY))

    results = await BraveWebSearchCore(api_key=API_KEY).search(
        QUERY, max_results=3, include_content=True
    )

    (req,) = requests
    assert req.method == "POST"
    assert req.url.path == "/res/v1/llm/context"
    assert req.url.query == b""  # the query stays out of the URL
    assert req.headers["X-Subscription-Token"] == API_KEY
    assert json.loads(req.content) == {
        "q": QUERY,
        "count": 3,
        "maximum_number_of_tokens": 8192,
    }
    assert results[0]["title"] == "C1"
    assert results[0]["link"] == "https://c.example/1"
    assert results[0]["snippet"] == ("first " + "x" * 600)[:500]
    assert results[0]["content"] == ("first " + "x" * 600) + "\n\nsecond"
    assert results[1] == {
        "title": "C2",
        "link": "https://c.example/2",
        "snippet": "",
        "content": "",
    }


@pytest.mark.asyncio
async def test_llm_context_content_is_bounded(transport, monkeypatch):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", "llm_context")
    body = {
        "grounding": {
            "generic": [
                {"url": "https://c.example/1", "title": "C", "snippets": ["y" * 9000]}
            ]
        }
    }
    transport(lambda r: httpx.Response(200, json=body))
    (result,) = await BraveWebSearchCore(api_key=API_KEY).search(
        QUERY, include_content=True
    )
    assert len(result["content"]) == 8192


@pytest.mark.asyncio
async def test_llm_context_without_include_content_has_no_content(
    transport, monkeypatch
):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", "llm_context")
    transport(lambda r: httpx.Response(200, json=CTX_BODY))
    results = await BraveWebSearchCore(api_key=API_KEY).search(QUERY)
    assert all("content" not in r for r in results)


@pytest.mark.asyncio
async def test_unknown_endpoint_uses_web(transport, monkeypatch, caplog):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", "bogus")
    requests = transport(lambda r: httpx.Response(200, json=WEB_BODY))
    with caplog.at_level(logging.WARNING):
        await BraveWebSearchCore(api_key=API_KEY).search(QUERY)
    assert requests[0].url.path == "/res/v1/web/search"
    assert "XAGENT_BRAVE_SEARCH_ENDPOINT" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["web", "llm_context"])
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failures(transport, monkeypatch, endpoint, status):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", endpoint)
    transport(lambda r: httpx.Response(status, json={"error": "nope"}))
    with pytest.raises(ValueError, match=f"auth failed \\({status}\\)"):
        await BraveWebSearchCore(api_key=API_KEY).search(QUERY)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["web", "llm_context"])
async def test_rate_limit(transport, monkeypatch, endpoint):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", endpoint)
    transport(lambda r: httpx.Response(429))
    with pytest.raises(ValueError, match="rate limit.*\\(429\\)"):
        await BraveWebSearchCore(api_key=API_KEY).search(QUERY)


@pytest.mark.asyncio
async def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BRAVE_SEARCH_API_KEY"):
        await BraveWebSearchCore().search(QUERY)


@pytest.mark.asyncio
async def test_env_key_fallback(transport, monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "env-key")
    requests = transport(lambda r: httpx.Response(200, json=WEB_BODY))
    await BraveWebSearchCore().search(QUERY)
    assert requests[0].headers["X-Subscription-Token"] == "env-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["web", "llm_context"])
async def test_query_limits_raise_without_request(transport, monkeypatch, endpoint):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", endpoint)
    requests = transport(lambda r: httpx.Response(200, json=WEB_BODY))
    core = BraveWebSearchCore(api_key=API_KEY)

    with pytest.raises(ValueError, match="600 characters"):
        await core.search("a" * 601)
    with pytest.raises(ValueError, match="75 words"):
        await core.search(" ".join(["w"] * 76))
    assert requests == []

    # Exactly at the limits is accepted.
    await core.search("a" * 600)
    await core.search(" ".join(["w"] * 75))
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["web", "llm_context"])
@pytest.mark.parametrize("status", [400, 429, 500])
@pytest.mark.parametrize("failure", ["status", "network"])
async def test_errors_never_leak_key_url_or_query(
    transport, monkeypatch, caplog, endpoint, status, failure
):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", endpoint)

    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            # httpx puts the request in the exception; its text carries the URL.
            raise httpx.ConnectError(f"connect failed for {request.url}")
        return httpx.Response(status, text=f"echo {request.url} {API_KEY}")

    transport(handler)
    # Force httpx's own request log line on: it carries the full request URL.
    with caplog.at_level(logging.DEBUG), caplog.at_level(logging.INFO, logger="httpx"):
        with pytest.raises(ValueError) as exc_info:
            await BraveWebSearchCore(api_key=API_KEY).search(QUERY)

    err = exc_info.value
    rendered = "".join(traceback.format_exception(err))
    logged = caplog.text + "".join(r.getMessage() for r in caplog.records)
    for text in (str(err), rendered, logged):
        assert API_KEY not in text
        assert "api.search.brave.com" not in text
        assert "forecast" not in text
        assert "jane.doe" not in text
        assert "jane.doe%40example.com" not in text
    assert err.__cause__ is None
    assert err.__context__ is None or err.__suppress_context__
    if failure == "status" and status not in {429}:
        assert str(status) in str(err)


@pytest.mark.asyncio
async def test_success_does_not_log_query(transport, monkeypatch, caplog):
    monkeypatch.delenv("XAGENT_BRAVE_SEARCH_ENDPOINT", raising=False)
    transport(lambda r: httpx.Response(200, json=WEB_BODY))
    with caplog.at_level(logging.DEBUG), caplog.at_level(logging.INFO, logger="httpx"):
        await BraveWebSearchCore(api_key=API_KEY).search(QUERY)
    assert "forecast" not in caplog.text
    assert "jane.doe" not in caplog.text
    assert API_KEY not in caplog.text


@pytest.mark.asyncio
async def test_non_json_body_raises_clean_error(transport):
    transport(lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(ValueError, match="invalid response body"):
        await BraveWebSearchCore(api_key=API_KEY).search(QUERY)
