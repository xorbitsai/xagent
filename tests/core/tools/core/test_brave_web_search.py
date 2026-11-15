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
QUERY = "acme quarterly revenue forecast"


@pytest.fixture(params=["httpx_warning", "httpx_notset"])
def httpx_log_level(request):
    """Run leak tests with httpx's own logger both quiet and fully verbose."""
    lg = logging.getLogger("httpx")
    saved = lg.level
    lg.setLevel(logging.WARNING if request.param == "httpx_warning" else logging.NOTSET)
    yield request.param
    lg.setLevel(saved)


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
async def test_country_sent_on_web_endpoint_when_set(transport, monkeypatch):
    monkeypatch.delenv("XAGENT_BRAVE_SEARCH_ENDPOINT", raising=False)
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_COUNTRY", " au ")
    requests = transport(lambda r: httpx.Response(200, json=WEB_BODY))
    await BraveWebSearchCore(api_key=API_KEY).search(QUERY)
    assert requests[0].url.params["country"] == "AU"
    assert "search_lang" not in requests[0].url.params


@pytest.mark.asyncio
async def test_country_sent_on_llm_context_endpoint_when_set(transport, monkeypatch):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", "llm_context")
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_COUNTRY", "AU")
    requests = transport(lambda r: httpx.Response(200, json=CTX_BODY))
    await BraveWebSearchCore(api_key=API_KEY).search(QUERY)
    body = json.loads(requests[0].content)
    assert body["country"] == "au"
    assert "search_lang" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["web", "llm_context"])
@pytest.mark.parametrize("value", [None, "", "  "])
async def test_country_absent_when_unset(transport, monkeypatch, endpoint, value):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", endpoint)
    if value is None:
        monkeypatch.delenv("XAGENT_BRAVE_SEARCH_COUNTRY", raising=False)
    else:
        monkeypatch.setenv("XAGENT_BRAVE_SEARCH_COUNTRY", value)
    requests = transport(lambda r: httpx.Response(200, json=WEB_BODY))
    await BraveWebSearchCore(api_key=API_KEY).search(QUERY)
    sent = requests[0].url.query.decode() + requests[0].content.decode()
    assert "country" not in sent


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["web", "llm_context"])
@pytest.mark.parametrize("value", ["AUS", "7", "1x", "éé", "au-AU"])
async def test_invalid_country_ignored_with_one_warning(
    transport, monkeypatch, caplog, endpoint, value
):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", endpoint)
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_COUNTRY", value)
    requests = transport(lambda r: httpx.Response(200, json=WEB_BODY))
    with caplog.at_level(logging.WARNING):
        await BraveWebSearchCore(api_key=API_KEY).search(QUERY)
    sent = requests[0].url.query.decode() + requests[0].content.decode()
    assert "country" not in sent
    warnings = [r for r in caplog.records if "XAGENT_BRAVE_SEARCH_COUNTRY" in r.message]
    assert len(warnings) == 1
    assert value not in warnings[0].getMessage()


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
    transport, monkeypatch, caplog, httpx_log_level, endpoint, status, failure
):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", endpoint)

    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            # httpx puts the request in the exception; its text carries the URL.
            raise httpx.ConnectError(f"connect failed for {request.url}")
        return httpx.Response(status, text=f"echo {request.url} {API_KEY}")

    transport(handler)
    with caplog.at_level(logging.DEBUG):
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
async def test_success_does_not_log_query(
    transport, monkeypatch, caplog, httpx_log_level
):
    monkeypatch.delenv("XAGENT_BRAVE_SEARCH_ENDPOINT", raising=False)
    transport(lambda r: httpx.Response(200, json=WEB_BODY))
    with caplog.at_level(logging.DEBUG):
        await BraveWebSearchCore(api_key=API_KEY).search(QUERY)
    assert "forecast" not in caplog.text
    assert "jane.doe" not in caplog.text
    assert API_KEY not in caplog.text


@pytest.mark.asyncio
async def test_non_json_body_raises_clean_error(transport):
    transport(lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(ValueError, match="invalid response body"):
        await BraveWebSearchCore(api_key=API_KEY).search(QUERY)


# --- redaction guard (xagent.core.tools.core.web_search_guard) ----------------

from urllib.parse import unquote_plus  # noqa: E402

from xagent.core.tools.core.web_search_guard import (  # noqa: E402
    SearchAuditEvent,
    SearchQueryRedactionError,
    set_search_audit_hook,
)

PII_EMAIL = "jane.doe@example.com"
PII_PHONE = "+61 412 345 678"
PII_QUERY = f"find {PII_EMAIL} {PII_PHONE} Acme Pty Ltd"
PII_REDACTED = "find [EMAIL] [PHONE] Acme Pty Ltd"


@pytest.fixture
def audit():
    events: list[SearchAuditEvent] = []
    set_search_audit_hook(events.append)
    yield events
    set_search_audit_hook(None)


def _wire(requests: list[httpx.Request]) -> str:
    return unquote_plus(" ".join(f"{r.url} {r.content.decode()}" for r in requests))


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["web", "llm_context"])
async def test_brave_sends_and_audits_only_the_redacted_query(
    transport, monkeypatch, caplog, audit, endpoint
):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", endpoint)
    body = WEB_BODY if endpoint == "web" else CTX_BODY
    requests = transport(lambda r: httpx.Response(200, json=body))
    with caplog.at_level(logging.DEBUG):
        results = await BraveWebSearchCore(api_key=API_KEY).search(PII_QUERY)

    wire = _wire(requests)
    assert PII_REDACTED in wire
    assert PII_EMAIL not in wire and "412 345 678" not in wire
    (event,) = audit
    assert event.provider == "brave"
    assert event.query == PII_REDACTED
    assert event.result_count == len(results) == 2
    assert event.outcome == "success"
    # The core logs no query text at all, raw or redacted.
    assert "Acme" not in caplog.text and "[EMAIL]" not in caplog.text
    assert PII_EMAIL not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["web", "llm_context"])
async def test_brave_query_limits_apply_to_the_redacted_text(
    transport, monkeypatch, audit, endpoint
):
    monkeypatch.setenv("XAGENT_BRAVE_SEARCH_ENDPOINT", endpoint)
    requests = transport(lambda r: httpx.Response(200, json=WEB_BODY))
    core = BraveWebSearchCore(api_key=API_KEY)

    # 601 raw characters shrink below the limit once the email is a placeholder.
    query = f"{PII_EMAIL} " + "b" * 585
    assert len(query) > 600
    await core.search(query)
    assert len(requests) == 1
    assert audit[-1].query == "[EMAIL] " + "b" * 585

    # An over-long query that stays over-long is refused after redaction, and
    # the failure event carries the redacted text, not the raw one.
    too_long = f"{PII_EMAIL} " + "c" * 600
    with pytest.raises(ValueError, match="600 characters"):
        await core.search(too_long)
    assert len(requests) == 1
    assert audit[-1].outcome == "failure"
    assert PII_EMAIL not in audit[-1].query


@pytest.mark.asyncio
async def test_brave_redaction_failure_sends_nothing(transport, monkeypatch, audit):
    import xagent.core.tools.core.search_query_redaction as redaction

    monkeypatch.setattr(
        redaction,
        "_sub",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(PII_EMAIL)),
    )
    requests = transport(lambda r: httpx.Response(200, json=WEB_BODY))
    with pytest.raises(SearchQueryRedactionError):
        await BraveWebSearchCore(api_key=API_KEY).search(PII_QUERY)
    assert requests == []
    assert audit[-1].failure_kind == "redaction_error"
