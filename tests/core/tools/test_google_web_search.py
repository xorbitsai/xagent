"""Tests for the Google Custom Search request address."""

from typing import Any

import pytest

from xagent.core.tools.core import web_search as web_search_module
from xagent.core.tools.core.web_search import WebSearchCore


class _Response:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"items": [{"title": "Example", "link": "https://example.com"}]}


@pytest.fixture
def requests(monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def get(self, url: str, **kwargs: Any) -> _Response:
            calls.append((url, kwargs))
            return _Response()

    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(WebSearchCore, "_get_proxy_url", lambda self: None)
    monkeypatch.delenv("GOOGLE_CSE_BASE_URL", raising=False)
    return calls


@pytest.mark.asyncio
async def test_google_search_defaults_to_googleapis(requests):
    results = await WebSearchCore(api_key="key", cse_id="cse").search("test")

    assert results == [
        {"title": "Example", "link": "https://example.com", "snippet": ""}
    ]
    url, kwargs = requests[0]
    assert url == "https://www.googleapis.com/customsearch/v1"
    assert kwargs["params"]["key"] == "key"
    assert kwargs["params"]["cx"] == "cse"


@pytest.mark.asyncio
async def test_google_search_uses_base_url_argument(requests, monkeypatch):
    monkeypatch.setenv("GOOGLE_CSE_BASE_URL", "https://env.example.com")
    searcher = WebSearchCore(
        api_key="key", cse_id="cse", base_url="https://search.example.com/"
    )

    await searcher.search("test")

    assert requests[0][0] == "https://search.example.com/customsearch/v1"


@pytest.mark.asyncio
async def test_google_search_uses_base_url_env(requests, monkeypatch):
    monkeypatch.setenv("GOOGLE_CSE_BASE_URL", "https://env.example.com")

    await WebSearchCore(api_key="key", cse_id="cse").search("test")

    assert requests[0][0] == "https://env.example.com/customsearch/v1"
