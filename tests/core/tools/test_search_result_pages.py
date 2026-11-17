"""Tests for refusing search-engine result pages in page-fetch tools."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from xagent.core.tools.adapters.vibe.api_tool import APITool
from xagent.core.tools.adapters.vibe.download_web_asset import DownloadWebAssetTool
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
from xagent.core.tools.core.api_tool import APIClientCore
from xagent.core.tools.core.search_result_pages import (
    SEARCH_RESULT_PAGE_REFUSAL,
    is_search_result_page_url,
)
from xagent.core.tools.core.web_content import WebContentFetcher
from xagent.core.workspace import TaskWorkspace

SECRET = "confidential-merger-codename"

SEARCH_URLS = [
    f"https://www.google.com/search?q={SECRET}",
    f"https://google.de/search?q={SECRET}",
    f"https://www.google.co.uk/search?q={SECRET}",
    f"https://www.google.com.au/search?q={SECRET}",
    f"https://www.google.com/search/?q={SECRET}",
    f"https://www.google.com/%73earch?q={SECRET}",
    f"https://www.google.com/a/../search?q={SECRET}",
    f"HTTPS://WWW.GOOGLE.COM./search?q={SECRET}",
    f"https://www.bing.com/search?q={SECRET}",
    f"https://cn.bing.com/search?q={SECRET}",
    f"https://www.bing.com/images/search?q={SECRET}",
    f"https://html.duckduckgo.com/html/?q={SECRET}",
    f"https://lite.duckduckgo.com/lite/?q={SECRET}",
    f"https://duckduckgo.com/html?q={SECRET}",
    f"https://duckduckgo.com/?q={SECRET}",
    f"https://search.yahoo.com/search?p={SECRET}",
    f"https://uk.search.yahoo.com/search;_ylt=abc?p={SECRET}",
    f"https://yandex.com/search/?text={SECRET}",
    f"https://yandex.com.tr/search?text={SECRET}",
    f"https://www.baidu.com/s?wd={SECRET}",
    f"https://search.brave.com/search?q={SECRET}",
    f"https://www.startpage.com/do/search?q={SECRET}",
    f"https://www.ecosia.org/search?q={SECRET}",
    f"https://www.sogou.com/web?query={SECRET}",
    f"https://www.so.com/s?q={SECRET}",
    f"https://search.naver.com/search.naver?query={SECRET}",
    f"https://m.search.naver.com/search.naver?query={SECRET}",
    f"https://search.yahoo.co.jp/search?p={SECRET}",
    f"https://ya.ru/search/?text={SECRET}",
    f"https://encrypted.google.com/search?q={SECRET}",
    f"https://www.bing.com/Search?q={SECRET}",
    f"https://www.baidu.com/S?wd={SECRET}",
]

NON_SEARCH_URLS = [
    "https://support.google.com/accounts/answer/1",
    "https://www.google.com/",
    "https://www.google.com/maps/place/x",
    "https://www.google.com/url?q=https://example.com",
    "https://www.google.com/searchstuff",
    "https://developers.google.com/search/docs",
    "https://google.com.evil.example/search?q=x",
    "https://www.google.evil.example/search?q=x",
    "https://evilgoogle.com/search?q=x",
    "https://notbing.com/search?q=x",
    "https://bing.com.evil.example/search?q=x",
    "https://www.bing.com/maps",
    "https://brave.com/search/api/",
    "https://brave.com/",
    "https://duckduckgo.com/",
    "https://duckduckgo.com/about",
    "https://duckduckgo.com/?ia=web",
    "https://support.yahoo.com/search",
    "https://www.yahoo.com/news",
    "https://www.baidu.com/",
    "https://www.baidu.com/sub",
    "https://www.ecosia.org/blog",
    "https://example.com/search?q=x",
    "https://search.yahoo.co.jp.evil.example/search?p=x",
    "https://search.yahoo.co.uk.example/search?p=x",
    "https://evil.m.search.naver.com.example/search?query=x",
    "https://xm.search.naver.com/search?query=x",
    "https://notya.ru/search?text=x",
    "https://ya.ru.evil.example/search?text=x",
    "https://ya.ru/about",
    "https://encrypted.google.com.evil.example/search?q=x",
    "https://evil.encrypted.google.com/search?q=x",
    "https://encrypted.google.com/",
    "ftp://www.google.com/search?q=x",
    "not a url",
    "",
]


@pytest.mark.parametrize("url", SEARCH_URLS)
def test_matches_search_result_pages(url: str) -> None:
    assert is_search_result_page_url(url)


@pytest.mark.parametrize("url", NON_SEARCH_URLS)
def test_leaves_other_pages_alone(url: str) -> None:
    assert not is_search_result_page_url(url)


@pytest.fixture
def http_request():
    with patch(
        "xagent.core.tools.core.web_content.fetch_public_http_bytes",
        new=AsyncMock(side_effect=AssertionError("network request was made")),
    ) as request:
        yield request


@pytest.mark.asyncio
@pytest.mark.parametrize("url", SEARCH_URLS[:3] + SEARCH_URLS[8:9])
async def test_fetch_refuses_without_request_or_url_in_logs(
    url: str, http_request: AsyncMock, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        result = await WebContentFetcher().fetch(url)

    assert result.blocked_by_policy is True
    assert result.blocked_by_site is False
    assert result.success is False
    assert result.error == SEARCH_RESULT_PAGE_REFUSAL
    assert "web_search" in result.error
    http_request.assert_not_called()
    assert SECRET not in caplog.text
    assert "/search" not in caplog.text


@pytest.mark.asyncio
async def test_fetch_tool_exposes_policy_flag(http_request: AsyncMock) -> None:
    result = await FetchWebContentTool().run_json_async(
        {"url": f"https://www.google.com/search?q={SECRET}"}
    )

    assert result["success"] is False
    assert result["blocked_by_policy"] is True
    assert "web_search" in result["error"]
    assert "search engine" in FetchWebContentTool().description.lower()
    http_request.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_switch_off_restores_old_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_REFUSE_SEARCH_RESULT_FETCH", "false")
    with patch(
        "xagent.core.tools.core.web_content.fetch_public_http_bytes",
        new=AsyncMock(side_effect=RuntimeError("reached network")),
    ) as request:
        result = await WebContentFetcher().fetch(
            f"https://www.google.com/search?q={SECRET}"
        )

    request.assert_awaited_once()
    assert result.blocked_by_policy is False
    assert "reached network" in (result.error or "")


@pytest.mark.asyncio
async def test_download_web_asset_refuses_search_page(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tool = DownloadWebAssetTool(TaskWorkspace("serp-test", base_dir=str(tmp_path)))
    with (
        patch(
            "xagent.core.tools.adapters.vibe.download_web_asset.fetch_public_http_bytes",
            new=AsyncMock(side_effect=AssertionError("network request was made")),
        ) as request,
        caplog.at_level(logging.DEBUG),
    ):
        result = await tool.run_json_async(
            {"url": f"https://www.bing.com/search?q={SECRET}"}
        )

    assert result["success"] is False
    assert result["error"] == SEARCH_RESULT_PAGE_REFUSAL
    request.assert_not_called()
    assert SECRET not in caplog.text


# --- api_call and download_web_asset -----------------------------------------

API_REFUSED_URLS = [
    (f"https://www.google.com/search?q={SECRET}", None),
    (f"https://html.duckduckgo.com/html/?q={SECRET}", None),
    ("https://www.google.com/search", {"q": SECRET}),
    ("https://duckduckgo.com/", {"q": SECRET}),
    ("https://www.bing.com/search?setlang=en", {"q": SECRET}),
]


def _api_client_without_network(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    request = AsyncMock(side_effect=AssertionError("network request was made"))
    monkeypatch.setattr(APIClientCore, "_make_request", request)
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize(("url", "params"), API_REFUSED_URLS)
async def test_api_call_refuses_search_page(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    url: str,
    params: dict[str, str] | None,
) -> None:
    request = _api_client_without_network(monkeypatch)
    args: dict[str, object] = {"url": url}
    if params is not None:
        args["params"] = params

    with caplog.at_level(logging.DEBUG):
        result = await APITool().run_json_async(args)

    assert result["success"] is False
    assert result["status_code"] == 0
    assert result["headers"] == {}
    assert result["body"] is None
    assert result["error"] == SEARCH_RESULT_PAGE_REFUSAL
    assert result["blocked_by_policy"] is True
    request.assert_not_called()
    # Only the host is logged; neither the query text nor the path.
    assert SECRET not in caplog.text
    assert "/search" not in caplog.text
    assert "host=" in caplog.text


@pytest.mark.asyncio
async def test_api_call_refuses_api_key_query_param_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _api_client_without_network(monkeypatch)

    result = await APIClientCore().call_api(
        "https://duckduckgo.com/",
        auth_type="api_key_query",
        auth_token=SECRET,
        api_key_param="q",
    )

    assert result["blocked_by_policy"] is True
    request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "params"),
    [
        ("https://www.google.com/maps/place/Sydney", None),
        ("https://www.google.com/search/about", None),
        ("https://www.bing.com/maps", {"q": "x"}),
        ("https://duckduckgo.com/", None),
        ("https://duckduckgo.com/about", {"q": "x"}),
        ("https://api.example.com/search", {"q": "x"}),
    ],
)
async def test_api_call_still_allows_other_pages_on_the_same_sites(
    monkeypatch: pytest.MonkeyPatch, url: str, params: dict[str, str] | None
) -> None:
    request = AsyncMock(
        return_value={
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": "ok",
            "error": None,
        }
    )
    monkeypatch.setattr(APIClientCore, "_make_request", request)
    args: dict[str, object] = {"url": url}
    if params is not None:
        args["params"] = params

    result = await APITool().run_json_async(args)

    assert result["success"] is True
    assert result["blocked_by_policy"] is False
    request.assert_awaited_once()


@pytest.mark.asyncio
async def test_api_call_switch_off_restores_old_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_REFUSE_SEARCH_RESULT_FETCH", "false")
    request = AsyncMock(
        return_value={
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": "ok",
            "error": None,
        }
    )
    monkeypatch.setattr(APIClientCore, "_make_request", request)

    result = await APITool().run_json_async(
        {"url": "https://www.google.com/search", "params": {"q": SECRET}}
    )

    request.assert_awaited_once()
    assert result["success"] is True
    assert result["blocked_by_policy"] is False


def test_api_call_description_forbids_searching() -> None:
    description = APITool().description.lower()
    assert "never use this tool to query a search engine" in description
    assert "web_search is the only way to search" in description


@pytest.mark.asyncio
async def test_download_web_asset_sets_blocked_by_policy(tmp_path: Path) -> None:
    tool = DownloadWebAssetTool(TaskWorkspace("serp-flag", base_dir=str(tmp_path)))
    with patch(
        "xagent.core.tools.adapters.vibe.download_web_asset.fetch_public_http_bytes",
        new=AsyncMock(side_effect=AssertionError("network request was made")),
    ):
        refused = await tool.run_json_async(
            {"url": f"https://www.bing.com/search?q={SECRET}"}
        )
        failed = await tool.run_json_async({"url": "not a url"})

    assert refused["blocked_by_policy"] is True
    # An ordinary failure must not be reported as a policy refusal.
    assert failed["success"] is False
    assert failed["blocked_by_policy"] is False
    assert "search" in tool.description.lower()


@pytest.mark.asyncio
async def test_download_web_asset_switch_off_restores_old_behaviour(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XAGENT_REFUSE_SEARCH_RESULT_FETCH", "false")
    tool = DownloadWebAssetTool(TaskWorkspace("serp-off", base_dir=str(tmp_path)))
    with patch(
        "xagent.core.tools.adapters.vibe.download_web_asset.fetch_public_http_bytes",
        new=AsyncMock(side_effect=RuntimeError("reached network")),
    ) as request:
        result = await tool.run_json_async(
            {"url": f"https://www.bing.com/search?q={SECRET}"}
        )

    request.assert_awaited_once()
    assert result["blocked_by_policy"] is False
    assert "reached network" in result["error"]


UNICODE_SEPARATOR_URLS = [
    f"https://www.google\u3002com/search?q={SECRET}",
    f"https://www.google\uff0ecom/search?q={SECRET}",
    f"https://www.google\uff61com/search?q={SECRET}",
    f"https://www.bing\u3002com/search?q={SECRET}",
]


@pytest.mark.parametrize("url", UNICODE_SEPARATOR_URLS)
def test_matcher_sees_the_host_httpx_connects_to(url: str) -> None:
    # httpx canonicalises these separators to "."; the matcher must too.
    assert httpx.URL(url).host in {"www.google.com", "www.bing.com"}
    assert is_search_result_page_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("url", UNICODE_SEPARATOR_URLS)
async def test_api_call_refuses_unicode_separator_host_without_params(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, url: str
) -> None:
    request = _api_client_without_network(monkeypatch)

    with caplog.at_level(logging.DEBUG):
        result = await APITool().run_json_async({"url": url})

    assert result["blocked_by_policy"] is True
    assert result["status_code"] == 0
    request.assert_not_called()
    assert SECRET not in caplog.text


@pytest.mark.asyncio
async def test_fetch_and_download_refuse_unicode_separator_host(
    tmp_path: Path,
) -> None:
    url = UNICODE_SEPARATOR_URLS[0]
    with patch(
        "xagent.core.tools.core.web_content.fetch_public_http_bytes",
        new=AsyncMock(side_effect=AssertionError("network request was made")),
    ):
        fetched = await WebContentFetcher().fetch(url)
    tool = DownloadWebAssetTool(TaskWorkspace("serp-idna", base_dir=str(tmp_path)))
    with patch(
        "xagent.core.tools.adapters.vibe.download_web_asset.fetch_public_http_bytes",
        new=AsyncMock(side_effect=AssertionError("network request was made")),
    ):
        downloaded = await tool.run_json_async({"url": url})

    assert fetched.blocked_by_policy is True
    assert downloaded["blocked_by_policy"] is True


def test_unparsable_host_keeps_the_raw_host_match() -> None:
    # httpx rejects a tab in the host, so it is never sent; staying strict is
    # harmless and must not raise.
    assert is_search_result_page_url(f"https://www.goog\tle.com/search?q={SECRET}")
