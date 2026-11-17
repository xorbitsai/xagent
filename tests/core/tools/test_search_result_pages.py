"""Tests for refusing search-engine result pages in page-fetch tools."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from xagent.core.tools.adapters.vibe.download_web_asset import DownloadWebAssetTool
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
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
