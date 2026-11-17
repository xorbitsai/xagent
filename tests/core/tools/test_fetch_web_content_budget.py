"""Page-text budget for fetch_web_content: boilerplate trimming and a char cap."""

from unittest.mock import AsyncMock, patch

import pytest

from xagent.config import get_fetch_web_content_budget_enabled
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
from xagent.core.tools.core.web_content import truncate_page_text

from .test_fetch_web_content import _MockStreamContext, _MockStreamResponse


@pytest.fixture(autouse=True)
def allow_public_test_hosts():
    with patch(
        "xagent.core.utils.security.validate_public_http_url",
        new=AsyncMock(return_value=["93.184.216.34"]),
    ):
        yield


def _page(paragraphs: int, para: str = "Answer sentence number {i}. " * 5) -> bytes:
    body = "".join(f"<p>{para.format(i=i)}</p>" for i in range(paragraphs))
    html = (
        "<html><head><title>T</title></head><body>"
        "<nav>SITE-NAV-LINKS</nav><header>SITE-HEADER</header>"
        f"<main>{body}</main>"
        "<aside>RELATED-STORIES</aside><footer>SITE-FOOTER</footer>"
        "</body></html>"
    )
    return html.encode()


async def _fetch(args: dict, body: bytes) -> dict:
    response = _MockStreamResponse(
        body=body,
        headers={"content-type": "text/html"},
        url="https://example.com/page",
    )
    with patch("httpx.AsyncClient.stream", return_value=_MockStreamContext(response)):
        return await FetchWebContentTool().run_json_async(
            {"url": "https://example.com/page", **args}
        )


@pytest.mark.asyncio
async def test_boilerplate_is_dropped_and_main_text_kept():
    result = await _fetch({}, _page(10))
    assert "Answer sentence number 0." in result["content"]
    for noise in ("SITE-NAV-LINKS", "SITE-HEADER", "RELATED-STORIES", "SITE-FOOTER"):
        assert noise not in result["content"]
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_default_budget_truncates_and_says_so():
    result = await _fetch({}, _page(400))
    assert result["truncated"] is True
    assert result["total_chars"] > 12_000
    assert len(result["content"]) < 12_000 + 300
    assert "Page text truncated" in result["content"]
    assert "Answer sentence number 0." in result["content"]


@pytest.mark.asyncio
async def test_max_chars_argument_raises_the_cap():
    default = await _fetch({}, _page(400))
    larger = await _fetch({"max_chars": 40_000}, _page(400))
    assert len(larger["content"]) > len(default["content"])


@pytest.mark.asyncio
async def test_short_main_region_falls_back_to_whole_page():
    body = (
        b"<html><body><nav>TOP-NAV</nav><main>tiny</main>"
        b"<div>Real article text lives outside main.</div></body></html>"
    )
    result = await _fetch({}, body)
    assert "Real article text lives outside main." in result["content"]


@pytest.mark.asyncio
async def test_switch_off_restores_full_untrimmed_page(monkeypatch):
    monkeypatch.setenv("XAGENT_FETCH_WEB_CONTENT_BUDGET", "false")
    assert get_fetch_web_content_budget_enabled() is False
    result = await _fetch({}, _page(400))
    assert result["truncated"] is False
    assert "SITE-NAV-LINKS" in result["content"]
    assert "SITE-FOOTER" in result["content"]
    assert len(result["content"]) > 20_000


def test_budget_is_on_by_default(monkeypatch):
    monkeypatch.delenv("XAGENT_FETCH_WEB_CONTENT_BUDGET", raising=False)
    assert get_fetch_web_content_budget_enabled() is True


def test_truncate_prefers_paragraph_boundary():
    text = "a" * 60 + "\n\n" + "b" * 60
    cut, truncated = truncate_page_text(text, 100)
    assert truncated is True
    assert cut == "a" * 60
    assert truncate_page_text("short", 100) == ("short", False)
