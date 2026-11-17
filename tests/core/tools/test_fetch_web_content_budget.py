"""Page-text budget for fetch_web_content: boilerplate trimming and a char cap."""

import re
from unittest.mock import AsyncMock, patch

import pytest

from xagent.config import get_fetch_web_content_budget_enabled
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
from xagent.core.tools.core.web_content import slice_page_text

from .test_fetch_web_content import _MockStreamContext, _MockStreamResponse


@pytest.fixture(autouse=True)
def clean_budget_env(monkeypatch):
    monkeypatch.delenv("XAGENT_FETCH_WEB_CONTENT_BUDGET", raising=False)
    monkeypatch.delenv("XAGENT_FETCH_WEB_CONTENT_MAX_CHARS", raising=False)


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


async def _fetch_raw(args: dict, body: bytes, content_type: str) -> dict:
    response = _MockStreamResponse(
        body=body,
        headers={"content-type": content_type},
        url="https://example.com/page",
    )
    with patch("httpx.AsyncClient.stream", return_value=_MockStreamContext(response)):
        return await FetchWebContentTool().run_json_async(
            {"url": "https://example.com/page", **args}
        )


LONG = "Body sentence about the topic. " * 20


@pytest.mark.asyncio
async def test_default_budget_truncates_and_says_so():
    result = await _fetch({}, _page(400))
    assert result["truncated"] is True
    assert result["total_chars"] > 12_000
    assert len(result["content"]) < 12_000 + 300
    assert "start_char=" in result["content"]
    assert "Answer sentence number 0." in result["content"]


@pytest.mark.asyncio
async def test_max_chars_sets_the_size_of_one_call():
    small = await _fetch({"max_chars": 2_000}, _page(400))
    large = await _fetch({"max_chars": 20_000}, _page(400))
    assert 1_500 < len(small["content"]) < 2_300
    assert 15_000 < len(large["content"]) < 20_300


async def _read_all(body: bytes, max_chars: int) -> tuple[str, dict]:
    start, pieces = 0, []
    for _ in range(200):
        r = await _fetch({"max_chars": max_chars, "start_char": start}, body)
        pieces.append(r["content"].split("\n\n[")[0])
        if not r["truncated"]:
            break
        start = int(re.search(r"start_char=(\d+)", r["content"]).group(1))
    return "".join("".join(pieces).split()), r


@pytest.mark.asyncio
async def test_start_char_continues_without_gap_or_overlap():
    body = _page(400)
    small, last = await _read_all(body, 5_000)
    big, _ = await _read_all(body, 9_000)
    assert small == big
    assert small.count("Answernumber0.") + small.count("Answersentencenumber0.") >= 1
    assert "End of page" in last["content"]
    assert last["truncated"] is False


@pytest.mark.asyncio
async def test_start_char_past_the_end_is_reported():
    r = await _fetch({"start_char": 10_000_000}, _page(5))
    assert "past the end" in r["content"]
    assert r["truncated"] is False


@pytest.mark.asyncio
async def test_nested_role_elements_do_not_crash_the_fetch():
    body = (
        b'<html><body><div role="navigation"><a role="button">x</a>'
        b'<ul role="list"><li role="menuitem">y</li></ul></div>'
        b"<p>kept paragraph</p></body></html>"
    )
    result = await _fetch({}, body)
    assert result["success"] is True
    assert "kept paragraph" in result["content"]


@pytest.mark.asyncio
async def test_form_wrapped_page_is_not_emptied():
    body = f"<html><body><form><h1>Title</h1><p>{LONG}</p></form></body></html>"
    result = await _fetch({}, body.encode())
    assert result["success"] is True
    assert "Body sentence about the topic." in result["content"]


@pytest.mark.asyncio
async def test_form_content_is_kept_even_when_other_text_remains():
    body = (
        f"<html><body><form><p>FORM-BODY {LONG}</p></form>"
        f"<p>Other text. {LONG}</p></body></html>"
    )
    result = await _fetch({}, body.encode())
    assert "FORM-BODY" in result["content"]


@pytest.mark.asyncio
async def test_untrimmed_page_returned_when_trimming_removes_almost_everything():
    body = f"<html><body><aside>ASIDE-BODY {LONG}</aside><p>tiny</p></body></html>"
    result = await _fetch({}, body.encode())
    assert "ASIDE-BODY" in result["content"]


@pytest.mark.asyncio
async def test_article_header_and_footer_are_kept_inside_the_main_region():
    body = (
        "<html><body><header>SITE-HEADER</header><main><article>"
        f"<header><h1>Headline</h1><time>2026-01-02</time></header><p>{LONG}</p>"
        "<footer>Byline: A. Writer</footer></article></main></body></html>"
    )
    result = await _fetch({}, body.encode())
    assert "Headline" in result["content"]
    assert "2026-01-02" in result["content"]
    assert "Byline: A. Writer" in result["content"]
    assert "SITE-HEADER" not in result["content"]


@pytest.mark.asyncio
async def test_lone_article_is_used_when_main_is_short():
    body = (
        "<html><body><nav>TOP-NAV</nav><main>tiny</main>"
        f"<article><p>{LONG}</p></article></body></html>"
    )
    result = await _fetch({}, body.encode())
    assert "Body sentence about the topic." in result["content"]
    assert "TOP-NAV" not in result["content"]


@pytest.mark.asyncio
async def test_several_articles_are_not_reduced_to_the_first():
    body = (
        "<html><body><nav>TOP-NAV</nav>"
        f"<article><p>FIRST-ITEM {LONG}</p></article>"
        f"<article><p>SECOND-ITEM {LONG}</p></article></body></html>"
    )
    result = await _fetch({}, body.encode())
    assert "FIRST-ITEM" in result["content"]
    assert "SECOND-ITEM" in result["content"]


@pytest.mark.asyncio
async def test_short_main_region_falls_back_to_whole_page():
    body = (
        b"<html><body><nav>TOP-NAV</nav><main>tiny</main>"
        b"<div>Real article text lives outside main.</div></body></html>"
    )
    result = await _fetch({}, body)
    assert "Real article text lives outside main." in result["content"]


@pytest.mark.asyncio
async def test_plain_text_and_json_are_sliced_too():
    text = await _fetch_raw({"max_chars": 1_000}, b"word " * 2_000, "text/plain")
    assert text["truncated"] is True
    assert len(text["content"]) < 1_300
    data = await _fetch_raw(
        {"max_chars": 1_000}, b'{"k": "' + b"v" * 5_000 + b'"}', "application/json"
    )
    assert data["truncated"] is True


@pytest.mark.asyncio
async def test_switch_off_restores_full_untrimmed_page(monkeypatch):
    monkeypatch.setenv("XAGENT_FETCH_WEB_CONTENT_BUDGET", "false")
    assert get_fetch_web_content_budget_enabled() is False
    result = await _fetch({"max_chars": 1_000, "start_char": 500}, _page(400))
    assert result["truncated"] is False
    assert result["total_chars"] is None
    assert "SITE-NAV-LINKS" in result["content"]
    assert "SITE-FOOTER" in result["content"]
    assert len(result["content"]) > 20_000


def test_description_follows_the_configuration(monkeypatch):
    assert "capped at 12000 characters" in FetchWebContentTool().description
    monkeypatch.setenv("XAGENT_FETCH_WEB_CONTENT_MAX_CHARS", "20000")
    assert "capped at 20000 characters" in FetchWebContentTool().description
    monkeypatch.setenv("XAGENT_FETCH_WEB_CONTENT_BUDGET", "false")
    description = FetchWebContentTool().description
    assert "capped" not in description
    assert "full page text is returned" in description


def test_slice_prefers_paragraph_boundary_and_hard_cuts_without_one():
    text = "a" * 60 + "\n\n" + "b" * 60
    piece, end, total = slice_page_text(text, 0, 100)
    assert (piece, end, total) == ("a" * 60, 62, 122)
    assert slice_page_text("x" * 500, 0, 100) == ("x" * 100, 100, 500)
    assert slice_page_text("short", 0, 100) == ("short", 5, 5)
    assert slice_page_text("short", 99, 100) == ("", 5, 5)


@pytest.mark.asyncio
async def test_page_level_trim_keeps_headline_and_date_inside_articles():
    body = (
        "<html><body><header>SITE-HEADER</header>"
        f"<article><header><h1>Post One</h1><time>2026-03-04</time></header><p>{LONG}</p></article>"
        f"<article><header><h1>Post Two</h1><time>2026-03-05</time></header><p>{LONG}</p></article>"
        "<footer>SITE-FOOTER</footer></body></html>"
    )
    result = await _fetch({}, body.encode())
    for kept in ("Post One", "2026-03-04", "Post Two", "2026-03-05"):
        assert kept in result["content"]
    assert "SITE-HEADER" not in result["content"]
    assert "SITE-FOOTER" not in result["content"]


@pytest.mark.asyncio
async def test_selected_main_is_trimmed_even_beside_a_large_sidebar():
    sidebar = "".join(f"<li>SIDEBAR-LINK-{i}</li>" for i in range(1500))
    body = (
        f"<html><body><aside><ul>{sidebar}</ul></aside>"
        f"<main><p>MAIN-TEXT {LONG}</p></main></body></html>"
    )
    result = await _fetch({}, body.encode())
    assert "MAIN-TEXT" in result["content"]
    assert "SIDEBAR-LINK" not in result["content"]


@pytest.mark.asyncio
async def test_empty_page_with_start_char_reports_past_the_end():
    result = await _fetch({"start_char": 5}, b"<html><body></body></html>")
    assert "past the end" in result["content"]
    assert "End of page" not in result["content"]
