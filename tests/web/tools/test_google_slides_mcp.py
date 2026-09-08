import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import google_slides


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    # Every test below replaces get_slides_service() wholesale via
    # _mock_slides_service, so nothing here exercises real credential
    # building — this just guards against a stray, unmocked call blowing up
    # with a confusing "missing env var" error instead of the actual assertion.
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")


def _mock_slides_service(monkeypatch, presentations_mock):
    service = Mock()
    service.presentations.return_value = presentations_mock
    monkeypatch.setattr(google_slides, "get_slides_service", lambda: service)
    return service


def _batch_update_requests(presentations_mock):
    return presentations_mock.batchUpdate.call_args.kwargs["body"]["requests"]


def _placeholder_object_id(create_slide, placeholder_type):
    return next(
        m["objectId"]
        for m in create_slide["placeholderIdMappings"]
        if m["layoutPlaceholder"]["type"] == placeholder_type
    )


def test_add_slide_default_layout_creates_title_and_body_with_bullets(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="Q3 Pipeline Highlights", body="Line one\nLine two"
        )
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)

    create_slide = requests[0]["createSlide"]
    assert create_slide["slideLayoutReference"] == {
        "predefinedLayout": "TITLE_AND_BODY"
    }
    mappings = {
        m["layoutPlaceholder"]["type"]: m["objectId"]
        for m in create_slide["placeholderIdMappings"]
    }
    assert set(mappings) == {"TITLE", "BODY"}

    title_req = next(
        r["insertText"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == mappings["TITLE"]
    )
    assert title_req["text"] == "Q3 Pipeline Highlights"
    body_req = next(
        r["insertText"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == mappings["BODY"]
    )
    assert body_req["text"] == "Line one\nLine two"

    bullets_req = next(
        r["createParagraphBullets"] for r in requests if "createParagraphBullets" in r
    )
    assert bullets_req["objectId"] == mappings["BODY"]
    assert bullets_req["textRange"] == {"type": "ALL"}

    # createParagraphBullets must come after the insertText that populates
    # the body — applying it to an empty range fails against the real API.
    body_insert_index = next(
        i
        for i, r in enumerate(requests)
        if "insertText" in r and r["insertText"]["objectId"] == mappings["BODY"]
    )
    bullets_index = next(
        i for i, r in enumerate(requests) if "createParagraphBullets" in r
    )
    assert bullets_index > body_insert_index


def test_add_slide_strips_literal_bullet_markers_before_inserting(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    body = "• First point\n- Second point\n* Third point\nFourth point (no marker)"
    google_slides.google_slides_add_slide("pres1", title="T", body=body)

    requests = _batch_update_requests(presentations)
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and "First point" in r["insertText"]["text"]
    )
    assert body_text == (
        "First point\nSecond point\nThird point\nFourth point (no marker)"
    )


def test_add_slide_strips_marker_glued_to_text_without_trailing_space(monkeypatch):
    """A marker with no space after it (e.g. "•First", "-Nospace") must
    still be recognized — otherwise the literal marker survives next to
    Slides' own bullet glyph, a visible double bullet."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    google_slides.google_slides_add_slide(
        "pres1", title="T", body="•First point\n-Nospace"
    )

    requests = _batch_update_requests(presentations)
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and "First point" in r["insertText"]["text"]
    )
    assert body_text == "First point\nNospace"


def test_add_slide_preserves_nested_bullet_indentation(monkeypatch):
    """Slides infers bullet nesting level from leading whitespace/tabs in
    the inserted text; stripping the marker must not also strip the
    indentation in front of it, or multi-level bullets flatten to one
    level."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    body = "- Top\n  - Nested\n    - Deeper"
    google_slides.google_slides_add_slide("pres1", title="T", body=body)

    requests = _batch_update_requests(presentations)
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and "Top" in r["insertText"]["text"]
    )
    # Slides infers nesting level from leading *tabs*, not spaces — plain
    # spaces would just be inserted as literal, non-nesting text — so
    # 2-space indentation must be converted to real tab characters, one
    # tab per level, for the nesting to actually render in Slides.
    assert body_text == "Top\n\tNested\n\t\tDeeper"


def test_add_slide_does_not_corrupt_non_bullet_content_starting_with_marker_chars(
    monkeypatch,
):
    """ "-"/"*" have common non-bullet meanings (a negative number's sign,
    a currency figure, a decimal, an em-/en-dash, markdown emphasis) — a
    leading marker glued to a digit, symbol, or another marker character,
    or a bare "*" with no trailing space, must be left alone rather than
    silently mangled. This is a regression guard for a real bug: an
    earlier fix for "•First" (marker glued to a word) also silently
    stripped the sign off negative figures like "-$5m loss"."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    body = (
        "-5% growth\n-$5m loss\n-.5% decline\n-- Author Name\n-– Author\n"
        "**Note\n*emphasis* not a bullet"
    )
    google_slides.google_slides_add_slide("pres1", title="T", body=body)

    requests = _batch_update_requests(presentations)
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and "growth" in r["insertText"]["text"]
    )
    assert body_text == body


def test_add_slide_strips_bare_marker_with_nothing_after_it(monkeypatch):
    """A line that's only a bullet marker with nothing after it at all
    (e.g. a stray "•" or "-" on its own line) previously wasn't recognized
    by the stripping regex (which required a character or whitespace
    after the marker), so it survived as literal text right next to
    Slides' own bullet glyph — a visible double bullet."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    google_slides.google_slides_add_slide(
        "pres1", title="T", body="Point one\n•\nPoint two"
    )

    requests = _batch_update_requests(presentations)
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and "Point one" in r["insertText"]["text"]
    )
    assert body_text == "Point one\nPoint two"


def test_add_slide_fully_unwraps_a_line_with_more_than_one_leading_marker(monkeypatch):
    """ "• - nested" must not leave a residual "-" as literal text inside a
    paragraph Slides is about to bullet — strip leading markers repeatedly
    until none remain, not just the outermost one."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    google_slides.google_slides_add_slide("pres1", title="T", body="• - nested")

    requests = _batch_update_requests(presentations)
    body_object_id = _placeholder_object_id(requests[0]["createSlide"], "BODY")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == body_object_id
    )
    assert body_text == "nested"


def test_add_slide_drops_blank_lines_and_trailing_newline_from_bulleted_body(
    monkeypatch,
):
    """A blank line inside body, or a trailing newline (very common in
    generated text), would otherwise become an empty paragraph that still
    gets createParagraphBullets applied to it — a visible floating bullet
    with no text next to it."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    google_slides.google_slides_add_slide(
        "pres1", title="T", body="Point one\n\nPoint two\n"
    )

    requests = _batch_update_requests(presentations)
    body_object_id = _placeholder_object_id(requests[0]["createSlide"], "BODY")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == body_object_id
    )
    assert body_text == "Point one\nPoint two"


def test_add_slide_rejects_marker_only_body_for_content_layout(monkeypatch):
    """A body that's only a bullet marker ("•   ") strips to an empty
    string before insertion — the body-required guard must check the
    post-stripping text, not the raw string, or this recreates the
    content-less-slide bug via a different input shape."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="T", body="•   ", layout="TITLE_AND_BODY"
        )
    )

    assert result["status"] == "error"
    assert "expects body content but none was provided" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_add_slide_title_layout_uses_subtitle_and_skips_bullets(monkeypatch):
    """The TITLE (cover) layout maps body to a SUBTITLE placeholder, which
    should not get a createParagraphBullets request — a subtitle line isn't a
    bulleted list."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1",
            title="Q3 2026 Sales & CRM Review",
            body="Pipeline Updates, Key Wins, and Q4 Strategy",
            layout="TITLE",
        )
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)

    create_slide = requests[0]["createSlide"]
    assert create_slide["slideLayoutReference"] == {"predefinedLayout": "TITLE"}
    mappings = {
        m["layoutPlaceholder"]["type"]: m["objectId"]
        for m in create_slide["placeholderIdMappings"]
    }
    assert set(mappings) == {"CENTERED_TITLE", "SUBTITLE"}
    assert not any("createParagraphBullets" in r for r in requests)


@pytest.mark.parametrize("layout", ["TITLE_ONLY", "SECTION_HEADER"])
def test_add_slide_title_only_layouts_need_no_body(monkeypatch, layout):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="Section Break", layout=layout
        )
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    create_slide = requests[0]["createSlide"]
    assert create_slide["slideLayoutReference"] == {"predefinedLayout": layout}
    mappings = {
        m["layoutPlaceholder"]["type"] for m in create_slide["placeholderIdMappings"]
    }
    assert mappings == {"TITLE"}


@pytest.mark.parametrize("layout", ["TITLE_ONLY", "SECTION_HEADER"])
def test_add_slide_rejects_empty_title_for_title_only_layout(monkeypatch, layout):
    """Regression guard, symmetric with the body-required check: a layout
    whose only content slot is the title must not silently create a fully
    empty slide when title is also omitted."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(google_slides.google_slides_add_slide("pres1", layout=layout))

    assert result["status"] == "error"
    assert "completely empty slide" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_add_slide_rejects_completely_empty_title_layout(monkeypatch):
    """The TITLE (cover) layout accepts an empty body (subtitle is
    optional), but title and body can't both be empty — that's a
    completely blank cover slide."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(google_slides.google_slides_add_slide("pres1", layout="TITLE"))

    assert result["status"] == "error"
    assert "completely empty slide" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_add_slide_skips_whitespace_only_title_insertion(monkeypatch):
    """A whitespace-only title on a content layout (where body carries the
    real content) must not be inserted verbatim — leave the placeholder
    empty instead of filling it with whitespace."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    google_slides.google_slides_add_slide(
        "pres1", title="   ", body="Real content", layout="TITLE_AND_BODY"
    )

    requests = _batch_update_requests(presentations)
    title_object_id = _placeholder_object_id(requests[0]["createSlide"], "TITLE")
    assert not any(
        "insertText" in r and r["insertText"]["objectId"] == title_object_id
        for r in requests
    )


def test_add_slide_rejects_unknown_layout(monkeypatch):
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="T", layout="TWO_COLUMNS")
    )

    assert result["status"] == "error"
    assert "TWO_COLUMNS" in result["message"]
    presentations.batchUpdate.assert_not_called()


@pytest.mark.parametrize("layout", ["TITLE_ONLY", "SECTION_HEADER"])
def test_add_slide_rejects_body_on_layout_without_body_placeholder(monkeypatch, layout):
    """Regression guard: these layouts have no body placeholder, so silently
    accepting `body` would drop it exactly like the reported bug — reject the
    call instead so the caller finds out immediately."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="T", body="This would be lost", layout=layout
        )
    )

    assert result["status"] == "error"
    assert "has no body placeholder" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_add_slide_rejects_missing_body_for_content_layout(monkeypatch):
    """Regression guard for the reported bug: a content layout (a real BODY
    placeholder) must not silently create a slide with no detail text."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="Q3 Pipeline Highlights", layout="TITLE_AND_BODY"
        )
    )

    assert result["status"] == "error"
    assert "expects body content but none was provided" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_add_slide_rejects_whitespace_only_body_for_content_layout(monkeypatch):
    """Regression guard: a whitespace-only body ("   ", "\\n\\n") must not
    slip past the "body required" check just because it's non-empty — that
    would recreate the exact content-less-slide bug the check exists for."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1",
            title="Q3 Pipeline Highlights",
            body="   \n\n  ",
            layout="TITLE_AND_BODY",
        )
    )

    assert result["status"] == "error"
    assert "expects body content but none was provided" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_add_slide_title_layout_preserves_literal_dash_in_subtitle(monkeypatch):
    """Regression guard: bullet-marker stripping must only apply to text
    that's actually being turned into a bulleted list (a real BODY
    placeholder) — a SUBTITLE is never bulleted, so a literal leading "-"
    the caller intended as part of the subtitle text must survive."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    google_slides.google_slides_add_slide(
        "pres1", title="Guide", body="- The Complete Guide", layout="TITLE"
    )

    requests = _batch_update_requests(presentations)
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and "Complete Guide" in r["insertText"]["text"]
    )
    assert body_text == "- The Complete Guide"


def test_add_slide_skips_whitespace_only_body_insertion_on_title_layout(monkeypatch):
    """Regression guard, symmetric with the whitespace-only-title fix: a
    whitespace-only body on a layout where body is optional (e.g. TITLE's
    subtitle) must not be inserted verbatim — leave the placeholder
    untouched instead of writing invisible whitespace into it."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="Cover", body="   ", layout="TITLE"
        )
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    subtitle_object_id = _placeholder_object_id(requests[0]["createSlide"], "SUBTITLE")
    assert not any(
        "insertText" in r and r["insertText"]["objectId"] == subtitle_object_id
        for r in requests
    )


def test_add_slide_title_layout_allows_missing_body(monkeypatch):
    """A cover slide's subtitle is optional, unlike a real BODY placeholder."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="Cover", layout="TITLE")
    )

    assert result["status"] == "success"


def test_add_slide_blank_layout_rejects_title_alone(monkeypatch):
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="T", layout="BLANK")
    )

    assert result["status"] == "error"
    assert "has no title placeholder" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_add_slide_blank_layout_rejects_body_alone(monkeypatch):
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide("pres1", body="B", layout="BLANK")
    )

    assert result["status"] == "error"
    assert "has no body placeholder" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_add_slide_blank_layout_allows_empty_and_omits_placeholder_mappings(
    monkeypatch,
):
    """BLANK is meant to be empty (custom content is added afterwards via
    google_slides_batch_update); the empty-slide guard must not reject it,
    and createSlide should omit placeholderIdMappings rather than sending
    an empty list."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(google_slides.google_slides_add_slide("pres1", layout="BLANK"))

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    assert "placeholderIdMappings" not in requests[0]["createSlide"]


def test_add_slide_resolves_full_presentation_url(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    url = "https://docs.google.com/presentation/d/abc123/edit#slide=id.p"
    result = json.loads(
        google_slides.google_slides_add_slide(url, title="T", body="detail line")
    )

    assert result["status"] == "success"
    assert presentations.batchUpdate.call_args.kwargs["presentationId"] == "abc123"


def test_add_slide_returns_error_payload_on_api_failure(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.side_effect = RuntimeError("boom")
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="T", body="detail")
    )

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_add_slide_returns_error_payload_without_ascii_escaping(monkeypatch):
    """_error() must use ensure_ascii=False like every success payload in
    this file, or non-ASCII error text (e.g. from a caller's own input
    echoed back, or a non-ASCII exception message) gets mangled into
    \\uXXXX escapes instead of staying human-readable."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.side_effect = RuntimeError(
        "读取失败 — 😀"
    )
    _mock_slides_service(monkeypatch, presentations)

    raw = google_slides.google_slides_add_slide("pres1", title="T", body="detail")

    assert "\\u" not in raw
    assert "读取失败" in raw
    assert json.loads(raw)["status"] == "error"


@pytest.mark.parametrize(
    ("layout", "kwargs"),
    [
        ("title_and_body", {"body": "detail"}),
        (" TITLE ", {}),
        ("Title_Only", {}),
    ],
)
def test_add_slide_normalizes_layout_case_and_whitespace(monkeypatch, layout, kwargs):
    """An LLM caller may not reproduce the exact enum casing/spacing —
    normalize before rejecting as unknown."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="T", layout=layout, **kwargs
        )
    )

    assert result["status"] == "success"
    assert result["layout"] == layout.strip().upper()


def test_add_slide_echoes_the_effective_layout_on_success(monkeypatch):
    """The caller passed no explicit layout, relying on the default — the
    response should confirm which layout was actually applied."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="T", body="detail")
    )

    assert result["layout"] == "TITLE_AND_BODY"


def test_add_slide_strips_padding_whitespace_from_title_before_inserting(monkeypatch):
    """Unlike body (whose leading whitespace is meaningful for bullet
    nesting), a title has no reason to keep incidental leading/trailing
    padding — insert the trimmed text, not the raw string."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    google_slides.google_slides_add_slide("pres1", title="  Q3 Review  ", body="detail")

    requests = _batch_update_requests(presentations)
    title_object_id = _placeholder_object_id(requests[0]["createSlide"], "TITLE")
    title_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == title_object_id
    )
    assert title_text == "Q3 Review"


def test_add_slide_normalizes_crlf_and_cr_newlines(monkeypatch):
    """Text pasted from a Windows editor may use \\r\\n or lone \\r line
    endings; leaving stray \\r characters embedded in the inserted text is
    a latent artifact even though it doesn't break bullet-marker matching
    (which only anchors on line starts)."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    google_slides.google_slides_add_slide(
        "pres1", title="T", body="Line one\r\nLine two\rLine three"
    )

    requests = _batch_update_requests(presentations)
    body_object_id = _placeholder_object_id(requests[0]["createSlide"], "BODY")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == body_object_id
    )
    assert "\r" not in body_text
    assert body_text == "Line one\nLine two\nLine three"


async def test_add_slide_rejects_unknown_layout_via_mcp_layer(monkeypatch):
    """layout is typed as a Literal enum so FastMCP validates it before the
    function body ever runs — exercise the real call path, which direct
    function calls (used by every other test in this file) bypass."""
    from mcp.server.fastmcp.exceptions import ToolError

    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    with pytest.raises(ToolError, match="validation error"):
        await google_slides.mcp.call_tool(
            "google_slides_add_slide",
            {"presentation_id": "pres1", "title": "T", "layout": "TWO_COLUMNS"},
        )
    presentations.batchUpdate.assert_not_called()


def test_get_presentation_returns_slide_summaries(monkeypatch):
    presentations = Mock()
    presentations.get.return_value.execute.return_value = {
        "presentationId": "pres1",
        "title": "My Deck",
        "slides": [
            {
                "objectId": "slide1",
                "pageElements": [
                    {
                        "objectId": "shape1",
                        "shape": {
                            "text": {
                                "textElements": [{"textRun": {"content": "Hello"}}]
                            }
                        },
                    }
                ],
            }
        ],
    }
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(google_slides.google_slides_get_presentation("pres1"))

    assert result["status"] == "success"
    assert result["title"] == "My Deck"
    assert result["slide_count"] == 1
    assert result["slides"][0]["text"] == ["Hello"]


def test_get_presentation_returns_error_payload_on_api_failure(monkeypatch):
    presentations = Mock()
    presentations.get.return_value.execute.side_effect = RuntimeError("boom")
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(google_slides.google_slides_get_presentation("pres1"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_create_presentation_returns_link_and_id(monkeypatch):
    presentations = Mock()
    presentations.create.return_value.execute.return_value = {
        "presentationId": "pres1",
        "title": "New Deck",
    }
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(google_slides.google_slides_create_presentation("New Deck"))

    assert result["status"] == "success"
    assert result["presentation_id"] == "pres1"
    assert result["link"] == "https://docs.google.com/presentation/d/pres1/edit"


def test_create_presentation_returns_error_payload_on_api_failure(monkeypatch):
    presentations = Mock()
    presentations.create.return_value.execute.side_effect = RuntimeError("boom")
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(google_slides.google_slides_create_presentation("New Deck"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_batch_update_forwards_requests_and_returns_replies(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {
        "presentationId": "pres1",
        "replies": [{"createShape": {"objectId": "shape1"}}],
    }
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_batch_update(
            "pres1", '[{"createShape": {"objectId": "shape1"}}]'
        )
    )

    assert result["status"] == "success"
    assert result["replies"] == [{"createShape": {"objectId": "shape1"}}]
    sent_requests = presentations.batchUpdate.call_args.kwargs["body"]["requests"]
    assert sent_requests == [{"createShape": {"objectId": "shape1"}}]


def test_batch_update_rejects_non_list_json(monkeypatch):
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_batch_update("pres1", '{"not": "a list"}')
    )

    assert result["status"] == "error"
    presentations.batchUpdate.assert_not_called()


def test_batch_update_returns_error_payload_on_api_failure(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.side_effect = RuntimeError("boom")
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(google_slides.google_slides_batch_update("pres1", "[]"))

    assert result["status"] == "error"
    assert "boom" in result["message"]
