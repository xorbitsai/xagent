import json
import time
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


def _placeholder_element(object_id, placeholder_type, text=""):
    text_elements = [{"textRun": {"content": text}}] if text else []
    return {
        "objectId": object_id,
        "shape": {
            "placeholder": {"type": placeholder_type},
            "text": {"textElements": text_elements},
        },
    }


def _mock_presentation_get(presentations_mock, slide_id, elements):
    presentations_mock.get.return_value.execute.return_value = {
        "slides": [{"objectId": slide_id, "pageElements": elements}]
    }


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
    assert bullets_req["bulletPreset"] == "BULLET_DISC_CIRCLE_SQUARE"

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
    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="T", body=body)
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    body_object_id = _placeholder_object_id(requests[0]["createSlide"], "BODY")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == body_object_id
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

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="T", body="•First point\n-Nospace"
        )
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    body_object_id = _placeholder_object_id(requests[0]["createSlide"], "BODY")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == body_object_id
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
    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="T", body=body)
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    body_object_id = _placeholder_object_id(requests[0]["createSlide"], "BODY")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == body_object_id
    )
    # Slides infers nesting level from leading *tabs*, not spaces — plain
    # spaces would just be inserted as literal, non-nesting text — so
    # 2-space indentation must be converted to real tab characters, one
    # tab per level, for the nesting to actually render in Slides.
    assert body_text == "Top\n\tNested\n\t\tDeeper"


def test_add_slide_preserves_literal_tab_indentation(monkeypatch):
    """A caller who already typed real tab characters (rather than
    spaces) for nesting must have them preserved 1:1 — this is the input
    shape Slides' createParagraphBullets actually keys off."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    body = "- Top\n\t- Nested\n\t\t- Deeper"
    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="T", body=body)
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    body_object_id = _placeholder_object_id(requests[0]["createSlide"], "BODY")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == body_object_id
    )
    assert body_text == "Top\n\tNested\n\t\tDeeper"


def test_add_slide_mixed_tab_and_space_indentation_uses_tab_count_only(monkeypatch):
    """Documented limitation: mixing tabs and spaces in one line's leading
    whitespace is not combined — if any tab is present, the tab count
    wins and spaces in that same run are ignored."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    body = "- Top\n\t  - Mixed"
    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="T", body=body)
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    body_object_id = _placeholder_object_id(requests[0]["createSlide"], "BODY")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == body_object_id
    )
    assert body_text == "Top\n\tMixed"


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
    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="T", body=body)
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    body_object_id = _placeholder_object_id(requests[0]["createSlide"], "BODY")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == body_object_id
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

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="T", body="Point one\n•\nPoint two"
        )
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    body_object_id = _placeholder_object_id(requests[0]["createSlide"], "BODY")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == body_object_id
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


def test_strip_bullet_prefixes_bounds_work_on_a_degenerate_glued_marker_run():
    """Regression guard for a real perf/DoS finding: _strip_line used to
    re-scan the whole remaining line on every stripped marker with no
    cap, making a line of N glued "•" characters (body has no length
    limit; plausible from a runaway/malformed LLM completion) O(n^2) —
    measured at over a second for N=200_000. _MAX_MARKER_STRIPS_PER_LINE
    must keep this bounded regardless of N."""
    line = "•" * 200_000 + "Hello"

    start = time.perf_counter()
    result = google_slides._strip_bullet_prefixes(line)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert result.endswith("Hello")


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


@pytest.mark.parametrize("layout", ["TITLE_ONLY", "SECTION_HEADER"])
def test_add_slide_rejects_title_that_is_only_a_newline(monkeypatch, layout):
    """Regression guard: before the newline-collapse + .strip()-based
    guards were added together, title="\\n" was truthy under the old bare
    truthiness check and slipped past the empty-slide guard, producing a
    createSlide with no insertText for the title at all (insertion always
    used title.strip()) — a silently blank slide. Collapsing "\\n" to " "
    and then stripping it must route this through the empty-slide guard
    with its real message, not a misleading one from elsewhere."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="\n", layout=layout)
    )

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


def test_add_slide_rejects_non_string_layout_from_direct_call(monkeypatch):
    """layout's Literal typing is only enforced by FastMCP's validation
    layer, which a direct Python call bypasses entirely — a non-string
    value (e.g. an int) must still surface as a clean error rather than
    an unhandled AttributeError from calling .strip() on it."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="T", layout=123)
    )

    assert result["status"] == "error"
    assert "'layout' must be a string" in result["message"]
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


@pytest.mark.parametrize("layout", ["TITLE_ONLY", "SECTION_HEADER"])
def test_add_slide_allows_whitespace_only_body_on_layout_without_body_placeholder(
    monkeypatch, layout
):
    """Regression guard: the "no body placeholder" rejection must key off
    stripped content, matching insertion (which already treats a
    whitespace-only body as absent) — otherwise body="\\n" is hard-rejected
    even though nothing would actually have been dropped."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="T", body="\n", layout=layout
        )
    )

    assert result["status"] == "success"


def test_add_slide_allows_whitespace_only_title_on_blank_layout(monkeypatch):
    """Regression guard, symmetric with the above: BLANK has no title
    placeholder either, and a whitespace-only title must not be
    hard-rejected when it would be treated as absent anyway."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="   ", layout="BLANK")
    )

    assert result["status"] == "success"


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

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="Guide", body="- The Complete Guide", layout="TITLE"
        )
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    subtitle_object_id = _placeholder_object_id(requests[0]["createSlide"], "SUBTITLE")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == subtitle_object_id
    )
    assert body_text == "- The Complete Guide"


def test_add_slide_drops_blank_lines_on_non_bulleted_body(monkeypatch):
    """Blank-line filtering must not be limited to bulleted content — a
    blank line or trailing newline in a non-bulleted body (e.g. TITLE's
    subtitle) would otherwise leave stray blank paragraphs in the
    placeholder even though there's no floating-bullet symptom to notice."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="Guide", body="\n\nSubtitle\n", layout="TITLE"
        )
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    subtitle_object_id = _placeholder_object_id(requests[0]["createSlide"], "SUBTITLE")
    body_text = next(
        r["insertText"]["text"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == subtitle_object_id
    )
    assert body_text == "Subtitle"


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
    ("layout", "kwargs", "expected_layout"),
    [
        ("title_and_body", {"body": "detail"}, "TITLE_AND_BODY"),
        (" TITLE ", {}, "TITLE"),
        ("Title_Only", {}, "TITLE_ONLY"),
    ],
)
def test_add_slide_normalizes_layout_case_and_whitespace(
    monkeypatch, layout, kwargs, expected_layout
):
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
    assert result["layout"] == expected_layout


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


def test_add_slide_collapses_embedded_newline_in_title(monkeypatch):
    """A title is expected to be a single line — an embedded newline
    (e.g. from a caller accidentally pasting multi-line content into
    title) must be collapsed to a space rather than silently producing a
    multi-paragraph title placeholder."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    google_slides.google_slides_add_slide("pres1", title="Q3\nReview", body="detail")

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


async def test_add_slide_normalizes_layout_via_mcp_layer(monkeypatch):
    """Regression guard: `layout`'s Literal type is validated by FastMCP's
    Pydantic layer *before* the function body runs — a naive Literal
    annotation would reject a differently-cased/spaced value there,
    making the function's own layout.strip().upper() dead code for every
    real (non-test, non-direct-call) caller. `_normalize_layout` must run
    as a BeforeValidator so normalization happens ahead of the Literal
    check, not after it."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    content, _ = await google_slides.mcp.call_tool(
        "google_slides_add_slide",
        {
            "presentation_id": "pres1",
            "title": "T",
            "body": "detail",
            "layout": " title_and_body ",
        },
    )

    result = json.loads(content[0].text)
    assert result["status"] == "success"
    assert result["layout"] == "TITLE_AND_BODY"


async def test_update_slide_via_mcp_layer(monkeypatch):
    """Smoke test through the real MCP dispatch path (JSON args dict,
    positional/keyword binding via call_tool), not just a direct Python
    call — the only path add_slide's Literal-normalization bug (fixed
    above) was actually reachable through."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations, "slide1", [_placeholder_element("title_obj", "TITLE")]
    )

    content, _ = await google_slides.mcp.call_tool(
        "google_slides_update_slide",
        {"presentation_id": "pres1", "slide_id": "slide1", "title": "New title"},
    )

    result = json.loads(content[0].text)
    assert result["status"] == "success"


async def test_delete_slide_via_mcp_layer(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(presentations, "slide1", [])

    content, _ = await google_slides.mcp.call_tool(
        "google_slides_delete_slide",
        {"presentation_id": "pres1", "slide_id": "slide1"},
    )

    result = json.loads(content[0].text)
    assert result["status"] == "success"


def test_update_slide_replaces_title_and_body_and_reapplies_bullets(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [
            _placeholder_element("title_obj", "TITLE", text="Old title"),
            _placeholder_element("body_obj", "BODY", text="Old body"),
        ],
    )

    result = json.loads(
        google_slides.google_slides_update_slide(
            "pres1", "slide1", title="New title", body="• Fixed detail"
        )
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)

    delete_ids = [r["deleteText"]["objectId"] for r in requests if "deleteText" in r]
    assert set(delete_ids) == {"title_obj", "body_obj"}

    title_insert = next(
        r["insertText"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == "title_obj"
    )
    assert title_insert["text"] == "New title"

    body_insert = next(
        r["insertText"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == "body_obj"
    )
    assert body_insert["text"] == "Fixed detail"

    bullets_req = next(
        r["createParagraphBullets"] for r in requests if "createParagraphBullets" in r
    )
    assert bullets_req["objectId"] == "body_obj"

    # Order matters: batchUpdate applies requests in list order, so a
    # deleteText must precede the insertText for the same objectId (an
    # insert-then-delete would wipe out the new text instead of the old),
    # and createParagraphBullets must come after the body insertText it
    # formats.
    def _index_of(predicate):
        return next(i for i, r in enumerate(requests) if predicate(r))

    title_delete_index = _index_of(
        lambda r: r.get("deleteText", {}).get("objectId") == "title_obj"
    )
    title_insert_index = _index_of(
        lambda r: r.get("insertText", {}).get("objectId") == "title_obj"
    )
    assert title_delete_index < title_insert_index

    body_delete_index = _index_of(
        lambda r: r.get("deleteText", {}).get("objectId") == "body_obj"
    )
    body_insert_index = _index_of(
        lambda r: r.get("insertText", {}).get("objectId") == "body_obj"
    )
    bullets_index = _index_of(lambda r: "createParagraphBullets" in r)
    assert body_delete_index < body_insert_index < bullets_index


def test_update_slide_skips_delete_text_when_placeholder_already_empty(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [_placeholder_element("title_obj", "TITLE", text="")],
    )

    google_slides.google_slides_update_slide("pres1", "slide1", title="First title")

    requests = _batch_update_requests(presentations)
    assert not any("deleteText" in r for r in requests)
    assert requests[0]["insertText"]["text"] == "First title"


def test_update_slide_skips_delete_text_when_placeholder_has_only_a_newline(
    monkeypatch,
):
    """Regression guard: the Slides API commonly represents a "cleared"
    placeholder as a lone trailing "\\n" (the paragraph terminator), not a
    truly empty string — the existing-text check must treat that the same
    as empty, or it triggers a needless deleteText."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [_placeholder_element("title_obj", "TITLE", text="\n")],
    )

    google_slides.google_slides_update_slide("pres1", "slide1", title="First title")

    requests = _batch_update_requests(presentations)
    assert not any("deleteText" in r for r in requests)
    assert requests[0]["insertText"]["text"] == "First title"


def test_update_slide_still_clears_whitespace_beyond_the_bare_terminator(monkeypatch):
    """Regression guard: only the exact "" or "\\n" (the implicit
    terminator) should skip deleteText — any other whitespace-only text
    (e.g. a stray " \\n" left by a slide edited by something other than
    this tool) must still be cleared. insertText has no insertionIndex
    here, so it defaults to prepending rather than replacing; skipping
    the delete for arbitrary whitespace would leave that stale text
    merged into what's supposed to be a clean replacement."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [_placeholder_element("title_obj", "TITLE", text=" \n")],
    )

    google_slides.google_slides_update_slide("pres1", "slide1", title="First title")

    requests = _batch_update_requests(presentations)
    delete_req = next(r["deleteText"] for r in requests if "deleteText" in r)
    assert delete_req["objectId"] == "title_obj"


def test_update_slide_only_touches_the_field_provided(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [
            _placeholder_element("title_obj", "TITLE", text="Old title"),
            _placeholder_element("body_obj", "BODY", text="Old body"),
        ],
    )

    google_slides.google_slides_update_slide("pres1", "slide1", title="New title")

    requests = _batch_update_requests(presentations)
    assert not any(
        r.get("deleteText", {}).get("objectId") == "body_obj"
        or r.get("insertText", {}).get("objectId") == "body_obj"
        for r in requests
    )


def test_update_slide_subtitle_body_is_not_bulleted_or_stripped(monkeypatch):
    """A TITLE-layout slide's body lands in a SUBTITLE, not a BODY —
    update_slide must follow the same non-bulleted, non-stripped rule as
    add_slide for that placeholder type."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [
            _placeholder_element("title_obj", "CENTERED_TITLE", text="Old"),
            _placeholder_element("subtitle_obj", "SUBTITLE", text="Old subtitle"),
        ],
    )

    google_slides.google_slides_update_slide(
        "pres1", "slide1", body="- Literal dash subtitle"
    )

    requests = _batch_update_requests(presentations)
    body_insert = next(
        r["insertText"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == "subtitle_obj"
    )
    assert body_insert["text"] == "- Literal dash subtitle"
    assert not any("createParagraphBullets" in r for r in requests)


def test_update_slide_requires_at_least_one_field(monkeypatch):
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(google_slides.google_slides_update_slide("pres1", "slide1"))

    assert result["status"] == "error"
    presentations.get.assert_not_called()


def test_update_slide_rejects_whitespace_only_title(monkeypatch):
    """Regression guard, symmetric with the body-side check above:
    whitespace-only title text ("   ") must not slip past the guard just
    because it's non-empty."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_update_slide("pres1", "slide1", title="   ")
    )

    assert result["status"] == "error"
    presentations.get.assert_not_called()


def test_update_slide_rejects_whitespace_only_body(monkeypatch):
    """Regression guard: whitespace-only text ("   ", "\\n\\n") must not
    slip past the guard just because it's non-empty — that would silently
    blank real slide content, the same bug class fixed for add_slide."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_update_slide("pres1", "slide1", body="   \n\n  ")
    )

    assert result["status"] == "error"
    presentations.get.assert_not_called()


def test_update_slide_rejects_marker_only_body(monkeypatch):
    """Regression guard, same class as add_slide's: a body that's only a
    bullet marker ("•   ") passes the raw whitespace-only check (it's
    non-blank) but strips down to nothing once bullet-marker/blank-line
    processing runs — must still be rejected, not silently applied as an
    empty update."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [_placeholder_element("body_obj", "BODY", text="Old body")],
    )

    result = json.loads(
        google_slides.google_slides_update_slide("pres1", "slide1", body="•   ")
    )

    assert result["status"] == "error"
    presentations.batchUpdate.assert_not_called()


def test_update_slide_rejects_unknown_slide_id(monkeypatch):
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(presentations, "other_slide", [])

    result = json.loads(
        google_slides.google_slides_update_slide("pres1", "slide1", title="T")
    )

    assert result["status"] == "error"
    assert "slide1" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_update_slide_rejects_title_when_slide_has_no_title_placeholder(monkeypatch):
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations, "slide1", [_placeholder_element("body_obj", "BODY", text="x")]
    )

    result = json.loads(
        google_slides.google_slides_update_slide("pres1", "slide1", title="T")
    )

    assert result["status"] == "error"
    assert "title" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_update_slide_rejects_body_when_slide_has_no_body_placeholder(monkeypatch):
    """Regression guard, symmetric with the title-side test above."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations, "slide1", [_placeholder_element("title_obj", "TITLE", text="x")]
    )

    result = json.loads(
        google_slides.google_slides_update_slide("pres1", "slide1", body="Detail text")
    )

    assert result["status"] == "error"
    assert "body" in result["message"]
    presentations.batchUpdate.assert_not_called()


def test_update_slide_collapses_embedded_newline_in_title(monkeypatch):
    """Regression guard, symmetric with add_slide's fix: a title is a
    single-line placeholder — an embedded newline must collapse to a
    space instead of silently producing a multi-paragraph title."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations, "slide1", [_placeholder_element("title_obj", "TITLE")]
    )

    google_slides.google_slides_update_slide(
        "pres1", "slide1", title="Line one\nLine two"
    )

    requests = _batch_update_requests(presentations)
    title_insert = next(
        r["insertText"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == "title_obj"
    )
    assert title_insert["text"] == "Line one Line two"


def test_update_slide_normalizes_crlf_and_cr_newlines(monkeypatch):
    """Regression guard, symmetric with add_slide's fix: text pasted from
    a Windows editor may use \\r\\n or lone \\r line endings — these must
    not survive as stray \\r characters embedded in the inserted text."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [_placeholder_element("body_obj", "BODY")],
    )

    google_slides.google_slides_update_slide(
        "pres1", "slide1", body="Line one\r\nLine two\rLine three"
    )

    requests = _batch_update_requests(presentations)
    body_insert = next(
        r["insertText"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == "body_obj"
    )
    assert "\r" not in body_insert["text"]
    assert body_insert["text"] == "Line one\nLine two\nLine three"


def test_update_slide_rejects_whole_call_when_body_becomes_empty_even_with_title(
    monkeypatch,
):
    """Regression guard: a marker-only body must reject the entire call
    (no batchUpdate at all), not silently apply just the title update and
    drop the body — the two are meant to be one atomic edit."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [
            _placeholder_element("title_obj", "TITLE", text="Old title"),
            _placeholder_element("body_obj", "BODY", text="Old body"),
        ],
    )

    result = json.loads(
        google_slides.google_slides_update_slide(
            "pres1", "slide1", title="New title", body="•   "
        )
    )

    assert result["status"] == "error"
    presentations.batchUpdate.assert_not_called()


def test_update_slide_writes_title_into_centered_title_placeholder(monkeypatch):
    """A TITLE-layout slide's title lands in a CENTERED_TITLE placeholder,
    not a plain TITLE — update_slide must write to it just the same."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [_placeholder_element("title_obj", "CENTERED_TITLE", text="Old")],
    )

    result = json.loads(
        google_slides.google_slides_update_slide("pres1", "slide1", title="New title")
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    title_insert = next(
        r["insertText"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == "title_obj"
    )
    assert title_insert["text"] == "New title"


def test_update_slide_writes_body_into_object_placeholder(monkeypatch):
    """OBJECT is a real Slides placeholder type common on slides from an
    imported/non-standard-theme presentation (e.g. one converted from
    PowerPoint) — update_slide must recognize it as a body-role
    placeholder like BODY/SUBTITLE, not reject it as unrecognized."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [_placeholder_element("body_obj", "OBJECT", text="Old detail")],
    )

    result = json.loads(
        google_slides.google_slides_update_slide("pres1", "slide1", body="New detail")
    )

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    body_insert = next(
        r["insertText"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == "body_obj"
    )
    assert body_insert["text"] == "New detail"
    # OBJECT is a generic content placeholder, not the same as a real
    # bulleted BODY list — bullets are only applied for the "BODY" type.
    assert not any("createParagraphBullets" in r for r in requests)


def test_update_slide_only_targets_the_first_placeholder_of_a_duplicated_role(
    monkeypatch,
):
    """Documented, accepted limitation: a slide with two BODY placeholders
    (not reachable via this file's own add_slide, but possible for a slide
    from another source) silently targets only the first one found. Lock
    in the current behavior so a future change to iteration order isn't
    silent."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [
            _placeholder_element("body_obj_1", "BODY", text="First"),
            _placeholder_element("body_obj_2", "BODY", text="Second"),
        ],
    )

    google_slides.google_slides_update_slide("pres1", "slide1", body="New detail")

    requests = _batch_update_requests(presentations)
    assert not any(
        r.get("insertText", {}).get("objectId") == "body_obj_2"
        or r.get("deleteText", {}).get("objectId") == "body_obj_2"
        for r in requests
    )
    body_insert = next(
        r["insertText"]
        for r in requests
        if "insertText" in r and r["insertText"]["objectId"] == "body_obj_1"
    )
    assert body_insert["text"] == "New detail"


def test_update_slide_resolves_full_presentation_url(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations, "slide1", [_placeholder_element("title_obj", "TITLE")]
    )

    url = "https://docs.google.com/presentation/d/abc123/edit#slide=id.p"
    result = json.loads(
        google_slides.google_slides_update_slide(url, "slide1", title="T")
    )

    assert result["status"] == "success"
    assert presentations.get.call_args.kwargs["presentationId"] == "abc123"
    assert presentations.batchUpdate.call_args.kwargs["presentationId"] == "abc123"


def test_update_slide_returns_error_payload_on_api_failure(monkeypatch):
    presentations = Mock()
    presentations.get.return_value.execute.side_effect = RuntimeError("boom")
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_update_slide("pres1", "slide1", title="T")
    )

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_update_slide_returns_error_payload_on_batch_update_failure(monkeypatch):
    """Symmetric with the get() failure case above and with delete_slide's
    own batchUpdate-failure test — the earlier presentations().get() call
    can succeed while the actual edit still fails."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.side_effect = RuntimeError("boom")
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations, "slide1", [_placeholder_element("title_obj", "TITLE")]
    )

    result = json.loads(
        google_slides.google_slides_update_slide("pres1", "slide1", title="T")
    )

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_delete_slide_sends_delete_object_request(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(presentations, "slide1", [])

    result = json.loads(google_slides.google_slides_delete_slide("pres1", "slide1"))

    assert result["status"] == "success"
    requests = _batch_update_requests(presentations)
    assert requests == [{"deleteObject": {"objectId": "slide1"}}]


def test_delete_slide_rejects_id_that_is_not_a_slide(monkeypatch):
    """Regression guard: a placeholder shape id (e.g. one this file itself
    mints as f"{slide_id}_title") must not be silently accepted — Slides'
    deleteObject would delete just that shape while reporting success as if
    the whole slide had been removed. The mocked presentation genuinely
    contains a shape with this id — nested inside slide1's pageElements,
    not as a top-level slide — confirming _find_slide's rejection comes
    from checking only top-level slide ids (by design), not from the id
    being absent from the API response altogether."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(
        presentations,
        "slide1",
        [_placeholder_element("slide1_title", "TITLE", text="Real title")],
    )

    result = json.loads(
        google_slides.google_slides_delete_slide("pres1", "slide1_title")
    )

    assert result["status"] == "error"
    presentations.batchUpdate.assert_not_called()


def test_delete_slide_resolves_full_presentation_url(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(presentations, "slide1", [])

    url = "https://docs.google.com/presentation/d/abc123/edit#slide=id.p"
    result = json.loads(google_slides.google_slides_delete_slide(url, "slide1"))

    assert result["status"] == "success"
    assert presentations.get.call_args.kwargs["presentationId"] == "abc123"
    assert presentations.batchUpdate.call_args.kwargs["presentationId"] == "abc123"


def test_delete_slide_returns_error_payload_on_api_failure(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.side_effect = RuntimeError("boom")
    _mock_slides_service(monkeypatch, presentations)
    _mock_presentation_get(presentations, "slide1", [])

    result = json.loads(google_slides.google_slides_delete_slide("pres1", "slide1"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


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


def test_get_presentation_resolves_full_presentation_url(monkeypatch):
    presentations = Mock()
    presentations.get.return_value.execute.return_value = {
        "presentationId": "abc123",
        "title": "My Deck",
        "slides": [],
    }
    _mock_slides_service(monkeypatch, presentations)

    url = "https://docs.google.com/presentation/d/abc123/edit#slide=id.p"
    result = json.loads(google_slides.google_slides_get_presentation(url))

    assert result["status"] == "success"
    assert presentations.get.call_args.kwargs["presentationId"] == "abc123"


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


def test_batch_update_rejects_malformed_json(monkeypatch):
    """Not merely wrong-shaped JSON (a dict instead of a list, covered
    above) but genuinely invalid JSON syntax — must surface as a clean
    error payload rather than an unhandled JSONDecodeError."""
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_batch_update("pres1", "{not valid json")
    )

    assert result["status"] == "error"
    presentations.batchUpdate.assert_not_called()


def test_batch_update_resolves_full_presentation_url(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {
        "presentationId": "abc123",
        "replies": [],
    }
    _mock_slides_service(monkeypatch, presentations)

    url = "https://docs.google.com/presentation/d/abc123/edit#slide=id.p"
    result = json.loads(google_slides.google_slides_batch_update(url, "[]"))

    assert result["status"] == "success"
    assert presentations.batchUpdate.call_args.kwargs["presentationId"] == "abc123"


def test_batch_update_returns_error_payload_on_api_failure(monkeypatch):
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.side_effect = RuntimeError("boom")
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(google_slides.google_slides_batch_update("pres1", "[]"))

    assert result["status"] == "error"
    assert "boom" in result["message"]
