import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import google_slides


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)


def _mock_slides_service(monkeypatch, presentations_mock):
    service = Mock()
    service.presentations.return_value = presentations_mock
    monkeypatch.setattr(google_slides, "get_slides_service", lambda: service)
    return service


def _batch_update_requests(presentations_mock):
    return presentations_mock.batchUpdate.call_args.kwargs["body"]["requests"]


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
    assert requests[0]["createSlide"]["slideLayoutReference"] == {
        "predefinedLayout": layout
    }


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
    assert "body" in result["message"]
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
    assert "body" in result["message"]
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
    assert "body" in result["message"]
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


def test_add_slide_title_layout_allows_missing_body(monkeypatch):
    """A cover slide's subtitle is optional, unlike a real BODY placeholder."""
    presentations = Mock()
    presentations.batchUpdate.return_value.execute.return_value = {}
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide("pres1", title="Cover", layout="TITLE")
    )

    assert result["status"] == "success"


def test_add_slide_blank_layout_rejects_title_and_body(monkeypatch):
    presentations = Mock()
    _mock_slides_service(monkeypatch, presentations)

    result = json.loads(
        google_slides.google_slides_add_slide(
            "pres1", title="T", body="B", layout="BLANK"
        )
    )

    assert result["status"] == "error"
    presentations.batchUpdate.assert_not_called()


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
