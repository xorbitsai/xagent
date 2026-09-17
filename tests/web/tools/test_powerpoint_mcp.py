import io
import json
from unittest.mock import Mock

import pytest
import requests
from pptx import Presentation
from pptx.util import Inches

from xagent.web.tools.mcp import powerpoint


def _pptx_bytes(build_fn=None) -> bytes:
    presentation = Presentation()
    if build_fn:
        build_fn(presentation)
    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


class MockResponse:
    def __init__(self, json_data=None, status_code=200, content=None, url=None):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.content = (
            json.dumps(self._json_data).encode("utf-8") if content is None else content
        )
        self.text = self.content.decode("utf-8", errors="replace")
        self.url = url or "https://graph.microsoft.com/v1.0/example"

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Error for url: {self.url}",
                response=self,
            )


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------


def test_item_path_defaults_to_own_onedrive():
    assert (
        powerpoint._item_path("Deck.pptx", None, None) == "/me/drive/root:/Deck.pptx:"
    )


def test_content_path_appends_content():
    assert (
        powerpoint._content_path("Deck.pptx", None, None)
        == "/me/drive/root:/Deck.pptx:/content"
    )


def test_normalize_relative_path_rejects_trailing_period():
    with pytest.raises(ValueError, match="must not end with a period"):
        powerpoint._normalize_relative_path("Deck.pptx.")


def test_normalize_relative_path_rejects_dot_segments():
    with pytest.raises(ValueError, match="must not contain"):
        powerpoint._normalize_relative_path("../secret.pptx")


# ---------------------------------------------------------------------------
# slide helpers
# ---------------------------------------------------------------------------


def test_shape_text_returns_none_for_non_text_shape():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    picture_placeholder_shapes = list(slide.shapes)
    # A blank layout has no shapes at all; simulate a shape without a text
    # frame via a connector, the simplest shape type with has_text_frame=False.
    from pptx.enum.shapes import MSO_CONNECTOR

    connector = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(0), Inches(0), Inches(1), Inches(1)
    )
    assert not connector.has_text_frame
    assert powerpoint._shape_text(connector) is None
    assert picture_placeholder_shapes == []


def test_require_slide_out_of_range():
    presentation = Presentation()
    with pytest.raises(ValueError, match="out of range"):
        powerpoint._require_slide(presentation, 0)


def test_delete_slide_removes_correct_slide():
    presentation = Presentation()
    for title in ("A", "B", "C"):
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = title

    powerpoint._delete_slide(presentation, 1)

    assert [s.shapes.title.text for s in presentation.slides] == ["A", "C"]


def test_delete_slide_out_of_range():
    presentation = Presentation()
    with pytest.raises(ValueError, match="out of range"):
        powerpoint._delete_slide(presentation, 0)


def test_delete_slide_prunes_relationship_and_part():
    """Removing only the <p:sldId> entry leaves the slide's own part and its
    relationship in the saved package; _delete_slide must also drop the
    relationship so the "deleted" slide's content doesn't survive a
    save/reload round trip at all."""
    presentation = Presentation()
    for title in ("A", "B"):
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = title
    rels_before = set(presentation.part.rels)

    powerpoint._delete_slide(presentation, 0)

    rels_after = set(presentation.part.rels)
    assert rels_after < rels_before
    assert len(rels_before) - len(rels_after) == 1

    buffer = io.BytesIO()
    presentation.save(buffer)
    reloaded = Presentation(io.BytesIO(buffer.getvalue()))
    assert [s.shapes.title.text for s in reloaded.slides] == ["B"]


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_presentation_rejects_when_session_creation_conflicts(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {"error": {"code": "nameAlreadyExists"}}, status_code=409
        )
    )
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(powerpoint.powerpoint_create_presentation("Deck.pptx"))

    assert result["status"] == "error"
    assert "already exists" in result["message"]
    assert mock_request.call_args.kwargs["url"].endswith("createUploadSession")


def test_create_presentation_uploads_blank_presentation(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"uploadUrl": "https://upload.example/session"})
    )
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "new-item"}, status_code=201))
    monkeypatch.setattr(powerpoint.requests, "put", mock_put)

    result = json.loads(powerpoint.powerpoint_create_presentation("Deck.pptx"))

    assert result["status"] == "success"
    assert result["item"]["id"] == "new-item"
    session_call = mock_request.call_args
    assert session_call.kwargs["json"] == {
        "item": {"@microsoft.graph.conflictBehavior": "fail"}
    }
    put_call = mock_put.call_args
    assert put_call.args[0] == "https://upload.example/session"
    assert (
        put_call.kwargs["headers"]["Content-Type"] == powerpoint._POWERPOINT_MIME_TYPE
    )
    assert "Authorization" not in put_call.kwargs.get("headers", {})
    # Re-parses as a real, valid (if blank) pptx.
    Presentation(io.BytesIO(put_call.kwargs["data"]))


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_get_presentation_text(monkeypatch):
    def build(prs):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Title A"
        slide.placeholders[1].text_frame.text = "Body A"

    content = _pptx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(powerpoint.powerpoint_get_presentation_text("Deck.pptx"))

    assert result["status"] == "success"
    assert result["slides"] == [{"slide_index": 0, "shapes": ["Title A", "Body A"]}]


def test_get_presentation_text_rejects_non_pptx_content(monkeypatch):
    mock_request = Mock(return_value=MockResponse(content=b"not a pptx file"))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(powerpoint.powerpoint_get_presentation_text("Deck.pptx"))

    assert result["status"] == "error"
    assert "PowerPoint presentation" in result["message"]


def test_list_slides(monkeypatch):
    def build(prs):
        prs.slides.add_slide(prs.slide_layouts[1])

    content = _pptx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(powerpoint.powerpoint_list_slides("Deck.pptx"))

    assert result["status"] == "success"
    assert result["slides"][0]["slide_index"] == 0
    assert result["slides"][0]["layout_name"] == "Title and Content"


def test_get_slide_text_out_of_range(monkeypatch):
    content = _pptx_bytes()
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(powerpoint.powerpoint_get_slide_text("Deck.pptx", 0))

    assert result["status"] == "error"
    assert "out of range" in result["message"]


def test_list_slide_layouts(monkeypatch):
    content = _pptx_bytes()
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(powerpoint.powerpoint_list_slide_layouts("Deck.pptx"))

    assert result["status"] == "success"
    assert result["layouts"][1] == {"layout_index": 1, "name": "Title and Content"}


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def test_add_slide_sets_title_and_body(monkeypatch):
    content = _pptx_bytes()
    responses = iter([MockResponse(content=content), MockResponse({"id": "item-1"})])
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(
        powerpoint.powerpoint_add_slide(
            "Deck.pptx", title="New Title", body_text="New Body"
        )
    )

    assert result["status"] == "success"
    assert result["slide_index"] == 0
    put_call = mock_request.call_args_list[1]
    uploaded = Presentation(io.BytesIO(put_call.kwargs["data"]))
    slide = uploaded.slides[0]
    assert slide.shapes.title.text == "New Title"
    assert slide.placeholders[1].text_frame.text == "New Body"


def test_add_slide_rejects_out_of_range_layout():
    result = json.loads(powerpoint.powerpoint_add_slide("Deck.pptx", layout_index=999))
    assert result["status"] == "error"


def test_set_shape_text_success(monkeypatch):
    def build(prs):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Old Title"

    content = _pptx_bytes(build)
    responses = iter([MockResponse(content=content), MockResponse({"id": "item-1"})])
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(
        powerpoint.powerpoint_set_shape_text("Deck.pptx", 0, 0, "New Title")
    )

    assert result["status"] == "success"
    put_call = mock_request.call_args_list[1]
    uploaded = Presentation(io.BytesIO(put_call.kwargs["data"]))
    assert uploaded.slides[0].shapes.title.text == "New Title"


def test_set_shape_text_rejects_shape_without_text_frame(monkeypatch):
    def build(prs):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        from pptx.enum.shapes import MSO_CONNECTOR

        slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(0), Inches(0), Inches(1), Inches(1)
        )

    content = _pptx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(powerpoint.powerpoint_set_shape_text("Deck.pptx", 0, 0, "text"))

    assert result["status"] == "error"
    assert "no text frame" in result["message"]


def test_text_frame_has_dynamic_content_detects_hyperlink():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(3), Inches(1))
    run = box.text_frame.paragraphs[0].add_run()
    run.text = "Click me"
    run.hyperlink.address = "https://example.com"

    assert powerpoint._text_frame_has_dynamic_content(box.text_frame)


def test_text_frame_has_dynamic_content_false_for_plain_text():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(3), Inches(1))
    box.text_frame.text = "Plain text"

    assert not powerpoint._text_frame_has_dynamic_content(box.text_frame)


def test_set_shape_text_rejects_shape_with_hyperlink(monkeypatch):
    def build(prs):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(3), Inches(1))
        run = box.text_frame.paragraphs[0].add_run()
        run.text = "Click me"
        run.hyperlink.address = "https://example.com"

    content = _pptx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(
        powerpoint.powerpoint_set_shape_text("Deck.pptx", 0, 0, "New text")
    )

    assert result["status"] == "error"
    assert "hyperlink" in result["message"]


def test_delete_slide_tool_uploads_updated_presentation(monkeypatch):
    def build(prs):
        prs.slides.add_slide(prs.slide_layouts[1]).shapes.title.text = "A"
        prs.slides.add_slide(prs.slide_layouts[1]).shapes.title.text = "B"

    content = _pptx_bytes(build)
    responses = iter([MockResponse(content=content), MockResponse({"id": "item-1"})])
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)

    result = json.loads(powerpoint.powerpoint_delete_slide("Deck.pptx", 0))

    assert result["status"] == "success"
    put_call = mock_request.call_args_list[1]
    uploaded = Presentation(io.BytesIO(put_call.kwargs["data"]))
    assert [s.shapes.title.text for s in uploaded.slides] == ["B"]


def test_upload_presentation_rejects_oversized_content(monkeypatch):
    presentation = Presentation()
    monkeypatch.setattr(powerpoint, "_SIMPLE_UPLOAD_MAX_BYTES", 10)

    with pytest.raises(ValueError, match="MB limit"):
        powerpoint._upload_presentation(presentation, "Deck.pptx", None, None)


# ---------------------------------------------------------------------------
# error propagation
# ---------------------------------------------------------------------------


def test_missing_auth_token_is_reported(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)

    result = json.loads(powerpoint.powerpoint_get_presentation_text("Deck.pptx"))

    assert result["status"] == "error"
    assert "AUTH_TOKEN" in result["message"]
