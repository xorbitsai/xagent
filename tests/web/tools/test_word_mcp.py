import io
import json
from unittest.mock import Mock

import pytest
import requests
from docx import Document

from xagent.web.tools.mcp import word


def _docx_bytes(build_fn=None) -> bytes:
    document = Document()
    if build_fn:
        build_fn(document)
    buffer = io.BytesIO()
    document.save(buffer)
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
    assert word._item_path("Report.docx", None, None) == "/me/drive/root:/Report.docx:"


def test_content_path_appends_content():
    assert (
        word._content_path("Report.docx", None, None)
        == "/me/drive/root:/Report.docx:/content"
    )


def test_normalize_relative_path_rejects_trailing_period():
    with pytest.raises(ValueError, match="must not end with a period"):
        word._normalize_relative_path("Report.docx.")


def test_normalize_relative_path_rejects_dot_segments():
    with pytest.raises(ValueError, match="must not contain"):
        word._normalize_relative_path("../secret.docx")


def test_set_paragraph_text_adds_run_when_empty():
    document = Document()
    paragraph = document.add_paragraph()
    assert not paragraph.runs
    word._set_paragraph_text(paragraph, "Hello")
    assert paragraph.text == "Hello"


def test_set_paragraph_text_reuses_first_run():
    document = Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("A")
    paragraph.add_run("B")
    paragraph.add_run("C")
    word._set_paragraph_text(paragraph, "Replaced")
    assert paragraph.text == "Replaced"


def test_set_paragraph_text_rejects_hyperlink_paragraph():
    """A hyperlink's own run lives inside <w:hyperlink>, outside
    paragraph.runs -- word._set_paragraph_text must refuse rather than
    appending a duplicate plain-text run alongside the untouched link."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    document = Document()
    paragraph = document.add_paragraph()
    hyperlink = OxmlElement("w:hyperlink")
    run_elm = OxmlElement("w:r")
    text_elm = OxmlElement("w:t")
    text_elm.text = "Click here"
    run_elm.append(text_elm)
    hyperlink.append(run_elm)
    paragraph._p.append(hyperlink)
    assert not paragraph.runs  # confirms the hyperlink run is invisible to .runs

    with pytest.raises(ValueError, match="hyperlink"):
        word._set_paragraph_text(paragraph, "New text")

    assert paragraph._p.findall(qn("w:hyperlink"))  # untouched, not duplicated


def test_set_paragraph_text_rejects_run_with_image():
    import base64

    from docx.shared import Inches

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    )
    document = Document()
    paragraph = document.add_paragraph("caption ")
    picture_run = paragraph.add_run()
    picture_run.add_picture(io.BytesIO(png_bytes), width=Inches(0.1))

    with pytest.raises(ValueError, match="non-text content"):
        word._set_paragraph_text(paragraph, "New text")


def test_set_paragraph_text_rejects_run_with_page_break():
    from docx.enum.text import WD_BREAK

    document = Document()
    paragraph = document.add_paragraph()
    run = paragraph.add_run()
    run.add_break(WD_BREAK.PAGE)

    with pytest.raises(ValueError, match="non-text content"):
        word._set_paragraph_text(paragraph, "New text")


# ---------------------------------------------------------------------------
# create / existence check
# ---------------------------------------------------------------------------


def test_create_document_rejects_when_session_creation_conflicts(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {"error": {"code": "nameAlreadyExists"}}, status_code=409
        )
    )
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_create_document("Report.docx"))

    assert result["status"] == "error"
    assert "already exists" in result["message"]
    assert mock_request.call_args.kwargs["url"].endswith("createUploadSession")


def test_create_document_rejects_when_upload_put_conflicts(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"uploadUrl": "https://upload.example/session"})
    )
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(
        return_value=MockResponse(
            {"error": {"code": "nameAlreadyExists"}}, status_code=409
        )
    )
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_create_document("Report.docx"))

    assert result["status"] == "error"
    assert "already exists" in result["message"]


def test_create_document_upload_failure_does_not_leak_session_url(monkeypatch):
    """The upload-session URL is a pre-authenticated bearer secret (a token
    in its own query string) -- a failed upload must never let that URL
    reach the caller through the error message, whether the failure comes
    from raise_for_status() or a lower-level connection/timeout error."""
    secret_url = "https://upload.example/session?token=super-secret-value"
    mock_request = Mock(return_value=MockResponse({"uploadUrl": secret_url}))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({}, status_code=500, url=secret_url))
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_create_document("Report.docx"))

    assert result["status"] == "error"
    assert "super-secret-value" not in result["message"]
    assert secret_url not in result["message"]


def test_create_document_upload_connection_error_does_not_leak_session_url(
    monkeypatch,
):
    secret_url = "https://upload.example/session?token=super-secret-value"
    mock_request = Mock(return_value=MockResponse({"uploadUrl": secret_url}))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(
        side_effect=requests.ConnectionError(f"Connection refused: {secret_url}")
    )
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_create_document("Report.docx"))

    assert result["status"] == "error"
    assert "super-secret-value" not in result["message"]
    assert secret_url not in result["message"]


def test_create_document_uploads_blank_document(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"uploadUrl": "https://upload.example/session"})
    )
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "new-item"}, status_code=201))
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_create_document("Report.docx"))

    assert result["status"] == "success"
    assert result["item"]["id"] == "new-item"
    session_call = mock_request.call_args
    assert session_call.kwargs["url"].endswith("createUploadSession")
    assert session_call.kwargs["json"] == {
        "item": {"@microsoft.graph.conflictBehavior": "fail"}
    }
    put_call = mock_put.call_args
    assert put_call.args[0] == "https://upload.example/session"
    assert put_call.kwargs["headers"]["Content-Type"] == word._WORD_MIME_TYPE
    assert "Authorization" not in put_call.kwargs.get("headers", {})
    # Re-parses as a real, valid (if blank) docx.
    Document(io.BytesIO(put_call.kwargs["data"]))


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_get_document_text_joins_paragraphs(monkeypatch):
    content = _docx_bytes(
        lambda d: (d.add_paragraph("First"), d.add_paragraph("Second"))
    )
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_get_document_text("Report.docx"))

    assert result["status"] == "success"
    assert result["text"] == "First\nSecond"
    assert result["table_count"] == 0


def test_get_document_text_rejects_non_docx_content(monkeypatch):
    mock_request = Mock(return_value=MockResponse(content=b"not a docx file"))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_get_document_text("Report.docx"))

    assert result["status"] == "error"
    assert "Word document" in result["message"]


def test_list_paragraphs_includes_style(monkeypatch):
    content = _docx_bytes(lambda d: d.add_heading("Title", level=1))
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_list_paragraphs("Report.docx"))

    assert result["status"] == "success"
    assert result["paragraphs"] == [{"index": 0, "text": "Title", "style": "Heading 1"}]


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def test_set_paragraph_text_out_of_range(monkeypatch):
    content = _docx_bytes(lambda d: d.add_paragraph("Only paragraph"))
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_set_paragraph_text("Report.docx", 5, "New text"))

    assert result["status"] == "error"
    assert "out of range" in result["message"]


def test_set_paragraph_text_uploads_updated_document(monkeypatch):
    content = _docx_bytes(lambda d: d.add_paragraph("Old text"))
    responses = iter(
        [
            MockResponse(content=content),
            MockResponse({"id": "item-1"}, status_code=200),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_set_paragraph_text("Report.docx", 0, "New text"))

    assert result["status"] == "success"
    put_call = mock_request.call_args_list[1]
    uploaded = Document(io.BytesIO(put_call.kwargs["data"]))
    assert uploaded.paragraphs[0].text == "New text"


def test_append_paragraph(monkeypatch):
    content = _docx_bytes()
    responses = iter([MockResponse(content=content), MockResponse({"id": "item-1"})])
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_append_paragraph("Report.docx", "New paragraph"))

    assert result["status"] == "success"
    put_call = mock_request.call_args_list[1]
    uploaded = Document(io.BytesIO(put_call.kwargs["data"]))
    assert uploaded.paragraphs[-1].text == "New paragraph"


def test_add_heading_validates_level():
    result = json.loads(word.word_add_heading("Report.docx", "Title", level=10))
    assert result["status"] == "error"
    assert "level" in result["message"]


def test_add_heading_uploads_updated_document(monkeypatch):
    content = _docx_bytes()
    responses = iter([MockResponse(content=content), MockResponse({"id": "item-1"})])
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_add_heading("Report.docx", "Section", level=2))

    assert result["status"] == "success"
    put_call = mock_request.call_args_list[1]
    uploaded = Document(io.BytesIO(put_call.kwargs["data"]))
    assert uploaded.paragraphs[-1].text == "Section"
    assert uploaded.paragraphs[-1].style.name == "Heading 2"


def test_replace_text_counts_matches_within_runs(monkeypatch):
    def build(d):
        p = d.add_paragraph()
        p.add_run("foo bar foo")

    content = _docx_bytes(build)
    responses = iter([MockResponse(content=content), MockResponse({"id": "item-1"})])
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "success"
    assert result["replacements"] == 2
    put_call = mock_request.call_args_list[1]
    uploaded = Document(io.BytesIO(put_call.kwargs["data"]))
    assert uploaded.paragraphs[0].text == "baz bar baz"


def test_replace_text_rejects_empty_find():
    result = json.loads(word.word_replace_text("Report.docx", "", "x"))
    assert result["status"] == "error"


def test_replace_text_rejects_match_in_run_with_image(monkeypatch):
    """A match inside a run that also holds a drawing must be refused, not
    silently applied -- run.text = ... would delete the drawing too."""
    import base64

    from docx.shared import Inches

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    )

    def build(d):
        p = d.add_paragraph()
        run = p.add_run("foo bar")
        run.add_picture(io.BytesIO(png_bytes), width=Inches(0.1))

    content = _docx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "error"
    assert "non-text content" in result["message"]
    # Only the download GET happened -- no upload was attempted.
    assert mock_request.call_count == 1


def test_upload_document_rejects_oversized_content(monkeypatch):
    document = Document()
    monkeypatch.setattr(word, "_SIMPLE_UPLOAD_MAX_BYTES", 10)

    with pytest.raises(ValueError, match="MB limit"):
        word._upload_document(document, "Report.docx", None, None)


# ---------------------------------------------------------------------------
# error propagation
# ---------------------------------------------------------------------------


def test_missing_auth_token_is_reported(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)

    result = json.loads(word.word_get_document_text("Report.docx"))

    assert result["status"] == "error"
    assert "AUTH_TOKEN" in result["message"]
