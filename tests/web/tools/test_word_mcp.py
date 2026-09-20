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
    def __init__(
        self,
        json_data=None,
        status_code=200,
        content=None,
        url=None,
        headers=None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.content = (
            json.dumps(self._json_data).encode("utf-8") if content is None else content
        )
        self.text = self.content.decode("utf-8", errors="replace")
        self.url = url or "https://graph.microsoft.com/v1.0/example"
        self.headers = headers if headers is not None else {}
        self.closed = False

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Error for url: {self.url}",
                response=self,
            )

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def iter_content(self, chunk_size=1):
        content = self.content
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------


def test_item_path_defaults_to_own_onedrive():
    assert word._item_path("Report.docx", None, None) == "/me/drive/root:/Report.docx:"


def test_item_path_closes_colon_for_path_shaped_site_id():
    """A "hostname:/relative-path" site id must be closed with a second
    colon before appending "/drive", per Graph's sharepoint-addressing
    docs -- otherwise "/drive" is parsed as part of the site's own
    server-relative path instead of as a sub-resource name."""
    path = word._item_path("Report.docx", "contoso.sharepoint.com:/teams/hr", None)
    assert path == ("/sites/contoso.sharepoint.com:/teams/hr:/drive/root:/Report.docx:")


def test_item_path_leaves_composite_site_id_unchanged():
    path = word._item_path("Report.docx", "contoso.sharepoint.com,site,web", None)
    assert path == ("/sites/contoso.sharepoint.com,site,web/drive/root:/Report.docx:")


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


def test_set_paragraph_text_rejects_hyperlink_nested_in_sdt():
    """A hyperlink can be wrapped in a structured document tag (<w:sdt>) --
    a direct-child-only check for <w:hyperlink> would miss it and let the
    zero-runs branch duplicate it instead of refusing."""
    from docx.oxml.shared import OxmlElement

    document = Document()
    paragraph = document.add_paragraph()
    sdt = OxmlElement("w:sdt")
    sdt_content = OxmlElement("w:sdtContent")
    hyperlink = OxmlElement("w:hyperlink")
    run_elm = OxmlElement("w:r")
    text_elm = OxmlElement("w:t")
    text_elm.text = "Click here"
    run_elm.append(text_elm)
    hyperlink.append(run_elm)
    sdt_content.append(hyperlink)
    sdt.append(sdt_content)
    paragraph._p.append(sdt)
    assert not paragraph.runs  # confirms the hyperlink run is invisible to .runs

    with pytest.raises(ValueError, match="hyperlink"):
        word._set_paragraph_text(paragraph, "New text")


def test_set_paragraph_text_rejects_tracked_insertion():
    """A <w:ins>-wrapped run's text is outside paragraph.runs, so rewriting
    only the direct-child runs would leave the tracked insertion's old text
    stale and still physically present -- verified: it stays in the saved
    XML even though it drops out of paragraph.text -- rather than replaced."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    document = Document()
    paragraph = document.add_paragraph("before ")
    ins = OxmlElement("w:ins")
    ins.set(qn("w:author"), "Alice")
    run_elm = OxmlElement("w:r")
    text_elm = OxmlElement("w:t")
    text_elm.text = "Q3 revenue"
    run_elm.append(text_elm)
    ins.append(run_elm)
    paragraph._p.append(ins)

    with pytest.raises(ValueError, match="tracked change"):
        word._set_paragraph_text(paragraph, "New text")


def test_set_paragraph_text_rejects_content_control():
    """A <w:sdt> (structured document tag / content control) holds its own
    runs the same way a hyperlink does -- invisible to paragraph.runs."""
    from docx.oxml.shared import OxmlElement

    document = Document()
    paragraph = document.add_paragraph()
    sdt = OxmlElement("w:sdt")
    sdt_content = OxmlElement("w:sdtContent")
    run_elm = OxmlElement("w:r")
    text_elm = OxmlElement("w:t")
    text_elm.text = "field value"
    run_elm.append(text_elm)
    sdt_content.append(run_elm)
    sdt.append(sdt_content)
    paragraph._p.append(sdt)

    with pytest.raises(ValueError, match="content control"):
        word._set_paragraph_text(paragraph, "New text")


@pytest.mark.parametrize(
    "tag",
    ["w:smartTag", "w:customXml", "w:fldSimple"],
)
def test_set_paragraph_text_rejects_other_run_owning_wrappers(tag):
    """w:smartTag (a legacy Office smart tag), w:customXml (a custom XML
    markup region), and w:fldSimple (a simple field's cached result, e.g.
    PAGE/DATE/a TOC entry) all hold their own runs outside paragraph.runs
    the same way a hyperlink or content control does -- verified directly:
    the wrapped text is still physically present in the saved XML after a
    rewrite, stale alongside the new text."""
    from docx.oxml.shared import OxmlElement

    document = Document()
    paragraph = document.add_paragraph()
    wrapper = OxmlElement(tag)
    run_elm = OxmlElement("w:r")
    text_elm = OxmlElement("w:t")
    text_elm.text = "wrapped value"
    run_elm.append(text_elm)
    wrapper.append(run_elm)
    paragraph._p.append(wrapper)
    assert not paragraph.runs  # confirms the wrapped run is invisible to .runs

    with pytest.raises(ValueError):
        word._set_paragraph_text(paragraph, "New text")


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


def test_run_has_non_text_content_rejects_unlisted_element_kinds():
    """_run_has_non_text_content treats any child other than <w:rPr>/<w:t>
    as unsafe, not just the specific tags this module happens to test --
    <w:noBreakHyphen> and <w:sym> are silently destroyed by Run.text just
    like an image or field character, verified directly against
    python-docx's own CT_R.text setter, but neither has a dedicated test
    above (nor is either tag hardcoded anywhere in the guard itself)."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    for tag, attrs in [
        ("w:noBreakHyphen", {}),
        ("w:sym", {"w:font": "Wingdings", "w:char": "F0E0"}),
    ]:
        document = Document()
        paragraph = document.add_paragraph()
        run = paragraph.add_run("A")
        elm = OxmlElement(tag)
        for name, value in attrs.items():
            elm.set(qn(name), value)
        run._r.append(elm)

        assert word._run_has_non_text_content(run), tag


def test_set_paragraph_text_rejects_run_with_tab():
    """A <w:tab> is not itself corrupted by Run.text (python-docx's setter
    reconstructs it from a literal \\t), but the guard is deliberately
    conservative -- it only special-cases <w:rPr>/<w:t> as known-safe
    rather than trying to keep an allowlist of every element that happens
    to round-trip correctly -- so a run with a tab is refused too."""
    document = Document()
    paragraph = document.add_paragraph()
    run = paragraph.add_run()
    run.add_tab()

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


def test_create_only_upload_does_not_chain_secret_bearing_exception(monkeypatch):
    """raise ... from exc would still attach the original, URL-bearing
    exception as __cause__ even though the raised message itself is
    sanitized -- a future traceback/log/APM capture could surface it.
    from None must be used instead, matching onedrive.py's precedent."""
    secret_url = "https://upload.example/session?token=super-secret-value"
    mock_request = Mock(return_value=MockResponse({"uploadUrl": secret_url}))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({}, status_code=500, url=secret_url))
    monkeypatch.setattr(word.requests, "put", mock_put)

    with pytest.raises(word._GraphRequestError) as exc_info:
        word._create_only_upload(b"content", "Report.docx", None, None)

    assert exc_info.value.__cause__ is None


def test_create_only_upload_connection_error_does_not_chain_exception(monkeypatch):
    secret_url = "https://upload.example/session?token=super-secret-value"
    mock_request = Mock(return_value=MockResponse({"uploadUrl": secret_url}))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(
        side_effect=requests.ConnectionError(f"Connection refused: {secret_url}")
    )
    monkeypatch.setattr(word.requests, "put", mock_put)

    with pytest.raises(RuntimeError) as exc_info:
        word._create_only_upload(b"content", "Report.docx", None, None)

    assert exc_info.value.__cause__ is None


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


def test_get_document_text_streams_with_identity_encoding(monkeypatch):
    content = _docx_bytes(lambda d: d.add_paragraph("hello"))
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    word.word_get_document_text("Report.docx")

    _, kwargs = mock_request.call_args
    assert kwargs["stream"] is True
    assert kwargs["headers"]["Accept-Encoding"] == "identity"


def test_get_document_text_rejects_declared_content_length_over_limit(monkeypatch):
    oversized = word._MAX_DOWNLOAD_BYTES + 1
    mock_request = Mock(
        return_value=MockResponse(
            content=b"irrelevant",
            headers={"Content-Length": str(oversized)},
        )
    )
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_get_document_text("Report.docx"))

    assert result["status"] == "error"
    assert "too large" in result["message"]


def test_get_document_text_rejects_streamed_body_over_limit(monkeypatch):
    # No (or a wrong) Content-Length header must not bypass the cap -- the
    # streamed byte count is the source of truth.
    oversized_content = b"x" * (word._MAX_DOWNLOAD_BYTES + 1)
    mock_request = Mock(return_value=MockResponse(content=oversized_content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_get_document_text("Report.docx"))

    assert result["status"] == "error"
    assert "too large" in result["message"]


def test_get_document_text_rejects_compressed_content_encoding(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            content=b"compressed-bytes",
            headers={"Content-Encoding": "gzip"},
        )
    )
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_get_document_text("Report.docx"))

    assert result["status"] == "error"
    assert "compressed" in result["message"]


def test_get_document_text_closes_connection_on_rejection(monkeypatch):
    # A stream=True response left open on a rejection path holds its
    # connection until garbage collection instead of returning it to the
    # pool immediately -- every rejection branch in _read_capped_content
    # must close the response, not just a fully-consumed one.
    response = MockResponse(
        content=b"compressed-bytes", headers={"Content-Encoding": "gzip"}
    )
    mock_request = Mock(return_value=response)
    monkeypatch.setattr(word.requests, "request", mock_request)

    word.word_get_document_text("Report.docx")

    assert response.closed is True


def test_get_document_text_download_error_does_not_leak_redirect_url(monkeypatch):
    """Graph's /content redirects to a pre-authenticated download URL --
    itself a bearer secret. A connection failure while following that
    redirect embeds the URL in requests' own exception message, so it must
    not be stringified into the returned error."""
    secret_url = "https://download.example/blob?token=super-secret-value"

    def side_effect(*args, **kwargs):
        if kwargs.get("params") == {"$select": "eTag"}:
            return MockResponse({})
        raise requests.ConnectionError(f"Connection refused: {secret_url}")

    mock_request = Mock(side_effect=side_effect)
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_get_document_text("Report.docx"))

    assert result["status"] == "error"
    assert "super-secret-value" not in result["message"]
    assert secret_url not in result["message"]


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
            MockResponse({"eTag": '"abc"'}),
            MockResponse(content=content),
            MockResponse({"uploadUrl": "https://upload.example/session"}),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "item-1"}, status_code=200))
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_set_paragraph_text("Report.docx", 0, "New text"))

    assert result["status"] == "success"
    session_call = mock_request.call_args_list[2]
    assert session_call.kwargs["headers"]["If-Match"] == '"abc"'
    uploaded = Document(io.BytesIO(mock_put.call_args.kwargs["data"]))
    assert uploaded.paragraphs[0].text == "New text"


def test_append_paragraph(monkeypatch):
    content = _docx_bytes()
    responses = iter(
        [
            MockResponse({"eTag": '"abc"'}),
            MockResponse(content=content),
            MockResponse({"uploadUrl": "https://upload.example/session"}),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_append_paragraph("Report.docx", "New paragraph"))

    assert result["status"] == "success"
    uploaded = Document(io.BytesIO(mock_put.call_args.kwargs["data"]))
    assert uploaded.paragraphs[-1].text == "New paragraph"


def test_add_heading_validates_level():
    result = json.loads(word.word_add_heading("Report.docx", "Title", level=10))
    assert result["status"] == "error"
    assert "level" in result["message"]


def test_add_heading_uploads_updated_document(monkeypatch):
    content = _docx_bytes()
    responses = iter(
        [
            MockResponse({}),
            MockResponse(content=content),
            MockResponse({"uploadUrl": "https://upload.example/session"}),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_add_heading("Report.docx", "Section", level=2))

    assert result["status"] == "success"
    uploaded = Document(io.BytesIO(mock_put.call_args.kwargs["data"]))
    assert uploaded.paragraphs[-1].text == "Section"
    assert uploaded.paragraphs[-1].style.name == "Heading 2"


def test_replace_text_counts_matches_within_runs(monkeypatch):
    def build(d):
        p = d.add_paragraph()
        p.add_run("foo bar foo")

    content = _docx_bytes(build)
    responses = iter(
        [
            MockResponse({}),
            MockResponse(content=content),
            MockResponse({"uploadUrl": "https://upload.example/session"}),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "success"
    assert result["replacements"] == 2
    uploaded = Document(io.BytesIO(mock_put.call_args.kwargs["data"]))
    assert uploaded.paragraphs[0].text == "baz bar baz"


def test_replace_text_rejects_empty_find():
    result = json.loads(word.word_replace_text("Report.docx", "", "x"))
    assert result["status"] == "error"


def test_replace_text_skips_upload_when_nothing_matches(monkeypatch):
    """A no-op request must not still create a new document version --
    uploading an unmodified document has no reason to happen."""
    content = _docx_bytes(lambda d: d.add_paragraph("nothing relevant here"))
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "absent", "x"))

    assert result["status"] == "success"
    assert result["replacements"] == 0
    assert "item" not in result
    # Only the metadata + content download GETs happened -- no upload.
    assert mock_request.call_count == 2


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
    # Only the metadata + content download GETs happened -- no upload.
    assert mock_request.call_count == 2


def test_replace_text_rejects_match_inside_content_control(monkeypatch):
    """A match inside a <w:sdt>'s run must be refused, not silently
    rewritten -- doing so would desync the control's data binding, which
    this tool has no way to update to match."""
    from docx.oxml.shared import OxmlElement

    def build(d):
        p = d.add_paragraph()
        sdt = OxmlElement("w:sdt")
        sdt_content = OxmlElement("w:sdtContent")
        run_elm = OxmlElement("w:r")
        text_elm = OxmlElement("w:t")
        text_elm.text = "foo bar"
        run_elm.append(text_elm)
        sdt_content.append(run_elm)
        sdt.append(sdt_content)
        p._p.append(sdt)

    content = _docx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "error"
    assert "content control" in result["message"]
    # Only the metadata + content download GETs happened -- no upload.
    assert mock_request.call_count == 2


def test_replace_text_rejects_match_inside_simple_field(monkeypatch):
    """A <w:fldSimple>'s cached result (e.g. a PAGE/DATE field or a TOC
    entry) must be refused -- rewriting the cached text leaves the field's
    own instruction unchanged, so Word's next refresh can revert it."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    def build(d):
        p = d.add_paragraph()
        fld_simple = OxmlElement("w:fldSimple")
        fld_simple.set(qn("w:instr"), "PAGE")
        run_elm = OxmlElement("w:r")
        text_elm = OxmlElement("w:t")
        text_elm.text = "foo bar"
        run_elm.append(text_elm)
        fld_simple.append(run_elm)
        p._p.append(fld_simple)

    content = _docx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "error"
    assert "field" in result["message"]
    # Only the metadata + content download GETs happened -- no upload.
    assert mock_request.call_count == 2


@pytest.mark.parametrize("tag", ["w:smartTag", "w:customXml"])
def test_replace_text_rejects_match_inside_smart_tag_or_custom_xml(monkeypatch, tag):
    """word_set_paragraph_text already refuses a paragraph containing
    w:smartTag/w:customXml -- word_replace_text's own, separate per-run
    guard must refuse a match inside one too, not just w:sdt/w:fldSimple."""
    from docx.oxml.shared import OxmlElement

    def build(d):
        p = d.add_paragraph()
        wrapper = OxmlElement(tag)
        run_elm = OxmlElement("w:r")
        text_elm = OxmlElement("w:t")
        text_elm.text = "foo bar"
        run_elm.append(text_elm)
        wrapper.append(run_elm)
        p._p.append(wrapper)

    content = _docx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "error"
    assert mock_request.call_count == 2


def test_replace_text_rejects_match_inside_complex_field_result(monkeypatch):
    """A complex field (begin/instrText/separate/result/end run sequence)
    has no single wrapper element -- its cached result run must still be
    recognized and refused via the fldChar state machine."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    def build(d):
        p = d.add_paragraph()

        def add_run_with(*children):
            run_elm = OxmlElement("w:r")
            for child in children:
                run_elm.append(child)
            p._p.append(run_elm)

        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        add_run_with(begin)

        instr = OxmlElement("w:instrText")
        instr.text = " PAGEREF _Toc123 "
        add_run_with(instr)

        separate = OxmlElement("w:fldChar")
        separate.set(qn("w:fldCharType"), "separate")
        add_run_with(separate)

        result_text = OxmlElement("w:t")
        result_text.text = "foo bar"
        add_run_with(result_text)

        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        add_run_with(end)

    content = _docx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "error"
    assert "field" in result["message"]
    # Only the metadata + content download GETs happened -- no upload.
    assert mock_request.call_count == 2


def test_replace_text_rejects_match_inside_field_spanning_paragraphs(monkeypatch):
    """A complex field commonly opens in one paragraph and closes several
    paragraphs later -- a multi-entry table of contents has its own begin/
    separate in the first entry's paragraph, one entry per paragraph, and
    end in the last. Tracking field state fresh per paragraph call would
    lose protection for every entry after the first; it must be threaded
    across the whole document instead."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    def add_run_with(paragraph, *children):
        run_elm = OxmlElement("w:r")
        for child in children:
            run_elm.append(child)
        paragraph._p.append(run_elm)

    def build(d):
        p0 = d.add_paragraph()
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        add_run_with(p0, begin)
        instr = OxmlElement("w:instrText")
        instr.text = " TOC "
        add_run_with(p0, instr)
        separate = OxmlElement("w:fldChar")
        separate.set(qn("w:fldCharType"), "separate")
        add_run_with(p0, separate)

        p1 = d.add_paragraph()
        entry_text = OxmlElement("w:t")
        entry_text.text = "foo bar"
        add_run_with(p1, entry_text)

        p2 = d.add_paragraph()
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        add_run_with(p2, end)

    content = _docx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "error"
    assert "field" in result["message"]


def test_replace_text_edits_ordinary_text_after_a_field_ends(monkeypatch):
    """State must not leak past a field's own "end" marker -- a paragraph
    with nothing but ordinary text, appearing after a completed field,
    must remain freely editable."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    def add_run_with(paragraph, *children):
        run_elm = OxmlElement("w:r")
        for child in children:
            run_elm.append(child)
        paragraph._p.append(run_elm)

    def build(d):
        p0 = d.add_paragraph()
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        add_run_with(p0, begin)
        separate = OxmlElement("w:fldChar")
        separate.set(qn("w:fldCharType"), "separate")
        add_run_with(p0, separate)
        result_text = OxmlElement("w:t")
        result_text.text = "cached result"
        add_run_with(p0, result_text)
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        add_run_with(p0, end)

        d.add_paragraph("foo bar")

    content = _docx_bytes(build)
    responses = iter(
        [
            MockResponse({}),
            MockResponse(content=content),
            MockResponse({"uploadUrl": "https://upload.example/session"}),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "success"
    assert result["replacements"] == 1


def test_replace_text_rejects_match_in_outer_field_result_after_nested_field(
    monkeypatch,
):
    """A field nested inside another field's result region (e.g. a TOC
    entry's page number is itself a nested PAGEREF field) must not let its
    own "end" prematurely un-protect the outer field's remaining result --
    a single boolean flag gets this wrong; a depth stack doesn't."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    def add_run_with(paragraph, *children):
        run_elm = OxmlElement("w:r")
        for child in children:
            run_elm.append(child)
        paragraph._p.append(run_elm)

    def fld_char(fld_type):
        elm = OxmlElement("w:fldChar")
        elm.set(qn("w:fldCharType"), fld_type)
        return elm

    def text_run(text):
        elm = OxmlElement("w:t")
        elm.text = text
        return elm

    def build(d):
        p = d.add_paragraph()
        add_run_with(p, fld_char("begin"))  # outer begin
        add_run_with(p, fld_char("separate"))  # outer separate
        add_run_with(p, fld_char("begin"))  # nested begin
        add_run_with(p, fld_char("separate"))  # nested separate
        add_run_with(p, text_run("nested result"))
        add_run_with(p, fld_char("end"))  # nested end
        add_run_with(p, text_run("foo bar"))  # outer's own trailing result
        add_run_with(p, fld_char("end"))  # outer end

    content = _docx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "error"
    assert "field" in result["message"]


def test_replace_text_ignores_field_state_from_tracked_insertion(monkeypatch):
    """A complex field entirely inside a tracked-change insertion is
    correctly skipped (it's not visible through this module's read tools
    either) -- but its fldChar markers must not leak field-protected state
    onto ordinary text later in the same paragraph."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    def build(d):
        p = d.add_paragraph()
        ins = OxmlElement("w:ins")
        for fld_type in ("begin", "separate"):
            run_elm = OxmlElement("w:r")
            fld = OxmlElement("w:fldChar")
            fld.set(qn("w:fldCharType"), fld_type)
            run_elm.append(fld)
            ins.append(run_elm)
        result_run = OxmlElement("w:r")
        result_text = OxmlElement("w:t")
        result_text.text = "tracked field result"
        result_run.append(result_text)
        ins.append(result_run)
        end_run = OxmlElement("w:r")
        end_fld = OxmlElement("w:fldChar")
        end_fld.set(qn("w:fldCharType"), "end")
        end_run.append(end_fld)
        ins.append(end_run)
        p._p.append(ins)
        p.add_run("foo bar")

    content = _docx_bytes(build)
    responses = iter(
        [
            MockResponse({}),
            MockResponse(content=content),
            MockResponse({"uploadUrl": "https://upload.example/session"}),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "success"
    assert result["replacements"] == 1


def test_replace_text_finds_match_inside_hyperlink_run(monkeypatch):
    """paragraph.runs excludes runs nested inside <w:hyperlink>. Iterating
    only paragraph.runs would silently skip a match inside hyperlink text
    even though word_get_document_text surfaces that same text."""
    from docx.oxml.shared import OxmlElement

    def build(d):
        p = d.add_paragraph("before ")
        hyperlink = OxmlElement("w:hyperlink")
        run_elm = OxmlElement("w:r")
        text_elm = OxmlElement("w:t")
        text_elm.text = "foo bar"
        run_elm.append(text_elm)
        hyperlink.append(run_elm)
        p._p.append(hyperlink)
        p.add_run(" after")

    content = _docx_bytes(build)
    responses = iter(
        [
            MockResponse({}),
            MockResponse(content=content),
            MockResponse({"uploadUrl": "https://upload.example/session"}),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(word.requests, "put", mock_put)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "success"
    assert result["replacements"] == 1
    uploaded = Document(io.BytesIO(mock_put.call_args.kwargs["data"]))
    assert uploaded.paragraphs[0].text == "before baz bar after"


def test_replace_text_ignores_match_inside_tracked_insertion(monkeypatch):
    """A <w:ins>-wrapped run holds someone else's pending, unaccepted
    tracked-change insertion. A plain recursive run search would reach it
    (unlike paragraph.runs, which excludes it) and silently rewrite that
    person's edit with no new revision recorded -- must not touch it."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    def build(d):
        p = d.add_paragraph("before ")
        ins = OxmlElement("w:ins")
        ins.set(qn("w:author"), "Alice")
        run_elm = OxmlElement("w:r")
        text_elm = OxmlElement("w:t")
        text_elm.text = "foo bar"
        run_elm.append(text_elm)
        ins.append(run_elm)
        p._p.append(ins)
        p.add_run(" after")

    content = _docx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "success"
    assert result["replacements"] == 0
    # No match means no change -- only the download GET happened.
    # Only the metadata + content download GETs happened -- no upload.
    assert mock_request.call_count == 2


def test_replace_text_ignores_match_inside_text_box(monkeypatch):
    """A text box's content is a separate stream anchored to (not part of)
    the run holding its <w:drawing> -- invisible to word_get_document_text
    and word_list_paragraphs, so word_replace_text must not silently edit
    it either."""
    from docx.oxml.shared import OxmlElement

    def build(d):
        p = d.add_paragraph("outer text")
        anchor = p.add_run()
        drawing = OxmlElement("w:drawing")
        txbx_content = OxmlElement("w:txbxContent")
        inner_p = OxmlElement("w:p")
        inner_run = OxmlElement("w:r")
        inner_text = OxmlElement("w:t")
        inner_text.text = "foo bar"
        inner_run.append(inner_text)
        inner_p.append(inner_run)
        txbx_content.append(inner_p)
        drawing.append(txbx_content)
        anchor._r.append(drawing)

    content = _docx_bytes(build)
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_replace_text("Report.docx", "foo", "baz"))

    assert result["status"] == "success"
    assert result["replacements"] == 0
    # No match means no change -- only the download GET happened.
    # Only the metadata + content download GETs happened -- no upload.
    assert mock_request.call_count == 2


def test_set_paragraph_text_ignores_hyperlink_inside_text_box():
    """A hyperlink nested inside a text box isn't part of this paragraph's
    own visible flow (word_get_document_text/word_list_paragraphs never
    show it), so it must not trigger the hyperlink-refusal guard -- though
    the paragraph is still refused for the real reason: the drawing-holding
    run itself is non-text content."""
    from docx.oxml.shared import OxmlElement

    document = Document()
    paragraph = document.add_paragraph("caption ")
    anchor = paragraph.add_run()
    drawing = OxmlElement("w:drawing")
    txbx_content = OxmlElement("w:txbxContent")
    inner_p = OxmlElement("w:p")
    hyperlink = OxmlElement("w:hyperlink")
    inner_run = OxmlElement("w:r")
    inner_text = OxmlElement("w:t")
    inner_text.text = "Click here"
    inner_run.append(inner_text)
    hyperlink.append(inner_run)
    inner_p.append(hyperlink)
    txbx_content.append(inner_p)
    drawing.append(txbx_content)
    anchor._r.append(drawing)

    with pytest.raises(ValueError, match="non-text content"):
        word._set_paragraph_text(paragraph, "New caption")


def test_upload_document_rejects_oversized_content(monkeypatch):
    document = Document()
    monkeypatch.setattr(word, "_MAX_UPLOAD_BYTES", 10)

    with pytest.raises(ValueError, match="MB limit"):
        word._upload_document(document, "Report.docx", None, None)


def test_upload_document_sends_if_match_when_etag_given(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"uploadUrl": "https://upload.example/session"})
    )
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(word.requests, "put", mock_put)

    word._upload_document(Document(), "Report.docx", None, None, etag='"abc123"')

    session_call = mock_request.call_args
    assert session_call.kwargs["headers"]["If-Match"] == '"abc123"'


def test_upload_document_omits_if_match_when_no_etag(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"uploadUrl": "https://upload.example/session"})
    )
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(word.requests, "put", mock_put)

    word._upload_document(Document(), "Report.docx", None, None, etag=None)

    session_call = mock_request.call_args
    assert "If-Match" not in session_call.kwargs["headers"]


def test_upload_document_rejects_stale_etag_at_session_creation(monkeypatch):
    """A 412 at createUploadSession means someone else changed the file
    since it was downloaded for this edit -- surfaced as a clear conflict,
    not the generic upload-failure message."""
    mock_request = Mock(
        return_value=MockResponse({"error": {"code": "..."}}, status_code=412)
    )
    monkeypatch.setattr(word.requests, "request", mock_request)

    with pytest.raises(ValueError, match="changed by someone else"):
        word._upload_document(Document(), "Report.docx", None, None, etag='"stale"')


def test_upload_document_rejects_stale_etag_at_content_put(monkeypatch):
    """Graph can also reject the conditional write at the content PUT step
    rather than session creation, depending on timing -- both paths must
    surface the same clear conflict message."""
    mock_request = Mock(
        return_value=MockResponse({"uploadUrl": "https://upload.example/session"})
    )
    monkeypatch.setattr(word.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({}, status_code=412))
    monkeypatch.setattr(word.requests, "put", mock_put)

    with pytest.raises(ValueError, match="changed by someone else"):
        word._upload_document(Document(), "Report.docx", None, None, etag='"stale"')


def test_download_document_returns_etag_from_metadata(monkeypatch):
    content = _docx_bytes()
    responses = iter([MockResponse({"eTag": '"xyz"'}), MockResponse(content=content)])
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)

    _document, etag = word._download_document("Report.docx", None, None)

    assert etag == '"xyz"'
    metadata_call = mock_request.call_args_list[0]
    assert metadata_call.kwargs["params"] == {"$select": "eTag"}


def test_download_document_tolerates_missing_etag(monkeypatch):
    content = _docx_bytes()
    responses = iter([MockResponse({}), MockResponse(content=content)])
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(word.requests, "request", mock_request)

    _document, etag = word._download_document("Report.docx", None, None)

    assert etag is None


def test_download_document_skips_etag_metadata_when_not_needed(monkeypatch):
    """word_get_document_text/word_list_paragraphs never upload, so they
    have no use for the etag -- need_etag=False must skip that extra Graph
    round trip entirely rather than fetching and discarding it."""
    content = _docx_bytes()
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    _document, etag = word._download_document(
        "Report.docx", None, None, need_etag=False
    )

    assert etag is None
    assert mock_request.call_count == 1


def test_get_document_text_does_not_fetch_etag(monkeypatch):
    content = _docx_bytes(lambda d: d.add_paragraph("hello"))
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_get_document_text("Report.docx"))

    assert result["status"] == "success"
    # Only the content GET happened -- no separate eTag metadata GET.
    assert mock_request.call_count == 1


def test_list_paragraphs_does_not_fetch_etag(monkeypatch):
    content = _docx_bytes(lambda d: d.add_paragraph("hello"))
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(word.requests, "request", mock_request)

    result = json.loads(word.word_list_paragraphs("Report.docx"))

    assert result["status"] == "success"
    assert mock_request.call_count == 1


# ---------------------------------------------------------------------------
# error propagation
# ---------------------------------------------------------------------------


def test_missing_auth_token_is_reported(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)

    result = json.loads(word.word_get_document_text("Report.docx"))

    assert result["status"] == "error"
    assert "AUTH_TOKEN" in result["message"]
