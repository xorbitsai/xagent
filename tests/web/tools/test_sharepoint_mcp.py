import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import sharepoint


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

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Error for url: {self.url}",
                response=self,
            )

    def iter_content(self, chunk_size=1):
        content = self.content
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")


@pytest.fixture(autouse=True)
def _upload_allowed_dirs_env(tmp_path, monkeypatch):
    allowed_dir = tmp_path / "workspace"
    allowed_dir.mkdir()
    monkeypatch.setenv("XAGENT_SHAREPOINT_FILE_ALLOWED_DIRS", str(allowed_dir))
    return allowed_dir


# ---------------------------------------------------------------------------
# pagination
# ---------------------------------------------------------------------------


def test_graph_paginate_follows_next_link_until_limit_reached(monkeypatch):
    next_link = "https://graph.microsoft.com/v1.0/sites/root/lists?$skiptoken=abc"
    responses = [
        MockResponse(
            {"value": [{"id": "1"}, {"id": "2"}], "@odata.nextLink": next_link}
        ),
        MockResponse({"value": [{"id": "3"}, {"id": "4"}]}),
    ]
    mock_request = Mock(side_effect=responses)
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    items, truncated = sharepoint._graph_paginate("/sites/root/lists", {}, limit=10)

    assert [item["id"] for item in items] == ["1", "2", "3", "4"]
    assert truncated is False
    # The second call must GET the nextLink verbatim -- not re-prefix it
    # with GRAPH_BASE_URL, which would double up the host/path.
    assert mock_request.call_args_list[1].kwargs["url"] == next_link


def test_graph_paginate_stops_and_reports_truncated_at_limit(monkeypatch):
    next_link = "https://graph.microsoft.com/v1.0/sites/root/lists?$skiptoken=abc"
    mock_request = Mock(
        return_value=MockResponse(
            {"value": [{"id": "1"}, {"id": "2"}], "@odata.nextLink": next_link}
        )
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    items, truncated = sharepoint._graph_paginate("/sites/root/lists", {}, limit=1)

    assert [item["id"] for item in items] == ["1"]
    assert truncated is True
    # The limit was already reached by the first page -- no nextLink fetch.
    assert mock_request.call_count == 1


def test_graph_paginate_not_truncated_when_collection_exactly_exhausted(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"value": [{"id": "1"}, {"id": "2"}]})
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    items, truncated = sharepoint._graph_paginate("/sites/root/lists", {}, limit=2)

    assert [item["id"] for item in items] == ["1", "2"]
    assert truncated is False


def test_success_with_capped_list_passes_through_small_list():
    result = json.loads(
        sharepoint._success_with_capped_list("items", [{"id": "1"}], truncated=False)
    )
    assert result["status"] == "success"
    assert result["items"] == [{"id": "1"}]
    assert result["truncated"] is False


def test_success_with_capped_list_halves_until_it_fits(monkeypatch):
    monkeypatch.setattr(sharepoint, "get_tool_max_output_length", lambda: 200)
    items = [{"id": str(i), "padding": "x" * 50} for i in range(10)]

    result = json.loads(
        sharepoint._success_with_capped_list("items", items, truncated=False)
    )

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["items"]) < len(items)
    assert "did not fit" in result["message"]


def test_list_list_items_reports_truncated_across_pages(monkeypatch):
    next_link = (
        "https://graph.microsoft.com/v1.0/sites/root/lists/Tasks/items?$skiptoken=abc"
    )
    responses = [
        MockResponse({"value": [{"id": "1"}], "@odata.nextLink": next_link}),
        MockResponse({"value": [{"id": "2"}]}),
    ]
    mock_request = Mock(side_effect=responses)
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_list_list_items("root", "Tasks", top=2))

    assert result["status"] == "success"
    assert [item["id"] for item in result["items"]] == ["1", "2"]
    assert result["truncated"] is False


# ---------------------------------------------------------------------------
# path/id helpers
# ---------------------------------------------------------------------------


def test_site_segment_rejects_empty():
    with pytest.raises(ValueError, match="site_id is required"):
        sharepoint._site_segment("")


def test_site_segment_rejects_dot_segments():
    with pytest.raises(ValueError, match="must not contain"):
        sharepoint._site_segment("contoso.sharepoint.com:/../etc")


def test_site_segment_preserves_colon_slash_comma():
    assert (
        sharepoint._site_segment("contoso.sharepoint.com:/sites/team")
        == "contoso.sharepoint.com:/sites/team"
    )
    assert (
        sharepoint._site_segment("contoso.sharepoint.com,abc-123,def-456")
        == "contoso.sharepoint.com,abc-123,def-456"
    )


def test_drive_base_defaults_to_site_default_drive():
    assert sharepoint._drive_base("root", None) == "/sites/root/drive"
    assert sharepoint._drive_base("root", "drive-1") == "/sites/root/drives/drive-1"


def test_drive_children_path_root_vs_folder():
    assert (
        sharepoint._drive_children_path("root", None, None)
        == "/sites/root/drive/root/children"
    )
    assert (
        sharepoint._drive_children_path("root", "Docs/Reports", None)
        == "/sites/root/drive/root:/Docs/Reports:/children"
    )


def test_drive_content_path_rejects_trailing_slash():
    with pytest.raises(ValueError, match="must include a filename"):
        sharepoint._drive_content_path("root", "Docs/", None)


def test_drive_content_path_rejects_dot_segments():
    with pytest.raises(ValueError, match="must not contain"):
        sharepoint._drive_content_path("root", "../secret.txt", None)


def test_drive_content_path_rejects_trailing_period():
    with pytest.raises(ValueError, match="must not end with a period"):
        sharepoint._drive_content_path("root", "report.docx.", None)


# ---------------------------------------------------------------------------
# sites
# ---------------------------------------------------------------------------


def test_search_sites_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"value": [{"id": "site-1", "name": "Team"}]})
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_search_sites("team"))

    assert result["status"] == "success"
    assert result["sites"] == [{"id": "site-1", "name": "Team"}]
    assert mock_request.call_args.kwargs["params"]["search"] == "team"


def test_search_sites_requires_query():
    result = json.loads(sharepoint.sharepoint_search_sites("   "))
    assert result["status"] == "error"


def test_get_root_site_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "root-site"}))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_root_site())

    assert result["status"] == "success"
    assert result["site"]["id"] == "root-site"
    assert mock_request.call_args.kwargs["url"].endswith("/sites/root")


def test_list_drives_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"value": [{"id": "drive-1", "name": "Documents"}]})
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_list_drives("root"))

    assert result["status"] == "success"
    assert result["drives"] == [{"id": "drive-1", "name": "Documents"}]
    assert result["truncated"] is False
    # Matching every other pagination call site: $top must be sent so
    # Graph's own default page size doesn't force extra round trips to
    # reach the 200-item cap this call already asks _graph_paginate for.
    assert mock_request.call_args.kwargs["params"]["$top"] == 200


def test_search_sites_reports_truncated_across_pages(monkeypatch):
    next_link = "https://graph.microsoft.com/v1.0/sites?$skiptoken=abc"
    responses = [
        MockResponse({"value": [{"id": "site-1"}], "@odata.nextLink": next_link}),
        MockResponse({"value": [{"id": "site-2"}]}),
    ]
    mock_request = Mock(side_effect=responses)
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_search_sites("team", top=2))

    assert result["status"] == "success"
    assert [s["id"] for s in result["sites"]] == ["site-1", "site-2"]
    assert result["truncated"] is False


# ---------------------------------------------------------------------------
# document library items
# ---------------------------------------------------------------------------


def test_list_items_strips_download_url(monkeypatch):
    download_url = "https://download.example/secret-token"
    mock_request = Mock(
        return_value=MockResponse(
            {
                "value": [
                    {
                        "id": "item-1",
                        "name": "report.pdf",
                        "@microsoft.graph.downloadUrl": download_url,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_list_items("root"))

    assert result["status"] == "success"
    item = result["items"][0]
    assert item["id"] == "item-1"
    assert "@microsoft.graph.downloadUrl" not in item


def test_search_files_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"value": [{"id": "item-1", "name": "report.pdf"}]})
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_search_files("root", "report"))

    assert result["status"] == "success"
    assert result["items"] == [{"id": "item-1", "name": "report.pdf"}]
    assert result["truncated"] is False


def test_search_files_requires_query():
    result = json.loads(sharepoint.sharepoint_search_files("root", "   "))
    assert result["status"] == "error"


def test_get_file_content_decodes_text(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(content=b"hello world", status_code=200)
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "notes.txt"))

    assert result["status"] == "success"
    assert result["text_content"] == "hello world"
    assert result["encoding"] == "utf-8"


def test_get_file_content_falls_back_to_base64_for_binary(monkeypatch):
    binary_content = b"\xff\xd8\xff\xe0binarydata"
    mock_request = Mock(return_value=MockResponse(content=binary_content))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "photo.jpg"))

    assert result["status"] == "success"
    assert result["encoding"] == "base64"


def test_get_file_content_streams_with_identity_encoding(monkeypatch):
    mock_request = Mock(return_value=MockResponse(content=b"hello world"))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    sharepoint.sharepoint_get_file_content("root", "notes.txt")

    _, kwargs = mock_request.call_args
    assert kwargs["stream"] is True
    assert kwargs["headers"]["Accept-Encoding"] == "identity"


def test_get_file_content_rejects_declared_content_length_over_limit(monkeypatch):
    oversized = sharepoint._MAX_DOWNLOAD_BYTES + 1
    mock_request = Mock(
        return_value=MockResponse(
            content=b"irrelevant",
            headers={"Content-Length": str(oversized)},
        )
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "huge.bin"))

    assert result["status"] == "error"
    assert "too large" in result["message"]


def test_get_file_content_rejects_streamed_body_over_limit(monkeypatch):
    # No (or a wrong) Content-Length header must not bypass the cap -- the
    # streamed byte count is the source of truth.
    oversized_content = b"x" * (sharepoint._MAX_DOWNLOAD_BYTES + 1)
    mock_request = Mock(return_value=MockResponse(content=oversized_content))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "huge.bin"))

    assert result["status"] == "error"
    assert "too large" in result["message"]


def test_get_file_content_rejects_raw_content_over_output_limit(monkeypatch):
    # Below _MAX_DOWNLOAD_BYTES (10MB) but well above a small output-length
    # limit -- rejected before _decode_bytes/_success ever run on it.
    monkeypatch.setattr(sharepoint, "get_tool_max_output_length", lambda: 100)
    mock_request = Mock(return_value=MockResponse(content=b"x" * 200))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "medium.bin"))

    assert result["status"] == "error"
    assert "too large" in result["message"]


def test_get_file_content_rejects_encoded_result_over_output_limit(monkeypatch):
    # Raw content itself is under the output limit, but base64 inflates it
    # (4/3x) past the limit -- must still be caught by the second check.
    monkeypatch.setattr(sharepoint, "get_tool_max_output_length", lambda: 100)
    binary_content = b"\xff\xd8\xff\xe0" + b"x" * 70  # 74 bytes, base64 ~100+ chars
    mock_request = Mock(return_value=MockResponse(content=binary_content))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "photo.jpg"))

    assert result["status"] == "error"
    assert "too large" in result["message"]


def test_get_file_content_accepts_multibyte_text_over_byte_count_limit(monkeypatch):
    # UTF-8 decoding SHRINKS byte count to character count for multi-byte
    # text (each of these Chinese characters is 3 raw bytes but 1 decoded
    # char) -- a byte-count-based early rejection would wrongly reject this
    # file even though the actual JSON response comfortably fits.
    monkeypatch.setattr(sharepoint, "get_tool_max_output_length", lambda: 500)
    content = ("中" * 200).encode("utf-8")  # 600 raw bytes, 200 chars
    mock_request = Mock(return_value=MockResponse(content=content))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "notes_zh.txt"))

    assert result["status"] == "success"
    assert result["text_content"] == "中" * 200


def test_get_file_content_rejects_compressed_content_encoding(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            content=b"compressed-bytes",
            headers={"Content-Encoding": "gzip"},
        )
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "notes.txt"))

    assert result["status"] == "error"
    assert "compressed" in result["message"]


def test_get_file_content_redacts_url_on_http_error(monkeypatch):
    # Graph's /content redirects to a short-lived preauthenticated download
    # URL; a failure on that final host must not leak it (it's a bearer
    # credential in its own right) into the returned error.
    sas_url = (
        "https://contoso.sharepoint.com/_layouts/download.aspx"
        "?sastoken=SECRET-CREDENTIAL"
    )
    mock_request = Mock(
        return_value=MockResponse(status_code=404, url=sas_url, content=b"")
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "notes.txt"))

    assert result["status"] == "error"
    assert "404" in result["message"]
    assert "SECRET-CREDENTIAL" not in result["message"]
    assert "contoso.sharepoint.com" not in result["message"]


def test_get_file_content_redacts_url_on_transport_error(monkeypatch):
    sas_url = (
        "https://contoso.sharepoint.com/_layouts/download.aspx"
        "?sastoken=SECRET-CREDENTIAL"
    )
    mock_request = Mock(
        side_effect=requests.exceptions.ConnectionError(
            f"Max retries exceeded with url: {sas_url}"
        )
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "notes.txt"))

    assert result["status"] == "error"
    assert "SECRET-CREDENTIAL" not in result["message"]
    assert "contoso.sharepoint.com" not in result["message"]


def test_get_file_content_redacts_url_on_mid_stream_error(monkeypatch):
    # A connection dropped partway through the body (after a 200 OK, while
    # response.iter_content() is being consumed inside _read_capped_content)
    # is a separate failure point from the connect-time and status-code
    # branches above -- it must be redacted the same way.
    sas_url = (
        "https://contoso.sharepoint.com/_layouts/download.aspx"
        "?sastoken=SECRET-CREDENTIAL"
    )

    class _MidStreamFailureResponse(MockResponse):
        def iter_content(self, chunk_size=1):
            raise requests.exceptions.ChunkedEncodingError(
                f"Connection broken while reading from {self.url}"
            )
            yield b""  # pragma: no cover -- makes this a generator

    mock_request = Mock(
        return_value=_MidStreamFailureResponse(url=sas_url, content=b"partial")
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "notes.txt"))

    assert result["status"] == "error"
    assert "SECRET-CREDENTIAL" not in result["message"]
    assert "contoso.sharepoint.com" not in result["message"]


def test_get_file_content_preserves_graph_error_detail_on_http_error(monkeypatch):
    # response.text is Graph's own JSON error body -- it describes the
    # failure (e.g. accessDenied) but does not itself echo the request URL,
    # so unlike str(exc)/response.url it is safe to keep for diagnostics.
    graph_error_body = b'{"error":{"code":"accessDenied","message":"Access denied"}}'
    mock_request = Mock(
        return_value=MockResponse(status_code=403, content=graph_error_body)
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_file_content("root", "notes.txt"))

    assert result["status"] == "error"
    assert "accessDenied" in result["message"]


def test_upload_text_file_rejects_binary_extension():
    result = json.loads(
        sharepoint.sharepoint_upload_text_file("root", "report.docx", "hello")
    )
    assert result["status"] == "error"
    assert "looks like a binary file" in result["message"]


def test_upload_text_file_rejects_unlisted_binary_extension():
    # Default-deny: an extension absent from the known-text allowlist is
    # rejected even though it isn't one of the small set of formats an
    # older denylist-based guard would have named explicitly.
    result = json.loads(
        sharepoint.sharepoint_upload_text_file("root", "report.doc", "hello")
    )
    assert result["status"] == "error"
    assert "looks like a binary file" in result["message"]


def test_upload_text_file_accepts_extensionless_name(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(
        sharepoint.sharepoint_upload_text_file("root", "Dockerfile", "hello")
    )

    assert result["status"] == "success"


def test_upload_text_file_rejects_dotfile_only_binary_name():
    # ".pdf" is entirely a leading dot plus extension -- Path(".pdf").suffix
    # is empty per pathlib's dotfile convention, which would otherwise slip
    # this past the guard as "extensionless" (accepted).
    result = json.loads(sharepoint.sharepoint_upload_text_file("root", ".pdf", "hello"))
    assert result["status"] == "error"
    assert "looks like a binary file" in result["message"]


def test_upload_text_file_rejects_trailing_period_instead_of_bypassing_guard():
    """A trailing period makes Path.suffix parse as "" (Path("report.docx.").suffix
    == ""), which would otherwise slip past the binary-extension denylist above --
    and SharePoint's backing storage can silently strip that trailing dot, landing
    the write as "report.docx" and overwriting a real document with text content.
    """
    result = json.loads(
        sharepoint.sharepoint_upload_text_file("root", "report.docx.", "hello")
    )
    assert result["status"] == "error"
    assert "must not end with a period" in result["message"]


def test_upload_text_file_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(
        sharepoint.sharepoint_upload_text_file("root", "notes.txt", "hello")
    )

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["data"] == b"hello"
    assert kwargs["headers"]["Content-Type"] == "text/plain; charset=utf-8"


def test_upload_file_rejects_path_outside_allowlist(monkeypatch, tmp_path):
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(b"\x00\x01")

    result = json.loads(sharepoint.sharepoint_upload_file(str(outside_file), "root"))

    assert result["status"] == "error"
    assert "allowed" in result["message"]


def test_upload_file_rejects_over_size_limit(monkeypatch, _upload_allowed_dirs_env):
    big_file = _upload_allowed_dirs_env / "big.bin"
    big_file.write_bytes(b"0" * (sharepoint._SIMPLE_UPLOAD_MAX_BYTES + 1))

    result = json.loads(sharepoint.sharepoint_upload_file(str(big_file), "root"))

    assert result["status"] == "error"
    assert "MB limit" in result["message"]


def test_upload_file_rejects_content_emptied_after_size_check(
    monkeypatch, _upload_allowed_dirs_env
):
    # Simulates the file being truncated to empty between the fstat() size
    # check and the read() -- fstat reports non-empty (so that check
    # passes), but the real, empty-on-disk file's read() genuinely returns
    # b"", which must still be caught rather than PUT as an empty body.
    local_file = _upload_allowed_dirs_env / "vanishes.bin"
    local_file.write_bytes(b"")

    class _FakeStat:
        st_size = 10

    monkeypatch.setattr(sharepoint.os, "fstat", lambda fd: _FakeStat())

    result = json.loads(sharepoint.sharepoint_upload_file(str(local_file), "root"))

    assert result["status"] == "error"
    assert "empty" in result["message"]


def test_upload_file_names_remote_path_in_path_errors(
    monkeypatch, _upload_allowed_dirs_env
):
    # _drive_content_path's error messages default to naming "file_path",
    # but sharepoint_upload_file's own parameter is "remote_path" -- the
    # error must name the parameter this tool actually has.
    local_file = _upload_allowed_dirs_env / "data.bin"
    local_file.write_bytes(b"\x00\x01")

    result = json.loads(
        sharepoint.sharepoint_upload_file(str(local_file), "root", remote_path="Docs/")
    )

    assert result["status"] == "error"
    assert "remote_path" in result["message"]
    assert "file_path" not in result["message"]


def test_upload_file_success(monkeypatch, _upload_allowed_dirs_env):
    local_file = _upload_allowed_dirs_env / "data.bin"
    local_file.write_bytes(b"\x00\x01\x02")
    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(
        sharepoint.sharepoint_upload_file(
            str(local_file), "root", remote_path="Docs/data.bin"
        )
    )

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["url"].endswith("/sites/root/drive/root:/Docs/data.bin:/content")


def test_guess_mime_type_covers_macro_enabled_and_opendocument_formats():
    # These are exactly the formats sharepoint_upload_text_file's own
    # _BINARY_ONLY_EXTENSIONS denylist already recognizes as real binary
    # Office formats; the override table must recognize them too so
    # sharepoint_upload_file doesn't mislabel them on a host whose stdlib
    # mimetypes has no system mime.types file to fall back on.
    assert (
        sharepoint._guess_mime_type("budget.xlsm")
        == "application/vnd.ms-excel.sheet.macroEnabled.12"
    )
    assert (
        sharepoint._guess_mime_type("notes.odt")
        == "application/vnd.oasis.opendocument.text"
    )
    assert sharepoint._guess_mime_type("book.epub") == "application/epub+zip"


def test_guess_mime_type_uses_encoding_not_decompressed_type():
    # mimetypes.guess_type("report.pdf.gz") reports the *decompressed*
    # type ("application/pdf") plus a "gzip" encoding; the actual bytes on
    # the wire are gzip, not PDF, so Content-Type must reflect that.
    assert sharepoint._guess_mime_type("report.pdf.gz") == "application/gzip"


# ---------------------------------------------------------------------------
# lists
# ---------------------------------------------------------------------------


def test_list_lists_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"value": [{"id": "list-1", "name": "Tasks"}]})
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_list_lists("root"))

    assert result["status"] == "success"
    assert result["lists"] == [{"id": "list-1", "name": "Tasks"}]
    assert result["truncated"] is False


def test_list_list_items_expands_fields(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"value": [{"id": "1"}]}))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_list_list_items("root", "Tasks"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["$expand"] == "fields"


def test_create_list_item_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(
        sharepoint.sharepoint_create_list_item("root", "Tasks", '{"Title": "New task"}')
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"] == {"fields": {"Title": "New task"}}


def test_create_list_item_rejects_invalid_json():
    result = json.loads(
        sharepoint.sharepoint_create_list_item("root", "Tasks", "{not json")
    )
    assert result["status"] == "error"
    assert "not valid JSON" in result["message"]


def test_create_list_item_rejects_non_object_json():
    result = json.loads(
        sharepoint.sharepoint_create_list_item("root", "Tasks", "[1, 2, 3]")
    )
    assert result["status"] == "error"
    assert "JSON object" in result["message"]


def test_update_list_item_requires_at_least_one_field():
    result = json.loads(
        sharepoint.sharepoint_update_list_item("root", "Tasks", "item-1", "{}")
    )
    assert result["status"] == "error"
    assert "at least one field" in result["message"]


def test_update_list_item_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"Title": "Done"}))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(
        sharepoint.sharepoint_update_list_item(
            "root", "Tasks", "item-1", '{"Title": "Done"}'
        )
    )

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["url"].endswith("/lists/Tasks/items/item-1/fields")
    assert kwargs["json"] == {"Title": "Done"}


def test_delete_list_item_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({}, status_code=204, content=b""))
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(
        sharepoint.sharepoint_delete_list_item("root", "Tasks", "item-1")
    )

    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# error propagation
# ---------------------------------------------------------------------------


def test_graph_error_response_is_surfaced_as_error(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"error": {"message": "Forbidden"}}, status_code=403)
    )
    monkeypatch.setattr(sharepoint.requests, "request", mock_request)

    result = json.loads(sharepoint.sharepoint_get_root_site())

    assert result["status"] == "error"


def test_missing_auth_token_is_reported(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)

    result = json.loads(sharepoint.sharepoint_get_root_site())

    assert result["status"] == "error"
    assert "AUTH_TOKEN" in result["message"]
