import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import sharepoint


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


@pytest.fixture(autouse=True)
def _upload_allowed_dirs_env(tmp_path, monkeypatch):
    allowed_dir = tmp_path / "workspace"
    allowed_dir.mkdir()
    monkeypatch.setenv("XAGENT_SHAREPOINT_FILE_ALLOWED_DIRS", str(allowed_dir))
    return allowed_dir


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
    assert result["text_content"] is None


def test_upload_text_file_rejects_binary_extension():
    result = json.loads(
        sharepoint.sharepoint_upload_text_file("root", "report.docx", "hello")
    )
    assert result["status"] == "error"
    assert "binary document format" in result["message"]


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


# ---------------------------------------------------------------------------
# lists
# ---------------------------------------------------------------------------


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
