"""A drive path that misses must tell the agent how to find the real one.

In #2875 the agent guessed ``Documents/deck.pptx``, then passed its local
workspace path ``output/deck.pptx``, and never listed the drive root where the
file actually was. These tests pin the hint a path-addressed 404 returns, and
that Graph error bodies are never copied into the caller's error message.
"""

import json
from unittest.mock import Mock

import pytest
import requests

from xagent.core.tools.adapters.vibe.mcp_adapter import _FIELD_TEXT_MAX_CHARS
from xagent.web.tools.mcp import onedrive, powerpoint

# Graph and its backing storage may echo a preauthenticated URL into an
# error body; none of it may reach the caller.
_SECRET_URL = "https://my.microsoftpersonalcontent.com/x?tempauth=SECRETVALUE"


class _Response:
    def __init__(self, payload=None, status_code=200, content=None):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.content = (
            json.dumps(self._payload).encode("utf-8") if content is None else content
        )
        self.text = self.content.decode("utf-8", errors="replace")
        self.headers = {}

    def json(self):
        # Like requests: a non-JSON body raises ValueError.
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            # requests puts the final URL in its own message. After a /content
            # redirect that URL is the preauthenticated one.
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Error for url: {_SECRET_URL}",
                response=self,
            )

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]

    def close(self):
        return None


def _graph_error(status_code: int, code: str) -> _Response:
    return _Response(
        {"error": {"code": code, "message": f"See {_SECRET_URL}"}},
        status_code=status_code,
    )


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "token")


def _patch_onedrive(monkeypatch, *responses):
    mock_request = Mock(side_effect=list(responses))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)
    return mock_request


def _assert_path_hint(message: str) -> None:
    assert "relative to the drive root" in message
    assert "not a local task-workspace path" in message
    assert "onedrive_list_items" in message


@pytest.mark.parametrize(
    "call",
    [
        lambda: onedrive.onedrive_get_item(path="Documents/deck.pptx"),
        lambda: onedrive.onedrive_get_file_content("Documents/notes.txt"),
        lambda: onedrive.onedrive_list_items(folder_path="Documents"),
        lambda: onedrive.onedrive_create_folder("Q3", parent_path="Documents"),
    ],
    ids=["get_item", "get_file_content", "list_items", "create_folder"],
)
def test_onedrive_path_lookup_miss_explains_how_to_find_the_path(monkeypatch, call):
    _patch_onedrive(monkeypatch, _graph_error(404, "itemNotFound"))

    result = json.loads(call())

    assert result["status"] == "error"
    assert "HTTP 404" in result["message"]
    assert "itemNotFound" in result["message"]
    _assert_path_hint(result["message"])
    assert "SECRETVALUE" not in result["message"]


def test_onedrive_non_json_path_miss_still_gets_the_hint(monkeypatch):
    _patch_onedrive(
        monkeypatch,
        _Response(status_code=404, content=f"<html>{_SECRET_URL}</html>".encode()),
    )

    result = json.loads(onedrive.onedrive_get_item(path="Documents/deck.pptx"))

    assert result["status"] == "error"
    _assert_path_hint(result["message"])
    assert "SECRETVALUE" not in result["message"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: onedrive.onedrive_get_item(path="Documents/deck.pptx"),
        lambda: onedrive.onedrive_download_file("Documents/deck.pptx"),
        lambda: powerpoint.powerpoint_list_slides("Documents/deck.pptx"),
    ],
    ids=["onedrive_get_item", "onedrive_download_file", "powerpoint_list_slides"],
)
def test_a_404_that_is_not_item_not_found_gets_no_path_hint(
    monkeypatch, tmp_path, call
):
    # e.g. a drive that does not exist: the path is not what is wrong.
    monkeypatch.setenv("XAGENT_ONEDRIVE_OUTPUT_DIR", str(tmp_path))
    _patch_onedrive(monkeypatch, _graph_error(404, "ResourceNotFound"))
    _patch_powerpoint(monkeypatch, _graph_error(404, "ResourceNotFound"))

    result = json.loads(call())

    assert result["status"] == "error"
    assert "HTTP 404" in result["message"]
    assert "relative to the drive root" not in result["message"]


def test_graph_request_adds_the_hint_only_for_an_opted_in_path_lookup(monkeypatch):
    # Internal lookups that expect a 404 (e.g. _current_upload_item probing an
    # upload target) must not build the hint.
    _patch_onedrive(
        monkeypatch,
        _graph_error(404, "itemNotFound"),
        _graph_error(404, "itemNotFound"),
    )

    with pytest.raises(onedrive._GraphRequestError) as plain:
        onedrive._graph_request("GET", "/me/drive/root:/a.txt:")
    with pytest.raises(onedrive._GraphRequestError) as lookup:
        onedrive._graph_request("GET", "/me/drive/root:/a.txt:", path_lookup=True)

    assert "relative to the drive root" not in str(plain.value)
    _assert_path_hint(str(lookup.value))


def test_onedrive_download_metadata_miss_explains_how_to_find_the_path(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_ONEDRIVE_OUTPUT_DIR", str(tmp_path))
    _patch_onedrive(monkeypatch, _graph_error(404, "itemNotFound"))

    result = json.loads(onedrive.onedrive_download_file("Documents/deck.pptx"))

    assert result["status"] == "error"
    assert result["message"].startswith("OneDrive file download failed with HTTP 404")
    _assert_path_hint(result["message"])


def test_onedrive_upload_put_miss_gets_no_path_hint(monkeypatch):
    # A content PUT targets a path that need not exist yet, so its 404 is
    # not reported as a wrong path.
    _patch_onedrive(monkeypatch, _graph_error(404, "itemNotFound"))

    result = json.loads(onedrive.onedrive_upload_text_file("Reports/a.txt", "hi"))

    assert result["status"] == "error"
    assert "relative to the drive root" not in result["message"]


def test_onedrive_item_id_miss_gets_no_path_hint(monkeypatch):
    _patch_onedrive(monkeypatch, _graph_error(404, "itemNotFound"))

    result = json.loads(onedrive.onedrive_get_item(item_id="ITEM-1"))

    assert result["status"] == "error"
    assert "HTTP 404" in result["message"]
    assert "relative to the drive root" not in result["message"]


def test_onedrive_non_404_path_error_gets_no_path_hint(monkeypatch):
    _patch_onedrive(monkeypatch, _graph_error(403, "accessDenied"))

    result = json.loads(onedrive.onedrive_get_item(path="Documents/deck.pptx"))

    assert result["status"] == "error"
    assert "HTTP 403" in result["message"]
    assert "accessDenied" in result["message"]
    assert "relative to the drive root" not in result["message"]


@pytest.mark.parametrize(
    "response",
    [
        _graph_error(500, "generalException"),
        _Response(status_code=502, content=f"<html>{_SECRET_URL}</html>".encode()),
    ],
    ids=["json-body", "non-json-body"],
)
def test_onedrive_graph_errors_never_forward_the_response_body(
    monkeypatch, caplog, response
):
    _patch_onedrive(monkeypatch, response)

    result = json.loads(onedrive.onedrive_list_items())

    assert result["status"] == "error"
    assert f"HTTP {response.status_code}" in result["message"]
    assert "SECRETVALUE" not in result["message"]
    assert "SECRETVALUE" not in caplog.text


def _patch_powerpoint(monkeypatch, *responses):
    mock_request = Mock(side_effect=list(responses))
    monkeypatch.setattr(powerpoint.requests, "request", mock_request)
    return mock_request


@pytest.mark.parametrize(
    "call",
    [
        lambda: powerpoint.powerpoint_list_slides("Documents/deck.pptx"),
        lambda: powerpoint.powerpoint_add_slide(
            "Documents/deck.pptx", expected_etag='"etag-1"', title="Quarterly"
        ),
        lambda: powerpoint.powerpoint_list_slides(
            "Documents/deck.pptx", site_id="contoso.sharepoint.com", drive_id="D1"
        ),
    ],
    ids=["list_slides", "add_slide", "site_drive"],
)
def test_powerpoint_path_miss_explains_how_to_find_the_path(monkeypatch, call):
    mock_request = _patch_powerpoint(monkeypatch, _graph_error(404, "itemNotFound"))

    result = json.loads(call())

    assert result["status"] == "error"
    assert "HTTP 404" in result["message"]
    assert "itemNotFound" in result["message"]
    _assert_path_hint(result["message"])
    assert "sharepoint_list_items" in result["message"]
    # The miss stops at the metadata lookup; no upload session is created.
    assert mock_request.call_count == 1


def test_powerpoint_non_404_lookup_error_gets_no_path_hint(monkeypatch):
    _patch_powerpoint(monkeypatch, _graph_error(403, "accessDenied"))

    result = json.loads(powerpoint.powerpoint_list_slides("deck.pptx"))

    assert result["status"] == "error"
    assert "relative to the drive root" not in result["message"]


def _path_descriptions(tools, field_names):
    descriptions = {}
    for tool in tools:
        for field_name in field_names:
            field = tool.inputSchema["properties"].get(field_name)
            if field is not None:
                descriptions[(tool.name, field_name)] = field.get("description", "")
    return descriptions


@pytest.mark.parametrize(
    ("module", "field_names", "expected_fields"),
    [
        (
            onedrive,
            ("path", "file_path", "folder_path", "remote_path", "parent_path"),
            {
                ("onedrive_list_items", "folder_path"),
                ("onedrive_create_folder", "parent_path"),
                ("onedrive_upload_file", "remote_path"),
                ("onedrive_get_item", "path"),
                ("onedrive_get_file_content", "file_path"),
                ("onedrive_download_file", "file_path"),
                ("onedrive_upload_text_file", "file_path"),
            },
        ),
        (
            powerpoint,
            ("file_path",),
            {
                ("powerpoint_create_presentation", "file_path"),
                ("powerpoint_get_presentation_text", "file_path"),
                ("powerpoint_list_slides", "file_path"),
                ("powerpoint_get_slide_text", "file_path"),
                ("powerpoint_set_shape_text", "file_path"),
                ("powerpoint_add_slide", "file_path"),
                ("powerpoint_list_slide_layouts", "file_path"),
                ("powerpoint_delete_slide", "file_path"),
            },
        ),
    ],
    ids=["onedrive", "powerpoint"],
)
async def test_drive_path_fields_say_they_are_not_workspace_paths(
    module, field_names, expected_fields
):
    descriptions = _path_descriptions(await module.mcp.list_tools(), field_names)

    assert set(descriptions) == expected_fields
    for key, description in descriptions.items():
        assert "relative to the drive root" in description, key
        assert "not a local task-workspace path" in description, key
        # The adapter clips a longer description before the model sees it.
        assert len(description) <= _FIELD_TEXT_MAX_CHARS, key


@pytest.mark.parametrize("value", [[], None], ids=["empty", "null"])
def test_onedrive_search_miss_points_at_listing_the_folder(monkeypatch, value):
    # In #2875 search returned no items for a file at the drive root, and the
    # agent guessed a folder instead of listing one.
    _patch_onedrive(monkeypatch, _Response({"value": value}))

    result = json.loads(onedrive.onedrive_search_files("August2026.pptx"))

    assert result["status"] == "success"
    assert result["items"] == value
    assert "can miss existing items" in result["hint"]
    assert "onedrive_list_items" in result["hint"]


def test_onedrive_search_hit_gets_no_hint(monkeypatch):
    item = {"id": "1", "name": "deck.pptx"}
    _patch_onedrive(monkeypatch, _Response({"value": [item]}))

    result = json.loads(onedrive.onedrive_search_files("deck.pptx"))

    assert result == {"status": "success", "items": [item]}


async def test_onedrive_search_description_points_at_listing_the_folder():
    tools = {tool.name: tool for tool in await onedrive.mcp.list_tools()}
    description = tools["onedrive_search_files"].description

    assert "can miss" in description
    assert "onedrive_list_items" in description


async def test_onedrive_download_prefers_editing_office_files_in_place():
    # In #2875 the agent downloaded a deck and edited a local copy instead of
    # calling the PowerPoint connector on the drive path it had just used.
    tools = {tool.name: tool for tool in await onedrive.mcp.list_tools()}
    description = tools["onedrive_download_file"].description

    for prefix in ("powerpoint_", "word_", "excel_"):
        assert prefix in description
    assert "in place" in description
    # The old wording recommended this tool for Office files to be edited.
    assert "intended for Office files" not in description
