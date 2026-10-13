import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import sharepoint

GRAPH = sharepoint.GRAPH_BASE_URL
SITE_ID = "site-1"
DRIVE_ID = "drive-1"


class _Response:
    def __init__(self, json_data=None, status_code=200):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.content = json.dumps(self._json_data).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Error for url: {GRAPH}/sites",
                response=self,
            )


def _file(item_id="file-1", name="report.pdf", parent_id="folder-a", **extra):
    return {
        "id": item_id,
        "name": name,
        "file": {"mimeType": "application/pdf"},
        "parentReference": {"driveId": DRIVE_ID, "id": parent_id},
        **extra,
    }


def _folder(item_id="folder-b", name="Archive", parent_id="root-id", **extra):
    return {
        "id": item_id,
        "name": name,
        "folder": {"childCount": 0},
        "parentReference": {"driveId": DRIVE_ID, "id": parent_id},
        **extra,
    }


def _root():
    return {
        "id": "root-id",
        "name": "root",
        "root": {},
        "folder": {"childCount": 3},
        "parentReference": {"driveId": DRIVE_ID},
    }


class _FakeGraph:
    def __init__(self, items, patch_response=None):
        self.items = items
        self.patch_response = patch_response
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        prefix = f"{GRAPH}/sites/{SITE_ID}/drives/{DRIVE_ID}/items/"
        assert url.startswith(prefix), url
        item_key = url[len(prefix) :]
        if method == "GET":
            item = self.items.get(item_key)
            if isinstance(item, _Response):
                return item
            if item is None:
                return _Response({"error": {"code": "itemNotFound"}}, 404)
            return _Response(item)
        if method == "PATCH":
            if isinstance(self.patch_response, _Response):
                return self.patch_response
            if self.patch_response is not None:
                return _Response(self.patch_response)
            moved = dict(self.items[item_key])
            body = kwargs["json"]
            if "parentReference" in body:
                moved["parentReference"] = {
                    "driveId": DRIVE_ID,
                    "id": body["parentReference"]["id"],
                }
            if "name" in body:
                moved["name"] = body["name"]
            return _Response(moved)
        raise AssertionError(f"unexpected {method} {url}")

    @property
    def patches(self):
        return [call for call in self.calls if call[0] == "PATCH"]


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")


def _install(monkeypatch, graph):
    monkeypatch.setattr(sharepoint.requests, "request", graph)
    return graph


def _move(item_id="file-1", destination_folder_id="folder-b", **kwargs):
    return json.loads(
        sharepoint.sharepoint_move_item(
            SITE_ID,
            DRIVE_ID,
            item_id,
            destination_folder_id,
            **kwargs,
        )
    )


def test_moves_a_file_into_another_folder(monkeypatch):
    graph = _install(
        monkeypatch, _FakeGraph({"file-1": _file(), "folder-b": _folder()})
    )

    result = _move()

    assert result["status"] == "success"
    assert result["item"]["parentReference"]["id"] == "folder-b"
    assert result["destination_folder_id"] == "folder-b"
    assert result["already_in_destination"] is False
    assert result["renamed"] is False
    [(_, url, kwargs)] = graph.patches
    assert url == f"{GRAPH}/sites/{SITE_ID}/drives/{DRIVE_ID}/items/file-1"
    assert kwargs["json"] == {"parentReference": {"id": "folder-b"}}


def test_moves_and_renames_in_one_request(monkeypatch):
    graph = _install(
        monkeypatch, _FakeGraph({"file-1": _file(), "folder-b": _folder()})
    )

    result = _move(new_name="  final.pdf ")

    assert result["status"] == "success"
    assert result["item"]["name"] == "final.pdf"
    assert result["renamed"] is True
    [(_, _, kwargs)] = graph.patches
    assert kwargs["json"] == {
        "parentReference": {"id": "folder-b"},
        "name": "final.pdf",
    }


def test_root_alias_is_resolved_to_the_real_root_id(monkeypatch):
    graph = _install(monkeypatch, _FakeGraph({"file-1": _file(), "root": _root()}))

    result = _move(destination_folder_id="root")

    assert result["status"] == "success"
    assert result["destination_folder_id"] == "root-id"
    [(_, _, kwargs)] = graph.patches
    assert kwargs["json"] == {"parentReference": {"id": "root-id"}}


def test_name_conflict_reports_a_clear_error(monkeypatch):
    _install(
        monkeypatch,
        _FakeGraph(
            {"file-1": _file(), "folder-b": _folder()},
            patch_response=_Response({"error": {"code": "nameAlreadyExists"}}, 409),
        ),
    )

    result = _move()

    assert result["status"] == "error"
    assert "already" in result["message"]
    assert "report.pdf" in result["message"]
    assert "new_name" in result["message"]


def test_other_conflicts_keep_graphs_own_error(monkeypatch):
    _install(
        monkeypatch,
        _FakeGraph(
            {"file-1": _file(), "folder-b": _folder()},
            patch_response=_Response({"error": {"code": "conflict"}}, 409),
        ),
    )

    result = _move()

    assert result["status"] == "error"
    assert "new_name" not in result["message"]
    assert "409" in result["message"]


@pytest.mark.parametrize("status_code", [401, 500])
def test_get_http_failures_are_reported_without_special_mapping(
    monkeypatch, status_code
):
    _install(
        monkeypatch,
        _FakeGraph(
            {
                "file-1": _Response({"error": {"code": "graphFailure"}}, status_code),
                "folder-b": _folder(),
            }
        ),
    )

    result = _move()

    assert result["status"] == "error"
    assert str(status_code) in result["message"]


def test_non_json_name_conflict_keeps_graph_error(monkeypatch):
    _install(
        monkeypatch,
        _FakeGraph(
            {"file-1": _file(), "folder-b": _folder()},
            patch_response=_Response("not-json", 409),
        ),
    )

    result = _move()

    assert result["status"] == "error"
    assert "new_name" not in result["message"]
    assert "409" in result["message"]


def test_permission_denied_on_move_reports_a_clear_error(monkeypatch):
    _install(
        monkeypatch,
        _FakeGraph(
            {"file-1": _file(), "folder-b": _folder()},
            patch_response=_Response({"error": {"code": "accessDenied"}}, 403),
        ),
    )

    result = _move()

    assert result["status"] == "error"
    assert "permission" in result["message"]


@pytest.mark.parametrize(
    ("missing", "field"),
    [("file-1", "item_id"), ("folder-b", "destination_folder_id")],
)
def test_unknown_item_or_destination_fails_before_any_change(
    monkeypatch, missing, field
):
    items = {"file-1": _file(), "folder-b": _folder()}
    del items[missing]
    graph = _install(monkeypatch, _FakeGraph(items))

    result = _move()

    assert result["status"] == "error"
    assert field in result["message"]
    assert "not found" in result["message"]
    if field == "destination_folder_id":
        assert "another document library" in result["message"]
    assert graph.patches == []


@pytest.mark.parametrize(
    ("denied", "field"),
    [("file-1", "item_id"), ("folder-b", "destination_folder_id")],
)
def test_inaccessible_item_or_destination_fails_before_any_change(
    monkeypatch, denied, field
):
    items = {"file-1": _file(), "folder-b": _folder()}
    items[denied] = _Response({"error": {"code": "accessDenied"}}, 403)
    graph = _install(monkeypatch, _FakeGraph(items))

    result = _move()

    assert result["status"] == "error"
    assert field in result["message"]
    assert "permission" in result["message"]
    assert graph.patches == []


def test_item_gone_by_the_time_of_the_move_reports_a_clear_error(monkeypatch):
    _install(
        monkeypatch,
        _FakeGraph(
            {"file-1": _file(), "folder-b": _folder()},
            patch_response=_Response({"error": {"code": "itemNotFound"}}, 404),
        ),
    )

    result = _move()

    assert result["status"] == "error"
    assert "no longer exists" in result["message"]


def test_destination_that_is_a_file_is_rejected(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(), "file-2": _file("file-2", "other.pdf")}),
    )

    result = _move(destination_folder_id="file-2")

    assert result["status"] == "error"
    assert "folder" in result["message"]
    assert graph.patches == []


def test_item_without_an_id_is_rejected(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph({"file-1": {"name": "report.pdf"}, "folder-b": _folder()}),
    )

    result = _move()

    assert result["status"] == "error"
    assert "invalid item" in result["message"]
    assert graph.patches == []


def test_shared_folder_from_another_drive_is_rejected(monkeypatch):
    shared = _folder(
        "shared-1",
        remoteItem={
            "id": "remote-folder",
            "folder": {},
            "parentReference": {"driveId": "other-drive"},
        },
    )
    graph = _install(monkeypatch, _FakeGraph({"file-1": _file(), "shared-1": shared}))

    result = _move(destination_folder_id="shared-1")

    assert result["status"] == "error"
    assert "between drives" in result["message"]
    assert graph.patches == []


def test_destination_outside_the_drive_is_rejected(monkeypatch):
    elsewhere = _folder("folder-x")
    elsewhere["parentReference"]["driveId"] = "other-drive"
    graph = _install(
        monkeypatch, _FakeGraph({"file-1": _file(), "folder-x": elsewhere})
    )

    result = _move(destination_folder_id="folder-x")

    assert result["status"] == "error"
    assert "between drives" in result["message"]
    assert graph.patches == []


def test_source_outside_the_selected_drive_is_rejected(monkeypatch):
    source = _file()
    source["parentReference"]["driveId"] = "other-drive"
    graph = _install(monkeypatch, _FakeGraph({"file-1": source, "folder-b": _folder()}))

    result = _move()

    assert result["status"] == "error"
    assert "between drives" in result["message"]
    assert graph.patches == []


def test_moving_an_item_into_itself_is_rejected_without_a_request(monkeypatch):
    graph = _install(monkeypatch, _FakeGraph({}))

    result = _move(item_id="folder-b", destination_folder_id="folder-b")

    assert result["status"] == "error"
    assert "itself" in result["message"]
    assert graph.calls == []


def test_ids_that_differ_only_in_case_are_not_treated_as_the_same_item(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph(
            {
                "folder-b": _file("folder-b"),
                "FOLDER-B": _folder("FOLDER-B"),
            }
        ),
    )

    result = _move(item_id="folder-b", destination_folder_id="FOLDER-B")

    assert result["status"] == "success"
    assert result["already_in_destination"] is False
    assert len(graph.patches) == 1


def test_the_drive_root_cannot_be_moved(monkeypatch):
    graph = _install(monkeypatch, _FakeGraph({"root": _root(), "folder-b": _folder()}))

    result = _move(item_id="root")

    assert result["status"] == "error"
    assert "root" in result["message"]
    assert graph.patches == []


def test_already_in_destination_without_rename_makes_no_change(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(parent_id="folder-b"), "folder-b": _folder()}),
    )

    result = _move(new_name="report.pdf")

    assert result["status"] == "success"
    assert result["already_in_destination"] is True
    assert result["renamed"] is False
    assert result["item"]["id"] == "file-1"
    assert graph.patches == []


def test_already_in_destination_with_rename_only_renames(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(parent_id="folder-b"), "folder-b": _folder()}),
    )

    result = _move(new_name="final.pdf")

    assert result["status"] == "success"
    assert result["already_in_destination"] is True
    assert result["renamed"] is True
    [(_, _, kwargs)] = graph.patches
    assert kwargs["json"] == {"name": "final.pdf"}


def test_blank_new_name_is_rejected_without_a_request(monkeypatch):
    graph = _install(monkeypatch, _FakeGraph({}))

    result = _move(new_name="   ")

    assert result["status"] == "error"
    assert "new_name" in result["message"]
    assert graph.calls == []


@pytest.mark.parametrize("bad_drive_id", [".", "..", "", " drive-1"])
def test_unsafe_drive_id_is_rejected_without_a_request(monkeypatch, bad_drive_id):
    graph = _install(monkeypatch, _FakeGraph({}))

    result = json.loads(
        sharepoint.sharepoint_move_item(SITE_ID, bad_drive_id, "file-1", "folder-b")
    )

    assert result["status"] == "error"
    assert graph.calls == []


@pytest.mark.parametrize("bad_id", [".", "..", "", " file-1"])
@pytest.mark.parametrize("position", ["item_id", "destination_folder_id"])
def test_unsafe_ids_are_rejected_without_a_request(monkeypatch, bad_id, position):
    graph = _install(monkeypatch, _FakeGraph({}))
    ids = {"item_id": "file-1", "destination_folder_id": "folder-b"}
    ids[position] = bad_id

    result = _move(**ids)

    assert result["status"] == "error"
    assert graph.calls == []


def test_move_response_without_the_destination_is_an_error(monkeypatch):
    _install(
        monkeypatch,
        _FakeGraph(
            {"file-1": _file(), "folder-b": _folder()},
            patch_response=_file(),
        ),
    )

    result = _move()

    assert result["status"] == "error"
    assert "may have moved" in result["message"]
    assert "sharepoint_list_items" in result["message"]


def test_move_response_without_a_parent_is_not_reported_as_a_failure(monkeypatch):
    moved = _file()
    del moved["parentReference"]
    _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(), "folder-b": _folder()}, patch_response=moved),
    )

    result = _move()

    assert result["status"] == "success"
    assert result["already_in_destination"] is False


def test_drive_ids_that_differ_only_in_case_are_different_drives(monkeypatch):
    destination = _folder()
    destination["parentReference"]["driveId"] = DRIVE_ID.upper()
    graph = _install(
        monkeypatch, _FakeGraph({"file-1": _file(), "folder-b": destination})
    )

    result = _move()

    assert result["status"] == "error"
    assert "between drives" in result["message"]
    assert graph.patches == []


def test_parent_ids_that_differ_only_in_case_do_not_skip_the_move(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(parent_id="FOLDER-B"), "folder-b": _folder()}),
    )

    result = _move()

    assert result["status"] == "success"
    assert result["already_in_destination"] is False
    assert len(graph.patches) == 1


def test_non_dict_move_response_is_rejected(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(), "folder-b": _folder()}, patch_response=[]),
    )

    result = _move()

    assert result["status"] == "error"
    assert "invalid item" in result["message"]
    assert len(graph.patches) == 1


def test_invalid_site_id_is_rejected_without_a_request(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(), "folder-b": _folder()}),
    )

    result = json.loads(
        sharepoint.sharepoint_move_item("..", DRIVE_ID, "file-1", "folder-b")
    )

    assert result["status"] == "error"
    assert graph.calls == []


@pytest.mark.parametrize("parent_id", ["folder-a", "folder-b"], ids=["moved", "no-op"])
def test_result_omits_preauthenticated_download_urls(monkeypatch, parent_id):
    signed = _file(
        parent_id=parent_id,
        **{"@microsoft.graph.downloadUrl": "https://signed.example/x"},
    )
    _install(monkeypatch, _FakeGraph({"file-1": signed, "folder-b": _folder()}))

    raw = sharepoint.sharepoint_move_item(SITE_ID, DRIVE_ID, "file-1", "folder-b")

    assert "signed.example" not in raw
    assert json.loads(raw)["status"] == "success"


def test_unexpected_graph_failure_is_reported(monkeypatch):
    monkeypatch.setattr(
        sharepoint.requests, "request", Mock(side_effect=requests.ConnectionError("x"))
    )

    result = _move()

    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_move_item_is_registered_with_annotations_and_optional_new_name():
    tools = {tool.name: tool for tool in await sharepoint.mcp.list_tools()}

    tool = tools["sharepoint_move_item"]
    schema = tool.inputSchema
    assert set(schema["properties"]) == {
        "site_id",
        "drive_id",
        "item_id",
        "destination_folder_id",
        "new_name",
    }
    assert sorted(schema["required"]) == [
        "destination_folder_id",
        "drive_id",
        "item_id",
        "site_id",
    ]
    assert schema["properties"]["new_name"]["default"] == ""
    assert tool.annotations.destructiveHint is True
    assert tool.annotations.idempotentHint is True
