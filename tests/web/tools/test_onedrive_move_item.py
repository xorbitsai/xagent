import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import onedrive

GRAPH = onedrive.GRAPH_BASE_URL
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
                f"{self.status_code} Client Error: Error for url: {GRAPH}/me/drive",
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
    """Route GET/PATCH on /me/drive/items/{id} to canned responses and record
    every request, so tests can assert both the outcome and that nothing was
    mutated when validation fails."""

    def __init__(self, items, patch_response=None):
        self.items = items
        self.patch_response = patch_response
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        prefix = f"{GRAPH}/me/drive/items/"
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
    monkeypatch.setattr(onedrive.requests, "request", graph)
    return graph


def _move(*args, **kwargs):
    return json.loads(onedrive.onedrive_move_item(*args, **kwargs))


def test_moves_a_file_into_another_folder(monkeypatch):
    graph = _install(
        monkeypatch, _FakeGraph({"file-1": _file(), "folder-b": _folder()})
    )

    result = _move("file-1", "folder-b")

    assert result["status"] == "success"
    assert result["item"]["parentReference"]["id"] == "folder-b"
    assert result["destination_folder_id"] == "folder-b"
    assert result["already_in_destination"] is False
    assert result["renamed"] is False
    [(_, url, kwargs)] = graph.patches
    assert url == f"{GRAPH}/me/drive/items/file-1"
    assert kwargs["json"] == {"parentReference": {"id": "folder-b"}}


def test_moves_and_renames_in_one_request(monkeypatch):
    graph = _install(
        monkeypatch, _FakeGraph({"file-1": _file(), "folder-b": _folder()})
    )

    result = _move("file-1", "folder-b", new_name="  final.pdf ")

    assert result["status"] == "success"
    assert result["item"]["name"] == "final.pdf"
    assert result["renamed"] is True
    [(_, _, kwargs)] = graph.patches
    assert kwargs["json"] == {
        "parentReference": {"id": "folder-b"},
        "name": "final.pdf",
    }


def test_root_alias_is_resolved_to_the_real_root_id(monkeypatch):
    """Graph rejects "root" as parentReference.id; the move must send the
    root folder's actual id."""
    graph = _install(monkeypatch, _FakeGraph({"file-1": _file(), "root": _root()}))

    result = _move("file-1", "root")

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

    result = _move("file-1", "folder-b")

    assert result["status"] == "error"
    assert "already" in result["message"]
    assert "new_name" in result["message"]


def test_other_conflicts_keep_graphs_own_error(monkeypatch):
    """409 is not only a name clash; a different conflict code must not be
    reported as one, or the agent would retry with a new name for nothing."""
    _install(
        monkeypatch,
        _FakeGraph(
            {"file-1": _file(), "folder-b": _folder()},
            patch_response=_Response({"error": {"code": "conflict"}}, 409),
        ),
    )

    result = _move("file-1", "folder-b")

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

    result = _move("file-1", "folder-b")

    assert result["status"] == "error"
    assert "permission" in result["message"]


@pytest.mark.parametrize("code", ["notAllowed", "resourceLocked", None])
def test_locked_item_reports_a_clear_error_and_asks_not_to_retry(monkeypatch, code):
    """Production answered 423 notAllowed for a file most likely open in
    Word for the web (#3021); the agent got only the raw status and repeated
    the call."""
    error = {"error": {"code": code}} if code else {}
    _install(
        monkeypatch,
        _FakeGraph(
            {"file-1": _file(), "folder-b": _folder()},
            patch_response=_Response(error, 423),
        ),
    )

    result = _move("file-1", "folder-b")

    assert result["status"] == "error"
    assert "locked" in result["message"]
    assert "close it or check it in" in result["message"]
    assert "Do not retry until they confirm" in result["message"]
    assert "423" not in result["message"]
    if code is not None:
        assert f"(Graph error code: {code})" in result["message"]
    else:
        assert "Graph error code" not in result["message"]


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

    result = _move("file-1", "folder-b")

    assert result["status"] == "error"
    assert field in result["message"]
    assert "not found" in result["message"]
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

    result = _move("file-1", "folder-b")

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

    result = _move("file-1", "folder-b")

    assert result["status"] == "error"
    assert "no longer exists" in result["message"]


@pytest.mark.parametrize("side", ["source", "destination"])
def test_missing_drive_id_does_not_block_the_move(monkeypatch, side):
    """The cross-drive check only clarifies Graph's own refusal, so it is
    skipped when either item leaves driveId out."""
    items = {"file-1": _file(), "folder-b": _folder()}
    key = "file-1" if side == "source" else "folder-b"
    del items[key]["parentReference"]["driveId"]
    graph = _install(monkeypatch, _FakeGraph(items))

    result = _move("file-1", "folder-b")

    assert result["status"] == "success"
    assert len(graph.patches) == 1


def test_ids_that_resolve_to_the_same_item_are_a_self_move(monkeypatch):
    """Ids that differ only in case pass the raw-equality pre-check, so the
    check on the resolved ids is what refuses them."""
    graph = _install(
        monkeypatch,
        _FakeGraph({"FILE-1": _folder("file-1"), "file-1": _folder("file-1")}),
    )

    result = _move("FILE-1", "file-1")

    assert result["status"] == "error"
    assert "itself" in result["message"]
    assert len(graph.calls) == 2
    assert graph.patches == []


@pytest.mark.asyncio
async def test_move_item_is_registered_with_an_optional_new_name():
    tools = {tool.name: tool for tool in await onedrive.mcp.list_tools()}

    schema = tools["onedrive_move_item"].inputSchema
    assert set(schema["properties"]) == {
        "item_id",
        "destination_folder_id",
        "new_name",
    }
    assert sorted(schema["required"]) == ["destination_folder_id", "item_id"]
    assert schema["properties"]["new_name"]["default"] == ""


def test_destination_that_is_a_file_is_rejected(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(), "file-2": _file("file-2", "other.pdf")}),
    )

    result = _move("file-1", "file-2")

    assert result["status"] == "error"
    assert "folder" in result["message"]
    assert graph.patches == []


def test_shared_folder_from_another_drive_is_rejected(monkeypatch):
    """A shared folder added to "My files" is a remoteItem whose contents live
    in its owner's drive; Graph cannot move items between drives."""
    shared = _folder(
        "shared-1",
        remoteItem={
            "id": "remote-folder",
            "folder": {},
            "parentReference": {"driveId": "other-drive"},
        },
    )
    graph = _install(monkeypatch, _FakeGraph({"file-1": _file(), "shared-1": shared}))

    result = _move("file-1", "shared-1")

    assert result["status"] == "error"
    assert "another drive" in result["message"]
    assert graph.patches == []


def test_destination_outside_the_drive_is_rejected(monkeypatch):
    elsewhere = _folder("folder-x")
    elsewhere["parentReference"]["driveId"] = "other-drive"
    graph = _install(
        monkeypatch, _FakeGraph({"file-1": _file(), "folder-x": elsewhere})
    )

    result = _move("file-1", "folder-x")

    assert result["status"] == "error"
    assert "another drive" in result["message"]
    assert graph.patches == []


def test_moving_an_item_into_itself_is_rejected_without_a_request(monkeypatch):
    graph = _install(monkeypatch, _FakeGraph({}))

    result = _move("folder-b", "folder-b")

    assert result["status"] == "error"
    assert "itself" in result["message"]
    assert graph.calls == []


def test_the_drive_root_cannot_be_moved(monkeypatch):
    graph = _install(monkeypatch, _FakeGraph({"root": _root(), "folder-b": _folder()}))

    result = _move("root", "folder-b")

    assert result["status"] == "error"
    assert "root" in result["message"]
    assert graph.patches == []


def test_already_in_destination_without_rename_makes_no_change(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(parent_id="folder-b"), "folder-b": _folder()}),
    )

    result = _move("file-1", "folder-b", new_name="report.pdf")

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

    result = _move("file-1", "folder-b", new_name="final.pdf")

    assert result["status"] == "success"
    assert result["already_in_destination"] is True
    assert result["renamed"] is True
    [(_, _, kwargs)] = graph.patches
    assert kwargs["json"] == {"name": "final.pdf"}


def test_blank_new_name_is_rejected_without_a_request(monkeypatch):
    graph = _install(monkeypatch, _FakeGraph({}))

    result = _move("file-1", "folder-b", new_name="   ")

    assert result["status"] == "error"
    assert "new_name" in result["message"]
    assert graph.calls == []


@pytest.mark.parametrize("bad_id", [".", "..", "", " file-1"])
@pytest.mark.parametrize("position", ["item_id", "destination_folder_id"])
def test_unsafe_ids_are_rejected_without_a_request(monkeypatch, bad_id, position):
    graph = _install(monkeypatch, _FakeGraph({}))
    ids = {"item_id": "file-1", "destination_folder_id": "folder-b"}
    ids[position] = bad_id

    result = _move(ids["item_id"], ids["destination_folder_id"])

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

    result = _move("file-1", "folder-b")

    assert result["status"] == "error"
    assert "may have moved" in result["message"]
    assert "onedrive_get_item" in result["message"]


def test_move_response_without_a_parent_is_not_reported_as_a_failure(monkeypatch):
    """The PATCH has already been applied; a response that leaves out
    parentReference must not turn a completed move into an error."""
    moved = _file()
    del moved["parentReference"]
    _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(), "folder-b": _folder()}, patch_response=moved),
    )

    result = _move("file-1", "folder-b")

    assert result["status"] == "success"
    assert result["already_in_destination"] is False


def test_drive_ids_that_differ_only_in_case_are_the_same_drive(monkeypatch):
    destination = _folder()
    destination["parentReference"]["driveId"] = DRIVE_ID.upper()
    graph = _install(
        monkeypatch, _FakeGraph({"file-1": _file(), "folder-b": destination})
    )

    result = _move("file-1", "folder-b")

    assert result["status"] == "success"
    assert len(graph.patches) == 1


def test_parent_id_that_differs_only_in_case_is_already_in_destination(monkeypatch):
    graph = _install(
        monkeypatch,
        _FakeGraph({"file-1": _file(parent_id="FOLDER-B"), "folder-b": _folder()}),
    )

    result = _move("file-1", "folder-b")

    assert result["status"] == "success"
    assert result["already_in_destination"] is True
    assert graph.patches == []


@pytest.mark.parametrize("parent_id", ["folder-a", "folder-b"], ids=["moved", "no-op"])
def test_result_omits_preauthenticated_download_urls(monkeypatch, parent_id):
    signed = _file(
        parent_id=parent_id,
        **{"@microsoft.graph.downloadUrl": "https://signed.example/x"},
    )
    _install(monkeypatch, _FakeGraph({"file-1": signed, "folder-b": _folder()}))

    raw = onedrive.onedrive_move_item("file-1", "folder-b")

    assert "signed.example" not in raw
    assert json.loads(raw)["status"] == "success"


def test_unexpected_graph_failure_is_reported(monkeypatch):
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(side_effect=requests.ConnectionError("x"))
    )

    result = _move("file-1", "folder-b")

    assert result["status"] == "error"
