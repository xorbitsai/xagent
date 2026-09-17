import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import planner


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
# helpers
# ---------------------------------------------------------------------------


def test_current_etag_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"id": "task-1", "@odata.etag": 'W/"etag-1"'})
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    assert planner._current_etag("/planner/tasks/task-1") == 'W/"etag-1"'


def test_current_etag_raises_when_missing(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "task-1"}))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    with pytest.raises(RuntimeError, match="did not return an @odata.etag"):
        planner._current_etag("/planner/tasks/task-1")


def test_etag_guarded_write_fetches_when_not_supplied(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"fetched"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    planner._etag_guarded_write("/planner/tasks/task-1", "PATCH", body={"title": "x"})

    assert mock_request.call_count == 2
    patch_call = mock_request.call_args_list[1]
    assert patch_call.kwargs["headers"]["If-Match"] == 'W/"fetched"'


def test_etag_guarded_write_uses_supplied_etag(monkeypatch):
    mock_request = Mock(return_value=MockResponse({}, status_code=204, content=b""))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    planner._etag_guarded_write("/planner/tasks/task-1", "DELETE", etag='W/"supplied"')

    assert mock_request.call_count == 1
    assert mock_request.call_args.kwargs["headers"]["If-Match"] == 'W/"supplied"'


def test_build_assignments_empty_returns_none():
    assert planner._build_assignments(None) is None
    assert planner._build_assignments([]) is None


def test_build_assignments_shape():
    result = planner._build_assignments(["user-1", "user-2"])
    assert result == {
        "user-1": {
            "@odata.type": "#microsoft.graph.plannerAssignment",
            "orderHint": " !",
        },
        "user-2": {
            "@odata.type": "#microsoft.graph.plannerAssignment",
            "orderHint": " !",
        },
    }


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------


def test_list_plans_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {"value": [{"id": "plan-1"}], "@odata.nextLink": "https://next"}
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_list_plans("group-1"))

    assert result["status"] == "success"
    assert result["plans"] == [{"id": "plan-1"}]
    assert result["next_link"] == "https://next"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/groups/group-1/planner/plans"
    )


def test_create_plan_builds_container_url(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"id": "plan-1", "title": "New plan"})
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_create_plan("group-1", "New plan"))

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["json"] == {
        "container": {"url": "https://graph.microsoft.com/v1.0/groups/group-1"},
        "title": "New plan",
    }


def test_create_plan_requires_title():
    result = json.loads(planner.planner_create_plan("group-1", "   "))
    assert result["status"] == "error"


def test_create_plan_strips_title_whitespace(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "plan-1"}))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    planner.planner_create_plan("group-1", "  New plan  ")

    assert mock_request.call_args.kwargs["json"]["title"] == "New plan"


# ---------------------------------------------------------------------------
# buckets
# ---------------------------------------------------------------------------


def test_create_bucket_uses_default_order_hint(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "bucket-1"}))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_create_bucket("plan-1", "To do"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"] == {
        "name": "To do",
        "planId": "plan-1",
        "orderHint": " !",
    }


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


def test_create_task_with_assignees_and_due_date(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "task-1"}))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(
        planner.planner_create_task(
            "plan-1",
            "Write report",
            bucket_id="bucket-1",
            assignee_user_ids=["user-1"],
            due_date_time="2026-09-30T00:00:00Z",
        )
    )

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]
    assert body["planId"] == "plan-1"
    assert body["bucketId"] == "bucket-1"
    assert body["dueDateTime"] == "2026-09-30T00:00:00Z"
    assert "user-1" in body["assignments"]


def test_create_task_requires_title():
    result = json.loads(planner.planner_create_task("plan-1", ""))
    assert result["status"] == "error"


def test_update_task_fetches_etag_and_sends_if_match(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"etag-1"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_update_task("task-1", title="New title"))

    assert result["status"] == "success"
    assert mock_request.call_count == 2
    get_call, patch_call = mock_request.call_args_list
    assert get_call.kwargs["method"] == "GET"
    assert patch_call.kwargs["method"] == "PATCH"
    assert patch_call.kwargs["headers"]["If-Match"] == 'W/"etag-1"'
    assert patch_call.kwargs["json"] == {"title": "New title"}


def test_update_task_with_supplied_etag_skips_get(monkeypatch):
    mock_request = Mock(return_value=MockResponse({}, status_code=204, content=b""))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(
        planner.planner_update_task("task-1", title="New title", etag='W/"caller-etag"')
    )

    assert result["status"] == "success"
    assert mock_request.call_count == 1
    (patch_call,) = mock_request.call_args_list
    assert patch_call.kwargs["method"] == "PATCH"
    assert patch_call.kwargs["headers"]["If-Match"] == 'W/"caller-etag"'


def test_update_task_requires_at_least_one_field():
    result = json.loads(planner.planner_update_task("task-1"))
    assert result["status"] == "error"
    assert "at least one field" in result["message"]


def test_update_task_validates_percent_complete():
    result = json.loads(planner.planner_update_task("task-1", percent_complete=150))
    assert result["status"] == "error"
    assert "percent_complete" in result["message"]


def test_update_task_validates_priority():
    result = json.loads(planner.planner_update_task("task-1", priority=11))
    assert result["status"] == "error"
    assert "priority" in result["message"]


def test_delete_task_fetches_etag_and_sends_if_match(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"etag-1"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_delete_task("task-1"))

    assert result["status"] == "success"
    delete_call = mock_request.call_args_list[1]
    assert delete_call.kwargs["method"] == "DELETE"
    assert delete_call.kwargs["headers"]["If-Match"] == 'W/"etag-1"'


def test_assign_task_requires_user_ids():
    result = json.loads(planner.planner_assign_task("task-1", []))
    assert result["status"] == "error"


def test_unassign_task_sends_null_per_user(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"etag-1"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_unassign_task("task-1", ["user-1", "user-2"]))

    assert result["status"] == "success"
    patch_call = mock_request.call_args_list[1]
    assert patch_call.kwargs["json"] == {
        "assignments": {"user-1": None, "user-2": None}
    }


# ---------------------------------------------------------------------------
# task details / checklist
# ---------------------------------------------------------------------------


def test_update_task_description_fetches_etag(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"details-etag"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(
        planner.planner_update_task_description("task-1", "New description")
    )

    assert result["status"] == "success"
    get_call, patch_call = mock_request.call_args_list
    assert get_call.kwargs["url"].endswith("/planner/tasks/task-1/details")
    assert patch_call.kwargs["headers"]["If-Match"] == 'W/"details-etag"'
    assert patch_call.kwargs["json"] == {"description": "New description"}


def test_add_checklist_item_generates_uuid(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"details-etag"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_add_checklist_item("task-1", "Step 1"))

    assert result["status"] == "success"
    item_id = result["checklist_item_id"]
    patch_call = mock_request.call_args_list[1]
    checklist_body = patch_call.kwargs["json"]["checklist"]
    assert item_id in checklist_body
    assert checklist_body[item_id]["title"] == "Step 1"
    assert checklist_body[item_id]["isChecked"] is False


def test_add_checklist_item_requires_title():
    result = json.loads(planner.planner_add_checklist_item("task-1", "  "))
    assert result["status"] == "error"


def test_set_checklist_item_checked(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"details-etag"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(
        planner.planner_set_checklist_item_checked("task-1", "item-1", True)
    )

    assert result["status"] == "success"
    patch_call = mock_request.call_args_list[1]
    assert patch_call.kwargs["json"] == {
        "checklist": {
            "item-1": {
                "@odata.type": "microsoft.graph.plannerChecklistItem",
                "isChecked": True,
            }
        }
    }


def test_delete_checklist_item_sends_null(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"details-etag"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_delete_checklist_item("task-1", "item-1"))

    assert result["status"] == "success"
    patch_call = mock_request.call_args_list[1]
    assert patch_call.kwargs["json"] == {"checklist": {"item-1": None}}


# ---------------------------------------------------------------------------
# error propagation
# ---------------------------------------------------------------------------


def test_graph_error_response_is_surfaced_as_error(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"error": {"message": "Not Found"}}, status_code=404)
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_get_plan("plan-1"))

    assert result["status"] == "error"


def test_missing_auth_token_is_reported(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)

    result = json.loads(planner.planner_list_my_tasks())

    assert result["status"] == "error"
    assert "AUTH_TOKEN" in result["message"]
