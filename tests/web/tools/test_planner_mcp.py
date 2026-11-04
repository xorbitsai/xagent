import base64
import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import planner

TENANT_ID = "11111111-2222-3333-4444-555555555555"


def _graph_token(*, tenant_id: object = TENANT_ID) -> str:
    payload = {} if tenant_id is None else {"tid": tenant_id}
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )
    return f"header.{encoded}.signature"


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
    monkeypatch.setenv("AUTH_TOKEN", _graph_token())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_tenant_id_from_auth_token(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", _graph_token())

    assert planner._tenant_id_from_auth_token() == TENANT_ID


@pytest.mark.parametrize(
    "token",
    [
        "opaque-token",
        "header.bm90LWpzb24.signature",
        "header.W10.signature",
        _graph_token(tenant_id=None),
        _graph_token(tenant_id="invalid"),
        _graph_token(tenant_id=123),
    ],
)
def test_tenant_id_from_auth_token_rejects_unusable_tokens(monkeypatch, token):
    monkeypatch.setenv("AUTH_TOKEN", token)

    assert planner._tenant_id_from_auth_token() is None


def test_planner_web_url_uses_plan_and_tenant_ids():
    assert planner._planner_web_url("plan-1", TENANT_ID) == (
        f"https://planner.cloud.microsoft/webui/plan/plan-1/view/board?tid={TENANT_ID}"
    )


def test_planner_web_url_rejects_invalid_plan_id():
    assert planner._planner_web_url(" . ", TENANT_ID) is None


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


def test_etag_guarded_write_raises_conflict_on_412(monkeypatch):
    mock_request = Mock(return_value=MockResponse({}, status_code=412))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    with pytest.raises(planner._EtagConflictError):
        planner._etag_guarded_write("/planner/tasks/task-1", "DELETE", etag='W/"stale"')


def test_etag_guarded_write_raises_conflict_on_409(monkeypatch):
    """Microsoft's own "Planner resource versioning" docs state client
    apps must handle both 409 and 412 as versioning conflicts, not just
    412 -- outlook.py's single-code precedent doesn't apply here."""
    mock_request = Mock(return_value=MockResponse({}, status_code=409))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    with pytest.raises(planner._EtagConflictError):
        planner._etag_guarded_write("/planner/tasks/task-1", "DELETE", etag='W/"stale"')


def test_etag_guarded_write_reraises_other_graph_error(monkeypatch):
    mock_request = Mock(return_value=MockResponse({}, status_code=404))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    with pytest.raises(planner._GraphRequestError):
        planner._etag_guarded_write("/planner/tasks/task-1", "DELETE", etag='W/"x"')


def test_build_assignments_empty_returns_none():
    assert planner._build_assignments(None) is None
    assert planner._build_assignments([]) is None


def test_build_assignments_shape():
    """Each assignee in the same call must get a DISTINCT orderHint --
    reusing the same literal for 2+ assignees ties their order, the exact
    bug this file already fixed once for buckets (see _sequential_order_
    hints)."""
    result = planner._build_assignments(["user-1", "user-2"])
    assert result == {
        "user-1": {
            "@odata.type": "#microsoft.graph.plannerAssignment",
            "orderHint": " !",
        },
        "user-2": {
            "@odata.type": "#microsoft.graph.plannerAssignment",
            "orderHint": " ! !",
        },
    }


def test_sequential_order_hints_are_distinct_and_increasing():
    hints = planner._sequential_order_hints(4)
    assert hints == [" !", " ! !", " ! ! !", " ! ! ! !"]
    assert len(set(hints)) == len(hints)


def test_build_assignments_rejects_malformed_user_id():
    with pytest.raises(ValueError, match="user_id"):
        planner._build_assignments([" user-1 "])


def test_bounded_list_response_projects_all_items_and_preserves_next_link(
    monkeypatch,
):
    monkeypatch.setattr(planner, "get_tool_max_output_length", lambda: 12_000)
    assignments = {
        f"user-{index:02d}": {"@odata.type": "#microsoft.graph.plannerAssignment"}
        for index in range(20)
    }
    tasks = [
        {
            "id": f"task-{index:02d}",
            "title": f"Task {index}",
            "planId": "plan-1",
            "bucketId": "bucket-1",
            "assignments": assignments,
            "percentComplete": 0,
            "largeFutureField": "x" * 500,
        }
        for index in range(30)
    ]
    server_next_link = f"{planner.GRAPH_BASE_URL}/me/planner/tasks?%24skiptoken=next"

    response = planner._bounded_list_response(
        "tasks",
        tasks,
        next_link=server_next_link,
        retry_next_link=None,
    )
    result = json.loads(response)

    assert len(response) <= 12_000
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["projection"] == "summary_fields"
    assert len(result["tasks"]) == len(tasks)
    assert result["tasks"][0]["assignee_count"] == 20
    assert "assignments" not in result["tasks"][0]
    assert result["next_link"] == server_next_link


def test_bounded_list_response_does_not_advance_when_page_cannot_fit(monkeypatch):
    monkeypatch.setattr(planner, "get_tool_max_output_length", lambda: 300)
    current_page = f"{planner.GRAPH_BASE_URL}/me/planner/tasks?%24skiptoken=current"
    server_next_page = f"{planner.GRAPH_BASE_URL}/me/planner/tasks?%24skiptoken=next"
    tasks = [{"id": f"task-{index}", "title": "x" * 256} for index in range(50)]

    response = planner._bounded_list_response(
        "tasks",
        tasks,
        next_link=server_next_page,
        retry_next_link=current_page,
    )
    result = json.loads(response)

    assert len(response) <= 300
    assert result["status"] == "error"
    assert result["retry_next_link"] == current_page
    assert result.get("retry_next_link") != server_next_page


def test_bounded_list_response_uses_real_output_limit_env_name(monkeypatch):
    monkeypatch.setattr(planner, "get_tool_max_output_length", lambda: 350)
    response = planner._bounded_list_response(
        "tasks",
        [{"id": f"task-{index}", "title": "x" * 256} for index in range(50)],
        next_link=None,
        retry_next_link=None,
    )

    assert "XAGENT_TOOL_MAX_OUTPUT_LENGTH" in response


def test_bounded_plan_list_compact_projection_keeps_web_url(monkeypatch):
    monkeypatch.setattr(planner, "get_tool_max_output_length", lambda: 2_000)
    plans = [
        {
            "web_url": f"https://planner.cloud.microsoft/webui/plan/plan-{index}",
            "id": f"plan-{index}",
            "title": "x" * 256,
            "owner": "group-1",
            "createdDateTime": "2026-01-01T00:00:00Z",
        }
        for index in range(5)
    ]

    response = planner._bounded_list_response(
        "plans", plans, next_link=None, retry_next_link=None
    )
    result = json.loads(response)

    assert result["projection"] == "identity_and_label"
    assert all("web_url" in plan for plan in result["plans"])


def test_bounded_plan_list_compact_projection_keeps_null_web_url(monkeypatch):
    monkeypatch.setattr(planner, "get_tool_max_output_length", lambda: 800)
    plans = [
        {
            "web_url": None,
            "id": f"plan-{index}",
            "title": "t" * 80,
            "owner": "o" * 256,
            "createdDateTime": "2026-01-01T00:00:00Z",
        }
        for index in range(5)
    ]

    response = planner._bounded_list_response(
        "plans", plans, next_link=None, retry_next_link=None
    )
    result = json.loads(response)

    assert result["projection"] == "identity_and_label"
    assert all(plan["web_url"] is None for plan in result["plans"])


def test_bounded_group_list_compact_projection_keeps_display_name(monkeypatch):
    monkeypatch.setattr(planner, "get_tool_max_output_length", lambda: 550)
    groups = [
        {
            "id": f"group-{index}",
            "displayName": f"Team {index}",
            "description": "x" * 256,
            "largeFutureField": "y" * 256,
        }
        for index in range(5)
    ]

    response = planner._bounded_list_response(
        "groups", groups, next_link=None, retry_next_link=None
    )
    result = json.loads(response)

    assert result["projection"] == "identity_and_label"
    assert result["groups"][0] == {
        "id": "group-0",
        "displayName": "Team 0",
    }


@pytest.mark.parametrize("configured_limit", [18, 0, -1])
def test_bounded_envelopes_remain_valid_below_minimum(monkeypatch, configured_limit):
    monkeypatch.setattr(planner, "get_tool_max_output_length", lambda: configured_limit)

    response = planner._error("x" * 10_000)
    assert len(response) <= planner._MIN_OUTPUT_LENGTH
    assert json.loads(response)["status"] == "error"


@pytest.mark.parametrize("configured_limit", [18, 0, -1])
def test_success_with_capped_dict_remains_valid_below_minimum(
    monkeypatch, configured_limit
):
    """success_with_capped_dict (in .utils, used by planner_get_plan and
    every other single-object read/create tool) has no floor of its own on
    get_tool_max_output_length -- unlike planner.py's own _bounded_json --
    so an aggressively low configured cap could make even its final
    fallback exceed the limit and get corrupted by the platform's output
    filter."""
    from xagent.web.tools.mcp import utils as mcp_utils

    monkeypatch.setattr(
        mcp_utils, "get_tool_max_output_length", lambda: configured_limit
    )
    mock_request = Mock(
        return_value=MockResponse({"id": "plan-1", "title": "x" * 10_000})
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    response = planner.planner_get_plan("plan-1")

    assert len(response) <= mcp_utils._MIN_OUTPUT_LENGTH
    assert json.loads(response)["status"] == "success"


def test_resolve_list_path_defaults_when_no_next_link():
    assert planner._resolve_list_path("/planner/plans/plan-1/tasks", None) == (
        "/planner/plans/plan-1/tasks"
    )


def test_resolve_list_path_strips_graph_base_url():
    default_path = "/planner/plans/plan-1/tasks"
    next_link = f"{planner.GRAPH_BASE_URL}{default_path}?%24skip=50"
    assert planner._resolve_list_path(default_path, next_link) == (
        "/planner/plans/plan-1/tasks?%24skip=50"
    )


def test_resolve_list_path_rejects_foreign_url():
    """A next_link that doesn't point at Graph must be rejected outright --
    passing it through to _graph_request would let a forged next_link
    redirect this bearer-token-carrying request to an attacker-controlled
    host (SSRF)."""
    with pytest.raises(ValueError, match="next_link"):
        planner._resolve_list_path("/default", "https://evil.example/steal-token")


def test_resolve_list_path_rejects_non_string():
    with pytest.raises(ValueError, match="next_link"):
        planner._resolve_list_path("/default", 12345)  # type: ignore[arg-type]


def test_resolve_list_path_rejects_mismatched_collection():
    """Even a next_link that is a genuine graph.microsoft.com URL must be
    rejected if it points at a different collection than the tool's own
    default_path -- otherwise a forged next_link could redirect a call to
    e.g. planner_list_plans into fetching an unrelated resource such as
    /me/messages using the same shared AUTH_TOKEN."""
    other_collection = f"{planner.GRAPH_BASE_URL}/me/messages?%24skip=50"
    with pytest.raises(ValueError, match="next_link"):
        planner._resolve_list_path("/planner/plans/plan-1/tasks", other_collection)


def test_resolve_list_path_accepts_percent_encoding_case_difference():
    """RFC 3986 percent-encoded octets are case-insensitive (%2F == %2f), so
    a next_link differing from default_path only in hex-digit casing must
    still be accepted rather than rejected as a mismatched collection."""
    default_path = "/groups/plan%2Dgroup/planner/plans"
    next_link = f"{planner.GRAPH_BASE_URL}/groups/plan%2dgroup/planner/plans?%24skip=50"
    assert planner._resolve_list_path(default_path, next_link) == (
        "/groups/plan%2dgroup/planner/plans?%24skip=50"
    )


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------


def test_list_my_groups_success(monkeypatch):
    next_link = f"{planner.GRAPH_BASE_URL}/me/joinedTeams?%24skiptoken=next"
    mock_request = Mock(
        return_value=MockResponse(
            {
                "value": [
                    {
                        "id": "group-1",
                        "displayName": "Product",
                        "description": "Product team",
                    }
                ],
                "@odata.nextLink": next_link,
            }
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_list_my_groups())

    assert result == {
        "status": "success",
        "groups": [
            {
                "id": "group-1",
                "displayName": "Product",
                "description": "Product team",
            }
        ],
        "next_link": next_link,
        "truncated": False,
    }
    assert mock_request.call_args.kwargs["url"].endswith("/me/joinedTeams")


def test_list_my_groups_fetches_next_page(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {"value": [{"id": "group-2", "displayName": "Sales"}]}
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)
    next_link = f"{planner.GRAPH_BASE_URL}/me/joinedTeams?%24skiptoken=next"

    result = json.loads(planner.planner_list_my_groups(next_link=next_link))

    assert result["status"] == "success"
    assert result["groups"][0]["id"] == "group-2"
    assert mock_request.call_args.kwargs["url"] == next_link


def test_list_my_groups_rejects_other_graph_collection(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(planner.requests, "request", mock_request)
    next_link = f"{planner.GRAPH_BASE_URL}/me/memberOf?%24skiptoken=next"

    result = json.loads(planner.planner_list_my_groups(next_link=next_link))

    assert result["status"] == "error"
    assert "next_link" in result["message"]
    mock_request.assert_not_called()


def test_list_my_groups_missing_scope_asks_for_reconnect(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {
                "error": {
                    "code": "Authorization_RequestDenied",
                    "message": "Insufficient privileges to complete the operation.",
                }
            },
            status_code=403,
            url=f"{planner.GRAPH_BASE_URL}/me/joinedTeams",
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_list_my_groups())

    assert result["status"] == "error"
    assert "Open Connectors" in result["message"]
    assert "disconnect Planner" in result["message"]
    assert "Team.ReadBasic.All" in result["message"]
    assert "Authorization_RequestDenied" in result["message"]
    assert "Insufficient privileges" in result["message"]


def test_list_my_groups_other_403_preserves_graph_error_without_reconnect(monkeypatch):
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "error": {
                        "code": "accessDenied",
                        "message": "Access is blocked by a conditional access policy.",
                    }
                },
                status_code=403,
            )
        ),
    )

    result = json.loads(planner.planner_list_my_groups())

    assert result["status"] == "error"
    assert "accessDenied" in result["message"]
    assert "conditional access policy" in result["message"]
    assert "disconnect Planner" not in result["message"]


def test_list_my_groups_server_error_preserves_graph_error_without_reconnect(
    monkeypatch,
):
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "error": {
                        "code": "InternalError",
                        "message": "Planner is temporarily unavailable.",
                    }
                },
                status_code=500,
            )
        ),
    )

    result = json.loads(planner.planner_list_my_groups())

    assert result["status"] == "error"
    assert "InternalError" in result["message"]
    assert "temporarily unavailable" in result["message"]
    assert "disconnect Planner" not in result["message"]


def test_list_plans_success(monkeypatch):
    next_link = f"{planner.GRAPH_BASE_URL}/groups/group-1/planner/plans?%24skip=50"
    mock_request = Mock(
        return_value=MockResponse(
            {"value": [{"id": "plan-1"}], "@odata.nextLink": next_link}
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_list_plans("group-1"))

    assert result["status"] == "success"
    assert result["plans"] == [
        {
            "web_url": (
                "https://planner.cloud.microsoft/webui/plan/plan-1/view/board?"
                f"tid={TENANT_ID}"
            ),
            "id": "plan-1",
        }
    ]
    assert result["next_link"] == next_link
    assert mock_request.call_args.kwargs["url"].endswith(
        "/groups/group-1/planner/plans"
    )


def test_list_plans_fetches_next_page(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"value": [{"id": "plan-2"}]}))
    monkeypatch.setattr(planner.requests, "request", mock_request)
    next_link = f"{planner.GRAPH_BASE_URL}/groups/group-1/planner/plans?%24skip=50"

    result = json.loads(planner.planner_list_plans("group-1", next_link=next_link))

    assert result["status"] == "success"
    assert result["plans"][0]["id"] == "plan-2"
    assert result["plans"][0]["web_url"].endswith(f"?tid={TENANT_ID}")
    assert mock_request.call_args.kwargs["url"] == next_link


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        pytest.param("planner_list_plans", ("group-1",), id="group-plans"),
        pytest.param("planner_list_my_plans", (), id="shared-plans"),
    ],
)
def test_list_plans_preserves_null_web_url_and_missing_id(monkeypatch, tool_name, args):
    monkeypatch.setenv("AUTH_TOKEN", "opaque-token")
    mock_request = Mock(
        return_value=MockResponse({"value": [{"title": "No id"}, {"id": "plan-1"}]})
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(getattr(planner, tool_name)(*args))

    assert result["status"] == "success"
    assert result["plans"] == [
        {"web_url": None, "title": "No id"},
        {"web_url": None, "id": "plan-1"},
    ]


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        pytest.param("planner_list_plans", ("group-1",), id="group-plans"),
        pytest.param("planner_list_my_plans", (), id="shared-plans"),
    ],
)
def test_list_plans_decodes_tenant_once(monkeypatch, tool_name, args):
    tenant_id = Mock(return_value=TENANT_ID)
    monkeypatch.setattr(planner, "_tenant_id_from_auth_token", tenant_id)
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(
            return_value=MockResponse({"value": [{"id": "plan-1"}, {"id": "plan-2"}]})
        ),
    )

    result = json.loads(getattr(planner, tool_name)(*args))

    assert result["status"] == "success"
    tenant_id.assert_called_once_with()


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        pytest.param("planner_list_plans", ("group-1",), id="group-plans"),
        pytest.param("planner_list_my_plans", (), id="shared-plans"),
    ],
)
def test_list_plans_logs_missing_tenant_once(monkeypatch, caplog, tool_name, args):
    monkeypatch.setenv("AUTH_TOKEN", "opaque-token")
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(
            return_value=MockResponse({"value": [{"id": "plan-1"}, {"id": "plan-2"}]})
        ),
    )

    with caplog.at_level("WARNING", logger="planner-mcp"):
        result = json.loads(getattr(planner, tool_name)(*args))

    assert result["status"] == "success"
    assert (
        sum("no usable tenant id" in record.message for record in caplog.records) == 1
    )


def test_list_plans_rejects_forged_next_link(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(
        planner.planner_list_plans(
            "group-1", next_link="https://evil.example/steal-token"
        )
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_list_my_plans_success(monkeypatch):
    next_link = f"{planner.GRAPH_BASE_URL}/me/planner/plans?%24skip=50"
    mock_request = Mock(
        return_value=MockResponse(
            {
                "value": [
                    {
                        "id": "plan-1",
                        "title": "Launch",
                        "owner": "group-1",
                        "createdDateTime": "2026-10-15T00:00:00Z",
                    }
                ],
                "@odata.nextLink": next_link,
            }
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_list_my_plans())

    assert result == {
        "status": "success",
        "plans": [
            {
                "web_url": (
                    "https://planner.cloud.microsoft/webui/plan/plan-1/view/board?"
                    f"tid={TENANT_ID}"
                ),
                "id": "plan-1",
                "title": "Launch",
                "owner": "group-1",
                "createdDateTime": "2026-10-15T00:00:00Z",
            }
        ],
        "next_link": next_link,
        "truncated": False,
    }
    assert mock_request.call_args.kwargs["url"].endswith("/me/planner/plans")


def test_list_my_plans_fetches_next_page(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"value": [{"id": "plan-2"}]}))
    monkeypatch.setattr(planner.requests, "request", mock_request)
    next_link = f"{planner.GRAPH_BASE_URL}/me/planner/plans?%24skip=50"

    result = json.loads(planner.planner_list_my_plans(next_link=next_link))

    assert result["status"] == "success"
    assert result["plans"][0]["id"] == "plan-2"
    assert mock_request.call_args.kwargs["url"] == next_link


def test_list_my_plans_rejects_other_graph_collection(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(planner.requests, "request", mock_request)
    next_link = f"{planner.GRAPH_BASE_URL}/me/planner/tasks?%24skip=50"

    result = json.loads(planner.planner_list_my_plans(next_link=next_link))

    assert result["status"] == "error"
    assert "next_link" in result["message"]
    mock_request.assert_not_called()


def test_list_my_plans_rejects_non_list_collection(monkeypatch):
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(return_value=MockResponse({"value": {"id": "plan-1"}})),
    )

    result = json.loads(planner.planner_list_my_plans())

    assert result["status"] == "error"
    assert "non-list" in result["message"]


def test_create_plan_builds_container_url(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"id": "plan-1", "title": "New plan"})
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_create_plan("group-1", "New plan"))

    assert result["status"] == "success"
    assert result["plan"]["web_url"] == (
        f"https://planner.cloud.microsoft/webui/plan/plan-1/view/board?tid={TENANT_ID}"
    )
    kwargs = mock_request.call_args.kwargs
    assert kwargs["json"] == {
        "container": {"url": "https://graph.microsoft.com/v1.0/groups/group-1"},
        "title": "New plan",
    }


def test_create_plan_keeps_success_when_web_url_cannot_be_built(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "opaque-token")
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(return_value=MockResponse({"id": "plan-1", "title": "New plan"})),
    )

    result = json.loads(planner.planner_create_plan("group-1", "New plan"))

    assert result["status"] == "success"
    assert result["plan"]["web_url"] is None
    assert result["plan"]["id"] == "plan-1"


def test_create_plan_keeps_success_when_plan_id_is_invalid(monkeypatch):
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(return_value=MockResponse({"id": " . ", "title": "New plan"})),
    )

    result = json.loads(planner.planner_create_plan("group-1", "New plan"))

    assert result["status"] == "success"
    assert result["plan"]["web_url"] is None
    assert result["plan"]["id"] == " . "


def test_create_plan_403_names_group_and_preserves_graph_error(monkeypatch):
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {
                    "error": {
                        "code": "Forbidden",
                        "message": "The signed-in user is not a member of the group.",
                    }
                },
                status_code=403,
            )
        ),
    )

    result = json.loads(
        planner.planner_create_plan("group-1", "New plan", group_name="Product Team")
    )

    assert result["status"] == "error"
    assert '"Product Team" (group-1)' in result["message"]
    assert "planner_list_my_groups" in result["message"]
    assert "Forbidden" in result["message"]
    assert "not a member" in result["message"]


def test_create_plan_transport_failure_is_indeterminate(monkeypatch):
    monkeypatch.setattr(
        planner.requests, "request", Mock(side_effect=requests.Timeout("timed out"))
    )

    result = json.loads(planner.planner_create_plan("group-1", "New plan"))

    assert result["status"] == "indeterminate"
    assert result["retryable"] is False
    assert result["mutation_may_have_completed"] is True
    assert result["reconciliation"] == {
        "read_tool": "planner_list_plans",
        "group_id": "group-1",
        "match": {"title": "New plan"},
    }


def test_indeterminate_reconciliation_survives_low_output_cap(monkeypatch):
    """A reconciliation-bearing indeterminate envelope (read_tool + a
    Planner id, no free-text message) runs 190-240+ chars -- comfortably
    over _MIN_OUTPUT_LENGTH (64) but under _INDETERMINATE_MIN_OUTPUT_LENGTH
    (320). An operator-configured cap in that range must not silently drop
    the reconciliation identity, or the whole point of this response --
    telling the caller what to check before retrying -- is lost exactly
    when it matters."""
    monkeypatch.setattr(planner, "get_tool_max_output_length", lambda: 100)
    monkeypatch.setattr(
        planner.requests, "request", Mock(side_effect=requests.Timeout("timed out"))
    )

    result = json.loads(planner.planner_create_plan("group-1", "New plan"))

    assert result["status"] == "indeterminate"
    assert result["reconciliation"]["read_tool"] == "planner_list_plans"
    assert result["reconciliation"]["group_id"] == "group-1"


def test_create_plan_unreadable_success_is_indeterminate(monkeypatch):
    response = MockResponse({"id": "plan-1"})
    response.json = Mock(side_effect=ValueError("invalid JSON"))
    monkeypatch.setattr(planner.requests, "request", Mock(return_value=response))

    result = json.loads(planner.planner_create_plan("group-1", "New plan"))

    assert result["status"] == "indeterminate"


@pytest.mark.parametrize("response_body", [{}, [], {"id": ""}])
def test_create_plan_anomalous_success_is_indeterminate(monkeypatch, response_body):
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(return_value=MockResponse(response_body)),
    )

    result = json.loads(planner.planner_create_plan("group-1", "New plan"))

    assert result["status"] == "indeterminate"
    assert result["mutation_may_have_completed"] is True


def test_create_plan_server_failure_is_indeterminate(monkeypatch):
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(return_value=MockResponse({"error": "gateway"}, status_code=503)),
    )

    result = json.loads(planner.planner_create_plan("group-1", "New plan"))

    assert result["status"] == "indeterminate"


def test_create_plan_http_timeout_is_indeterminate(monkeypatch):
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(return_value=MockResponse({"error": "timeout"}, status_code=408)),
    )

    result = json.loads(planner.planner_create_plan("group-1", "New plan"))

    assert result["status"] == "indeterminate"
    assert result["retryable"] is False


def test_create_plan_requires_title():
    result = json.loads(planner.planner_create_plan("group-1", "   "))
    assert result["status"] == "error"


def test_create_plan_strips_title_whitespace(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "plan-1"}))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    planner.planner_create_plan("group-1", "  New plan  ")

    assert mock_request.call_args.kwargs["json"]["title"] == "New plan"


def test_get_plan_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "plan-1", "title": "Plan"}))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_get_plan("plan-1"))

    assert result["status"] == "success"
    assert result["plan"] == {
        "web_url": (
            "https://planner.cloud.microsoft/webui/plan/plan-1/view/board?"
            f"tid={TENANT_ID}"
        ),
        "id": "plan-1",
        "title": "Plan",
    }
    assert mock_request.call_args.kwargs["url"].endswith("/planner/plans/plan-1")


def test_get_plan_preserves_success_without_tenant(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "opaque-token")
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(return_value=MockResponse({"id": "plan-1", "title": "Plan"})),
    )

    result = json.loads(planner.planner_get_plan("plan-1"))

    assert result["status"] == "success"
    assert result["plan"]["web_url"] is None


def test_get_plan_caps_large_singleton_response(monkeypatch):
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "300")
    mock_request = Mock(
        return_value=MockResponse(
            {"id": "plan-1", "large": {str(index): "x" * 80 for index in range(50)}}
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    response = planner.planner_get_plan("plan-1")
    result = json.loads(response)

    assert len(response) <= 300
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["plan"]["web_url"].endswith(f"?tid={TENANT_ID}")


def test_get_plan_caps_and_redacts_large_graph_error(monkeypatch):
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "300")
    response = MockResponse(
        content=(b"Authorization: Bearer secret-token " + b"x" * 10_000),
        status_code=400,
    )
    monkeypatch.setattr(planner.requests, "request", Mock(return_value=response))

    serialized = planner.planner_get_plan("plan-1")
    result = json.loads(serialized)

    assert len(serialized) <= 300
    assert result["status"] == "error"
    assert "secret-token" not in serialized


def test_get_plan_rejects_and_bounds_non_object_response(monkeypatch):
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "300")
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(return_value=MockResponse(["x" * 10_000])),
    )

    serialized = planner.planner_get_plan("plan-1")
    result = json.loads(serialized)

    assert len(serialized) <= 300
    assert result["status"] == "error"
    assert "invalid plan object" in result["message"]


def test_get_plan_transport_failure_does_not_expose_exception_text(monkeypatch):
    monkeypatch.setattr(
        planner.requests,
        "request",
        Mock(side_effect=requests.Timeout("token=secret-transport-value")),
    )

    serialized = planner.planner_get_plan("plan-1")
    result = json.loads(serialized)

    assert result["status"] == "error"
    assert "secret-transport-value" not in serialized
    assert "before a response was received" in result["message"]


# ---------------------------------------------------------------------------
# buckets
# ---------------------------------------------------------------------------


def test_list_buckets_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {
                "value": [{"id": "bucket-1"}],
                "@odata.nextLink": f"{planner.GRAPH_BASE_URL}/planner/plans/plan-1/buckets?%24skip=50",
            }
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_list_buckets("plan-1"))

    assert result["status"] == "success"
    assert result["buckets"] == [{"id": "bucket-1"}]
    assert result["next_link"] == (
        f"{planner.GRAPH_BASE_URL}/planner/plans/plan-1/buckets?%24skip=50"
    )
    assert mock_request.call_args.kwargs["url"].endswith(
        "/planner/plans/plan-1/buckets"
    )


def test_list_buckets_fetches_next_page(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"value": [{"id": "bucket-2"}]}))
    monkeypatch.setattr(planner.requests, "request", mock_request)
    next_link = f"{planner.GRAPH_BASE_URL}/planner/plans/plan-1/buckets?%24skip=50"

    result = json.loads(planner.planner_list_buckets("plan-1", next_link=next_link))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"] == next_link


def test_create_bucket_omits_order_hint(monkeypatch):
    """orderHint is intentionally left unset so the service auto-generates
    it -- sending the same literal for every bucket would give the 2nd+
    bucket in a plan an identical, unordered hint (see module docstring)."""
    mock_request = Mock(return_value=MockResponse({"id": "bucket-1"}))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_create_bucket("plan-1", "To do"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"] == {
        "name": "To do",
        "planId": "plan-1",
    }


def test_create_bucket_rejects_malformed_plan_id():
    result = json.loads(planner.planner_create_bucket(" plan-1 ", "To do"))
    assert result["status"] == "error"
    assert "plan_id" in result["message"]


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


def test_list_tasks_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {
                "value": [{"id": "task-1"}],
                "@odata.nextLink": f"{planner.GRAPH_BASE_URL}/planner/plans/plan-1/tasks?%24skip=50",
            }
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_list_tasks("plan-1"))

    assert result["status"] == "success"
    assert result["tasks"] == [{"id": "task-1"}]
    assert result["next_link"] == (
        f"{planner.GRAPH_BASE_URL}/planner/plans/plan-1/tasks?%24skip=50"
    )
    assert mock_request.call_args.kwargs["url"].endswith("/planner/plans/plan-1/tasks")


def test_list_tasks_fetches_next_page(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"value": [{"id": "task-2"}]}))
    monkeypatch.setattr(planner.requests, "request", mock_request)
    next_link = f"{planner.GRAPH_BASE_URL}/planner/plans/plan-1/tasks?%24skip=50"

    result = json.loads(planner.planner_list_tasks("plan-1", next_link=next_link))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"] == next_link


def test_list_my_tasks_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {
                "value": [{"id": "task-1"}],
                "@odata.nextLink": f"{planner.GRAPH_BASE_URL}/me/planner/tasks?%24skip=50",
            }
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_list_my_tasks())

    assert result["status"] == "success"
    assert result["tasks"] == [{"id": "task-1"}]
    assert (
        result["next_link"] == f"{planner.GRAPH_BASE_URL}/me/planner/tasks?%24skip=50"
    )
    assert mock_request.call_args.kwargs["url"].endswith("/me/planner/tasks")


def test_list_my_tasks_fetches_next_page(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"value": [{"id": "task-2"}]}))
    monkeypatch.setattr(planner.requests, "request", mock_request)
    next_link = f"{planner.GRAPH_BASE_URL}/me/planner/tasks?%24skip=50"

    result = json.loads(planner.planner_list_my_tasks(next_link=next_link))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"] == next_link


def test_get_task_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"id": "task-1", "title": "Write report"})
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_get_task("task-1"))

    assert result["status"] == "success"
    assert result["task"] == {"id": "task-1", "title": "Write report"}
    assert mock_request.call_args.kwargs["url"].endswith("/planner/tasks/task-1")


def test_get_task_details_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"description": "Notes", "checklist": {}})
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_get_task_details("task-1"))

    assert result["status"] == "success"
    assert result["details"] == {"description": "Notes", "checklist": {}}
    assert mock_request.call_args.kwargs["url"].endswith(
        "/planner/tasks/task-1/details"
    )


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
    assert body["assignments"]["user-1"]["orderHint"] == " !"


def test_create_task_requires_title():
    result = json.loads(planner.planner_create_task("plan-1", ""))
    assert result["status"] == "error"


def test_create_task_rejects_malformed_plan_id():
    result = json.loads(planner.planner_create_task(" plan-1 ", "Write report"))
    assert result["status"] == "error"
    assert "plan_id" in result["message"]


def test_create_task_rejects_malformed_bucket_id():
    result = json.loads(
        planner.planner_create_task("plan-1", "Write report", bucket_id=" bucket-1 ")
    )
    assert result["status"] == "error"
    assert "bucket_id" in result["message"]


def test_create_task_rejects_empty_bucket_id():
    """Unlike omitting bucket_id, an explicit empty string is rejected --
    consistent with planner_update_task's handling of the same parameter,
    rather than silently creating the task unbucketed."""
    result = json.loads(
        planner.planner_create_task("plan-1", "Write report", bucket_id="")
    )
    assert result["status"] == "error"
    assert "bucket_id" in result["message"]


def test_create_task_rejects_empty_due_date_time():
    result = json.loads(
        planner.planner_create_task("plan-1", "Write report", due_date_time="  ")
    )
    assert result["status"] == "error"
    assert "due_date_time" in result["message"]


def test_create_task_rejects_malformed_assignee_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "task-1"}))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(
        planner.planner_create_task(
            "plan-1", "Write report", assignee_user_ids=[" user-1 "]
        )
    )

    assert result["status"] == "error"
    assert "user_id" in result["message"]


def test_create_task_rejects_non_list_assignee_user_ids():
    """A bare string would otherwise be iterated character-by-character by
    _build_assignments, silently assigning to single-character "user ids"
    instead of raising -- matches the isinstance(list) guard already on
    planner_assign_task/planner_unassign_task."""
    result = json.loads(
        planner.planner_create_task(
            "plan-1", "Write report", assignee_user_ids="user-1"
        )
    )
    assert result["status"] == "error"
    assert "assignee_user_ids" in result["message"]


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


def test_update_task_rejects_blank_title():
    result = json.loads(planner.planner_update_task("task-1", title="   "))
    assert result["status"] == "error"
    assert "title" in result["message"]


def test_update_task_rejects_malformed_bucket_id():
    result = json.loads(planner.planner_update_task("task-1", bucket_id=" bucket-1 "))
    assert result["status"] == "error"
    assert "bucket_id" in result["message"]


def test_update_task_clears_bucket_due_and_start_date_with_empty_string(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"etag-1"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(
        planner.planner_update_task(
            "task-1", bucket_id="", due_date_time="", start_date_time=""
        )
    )

    assert result["status"] == "success"
    patch_call = mock_request.call_args_list[1]
    assert patch_call.kwargs["json"] == {
        "bucketId": None,
        "dueDateTime": None,
        "startDateTime": None,
    }


def test_update_task_rejects_blank_due_date_time():
    """Unlike "" (the clear-the-field sentinel), a whitespace-only value
    is neither a real timestamp nor a clear request and must be rejected
    locally rather than forwarded to Graph as an opaque 400."""
    result = json.loads(planner.planner_update_task("task-1", due_date_time="   "))
    assert result["status"] == "error"
    assert "due_date_time" in result["message"]


def test_update_task_strips_due_and_start_date_time(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"etag-1"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(
        planner.planner_update_task(
            "task-1",
            due_date_time=" 2026-09-30T00:00:00Z ",
            start_date_time=" 2026-09-01T00:00:00Z ",
        )
    )

    assert result["status"] == "success"
    patch_call = mock_request.call_args_list[1]
    assert patch_call.kwargs["json"] == {
        "dueDateTime": "2026-09-30T00:00:00Z",
        "startDateTime": "2026-09-01T00:00:00Z",
    }


def test_update_task_rejects_blank_start_date_time():
    result = json.loads(planner.planner_update_task("task-1", start_date_time="   "))
    assert result["status"] == "error"
    assert "start_date_time" in result["message"]


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


def test_delete_task_returns_conflict_on_stale_etag(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"etag-1"'}),
            MockResponse({}, status_code=412),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_delete_task("task-1"))

    assert result["status"] == "conflict_stale_version"


def _delete_plan(**kwargs):
    return planner.planner_delete_plan("plan-1", **kwargs)


def _delete_bucket(**kwargs):
    return planner.planner_delete_bucket("plan-1", "bucket-1", **kwargs)


_DELETE_TOOLS = [
    pytest.param(_delete_plan, "/planner/plans/plan-1", id="plan"),
    pytest.param(_delete_bucket, "/planner/buckets/bucket-1", id="bucket"),
]


@pytest.mark.parametrize("tool_name", ["planner_delete_plan", "planner_delete_bucket"])
def test_delete_container_declares_destructive_idempotent_annotation(tool_name):
    tool = planner.mcp._tool_manager.get_tool(tool_name)

    assert tool.annotations is not None
    assert tool.annotations.destructiveHint is True
    assert tool.annotations.idempotentHint is True


@pytest.mark.parametrize(("delete", "path"), _DELETE_TOOLS)
def test_delete_container_fetches_etag_and_sends_if_match(monkeypatch, delete, path):
    responses = iter(
        [
            MockResponse({"id": "x", "@odata.etag": 'W/"etag-1"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(delete())

    assert result["status"] == "success"
    get_call, delete_call = mock_request.call_args_list
    assert get_call.kwargs["method"] == "GET"
    assert get_call.kwargs["url"] == f"{planner.GRAPH_BASE_URL}{path}"
    assert delete_call.kwargs["method"] == "DELETE"
    assert delete_call.kwargs["url"] == f"{planner.GRAPH_BASE_URL}{path}"
    assert delete_call.kwargs["headers"]["If-Match"] == 'W/"etag-1"'


@pytest.mark.parametrize(("delete", "path"), _DELETE_TOOLS)
def test_delete_container_uses_supplied_etag(monkeypatch, delete, path):
    mock_request = Mock(return_value=MockResponse({}, status_code=204, content=b""))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(delete(etag='W/"supplied"'))

    assert result["status"] == "success"
    mock_request.assert_called_once()
    assert mock_request.call_args.kwargs["method"] == "DELETE"
    assert mock_request.call_args.kwargs["url"] == f"{planner.GRAPH_BASE_URL}{path}"
    assert mock_request.call_args.kwargs["headers"]["If-Match"] == 'W/"supplied"'


@pytest.mark.parametrize("status_code", [409, 412])
@pytest.mark.parametrize(("delete", "path"), _DELETE_TOOLS)
def test_delete_container_returns_conflict_on_stale_etag(
    monkeypatch, delete, path, status_code
):
    mock_request = Mock(return_value=MockResponse({}, status_code=status_code))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(delete(etag='W/"stale"'))

    assert result["status"] == "conflict_stale_version"
    assert "not deleted" in result["message"]
    assert "confirm with the user" in result["message"]
    mock_request.assert_called_once()
    assert mock_request.call_args.kwargs["url"] == f"{planner.GRAPH_BASE_URL}{path}"


@pytest.mark.parametrize(
    ("etag", "failing_method"),
    [(None, "GET"), ('W/"etag-1"', "DELETE")],
    ids=["on-etag-read", "on-delete"],
)
@pytest.mark.parametrize(("delete", "path"), _DELETE_TOOLS)
def test_delete_container_403_names_group_membership(
    monkeypatch, delete, path, etag, failing_method
):
    mock_request = Mock(
        return_value=MockResponse(
            {
                "error": {
                    "code": "Forbidden",
                    "message": "You do not have the required permissions.",
                }
            },
            status_code=403,
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(delete(etag=etag))

    assert result["status"] == "error"
    assert "member of the Microsoft 365 group" in result["message"]
    assert "Forbidden" in result["message"]
    assert "disconnect Planner" not in result["message"]
    mock_request.assert_called_once()
    assert mock_request.call_args.kwargs["method"] == failing_method
    assert mock_request.call_args.kwargs["url"] == f"{planner.GRAPH_BASE_URL}{path}"


@pytest.mark.parametrize(
    ("etag", "failing_method"),
    [(None, "GET"), ('W/"etag-1"', "DELETE")],
    ids=["on-etag-read", "on-delete"],
)
@pytest.mark.parametrize(("delete", "path"), _DELETE_TOOLS)
def test_delete_container_missing_scope_asks_for_reconnect(
    monkeypatch, delete, path, etag, failing_method
):
    mock_request = Mock(
        return_value=MockResponse(
            {
                "error": {
                    "code": "Authorization_RequestDenied",
                    "message": "Insufficient privileges to complete the operation.",
                }
            },
            status_code=403,
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(delete(etag=etag))

    assert result["status"] == "error"
    assert "disconnect Planner" in result["message"]
    assert "Tasks.ReadWrite" in result["message"]
    assert "member of the Microsoft 365 group" not in result["message"]
    mock_request.assert_called_once()
    assert mock_request.call_args.kwargs["method"] == failing_method
    assert mock_request.call_args.kwargs["url"] == f"{planner.GRAPH_BASE_URL}{path}"


@pytest.mark.parametrize(
    ("etag", "failing_method"),
    [(None, "GET"), ('W/"etag-1"', "DELETE")],
    ids=["on-etag-read", "on-delete"],
)
@pytest.mark.parametrize(("delete", "path"), _DELETE_TOOLS)
def test_delete_container_404_reports_possibly_already_deleted(
    monkeypatch, delete, path, etag, failing_method
):
    mock_request = Mock(
        return_value=MockResponse(
            {"error": {"code": "NotFound", "message": "Not found."}},
            status_code=404,
        )
    )
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(delete(etag=etag))

    assert result["status"] == "error"
    assert "not found" in result["message"]
    assert "already been deleted" in result["message"]
    mock_request.assert_called_once()
    assert mock_request.call_args.kwargs["method"] == failing_method
    assert mock_request.call_args.kwargs["url"] == f"{planner.GRAPH_BASE_URL}{path}"


@pytest.mark.parametrize(
    ("delete", "reconciliation"),
    [
        pytest.param(
            _delete_plan,
            {"read_tool": "planner_get_plan", "plan_id": "plan-1"},
            id="plan",
        ),
        pytest.param(
            _delete_bucket,
            {
                "read_tool": "planner_list_buckets",
                "plan_id": "plan-1",
                "bucket_id": "bucket-1",
            },
            id="bucket",
        ),
    ],
)
def test_delete_container_transport_failure_is_indeterminate(
    monkeypatch, delete, reconciliation
):
    monkeypatch.setattr(
        planner.requests, "request", Mock(side_effect=requests.Timeout("timed out"))
    )

    result = json.loads(delete(etag='W/"etag-1"'))

    assert result["status"] == "indeterminate"
    assert result["retryable"] is False
    assert result["reconciliation"] == reconciliation


def test_delete_bucket_rejects_malformed_plan_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_delete_bucket(" plan-1 ", "bucket-1"))

    assert result["status"] == "error"
    assert "plan_id" in result["message"]
    mock_request.assert_not_called()


def test_update_task_returns_conflict_on_stale_etag(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"etag-1"'}),
            MockResponse({}, status_code=412),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_update_task("task-1", title="New title"))

    assert result["status"] == "conflict_stale_version"


def test_assign_task_requires_user_ids():
    result = json.loads(planner.planner_assign_task("task-1", []))
    assert result["status"] == "error"


def test_assign_task_rejects_non_list_user_ids():
    result = json.loads(planner.planner_assign_task("task-1", "user-1"))
    assert result["status"] == "error"
    assert "user_ids" in result["message"]


def test_assign_task_sends_body_and_if_match(monkeypatch):
    responses = iter(
        [
            MockResponse({"id": "task-1", "@odata.etag": 'W/"etag-1"'}),
            MockResponse({}, status_code=204, content=b""),
        ]
    )
    mock_request = Mock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(planner.requests, "request", mock_request)

    result = json.loads(planner.planner_assign_task("task-1", ["user-1"]))

    assert result["status"] == "success"
    patch_call = mock_request.call_args_list[1]
    assert patch_call.kwargs["headers"]["If-Match"] == 'W/"etag-1"'
    assert patch_call.kwargs["json"] == {
        "assignments": {
            "user-1": {
                "@odata.type": "#microsoft.graph.plannerAssignment",
                "orderHint": " !",
            }
        }
    }


def test_assign_task_transport_failure_is_indeterminate(monkeypatch):
    monkeypatch.setattr(
        planner.requests, "request", Mock(side_effect=requests.Timeout("timed out"))
    )

    result = json.loads(
        planner.planner_assign_task("task-1", ["user-1"], etag='W/"etag"')
    )

    assert result["status"] == "indeterminate"
    assert result["mutation_may_have_completed"] is True
    assert result["reconciliation"] == {
        "read_tool": "planner_get_task",
        "task_id": "task-1",
    }


def test_unassign_task_rejects_non_list_user_ids():
    result = json.loads(planner.planner_unassign_task("task-1", "user-1"))
    assert result["status"] == "error"
    assert "user_ids" in result["message"]


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


def test_unassign_task_rejects_malformed_user_id():
    result = json.loads(planner.planner_unassign_task("task-1", [" user-1 "]))
    assert result["status"] == "error"
    assert "user_id" in result["message"]


def test_assign_task_rejects_malformed_user_id():
    result = json.loads(planner.planner_assign_task("task-1", [" user-1 "]))
    assert result["status"] == "error"
    assert "user_id" in result["message"]


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


def test_add_checklist_item_indeterminate_keeps_reconciliation_id(monkeypatch):
    item_id = "00000000-0000-0000-0000-000000000001"
    monkeypatch.setattr(planner.uuid, "uuid4", Mock(return_value=item_id))
    monkeypatch.setattr(
        planner.requests, "request", Mock(side_effect=requests.Timeout("timed out"))
    )

    result = json.loads(
        planner.planner_add_checklist_item("task-1", "Step 1", etag='W/"details-etag"')
    )

    assert result["status"] == "indeterminate"
    assert result["reconciliation"] == {
        "read_tool": "planner_get_task_details",
        "task_id": "task-1",
        "checklist_item_id": item_id,
    }


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


def test_set_checklist_item_checked_rejects_malformed_item_id():
    result = json.loads(
        planner.planner_set_checklist_item_checked("task-1", " item-1 ", True)
    )
    assert result["status"] == "error"
    assert "item_id" in result["message"]


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


def test_delete_checklist_item_rejects_malformed_item_id():
    result = json.loads(planner.planner_delete_checklist_item("task-1", " item-1 "))
    assert result["status"] == "error"
    assert "item_id" in result["message"]


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
