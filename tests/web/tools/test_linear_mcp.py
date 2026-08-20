import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import linear


class MockResponse:
    def __init__(
        self,
        json_data=None,
        status_code: int = 200,
        text: str = "",
        headers: dict | None = None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.headers = headers or {}

    def json(self):
        return self._json_data


_TEAM_UUID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("LINEAR_ACCESS_TOKEN", "access-token")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="LINEAR_ACCESS_TOKEN"):
        linear._headers()


def test_headers_include_bearer_token_and_json_content_type():
    assert linear._headers() == {
        "Authorization": "Bearer access-token",
        "Content-Type": "application/json",
    }


def test_clamp_limit_caps_at_max():
    assert linear._clamp_limit(500) == linear.MAX_LIMIT


def test_clamp_limit_floors_zero_and_negative_to_one():
    assert linear._clamp_limit(0) == 1
    assert linear._clamp_limit(-5) == 1


def test_clamp_limit_leaves_in_range_value_unchanged():
    assert linear._clamp_limit(10) == 10


def test_graphql_sends_query_and_variables_as_json_body(monkeypatch):
    mock_post = Mock(return_value=MockResponse(json_data={"data": {"ok": True}}))
    monkeypatch.setattr(linear.requests, "post", mock_post)

    data, errors = linear._graphql("query { ok }", {"a": 1})

    assert data == {"ok": True}
    assert errors == []
    assert mock_post.call_args.args[0] == linear.LINEAR_GRAPHQL_URL
    assert mock_post.call_args.kwargs["json"] == {
        "query": "query { ok }",
        "variables": {"a": 1},
    }


def test_graphql_raises_on_top_level_errors_with_200_status(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={"errors": [{"message": "Field not found"}]}
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Field not found"):
        linear._graphql("query { bad }")


def test_graphql_raises_with_structured_error_body_on_http_error(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data={"errors": [{"message": "Invalid token"}]},
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Invalid token"):
        linear._graphql("query { viewer { id } }")


def test_graphql_truncates_unstructured_error_body_on_http_error(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        linear._graphql("query { viewer { id } }")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def _ratelimited_response(reset_epoch_ms: str | None) -> "MockResponse":
    headers = {}
    if reset_epoch_ms is not None:
        headers["X-RateLimit-Requests-Reset"] = reset_epoch_ms
    return MockResponse(
        status_code=400,
        json_data={
            "errors": [
                {"message": "Rate limited", "extensions": {"code": "RATELIMITED"}}
            ]
        },
        headers=headers,
    )


def test_graphql_retries_once_on_ratelimited_and_waits_until_reset(monkeypatch):
    """Linear signals rate limiting as HTTP 400 + a RATELIMITED extensions
    code, with the reset time in X-RateLimit-Requests-Reset (epoch ms) --
    not HTTP 429 / Retry-After like the REST-shaped Slack/Intercom
    siblings."""
    now = 1_700_000_000.0
    reset_ms = str(int((now + 5) * 1000))
    mock_post = Mock(
        side_effect=[
            _ratelimited_response(reset_ms),
            MockResponse(json_data={"data": {"viewer": {"id": "u1"}}}),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)
    monkeypatch.setattr(linear.time, "time", lambda: now)
    sleep_calls = []
    monkeypatch.setattr(linear.time, "sleep", sleep_calls.append)

    data, errors = linear._graphql("query { viewer { id } }")

    assert data == {"viewer": {"id": "u1"}}
    assert errors == []
    assert mock_post.call_count == 2
    assert sleep_calls == [5.0]


def test_graphql_does_not_retry_on_plain_429_without_ratelimited_extension(monkeypatch):
    """A REST-shaped 429 (no RATELIMITED extensions code) is not part of
    Linear's documented rate-limit contract and must not trigger a retry."""
    mock_post = Mock(
        return_value=MockResponse(status_code=429, headers={"Retry-After": "1"})
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    with pytest.raises(RuntimeError):
        linear._graphql("query { viewer { id } }")

    assert mock_post.call_count == 1


def test_graphql_does_not_retry_when_reset_wait_exceeds_max_retry_after(monkeypatch):
    now = 1_700_000_000.0
    reset_ms = str(int((now + linear.MAX_RETRY_AFTER_SECONDS + 1) * 1000))
    mock_post = Mock(return_value=_ratelimited_response(reset_ms))
    monkeypatch.setattr(linear.requests, "post", mock_post)
    monkeypatch.setattr(linear.time, "time", lambda: now)

    with pytest.raises(RuntimeError):
        linear._graphql("query { viewer { id } }")

    assert mock_post.call_count == 1


def test_graphql_retries_immediately_when_reset_has_already_elapsed(monkeypatch):
    """A reset timestamp already in the past (clock skew, or the response
    just arrived after the window rolled over) computes wait_seconds=0.0 --
    that must still retry (immediately), not be treated the same as no
    header/information at all."""
    now = 1_700_000_000.0
    reset_ms = str(int((now - 5) * 1000))
    mock_post = Mock(
        side_effect=[
            _ratelimited_response(reset_ms),
            MockResponse(json_data={"data": {"viewer": {"id": "u1"}}}),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)
    monkeypatch.setattr(linear.time, "time", lambda: now)
    sleep_calls = []
    monkeypatch.setattr(linear.time, "sleep", sleep_calls.append)

    data, errors = linear._graphql("query { viewer { id } }")

    assert data == {"viewer": {"id": "u1"}}
    assert mock_post.call_count == 2
    assert sleep_calls == [0.0]


def test_graphql_does_not_retry_when_reset_header_missing(monkeypatch):
    mock_post = Mock(return_value=_ratelimited_response(None))
    monkeypatch.setattr(linear.requests, "post", mock_post)

    with pytest.raises(RuntimeError):
        linear._graphql("query { viewer { id } }")

    assert mock_post.call_count == 1


def test_rate_limit_wait_seconds_returns_none_for_missing_header():
    """Pins the return-value contract directly, not just the resulting
    no-retry behavior -- a future regression that reintroduces 0.0 as the
    missing-header sentinel would still pass the no-retry behavioral test
    (0.0 is also within retry range) without this."""
    response = _ratelimited_response(None)

    assert linear._rate_limit_wait_seconds(response) is None


def test_rate_limit_wait_seconds_returns_float_for_present_header(monkeypatch):
    now = 1_700_000_000.0
    response = _ratelimited_response(str(int((now + 5) * 1000)))
    monkeypatch.setattr(linear.time, "time", lambda: now)

    assert linear._rate_limit_wait_seconds(response) == 5.0


def test_graphql_returns_partial_data_and_errors_instead_of_raising(
    monkeypatch, caplog
):
    """A 200 response can carry both usable data and a truthy errors array
    when only one sub-field's resolver failed -- the whole response must
    not be discarded over that one bad field, and the errors must be
    returned to the caller (not only logged) so tool functions can surface
    them as warnings."""
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"viewer": {"id": "u1"}},
                    "errors": [{"message": "some.unrelated.field failed"}],
                }
            )
        ),
    )

    with caplog.at_level("WARNING"):
        data, errors = linear._graphql("query { viewer { id } }")

    assert data == {"viewer": {"id": "u1"}}
    assert errors == [{"message": "some.unrelated.field failed"}]
    assert "some.unrelated.field failed" in caplog.text


def test_graphql_raises_when_the_requested_root_field_is_null_with_errors(monkeypatch):
    """Every query/mutation in this module selects exactly one top-level
    field. If that field is null and errors is non-empty (e.g. a permission
    failure on the requested object), there is nothing usable to return --
    this must raise, not come back as a quiet {"issue": None} that a caller
    could mistake for "not found"."""
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"issue": None},
                    "errors": [{"message": "You do not have permission"}],
                }
            )
        ),
    )

    with pytest.raises(RuntimeError, match="You do not have permission"):
        linear._graphql("query($id: String!) { issue(id: $id) { id } }", {"id": "x"})


def test_graphql_raises_on_non_dict_200_body(monkeypatch):
    """A non-object 200 JSON body (e.g. a bare list or string) must raise a
    clear RuntimeError, not an unhandled AttributeError from .get()."""
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(return_value=MockResponse(json_data=["unexpected", "list"])),
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        linear._graphql("query { viewer { id } }")


def test_graphql_raises_on_non_json_200_body(monkeypatch):
    """A non-JSON 200 body (e.g. an HTML proxy error page) must raise a
    clear, bounded RuntimeError, not a raw JSONDecodeError."""
    long_body = "<html>" + "x" * 5000

    class NonJsonResponse(MockResponse):
        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(return_value=NonJsonResponse(status_code=200, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        linear._graphql("query { viewer { id } }")

    assert "[truncated]" in str(excinfo.value)


def test_get_current_user_returns_profile(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "viewer": {
                            "id": "u1",
                            "name": "Ada",
                            "email": "ada@example.com",
                            "displayName": "ada",
                            "admin": False,
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["email"] == "ada@example.com"


def test_get_current_user_surfaces_partial_graphql_errors_as_warnings(monkeypatch):
    """A genuine partial success (root field populated, but a sub-field
    resolver failed) must surface `errors` in the tool result, not only in
    the server log -- otherwise the caller has no way to know part of the
    response is unreliable."""
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"viewer": {"id": "u1", "name": "Ada"}},
                    "errors": [{"message": "email field failed"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["id"] == "u1"
    assert "email field failed" in result["warnings"][0]


def test_get_current_user_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data={"errors": [{"message": "Authentication required"}]},
            )
        ),
    )

    result = json.loads(linear.linear_get_current_user())

    assert result["status"] == "error"
    assert "Authentication required" in result["message"]


def test_resolve_team_uuid_returns_already_uuid_input_unchanged(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    assert linear._resolve_team_uuid(_TEAM_UUID) == _TEAM_UUID
    mock_post.assert_not_called()


def test_resolve_team_uuid_filters_by_key_not_by_id(monkeypatch):
    """The unambiguous, documented way to resolve a team key: filter the
    teams collection by key, rather than passing the raw key to the
    top-level team(id:) field (which Linear's own SDK only documents as
    accepting a UUID)."""
    mock_post = Mock(
        return_value=MockResponse(
            json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = linear._resolve_team_uuid("ENG")

    assert result == _TEAM_UUID
    sent = mock_post.call_args.kwargs["json"]
    assert "teams(filter:" in sent["query"]
    assert "team(id:" not in sent["query"]
    assert sent["variables"] == {"key": "ENG"}


def test_resolve_team_uuid_matches_case_insensitively(monkeypatch):
    """Team keys are conventionally uppercase (e.g. "ENG"), but nothing
    tells an LLM caller that -- a lowercased key like "eng" must still
    resolve, so the filter must use eqIgnoreCase, not eq."""
    mock_post = Mock(
        return_value=MockResponse(
            json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = linear._resolve_team_uuid("eng")

    assert result == _TEAM_UUID
    sent = mock_post.call_args.kwargs["json"]
    assert "eqIgnoreCase" in sent["query"]
    assert "{ eq:" not in sent["query"]
    assert sent["variables"] == {"key": "eng"}


def test_resolve_team_uuid_raises_when_key_does_not_resolve(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(return_value=MockResponse(json_data={"data": {"teams": {"nodes": []}}})),
    )

    with pytest.raises(ValueError, match="not found"):
        linear._resolve_team_uuid("NOPE")


def test_list_teams_returns_nodes(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "teams": {
                            "nodes": [{"id": "t1", "key": "ENG", "name": "Engineering"}]
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_list_teams())

    assert result["status"] == "success"
    assert result["teams"] == [{"id": "t1", "key": "ENG", "name": "Engineering"}]
    assert result["truncated"] is False


def test_list_teams_clamps_over_limit(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(json_data={"data": {"teams": {"nodes": []}}})
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_list_teams(limit=500)

    assert mock_post.call_args.kwargs["json"]["variables"]["first"] == linear.MAX_LIMIT


def test_list_teams_surfaces_partial_graphql_errors_as_warnings(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"teams": {"nodes": [{"id": "t1", "key": "ENG"}]}},
                    "errors": [{"message": "some.other.field failed"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_list_teams())

    assert result["status"] == "success"
    assert result["teams"] == [{"id": "t1", "key": "ENG"}]
    assert "some.other.field failed" in result["warnings"][0]


def test_list_teams_reports_truncated_when_more_pages_exist(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "teams": {
                            "nodes": [
                                {"id": "t1", "key": "ENG", "name": "Engineering"}
                            ],
                            "pageInfo": {"hasNextPage": True},
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_list_teams())

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_list_workflow_states_resolves_team_key_to_uuid(monkeypatch):
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
            ),
            MockResponse(
                json_data={
                    "data": {
                        "team": {"states": {"nodes": [{"id": "s1", "name": "Todo"}]}}
                    }
                }
            ),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_list_workflow_states("ENG"))

    assert result["status"] == "success"
    assert result["states"] == [{"id": "s1", "name": "Todo"}]
    states_call = mock_post.call_args_list[1]
    assert states_call.kwargs["json"]["variables"]["teamId"] == _TEAM_UUID


def test_list_workflow_states_reports_error_when_team_missing(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            side_effect=[
                MockResponse(
                    json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
                ),
                MockResponse(json_data={"data": {"team": None}}),
            ]
        ),
    )

    result = json.loads(linear.linear_list_workflow_states("ENG"))

    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_list_workflow_states_reports_truncated_when_more_pages_exist(monkeypatch):
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
            ),
            MockResponse(
                json_data={
                    "data": {
                        "team": {
                            "states": {
                                "nodes": [{"id": "s1", "name": "Todo"}],
                                "pageInfo": {"hasNextPage": True},
                            }
                        }
                    }
                }
            ),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_list_workflow_states("ENG"))

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_list_workflow_states_clamps_over_limit(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={"data": {"team": {"states": {"nodes": []}}}}
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_list_workflow_states(_TEAM_UUID, limit=500)

    assert mock_post.call_args.kwargs["json"]["variables"]["first"] == linear.MAX_LIMIT


def test_list_workflow_states_surfaces_partial_graphql_errors_as_warnings(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"team": {"states": {"nodes": [{"id": "s1"}]}}},
                    "errors": [{"message": "some.other.field failed"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_list_workflow_states(_TEAM_UUID))

    assert result["status"] == "success"
    assert result["states"] == [{"id": "s1"}]
    assert "some.other.field failed" in result["warnings"][0]


def test_list_labels_resolves_team_key_to_uuid(monkeypatch):
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
            ),
            MockResponse(
                json_data={
                    "data": {
                        "team": {"labels": {"nodes": [{"id": "l1", "name": "bug"}]}}
                    }
                }
            ),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_list_labels("ENG"))

    assert result["status"] == "success"
    assert result["labels"] == [{"id": "l1", "name": "bug"}]
    assert result["truncated"] is False


def test_list_labels_clamps_over_limit(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={"data": {"team": {"labels": {"nodes": []}}}}
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_list_labels(_TEAM_UUID, limit=500)

    assert mock_post.call_args.kwargs["json"]["variables"]["first"] == linear.MAX_LIMIT


def test_list_labels_reports_error_when_team_missing(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(return_value=MockResponse(json_data={"data": {"team": None}})),
    )

    result = json.loads(linear.linear_list_labels(_TEAM_UUID))

    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_list_labels_surfaces_partial_graphql_errors_as_warnings(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"team": {"labels": {"nodes": [{"id": "l1"}]}}},
                    "errors": [{"message": "some.other.field failed"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_list_labels(_TEAM_UUID))

    assert result["status"] == "success"
    assert result["labels"] == [{"id": "l1"}]
    assert "some.other.field failed" in result["warnings"][0]


def test_list_labels_reports_truncated_when_more_pages_exist(monkeypatch):
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
            ),
            MockResponse(
                json_data={
                    "data": {
                        "team": {
                            "labels": {
                                "nodes": [{"id": "l1", "name": "bug"}],
                                "pageInfo": {"hasNextPage": True},
                            }
                        }
                    }
                }
            ),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_list_labels("ENG"))

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_list_projects_resolves_team_key_to_uuid(monkeypatch):
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
            ),
            MockResponse(
                json_data={
                    "data": {
                        "team": {
                            "projects": {
                                "nodes": [
                                    {
                                        "id": "p1",
                                        "name": "Q3",
                                        "status": {
                                            "id": "s1",
                                            "name": "In Progress",
                                            "type": "started",
                                        },
                                    }
                                ]
                            }
                        }
                    }
                }
            ),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_list_projects(team_id="ENG"))

    assert result["status"] == "success"
    # Asserting the parsed response carries status.type (not just that the
    # hardcoded query string mentions "status") proves the deprecated
    # Project.state -> status migration actually reaches the tool's output.
    assert result["projects"][0]["status"]["type"] == "started"
    assert result["truncated"] is False


def test_list_projects_reports_truncated_when_team_scoped_has_more_pages(monkeypatch):
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
            ),
            MockResponse(
                json_data={
                    "data": {
                        "team": {
                            "projects": {
                                "nodes": [{"id": "p1", "name": "Q3"}],
                                "pageInfo": {"hasNextPage": True},
                            }
                        }
                    }
                }
            ),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_list_projects(team_id="ENG"))

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_list_projects_clamps_over_limit_when_team_scoped(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={"data": {"team": {"projects": {"nodes": []}}}}
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_list_projects(team_id=_TEAM_UUID, limit=500)

    assert mock_post.call_args.kwargs["json"]["variables"]["first"] == linear.MAX_LIMIT


def test_list_projects_clamps_over_limit_without_team_id(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(json_data={"data": {"projects": {"nodes": []}}})
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_list_projects(limit=500)

    assert mock_post.call_args.kwargs["json"]["variables"]["first"] == linear.MAX_LIMIT


def test_list_projects_without_team_id_skips_resolution(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "projects": {
                        "nodes": [
                            {
                                "id": "p1",
                                "name": "Q3",
                                "status": {
                                    "id": "s1",
                                    "name": "In Progress",
                                    "type": "started",
                                },
                            }
                        ]
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_list_projects())

    assert result["status"] == "success"
    assert mock_post.call_count == 1
    assert result["truncated"] is False
    # Asserting the parsed response carries status.type proves the
    # deprecated Project.state -> status migration reaches the output,
    # not just that the hardcoded query string happens to mention "status".
    assert result["projects"][0]["status"]["type"] == "started"


def test_list_projects_reports_error_when_team_missing(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(return_value=MockResponse(json_data={"data": {"team": None}})),
    )

    result = json.loads(linear.linear_list_projects(team_id=_TEAM_UUID))

    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_list_projects_surfaces_partial_graphql_errors_as_warnings(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"projects": {"nodes": [{"id": "p1"}]}},
                    "errors": [{"message": "some.other.field failed"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_list_projects())

    assert result["status"] == "success"
    assert result["projects"] == [{"id": "p1"}]
    assert "some.other.field failed" in result["warnings"][0]


def test_list_projects_without_team_id_reports_truncated_when_more_pages_exist(
    monkeypatch,
):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "projects": {
                        "nodes": [{"id": "p1", "name": "Q3"}],
                        "pageInfo": {"hasNextPage": True},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_list_projects())

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_search_issues_rejects_invalid_state_type(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_issues(state_type="in-progress"))

    assert result["status"] == "error"
    assert "state_type" in result["message"]
    mock_post.assert_not_called()


@pytest.mark.parametrize(
    "state_type",
    ["triage", "backlog", "unstarted", "started", "completed", "canceled", "duplicate"],
)
def test_search_issues_accepts_every_documented_state_type(monkeypatch, state_type):
    mock_post = Mock(
        return_value=MockResponse(json_data={"data": {"issues": {"nodes": []}}})
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_issues(state_type=state_type))

    assert result["status"] == "success"
    assert mock_post.call_args.kwargs["json"]["variables"]["stateType"] == state_type


def test_search_issues_applies_team_filter(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "issues": {
                        "nodes": [
                            {
                                "id": "i1",
                                "identifier": "ENG-1",
                                "title": "Bug",
                            }
                        ]
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    # An already-UUID team_id skips resolution entirely (see the dedicated
    # key-resolution test below), keeping this test focused on filter wiring.
    result = json.loads(linear.linear_search_issues(team_id=_TEAM_UUID))

    assert result["status"] == "success"
    assert result["issues"][0]["identifier"] == "ENG-1"
    variables = mock_post.call_args.kwargs["json"]["variables"]
    assert variables["teamId"] == _TEAM_UUID
    query_text = mock_post.call_args.kwargs["json"]["query"]
    assert query_text.count("{") == query_text.count("}")


def test_search_issues_combines_assignee_and_state_type_filters(monkeypatch):
    """The type_signature lookup is keyed by every variable name actually
    present in `variables` -- this must not KeyError when two (or more)
    optional filters are combined, only tested individually elsewhere."""
    mock_post = Mock(
        return_value=MockResponse(json_data={"data": {"issues": {"nodes": []}}})
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(
        linear.linear_search_issues(assignee_id="u1", state_type="started")
    )

    assert result["status"] == "success"
    variables = mock_post.call_args.kwargs["json"]["variables"]
    assert variables["assigneeId"] == "u1"
    assert variables["stateType"] == "started"
    query_text = mock_post.call_args.kwargs["json"]["query"]
    assert query_text.count("{") == query_text.count("}")
    assert "$assigneeId: ID!" in query_text
    assert "$stateType: String!" in query_text


def test_search_issues_combines_all_three_filters(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(json_data={"data": {"issues": {"nodes": []}}})
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(
        linear.linear_search_issues(
            team_id=_TEAM_UUID, assignee_id="u1", state_type="started"
        )
    )

    assert result["status"] == "success"
    variables = mock_post.call_args.kwargs["json"]["variables"]
    assert variables == {
        "teamId": _TEAM_UUID,
        "assigneeId": "u1",
        "stateType": "started",
        "first": 20,
        "after": None,
    }
    query_text = mock_post.call_args.kwargs["json"]["query"]
    assert query_text.count("{") == query_text.count("}")


def test_search_issues_resolves_team_key_to_uuid(monkeypatch):
    """The `team: { id: { eq: ... } }` filter requires the team's real UUID,
    not its human-readable key (e.g. "ENG") — team_id must be resolved
    first, same as issue identifiers for mutation inputs."""
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
            ),
            MockResponse(json_data={"data": {"issues": {"nodes": []}}}),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_issues(team_id="ENG"))

    assert result["status"] == "success"
    assert mock_post.call_count == 2
    resolve_call, search_call = mock_post.call_args_list
    assert resolve_call.kwargs["json"]["variables"] == {"key": "ENG"}
    assert search_call.kwargs["json"]["variables"]["teamId"] == _TEAM_UUID


def test_search_issues_filters_by_title_query_client_side(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "issues": {
                            "nodes": [
                                {
                                    "id": "i1",
                                    "identifier": "ENG-1",
                                    "title": "Login bug",
                                },
                                {
                                    "id": "i2",
                                    "identifier": "ENG-2",
                                    "title": "Docs typo",
                                },
                            ]
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_search_issues(query="login"))

    assert result["status"] == "success"
    assert len(result["issues"]) == 1
    assert result["issues"][0]["identifier"] == "ENG-1"


def test_search_issues_query_omits_description_field(monkeypatch):
    """Up to MAX_ISSUE_SEARCH_PAGES * MAX_LIMIT issues can be fetched
    server-side to answer one search -- the list/search query must not ask
    for the (potentially large, unbounded) description body per row."""
    mock_post = Mock(
        return_value=MockResponse(json_data={"data": {"issues": {"nodes": []}}})
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_search_issues()

    query_text = mock_post.call_args.kwargs["json"]["query"]
    assert "description" not in query_text


def test_get_issue_query_includes_description_field(monkeypatch):
    """A single-issue fetch is exactly the case where the full body is the
    point, unlike the bulk list/search path."""
    mock_post = Mock(
        return_value=MockResponse(json_data={"data": {"issue": {"id": "i1"}}})
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_get_issue("ENG-1")

    query_text = mock_post.call_args.kwargs["json"]["query"]
    assert "description" in query_text


def test_search_issues_surfaces_error_instead_of_confident_empty_result(monkeypatch):
    """A partial error on the `issues` root field itself yields
    {"issues": null} -- this must be a hard failure, not a confident
    _success(issues=[], truncated=False) for a request that actually
    failed."""
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"issues": None},
                    "errors": [{"message": "Internal server error"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_search_issues(query="login"))

    assert result["status"] == "error"
    assert "Internal server error" in result["message"]


def test_search_issues_title_query_keeps_paging_past_the_first_page(monkeypatch):
    """Linear has no server-side title filter -- a match beyond the first
    MAX_LIMIT-sized page must not be silently missed. The first page has no
    matches at all, so the loop must fetch a second page using the cursor
    from the first page's pageInfo."""
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={
                    "data": {
                        "issues": {
                            "nodes": [
                                {"id": "i1", "identifier": "ENG-1", "title": "Docs"}
                            ],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                        }
                    }
                }
            ),
            MockResponse(
                json_data={
                    "data": {
                        "issues": {
                            "nodes": [
                                {
                                    "id": "i2",
                                    "identifier": "ENG-2",
                                    "title": "Login bug",
                                }
                            ],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            ),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_issues(query="login"))

    assert result["status"] == "success"
    assert [issue["identifier"] for issue in result["issues"]] == ["ENG-2"]
    assert result["truncated"] is False
    assert mock_post.call_count == 2
    first_call, second_call = mock_post.call_args_list
    assert first_call.kwargs["json"]["variables"]["after"] is None
    assert second_call.kwargs["json"]["variables"]["after"] == "cursor-1"


def test_search_issues_title_query_stops_when_next_page_has_no_cursor(monkeypatch):
    """hasNextPage: true with a missing/null endCursor must stop paging
    instead of re-sending after: null, which would refetch page 1 forever
    and duplicate nodes."""
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "issues": {
                        "nodes": [{"id": "i1", "identifier": "ENG-1", "title": "Docs"}],
                        "pageInfo": {"hasNextPage": True, "endCursor": None},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_issues(query="login"))

    assert result["status"] == "success"
    assert result["issues"] == []
    assert result["truncated"] is True
    assert mock_post.call_count == 1


def test_search_issues_title_query_returns_partial_matches_on_mid_pagination_failure(
    monkeypatch,
):
    """A page failure after at least one page already succeeded must not
    discard matches collected so far -- mirrors slack_list_channels."""
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={
                    "data": {
                        "issues": {
                            "nodes": [
                                {"id": "i1", "identifier": "ENG-1", "title": "Docs"}
                            ],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                        }
                    }
                }
            ),
            RuntimeError("Linear API error (status 500): boom"),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_issues(query="docs"))

    assert result["status"] == "success"
    assert [issue["identifier"] for issue in result["issues"]] == ["ENG-1"]
    assert result["truncated"] is True
    assert "boom" in result["error"]


def test_search_issues_title_query_reports_error_on_first_page_failure(monkeypatch):
    """With no prior page to fall back on, a first-page failure must
    surface as a hard error, not a `_success` with an empty partial list --
    there is nothing to distinguish that from a genuine no-match result."""
    mock_post = Mock(side_effect=[RuntimeError("Linear API error (status 500): boom")])
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_issues(query="docs"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_search_issues_preserves_earlier_page_warnings_on_later_page_failure(
    monkeypatch,
):
    """A genuine partial GraphQL error surfaced on an earlier, successfully
    fetched page must not be silently dropped just because a later page
    then fails outright -- both must reach the caller."""
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={
                    "data": {
                        "issues": {
                            "nodes": [
                                {"id": "i1", "identifier": "ENG-1", "title": "Docs"}
                            ],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                        }
                    },
                    "errors": [{"message": "some.unrelated.field failed"}],
                }
            ),
            RuntimeError("Linear API error (status 500): boom"),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_issues(query="docs"))

    assert result["status"] == "success"
    assert [issue["identifier"] for issue in result["issues"]] == ["ENG-1"]
    assert result["truncated"] is True
    assert "boom" in result["error"]
    assert "some.unrelated.field failed" in result["warnings"][0]


def test_search_issues_title_query_stops_at_max_pages_and_reports_truncated(
    monkeypatch,
):
    """If MAX_ISSUE_SEARCH_PAGES is exhausted while the server still reports
    more pages, matches may exist further out -- must report truncated
    rather than a false "complete" result. Uses a finite side_effect (one
    response per page, each with a distinct cursor) rather than a
    Mock(return_value=...) that would return the same non-matching page
    forever regardless of how many pages the loop actually fetches -- that
    shape can't distinguish "looped MAX_ISSUE_SEARCH_PAGES times" from
    "looped once", nor pin that the cursor advances."""
    pages = [
        MockResponse(
            json_data={
                "data": {
                    "issues": {
                        "nodes": [
                            {"id": f"i{i}", "identifier": f"ENG-{i}", "title": "Docs"}
                        ],
                        "pageInfo": {"hasNextPage": True, "endCursor": f"cursor-{i}"},
                    }
                }
            }
        )
        for i in range(linear.MAX_ISSUE_SEARCH_PAGES)
    ]
    mock_post = Mock(side_effect=pages)
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_issues(query="login"))

    assert result["status"] == "success"
    assert result["issues"] == []
    assert result["truncated"] is True
    assert mock_post.call_count == linear.MAX_ISSUE_SEARCH_PAGES
    sent_afters = [
        call.kwargs["json"]["variables"]["after"] for call in mock_post.call_args_list
    ]
    assert sent_afters == [None] + [
        f"cursor-{i}" for i in range(linear.MAX_ISSUE_SEARCH_PAGES - 1)
    ]


def test_search_issues_reports_truncated_via_page_info_without_query(monkeypatch):
    """With no client-side query filter, `issues` is always bounded to
    max_results by the `first` fetch itself, so hasNextPage from the server
    is the only real signal that more results exist beyond this page."""
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "issues": {
                            "nodes": [
                                {"id": "i1", "identifier": "ENG-1", "title": "Bug"}
                            ],
                            "pageInfo": {"hasNextPage": True},
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_search_issues(limit=1))

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_search_issues_not_truncated_when_no_next_page(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "issues": {
                            "nodes": [
                                {"id": "i1", "identifier": "ENG-1", "title": "Bug"}
                            ],
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_search_issues())

    assert result["status"] == "success"
    assert result["truncated"] is False


def test_get_issue_returns_issue_on_success(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "issue": {
                            "id": _TEAM_UUID,
                            "identifier": "ENG-1",
                            "title": "Fix bug",
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_get_issue("ENG-1"))

    assert result["status"] == "success"
    assert result["issue"]["identifier"] == "ENG-1"
    assert result["issue"]["title"] == "Fix bug"


def test_get_issue_surfaces_partial_graphql_errors_as_warnings(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"issue": {"id": "i1", "identifier": "ENG-1"}},
                    "errors": [{"message": "some.other.field failed"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_get_issue("ENG-1"))

    assert result["status"] == "success"
    assert result["issue"]["identifier"] == "ENG-1"
    assert "some.other.field failed" in result["warnings"][0]


def test_get_issue_returns_not_found_error_when_missing(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(return_value=MockResponse(json_data={"data": {"issue": None}})),
    )

    result = json.loads(linear.linear_get_issue("ENG-999"))

    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_get_issue_surfaces_permission_error_instead_of_reporting_not_found(
    monkeypatch,
):
    """A permission failure comes back as HTTP 200 with
    {"data": {"issue": null}, "errors": [...]} -- this is NOT the same as a
    genuinely nonexistent issue (no errors) and must surface the real cause
    rather than the generic "not found" message."""
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"issue": None},
                    "errors": [{"message": "You do not have permission"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_get_issue("ENG-1"))

    assert result["status"] == "error"
    assert "permission" in result["message"]
    assert "not found" not in result["message"]


def test_get_issue_returns_error_payload_on_network_exception(monkeypatch):
    """Every tool wraps its body in try/except Exception -> _error(str(e)) --
    a network-level failure (not just an HTTP error status) must surface as
    a structured error payload, not an unhandled exception escaping the
    FastMCP tool call."""
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(side_effect=requests.ConnectionError("Connection refused")),
    )

    result = json.loads(linear.linear_get_issue("ENG-1"))

    assert result["status"] == "error"
    assert "Connection refused" in result["message"]


def test_create_issue_sends_expected_input(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "id": "i1",
                            "identifier": "ENG-1",
                            "title": "New bug",
                        },
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    # An already-UUID team_id skips resolution (see the dedicated
    # key-resolution test below), keeping this test focused on input wiring.
    result = json.loads(
        linear.linear_create_issue(
            team_id=_TEAM_UUID, title="New bug", assignee_id="u1", priority=2
        )
    )

    assert result["status"] == "success"
    assert result["issue"]["identifier"] == "ENG-1"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {
        "teamId": _TEAM_UUID,
        "title": "New bug",
        "assigneeId": "u1",
        "priority": 2,
    }


def test_create_issue_sends_label_ids(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {"issueCreate": {"success": True, "issue": {"id": "i1"}}}
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_create_issue(
        team_id=_TEAM_UUID, title="New bug", label_ids=["l1", "l2"]
    )

    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input["labelIds"] == ["l1", "l2"]


def test_create_issue_sends_explicit_empty_label_ids(monkeypatch):
    """label_ids must use is-not-None (like linear_update_issue's label_ids),
    not truthiness -- otherwise an explicitly empty list is silently dropped
    instead of being sent through."""
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {"issueCreate": {"success": True, "issue": {"id": "i1"}}}
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_create_issue(team_id=_TEAM_UUID, title="New bug", label_ids=[])

    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input["labelIds"] == []


def test_create_issue_resolves_team_key_to_uuid(monkeypatch):
    """IssueCreateInput.teamId requires the team's real UUID, not its key
    (e.g. "ENG") — team_id must be resolved first."""
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={"data": {"teams": {"nodes": [{"id": _TEAM_UUID}]}}}
            ),
            MockResponse(
                json_data={
                    "data": {
                        "issueCreate": {
                            "success": True,
                            "issue": {"id": "i1", "identifier": "ENG-1"},
                        }
                    }
                }
            ),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_create_issue(team_id="ENG", title="New bug"))

    assert result["status"] == "success"
    assert mock_post.call_count == 2
    create_call = mock_post.call_args_list[1]
    assert create_call.kwargs["json"]["variables"]["input"]["teamId"] == _TEAM_UUID


def test_create_issue_omits_priority_when_not_provided(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {"issueCreate": {"success": True, "issue": {"id": "i1"}}}
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_create_issue(team_id=_TEAM_UUID, title="New bug")

    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert "priority" not in sent_input


def test_create_issue_sends_explicit_priority_zero(monkeypatch):
    """priority=0 ("no priority") is a valid explicit choice, distinct from
    the caller not specifying priority at all -- both happen to produce the
    same server-side outcome today, but the input contract should still
    distinguish them, matching linear_update_issue's is-not-None handling."""
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {"issueCreate": {"success": True, "issue": {"id": "i1"}}}
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_create_issue(team_id=_TEAM_UUID, title="New bug", priority=0)

    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input["priority"] == 0


def test_create_issue_rejects_out_of_range_priority(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(
        linear.linear_create_issue(team_id=_TEAM_UUID, title="New bug", priority=5)
    )

    assert result["status"] == "error"
    assert "priority" in result["message"]
    mock_post.assert_not_called()


def test_create_issue_surfaces_partial_graphql_errors_as_warnings(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"issueCreate": {"success": True, "issue": {"id": "i1"}}},
                    "errors": [{"message": "some.other.field failed"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_create_issue(team_id=_TEAM_UUID, title="New bug"))

    assert result["status"] == "success"
    assert result["issue"]["id"] == "i1"
    assert "some.other.field failed" in result["warnings"][0]


def test_create_issue_rejects_empty_title(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_create_issue(team_id=_TEAM_UUID, title="   "))

    assert result["status"] == "error"
    assert "title" in result["message"]
    mock_post.assert_not_called()


def test_create_issue_reports_error_when_linear_reports_failure(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={"data": {"issueCreate": {"success": False, "issue": None}}}
            )
        ),
    )

    result = json.loads(linear.linear_create_issue(team_id=_TEAM_UUID, title="New bug"))

    assert result["status"] == "error"
    assert "not created" in result["message"]


def test_update_issue_requires_at_least_one_field(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1"))

    assert result["status"] == "error"
    assert "No fields" in result["message"]
    mock_post.assert_not_called()


def test_update_issue_reports_error_when_linear_reports_failure(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={"data": {"issueUpdate": {"success": False, "issue": None}}}
            )
        ),
    )

    result = json.loads(linear.linear_update_issue("ENG-1", title="Renamed"))

    assert result["status"] == "error"
    assert "not updated" in result["message"]


def test_update_issue_omits_priority_when_left_default(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "issueUpdate": {
                        "success": True,
                        "issue": {"id": "i1", "identifier": "ENG-1"},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", state_id="s1"))

    assert result["status"] == "success"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {"stateId": "s1"}


def test_update_issue_unassigns_on_explicit_empty_assignee_id(monkeypatch):
    """assignee_id="" (explicitly passed) must clear the assignee (send
    None), distinct from leaving assignee_id unset entirely (which omits
    "assigneeId" from the input so it is left untouched)."""
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "issueUpdate": {
                        "success": True,
                        "issue": {"id": "i1", "identifier": "ENG-1"},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", assignee_id=""))

    assert result["status"] == "success"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {"assigneeId": None}


def test_update_issue_can_clear_description(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "issueUpdate": {
                        "success": True,
                        "issue": {"id": "i1", "identifier": "ENG-1"},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", description=""))

    assert result["status"] == "success"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {"description": ""}


_ISSUE_UPDATE_SUCCESS = MockResponse(
    json_data={
        "data": {
            "issueUpdate": {
                "success": True,
                "issue": {"id": "i1", "identifier": "ENG-1"},
            }
        }
    }
)


def test_update_issue_surfaces_partial_graphql_errors_as_warnings(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"issueUpdate": {"success": True, "issue": {"id": "i1"}}},
                    "errors": [{"message": "some.other.field failed"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_update_issue("ENG-1", title="Renamed"))

    assert result["status"] == "success"
    assert result["issue"]["id"] == "i1"
    assert "some.other.field failed" in result["warnings"][0]


def test_update_issue_leaves_labels_untouched_when_not_provided(monkeypatch):
    mock_post = Mock(return_value=_ISSUE_UPDATE_SUCCESS)
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_update_issue("ENG-1", title="Renamed")

    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert "labelIds" not in sent_input
    assert "addedLabelIds" not in sent_input
    assert "removedLabelIds" not in sent_input


def test_update_issue_adds_labels_via_added_label_ids(monkeypatch):
    mock_post = Mock(return_value=_ISSUE_UPDATE_SUCCESS)
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", add_label_ids=["l1"]))

    assert result["status"] == "success"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {"addedLabelIds": ["l1"]}


def test_update_issue_removes_labels_via_removed_label_ids(monkeypatch):
    mock_post = Mock(return_value=_ISSUE_UPDATE_SUCCESS)
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", remove_label_ids=["l1"]))

    assert result["status"] == "success"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {"removedLabelIds": ["l1"]}


def test_update_issue_replaces_full_label_set_via_label_ids(monkeypatch):
    mock_post = Mock(return_value=_ISSUE_UPDATE_SUCCESS)
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", label_ids=["l1", "l2"]))

    assert result["status"] == "success"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {"labelIds": ["l1", "l2"]}


def test_update_issue_clears_all_labels_via_empty_label_ids(monkeypatch):
    """An explicit empty list for label_ids (distinct from leaving it
    unset/None) must clear every label, mirroring the create-time
    labelIds field's full-replace semantics."""
    mock_post = Mock(return_value=_ISSUE_UPDATE_SUCCESS)
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", label_ids=[]))

    assert result["status"] == "success"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {"labelIds": []}


def test_update_issue_distinguishes_omitted_from_explicit_empty_add_remove_labels(
    monkeypatch,
):
    """add_label_ids/remove_label_ids must use is-not-None (like label_ids),
    not truthiness, so an explicit [] is sent through rather than treated
    the same as an omitted argument."""
    mock_post = Mock(return_value=_ISSUE_UPDATE_SUCCESS)
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(
        linear.linear_update_issue("ENG-1", add_label_ids=[], remove_label_ids=[])
    )

    assert result["status"] == "success"
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {"addedLabelIds": [], "removedLabelIds": []}


def test_update_issue_rejects_label_ids_combined_with_add_label_ids(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(
        linear.linear_update_issue("ENG-1", label_ids=["l1"], add_label_ids=["l2"])
    )

    assert result["status"] == "error"
    assert "label_ids" in result["message"]
    mock_post.assert_not_called()


def test_update_issue_rejects_label_ids_combined_with_remove_label_ids(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(
        linear.linear_update_issue("ENG-1", label_ids=["l1"], remove_label_ids=["l2"])
    )

    assert result["status"] == "error"
    assert "label_ids" in result["message"]
    mock_post.assert_not_called()


def test_update_issue_rejects_overlapping_add_and_remove_label_ids(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(
        linear.linear_update_issue(
            "ENG-1", add_label_ids=["l1", "l2"], remove_label_ids=["l2", "l3"]
        )
    )

    assert result["status"] == "error"
    assert "label id" in result["message"]
    mock_post.assert_not_called()


def test_update_issue_rejects_label_ids_combined_with_explicitly_empty_add_label_ids(
    monkeypatch,
):
    """The conflict check must use is-not-None (like the field-building code
    a few lines below it), not truthiness -- otherwise an explicitly empty
    add_label_ids=[] (falsy, but a deliberate argument, not an omitted one)
    slips past the check and both labelIds and addedLabelIds end up in the
    same IssueUpdateInput."""
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(
        linear.linear_update_issue("ENG-1", label_ids=["l1"], add_label_ids=[])
    )

    assert result["status"] == "error"
    assert "label_ids" in result["message"]
    mock_post.assert_not_called()


def test_update_issue_rejects_label_ids_combined_with_explicitly_empty_remove_label_ids(
    monkeypatch,
):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(
        linear.linear_update_issue("ENG-1", label_ids=["l1"], remove_label_ids=[])
    )

    assert result["status"] == "error"
    assert "label_ids" in result["message"]
    mock_post.assert_not_called()


def test_update_issue_rejects_out_of_range_priority(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", priority=5))

    assert result["status"] == "error"
    assert "priority" in result["message"]
    mock_post.assert_not_called()


def test_update_issue_rejects_explicitly_empty_title(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", title="   "))

    assert result["status"] == "error"
    assert "title" in result["message"]
    mock_post.assert_not_called()


def test_update_issue_allows_title_left_unset(monkeypatch):
    """title=None (the default) means "leave untouched", not "empty" --
    must not be rejected by the same guard that catches title=""."""
    mock_post = Mock(return_value=_ISSUE_UPDATE_SUCCESS)
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_update_issue("ENG-1", state_id="s1"))

    assert result["status"] == "success"


def test_list_comments_returns_nodes(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "issue": {
                        "comments": {
                            "nodes": [
                                {
                                    "id": "c1",
                                    "body": "Looks good",
                                    "createdAt": "2026-08-01T00:00:00Z",
                                    "user": {"id": "u1", "name": "Ada"},
                                }
                            ]
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_list_comments("ENG-1"))

    assert result["status"] == "success"
    assert result["comments"] == [
        {
            "id": "c1",
            "body": "Looks good",
            "createdAt": "2026-08-01T00:00:00Z",
            "user": {"id": "u1", "name": "Ada"},
        }
    ]
    assert result["truncated"] is False
    assert mock_post.call_args.kwargs["json"]["variables"] == {
        "id": "ENG-1",
        "first": 50,
    }


def test_list_comments_surfaces_partial_graphql_errors_as_warnings(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"issue": {"comments": {"nodes": [{"id": "c1"}]}}},
                    "errors": [{"message": "some.other.field failed"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_list_comments("ENG-1"))

    assert result["status"] == "success"
    assert result["comments"] == [{"id": "c1"}]
    assert "some.other.field failed" in result["warnings"][0]


def test_list_comments_clamps_over_limit(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={"data": {"issue": {"comments": {"nodes": []}}}}
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    linear.linear_list_comments("ENG-1", limit=500)

    assert mock_post.call_args.kwargs["json"]["variables"]["first"] == linear.MAX_LIMIT


def test_list_comments_reports_truncated_when_more_pages_exist(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "issue": {
                            "comments": {
                                "nodes": [{"id": "c1", "body": "First"}],
                                "pageInfo": {"hasNextPage": True},
                            }
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_list_comments("ENG-1"))

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_list_comments_reports_error_when_issue_not_found(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(return_value=MockResponse(json_data={"data": {"issue": None}})),
    )

    result = json.loads(linear.linear_list_comments("ENG-999"))

    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_list_comments_surfaces_api_error(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={"errors": [{"message": "Internal error"}]}
            )
        ),
    )

    result = json.loads(linear.linear_list_comments("ENG-1"))

    assert result["status"] == "error"
    assert "Internal error" in result["message"]


def test_add_comment_passes_issue_id_through_without_a_lookup(monkeypatch):
    """CommentCreateInput.issueId accepts either an issue's UUID or its
    human-readable identifier directly (confirmed against Linear's own SDK
    type definitions), so a human-readable identifier like "ENG-1" must be
    sent as-is -- no resolution lookup, no extra request."""
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "commentCreate": {
                        "success": True,
                        "comment": {"id": "c1", "body": "Looks good"},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_add_comment("ENG-1", "Looks good"))

    assert result["status"] == "success"
    assert result["comment"]["body"] == "Looks good"
    assert mock_post.call_count == 1
    sent_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert sent_input == {"issueId": "ENG-1", "body": "Looks good"}


def test_add_comment_surfaces_partial_graphql_errors_as_warnings(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "commentCreate": {"success": True, "comment": {"id": "c1"}}
                    },
                    "errors": [{"message": "some.other.field failed"}],
                }
            )
        ),
    )

    result = json.loads(linear.linear_add_comment("ENG-1", "Looks good"))

    assert result["status"] == "success"
    assert result["comment"]["id"] == "c1"
    assert "some.other.field failed" in result["warnings"][0]


def test_add_comment_reports_error_when_linear_reports_failure(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"commentCreate": {"success": False, "comment": None}}
                }
            )
        ),
    )

    result = json.loads(linear.linear_add_comment("ENG-1", "Looks good"))

    assert result["status"] == "error"
    assert "not created" in result["message"]


def test_add_comment_rejects_empty_body(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_add_comment("ENG-1", "   "))

    assert result["status"] == "error"
    assert "body" in result["message"]
    mock_post.assert_not_called()


def test_search_users_filters_by_name_or_email(monkeypatch):
    monkeypatch.setattr(
        linear.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "users": {
                            "nodes": [
                                {"id": "u1", "name": "Ada", "email": "ada@example.com"},
                                {"id": "u2", "name": "Bob", "email": "bob@example.com"},
                            ]
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(linear.linear_search_users("ada"))

    assert result["status"] == "success"
    assert len(result["users"]) == 1
    assert result["users"][0]["id"] == "u1"
    assert result["truncated"] is False


def test_search_users_keeps_paging_past_the_first_page(monkeypatch):
    """A match beyond the first MAX_LIMIT-sized page (e.g. the 101st member
    of a large workspace) must not be silently missed as a false "no
    match" result."""
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={
                    "data": {
                        "users": {
                            "nodes": [
                                {"id": "u1", "name": "Bob", "email": "bob@example.com"}
                            ],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                        }
                    }
                }
            ),
            MockResponse(
                json_data={
                    "data": {
                        "users": {
                            "nodes": [
                                {
                                    "id": "u2",
                                    "name": "Ada",
                                    "email": "ada@example.com",
                                }
                            ],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            ),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_users("ada"))

    assert result["status"] == "success"
    assert [user["id"] for user in result["users"]] == ["u2"]
    assert result["truncated"] is False
    assert mock_post.call_count == 2
    first_call, second_call = mock_post.call_args_list
    assert first_call.kwargs["json"]["variables"]["after"] is None
    assert second_call.kwargs["json"]["variables"]["after"] == "cursor-1"


def test_search_users_stops_when_next_page_has_no_cursor(monkeypatch):
    """hasNextPage: true with a missing/null endCursor must stop paging
    instead of re-sending after: null, which would refetch page 1 forever
    and duplicate nodes."""
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "users": {
                        "nodes": [
                            {"id": "u1", "name": "Bob", "email": "b@example.com"}
                        ],
                        "pageInfo": {"hasNextPage": True, "endCursor": None},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_users("ada"))

    assert result["status"] == "success"
    assert result["users"] == []
    assert result["truncated"] is True
    assert mock_post.call_count == 1


def test_search_users_returns_partial_matches_on_mid_pagination_failure(monkeypatch):
    """A page failure after at least one page already succeeded must not
    discard matches collected so far -- mirrors slack_list_channels."""
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={
                    "data": {
                        "users": {
                            "nodes": [
                                {"id": "u1", "name": "Ada", "email": "ada@example.com"}
                            ],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                        }
                    }
                }
            ),
            RuntimeError("Linear API error (status 500): boom"),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_users("ada"))

    assert result["status"] == "success"
    assert [user["id"] for user in result["users"]] == ["u1"]
    assert result["truncated"] is True
    assert "boom" in result["error"]


def test_search_users_reports_error_on_first_page_failure(monkeypatch):
    """With no prior page to fall back on, a first-page failure must
    surface as a hard error, not a `_success` with an empty partial list --
    there is nothing to distinguish that from a genuine no-match result."""
    mock_post = Mock(side_effect=[RuntimeError("Linear API error (status 500): boom")])
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_users("ada"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_search_users_preserves_earlier_page_warnings_on_later_page_failure(
    monkeypatch,
):
    """A genuine partial GraphQL error surfaced on an earlier, successfully
    fetched page must not be silently dropped just because a later page
    then fails outright -- both must reach the caller."""
    mock_post = Mock(
        side_effect=[
            MockResponse(
                json_data={
                    "data": {
                        "users": {
                            "nodes": [
                                {"id": "u1", "name": "Ada", "email": "ada@example.com"}
                            ],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                        }
                    },
                    "errors": [{"message": "some.unrelated.field failed"}],
                }
            ),
            RuntimeError("Linear API error (status 500): boom"),
        ]
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_users("ada"))

    assert result["status"] == "success"
    assert [user["id"] for user in result["users"]] == ["u1"]
    assert result["truncated"] is True
    assert "boom" in result["error"]
    assert "some.unrelated.field failed" in result["warnings"][0]


def test_search_users_lists_all_members_when_query_omitted(monkeypatch):
    """query defaults to "" (matching linear_search_issues), listing every
    workspace member instead of requiring a throwaway empty-string call."""
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "users": {
                        "nodes": [
                            {"id": "u1", "name": "Ada", "email": "ada@example.com"},
                            {"id": "u2", "name": "Bob", "email": "bob@example.com"},
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_users())

    assert result["status"] == "success"
    assert [user["id"] for user in result["users"]] == ["u1", "u2"]


def test_search_users_stops_at_max_pages_and_reports_truncated(monkeypatch):
    """Finite side_effect (distinct cursor per page), not
    Mock(return_value=...) -- see the analogous issue-search test for why a
    same-response-forever mock can't pin the page bound or cursor
    advancement."""
    pages = [
        MockResponse(
            json_data={
                "data": {
                    "users": {
                        "nodes": [
                            {"id": f"u{i}", "name": "Bob", "email": "bob@example.com"}
                        ],
                        "pageInfo": {"hasNextPage": True, "endCursor": f"cursor-{i}"},
                    }
                }
            }
        )
        for i in range(linear.MAX_USER_SEARCH_PAGES)
    ]
    mock_post = Mock(side_effect=pages)
    monkeypatch.setattr(linear.requests, "post", mock_post)

    result = json.loads(linear.linear_search_users("ada"))

    assert result["status"] == "success"
    assert result["users"] == []
    assert result["truncated"] is True
    assert mock_post.call_count == linear.MAX_USER_SEARCH_PAGES
    sent_afters = [
        call.kwargs["json"]["variables"]["after"] for call in mock_post.call_args_list
    ]
    assert sent_afters == [None] + [
        f"cursor-{i}" for i in range(linear.MAX_USER_SEARCH_PAGES - 1)
    ]


def test_linear_app_registry_requests_read_and_write_scopes():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    linear_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "linear"
    )
    assert linear_app["oauth_scopes"] == ["read", "write"]
