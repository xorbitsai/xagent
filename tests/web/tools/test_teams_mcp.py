import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import teams


class MockResponse:
    def __init__(self, json_data=None, status_code=200):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.content = json.dumps(self._json_data).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "graph-token")


def _sent_params(mock_request: Mock) -> dict:
    return mock_request.call_args.kwargs["params"] or {}


# ---------------------------------------------------------------------------
# list_teams: /me/joinedTeams accepts no OData query options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top", [1, 25, 500])
def test_list_teams_sends_no_query_options(monkeypatch, top):
    mock_request = Mock(return_value=MockResponse({"value": []}))
    monkeypatch.setattr(teams.requests, "request", mock_request)

    result = json.loads(teams.list_teams(top=top))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/me/joinedTeams")
    assert _sent_params(mock_request) == {}


def test_list_teams_returns_every_team_and_next_link(monkeypatch):
    joined = [{"id": f"team-{i}", "displayName": f"Team {i}"} for i in range(3)]
    next_link = f"{teams.GRAPH_BASE_URL}/me/joinedTeams?%24skiptoken=next"
    mock_request = Mock(
        return_value=MockResponse({"value": joined, "@odata.nextLink": next_link})
    )
    monkeypatch.setattr(teams.requests, "request", mock_request)

    result = json.loads(teams.list_teams(top=1))

    assert result == {"status": "success", "teams": joined, "next_link": next_link}


def test_list_teams_reports_graph_error(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {"error": {"message": "Query option 'Top' is not allowed."}},
            status_code=400,
        )
    )
    monkeypatch.setattr(teams.requests, "request", mock_request)

    result = json.loads(teams.list_teams())

    assert result["status"] == "error"
    assert "Query option 'Top' is not allowed." in result["message"]


# ---------------------------------------------------------------------------
# list_channels: /teams/{id}/channels documents only $filter and $select
# ---------------------------------------------------------------------------


def test_list_channels_sends_no_top(monkeypatch):
    channels = [{"id": "19:general@thread.tacv2", "displayName": "General"}]
    next_link = f"{teams.GRAPH_BASE_URL}/teams/team-1/channels?%24skiptoken=next"
    mock_request = Mock(
        return_value=MockResponse({"value": channels, "@odata.nextLink": next_link})
    )
    monkeypatch.setattr(teams.requests, "request", mock_request)

    result = json.loads(teams.list_channels("team-1", top=10))

    assert result == {
        "status": "success",
        "channels": channels,
        "next_link": next_link,
    }
    assert mock_request.call_args.kwargs["url"].endswith("/teams/team-1/channels")
    assert "$top" not in _sent_params(mock_request)


# ---------------------------------------------------------------------------
# list_chats: /me/chats allows $top up to 50
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("top", "expected"), [(0, 1), (20, 20), (50, 50), (100, 50)])
def test_list_chats_caps_top_at_graph_maximum(monkeypatch, top, expected):
    mock_request = Mock(return_value=MockResponse({"value": []}))
    monkeypatch.setattr(teams.requests, "request", mock_request)

    result = json.loads(teams.list_chats(top=top))

    assert result["status"] == "success"
    assert _sent_params(mock_request) == {"$top": expected}
