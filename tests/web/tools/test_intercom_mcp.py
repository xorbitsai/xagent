import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import intercom


class MockResponse:
    def __init__(self, json_data=None, text="", status_code=200, headers=None):
        self._json_data = json_data if json_data is not None else {}
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.status_code = status_code
        self.content = self.text.encode()
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("INTERCOM_ACCESS_TOKEN", "access-token")
    intercom._admin_id_cache.clear()


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("INTERCOM_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="INTERCOM_ACCESS_TOKEN"):
        intercom._headers()


def test_headers_include_bearer_token_and_api_version():
    assert intercom._headers() == {
        "Authorization": "Bearer access-token",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Intercom-Version": intercom.INTERCOM_API_VERSION,
    }


def test_get_contact_percent_encodes_id_with_path_separator(monkeypatch):
    """A contact_id like "../admins" must not be able to redirect the request
    to a different endpoint under the same bearer token (confused deputy)."""
    mock_request = Mock(return_value=MockResponse(json_data={"id": "c1"}))
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_get_contact("../admins")

    called_url = mock_request.call_args.kwargs["url"]
    assert called_url == "https://api.intercom.io/contacts/..%2Fadmins"


def test_get_contact_percent_encodes_id_with_query_injection(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "c1"}))
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_get_contact("123?display_as=plaintext")

    called_url = mock_request.call_args.kwargs["url"]
    assert called_url == "https://api.intercom.io/contacts/123%3Fdisplay_as%3Dplaintext"
    # _request has no params kwarg (dead code, removed) -- the encoded "?"
    # must not somehow still end up attached as a query string.
    assert "params" not in mock_request.call_args.kwargs


def test_get_conversation_percent_encodes_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "conv1"}))
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_get_conversation("../admins")

    called_url = mock_request.call_args.kwargs["url"]
    assert called_url == "https://api.intercom.io/conversations/..%2Fadmins"


def test_get_conversation_projects_populated_parts(monkeypatch):
    """The only prior get_conversation test mocked an empty conversation, so
    the conversation_parts projection never actually ran."""
    monkeypatch.setattr(
        intercom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "id": "conv1",
                    "conversation_parts": {
                        "conversation_parts": [
                            {
                                "id": "part1",
                                "author": {"name": "Alice"},
                                "body": "<p>hi</p>",
                                "created_at": 1700000000,
                            }
                        ]
                    },
                }
            )
        ),
    )

    result = json.loads(intercom.intercom_get_conversation("conv1"))

    assert result["status"] == "success"
    assert result["parts"] == [
        {
            "id": "part1",
            "author": "Alice",
            "body": "<p>hi</p>",
            "created_at": 1700000000,
        }
    ]


def test_reply_percent_encodes_conversation_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "u1"}))
    monkeypatch.setattr(intercom.requests, "request", mock_request)
    # First call resolves the admin id via /me, second is the reply itself.
    mock_request.side_effect = [
        MockResponse(json_data={"id": "admin-1"}),
        MockResponse(json_data={"id": "conv1"}),
    ]

    intercom.intercom_reply_to_conversation("../admins", "hello")

    reply_call = mock_request.call_args_list[1]
    assert (
        reply_call.kwargs["url"]
        == "https://api.intercom.io/conversations/..%2Fadmins/reply"
    )
    # Not just the URL -- swapping message_type comment<->note would change
    # no URL, so the body must be asserted too.
    assert reply_call.kwargs["json"] == {
        "message_type": "comment",
        "type": "admin",
        "admin_id": "admin-1",
        "body": "hello",
    }


def test_reply_rejects_blank_body(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    result = json.loads(intercom.intercom_reply_to_conversation("conv1", "   "))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_add_internal_note_percent_encodes_conversation_id(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"id": "admin-1"}),
            MockResponse(json_data={"id": "conv1"}),
        ]
    )
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_add_internal_note("../admins", "internal note")

    note_call = mock_request.call_args_list[1]
    assert (
        note_call.kwargs["url"]
        == "https://api.intercom.io/conversations/..%2Fadmins/reply"
    )
    assert note_call.kwargs["json"] == {
        "message_type": "note",
        "type": "admin",
        "admin_id": "admin-1",
        "body": "internal note",
    }


def test_add_internal_note_rejects_blank_body(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    result = json.loads(intercom.intercom_add_internal_note("conv1", ""))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_close_conversation_percent_encodes_conversation_id(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"id": "admin-1"}),
            MockResponse(json_data={"id": "conv1"}),
        ]
    )
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_close_conversation("../admins")

    close_call = mock_request.call_args_list[1]
    assert (
        close_call.kwargs["url"]
        == "https://api.intercom.io/conversations/..%2Fadmins/parts"
    )
    assert close_call.kwargs["json"] == {
        "message_type": "close",
        "type": "admin",
        "admin_id": "admin-1",
    }


def test_list_conversations_open_and_closed_filter_by_state(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"conversations": []}))
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_list_conversations(state="open")
    body = mock_request.call_args.kwargs["json"]
    assert body["query"] == {"field": "state", "operator": "=", "value": "open"}

    intercom.intercom_list_conversations(state="closed")
    body = mock_request.call_args.kwargs["json"]
    assert body["query"] == {"field": "state", "operator": "=", "value": "closed"}


def test_list_conversations_snoozed_filters_by_state(monkeypatch):
    """Intercom's conversation state enum is open/closed/snoozed; snoozed
    must be a selectable state, not silently unreachable except via "all"."""
    mock_request = Mock(return_value=MockResponse(json_data={"conversations": []}))
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_list_conversations(state="snoozed")

    body = mock_request.call_args.kwargs["json"]
    assert body["query"] == {"field": "state", "operator": "=", "value": "snoozed"}


def test_list_conversations_returns_total_count_and_has_more(monkeypatch):
    monkeypatch.setattr(
        intercom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "conversations": [{"id": "c1"}],
                    "total_count": 5,
                    "pages": {"next": {"starting_after": "cursor-2"}},
                }
            )
        ),
    )

    result = json.loads(intercom.intercom_list_conversations())

    assert result["status"] == "success"
    assert result["total_count"] == 5
    assert result["has_more"] is True


def test_list_conversations_has_more_false_on_last_page(monkeypatch):
    monkeypatch.setattr(
        intercom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "conversations": [{"id": "c1"}],
                    "total_count": 1,
                    "pages": {"next": None},
                }
            )
        ),
    )

    result = json.loads(intercom.intercom_list_conversations())

    assert result["has_more"] is False


def test_list_conversations_all_state_still_sends_a_query(monkeypatch):
    """Regression test: /conversations/search requires a `query` field on
    every request, including "all" -- omitting it previously produced a
    request Intercom would very likely reject with a 400."""
    mock_request = Mock(return_value=MockResponse(json_data={"conversations": []}))
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    result = json.loads(intercom.intercom_list_conversations(state="all"))

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]
    assert "query" in body
    assert body["query"]["field"] == "created_at"


def test_list_conversations_rejects_unknown_state(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    result = json.loads(intercom.intercom_list_conversations(state="bogus"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_request_does_not_retry_on_zero_retry_after(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(status_code=429, headers={"Retry-After": "0"}),
            MockResponse(json_data={"id": "c1"}),
        ]
    )
    monkeypatch.setattr(intercom.requests, "request", mock_request)
    monkeypatch.setattr(intercom.time, "sleep", Mock())

    # Retry-After of "0" does not satisfy `0 < retry_after`, so this call
    # should NOT sleep/retry and must surface the 429 as an error instead.
    result = json.loads(intercom.intercom_get_contact("c1"))
    assert result["status"] == "error"
    assert mock_request.call_count == 1


def test_request_retries_once_on_429_with_positive_retry_after(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(status_code=429, headers={"Retry-After": "1"}),
            MockResponse(json_data={"id": "c1"}),
        ]
    )
    sleep_calls = []
    monkeypatch.setattr(intercom.requests, "request", mock_request)
    monkeypatch.setattr(intercom.time, "sleep", lambda s: sleep_calls.append(s))

    result = json.loads(intercom.intercom_get_contact("c1"))

    assert result["status"] == "success"
    assert mock_request.call_count == 2
    assert sleep_calls == [1]


def test_request_does_not_retry_beyond_max_retry_after(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            status_code=429,
            headers={"Retry-After": str(intercom.MAX_RETRY_AFTER_SECONDS + 1)},
        )
    )
    monkeypatch.setattr(intercom.requests, "request", mock_request)
    monkeypatch.setattr(intercom.time, "sleep", Mock())

    result = json.loads(intercom.intercom_get_contact("c1"))

    assert result["status"] == "error"
    assert mock_request.call_count == 1


def test_admin_id_is_cached_per_token(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"id": "admin-1"}),
            MockResponse(json_data={"id": "conv1"}),
            MockResponse(json_data={"id": "conv2"}),
        ]
    )
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_reply_to_conversation("conv1", "hi")
    intercom.intercom_reply_to_conversation("conv2", "hi again")

    # /me is only called once across the two replies: the second reply hits
    # the token-keyed cache instead of re-resolving the admin id.
    me_calls = [
        call
        for call in mock_request.call_args_list
        if call.kwargs["url"].endswith("/me")
    ]
    assert len(me_calls) == 1
    assert mock_request.call_count == 3


def test_admin_id_cache_does_not_leak_across_different_tokens(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"id": "admin-token-a"}),
            MockResponse(json_data={"id": "conv1"}),
        ]
    )
    monkeypatch.setattr(intercom.requests, "request", mock_request)
    monkeypatch.setenv("INTERCOM_ACCESS_TOKEN", "token-a")
    intercom.intercom_reply_to_conversation("conv1", "hi")
    reply_body_a = mock_request.call_args.kwargs["json"]
    assert reply_body_a["admin_id"] == "admin-token-a"

    mock_request.side_effect = [
        MockResponse(json_data={"id": "admin-token-b"}),
        MockResponse(json_data={"id": "conv2"}),
    ]
    monkeypatch.setenv("INTERCOM_ACCESS_TOKEN", "token-b")
    intercom.intercom_reply_to_conversation("conv2", "hi")
    reply_body_b = mock_request.call_args.kwargs["json"]

    # A different token must resolve (and use) its own admin id, not the
    # first token's cached value.
    assert reply_body_b["admin_id"] == "admin-token-b"


def test_search_contacts_returns_success_payload(monkeypatch):
    monkeypatch.setattr(
        intercom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": [{"id": "c1", "name": "Ada", "email": "ada@example.com"}],
                    "total_count": 1,
                }
            )
        ),
    )

    result = json.loads(intercom.intercom_search_contacts("ada"))

    assert result["status"] == "success"
    assert result["contacts"] == [
        {
            "id": "c1",
            "name": "Ada",
            "email": "ada@example.com",
            "phone": None,
            "role": None,
            "last_seen_at": None,
        }
    ]
    assert result["total"] == 1


def test_search_contacts_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        intercom.requests,
        "request",
        Mock(
            return_value=MockResponse(status_code=401, text='{"message": "bad token"}')
        ),
    )

    result = json.loads(intercom.intercom_search_contacts("ada"))

    assert result["status"] == "error"
    assert "bad token" in result["message"]


def test_search_contacts_rejects_blank_query(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    result = json.loads(intercom.intercom_search_contacts("   "))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_get_contact_curates_the_response_like_search_does(monkeypatch):
    """intercom_get_contact used to return the raw provider object while
    search_contacts and get_conversation both curate via a summary helper --
    internally inconsistent, and an unbounded passthrough of whatever fields
    Intercom's API happens to include."""
    monkeypatch.setattr(
        intercom.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "id": "c1",
                    "name": "Ada",
                    "email": "ada@example.com",
                    "phone": None,
                    "role": "user",
                    "last_seen_at": 1700000000,
                    "custom_attributes": {"internal_notes": "sensitive stuff"},
                }
            )
        ),
    )

    result = json.loads(intercom.intercom_get_contact("c1"))

    assert result["contact"] == {
        "id": "c1",
        "name": "Ada",
        "email": "ada@example.com",
        "phone": None,
        "role": "user",
        "last_seen_at": 1700000000,
    }
    assert "custom_attributes" not in result["contact"]


def test_request_truncates_long_unstructured_error_body(monkeypatch):
    """An HTML gateway error page (or similar) landing in an unstructured
    error body must not be forwarded to the LLM/logs verbatim and
    unbounded, mirroring the Zoom sibling module."""
    long_body = "x" * 5000
    monkeypatch.setattr(
        intercom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    result = json.loads(intercom.intercom_get_contact("c1"))

    assert result["status"] == "error"
    assert "[truncated]" in result["message"]
    assert len(result["message"]) < len(long_body)


def test_request_returns_empty_dict_on_204(monkeypatch):
    monkeypatch.setattr(
        intercom.requests,
        "request",
        Mock(return_value=MockResponse(status_code=204)),
    )

    assert intercom._request("POST", "/conversations/c1/parts") == {}


def test_list_conversations_clamps_limit_to_the_documented_range(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"conversations": []}))
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_list_conversations(limit=500)
    assert mock_request.call_args.kwargs["json"]["pagination"]["per_page"] == 100

    intercom.intercom_list_conversations(limit=0)
    assert mock_request.call_args.kwargs["json"]["pagination"]["per_page"] == 1


def test_current_admin_id_raises_when_me_has_no_id(monkeypatch):
    monkeypatch.setattr(
        intercom.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"type": "admin"})),
    )

    with pytest.raises(RuntimeError, match="Could not resolve"):
        intercom._current_admin_id()
