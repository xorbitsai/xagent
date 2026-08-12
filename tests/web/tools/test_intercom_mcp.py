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
    assert mock_request.call_args.kwargs["params"] is None


def test_get_conversation_percent_encodes_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": "conv1"}))
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_get_conversation("../admins")

    called_url = mock_request.call_args.kwargs["url"]
    assert called_url == "https://api.intercom.io/conversations/..%2Fadmins"


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


def test_list_conversations_open_and_closed_filter_by_state(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"conversations": []}))
    monkeypatch.setattr(intercom.requests, "request", mock_request)

    intercom.intercom_list_conversations(state="open")
    body = mock_request.call_args.kwargs["json"]
    assert body["query"] == {"field": "state", "operator": "=", "value": "open"}

    intercom.intercom_list_conversations(state="closed")
    body = mock_request.call_args.kwargs["json"]
    assert body["query"] == {"field": "state", "operator": "=", "value": "closed"}


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
