import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import whatsapp

GRAPH = "https://graph.facebook.com/v25.0"


class MockResponse:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json_data = json_data or {}
        self.text = text
        self.status_code = status_code
        self.content = text.encode("utf-8") if text else b"{}"

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


def _payload(result: str):
    return json.loads(result)


def _graph_error(code: int, message: str = "boom", status_code: int = 400):
    body = {"error": {"message": message, "type": "OAuthException", "code": code}}
    return MockResponse(body, text=json.dumps(body), status_code=status_code)


SEND_OK = {
    "messaging_product": "whatsapp",
    "contacts": [{"input": "15551234567", "wa_id": "15551234567"}],
    "messages": [{"id": "wamid.ABC"}],
}


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")


def _mock_request(monkeypatch, response=None, side_effect=None):
    mock = Mock(return_value=response, side_effect=side_effect)
    monkeypatch.setattr(whatsapp.requests, "request", mock)
    return mock


# --------------------------------------------------------------------------
# auth / discovery
# --------------------------------------------------------------------------


def test_auth_status_uses_injected_meta_token(token, monkeypatch):
    mock = _mock_request(monkeypatch, MockResponse({"id": "u1", "name": "Alice"}))

    result = _payload(whatsapp.whatsapp_auth_status())

    assert result == {
        "status": "success",
        "authenticated": True,
        "user": {"id": "u1", "name": "Alice", "email": None},
    }
    mock.assert_called_once_with(
        method="GET",
        url=f"{GRAPH}/me",
        headers={"Authorization": "Bearer user-token", "Accept": "application/json"},
        params={"fields": "id,name,email"},
        data=None,
        json=None,
        timeout=30,
    )


def test_auth_status_without_token_returns_error(monkeypatch):
    monkeypatch.delenv("META_ACCESS_TOKEN", raising=False)
    mock = _mock_request(monkeypatch)

    result = _payload(whatsapp.whatsapp_auth_status())

    assert result["status"] == "error"
    assert "META_ACCESS_TOKEN" in result["message"]
    mock.assert_not_called()


def test_list_business_accounts_expands_owned_and_client_wabas(token, monkeypatch):
    mock = _mock_request(
        monkeypatch,
        MockResponse(
            {
                "data": [
                    {
                        "id": "biz-1",
                        "name": "Acme",
                        "owned_whatsapp_business_accounts": {
                            "data": [
                                {
                                    "id": "waba-1",
                                    "name": "Acme WA",
                                    "currency": "USD",
                                    "timezone_id": "1",
                                    "message_template_namespace": "ns-1",
                                    "account_review_status": "APPROVED",
                                }
                            ],
                            "paging": {"cursors": {"after": "x"}},
                        },
                        "client_whatsapp_business_accounts": {
                            "data": [{"id": "waba-2", "name": "Client WA"}]
                        },
                    },
                    # waba-1 shows up again under the second business: must be
                    # deduped. No client edge at all: must be tolerated.
                    {
                        "id": "biz-2",
                        "name": "Beta",
                        "owned_whatsapp_business_accounts": {
                            "data": [{"id": "waba-1", "name": "Acme WA"}]
                        },
                    },
                    {"id": "biz-3", "name": "No WhatsApp"},
                ]
            }
        ),
    )

    result = _payload(whatsapp.whatsapp_list_business_accounts())

    assert result == {
        "status": "success",
        "accounts": [
            {
                "id": "waba-1",
                "name": "Acme WA",
                "currency": "USD",
                "timezone_id": "1",
                "message_template_namespace": "ns-1",
                "account_review_status": "APPROVED",
                "relationship": "owned",
                "business": {"id": "biz-1", "name": "Acme"},
            },
            {
                "id": "waba-2",
                "name": "Client WA",
                "currency": None,
                "timezone_id": None,
                "message_template_namespace": None,
                "account_review_status": None,
                "relationship": "client",
                "business": {"id": "biz-1", "name": "Acme"},
            },
        ],
        "next_link": None,
    }
    mock.assert_called_once()
    assert mock.call_args.kwargs["url"] == f"{GRAPH}/me/businesses"
    assert mock.call_args.kwargs["params"] == {
        "fields": (
            "id,name,"
            "owned_whatsapp_business_accounts.limit(100){id,name,currency,"
            "timezone_id,message_template_namespace,account_review_status},"
            "client_whatsapp_business_accounts.limit(100){id,name,currency,"
            "timezone_id,message_template_namespace,account_review_status}"
        ),
        "limit": 100,
    }


def test_list_business_accounts_with_no_businesses(token, monkeypatch):
    mock = _mock_request(monkeypatch, MockResponse({"data": []}))

    result = _payload(whatsapp.whatsapp_list_business_accounts())

    assert result == {"status": "success", "accounts": [], "next_link": None}
    mock.assert_called_once()


def test_list_business_accounts_surfaces_next_link_past_100_businesses(
    token, monkeypatch
):
    """A user in more than 100 businesses gets a next_link signal rather than
    a silent truncation to the first page (see BUSINESS_FIELDS's .limit(100)
    on the nested WABA edges for the corresponding per-business ceiling)."""
    mock = _mock_request(
        monkeypatch,
        MockResponse(
            {
                "data": [{"id": "biz-1", "name": "Acme"}],
                "paging": {
                    "next": "https://graph.facebook.com/v25.0/me/businesses?after=x"
                },
            }
        ),
    )

    result = _payload(whatsapp.whatsapp_list_business_accounts())

    assert result["next_link"] == (
        "https://graph.facebook.com/v25.0/me/businesses?after=x"
    )
    mock.assert_called_once()


def test_list_business_accounts_surfaces_graph_error(token, monkeypatch):
    _mock_request(
        monkeypatch,
        _graph_error(10, "(#10) Requires business_management permission"),
    )

    result = _payload(whatsapp.whatsapp_list_business_accounts())

    assert result["status"] == "error"
    assert result["details"]["error"]["code"] == 10


def test_list_phone_numbers(token, monkeypatch):
    mock = _mock_request(
        monkeypatch,
        MockResponse(
            {
                "data": [
                    {
                        "id": "pn-1",
                        "display_phone_number": "+1 555-123-4567",
                        "verified_name": "Acme",
                        "quality_rating": "GREEN",
                    }
                ],
                "paging": {"next": "https://graph.facebook.com/next"},
            }
        ),
    )

    result = _payload(whatsapp.whatsapp_list_phone_numbers("waba-1"))

    assert result["status"] == "success"
    assert result["phone_numbers"][0]["id"] == "pn-1"
    assert result["next_link"] == "https://graph.facebook.com/next"
    assert mock.call_args.kwargs["url"] == f"{GRAPH}/waba-1/phone_numbers"
    assert mock.call_args.kwargs["params"] == {"fields": whatsapp.PHONE_NUMBER_FIELDS}


def test_list_phone_numbers_url_encodes_waba_id(token, monkeypatch):
    mock = _mock_request(monkeypatch, MockResponse({"data": []}))

    whatsapp.whatsapp_list_phone_numbers("waba/../evil")

    assert mock.call_args.kwargs["url"] == f"{GRAPH}/waba%2F..%2Fevil/phone_numbers"


def test_list_phone_numbers_requires_waba_id(token, monkeypatch):
    mock = _mock_request(monkeypatch)

    result = _payload(whatsapp.whatsapp_list_phone_numbers("  "))

    assert result["status"] == "error"
    assert "required" in result["message"]
    mock.assert_not_called()


def test_get_business_profile_unwraps_data_list(token, monkeypatch):
    mock = _mock_request(
        monkeypatch,
        MockResponse(
            {"data": [{"about": "Hi", "websites": ["https://acme.test"]}]},
        ),
    )

    result = _payload(whatsapp.whatsapp_get_business_profile("pn-1"))

    assert result == {
        "status": "success",
        "profile": {"about": "Hi", "websites": ["https://acme.test"]},
    }
    assert mock.call_args.kwargs["url"] == f"{GRAPH}/pn-1/whatsapp_business_profile"
    assert mock.call_args.kwargs["params"] == {
        "fields": whatsapp.BUSINESS_PROFILE_FIELDS
    }


def test_get_business_profile_empty(token, monkeypatch):
    _mock_request(monkeypatch, MockResponse({"data": []}))

    result = _payload(whatsapp.whatsapp_get_business_profile("pn-1"))

    assert result == {"status": "success", "profile": {}}


# --------------------------------------------------------------------------
# templates
# --------------------------------------------------------------------------


def test_list_message_templates_defaults(token, monkeypatch):
    mock = _mock_request(
        monkeypatch,
        MockResponse(
            {
                "data": [{"id": "t1", "name": "hello", "status": "APPROVED"}],
                "paging": {"cursors": {"after": "x"}},
            }
        ),
    )

    result = _payload(whatsapp.whatsapp_list_message_templates("waba-1"))

    assert result == {
        "status": "success",
        "templates": [{"id": "t1", "name": "hello", "status": "APPROVED"}],
        "next_link": None,
    }
    assert mock.call_args.kwargs["url"] == f"{GRAPH}/waba-1/message_templates"
    assert mock.call_args.kwargs["params"] == {
        "fields": whatsapp.TEMPLATE_FIELDS,
        "limit": 25,
    }


def test_list_message_templates_normalizes_status_and_bounds_limit(token, monkeypatch):
    mock = _mock_request(monkeypatch, MockResponse({"data": []}))

    whatsapp.whatsapp_list_message_templates("waba-1", status=" approved ", limit=999)

    assert mock.call_args.kwargs["params"] == {
        "fields": whatsapp.TEMPLATE_FIELDS,
        "limit": 100,
        "status": "APPROVED",
    }


@pytest.mark.parametrize("status", [None, "", "   "])
def test_list_message_templates_ignores_blank_status(token, monkeypatch, status):
    mock = _mock_request(monkeypatch, MockResponse({"data": []}))

    whatsapp.whatsapp_list_message_templates("waba-1", status=status)

    assert "status" not in mock.call_args.kwargs["params"]


def test_list_message_templates_rejects_unknown_status(token, monkeypatch):
    mock = _mock_request(monkeypatch)

    result = _payload(whatsapp.whatsapp_list_message_templates("waba-1", status="live"))

    assert result["status"] == "error"
    assert "status must be one of" in result["message"]
    mock.assert_not_called()


# --------------------------------------------------------------------------
# sending: text
# --------------------------------------------------------------------------


def test_send_text_message_posts_json_payload(token, monkeypatch):
    mock = _mock_request(monkeypatch, MockResponse(SEND_OK))

    result = _payload(
        whatsapp.whatsapp_send_text_message(
            "pn-1", "+1 (555) 123-4567", "Hello there", preview_url=True
        )
    )

    assert result == {
        "status": "success",
        "message_id": "wamid.ABC",
        "recipient": "15551234567",
        "wa_id": "15551234567",
    }
    mock.assert_called_once_with(
        method="POST",
        url=f"{GRAPH}/pn-1/messages",
        headers={
            "Authorization": "Bearer user-token",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        params=None,
        data=None,
        timeout=30,
        json={
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": "15551234567",
            "type": "text",
            "text": {"preview_url": True, "body": "Hello there"},
        },
    )


def test_send_text_message_threads_reply_context(token, monkeypatch):
    mock = _mock_request(monkeypatch, MockResponse(SEND_OK))

    whatsapp.whatsapp_send_text_message(
        "pn-1", "15551234567", "Re: hi", reply_to_message_id=" wamid.PREV "
    )

    body = mock.call_args.kwargs["json"]
    assert body["context"] == {"message_id": "wamid.PREV"}
    assert body["text"] == {"preview_url": False, "body": "Re: hi"}


def test_send_text_message_surfaces_held_message_status(token, monkeypatch):
    _mock_request(
        monkeypatch,
        MockResponse(
            {
                "messaging_product": "whatsapp",
                "contacts": [{"input": "15551234567", "wa_id": "15551234567"}],
                "messages": [
                    {
                        "id": "wamid.HELD",
                        "message_status": "held_for_quality_assessment",
                    }
                ],
            }
        ),
    )

    result = _payload(whatsapp.whatsapp_send_text_message("pn-1", "15551234567", "x"))

    assert result["message_id"] == "wamid.HELD"
    assert result["message_status"] == "held_for_quality_assessment"


def test_send_text_message_errors_when_no_message_id_returned(token, monkeypatch):
    _mock_request(monkeypatch, MockResponse({"messaging_product": "whatsapp"}))

    result = _payload(whatsapp.whatsapp_send_text_message("pn-1", "15551234567", "x"))

    assert result["status"] == "error"
    assert "did not return a message id" in result["message"]


@pytest.mark.parametrize(
    ("to", "expected"),
    [
        ("+15551234567", "15551234567"),
        ("15551234567", "15551234567"),
        ("+44 20 7946 0958", "442079460958"),
        ("+1 (555) 123-4567", "15551234567"),
        ("+49.30.123456", "4930123456"),
        # "00" is the international access code many countries (e.g. the
        # UK) use in place of "+" when dialing abroad.
        ("0044 20 7946 0958", "442079460958"),
        ("00 1 555 123 4567", "15551234567"),
    ],
)
def test_normalize_recipient_accepts_common_formats(to, expected):
    assert whatsapp._normalize_recipient(to) == expected


@pytest.mark.parametrize(
    "to",
    ["", "   ", "abc", "+1555abc", "1234", "1" * 16, "1555+1234567", "+1 555 1234;"],
)
def test_normalize_recipient_rejects_malformed(to):
    with pytest.raises(ValueError):
        whatsapp._normalize_recipient(to)


def test_send_text_message_rejects_bad_recipient_before_calling_graph(
    token, monkeypatch
):
    mock = _mock_request(monkeypatch)

    result = _payload(whatsapp.whatsapp_send_text_message("pn-1", "not-a-number", "x"))

    assert result["status"] == "error"
    assert "international format" in result["message"]
    mock.assert_not_called()


@pytest.mark.parametrize("body", ["", "   "])
def test_send_text_message_requires_body(token, monkeypatch, body):
    mock = _mock_request(monkeypatch)

    result = _payload(whatsapp.whatsapp_send_text_message("pn-1", "15551234567", body))

    assert result["status"] == "error"
    assert "body is required" in result["message"]
    mock.assert_not_called()


def test_send_text_message_enforces_body_length(token, monkeypatch):
    mock = _mock_request(monkeypatch)

    result = _payload(
        whatsapp.whatsapp_send_text_message("pn-1", "15551234567", "x" * 4097)
    )

    assert result["status"] == "error"
    assert "4096" in result["message"]
    mock.assert_not_called()


def test_send_text_message_requires_phone_number_id(token, monkeypatch):
    mock = _mock_request(monkeypatch)

    result = _payload(whatsapp.whatsapp_send_text_message("", "15551234567", "hi"))

    assert result["status"] == "error"
    mock.assert_not_called()


def test_send_text_message_adds_reengagement_hint(token, monkeypatch):
    _mock_request(monkeypatch, _graph_error(131047, "Re-engagement message"))

    result = _payload(whatsapp.whatsapp_send_text_message("pn-1", "15551234567", "hi"))

    assert result["status"] == "error"
    assert result["details"]["error"]["code"] == 131047
    assert "whatsapp_send_template_message" in result["details"]["hint"]


def test_send_text_message_unknown_error_code_has_no_hint(token, monkeypatch):
    _mock_request(monkeypatch, _graph_error(999999, "weird"))

    result = _payload(whatsapp.whatsapp_send_text_message("pn-1", "15551234567", "hi"))

    assert result["status"] == "error"
    assert result["details"] == {
        "error": {"message": "weird", "type": "OAuthException", "code": 999999}
    }
    assert "hint" not in result["details"]


def test_send_error_redacts_tokens(token, monkeypatch):
    _mock_request(
        monkeypatch,
        _graph_error(131047, "token user-token rejected"),
    )

    result = whatsapp.whatsapp_send_text_message("pn-1", "15551234567", "hi")

    assert "user-token" not in result
    assert "[redacted]" in result


# --------------------------------------------------------------------------
# sending: template
# --------------------------------------------------------------------------


def test_send_template_message_with_components(token, monkeypatch):
    mock = _mock_request(monkeypatch, MockResponse(SEND_OK))
    components = [
        {"type": "body", "parameters": [{"type": "text", "text": "Alice"}]},
    ]

    result = _payload(
        whatsapp.whatsapp_send_template_message(
            "pn-1", "+15551234567", " order_update ", " en_US ", components=components
        )
    )

    assert result["status"] == "success"
    assert result["message_id"] == "wamid.ABC"
    assert mock.call_args.kwargs["json"] == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": "15551234567",
        "type": "template",
        "template": {
            "name": "order_update",
            "language": {"code": "en_US"},
            "components": components,
        },
    }


@pytest.mark.parametrize("components", [None, []])
def test_send_template_message_omits_empty_components(token, monkeypatch, components):
    mock = _mock_request(monkeypatch, MockResponse(SEND_OK))

    whatsapp.whatsapp_send_template_message(
        "pn-1", "15551234567", "hello_world", "en_US", components=components
    )

    assert "components" not in mock.call_args.kwargs["json"]["template"]


@pytest.mark.parametrize(
    ("template_name", "language_code", "needle"),
    [("", "en_US", "template_name"), ("hello", "  ", "language_code")],
)
def test_send_template_message_requires_name_and_language(
    token, monkeypatch, template_name, language_code, needle
):
    mock = _mock_request(monkeypatch)

    result = _payload(
        whatsapp.whatsapp_send_template_message(
            "pn-1", "15551234567", template_name, language_code
        )
    )

    assert result["status"] == "error"
    assert needle in result["message"]
    mock.assert_not_called()


def test_send_template_message_adds_template_hint(token, monkeypatch):
    _mock_request(monkeypatch, _graph_error(132001, "Template name does not exist"))

    result = _payload(
        whatsapp.whatsapp_send_template_message("pn-1", "15551234567", "nope", "en")
    )

    assert result["status"] == "error"
    assert "whatsapp_list_message_templates" in result["details"]["hint"]


# --------------------------------------------------------------------------
# sending: media
# --------------------------------------------------------------------------


def test_send_media_message_image_with_caption(token, monkeypatch):
    mock = _mock_request(monkeypatch, MockResponse(SEND_OK))

    result = _payload(
        whatsapp.whatsapp_send_media_message(
            "pn-1",
            "15551234567",
            " Image ",
            "https://cdn.example.com/a.png",
            caption="Look",
        )
    )

    assert result["status"] == "success"
    assert mock.call_args.kwargs["json"] == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": "15551234567",
        "type": "image",
        "image": {"link": "https://cdn.example.com/a.png", "caption": "Look"},
    }


def test_send_media_message_strips_media_url(token, monkeypatch):
    """A URL pasted with surrounding whitespace is sent trimmed, not with the
    whitespace intact -- urlparse (and so _is_public_http_url) tolerates
    leading/trailing whitespace, so it must be stripped before use, not just
    before validation."""
    mock = _mock_request(monkeypatch, MockResponse(SEND_OK))

    whatsapp.whatsapp_send_media_message(
        "pn-1", "15551234567", "image", "  https://cdn.example.com/a.png  "
    )

    assert mock.call_args.kwargs["json"]["image"] == {
        "link": "https://cdn.example.com/a.png"
    }


def test_send_media_message_document_with_filename(token, monkeypatch):
    mock = _mock_request(monkeypatch, MockResponse(SEND_OK))

    whatsapp.whatsapp_send_media_message(
        "pn-1",
        "15551234567",
        "document",
        "https://cdn.example.com/invoice.pdf",
        caption="Your invoice",
        filename=" invoice.pdf ",
    )

    assert mock.call_args.kwargs["json"]["document"] == {
        "link": "https://cdn.example.com/invoice.pdf",
        "caption": "Your invoice",
        "filename": "invoice.pdf",
    }


@pytest.mark.parametrize("caption", [None, "", "  "])
def test_send_media_message_omits_blank_caption(token, monkeypatch, caption):
    mock = _mock_request(monkeypatch, MockResponse(SEND_OK))

    whatsapp.whatsapp_send_media_message(
        "pn-1", "15551234567", "video", "https://cdn.example.com/v.mp4", caption=caption
    )

    assert mock.call_args.kwargs["json"]["video"] == {
        "link": "https://cdn.example.com/v.mp4"
    }


def test_send_media_message_rejects_caption_on_audio(token, monkeypatch):
    mock = _mock_request(monkeypatch)

    result = _payload(
        whatsapp.whatsapp_send_media_message(
            "pn-1",
            "15551234567",
            "audio",
            "https://cdn.example.com/a.ogg",
            caption="hi",
        )
    )

    assert result["status"] == "error"
    assert "audio messages do not support a caption" in result["message"]
    mock.assert_not_called()


def test_send_media_message_rejects_filename_on_non_document(token, monkeypatch):
    mock = _mock_request(monkeypatch)

    result = _payload(
        whatsapp.whatsapp_send_media_message(
            "pn-1",
            "15551234567",
            "image",
            "https://cdn.example.com/a.png",
            filename="a.png",
        )
    )

    assert result["status"] == "error"
    assert "filename is only supported for document" in result["message"]
    mock.assert_not_called()


def test_send_media_message_enforces_caption_length(token, monkeypatch):
    mock = _mock_request(monkeypatch)

    result = _payload(
        whatsapp.whatsapp_send_media_message(
            "pn-1",
            "15551234567",
            "image",
            "https://cdn.example.com/a.png",
            caption="x" * 1025,
        )
    )

    assert result["status"] == "error"
    assert "1024" in result["message"]
    mock.assert_not_called()


@pytest.mark.parametrize("media_type", ["", "sticker", "gif", None])
def test_send_media_message_rejects_unknown_media_type(token, monkeypatch, media_type):
    mock = _mock_request(monkeypatch)

    result = _payload(
        whatsapp.whatsapp_send_media_message(
            "pn-1", "15551234567", media_type, "https://cdn.example.com/a"
        )
    )

    assert result["status"] == "error"
    assert "media_type must be one of" in result["message"]
    mock.assert_not_called()


@pytest.mark.parametrize(
    "media_url", ["", "ftp://x/a.png", "file:///etc/passwd", "cdn.example.com/a.png"]
)
def test_send_media_message_requires_public_http_url(token, monkeypatch, media_url):
    mock = _mock_request(monkeypatch)

    result = _payload(
        whatsapp.whatsapp_send_media_message("pn-1", "15551234567", "image", media_url)
    )

    assert result["status"] == "error"
    assert "public http or https URL" in result["message"]
    mock.assert_not_called()


# --------------------------------------------------------------------------
# read receipts
# --------------------------------------------------------------------------


def test_mark_message_read(token, monkeypatch):
    mock = _mock_request(monkeypatch, MockResponse({"success": True}))

    result = _payload(whatsapp.whatsapp_mark_message_read("pn-1", " wamid.IN "))

    assert result == {
        "status": "success",
        "message_id": "wamid.IN",
        "marked_read": True,
    }
    assert mock.call_args.kwargs["url"] == f"{GRAPH}/pn-1/messages"
    assert mock.call_args.kwargs["json"] == {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": "wamid.IN",
    }
    assert mock.call_args.kwargs["headers"]["Content-Type"] == "application/json"


def test_mark_message_read_requires_message_id(token, monkeypatch):
    mock = _mock_request(monkeypatch)

    result = _payload(whatsapp.whatsapp_mark_message_read("pn-1", ""))

    assert result["status"] == "error"
    assert "message_id is required" in result["message"]
    mock.assert_not_called()


def test_mark_message_read_surfaces_graph_error(token, monkeypatch):
    _mock_request(monkeypatch, _graph_error(100, "Invalid parameter"))

    result = _payload(whatsapp.whatsapp_mark_message_read("pn-1", "wamid.X"))

    assert result["status"] == "error"
    assert result["details"]["error"]["code"] == 100


# --------------------------------------------------------------------------
# MCP schema: explicit nulls for optional args must not 422 before our code
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "whatsapp_list_message_templates",
            {"waba_id": "waba-1", "status": None},
        ),
        (
            "whatsapp_send_text_message",
            {
                "phone_number_id": "pn-1",
                "to": "15551234567",
                "body": "hi",
                "reply_to_message_id": None,
            },
        ),
        (
            "whatsapp_send_template_message",
            {
                "phone_number_id": "pn-1",
                "to": "15551234567",
                "template_name": "hello_world",
                "language_code": "en_US",
                "components": None,
            },
        ),
        (
            "whatsapp_send_media_message",
            {
                "phone_number_id": "pn-1",
                "to": "15551234567",
                "media_type": "image",
                "media_url": "https://cdn.example.com/a.png",
                "caption": None,
                "filename": None,
            },
        ),
    ],
)
async def test_tools_accept_explicit_null_optional_arguments(
    token, monkeypatch, tool_name, arguments
):
    _mock_request(monkeypatch, MockResponse({**SEND_OK, "data": []}))

    _content, structured = await whatsapp.mcp.call_tool(tool_name, arguments)

    assert json.loads(structured["result"])["status"] == "success"


@pytest.mark.asyncio
async def test_all_tools_are_registered():
    tools = {tool.name for tool in await whatsapp.mcp.list_tools()}

    assert tools == {
        "whatsapp_auth_status",
        "whatsapp_list_business_accounts",
        "whatsapp_list_phone_numbers",
        "whatsapp_get_business_profile",
        "whatsapp_list_message_templates",
        "whatsapp_send_text_message",
        "whatsapp_send_template_message",
        "whatsapp_send_media_message",
        "whatsapp_mark_message_read",
    }


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_whatsapp_app_registry_row():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "whatsapp"
    )
    assert row["transport"] == "oauth"
    assert row["provider_name"] == "meta"
    assert row["category"] == "Communication"
    assert row["oauth_scopes"] == [
        "business_management",
        "whatsapp_business_management",
        "whatsapp_business_messaging",
    ]
    assert row["launch_config"] == {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.whatsapp"],
        "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
    }
    assert row["is_visible_in_connector"] is True
