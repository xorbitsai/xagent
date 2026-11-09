"""The OAuth client verification command (#3580)."""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest

import xagent.web.api.auth  # noqa: F401  (see test_oauth_refresh_retry.py)
from xagent.core.utils.encryption import encrypt_value
from xagent.web import oauth_client_verification as verification
from xagent.web.builtin_mcp_registry import get_builtin_oauth_provider_rows
from xagent.web.tools import config as tool_config

CLIENT_ID = "platform-client-id"
CLIENT_SECRET = "platform-client-secret-value"
BUILTIN = {str(row["provider_name"]): row for row in get_builtin_oauth_provider_rows()}

# What each vendor answered a made-up client on 2026-11-09 (status, body),
# trimmed of trace ids.
RECORDED_REJECTED = {
    "google": (
        401,
        {
            "error": "invalid_client",
            "error_description": "The OAuth client was not found.",
        },
    ),
    "meta": (
        400,
        {
            "error": {
                "message": "Invalid Client ID",
                "type": "OAuthException",
                "code": 101,
            }
        },
    ),
    "zoom": (
        400,
        {"reason": "Invalid client_id or client_secret", "error": "invalid_client"},
    ),
    "salesforce": (
        400,
        {
            "error": "invalid_client_id",
            "error_description": "client identifier invalid",
        },
    ),
    "linear": (
        400,
        {
            "error": "invalid_client",
            "error_description": "Invalid client: client is invalid",
        },
    ),
    "xero": (400, {"error": "invalid_client"}),
    "myob": (
        400,
        {
            "error": "invalid_client",
            "error_description": "Client authentication failed.",
        },
    ),
    "employment-hero": (
        401,
        {
            "error": "invalid_client",
            "error_description": "The client identifier provided is invalid.",
        },
    ),
    "intercom": (
        404,
        {
            "type": "error.list",
            "errors": [{"code": "not_found", "message": "Client not found"}],
        },
    ),
}
# A known client with an invalid token, per each vendor's documentation.
DOCUMENTED_OK = {
    "google": (
        400,
        {
            "error": "invalid_grant",
            "error_description": "Token has been expired or revoked.",
        },
    ),
    "meta": (
        400,
        {
            "error": {
                "message": "Invalid OAuth access token.",
                "type": "OAuthException",
                "code": 190,
            }
        },
    ),
    "zoom": (400, {"error": "invalid_grant"}),
    "salesforce": (
        400,
        {"error": "invalid_grant", "error_description": "expired access/refresh token"},
    ),
    "linear": (400, {"error": "invalid_grant"}),
    "xero": (400, {"error": "invalid_grant"}),
    "myob": (400, {"error": "invalid_grant"}),
    "employment-hero": (400, {"error": "invalid_grant"}),
    "github": (200, {"error": "bad_refresh_token"}),
    "jira": (
        403,
        {
            "error": "invalid_grant",
            "error_description": "Unknown or invalid refresh token.",
        },
    ),
}
# Answers that name neither side clearly (measured 2026-11-09).
RECORDED_UNCLEAR = {
    "github": (404, {"error": "Not Found"}),
    "jira": (
        403,
        {
            "error": "unauthorized_client",
            "error_description": "refresh_token is invalid",
        },
    ),
}


def _verdict(provider: str, status: int, payload: object) -> str:
    return verification.RULES[provider](verification.vendor_answer(status, payload))


@pytest.mark.parametrize("provider", sorted(RECORDED_REJECTED))
def test_a_recorded_client_rejection_is_client_rejected(provider):
    assert _verdict(provider, *RECORDED_REJECTED[provider]) == "client_rejected"


@pytest.mark.parametrize("provider", sorted(DOCUMENTED_OK))
def test_an_invalid_token_for_an_accepted_client_is_client_ok(provider):
    assert _verdict(provider, *DOCUMENTED_OK[provider]) == "client_ok"


@pytest.mark.parametrize("provider", sorted(RECORDED_UNCLEAR))
def test_an_answer_naming_neither_side_is_inconclusive(provider):
    assert _verdict(provider, *RECORDED_UNCLEAR[provider]) == "inconclusive"


@pytest.mark.parametrize("provider", sorted(verification.RULES))
def test_a_server_error_is_inconclusive_whatever_it_says(provider):
    assert _verdict(provider, 503, {"error": "invalid_client"}) == "inconclusive"


def test_every_builtin_provider_is_covered():
    assert set(BUILTIN) == set(verification.RULES) | set(verification.UNVERIFIABLE)


def _row(provider: str, *, secret: str = CLIENT_SECRET) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            **BUILTIN[provider],
            "client_id": encrypt_value(CLIENT_ID),
            "client_secret": encrypt_value(secret),
            "redirect_uri": f"https://app.example.com/api/auth/{provider}/callback",
        }
    )


def _vendor(monkeypatch, status: int, payload: object) -> list[httpx.Request]:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, json=payload)

    monkeypatch.setattr(
        tool_config,
        "_oauth_refresh_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return sent


def test_a_rejected_client_sends_one_refresh_with_an_invalid_token(monkeypatch):
    sent = _vendor(monkeypatch, 400, {"error": "invalid_client"})

    result = asyncio.run(verification.verify_builtin_provider("xero", _row("xero")))

    assert result.verdict == "client_rejected"
    (request,) = sent
    body = {
        key: values[0] for key, values in parse_qs(request.content.decode()).items()
    }
    assert body == {
        "grant_type": "refresh_token",
        "refresh_token": verification.INVALID_TOKEN,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    assert CLIENT_SECRET not in verification.format_result(result)
    assert CLIENT_ID not in verification.format_result(result)


def test_meta_sends_the_invalid_token_as_the_token_to_exchange(monkeypatch):
    sent = _vendor(monkeypatch, *DOCUMENTED_OK["meta"])

    result = asyncio.run(verification.verify_builtin_provider("meta", _row("meta")))

    assert result.verdict == "client_ok"
    assert sent[0].method == "GET"
    assert sent[0].url.params["fb_exchange_token"] == verification.INVALID_TOKEN


def test_no_answer_is_inconclusive(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    monkeypatch.setattr(
        tool_config,
        "_oauth_refresh_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    result = asyncio.run(verification.verify_builtin_provider("linear", _row("linear")))

    assert result.verdict == "inconclusive"
    assert "ConnectTimeout" in result.detail


@pytest.mark.parametrize("provider", sorted(verification.UNVERIFIABLE))
def test_an_unverifiable_provider_sends_nothing(monkeypatch, provider):
    sent = _vendor(monkeypatch, 400, {"error": "invalid_client"})

    result = asyncio.run(verification.verify_builtin_provider(provider, _row(provider)))

    assert result.verdict == "unverifiable"
    assert result.detail == verification.UNVERIFIABLE[provider]
    assert sent == []


def test_a_missing_credential_is_left_to_oauth_readiness(monkeypatch):
    sent = _vendor(monkeypatch, 400, {"error": "invalid_client"})
    row = _row("xero")
    row.client_secret = ""
    monkeypatch.delenv("XERO_CLIENT_SECRET", raising=False)

    result = asyncio.run(verification.verify_builtin_provider("xero", row))

    assert result.verdict == "inconclusive"
    assert "oauth_readiness" in result.detail
    assert sent == []


def test_hubspot_is_listed_as_unverifiable_and_sent_nothing(monkeypatch):
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    sent = _vendor(monkeypatch, 400, {"error": "invalid_client"})

    result = verification.verify_mcp_client(get_builtin_public_mcp_app("hubspot"))

    assert (result.kind, result.name, result.verdict) == (
        "mcp-client",
        "hubspot",
        "unverifiable",
    )
    assert result.detail == verification.MCP_UNVERIFIABLE["hubspot"]
    assert sent == []


def test_a_connector_without_a_preregistered_client_is_not_listed():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    assert verification.verify_mcp_client(get_builtin_public_mcp_app("notion")) is None


def _run_main(
    monkeypatch, answers: dict[str, tuple[int, object]], handler=None
) -> tuple[int, str]:
    def answer(request: httpx.Request) -> httpx.Response:
        for provider, (status, payload) in answers.items():
            if request.url.host in str(BUILTIN[provider]["token_url"]):
                return httpx.Response(status, json=payload)
        return httpx.Response(503)

    monkeypatch.setattr(
        tool_config,
        "_oauth_refresh_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler or answer)),
    )
    monkeypatch.setattr(
        "xagent.web.oauth_readiness._load_provider_rows",
        lambda: [_row(name) for name in answers],
    )
    monkeypatch.setattr(
        "xagent.web.builtin_mcp_registry.get_builtin_oauth_provider_rows",
        lambda: [BUILTIN[name] for name in answers],
    )
    monkeypatch.setattr(
        "xagent.web.builtin_mcp_registry.get_builtin_public_mcp_app_rows", lambda: []
    )
    out = io.StringIO()
    code = verification.main([], out=out)
    return code, out.getvalue()


def test_the_command_exits_1_when_a_client_is_rejected(monkeypatch):
    code, output = _run_main(
        monkeypatch,
        {
            "xero": (400, {"error": "invalid_client"}),
            "linear": (400, {"error": "invalid_grant"}),
        },
    )

    assert code == 1
    assert (
        "CLIENT_REJECTED oauth-client:xero: client_id=row client_secret=row" in output
    )
    assert "CLIENT_OK oauth-client:linear:" in output
    assert (
        "2 credentials: 1 client_ok, 1 client_rejected, 0 inconclusive, 0 unverifiable"
        in output
    )
    assert CLIENT_SECRET not in output and CLIENT_ID not in output


def test_the_command_exits_0_when_every_client_is_accepted_or_unclear(monkeypatch):
    code, output = _run_main(
        monkeypatch,
        {
            "xero": (400, {"error": "invalid_grant"}),
            "github": (404, {"error": "Not Found"}),
        },
    )

    assert code == 0
    assert "INCONCLUSIVE oauth-client:github:" in output
    assert json.dumps(CLIENT_SECRET)[1:-1] not in output


def test_one_unreachable_vendor_does_not_hide_the_others(monkeypatch):
    """A refused egress policy raises ValueError; it is that vendor's result
    only (seen in a real run behind a proxy that resolves to private IPs)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "identity.xero.com":
            raise ValueError("Host must not resolve to a private network.")
        return httpx.Response(400, json={"error": "invalid_client"})

    code, output = _run_main(
        monkeypatch, {"xero": (0, None), "linear": (0, None)}, handler=handler
    )

    assert (
        "INCONCLUSIVE oauth-client:xero: client_id=row client_secret=row; no answer (ValueError)"
        in output
    )
    assert "CLIENT_REJECTED oauth-client:linear:" in output
    assert code == 1


def test_an_unknown_exclusion_is_a_usage_error(monkeypatch):
    _run_main(monkeypatch, {"xero": (400, {"error": "invalid_grant"})})

    with pytest.raises(SystemExit) as exited:
        verification.main(["--exclude", "not-a-provider"], out=io.StringIO())

    assert exited.value.code == 2


@pytest.mark.parametrize(
    ("provider", "status", "payload"),
    [
        # Intercom: only the measured "Client not found" answer rejects.
        (
            "intercom",
            400,
            {
                "type": "error.list",
                "errors": [
                    {
                        "code": "parameter_invalid",
                        "message": "Grant type not allowed for this client",
                    }
                ],
            },
        ),
        # Meta's code 1 is also its generic unknown error.
        (
            "meta",
            400,
            {
                "error": {
                    "type": "OAuthException",
                    "code": 1,
                    "message": "An unknown error occurred",
                }
            },
        ),
    ],
)
def test_an_answer_merely_mentioning_the_client_is_not_a_rejection(
    provider, status, payload
):
    assert _verdict(provider, status, payload) == "inconclusive"


def test_meta_code_1_about_the_client_secret_is_a_rejection():
    payload = {
        "error": {
            "type": "OAuthException",
            "code": 1,
            "message": "Error validating client secret.",
        }
    }

    assert _verdict("meta", 400, payload) == "client_rejected"


def test_a_failure_before_the_request_is_sent_is_that_credentials_result(monkeypatch):
    def refuse_proxy():
        raise ValueError("proxy must be trusted")

    monkeypatch.setattr(tool_config, "_oauth_refresh_http_client", refuse_proxy)

    result = asyncio.run(verification.verify_builtin_provider("xero", _row("xero")))

    assert result.verdict == "inconclusive"
    assert "no answer (ValueError)" in result.detail


def test_a_provider_name_is_matched_case_insensitively(monkeypatch):
    _vendor(monkeypatch, 400, {"error": "invalid_client"})
    row = _row("zoom")
    row.provider_name = "Zoom"

    result = asyncio.run(verification.verify_builtin_provider("Zoom", row))

    assert result.verdict == "client_rejected"


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (502, b"<html>bad gateway</html>"),
        (400, b"not json"),
        (200, b'{"access_token": "x"}'),
    ],
)
def test_an_answer_without_a_rejection_or_an_invalid_token_is_inconclusive(
    monkeypatch, status, body
):
    monkeypatch.setattr(
        tool_config,
        "_oauth_refresh_http_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status, content=body)
            )
        ),
    )

    result = asyncio.run(verification.verify_builtin_provider("linear", _row("linear")))

    assert result.verdict == "inconclusive"


def test_a_database_only_provider_has_no_rule(monkeypatch):
    sent = _vendor(monkeypatch, 400, {"error": "invalid_client"})
    row = SimpleNamespace(**{**vars(_row("xero")), "provider_name": "acme"})

    result = asyncio.run(verification.verify_builtin_provider("acme", row))

    assert (result.verdict, result.detail) == ("unverifiable", "no verification rule")
    assert sent == []


def test_an_excluded_provider_is_not_checked(monkeypatch):
    _run_main(
        monkeypatch,
        {
            "xero": (400, {"error": "invalid_client"}),
            "linear": (400, {"error": "invalid_grant"}),
        },
    )
    out = io.StringIO()

    code = verification.main(["--exclude", "xero"], out=out)

    assert code == 0
    assert "oauth-client:xero" not in out.getvalue()
    assert "CLIENT_OK oauth-client:linear:" in out.getvalue()


def test_none_is_not_a_name_to_exclude(monkeypatch):
    _run_main(monkeypatch, {"xero": (400, {"error": "invalid_grant"})})
    monkeypatch.setattr(
        "xagent.web.builtin_mcp_registry.get_builtin_public_mcp_app_rows",
        lambda: [
            {"app_id": "notion", "launch_config": {"auth": {"type": "mcp_oauth"}}}
        ],
    )

    with pytest.raises(SystemExit):
        verification.main(["--exclude", "None"], out=io.StringIO())
