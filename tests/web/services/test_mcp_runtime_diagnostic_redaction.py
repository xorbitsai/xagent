"""Regression test for issue #2239: MCP OAuth diagnostic dicts must not leak
query-string/userinfo secrets from a user-configured ``resource`` URL.

``mcp_oauth_runtime_diagnostic`` builds the payload that
``build_mcp_runtime_connection`` returns as-is, and that the
``/api/mcp/servers/{id}/tools`` endpoint (api/mcp.py) surfaces verbatim as an
HTTP 400 body to any user with an active (not necessarily owning) association
to the server. ``resource`` is free text the connector's creator typed in and
commonly carries an API key or token in its query string.
"""

from types import SimpleNamespace

import pytest

from xagent.web.services.mcp_runtime import mcp_oauth_runtime_diagnostic


def _server() -> SimpleNamespace:
    return SimpleNamespace(id=42, name="internal-mcp")


def test_resource_query_string_and_userinfo_are_redacted():
    leaking_resource = "https://user:token@mcp.example.test/oauth?api_key=SECRET"

    diagnostic = mcp_oauth_runtime_diagnostic(
        _server(),
        code="authorization_required",
        message="MCP OAuth grant is expired and requires reauthorization",
        resource=leaking_resource,
        scope="read",
        issuer="https://issuer.example.test/authorize?client_secret=OTHERSECRET",
    )

    assert "SECRET" not in diagnostic["resource"]
    assert "token" not in diagnostic["resource"]
    assert "?" not in diagnostic["resource"]
    assert "OTHERSECRET" not in diagnostic["issuer"]
    # Host/path survive redaction -- this is still a useful diagnostic, not a
    # blanket "authorization required" with no context.
    assert diagnostic["resource"] == "https://mcp.example.test/oauth"
    assert diagnostic["issuer"] == "https://issuer.example.test/authorize"
    # Non-URL fields are untouched by the redaction path.
    assert diagnostic["scope"] == "read"
    assert diagnostic["code"] == "authorization_required"
    assert diagnostic["server_id"] == 42


@pytest.mark.parametrize(
    "value",
    [
        "mcp.example.test/oauth?api_key=SECRET",
        "user:SECRET@mcp.example.test/oauth?api_key=OTHER",
        "user%40example.test:SECRET@mcp.example.test/oauth?api_key=OTHER",
    ],
)
def test_urls_without_authority_are_omitted(value):
    diagnostic = mcp_oauth_runtime_diagnostic(
        _server(),
        code="authorization_required",
        message="MCP OAuth grant is expired and requires reauthorization",
        resource=value,
        issuer=value,
    )

    assert diagnostic["resource"] is None
    assert diagnostic["issuer"] is None


def test_protocol_relative_resource_userinfo_and_query_are_redacted():
    # No scheme, but "//" gives urlsplit a netloc (with userinfo) to parse.
    diagnostic = mcp_oauth_runtime_diagnostic(
        _server(),
        code="authorization_required",
        message="MCP OAuth grant is expired and requires reauthorization",
        resource="//user:token@mcp.example.test/oauth?api_key=SECRET",
    )

    assert "SECRET" not in diagnostic["resource"]
    assert "token" not in diagnostic["resource"]
    assert "?" not in diagnostic["resource"]


def test_missing_resource_and_issuer_stay_none():
    diagnostic = mcp_oauth_runtime_diagnostic(
        _server(),
        code="authorization_required",
        message="MCP server authorization is required",
    )

    assert diagnostic["resource"] is None
    assert diagnostic["issuer"] is None
    assert diagnostic["scope"] == ""
