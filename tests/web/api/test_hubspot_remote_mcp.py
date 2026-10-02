from fastapi import HTTPException

from xagent.web.api import mcp as mcp_api
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app


def test_hubspot_catalog_uses_hosted_mcp_without_exposing_credentials():
    app = get_builtin_public_mcp_app("hubspot")

    assert app is not None
    assert app["transport"] == "streamable_http"
    assert app["provider_name"] is None
    assert app["oauth_scopes"] is None
    assert app["launch_config"]["url"] == "https://mcp.hubspot.com"
    auth = app["launch_config"]["auth"]
    assert auth == {
        "type": "mcp_oauth",
        "credential_provider": "hubspot",
        "token_endpoint_auth_method": "client_secret_post",
    }
    assert "client_id" not in auth
    assert "client_secret" not in auth


def test_hubspot_catalog_credentials_resolve_from_config(monkeypatch):
    monkeypatch.setenv("XAGENT_HUBSPOT_MCP_CLIENT_ID", "client-id")
    monkeypatch.setenv("XAGENT_HUBSPOT_MCP_CLIENT_SECRET", "client-secret")

    resolved = mcp_api._resolve_catalog_mcp_oauth_auth(
        "hubspot",
        {
            "type": "mcp_oauth",
            "credential_provider": "hubspot",
            "token_endpoint_auth_method": "client_secret_post",
        },
    )

    assert resolved == {
        "type": "mcp_oauth",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "token_endpoint_auth_method": "client_secret_post",
    }


def test_hubspot_catalog_reconciliation_uses_empty_placeholders_when_missing(
    monkeypatch,
):
    monkeypatch.delenv("XAGENT_HUBSPOT_MCP_CLIENT_ID", raising=False)
    monkeypatch.delenv("XAGENT_HUBSPOT_MCP_CLIENT_SECRET", raising=False)

    assert mcp_api._resolve_catalog_mcp_oauth_auth(
        "hubspot",
        {"type": "mcp_oauth", "credential_provider": "hubspot"},
    ) == {
        "type": "mcp_oauth",
        "client_id": "",
        "client_secret": "",
    }


def test_hubspot_connect_flow_fails_closed_when_credentials_are_missing(monkeypatch):
    monkeypatch.delenv("XAGENT_HUBSPOT_MCP_CLIENT_ID", raising=False)
    monkeypatch.delenv("XAGENT_HUBSPOT_MCP_CLIENT_SECRET", raising=False)

    try:
        mcp_api._require_catalog_mcp_oauth_credentials("hubspot")
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "XAGENT_HUBSPOT_MCP_CLIENT_ID" in exc.detail
        assert "XAGENT_HUBSPOT_MCP_CLIENT_SECRET" in exc.detail
    else:
        raise AssertionError("missing HubSpot MCP credentials must fail closed")


def test_other_remote_mcp_auth_is_unchanged():
    auth = {"type": "mcp_oauth", "scope": "notes.read"}

    assert mcp_api._resolve_catalog_mcp_oauth_auth("notion", auth) == auth
