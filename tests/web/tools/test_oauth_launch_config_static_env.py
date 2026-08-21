"""Tests for forwarding platform-level static secrets (e.g. a shared API
developer token) into OAuth-transport MCP subprocess environments, alongside
the existing per-user OAuth access token forwarded via env_mapping."""

from types import SimpleNamespace

import pytest

from xagent.web.tools.config import (
    WebToolConfig,
    _oauth_launch_config_static_env,
    _OAuthInstanceUrlRequired,
)


def test_static_env_returns_empty_mapping_when_absent():
    assert _oauth_launch_config_static_env({}) == {}


def test_static_env_returns_mapping_when_present():
    launch_config = {
        "static_env": {"GOOGLE_ADS_DEVELOPER_TOKEN": "GOOGLE_ADS_DEVELOPER_TOKEN"}
    }

    assert _oauth_launch_config_static_env(launch_config) == {
        "GOOGLE_ADS_DEVELOPER_TOKEN": "GOOGLE_ADS_DEVELOPER_TOKEN"
    }


def test_static_env_ignores_non_mapping_value(caplog):
    launch_config = {"static_env": ["not", "a", "mapping"]}

    assert _oauth_launch_config_static_env(launch_config) == {}


def test_transport_config_forwards_static_env_value(monkeypatch):
    monkeypatch.setenv("GOOGLE_ADS_DEVELOPER_TOKEN", "dev-token-value")

    cfg = WebToolConfig(db=None, request=None)
    app_info = {
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.google_ads"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            "static_env": {"GOOGLE_ADS_DEVELOPER_TOKEN": "GOOGLE_ADS_DEVELOPER_TOKEN"},
        }
    }

    transport_config = cfg._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Google Ads"),
        app_info=app_info,
        access_token="user-access-token",
    )

    assert transport_config["env"]["GOOGLE_ACCESS_TOKEN"] == "user-access-token"
    assert transport_config["env"]["GOOGLE_ADS_DEVELOPER_TOKEN"] == "dev-token-value"


def test_transport_config_forwards_empty_string_static_env_value(monkeypatch):
    """An explicitly empty host value is still forwarded (distinct from unset)."""
    monkeypatch.setenv("GOOGLE_ADS_DEVELOPER_TOKEN", "")

    cfg = WebToolConfig(db=None, request=None)
    app_info = {
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.google_ads"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            "static_env": {"GOOGLE_ADS_DEVELOPER_TOKEN": "GOOGLE_ADS_DEVELOPER_TOKEN"},
        }
    }

    transport_config = cfg._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Google Ads"),
        app_info=app_info,
        access_token="user-access-token",
    )

    assert transport_config["env"]["GOOGLE_ADS_DEVELOPER_TOKEN"] == ""


def test_transport_config_omits_static_env_when_host_var_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_ADS_DEVELOPER_TOKEN", raising=False)

    cfg = WebToolConfig(db=None, request=None)
    app_info = {
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.google_ads"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            "static_env": {"GOOGLE_ADS_DEVELOPER_TOKEN": "GOOGLE_ADS_DEVELOPER_TOKEN"},
        }
    }

    transport_config = cfg._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Google Ads"),
        app_info=app_info,
        access_token="user-access-token",
    )

    assert "GOOGLE_ADS_DEVELOPER_TOKEN" not in transport_config["env"]


def test_transport_config_forwards_instance_url_when_mapped_and_provided():
    """Salesforce (and no other provider) maps a second env_mapping entry to
    "instance_url" -- the per-org API host from the OAuth grant, distinct
    from static_env's operator-wide, non-per-user secrets."""
    cfg = WebToolConfig(db=None, request=None)
    app_info = {
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.salesforce"],
            "env_mapping": {
                "SALESFORCE_ACCESS_TOKEN": "access_token",
                "SALESFORCE_INSTANCE_URL": "instance_url",
            },
        }
    }

    transport_config = cfg._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Salesforce"),
        app_info=app_info,
        access_token="user-access-token",
        instance_url="https://acme.my.salesforce.com",
    )

    assert transport_config["env"]["SALESFORCE_ACCESS_TOKEN"] == "user-access-token"
    assert (
        transport_config["env"]["SALESFORCE_INSTANCE_URL"]
        == "https://acme.my.salesforce.com"
    )


def test_transport_config_raises_when_instance_url_mapped_but_not_provided():
    """A launch_config that maps an env_mapping entry to "instance_url" but
    gets no value for it (pre-migration UserOAuth row, hook path that hasn't
    wired instance_url through, etc.) must not silently launch a connector
    missing a value it declared as required -- that only turns into an
    opaque failure on the connector's first real tool call. It should
    surface as unavailable/reconnect-required instead, same as a missing
    access_token."""
    cfg = WebToolConfig(db=None, request=None)
    app_info = {
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.salesforce"],
            "env_mapping": {
                "SALESFORCE_ACCESS_TOKEN": "access_token",
                "SALESFORCE_INSTANCE_URL": "instance_url",
            },
        }
    }

    with pytest.raises(_OAuthInstanceUrlRequired):
        cfg._build_oauth_mcp_stdio_transport_config(
            server=SimpleNamespace(name="Salesforce"),
            app_info=app_info,
            access_token="user-access-token",
        )


def test_transport_config_warns_and_omits_env_var_for_unrecognized_token_type(
    caplog,
):
    """launch_config is not developer-only: POST /admin/mcp/apps accepts it
    as an unvalidated free-form dict (PublicMCPAppCreate's validator only
    checks command/required_env/url/auth.type), so a hand-typed custom
    OAuth app's env_mapping (e.g. "acess_token", a typo) can reach this
    branch too, not just a typo in this codebase's own registry."""
    cfg = WebToolConfig(db=None, request=None)
    app_info = {
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.custom_app"],
            "env_mapping": {"CUSTOM_ACCESS_TOKEN": "acess_token"},
        }
    }

    with caplog.at_level("WARNING"):
        transport_config = cfg._build_oauth_mcp_stdio_transport_config(
            server=SimpleNamespace(name="Custom App"),
            app_info=app_info,
            access_token="user-access-token",
        )

    assert "CUSTOM_ACCESS_TOKEN" not in transport_config["env"]
    assert "acess_token" in caplog.text
