"""Tests for forwarding platform-level static secrets (e.g. a shared API
developer token) into OAuth-transport MCP subprocess environments, alongside
the existing per-user OAuth access token forwarded via env_mapping."""

from types import SimpleNamespace

from xagent.web.tools.config import (
    WebToolConfig,
    _oauth_launch_config_static_env,
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
