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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mapping", ["env_mapping", "static_env"])
def test_stdio_transport_forwards_operator_proxy_and_ca_policy(
    monkeypatch, enabled, mapping
):
    values = {
        "HTTPS_PROXY": "http://proxy.example:3128",
        "HTTP_PROXY": "http://proxy.example:3128",
        "https_proxy": "http://lower.example:3128",
        "http_proxy": "http://lower.example:3128",
        "REQUESTS_CA_BUNDLE": "/certs/requests.pem",
        "CURL_CA_BUNDLE": "/certs/curl.pem",
        "SSL_CERT_FILE": "/certs/ssl.pem",
        "SSL_CERT_DIR": "/certs",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    if enabled:
        monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    else:
        monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)
    monkeypatch.setenv("MAPPED_UNTRUSTED_VALUE", "not-operator-policy")
    policy_keys = [
        "XAGENT_TRUSTED_EGRESS_PROXY",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ]
    launch_config = {
        "command": "python",
        "args": [],
        "env_mapping": {"LINKEDIN_ACCESS_TOKEN": "access_token"},
    }
    launch_config.setdefault(mapping, {}).update(
        {
            key: "access_token"
            if mapping == "env_mapping"
            else "MAPPED_UNTRUSTED_VALUE"
            for key in policy_keys
        }
    )
    config = WebToolConfig(
        db=None, request=None
    )._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="LinkedIn"),
        app_info={"launch_config": launch_config},
        access_token="caller-token-must-not-enable-proxy-trust",
    )
    assert config["env"]["XAGENT_TRUSTED_EGRESS_PROXY"] == ("1" if enabled else "")
    for key, value in values.items():
        assert config["env"][key] == value


@pytest.mark.parametrize("mapping", ["env_mapping", "static_env"])
def test_stdio_transport_does_not_create_ca_configuration_from_token_mapping(
    monkeypatch,
    mapping,
):
    ca_keys = ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR")
    for key in ca_keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MAPPED_UNTRUSTED_VALUE", "not-an-operator-CA-path")
    config = WebToolConfig(
        db=None, request=None
    )._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Example"),
        app_info={
            "launch_config": {
                "command": "python",
                "args": [],
                mapping: {
                    key: "access_token"
                    if mapping == "env_mapping"
                    else "MAPPED_UNTRUSTED_VALUE"
                    for key in ca_keys
                },
            }
        },
        access_token="not-a-CA-path",
    )
    assert all(key not in config["env"] for key in ca_keys)


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


def test_word_transport_inherits_parent_output_cap(monkeypatch):
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "100")
    app_info = get_builtin_public_mcp_app("word")
    assert app_info is not None

    transport_config = WebToolConfig(
        db=None, request=None
    )._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Word"),
        app_info={
            "launch_config": app_info["launch_config"],
        },
        access_token="user-access-token",
    )

    assert transport_config["env"]["XAGENT_TOOL_MAX_OUTPUT_LENGTH"] == "100"


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
