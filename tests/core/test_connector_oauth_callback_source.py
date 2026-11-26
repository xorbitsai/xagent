"""Source-origin checks for the connector OAuth callback migration."""

import json

import pytest

from xagent.config import (
    get_connector_oauth_callback_base_url,
    get_connector_oauth_callback_uri,
    get_mcp_oauth_callback_base_url,
)

CALLBACK_PATH = "/api/mcp/oauth/callback"


@pytest.fixture(autouse=True)
def clean_callback_environment(monkeypatch):
    for name in (
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL",
        "XAGENT_MCP_OAUTH_CALLBACK_BASE_URL",
        "XAGENT_PUBLIC_ENTRIES_JSON",
        "XAGENT_PUBLIC_API_BASE_URL",
        "XAGENT_APP_BASE_URL",
        "XAGENT_FRONTEND_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def rewrite(uri: str) -> str:
    return get_connector_oauth_callback_uri(uri, callback_path=CALLBACK_PATH)


def test_custom_external_same_path_callback_is_untouched(monkeypatch):
    monkeypatch.setenv(
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "https://cloud.tobybot.ai"
    )
    uri = "https://customer.example/api/mcp/oauth/callback"

    assert rewrite(uri) == uri


def test_explicit_app_origin_is_moved(monkeypatch):
    monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://app.tobybot.ai")
    monkeypatch.setenv(
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "https://cloud.tobybot.ai"
    )

    assert (
        rewrite("https://app.tobybot.ai/api/mcp/oauth/callback")
        == "https://cloud.tobybot.ai/api/mcp/oauth/callback"
    )


def test_public_api_origin_is_moved(monkeypatch):
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.tobybot.ai")
    monkeypatch.setenv(
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "https://cloud.tobybot.ai"
    )

    assert (
        rewrite("https://api.tobybot.ai/api/mcp/oauth/callback")
        == "https://cloud.tobybot.ai/api/mcp/oauth/callback"
    )


@pytest.mark.parametrize(
    ("frontend_base", "uri", "expected"),
    [
        (
            "http://xagent_web:3000",
            "https://api.tobybot.ai/api/mcp/oauth/callback",
            "https://cloud.tobybot.ai/api/mcp/oauth/callback",
        ),
        (
            "http://[::1]:3000",
            "https://api.tobybot.ai/api/mcp/oauth/callback",
            "https://cloud.tobybot.ai/api/mcp/oauth/callback",
        ),
        (
            "http://xagent_web:3000",
            "http://xagent_web:3000/api/mcp/oauth/callback",
            "http://xagent_web:3000/api/mcp/oauth/callback",
        ),
        (
            "http://[::1]:3000",
            "http://[::1]:3000/api/mcp/oauth/callback",
            "http://[::1]:3000/api/mcp/oauth/callback",
        ),
        (
            "http://xagent_web:3000",
            "https://customer.example/api/mcp/oauth/callback",
            "https://customer.example/api/mcp/oauth/callback",
        ),
    ],
)
def test_unmatchable_frontend_base_does_not_abort_owned_sources(
    monkeypatch, frontend_base, uri, expected
):
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.tobybot.ai")
    monkeypatch.setenv("XAGENT_APP_BASE_URL", frontend_base)
    monkeypatch.setenv(
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "https://cloud.tobybot.ai"
    )

    assert rewrite(uri) == expected


def test_each_trusted_public_entry_is_a_source(monkeypatch):
    monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://cloud.tobybot.ai")
    monkeypatch.setenv(
        "XAGENT_PUBLIC_ENTRIES_JSON",
        json.dumps(
            [
                {
                    "origin": "https://cloud.tobybot.ai",
                    "region_cookie_domain": ".tobybot.ai",
                    "google_oidc_redirect_uri": (
                        "https://cloud.tobybot.ai/api/auth/oidc/google/callback"
                    ),
                },
                {
                    "origin": "https://tobybot.ai",
                    "region_cookie_domain": None,
                    "google_oidc_redirect_uri": (
                        "https://tobybot.ai/api/auth/oidc/google/callback"
                    ),
                },
            ]
        ),
    )
    monkeypatch.setenv(
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "https://cloud.tobybot.ai"
    )

    assert (
        rewrite("https://tobybot.ai/api/mcp/oauth/callback")
        == "https://cloud.tobybot.ai/api/mcp/oauth/callback"
    )


def test_unconfigured_localhost_fallback_is_not_trusted(monkeypatch):
    monkeypatch.setenv(
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "https://cloud.tobybot.ai"
    )
    uri = "http://localhost:3000/api/mcp/oauth/callback"

    assert rewrite(uri) == uri


def test_ipv6_loopback_destination_is_moved(monkeypatch):
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.tobybot.ai")
    monkeypatch.setenv("XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "http://[::1]:8765")

    assert get_connector_oauth_callback_base_url() == "http://[::1]:8765"
    assert (
        rewrite("https://api.tobybot.ai/api/mcp/oauth/callback")
        == "http://[::1]:8765/api/mcp/oauth/callback"
    )


def test_external_source_is_unchanged_with_an_ipv6_destination(monkeypatch):
    monkeypatch.setenv("XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "http://[::1]:8765")
    uri = "https://customer.example/api/mcp/oauth/callback"

    assert rewrite(uri) == uri


def test_equal_origin_keeps_the_current_value(monkeypatch):
    monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://APP.TOBYBOT.AI:443/")
    monkeypatch.setenv(
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "https://app.tobybot.ai"
    )
    uri = "https://APP.TOBYBOT.AI:443/api/mcp/oauth/callback"

    assert rewrite(uri) == uri


def test_ipv6_base_keeps_its_configured_presentation(monkeypatch):
    base = "https://[0:0:0:0:0:0:0:1]:443"
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.tobybot.ai")
    monkeypatch.setenv("XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", base)

    # The origin dynamic registration advertises is exactly the one the
    # static-client rewrite applies, including the configured IPv6 spelling.
    assert get_connector_oauth_callback_base_url() == base
    assert (
        rewrite("https://api.tobybot.ai/api/mcp/oauth/callback")
        == f"{base}/api/mcp/oauth/callback"
    )


@pytest.mark.parametrize(
    "uri",
    [
        "https://api.tobybot.ai/api/mcp/oauth/callback?tenant=acme",
        "https://api.tobybot.ai/api/mcp/oauth/callback#fragment",
        "https://user:pass@api.tobybot.ai/api/mcp/oauth/callback",
        "https://api.tobybot.ai/api/mcp/oauth/callback?",
        "https://api.tobybot.ai/api/mcp/oauth/callback#",
    ],
)
def test_custom_query_fragment_and_credentials_are_untouched(monkeypatch, uri):
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.tobybot.ai")
    monkeypatch.setenv(
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "https://cloud.tobybot.ai"
    )

    assert rewrite(uri) == uri


def test_non_canonical_path_is_untouched(monkeypatch):
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.tobybot.ai")
    monkeypatch.setenv(
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "https://cloud.tobybot.ai"
    )
    uri = "https://api.tobybot.ai/api/mcp/oauth/callback-extra"

    assert rewrite(uri) == uri


def test_domain_case_and_default_port_are_normalized(monkeypatch):
    monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://APP.TOBYBOT.AI:443/")
    monkeypatch.setenv(
        "XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL", "https://CLOUD.TOBYBOT.AI:443/"
    )

    assert (
        rewrite("https://App.Tobybot.AI:443/api/mcp/oauth/callback")
        == "https://cloud.tobybot.ai/api/mcp/oauth/callback"
    )


def test_legacy_mcp_advertised_origin_is_unchanged(monkeypatch):
    monkeypatch.setenv("XAGENT_MCP_OAUTH_CALLBACK_BASE_URL", "https://au.brand.example")
    uri = "https://api.tobybot.ai/api/mcp/oauth/callback"
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.tobybot.ai")

    assert get_mcp_oauth_callback_base_url() == "https://au.brand.example"
    assert get_connector_oauth_callback_base_url() is None
    assert rewrite(uri) == uri
