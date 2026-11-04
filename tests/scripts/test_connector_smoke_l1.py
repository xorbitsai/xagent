"""Assertion logic of the L1 connector configuration smoke check (#3245).

Every input here is a recorded response shape from the routes the check
calls; nothing touches the network.
"""

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

# Importable under both pytest and `unittest discover -s tests/scripts`.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
l1 = importlib.import_module("connector_smoke.l1")

REGISTRY_ROWS: list[dict[str, Any]] = [
    {"app_id": "slack", "is_visible_in_connector": True},
    {"app_id": "google-calendar", "is_visible_in_connector": True},
    {"app_id": "google-drive", "is_visible_in_connector": True},
    {"app_id": "gmail", "is_visible_in_connector": False},
    {"app_id": "zendesk", "is_visible_in_connector": False},
    {"app_id": "notion", "is_visible_in_connector": True},
]


def _app(
    app_id: str,
    provider: str | None,
    auth_type: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": app_id,
        "name": app_id.title(),
        "provider": provider,
        "auth_type": auth_type,
        "is_visible_in_connector": True,
        "is_connected": False,
        **extra,
    }


ZENDESK_ACCOUNT_INPUT = {
    "label": "Zendesk subdomain",
    "placeholder": "yourcompany",
    "help": "The part before .zendesk.com",
    "suffix": ".zendesk.com",
}


def _by_name(results: list[l1.CheckResult]) -> dict[str, l1.CheckResult]:
    return {result.name: result for result in results}


# --- expected app set ---------------------------------------------------------


def test_expected_set_is_registry_default_visible_apps() -> None:
    assert l1.expected_app_ids(REGISTRY_ROWS) == {
        "slack",
        "google-calendar",
        "google-drive",
        "notion",
    }


def test_expected_set_drops_excluded_and_adds_included_apps() -> None:
    expected = l1.expected_app_ids(
        REGISTRY_ROWS, exclude={"google-drive"}, include={"zendesk", "custom-crm"}
    )

    assert expected == {"slack", "google-calendar", "notion", "zendesk", "custom-crm"}


def test_excluding_an_app_the_registry_does_not_show_is_rejected() -> None:
    # A typo, or an app the registry already hides, must not silently pass.
    with pytest.raises(ValueError, match="gmail"):
        l1.expected_app_ids(REGISTRY_ROWS, exclude={"gmail"})


def test_an_app_both_included_and_excluded_is_rejected() -> None:
    with pytest.raises(ValueError, match="slack"):
        l1.expected_app_ids(REGISTRY_ROWS, exclude={"slack"}, include={"slack"})


def test_real_registry_hides_the_gated_google_apps_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The registry reads the restricted-scope gate from the runner's own
    # environment, so a deployment with the gate open needs --include.
    from xagent.config import GOOGLE_RESTRICTED_SCOPES_ENABLED
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    monkeypatch.delenv(GOOGLE_RESTRICTED_SCOPES_ENABLED, raising=False)
    rows = get_builtin_public_mcp_app_rows()

    expected = l1.expected_app_ids(rows)
    assert {"gmail", "google-drive"}.isdisjoint(expected)
    assert {"slack", "google-calendar", "notion"} <= expected
    assert {"gmail", "google-drive"} <= l1.expected_app_ids(
        rows, include={"gmail", "google-drive"}
    )
    with pytest.raises(ValueError, match="google-drive"):
        l1.expected_app_ids(rows, exclude={"google-drive"})


# --- catalog check -------------------------------------------------------------


def test_catalog_passes_when_every_expected_app_is_present_and_connectable() -> None:
    catalog = [
        _app("slack", "slack", "builtin_oauth"),
        _app("notion", None, "mcp_oauth"),
    ]

    results = l1.check_catalog(catalog, {"slack", "notion"})

    assert [(r.name, r.ok) for r in results] == [
        ("catalog:notion", True),
        ("catalog:slack", True),
    ]


def test_catalog_names_a_missing_expected_app() -> None:
    results = _by_name(
        l1.check_catalog([_app("slack", "slack", "builtin_oauth")], {"slack", "notion"})
    )

    assert results["catalog:slack"].ok
    assert not results["catalog:notion"].ok
    assert "missing" in results["catalog:notion"].reason


def test_catalog_fails_an_unconnectable_expected_app() -> None:
    results = _by_name(
        l1.check_catalog([_app("notion", None, "unconnectable")], {"notion"})
    )

    assert not results["catalog:notion"].ok
    assert "unconnectable" in results["catalog:notion"].reason


def test_catalog_fails_an_app_listed_as_hidden() -> None:
    hidden = _app("slack", "slack", "builtin_oauth", is_visible_in_connector=False)

    results = _by_name(l1.check_catalog([hidden], {"slack"}))

    assert not results["catalog:slack"].ok
    assert "hidden" in results["catalog:slack"].reason


def test_catalog_reports_an_unexpected_app() -> None:
    catalog = [
        _app("slack", "slack", "builtin_oauth"),
        _app("gmail", "google", "builtin_oauth"),
    ]

    results = _by_name(l1.check_catalog(catalog, {"slack"}))

    assert results["catalog:slack"].ok
    assert not results["catalog:gmail"].ok
    assert "unexpected" in results["catalog:gmail"].reason


def test_catalog_ignores_entries_without_an_id() -> None:
    results = l1.check_catalog([{"name": "broken"}], set())

    assert results == []


# --- OAuth launch targets ------------------------------------------------------


def test_one_target_per_builtin_oauth_provider() -> None:
    catalog = [
        _app("google-sheets", "google", "builtin_oauth"),
        _app("google-calendar", "google", "builtin_oauth"),
        _app("slack", "slack", "builtin_oauth"),
        _app("notion", None, "mcp_oauth"),
        _app("aws", None, "api_key"),
    ]

    targets = l1.select_oauth_targets(catalog)

    assert targets == [
        l1.OAuthTarget(provider="google", app_id="google-calendar"),
        l1.OAuthTarget(provider="slack", app_id="slack"),
    ]


def test_gmail_is_never_the_representative_app_when_another_exists() -> None:
    catalog = [
        _app("gmail", "google", "builtin_oauth"),
        _app("google-sheets", "google", "builtin_oauth"),
    ]

    assert l1.select_oauth_targets(catalog) == [
        l1.OAuthTarget(provider="google", app_id="google-sheets")
    ]


def test_registry_app_represents_its_provider_over_an_admin_created_one() -> None:
    # An operator-created app can request scopes the registry apps do not, and
    # a refusal of those would be blamed on the provider's configuration.
    catalog = [
        _app("a-custom-drive", "google", "builtin_oauth"),
        _app("google-sheets", "google", "builtin_oauth"),
    ]

    assert l1.select_oauth_targets(catalog, preferred={"google-sheets"}) == [
        l1.OAuthTarget(provider="google", app_id="google-sheets")
    ]


def test_account_hosted_provider_target_is_marked() -> None:
    catalog = [
        _app(
            "zendesk",
            "zendesk",
            "builtin_oauth",
            oauth_account_input=ZENDESK_ACCOUNT_INPUT,
        )
    ]

    assert l1.select_oauth_targets(catalog) == [
        l1.OAuthTarget(provider="zendesk", app_id="zendesk", needs_account=True)
    ]


def test_hidden_or_providerless_oauth_apps_are_not_targets() -> None:
    catalog = [
        _app("slack", "slack", "builtin_oauth", is_visible_in_connector=False),
        _app("orphan", None, "builtin_oauth"),
    ]

    assert l1.select_oauth_targets(catalog) == []


# --- callback origin (from the launch start) ------------------------------------


def test_callback_origin_on_the_environment_host_passes() -> None:
    result = l1.check_callback_origin(
        "slack",
        status_code=200,
        payload={
            "authorization_url": "/api/auth/slack/login?ticket=x",
            "callback_origin": "https://Cloud.Example.com:443",
        },
        expected_origin="https://cloud.example.com",
    )

    assert result.ok
    assert result.name == "oauth:slack:callback"


def test_callback_origin_falling_back_to_localhost_fails() -> None:
    # An unset <PROVIDER>_REDIRECT_URI resolves to http://localhost:8000.
    result = l1.check_callback_origin(
        "slack",
        status_code=200,
        payload={"callback_origin": "http://localhost:8000"},
        expected_origin="https://cloud.example.com",
    )

    assert not result.ok
    assert "http://localhost:8000" in result.reason


def test_callback_check_fails_when_the_launch_start_is_refused() -> None:
    result = l1.check_callback_origin(
        "slack",
        status_code=400,
        payload={"detail": "Unsupported provider"},
        expected_origin="https://cloud.example.com",
    )

    assert not result.ok
    assert "HTTP 400" in result.reason
    assert "Unsupported provider" in result.reason


# --- authorization redirect (from the launch redeem) ----------------------------

AUTHORIZE = "https://slack.com/oauth/v2/authorize"
CALLBACK = "https://cloud.example.com/api/auth/slack/callback"


def _location(**params: str) -> str:
    from urllib.parse import urlencode

    return f"{AUTHORIZE}?{urlencode(params)}"


def test_redirect_with_client_id_and_own_callback_passes() -> None:
    location = _location(
        response_type="code",
        client_id="123.456",
        redirect_uri=CALLBACK,
        state="signed-state",
    )

    result = l1.check_authorize_redirect(
        "slack",
        status_code=307,
        location=location,
        body="",
        expected_origin="https://cloud.example.com",
    )

    assert result.ok
    assert result.name == "oauth:slack:authorize"


def test_redirect_with_empty_client_id_fails() -> None:
    location = _location(client_id="", redirect_uri=CALLBACK, state="s")

    result = l1.check_authorize_redirect(
        "slack",
        status_code=307,
        location=location,
        body="",
        expected_origin="https://cloud.example.com",
    )

    assert not result.ok
    assert "client_id" in result.reason


def test_redirect_uri_on_another_host_fails() -> None:
    location = _location(
        client_id="123",
        redirect_uri="http://localhost:8000/api/auth/slack/callback",
        state="s",
    )

    result = l1.check_authorize_redirect(
        "slack",
        status_code=307,
        location=location,
        body="",
        expected_origin="https://cloud.example.com",
    )

    assert not result.ok
    assert "redirect_uri" in result.reason


def test_redirect_uri_with_the_wrong_callback_path_fails() -> None:
    location = _location(
        client_id="123",
        redirect_uri="https://cloud.example.com/api/auth/github/callback",
        state="s",
    )

    result = l1.check_authorize_redirect(
        "slack",
        status_code=307,
        location=location,
        body="",
        expected_origin="https://cloud.example.com",
    )

    assert not result.ok
    assert "/api/auth/slack/callback" in result.reason


def test_unconfigured_provider_page_fails_and_names_the_variable() -> None:
    body = (
        "<h1>Error: OAuth provider not configured</h1>"
        "<p>Missing SLACK_CLIENT_ID for provider slack.</p>"
        "<p>Configure the provider in admin settings ...</p>"
    )

    result = l1.check_authorize_redirect(
        "slack",
        status_code=500,
        location=None,
        body=body,
        expected_origin="https://cloud.example.com",
    )

    assert not result.ok
    assert "HTTP 500" in result.reason
    assert "Missing SLACK_CLIENT_ID" in result.reason


def test_failure_reasons_never_echo_the_redirect_url_or_its_values() -> None:
    secret_state = "eyJhbGciOiJIUzI1NiJ9.payload.signature"
    location = _location(
        client_id="real-client-id",
        redirect_uri="http://localhost:8000/api/auth/slack/callback",
        state=secret_state,
    )

    result = l1.check_authorize_redirect(
        "slack",
        status_code=307,
        location=location,
        body="",
        expected_origin="https://cloud.example.com",
    )

    assert not result.ok
    assert secret_state not in result.reason
    assert "real-client-id" not in result.reason
    assert AUTHORIZE not in result.reason


def test_format_result_line() -> None:
    assert l1.format_result(l1.CheckResult("catalog:slack", True)) == (
        "PASS catalog:slack"
    )
    assert l1.format_result(l1.CheckResult("catalog:x", False, "missing")) == (
        "FAIL catalog:x: missing"
    )
