"""A provider is Composio-only only when every app it serves is."""

from __future__ import annotations

import pytest

from xagent.web import composio_apps

SWITCHES = (
    (
        composio_apps.FACEBOOK_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV,
    ),
    (
        composio_apps.INSTAGRAM_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID_ENV,
    ),
    (
        composio_apps.GMAIL_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_GMAIL_AUTH_CONFIG_ID_ENV,
    ),
    (
        composio_apps.GOOGLE_DRIVE_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_GOOGLE_DRIVE_AUTH_CONFIG_ID_ENV,
    ),
    (
        composio_apps.SALESFORCE_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_SALESFORCE_AUTH_CONFIG_ID_ENV,
    ),
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for switch, auth_config in SWITCHES:
        monkeypatch.delenv(switch, raising=False)
        monkeypatch.delenv(auth_config, raising=False)
    monkeypatch.delenv(composio_apps.COMPOSIO_API_KEY_ENV, raising=False)
    monkeypatch.delenv(composio_apps.COMPOSIO_USER_ID_SECRET_ENV, raising=False)
    composio_apps._WARNED_BACKENDS.clear()


def _on(monkeypatch, switch, auth_config, *, api_key="key"):
    monkeypatch.setenv(switch, "composio")
    monkeypatch.setenv(auth_config, "ac_test")
    monkeypatch.setenv(composio_apps.COMPOSIO_USER_ID_SECRET_ENV, "secret")
    if api_key is not None:
        monkeypatch.setenv(composio_apps.COMPOSIO_API_KEY_ENV, api_key)


def test_every_app_on_makes_the_provider_composio_only(monkeypatch):
    _on(
        monkeypatch,
        composio_apps.SALESFORCE_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_SALESFORCE_AUTH_CONFIG_ID_ENV,
    )

    assert composio_apps.provider_is_composio_only("salesforce", ["salesforce"])


def test_one_app_off_keeps_the_native_client(monkeypatch):
    _on(
        monkeypatch,
        composio_apps.FACEBOOK_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV,
    )

    assert not composio_apps.provider_is_composio_only(
        "meta", ["facebook", "instagram"]
    )


def test_a_switched_on_app_whose_web_connect_stays_native_keeps_the_native_client(
    monkeypatch,
):
    """Every app's switch is on, yet one web connect does not go through Composio."""
    _on(
        monkeypatch,
        composio_apps.SALESFORCE_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_SALESFORCE_AUTH_CONFIG_ID_ENV,
    )
    extra = composio_apps.COMPOSIO_APPS["salesforce"]._replace(
        app_id="salesforce-extra", web_start=False
    )
    monkeypatch.setitem(composio_apps.COMPOSIO_APPS, "salesforce-extra", extra)

    assert not composio_apps.provider_is_composio_only(
        "salesforce", ["salesforce", "salesforce-extra"]
    )


def test_one_app_without_a_composio_app_keeps_the_native_client(monkeypatch):
    _on(
        monkeypatch,
        composio_apps.SALESFORCE_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_SALESFORCE_AUTH_CONFIG_ID_ENV,
    )

    assert not composio_apps.provider_is_composio_only(
        "salesforce", ["salesforce", "salesforce-admin"]
    )


def test_a_provider_with_a_native_only_app_is_never_composio_only(monkeypatch):
    _on(
        monkeypatch,
        composio_apps.GMAIL_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_GMAIL_AUTH_CONFIG_ID_ENV,
    )
    _on(
        monkeypatch,
        composio_apps.GOOGLE_DRIVE_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_GOOGLE_DRIVE_AUTH_CONFIG_ID_ENV,
    )
    _on(
        monkeypatch,
        composio_apps.FACEBOOK_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV,
    )
    _on(
        monkeypatch,
        composio_apps.INSTAGRAM_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID_ENV,
    )

    assert not composio_apps.provider_is_composio_only(
        "google", ["gmail", "google-drive", "google-calendar"]
    )
    assert not composio_apps.provider_is_composio_only(
        "meta", ["facebook", "instagram", "meta-ads", "whatsapp"]
    )


def test_missing_composio_config_keeps_the_native_client(monkeypatch):
    _on(
        monkeypatch,
        composio_apps.SALESFORCE_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_SALESFORCE_AUTH_CONFIG_ID_ENV,
        api_key=None,
    )

    assert not composio_apps.provider_is_composio_only("salesforce", ["salesforce"])


def test_the_real_registry_keeps_google_and_meta_native(monkeypatch):
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    for switch, auth_config in SWITCHES:
        _on(monkeypatch, switch, auth_config)
    rows = get_builtin_public_mcp_app_rows()

    def app_ids(provider: str) -> list[str]:
        return [
            str(row["app_id"]) for row in rows if row.get("provider_name") == provider
        ]

    assert not composio_apps.provider_is_composio_only("google", app_ids("google"))
    assert not composio_apps.provider_is_composio_only("meta", app_ids("meta"))
    assert composio_apps.provider_is_composio_only("salesforce", app_ids("salesforce"))


def test_a_provider_without_apps_is_not_composio_only(monkeypatch):
    _on(
        monkeypatch,
        composio_apps.SALESFORCE_AUTH_BACKEND_ENV,
        composio_apps.COMPOSIO_SALESFORCE_AUTH_CONFIG_ID_ENV,
    )

    assert not composio_apps.provider_is_composio_only("salesforce", [])
