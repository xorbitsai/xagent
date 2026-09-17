"""Tests for the degraded-mode ops-signal registry and its Gmail OIDC check."""

from __future__ import annotations

import pytest

from xagent.web.services.ops_signals import (
    GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED,
    GMAIL_WATCH_REGISTRATION_DISABLED,
    active_degradations,
    clear_degradation,
    register_degradation,
)
from xagent.web.services.trigger_providers.gmail import (
    warn_if_gmail_oidc_verification_degraded,
    warn_if_gmail_watch_registration_degraded,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED)
    clear_degradation(GMAIL_WATCH_REGISTRATION_DISABLED)
    yield
    clear_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED)
    clear_degradation(GMAIL_WATCH_REGISTRATION_DISABLED)


def test_register_and_clear_degradation() -> None:
    assert GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED not in active_degradations()
    register_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED, "detail one")
    register_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED, "detail two")
    assert active_degradations()[GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED] == "detail two"
    clear_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED)
    clear_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED)  # idempotent
    assert GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED not in active_degradations()


def test_startup_check_registers_signal_on_config_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gmail Pub/Sub configured without the push service account is config
    drift: OIDC verification would silently skip service-account email
    checks, so the degradation signal is raised at startup."""
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", "demo-project")
    monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT", raising=False)

    warn_if_gmail_oidc_verification_degraded()

    assert GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED in active_degradations()


def test_startup_check_is_silent_when_service_account_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", "demo-project")
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "pubsub-push@demo-project.iam.gserviceaccount.com",
    )

    warn_if_gmail_oidc_verification_degraded()

    assert GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED not in active_degradations()


def test_startup_check_is_silent_when_gmail_pubsub_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", raising=False)
    monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT", raising=False)

    warn_if_gmail_oidc_verification_degraded()

    assert GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED not in active_degradations()


@pytest.mark.parametrize(
    ("project_id_set", "watch_enabled", "expect_signal"),
    [
        (True, "false", True),
        (True, "true", False),
        (False, "false", False),
        (False, "true", False),
    ],
)
def test_watch_registration_degraded_signal_only_when_project_set_and_flag_off(
    monkeypatch: pytest.MonkeyPatch,
    project_id_set: bool,
    watch_enabled: str,
    expect_signal: bool,
) -> None:
    """A configured Pub/Sub project with the watch feature off means Gmail
    watch registration and renewal are disabled: Gmail triggers report
    failed provisioning and existing watches expire unrenewed (#1231)."""
    if project_id_set:
        monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", "demo-project")
    else:
        monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", raising=False)
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", watch_enabled)

    warn_if_gmail_watch_registration_degraded()

    if expect_signal:
        assert GMAIL_WATCH_REGISTRATION_DISABLED in active_degradations()
    else:
        assert GMAIL_WATCH_REGISTRATION_DISABLED not in active_degradations()
