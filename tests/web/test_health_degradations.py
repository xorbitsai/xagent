"""/health surfaces active degraded-mode signals for monitoring."""

from __future__ import annotations

import asyncio
from importlib import import_module

import pytest

from xagent.web.services.ops_signals import (
    GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED,
    active_degradations,
    clear_degradation,
    register_degradation,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Empty the whole registry, not just this suite's own signal.

    Both tests below assert an exact /health payload, so any signal left
    active by an earlier module on the same xdist worker fails them. Some
    of those signals -- ``INTERACTION_HANDOFF_DEGRADED`` and
    ``INTERACTION_RUN_PARTITION_MISMATCH_DEGRADED`` -- have no production
    clear site at all, so a producing suite is the only thing that could
    remove them; clearing everything here keeps that a shared convention
    rather than a single-file obligation.
    """
    for name in list(active_degradations()):
        clear_degradation(name)
    yield
    for name in list(active_degradations()):
        clear_degradation(name)


def test_health_is_plain_ok_without_degradations() -> None:
    app_module = import_module("xagent.web.app")

    payload = asyncio.run(app_module.health_check())

    assert payload == {"status": "ok"}


def test_health_reports_active_degradations_but_stays_ok() -> None:
    """Degradations ride along for monitoring to alert on; the status stays
    healthy so container probes keep passing. Only signal names appear —
    /health is unauthenticated and the detail strings describe
    security-relevant misconfiguration."""
    app_module = import_module("xagent.web.app")
    register_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED, "service account unset")

    payload = asyncio.run(app_module.health_check())

    assert payload["status"] == "ok"
    assert payload["degradations"] == [GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED]
    assert "service account unset" not in str(payload)
