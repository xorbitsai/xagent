"""Unit tests for the share-channel rate limiter / run quota (#973, PR2)."""

from __future__ import annotations

import pytest

from xagent.web.services.share_rate_limit import (
    ShareRateLimiter,
    get_share_rate_limiter,
    reset_share_rate_limiter,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_share_rate_limiter()
    yield
    reset_share_rate_limiter()


def _limiter_with(monkeypatch: pytest.MonkeyPatch, **env: str) -> ShareRateLimiter:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    reset_share_rate_limiter()
    return get_share_rate_limiter()


def test_auth_per_token_bucket_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_SHARE_AUTH_RATE_LIMIT="2/minute",
        XAGENT_SHARE_AUTH_IP_RATE_LIMIT="100/minute",
    )
    assert limiter.allow_auth("tok", "1.1.1.1") is True
    assert limiter.allow_auth("tok", "1.1.1.1") is True
    assert limiter.allow_auth("tok", "1.1.1.1") is False


def test_auth_per_ip_ceiling_trips_across_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_SHARE_AUTH_RATE_LIMIT="100/minute",
        XAGENT_SHARE_AUTH_IP_RATE_LIMIT="2/minute",
    )
    # Rotating the share token must not escape the per-IP ceiling.
    assert limiter.allow_auth("tok-a", "9.9.9.9") is True
    assert limiter.allow_auth("tok-b", "9.9.9.9") is True
    assert limiter.allow_auth("tok-c", "9.9.9.9") is False


def test_task_create_per_guest_bucket_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_SHARE_TASK_CREATE_RATE_LIMIT="1/minute",
        XAGENT_SHARE_TASK_CREATE_TOKEN_RATE_LIMIT="100/minute",
    )
    assert limiter.allow_task_create("tok", "guest-1") is True
    assert limiter.allow_task_create("tok", "guest-1") is False
    # A different guest on the same link has an independent bucket.
    assert limiter.allow_task_create("tok", "guest-2") is True


def test_task_create_token_ceiling_trips_across_guests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_SHARE_TASK_CREATE_RATE_LIMIT="100/minute",
        XAGENT_SHARE_TASK_CREATE_TOKEN_RATE_LIMIT="2/minute",
    )
    assert limiter.allow_task_create("tok", "guest-1") is True
    assert limiter.allow_task_create("tok", "guest-2") is True
    assert limiter.allow_task_create("tok", "guest-3") is False


def test_ws_turn_and_upload_are_per_guest(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_SHARE_WS_TURN_RATE_LIMIT="1/minute",
        XAGENT_SHARE_UPLOAD_RATE_LIMIT="1/minute",
    )
    assert limiter.allow_ws_turn("g") is True
    assert limiter.allow_ws_turn("g") is False
    assert limiter.allow_ws_turn("other") is True
    assert limiter.allow_upload("g") is True
    assert limiter.allow_upload("g") is False


def test_widget_upload_per_ip_bucket_trips_across_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-IP is the tight abuser bucket: rotating the target entity (or the
    client-supplied widget guest_id, which is deliberately NOT a key) must
    not escape it."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_UPLOAD_RATE_LIMIT="100/minute",
        XAGENT_WIDGET_UPLOAD_IP_RATE_LIMIT="2/minute",
    )
    assert limiter.allow_widget_upload("agent:1", "9.9.9.9") is True
    assert limiter.allow_widget_upload("workforce:2", "9.9.9.9") is True
    assert limiter.allow_widget_upload("agent:3", "9.9.9.9") is False
    # A different caller is unaffected.
    assert limiter.allow_widget_upload("agent:1", "8.8.8.8") is True


def test_widget_upload_entity_ceiling_trips_across_ips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-entity is the loose backstop bounding total production through one
    widget, across many callers."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_UPLOAD_RATE_LIMIT="2/minute",
        XAGENT_WIDGET_UPLOAD_IP_RATE_LIMIT="100/minute",
    )
    assert limiter.allow_widget_upload("workforce:7", "1.1.1.1") is True
    assert limiter.allow_widget_upload("workforce:7", "2.2.2.2") is True
    assert limiter.allow_widget_upload("workforce:7", "3.3.3.3") is False
    # A different widget entity has its own bucket.
    assert limiter.allow_widget_upload("workforce:8", "4.4.4.4") is True


def test_ws_connect_is_per_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-auth handshake budget is keyed per caller IP (#993 F5)."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_SHARE_WS_CONNECT_IP_RATE_LIMIT="2/minute",
    )
    assert limiter.allow_ws_connect("1.2.3.4") is True
    assert limiter.allow_ws_connect("1.2.3.4") is True
    assert limiter.allow_ws_connect("1.2.3.4") is False
    # Another caller is unaffected; a missing IP degrades to a shared bucket
    # rather than bypassing the gate.
    assert limiter.allow_ws_connect("5.6.7.8") is True
    assert limiter.allow_ws_connect(None) is True
    assert limiter.allow_ws_connect(None) is True
    assert limiter.allow_ws_connect(None) is False


def test_run_quota_per_share_and_per_guest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_SHARE_RUN_QUOTA="3/day",
        XAGENT_SHARE_RUN_GUEST_QUOTA="2/hour",
    )
    # Per-guest window (2) trips before the per-share quota (3) for one guest.
    assert limiter.allow_run("agent:1", "g1") is True
    assert limiter.allow_run("agent:1", "g1") is True
    assert limiter.allow_run("agent:1", "g1") is False


def test_run_guest_denial_does_not_consume_share_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-guest window denial must not burn a per-share-quota slot."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_SHARE_RUN_QUOTA="3/day",
        XAGENT_SHARE_RUN_GUEST_QUOTA="1/hour",
    )
    assert limiter.allow_run("agent:1", "g1") is True  # share=1, g1=1
    assert limiter.allow_run("agent:1", "g1") is False  # g1 window blocks
    # g1's denials didn't consume the share quota, so two fresh guests still fit.
    assert limiter.allow_run("agent:1", "g2") is True  # share=2
    assert limiter.allow_run("agent:1", "g3") is True  # share=3
    assert limiter.allow_run("agent:1", "g4") is False  # share quota (3) hit


def test_run_share_quota_trips_across_guests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_SHARE_RUN_QUOTA="2/day",
        XAGENT_SHARE_RUN_GUEST_QUOTA="100/hour",
    )
    assert limiter.allow_run("workforce:5", "g1") is True
    assert limiter.allow_run("workforce:5", "g2") is True
    assert limiter.allow_run("workforce:5", "g3") is False


def test_all_gates_fail_open_on_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising storage backend (e.g. Redis down) must ADMIT, not 500/lock
    out — every allow_* gate is decorated to fail open. A genuine over-limit
    (returning False) is unaffected; only raised errors admit."""

    def _boom(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("redis unavailable")

    limiter = _limiter_with(monkeypatch)
    monkeypatch.setattr(limiter._limiter, "hit", _boom)
    monkeypatch.setattr(limiter._limiter, "test", _boom)

    assert limiter.allow_auth("tok", "1.1.1.1") is True
    assert limiter.allow_task_create("tok", "g") is True
    assert limiter.allow_ws_turn("g") is True
    assert limiter.allow_ws_connect("1.1.1.1") is True
    assert limiter.allow_upload("g") is True
    assert limiter.allow_widget_upload("agent:1", "1.1.1.1") is True
    assert limiter.allow_run("agent:1", "g") is True
