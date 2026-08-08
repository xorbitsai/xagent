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


def test_construction_degrades_to_memory_on_unusable_redis_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N12: an invalid XAGENT_REDIS_URL must not 500 the public endpoints.
    Construction runs lazily on first request, outside the per-call fail-open
    boundary, so a raising storage backend has to degrade to in-process memory
    here instead of propagating."""
    limiter = _limiter_with(monkeypatch, XAGENT_REDIS_URL="not-a-valid-scheme://x")
    assert limiter.backend == "memory"
    # And the gates still function on the fallback storage.
    assert limiter.allow_widget_auth("agent:1", "1.1.1.1") is True


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


def test_widget_ws_connect_is_per_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The widget WS handshake budget mirrors the share one (#1056): keyed per
    caller IP because it runs pre-auth, but in its own bucket so probes against
    one public channel cannot consume the other's budget."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_WS_CONNECT_IP_RATE_LIMIT="2/minute",
        XAGENT_SHARE_WS_CONNECT_IP_RATE_LIMIT="100/minute",
    )
    assert limiter.allow_widget_ws_connect("1.2.3.4") is True
    assert limiter.allow_widget_ws_connect("1.2.3.4") is True
    assert limiter.allow_widget_ws_connect("1.2.3.4") is False
    # Another caller is unaffected; a missing IP degrades to a shared bucket
    # rather than bypassing the gate.
    assert limiter.allow_widget_ws_connect("5.6.7.8") is True
    assert limiter.allow_widget_ws_connect(None) is True
    assert limiter.allow_widget_ws_connect(None) is True
    assert limiter.allow_widget_ws_connect(None) is False
    # The share connect bucket is untouched by widget attempts.
    assert limiter.allow_ws_connect("1.2.3.4") is True


def test_widget_ws_turn_per_ip_bucket_trips_across_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-IP is the tight abuser bucket (#1056): the widget guest_id is
    client-supplied (rotatable at will), so unlike the share turn gate the
    widget one is keyed on caller IP — rotating the entity (or guest) must not
    escape it."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_WS_TURN_RATE_LIMIT="100/minute",
        XAGENT_WIDGET_WS_TURN_IP_RATE_LIMIT="2/minute",
    )
    assert limiter.allow_widget_ws_turn("agent:1", "9.9.9.9") is True
    assert limiter.allow_widget_ws_turn("workforce:2", "9.9.9.9") is True
    assert limiter.allow_widget_ws_turn("agent:3", "9.9.9.9") is False
    # A different caller is unaffected.
    assert limiter.allow_widget_ws_turn("agent:1", "8.8.8.8") is True


def test_widget_ws_turn_entity_ceiling_trips_across_ips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-entity is the loose backstop bounding total owner-billed turn volume
    through one widget, across many callers."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_WS_TURN_RATE_LIMIT="2/minute",
        XAGENT_WIDGET_WS_TURN_IP_RATE_LIMIT="100/minute",
    )
    assert limiter.allow_widget_ws_turn("workforce:7", "1.1.1.1") is True
    assert limiter.allow_widget_ws_turn("workforce:7", "2.2.2.2") is True
    assert limiter.allow_widget_ws_turn("workforce:7", "3.3.3.3") is False
    # A different widget entity has its own bucket.
    assert limiter.allow_widget_ws_turn("workforce:8", "4.4.4.4") is True


def test_widget_auth_per_ip_bucket_trips_across_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-IP is the tight per-visitor/per-abuser bound (#1108 F2): rotating
    the entity key (fake widget keys, forged tickets) must not escape it."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_AUTH_RATE_LIMIT="100/minute",
        XAGENT_WIDGET_AUTH_IP_RATE_LIMIT="2/minute",
    )
    assert limiter.allow_widget_auth("agent:1", "9.9.9.9") is True
    assert limiter.allow_widget_auth("key:rotated", "9.9.9.9") is True
    assert limiter.allow_widget_auth("agent:2", "9.9.9.9") is False
    # A different caller is unaffected.
    assert limiter.allow_widget_auth("agent:1", "8.8.8.8") is True


def test_widget_auth_entity_ceiling_is_loose_across_ips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-entity is the loose aggregate backstop across a widget's visitors:
    distinct-IP visitors of one widget share it, so it must not 429 them until
    the loose cap — the F2 fix that keeps busy embeds working."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_AUTH_RATE_LIMIT="2/minute",
        XAGENT_WIDGET_AUTH_IP_RATE_LIMIT="100/minute",
    )
    assert limiter.allow_widget_auth("agent:7", "1.1.1.1") is True
    assert limiter.allow_widget_auth("agent:7", "2.2.2.2") is True
    assert limiter.allow_widget_auth("agent:7", "3.3.3.3") is False
    # A different widget entity has its own backstop.
    assert limiter.allow_widget_auth("agent:8", "4.4.4.4") is True


def test_widget_auth_malformed_env_falls_back_to_1200_not_600(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N1: a malformed XAGENT_WIDGET_AUTH_RATE_LIMIT must fall back to the same
    1200/min default the getter uses (the 4:1 ratio F1 set), not the old 600 —
    otherwise the degraded path silently halves the backstop, reintroducing
    the exact ratio problem F1 fixed."""
    limiter = _limiter_with(
        monkeypatch, XAGENT_WIDGET_AUTH_RATE_LIMIT="not-a-valid-rate"
    )
    assert limiter._widget_auth_entity_limit.amount == 1200


def test_widget_task_create_malformed_env_falls_back_to_240_not_120(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N13: the task-create entity fallback must match the getter's 240/min
    default (4:1), not the old 120 — otherwise a malformed env var silently
    runs the highest-value gate at 2:1, the same drift N1 fixed for auth."""
    limiter = _limiter_with(
        monkeypatch, XAGENT_WIDGET_TASK_CREATE_RATE_LIMIT="not-a-valid-rate"
    )
    assert limiter._widget_task_create_entity_limit.amount == 240


def test_widget_auth_entity_denial_does_not_burn_ip_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F4: a per-entity denial must not consume the caller's per-IP allowance,
    so the same IP can still reach *other* widgets (non-destructive test→hit)."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_AUTH_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_AUTH_IP_RATE_LIMIT="3/minute",
    )
    assert limiter.allow_widget_auth("agent:1", "9.9.9.9") is True  # entity=1, ip=1
    assert limiter.allow_widget_auth("agent:1", "9.9.9.9") is False  # entity backstop
    # The denial didn't burn the IP budget: the same IP reaches other widgets.
    assert limiter.allow_widget_auth("agent:2", "9.9.9.9") is True
    assert limiter.allow_widget_auth("agent:3", "9.9.9.9") is True


def test_widget_auth_is_isolated_from_share_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The widget auth buckets must not consume the share auth budget."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_AUTH_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_AUTH_IP_RATE_LIMIT="1/minute",
        XAGENT_SHARE_AUTH_RATE_LIMIT="5/minute",
        XAGENT_SHARE_AUTH_IP_RATE_LIMIT="5/minute",
    )
    assert limiter.allow_widget_auth("agent:1", "1.1.1.1") is True
    assert limiter.allow_widget_auth("agent:1", "1.1.1.1") is False
    # Same IP + token on the share endpoint still admits from its own budget.
    assert limiter.allow_auth("tok", "1.1.1.1") is True


def test_widget_gates_use_disjoint_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhausting one widget gate must not block the others for the same
    entity+IP (a copy-paste typo collapsing two widget namespaces would
    otherwise pass CI). N6: every widget limit — including the run and WS
    gates — is set to the SAME 1/minute shape, because ``limits``' MemoryStorage
    key encodes the rate-limit item, so two gates with different shapes never
    collide observably even under a merged namespace; only same-shape gates can.
    Production defaults already make ws-turn-ip / task-create-ip / upload-ip all
    60/min, so a real merge there would take effect."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_AUTH_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_AUTH_IP_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_TASK_CREATE_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_TASK_CREATE_IP_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_UPLOAD_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_UPLOAD_IP_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_WS_TURN_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_WS_TURN_IP_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_WS_CONNECT_IP_RATE_LIMIT="1/minute",
        XAGENT_WIDGET_RUN_QUOTA="1/minute",
        XAGENT_WIDGET_RUN_IP_QUOTA="1/minute",
    )
    # Exhaust the auth gate for one entity + IP.
    assert limiter.allow_widget_auth("agent:1", "1.1.1.1") is True
    assert limiter.allow_widget_auth("agent:1", "1.1.1.1") is False
    # The other widget gates for the SAME entity + IP still admit once each —
    # they live in their own namespaces, unconsumed by the auth gate. All are
    # 1/minute now, so a namespace merged onto auth's would be observable here.
    assert limiter.allow_widget_task_create("agent:1", "1.1.1.1") is True
    assert limiter.allow_widget_upload("agent:1", "1.1.1.1") is True
    assert limiter.allow_widget_ws_turn("agent:1", "1.1.1.1") is True
    assert limiter.allow_widget_ws_connect("1.1.1.1") is True
    assert limiter.widget_run_denial_reason("agent:1", "1.1.1.1") is None
    # And each is now independently exhausted, confirming they are distinct.
    assert limiter.allow_widget_task_create("agent:1", "1.1.1.1") is False
    assert limiter.allow_widget_upload("agent:1", "1.1.1.1") is False
    assert limiter.allow_widget_ws_turn("agent:1", "1.1.1.1") is False
    assert limiter.allow_widget_ws_connect("1.1.1.1") is False
    assert limiter.widget_run_denial_reason("agent:1", "1.1.1.1") is not None


def test_widget_task_create_per_ip_bucket_trips_across_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-IP is the tight abuser bucket: the widget guest_id is client-supplied
    (rotatable), so rotating the entity must not escape the per-caller gate."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_TASK_CREATE_RATE_LIMIT="100/minute",
        XAGENT_WIDGET_TASK_CREATE_IP_RATE_LIMIT="2/minute",
    )
    assert limiter.allow_widget_task_create("agent:1", "9.9.9.9") is True
    assert limiter.allow_widget_task_create("workforce:2", "9.9.9.9") is True
    assert limiter.allow_widget_task_create("agent:3", "9.9.9.9") is False
    # A different caller is unaffected.
    assert limiter.allow_widget_task_create("agent:1", "8.8.8.8") is True


def test_widget_task_create_entity_ceiling_trips_across_ips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-entity is the loose backstop bounding total task-create volume through
    one widget, across many callers."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_TASK_CREATE_RATE_LIMIT="2/minute",
        XAGENT_WIDGET_TASK_CREATE_IP_RATE_LIMIT="100/minute",
    )
    assert limiter.allow_widget_task_create("workforce:7", "1.1.1.1") is True
    assert limiter.allow_widget_task_create("workforce:7", "2.2.2.2") is True
    assert limiter.allow_widget_task_create("workforce:7", "3.3.3.3") is False
    # A different widget entity has its own bucket.
    assert limiter.allow_widget_task_create("workforce:8", "4.4.4.4") is True


def test_widget_task_create_ip_denial_does_not_consume_entity_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-IP denial must not burn an entity slot (paired non-destructive
    test→hit), so other callers of the same widget still fit."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_TASK_CREATE_RATE_LIMIT="3/minute",
        XAGENT_WIDGET_TASK_CREATE_IP_RATE_LIMIT="1/minute",
    )
    assert limiter.allow_widget_task_create("agent:1", "1.1.1.1") is True
    assert limiter.allow_widget_task_create("agent:1", "1.1.1.1") is False  # IP trips
    # The denials above didn't consume the entity budget: two fresh callers fit.
    assert limiter.allow_widget_task_create("agent:1", "2.2.2.2") is True
    assert limiter.allow_widget_task_create("agent:1", "3.3.3.3") is True


def test_widget_run_quota_per_entity_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_RUN_QUOTA="2/day",
        XAGENT_WIDGET_RUN_IP_QUOTA="100/hour",
    )
    assert limiter.widget_run_denial_reason("agent:1", "1.1.1.1") is None
    assert limiter.widget_run_denial_reason("agent:1", "2.2.2.2") is None
    assert limiter.widget_run_denial_reason("agent:1", "3.3.3.3") == "entity"
    # A different widget entity has its own rolling quota.
    assert limiter.widget_run_denial_reason("workforce:2", "4.4.4.4") is None


def test_widget_run_ip_quota_bounds_one_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-creating-IP window trips before the entity quota for one
    caller — a single IP can no longer drain a widget's whole rolling quota
    and lock other visitors out."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_RUN_QUOTA="3/day",
        XAGENT_WIDGET_RUN_IP_QUOTA="1/hour",
    )
    assert limiter.widget_run_denial_reason("agent:1", "9.9.9.9") is None
    # The IP window refuses, and says so — the caller needs "ip" vs "entity" to
    # tell a retryable per-caller throttle from an exhausted owner budget.
    assert limiter.widget_run_denial_reason("agent:1", "9.9.9.9") == "ip"
    # The denial didn't consume entity slots: two other visitors still fit.
    assert limiter.widget_run_denial_reason("agent:1", "8.8.8.8") is None  # entity=2
    assert limiter.widget_run_denial_reason("agent:1", "7.7.7.7") is None  # entity=3
    assert limiter.widget_run_denial_reason("agent:1", "6.6.6.6") == "entity"


def test_widget_run_ip_quota_is_scoped_per_widget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D1/F1: the IP sub-quota bucket is keyed ``entity|ip``, not bare IP, so
    one caller exhausting their budget on one widget must not lock that whole
    network (NAT/CGNAT egress) out of every *other* widget on the instance.
    This gate is charged per turn, so a platform-wide bucket would be
    collateral damage on ordinary shared-egress traffic."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_RUN_QUOTA="100/day",
        XAGENT_WIDGET_RUN_IP_QUOTA="1/hour",
    )
    assert limiter.widget_run_denial_reason("agent:1", "9.9.9.9") is None
    assert limiter.widget_run_denial_reason("agent:1", "9.9.9.9") == "ip"
    # Same IP, different widgets: unaffected by the exhausted agent:1 bucket.
    assert limiter.widget_run_denial_reason("agent:2", "9.9.9.9") is None
    assert limiter.widget_run_denial_reason("workforce:3", "9.9.9.9") is None


def test_widget_run_quota_without_ip_marker_is_entity_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tasks created before the IP marker existed (client_ip=None) are bounded
    by the entity quota alone — they must not collapse into one shared IP
    bucket."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_RUN_QUOTA="2/day",
        XAGENT_WIDGET_RUN_IP_QUOTA="1/hour",
    )
    # Two legacy runs pass despite the 1/hour IP window (no IP → no IP bucket).
    assert limiter.widget_run_denial_reason("agent:1", None) is None
    assert limiter.widget_run_denial_reason("agent:1", None) is None
    assert limiter.widget_run_denial_reason("agent:1", None) == "entity"


def test_widget_run_quota_is_isolated_from_share_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The widget run quota must not consume the share run budget (same key
    string is a different namespace)."""
    limiter = _limiter_with(
        monkeypatch,
        XAGENT_WIDGET_RUN_QUOTA="1/day",
        XAGENT_SHARE_RUN_QUOTA="5/day",
        XAGENT_SHARE_RUN_GUEST_QUOTA="5/hour",
    )
    assert limiter.widget_run_denial_reason("agent:1", "1.1.1.1") is None
    assert limiter.widget_run_denial_reason("agent:1", "2.2.2.2") == "entity"
    # The share run quota under the same entity key still admits.
    assert limiter.allow_run("agent:1", "g1") is True


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
    assert limiter.allow_widget_ws_connect("1.1.1.1") is True
    assert limiter.allow_widget_ws_turn("agent:1", "1.1.1.1") is True
    assert limiter.allow_run("agent:1", "g") is True
    assert limiter.allow_widget_auth("key", "1.1.1.1") is True
    assert limiter.allow_widget_task_create("agent:1", "1.1.1.1") is True
    assert limiter.widget_run_denial_reason("agent:1", "1.1.1.1") is None
    assert limiter.widget_run_denial_reason("agent:1", None) is None
