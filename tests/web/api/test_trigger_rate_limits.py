"""Rate limiting and ingress hardening tests for trigger endpoints."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.models.trigger import TriggerAudit, TriggerRun
from xagent.web.services.trigger_providers import sign_webhook_payload
from xagent.web.services.trigger_rate_limit import (
    TriggerRateLimiter,
    remote_ip_from_request,
    reset_trigger_rate_limiter,
    warn_if_rate_limits_are_per_process,
)

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def mock_bg_scheduler():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


@pytest.fixture(autouse=True)
def _fresh_rate_limiter():
    reset_trigger_rate_limiter()
    yield
    reset_trigger_rate_limiter()


def _create_agent(headers: dict[str, str]) -> int:
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "Rate Limit Agent",
            "description": "test",
            "instructions": "You are a rate limit test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    return int(resp.json()["id"])


def _create_webhook(headers: dict[str, str], agent_id: int) -> dict:
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Rate limited webhook"},
    )
    assert created.status_code == 200, created.text
    return created.json()


def _signed_headers(secret: str, raw_body: bytes, event_id: str) -> dict[str, str]:
    timestamp = str(int(time.time()))
    return {
        "x-xagent-signature": sign_webhook_payload(secret, timestamp, raw_body),
        "x-xagent-timestamp": timestamp,
        "x-xagent-event-id": event_id,
    }


class TestCallbackRateLimit:
    def test_callback_over_limit_returns_429(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        headers = _admin_headers()
        agent_id = _create_agent(headers)
        trigger = _create_webhook(headers, agent_id)
        monkeypatch.setenv("XAGENT_TRIGGER_CALLBACK_RATE_LIMIT", "2/minute")
        reset_trigger_rate_limiter()

        url = f"/api/triggers/callback/webhook/{trigger['callback_id']}"
        raw_body = b"{}"
        for index in range(2):
            fired = client.post(
                url,
                headers=_signed_headers(
                    trigger["webhook_secret"], raw_body, f"evt-{index}"
                ),
                content=raw_body,
            )
            assert fired.status_code == 200, fired.text

        limited = client.post(
            url,
            headers=_signed_headers(trigger["webhook_secret"], raw_body, "evt-x"),
            content=raw_body,
        )
        assert limited.status_code == 429

    def test_rate_limited_requests_leave_one_deduplicated_audit_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """429s are audited (#722 AC) but deduplicated per source per window,
        so sustained throttled traffic cannot amplify database writes."""
        monkeypatch.setenv("XAGENT_TRIGGER_CALLBACK_RATE_LIMIT", "1/minute")
        reset_trigger_rate_limiter()

        url = "/api/triggers/callback/webhook/garbage-callback-id"
        first = client.post(url, content=b"\x00garbage")
        # First request passes the limiter and is audited as unknown callback.
        assert first.status_code == 401 or first.status_code == 404

        db = _direct_db_session()
        try:
            audits_after_first = db.query(TriggerAudit).count()
        finally:
            db.close()

        for _ in range(3):
            limited = client.post(url, content=b"\x00garbage")
            assert limited.status_code == 429

        db = _direct_db_session()
        try:
            # Exactly one rate_limited row for the whole 429 burst.
            rate_limited = (
                db.query(TriggerAudit)
                .filter(TriggerAudit.outcome == "rate_limited")
                .all()
            )
            assert len(rate_limited) == 1
            assert rate_limited[0].callback_id == "garbage-callback-id"
            assert rate_limited[0].detail == {"route": "unified_callback"}
            assert db.query(TriggerAudit).count() == audits_after_first + 1
            assert db.query(TriggerRun).count() == 0
        finally:
            db.close()

    def test_legacy_route_rate_limited_audit_records_callback_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deprecated webhook route's 429 audit row carries the webhook
        token as callback_id, matching the unified route's forensic detail."""
        monkeypatch.setenv("XAGENT_TRIGGER_CALLBACK_RATE_LIMIT", "1/minute")
        reset_trigger_rate_limiter()

        url = "/api/triggers/webhook/legacy-token"
        first = client.post(url, content=b"{}")
        assert first.status_code != 429

        limited = client.post(url, content=b"{}")
        assert limited.status_code == 429

        db = _direct_db_session()
        try:
            rate_limited = (
                db.query(TriggerAudit)
                .filter(TriggerAudit.outcome == "rate_limited")
                .all()
            )
            assert len(rate_limited) == 1
            assert rate_limited[0].callback_id == "legacy-token"
            assert rate_limited[0].detail == {"route": "legacy_webhook"}
        finally:
            db.close()

    def test_rate_limit_key_includes_callback_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAGENT_TRIGGER_CALLBACK_RATE_LIMIT", "1/minute")
        reset_trigger_rate_limiter()

        first = client.post("/api/triggers/callback/webhook/cb-a", content=b"{}")
        assert first.status_code != 429
        other_callback = client.post(
            "/api/triggers/callback/webhook/cb-b", content=b"{}"
        )
        # A different callback id has its own bucket.
        assert other_callback.status_code != 429
        same_callback = client.post(
            "/api/triggers/callback/webhook/cb-a", content=b"{}"
        )
        assert same_callback.status_code == 429


class TestQueryStringSecrets:
    def test_secret_in_query_string_is_rejected(self) -> None:
        headers = _admin_headers()
        agent_id = _create_agent(headers)
        trigger = _create_webhook(headers, agent_id)

        response = client.post(
            f"/api/triggers/callback/webhook/{trigger['callback_id']}"
            f"?token={trigger['webhook_secret']}",
            content=b"{}",
        )
        assert response.status_code == 400
        assert "header" in response.json()["detail"].lower()

        db = _direct_db_session()
        try:
            assert db.query(TriggerRun).count() == 0
        finally:
            db.close()


class TestCrudRateLimit:
    def test_trigger_crud_over_limit_returns_429(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        headers = _admin_headers()
        agent_id = _create_agent(headers)
        monkeypatch.setenv("XAGENT_TRIGGER_CRUD_RATE_LIMIT", "2/minute")
        reset_trigger_rate_limiter()

        first = _create_webhook(headers, agent_id)
        patched = client.patch(
            f"/api/agents/{agent_id}/triggers/{first['id']}",
            headers=headers,
            json={"name": "Renamed"},
        )
        assert patched.status_code == 200, patched.text

        limited = client.delete(
            f"/api/agents/{agent_id}/triggers/{first['id']}",
            headers=headers,
        )
        assert limited.status_code == 429

    def test_trigger_list_is_not_rate_limited_by_crud_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        headers = _admin_headers()
        agent_id = _create_agent(headers)
        monkeypatch.setenv("XAGENT_TRIGGER_CRUD_RATE_LIMIT", "1/minute")
        reset_trigger_rate_limiter()
        _create_webhook(headers, agent_id)

        for _ in range(3):
            listed = client.get(f"/api/agents/{agent_id}/triggers", headers=headers)
            assert listed.status_code == 200


class TestRemoteIpDerivation:
    def _request(self, peer: str, forwarded: str | None) -> SimpleNamespace:
        headers = {"x-forwarded-for": forwarded} if forwarded else {}
        return SimpleNamespace(
            client=SimpleNamespace(host=peer),
            headers=headers,
        )

    def test_peer_address_without_trusted_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XAGENT_TRUSTED_PROXY_HOPS", raising=False)
        request = self._request("10.0.0.1", "203.0.113.7")
        assert remote_ip_from_request(request) == "10.0.0.1"

    def test_forwarded_header_with_one_trusted_hop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAGENT_TRUSTED_PROXY_HOPS", "1")
        request = self._request("10.0.0.1", "203.0.113.7")
        assert remote_ip_from_request(request) == "203.0.113.7"

    def test_forged_prefix_entries_are_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAGENT_TRUSTED_PROXY_HOPS", "1")
        # Attacker sends a forged X-Forwarded-For; the trusted proxy appends
        # the real client, which is the rightmost (and only trusted) entry.
        request = self._request("10.0.0.1", "1.2.3.4, 203.0.113.7")
        assert remote_ip_from_request(request) == "203.0.113.7"

    def test_non_ip_forwarded_entry_falls_back_to_peer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#1108 F3: the selected entry is client-controlled when hops
        over-counts the real chain, and callers now persist this value and use
        it as rate-limit key material — so a non-IP-shaped entry must be
        rejected in favour of the peer address rather than passed through."""
        monkeypatch.setenv("XAGENT_TRUSTED_PROXY_HOPS", "1")
        for forged in ("not-an-ip", "a" * 5000, "203.0.113.7 evil", ""):
            request = self._request("10.0.0.1", forged or None)
            assert remote_ip_from_request(request) == "10.0.0.1", forged
        # A genuine IPv6 entry still resolves normally.
        request = self._request("10.0.0.1", "2001:db8::1")
        assert remote_ip_from_request(request) == "2001:db8::1"

    def test_missing_forwarded_header_falls_back_to_peer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAGENT_TRUSTED_PROXY_HOPS", "1")
        request = self._request("10.0.0.1", None)
        assert remote_ip_from_request(request) == "10.0.0.1"

    def test_port_suffixed_entries_resolve_to_the_bare_ip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#1185: some proxies append the client's source port. Rejecting
        those would collapse every client behind such a proxy onto the peer
        address — one shared rate-limit bucket. The port is dropped so the
        bucket stays keyed per client, not per ephemeral connection."""
        monkeypatch.setenv("XAGENT_TRUSTED_PROXY_HOPS", "1")
        cases = {
            "203.0.113.7:8080": "203.0.113.7",
            "203.0.113.7:0": "203.0.113.7",
            "203.0.113.7:65535": "203.0.113.7",
            "[2001:db8::1]:8080": "2001:db8::1",
            # Brackets with no port are still a valid IPv6 reference.
            "[2001:db8::1]": "2001:db8::1",
            # Brackets are stripped without checking the host is IPv6. No real
            # proxy emits this (RFC 3986 reserves brackets for IPv6), but it
            # yields the same string the bare form would, so it cannot reach a
            # different bucket than its canonical spelling.
            "[1.2.3.4]:80": "1.2.3.4",
        }
        for forwarded, expected in cases.items():
            request = self._request("10.0.0.1", forwarded)
            assert remote_ip_from_request(request) == expected, forwarded

    def test_bare_ipv6_is_never_split_on_its_last_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#1185 regression guard: an unbracketed IPv6 literal can end in a
        group that looks like a port. Stripping it would silently rewrite the
        entry into a *different*, still-valid address — worse than the
        peer-address fallback, since nothing would flag it."""
        monkeypatch.setenv("XAGENT_TRUSTED_PROXY_HOPS", "1")
        for forwarded in ("2001:db8::1:8080", "2001:db8::1", "::1"):
            request = self._request("10.0.0.1", forwarded)
            assert remote_ip_from_request(request) == forwarded, forwarded

    def test_malformed_port_and_bracket_forms_fall_back_to_peer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Port stripping must not widen what counts as IP-shaped: the value
        is persisted and used as rate-limit key material (#1108)."""
        monkeypatch.setenv("XAGENT_TRUSTED_PROXY_HOPS", "1")
        forged = (
            "203.0.113.7:evil",
            "203.0.113.7:99999",
            "203.0.113.7:8080x",
            # Longer than sys.get_int_max_str_digits() (4300): int() raises on
            # these, so the length bound must reject them before conversion or
            # a client-controlled header becomes an unhandled 500.
            "203.0.113.7:" + "9" * 5000,
            # str.isdigit() accepts non-ASCII digits that int() then rejects.
            "203.0.113.7:٨٠",
            "203.0.113.7:²",
            "203.0.113.7:",
            ":8080",
            "[2001:db8::1",
            "[2001:db8::1]8080",
            "[2001:db8::1]:",
            "[2001:db8::1]:evil",
            "[]",
            "[]:80",
            "[not-an-ip]:80",
            "evil:8080",
            # Multi-colon non-IP: pins the no-bracket, no-port fallthrough in
            # _split_host_port, which is otherwise only reached by entries that
            # do parse as IPv6.
            "a:b:c",
            # ipaddress.ip_address() accepts an unbounded IPv6 zone-id, which
            # would otherwise be returned verbatim and persisted as the client
            # IP — the exact unbounded-string injection this validation exists
            # to prevent.
            "::1%eth0",
            "::1%" + "A" * 4000,
            "[fe80::1%eth0]:80",
        )
        for entry in forged:
            request = self._request("10.0.0.1", entry)
            assert remote_ip_from_request(request) == "10.0.0.1", entry


class TestLimiterConfiguration:
    def test_memory_storage_without_redis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
        limiter = TriggerRateLimiter()
        assert limiter.backend == "memory"

    def test_redis_storage_when_redis_url_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6399/0")
        limiter = TriggerRateLimiter()
        assert limiter.backend == "redis"

    def test_invalid_rate_string_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
        monkeypatch.setenv("XAGENT_TRIGGER_CALLBACK_RATE_LIMIT", "not-a-rate")
        limiter = TriggerRateLimiter()
        assert limiter.hit_callback("cb", "1.1.1.1") is True

    def test_multiprocess_without_redis_warns_instead_of_failing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
        monkeypatch.setenv("WEB_CONCURRENCY", "4")
        with caplog.at_level("WARNING"):
            warn_if_rate_limits_are_per_process()
        assert any("per process" in record.message for record in caplog.records)

    def test_single_process_without_redis_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
        monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
        monkeypatch.delenv("UVICORN_WORKERS", raising=False)
        monkeypatch.delenv("GUNICORN_WORKERS", raising=False)
        with caplog.at_level("WARNING"):
            warn_if_rate_limits_are_per_process()
        assert not [
            record for record in caplog.records if "per process" in record.message
        ]


class TestCallbackIpCeiling:
    def test_rotating_callback_ids_hit_the_ip_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single IP cannot bypass limiting by minting fresh callback ids."""
        monkeypatch.setenv("XAGENT_TRIGGER_CALLBACK_RATE_LIMIT", "100/minute")
        monkeypatch.setenv("XAGENT_TRIGGER_CALLBACK_IP_RATE_LIMIT", "3/minute")
        reset_trigger_rate_limiter()

        for index in range(3):
            response = client.post(
                f"/api/triggers/callback/webhook/rotating-{index}", content=b"{}"
            )
            assert response.status_code != 429

        db = _direct_db_session()
        try:
            audits_at_ceiling = db.query(TriggerAudit).count()
        finally:
            db.close()

        for index in range(3, 8):
            limited = client.post(
                f"/api/triggers/callback/webhook/rotating-{index}", content=b"{}"
            )
            assert limited.status_code == 429

        db = _direct_db_session()
        try:
            # No audit-write amplification once the IP ceiling kicks in:
            # rotating callback ids share the per-IP dedup bucket, so the
            # whole 429 burst produces exactly one rate_limited row.
            assert db.query(TriggerAudit).count() == audits_at_ceiling + 1
            assert (
                db.query(TriggerAudit)
                .filter(TriggerAudit.outcome == "rate_limited")
                .count()
                == 1
            )
        finally:
            db.close()
