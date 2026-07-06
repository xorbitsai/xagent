"""Integration tests for the agent API key admin endpoints.

Covers the three endpoints at ``/api/agents/{agent_id}/api-key``:

  - POST: generate or rotate the active key (happy + reset + 401 + 404)
  - GET:  read active key metadata (happy + no-key 404 + cross-user 404)
  - DELETE: idempotent revoke (active -> revoked / no-active -> no-op /
            double-call true idempotency)

Test plumbing (TestClient, _test_db fixture, auth helpers) is shared
via ``tests/web/api/conftest.py``.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, AgentOrigin
from xagent.web.models.agent_api_key import AgentApiKey

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

# Opt this file into the shared conftest ``_test_db`` fixture without
# making it autouse globally (sibling test_tools_api.py defines its own
# DB fixture and would double-init). ``usefixtures`` takes the name as
# a string so we don't import the fixture (which would trip ruff F811).
pytestmark = pytest.mark.usefixtures("_test_db")


# Keep the prior local helper name so existing test bodies stay readable.
_headers = _admin_headers


def _create_agent(headers: dict[str, str], name: str = "Test Agent") -> int:
    """Create a minimal agent under the given user; return its id."""
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "test",
            "instructions": "You are a test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _mark_generated_manager(agent_id: int) -> None:
    db = _direct_db_session()
    try:
        db.query(Agent).filter(Agent.id == agent_id).update(
            {"origin": AgentOrigin.WORKFORCE_GENERATED_MANAGER.value}
        )
        db.commit()
    finally:
        db.close()


# ===== POST /{agent_id}/api-key =====


class TestPostGenerateApiKey:
    """POST /api/agents/{agent_id}/api-key — generate or rotate."""

    def test_happy_path_returns_full_key_and_creates_row(self):
        headers = _headers()
        agent_id = _create_agent(headers)

        resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 200, resp.text

        body = resp.json()
        # full_key format: xag_<6 alnum>_<32 alnum>
        full_key = body["full_key"]
        assert full_key.startswith("xag_")
        parts = full_key.split("_")
        assert len(parts) == 3
        assert parts[0] == "xag"
        assert len(parts[1]) == 6
        assert len(parts[2]) == 32
        assert body["key_prefix"] == parts[1]
        assert "created_at" in body

        # DB row exists, active, prefix matches
        db = _direct_db_session()
        try:
            rows = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).all()
            assert len(rows) == 1
            assert rows[0].key_prefix == body["key_prefix"]
            assert rows[0].revoked_at is None
            # Hash is bcrypt, NOT the plaintext full_key
            assert rows[0].key_hash != full_key
            assert rows[0].key_hash.startswith("$2b$12$")
        finally:
            db.close()

    def test_second_post_rotates_and_revokes_old(self):
        """Second POST revokes the old active row and creates a new one."""
        headers = _headers()
        agent_id = _create_agent(headers)

        first = client.post(f"/api/agents/{agent_id}/api-key", headers=headers).json()
        second = client.post(f"/api/agents/{agent_id}/api-key", headers=headers).json()
        assert first["full_key"] != second["full_key"]
        assert first["key_prefix"] != second["key_prefix"]

        db = _direct_db_session()
        try:
            rows = (
                db.query(AgentApiKey)
                .filter(AgentApiKey.agent_id == agent_id)
                .order_by(AgentApiKey.id)
                .all()
            )
            assert len(rows) == 2
            # First row is revoked, second is active.
            assert rows[0].revoked_at is not None
            assert rows[0].key_prefix == first["key_prefix"]
            assert rows[1].revoked_at is None
            assert rows[1].key_prefix == second["key_prefix"]
        finally:
            db.close()

    def test_unauthorized_returns_401(self):
        """No Authorization header -> 401."""
        # We still need an agent to target, but the auth gate fires before
        # ownership; create the agent under admin, then call without header.
        headers = _headers()
        agent_id = _create_agent(headers)
        resp = client.post(f"/api/agents/{agent_id}/api-key")
        # python-jose / HTTPBearer raises 401 with "Not authenticated"
        # when the header is missing; 403 is the FastAPI default for
        # HTTPBearer missing credentials. Accept either.
        assert resp.status_code in (401, 403)

    def test_other_users_agent_returns_404(self):
        """Calling POST on someone else's agent returns 404 (not 403)."""
        admin_headers = _headers()
        admin_agent_id = _create_agent(admin_headers, name="admin agent")

        bob_headers = _register_second_user()
        # Bob tries to generate a key for the admin's agent
        resp = client.post(f"/api/agents/{admin_agent_id}/api-key", headers=bob_headers)
        assert resp.status_code == 404
        # The detail must NOT indicate "permission denied" -- it must
        # look identical to "this agent does not exist".
        assert "Agent not found" in resp.json()["detail"]

    def test_nonexistent_agent_returns_404(self):
        headers = _headers()
        resp = client.post("/api/agents/9999999/api-key", headers=headers)
        assert resp.status_code == 404

    def test_integrity_error_returns_409_rotation_conflict(self):
        """Concurrent rotate race -- partial unique constraint fires at commit.

        We can't easily orchestrate two real concurrent connections in
        SQLite tests, so we monkey-patch ``Session.commit`` to raise
        ``IntegrityError`` once. That exercises the exact branch the
        production race would hit (commit fails because partial unique
        index rejects the second active row), and asserts the endpoint
        translates it into HTTP 409 with the stable ``rotation_conflict``
        code rather than leaking a 500 + raw SQL message.

        ``_create_agent`` and ``_headers`` run BEFORE the patch context,
        so the setup commits succeed; only the commit inside POST
        /api-key sees the simulated race.
        """
        headers = _headers()
        agent_id = _create_agent(headers)

        # Patch the SQLAlchemy ``Session.commit`` only inside the POST
        # call. The handler catches IntegrityError -> 409, rolls back,
        # and never calls commit again on this session, so the patch is
        # exercised exactly once.
        fake_error = IntegrityError(
            "UNIQUE constraint failed: agent_api_keys.agent_id",
            params=None,
            orig=Exception("simulated race"),
        )
        with patch.object(Session, "commit", side_effect=fake_error):
            resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        assert resp.status_code == 409
        assert resp.json()["detail"] == "rotation_conflict"
        # Crucially: the raw SQL error string must NOT appear in the
        # client-visible response.
        assert "UNIQUE constraint failed" not in resp.text
        assert "agent_api_keys" not in resp.text

    def test_internal_error_response_does_not_leak_str_e(self):
        """Non-IntegrityError 500 path must not echo str(e) to the client."""
        headers = _headers()
        agent_id = _create_agent(headers)

        # Patch commit to raise an unrelated RuntimeError -- this should
        # hit the generic ``except Exception`` branch and surface as a
        # sanitized 500 ("Internal server error"), not the raw message.
        secret_message = "secret-internal-detail-do-not-leak"
        with patch.object(Session, "commit", side_effect=RuntimeError(secret_message)):
            resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"
        assert secret_message not in resp.text

    def test_generated_manager_returns_404_without_rotating_existing_key(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        first = client.post(f"/api/agents/{agent_id}/api-key", headers=headers).json()
        _mark_generated_manager(agent_id)

        resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Agent not found"
        db = _direct_db_session()
        try:
            rows = (
                db.query(AgentApiKey)
                .filter(AgentApiKey.agent_id == agent_id)
                .order_by(AgentApiKey.id)
                .all()
            )
            assert len(rows) == 1
            assert rows[0].key_prefix == first["key_prefix"]
            assert rows[0].revoked_at is None
        finally:
            db.close()


# ===== GET /{agent_id}/api-key =====


class TestGetActiveApiKey:
    """GET /api/agents/{agent_id}/api-key — read active key metadata."""

    def test_happy_path_returns_masked(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        post_resp = client.post(
            f"/api/agents/{agent_id}/api-key", headers=headers
        ).json()
        prefix = post_resp["key_prefix"]

        resp = client.get(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["key_prefix"] == prefix
        assert body["masked_key"] == f"xag_{prefix}_••••••••"
        assert "created_at" in body
        # full_key MUST NOT be in the GET response
        assert "full_key" not in body

    def test_no_active_key_returns_404(self):
        """Agent owned but never had a key -> 404 no_active_key."""
        headers = _headers()
        agent_id = _create_agent(headers)
        resp = client.get(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 404
        assert resp.json()["detail"] == "no_active_key"

    def test_revoked_key_returns_404(self):
        """After DELETE, GET returns 404 no_active_key."""
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
        client.delete(f"/api/agents/{agent_id}/api-key", headers=headers)

        resp = client.get(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 404
        assert resp.json()["detail"] == "no_active_key"

    def test_other_users_agent_returns_404(self):
        admin_headers = _headers()
        admin_agent_id = _create_agent(admin_headers)
        client.post(f"/api/agents/{admin_agent_id}/api-key", headers=admin_headers)

        bob_headers = _register_second_user()
        resp = client.get(f"/api/agents/{admin_agent_id}/api-key", headers=bob_headers)
        assert resp.status_code == 404
        # Same "agent not found" detail -- never reveals the existence
        # of admin's key, only that the agent itself isn't bob's.
        assert resp.json()["detail"] == "Agent not found"

    def test_generated_manager_returns_404_even_if_key_exists(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
        _mark_generated_manager(agent_id)

        resp = client.get(f"/api/agents/{agent_id}/api-key", headers=headers)

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Agent not found"


# ===== DELETE /{agent_id}/api-key =====


class TestDeleteApiKey:
    """DELETE /api/agents/{agent_id}/api-key — idempotent revoke."""

    def test_revoke_active_returns_true(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        resp = client.delete(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["revoked"] is True
        assert body["revoked_at"] is not None

        # DB confirms the row is now revoked
        db = _direct_db_session()
        try:
            row = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).first()
            assert row is not None
            assert row.revoked_at is not None
        finally:
            db.close()

    def test_revoke_with_no_active_returns_false_idempotent(self):
        """DELETE on an agent with no active key is a 200 no-op."""
        headers = _headers()
        agent_id = _create_agent(headers)
        resp = client.delete(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["revoked"] is False
        assert body["revoked_at"] is None

    def test_double_revoke_is_idempotent(self):
        """Two consecutive DELETEs: first revokes, second is a no-op."""
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        first = client.delete(f"/api/agents/{agent_id}/api-key", headers=headers).json()
        assert first["revoked"] is True

        second = client.delete(
            f"/api/agents/{agent_id}/api-key", headers=headers
        ).json()
        assert second["revoked"] is False
        assert second["revoked_at"] is None

    def test_other_users_agent_returns_404(self):
        admin_headers = _headers()
        admin_agent_id = _create_agent(admin_headers)
        client.post(f"/api/agents/{admin_agent_id}/api-key", headers=admin_headers)

        bob_headers = _register_second_user()
        resp = client.delete(
            f"/api/agents/{admin_agent_id}/api-key", headers=bob_headers
        )
        assert resp.status_code == 404

    def test_generated_manager_returns_404_without_revoking_existing_key(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        first = client.post(f"/api/agents/{agent_id}/api-key", headers=headers).json()
        _mark_generated_manager(agent_id)

        resp = client.delete(f"/api/agents/{agent_id}/api-key", headers=headers)

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Agent not found"
        db = _direct_db_session()
        try:
            row = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).one()
            assert row.key_prefix == first["key_prefix"]
            assert row.revoked_at is None
        finally:
            db.close()


# ===== /api/agent-api-keys (multi-key admin surface) =====


class TestCreateMultipleKeys:
    """POST /api/agent-api-keys — unlike the legacy endpoint, does not revoke."""

    def test_second_create_does_not_revoke_first(self):
        headers = _headers()
        agent_id = _create_agent(headers)

        first = client.post(
            "/api/agent-api-keys",
            headers=headers,
            json={"agent_id": agent_id, "label": "prod"},
        ).json()
        second = client.post(
            "/api/agent-api-keys",
            headers=headers,
            json={"agent_id": agent_id, "label": "staging"},
        ).json()
        assert first["full_key"] != second["full_key"]

        db = _direct_db_session()
        try:
            rows = (
                db.query(AgentApiKey)
                .filter(AgentApiKey.agent_id == agent_id)
                .order_by(AgentApiKey.id)
                .all()
            )
            assert len(rows) == 2
            assert all(r.revoked_at is None for r in rows)
            assert rows[0].label == "prod"
            assert rows[1].label == "staging"
        finally:
            db.close()

    def test_other_users_agent_returns_404(self):
        admin_headers = _headers()
        admin_agent_id = _create_agent(admin_headers)

        bob_headers = _register_second_user()
        resp = client.post(
            "/api/agent-api-keys",
            headers=bob_headers,
            json={"agent_id": admin_agent_id, "label": "x"},
        )
        assert resp.status_code == 404

    def test_key_prefix_collision_returns_409(self):
        """Mirrors the legacy endpoint's IntegrityError -> 409 mapping."""
        headers = _headers()
        agent_id = _create_agent(headers)

        fake_error = IntegrityError(
            "UNIQUE constraint failed: agent_api_keys.key_prefix",
            params=None,
            orig=Exception("simulated race"),
        )
        with patch.object(Session, "commit", side_effect=fake_error):
            resp = client.post(
                "/api/agent-api-keys",
                headers=headers,
                json={"agent_id": agent_id, "label": "x"},
            )

        assert resp.status_code == 409
        assert resp.json()["detail"] == "rotation_conflict"
        assert "UNIQUE constraint failed" not in resp.text

    def test_internal_error_does_not_leak_str_e(self):
        headers = _headers()
        agent_id = _create_agent(headers)

        secret_message = "secret-internal-detail-do-not-leak"
        with patch.object(Session, "commit", side_effect=RuntimeError(secret_message)):
            resp = client.post(
                "/api/agent-api-keys",
                headers=headers,
                json={"agent_id": agent_id, "label": "x"},
            )

        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"
        assert secret_message not in resp.text


class TestListAndStats:
    """GET /api/agent-api-keys and /api/agent-api-keys/stats."""

    def test_list_scoped_to_caller_and_optional_agent_filter(self):
        admin_headers = _headers()
        agent_a = _create_agent(admin_headers, name="agent a")
        agent_b = _create_agent(admin_headers, name="agent b")
        client.post(
            "/api/agent-api-keys", headers=admin_headers, json={"agent_id": agent_a}
        )
        client.post(
            "/api/agent-api-keys", headers=admin_headers, json={"agent_id": agent_b}
        )

        bob_headers = _register_second_user()
        bob_agent = _create_agent(bob_headers, name="bob agent")
        client.post(
            "/api/agent-api-keys", headers=bob_headers, json={"agent_id": bob_agent}
        )

        all_admin_keys = client.get("/api/agent-api-keys", headers=admin_headers).json()
        assert len(all_admin_keys) == 2
        assert {k["agent_id"] for k in all_admin_keys} == {agent_a, agent_b}

        scoped = client.get(
            f"/api/agent-api-keys?agent_id={agent_a}", headers=admin_headers
        ).json()
        assert len(scoped) == 1
        assert scoped[0]["agent_id"] == agent_a

        bob_keys = client.get("/api/agent-api-keys", headers=bob_headers).json()
        assert len(bob_keys) == 1
        assert bob_keys[0]["agent_id"] == bob_agent

    def test_stats_counts_active_and_this_months_calls(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        created = client.post(
            "/api/agent-api-keys", headers=headers, json={"agent_id": agent_id}
        ).json()
        key_id = client.get("/api/agent-api-keys", headers=headers).json()[0]["id"]
        client.post(f"/api/agent-api-keys/{key_id}/pause", headers=headers)

        stats = client.get("/api/agent-api-keys/stats", headers=headers).json()
        assert stats["total_keys"] == 1
        assert stats["active_keys"] == 0  # paused, not active
        assert stats["calls_this_month"] == 0
        assert stats["last_api_call"] is None
        assert created["full_key"]  # sanity: creation itself succeeded

    def test_calls_this_month_survives_revocation(self):
        """Historical usage counts toward the stat even after the key that
        made those calls is revoked -- revoked keys are kept forever as an
        audit trail, so their usage shouldn't vanish from "calls this
        month" the instant they're deactivated. Deliberately pins this so
        an accidental future ``revoked_at IS NULL`` filter on the calls
        aggregate (unlike the intentional one on ``active_keys``) would
        fail this test instead of silently changing behavior.
        """
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post("/api/agent-api-keys", headers=headers, json={"agent_id": agent_id})
        key_id = client.get("/api/agent-api-keys", headers=headers).json()[0]["id"]

        current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        db = _direct_db_session()
        try:
            db.query(AgentApiKey).filter(AgentApiKey.id == key_id).update(
                {"usage_month": current_month, "usage_month_calls": 5}
            )
            db.commit()
        finally:
            db.close()

        client.delete(f"/api/agent-api-keys/{key_id}", headers=headers)

        stats = client.get("/api/agent-api-keys/stats", headers=headers).json()
        assert stats["active_keys"] == 0
        assert stats["calls_this_month"] == 5


class TestPauseResume:
    def test_pause_then_resume_round_trips_status(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post("/api/agent-api-keys", headers=headers, json={"agent_id": agent_id})
        key_id = client.get("/api/agent-api-keys", headers=headers).json()[0]["id"]

        paused = client.post(f"/api/agent-api-keys/{key_id}/pause", headers=headers)
        assert paused.status_code == 200
        assert paused.json()["status"] == "paused"

        resumed = client.post(f"/api/agent-api-keys/{key_id}/resume", headers=headers)
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "active"

    def test_pause_someone_elses_key_returns_404(self):
        admin_headers = _headers()
        agent_id = _create_agent(admin_headers)
        client.post(
            "/api/agent-api-keys", headers=admin_headers, json={"agent_id": agent_id}
        )
        key_id = client.get("/api/agent-api-keys", headers=admin_headers).json()[0][
            "id"
        ]

        bob_headers = _register_second_user()
        resp = client.post(f"/api/agent-api-keys/{key_id}/pause", headers=bob_headers)
        assert resp.status_code == 404


class TestRegenerateAndDelete:
    def test_regenerate_preserves_id_and_label_but_changes_secret(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        created = client.post(
            "/api/agent-api-keys",
            headers=headers,
            json={"agent_id": agent_id, "label": "keep-me"},
        ).json()
        key_id = client.get("/api/agent-api-keys", headers=headers).json()[0]["id"]

        regenerated = client.post(
            f"/api/agent-api-keys/{key_id}/regenerate", headers=headers
        ).json()
        assert regenerated["full_key"] != created["full_key"]
        assert regenerated["key_prefix"] != created["key_prefix"]

        listed = client.get("/api/agent-api-keys", headers=headers).json()[0]
        assert listed["id"] == key_id
        assert listed["label"] == "keep-me"
        assert listed["status"] == "active"

    def test_regenerate_key_prefix_collision_returns_409(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post("/api/agent-api-keys", headers=headers, json={"agent_id": agent_id})
        key_id = client.get("/api/agent-api-keys", headers=headers).json()[0]["id"]

        fake_error = IntegrityError(
            "UNIQUE constraint failed: agent_api_keys.key_prefix",
            params=None,
            orig=Exception("simulated race"),
        )
        with patch.object(Session, "commit", side_effect=fake_error):
            resp = client.post(
                f"/api/agent-api-keys/{key_id}/regenerate", headers=headers
            )

        assert resp.status_code == 409
        assert resp.json()["detail"] == "rotation_conflict"
        assert "UNIQUE constraint failed" not in resp.text

    def test_regenerate_internal_error_does_not_leak_str_e(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post("/api/agent-api-keys", headers=headers, json={"agent_id": agent_id})
        key_id = client.get("/api/agent-api-keys", headers=headers).json()[0]["id"]

        secret_message = "secret-internal-detail-do-not-leak"
        with patch.object(Session, "commit", side_effect=RuntimeError(secret_message)):
            resp = client.post(
                f"/api/agent-api-keys/{key_id}/regenerate", headers=headers
            )

        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"
        assert secret_message not in resp.text

    def test_delete_marks_revoked_and_hides_from_active_count(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post("/api/agent-api-keys", headers=headers, json={"agent_id": agent_id})
        key_id = client.get("/api/agent-api-keys", headers=headers).json()[0]["id"]

        resp = client.delete(f"/api/agent-api-keys/{key_id}", headers=headers)
        assert resp.status_code == 200

        listed = client.get("/api/agent-api-keys", headers=headers).json()[0]
        assert listed["status"] == "revoked"

    def test_delete_nonexistent_returns_404(self):
        headers = _headers()
        resp = client.delete("/api/agent-api-keys/9999999", headers=headers)
        assert resp.status_code == 404

    def test_regenerate_preserves_paused_status(self):
        """Regenerating a paused key must not silently resume it."""
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post("/api/agent-api-keys", headers=headers, json={"agent_id": agent_id})
        key_id = client.get("/api/agent-api-keys", headers=headers).json()[0]["id"]
        client.post(f"/api/agent-api-keys/{key_id}/pause", headers=headers)

        resp = client.post(f"/api/agent-api-keys/{key_id}/regenerate", headers=headers)
        assert resp.status_code == 200, resp.text

        listed = client.get("/api/agent-api-keys", headers=headers).json()[0]
        assert listed["status"] == "paused"

    def test_regenerate_revoked_key_returns_404(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post("/api/agent-api-keys", headers=headers, json={"agent_id": agent_id})
        key_id = client.get("/api/agent-api-keys", headers=headers).json()[0]["id"]
        client.delete(f"/api/agent-api-keys/{key_id}", headers=headers)

        resp = client.post(f"/api/agent-api-keys/{key_id}/regenerate", headers=headers)
        assert resp.status_code == 404


class TestLegacyMultiKeyInteraction:
    """The legacy single-key endpoints (/api/agents/{id}/api-key) once an
    agent has multiple keys via the new admin surface.
    """

    def test_legacy_get_returns_most_recent_non_paused_key(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post(
            "/api/agent-api-keys",
            headers=headers,
            json={"agent_id": agent_id, "label": "older"},
        )
        newer = client.post(
            "/api/agent-api-keys",
            headers=headers,
            json={"agent_id": agent_id, "label": "newer"},
        ).json()

        resp = client.get(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 200, resp.text
        assert resp.json()["key_prefix"] == newer["key_prefix"]

    def test_legacy_get_skips_paused_most_recent_key(self):
        """A paused key must never be surfaced by the legacy GET as active."""
        headers = _headers()
        agent_id = _create_agent(headers)
        older = client.post(
            "/api/agent-api-keys",
            headers=headers,
            json={"agent_id": agent_id, "label": "older"},
        ).json()
        client.post(
            "/api/agent-api-keys",
            headers=headers,
            json={"agent_id": agent_id, "label": "newer-but-paused"},
        )
        # The list is ordered created_at desc, so the just-created key is first.
        newest_key_id = client.get(
            f"/api/agent-api-keys?agent_id={agent_id}", headers=headers
        ).json()[0]["id"]
        client.post(f"/api/agent-api-keys/{newest_key_id}/pause", headers=headers)

        resp = client.get(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 200, resp.text
        assert resp.json()["key_prefix"] == older["key_prefix"]

    def test_legacy_rotate_revokes_every_multi_created_key(self):
        """Documents intended (if surprising) behavior: legacy POST rotate
        is a blunt instrument that invalidates every active key on the
        agent, not just "the" legacy one.
        """
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post("/api/agent-api-keys", headers=headers, json={"agent_id": agent_id})
        client.post("/api/agent-api-keys", headers=headers, json={"agent_id": agent_id})

        rotate_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert rotate_resp.status_code == 200, rotate_resp.text

        listed = client.get(
            f"/api/agent-api-keys?agent_id={agent_id}", headers=headers
        ).json()
        # The two pre-existing keys plus the legacy endpoint's new one.
        assert len(listed) == 3
        active = [k for k in listed if k["status"] == "active"]
        assert len(active) == 1
        assert active[0]["key_prefix"] == rotate_resp.json()["key_prefix"]
