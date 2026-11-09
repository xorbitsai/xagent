"""A refresh the vendor rejects for the platform's client (#3580)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.web.api.auth  # noqa: F401  (see test_oauth_refresh_retry.py)
from xagent.core.utils.encryption import encrypt_value
from xagent.web.builtin_mcp_registry import get_builtin_oauth_provider_rows
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.oauth_client_verification import is_client_rejected
from xagent.web.services import mcp_oauth
from xagent.web.tools import config as tool_config

# Microsoft's answer once a client secret has expired, as on 2026-11-05/06.
MICROSOFT_EXPIRED_SECRET = {
    "error": "invalid_client",
    "error_description": "AADSTS7000222: The provided client secret keys for app "
    "'00000000-0000-0000-0000-000000000000' are expired.",
}


@pytest.mark.parametrize(
    ("provider", "status", "payload", "rejected"),
    [
        ("microsoft", 401, MICROSOFT_EXPIRED_SECRET, True),
        # Providers the verification command cannot check: a real refresh
        # carries a valid token, so invalid_client means the client.
        ("linkedin", 401, {"error": "invalid_client"}, True),
        ("zendesk", 401, {"error": "invalid_client"}, True),
        ("slack", 200, {"ok": False, "error": "bad_client_secret"}, True),
        ("slack", 200, {"ok": False, "error": "invalid_client_id"}, True),
        ("meta", 400, {"error": {"type": "OAuthException", "code": 101}}, True),
        ("github", 200, {"error": "incorrect_client_credentials"}, True),
        ("salesforce", 400, {"error": "invalid_client_id"}, True),
        # A dead grant, an unclear answer or a vendor outage is not the client.
        ("microsoft", 400, {"error": "invalid_grant"}, False),
        ("slack", 200, {"ok": False, "error": "invalid_refresh_token"}, False),
        ("meta", 400, {"error": {"type": "OAuthException", "code": 190}}, False),
        ("jira", 403, {"error": "unauthorized_client"}, False),
        ("google", 503, {"error": "invalid_client"}, False),
        ("xero", 400, None, False),
    ],
)
def test_which_refresh_failures_reject_the_client(provider, status, payload, rejected):
    assert is_client_rejected(provider, status, payload) is rejected


@pytest.fixture()
def grant(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'rejected.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False)()
    row = next(
        r
        for r in get_builtin_oauth_provider_rows()
        if r["provider_name"] == "microsoft"
    )
    db.add(
        OAuthProvider(
            **{
                **row,
                "client_id": encrypt_value("client-id"),
                "client_secret": encrypt_value("expired-secret"),
            }
        )
    )
    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    account = UserOAuth(
        user_id=user.id,
        provider="microsoft",
        access_token="old-access",
        refresh_token="good-refresh",
        provider_user_id="1",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    db.add(account)
    db.commit()
    yield db, account
    db.close()
    engine.dispose()


def _vendor_answers(monkeypatch, status: int, payload: object) -> None:
    monkeypatch.setattr(
        tool_config,
        "_oauth_refresh_http_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status, json=payload)
            )
        ),
    )


def test_a_rejected_client_raises_logs_once_and_keeps_the_grant(
    grant, monkeypatch, caplog
):
    db, account = grant
    _vendor_answers(monkeypatch, 401, MICROSOFT_EXPIRED_SECRET)

    with caplog.at_level(logging.INFO, logger=tool_config.logger.name):
        with pytest.raises(tool_config._OAuthRefreshClientRejected) as rejected:
            asyncio.run(
                tool_config.refresh_oauth_token_if_needed(db, account, "microsoft")
            )

    assert rejected.value.provider == "microsoft"
    assert (account.access_token, account.refresh_token) == (
        "old-access",
        "good-refresh",
    )
    lines = [r for r in caplog.records if "client" in r.getMessage().lower()]
    assert [r.getMessage() for r in lines] == [
        "OAuth client rejected (provider=microsoft classification=client_rejected)"
    ]
    assert (lines[0].oauth_provider, lines[0].oauth_classification) == (
        "microsoft",
        "client_rejected",
    )
    assert "expired-secret" not in caplog.text and "good-refresh" not in caplog.text


def test_a_dead_grant_is_still_permanent(grant, monkeypatch):
    db, account = grant
    _vendor_answers(monkeypatch, 400, {"error": "invalid_grant"})

    assert (
        asyncio.run(tool_config.refresh_oauth_token_if_needed(db, account, "microsoft"))
        is False
    )


@pytest.mark.parametrize(
    ("reason", "admitted"),
    [
        ("client_rejected", True),
        ("client_rejected:microsoft", True),
        ("client_rejected:employment-hero", True),
        ("client_rejected:Microsoft", False),
        ("client_rejected:a b", False),
        ("client_rejected:", False),
    ],
)
def test_the_unavailable_reason_admits_client_rejected(reason, admitted):
    assert tool_config._is_mcp_unavailable_reason(reason) is admitted


def _mcp_client() -> SimpleNamespace:
    return SimpleNamespace(
        token_endpoint="https://mcp-auth.example.com/oauth/token",
        client_id="platform-client",
        client_secret="platform-secret",
        token_endpoint_auth_method="client_secret_post",
    )


async def _refresh_mcp(status: int, payload: object, *, preregistered: bool) -> str:
    async def no_url_policy(url, *, resolve_dns=False):
        return None

    original = mcp_oauth.validate_oauth_http_url
    mcp_oauth.validate_oauth_http_url = no_url_policy
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status, json=payload)
            )
        ) as client:
            await mcp_oauth._refresh_mcp_oauth_grant(
                _mcp_client(),
                refresh_token="good-refresh",
                resource="https://mcp.example.com",
                client=client,
                classify_client_rejection=preregistered,
            )
    except mcp_oauth.MCPOAuthRuntimeError as exc:
        return exc.code
    finally:
        mcp_oauth.validate_oauth_http_url = original
    return "ok"


@pytest.mark.parametrize(
    ("payload", "preregistered", "code"),
    [
        ({"error": "invalid_client"}, True, "client_rejected"),
        # A per-user registered client only affects that user.
        ({"error": "invalid_client"}, False, "token_refresh_failed"),
        ({"error": "invalid_grant"}, True, "token_refresh_failed"),
    ],
)
def test_an_mcp_refresh_rejects_only_a_preregistered_client(
    payload, preregistered, code
):
    assert asyncio.run(_refresh_mcp(400, payload, preregistered=preregistered)) == code
