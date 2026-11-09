"""The token request each builtin OAuth provider's refresh sends (#3580).

Captured on the wire through ``httpx.MockTransport``, so the expectation is
what the vendor receives, not how the request is assembled. The client
verification command must send exactly this shape with an invalid token.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.web.api.auth  # noqa: F401  (see test_oauth_refresh_retry.py)
from xagent.core.utils.encryption import encrypt_value
from xagent.web import oauth_account_hosts
from xagent.web.builtin_mcp_registry import get_builtin_oauth_provider_rows
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.tools import config as tool_config

CLIENT_ID = "test-client-id"
CLIENT_SECRET = "test-client-secret"
REFRESH_TOKEN = "test-refresh-token"
ACCESS_TOKEN = "test-access-token"
ZENDESK_MARKETPLACE_ENV = {
    "ZENDESK_MARKETPLACE_NAME": "Xagent",
    "ZENDESK_MARKETPLACE_ORGANIZATION_ID": "12345",
    "ZENDESK_MARKETPLACE_APP_ID": "67890",
}
# Where a provider's grant records the account its refresh goes to.
INSTANCE_URLS = {"zendesk": "acme", "deputy": "https://acme.as.deputy.com"}


@pytest.fixture()
def db(tmp_path, monkeypatch):
    for name, value in ZENDESK_MARKETPLACE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        oauth_account_hosts.socket,
        "getaddrinfo",
        lambda host, port, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    engine = create_engine(
        f"sqlite:///{tmp_path / 'shapes.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    for row in get_builtin_oauth_provider_rows():
        session.add(
            OAuthProvider(
                **{
                    **row,
                    "client_id": encrypt_value(CLIENT_ID),
                    "client_secret": encrypt_value(CLIENT_SECRET),
                    "redirect_uri": f"https://app.example.com/api/auth/"
                    f"{row['provider_name']}/callback",
                }
            )
        )
    session.add(User(username="alice", password_hash="x", is_admin=False))
    session.commit()
    yield session
    session.close()
    engine.dispose()


def _captured_refresh(db, monkeypatch, provider: str) -> httpx.Request:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})

    monkeypatch.setattr(
        tool_config,
        "_oauth_refresh_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    user = db.query(User).one()
    grant = UserOAuth(
        user_id=user.id,
        provider=provider,
        access_token=ACCESS_TOKEN,
        refresh_token=REFRESH_TOKEN,
        instance_url=INSTANCE_URLS.get(provider),
        provider_user_id="1",
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    db.add(grant)
    db.commit()
    assert asyncio.run(tool_config.refresh_oauth_token_if_needed(db, grant, provider))
    (request,) = captured
    return request


def _shape(request: httpx.Request) -> dict:
    url = urlsplit(str(request.url))
    content_type = request.headers.get("content-type", "")
    raw = request.content.decode()
    if content_type.startswith("application/json"):
        body = json.loads(raw)
    else:
        body = {key: values[0] for key, values in parse_qs(raw).items()}
    shape = {
        "method": request.method,
        "url": f"{url.scheme}://{url.netloc}{url.path}",
        "query": {key: values[0] for key, values in parse_qs(url.query).items()},
        "body": body,
        "content_type": content_type or None,
        # httpx's default */* is left out, so only a header the code sets shows.
        "accept": accept
        if (accept := request.headers.get("accept")) != "*/*"
        else None,
    }
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, value = authorization.partition(" ")
        shape["basic"] = (
            base64.b64decode(value).decode() if scheme == "Basic" else authorization
        )
    extra = {
        key: value
        for key, value in request.headers.items()
        if key.lower().startswith("x-zendesk")
    }
    if extra:
        shape["headers"] = extra
    return shape


EXPECTED_SHAPES: dict[str, dict] = {
    "google": {
        "method": "POST",
        "url": "https://oauth2.googleapis.com/token",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
    "linkedin": {
        "method": "POST",
        "url": "https://www.linkedin.com/oauth/v2/accessToken",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
    "microsoft": {
        "method": "POST",
        "url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
    "meta": {
        "method": "GET",
        "url": "https://graph.facebook.com/v25.0/oauth/access_token",
        "query": {
            "grant_type": "fb_exchange_token",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "fb_exchange_token": ACCESS_TOKEN,
        },
        "body": {},
        "content_type": None,
        "accept": None,
    },
    "slack": {
        "method": "POST",
        "url": "https://slack.com/api/oauth.v2.access",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
    "zoom": {
        "method": "POST",
        "url": "https://zoom.us/oauth/token",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
        "basic": f"{CLIENT_ID}:{CLIENT_SECRET}",
    },
    "github": {
        "method": "POST",
        "url": "https://github.com/login/oauth/access_token",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": "application/json",
    },
    "intercom": {
        "method": "POST",
        "url": "https://api.intercom.io/auth/eagle/token",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
    "zendesk": {
        "method": "POST",
        "url": "https://acme.zendesk.com/oauth/tokens",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "expires_in": "172800",
            "refresh_token_expires_in": "7776000",
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
        "headers": {
            "x-zendesk-marketplace-name": "Xagent",
            "x-zendesk-marketplace-organization-id": "12345",
            "x-zendesk-marketplace-app-id": "67890",
        },
    },
    "salesforce": {
        "method": "POST",
        "url": "https://login.salesforce.com/services/oauth2/token",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
    "deputy": {
        "method": "POST",
        "url": "https://acme.as.deputy.com/oauth/access_token",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": "https://app.example.com/api/auth/deputy/callback",
            "scope": "longlife_refresh_token",
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
    "linear": {
        "method": "POST",
        "url": "https://api.linear.app/oauth/token",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
    "jira": {
        "method": "POST",
        "url": "https://auth.atlassian.com/oauth/token",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/json",
        "accept": None,
    },
    "xero": {
        "method": "POST",
        "url": "https://identity.xero.com/connect/token",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
    "myob": {
        "method": "POST",
        "url": "https://secure.myob.com/oauth2/v1/authorize/",
        "query": {},
        "body": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
    "employment-hero": {
        "method": "POST",
        "url": "https://oauth.employmenthero.com/oauth2/token",
        "query": {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
        },
        "body": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        "content_type": "application/x-www-form-urlencoded",
        "accept": None,
    },
}


def test_every_builtin_provider_has_a_pinned_shape():
    providers = {str(row["provider_name"]) for row in get_builtin_oauth_provider_rows()}
    assert providers == set(EXPECTED_SHAPES)


@pytest.mark.parametrize("provider", sorted(EXPECTED_SHAPES))
def test_refresh_sends_the_pinned_request(db, monkeypatch, provider):
    assert (
        _shape(_captured_refresh(db, monkeypatch, provider))
        == EXPECTED_SHAPES[provider]
    )


@pytest.mark.parametrize("provider", sorted(EXPECTED_SHAPES))
def test_refresh_sends_with_a_ten_second_timeout(db, monkeypatch, provider):
    request = _captured_refresh(db, monkeypatch, provider)

    assert request.extensions["timeout"] == {
        "connect": 10.0,
        "read": 10.0,
        "write": 10.0,
        "pool": 10.0,
    }


def test_an_account_host_refresh_does_not_follow_a_redirect(db, monkeypatch):
    """A redirect would skip the account host's private-network check."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/"})

    monkeypatch.setattr(
        tool_config,
        "_oauth_refresh_http_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ),
    )
    user = db.query(User).one()
    grant = UserOAuth(
        user_id=user.id,
        provider="zendesk",
        access_token=ACCESS_TOKEN,
        refresh_token=REFRESH_TOKEN,
        instance_url=INSTANCE_URLS["zendesk"],
        provider_user_id="1",
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    db.add(grant)
    db.commit()

    refreshed = asyncio.run(
        tool_config.refresh_oauth_token_if_needed(db, grant, "zendesk")
    )

    assert refreshed is False
    assert len(sent) == 1
