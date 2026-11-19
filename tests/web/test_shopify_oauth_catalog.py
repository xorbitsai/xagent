"""The Shopify catalog row and provider row that put the OAuth flow of
test_shopify_oauth.py into the product (#2810): the real registry rows, not a
stand-in, from login to the connector's launch environment.
"""

import socket
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.shared.oauth_browser_fixture import create_access_token
from xagent.web import oauth_account_hosts
from xagent.web.api.auth import generic_oauth_login
from xagent.web.builtin_mcp_registry import (
    get_builtin_oauth_provider_rows,
    get_builtin_public_mcp_app,
)
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.shopify_oauth import SHOPIFY_REQUIRED_SCOPES
from xagent.web.tools.config import WebToolConfig

PUBLIC_ADDRESS = "23.227.38.65"


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    def resolve(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_ADDRESS, port))]

    monkeypatch.setattr(oauth_account_hosts.socket, "getaddrinfo", resolve)


def _registry_provider() -> dict:
    return next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "shopify"
    )


def _registry_app() -> dict:
    row = get_builtin_public_mcp_app("shopify")
    assert row is not None
    return row


@pytest.fixture()
def shopify_db(tmp_path, monkeypatch):
    """The registry's own rows; client credentials come from env, as they
    would on a deployment that only sets SHOPIFY_CLIENT_*."""
    monkeypatch.setenv("SHOPIFY_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("SHOPIFY_CLIENT_SECRET", "env-client-secret")
    monkeypatch.setenv(
        "SHOPIFY_REDIRECT_URI", "https://xagent.example/api/auth/shopify/callback"
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'shopify-catalog.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="alice", password_hash="hash")
        db.add(user)
        db.add(PublicMCPApp(**_registry_app()))
        db.add(OAuthProvider(**_registry_provider()))
        db.commit()
        db.refresh(user)
        yield db, user
    engine.dispose()


def _show(db) -> None:
    app = db.query(PublicMCPApp).filter_by(app_id="shopify").one()
    app.is_visible_in_connector = True
    db.commit()


def _login(db, user, account: str = "acme"):
    token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "session_epoch": user.session_epoch,
            "type": "access",
        },
        expires_delta=timedelta(minutes=5),
    )
    return generic_oauth_login(
        "shopify",
        token=token,
        app_id="shopify",
        redirect=None,
        db=db,
        db_provider=db.query(OAuthProvider).filter_by(provider_name="shopify").one(),
        account=account,
    )


# ---------- registry rows ----------------------------------------------------


def test_catalog_row_connects_through_shopify_oauth():
    row = _registry_app()

    assert row["transport"] == "oauth"
    assert row["provider_name"] == "shopify"
    # Exactly what the callback requires (shopify_oauth); read_all_orders is
    # not requested by default.
    assert set(row["oauth_scopes"]) == SHOPIFY_REQUIRED_SCOPES
    launch = row["launch_config"]
    assert launch["args"] == ["-m", "xagent.web.tools.mcp.shopify"]
    assert launch["env_mapping"] == {
        "SHOPIFY_ACCESS_TOKEN": "access_token",
        "SHOPIFY_STORE_DOMAIN": "instance_url",
    }
    # Nothing left that asks the user to paste a token.
    assert "required_env" not in launch
    assert "token" not in row["description"].lower()
    assert launch["builtin_provenance"] == {
        "registry": "xagent",
        "app_id": "shopify",
        "version": 1,
    }


def test_catalog_row_ships_hidden():
    """Shown only after a development-store check, by an operator."""
    assert _registry_app()["is_visible_in_connector"] is False


def test_provider_row_reads_credentials_and_hosts_at_request_time():
    row = _registry_provider()

    for field in (
        "client_id",
        "client_secret",
        "auth_url",
        "token_url",
        "redirect_uri",
        "userinfo_url",
    ):
        assert row[field] == ""
    assert row["default_scopes"] == []


# ---------- connect ----------------------------------------------------------


def test_hidden_catalog_row_cannot_start_a_connection(shopify_db):
    db, user = shopify_db

    response = _login(db, user)

    assert response.status_code == 404
    assert "location" not in response.headers


def test_shown_catalog_row_sends_the_merchant_to_the_stores_consent_page(
    shopify_db,
):
    db, user = shopify_db
    _show(db)

    response = _login(db, user)

    location = urlparse(response.headers["location"])
    assert (location.netloc, location.path) == (
        "acme.myshopify.com",
        "/admin/oauth/authorize",
    )
    query = parse_qs(location.query)
    assert query["client_id"] == ["env-client-id"]
    assert query["redirect_uri"] == ["https://xagent.example/api/auth/shopify/callback"]
    assert set(query["scope"][0].split(",")) == SHOPIFY_REQUIRED_SCOPES


@pytest.mark.asyncio
async def test_connector_gets_the_token_and_the_grants_store(shopify_db):
    db, user = shopify_db
    _show(db)
    server = MCPServer(
        name="Shopify",
        description="Shopify",
        managed="external",
        transport="oauth",
        auth={
            "app_id": "shopify",
            "provider": "shopify",
            "builtin_provenance": _registry_app()["launch_config"][
                "builtin_provenance"
            ],
        },
    )
    db.add(server)
    db.commit()
    db.add(
        UserMCPServer(
            user_id=user.id, mcpserver_id=server.id, is_owner=True, is_active=True
        )
    )
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="shopify",
            access_token="shpat-live",
            refresh_token="shprt-live",
            instance_url="acme",
            provider_user_id="gid://shopify/Shop/1",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
    )
    db.commit()

    configs = await WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=user.id,
        workspace_config={"base_dir": "/tmp", "task_id": "task-1"},
    ).get_mcp_server_configs()

    shopify = next(config for config in configs if config["name"] == "Shopify")
    assert shopify["config"]["args"] == ["-m", "xagent.web.tools.mcp.shopify"]
    env = shopify["config"]["env"]
    assert env["SHOPIFY_ACCESS_TOKEN"] == "shpat-live"
    assert env["SHOPIFY_STORE_DOMAIN"] == "acme"
