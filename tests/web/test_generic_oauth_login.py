"""Regression tests for generic_oauth_login URL construction.

Covers two bugs the PR fixed:
  1. When auth_url already contains a query string, params must be appended
     with '&' (no second '?').
  2. Zoom provider must include `prompt=login` in the redirect URL.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api.auth import (
    _resolve_oauth_secret,
    create_access_token,
    generic_oauth_login,
)
from xagent.web.models.database import Base
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User

# ---------- helpers ---------------------------------------------------------


@pytest.fixture()
def db_session(tmp_path):
    """Fresh SQLite DB + a single user for each test."""
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _token_for(user: User) -> str:
    return create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )


def _provider(
    auth_url: str,
    default_scopes=None,
    redirect_uri=None,
    client_id: str = "test-client-id",
):
    """A duck-typed stand-in for the OAuthProvider ORM row."""
    return SimpleNamespace(
        client_id=encrypt_value(client_id),
        auth_url=auth_url,
        redirect_uri=redirect_uri,
        default_scopes=default_scopes or [],
    )


def _location(response) -> str:
    # RedirectResponse stores the target in the Location header.
    return response.headers["location"]


# ---------- the actual regression checks ------------------------------------


def test_auth_url_with_query_uses_ampersand_separator(db_session):
    """If db_provider.auth_url already has '?', params must be appended with '&'."""
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://example.com/oauth/authorize?tenant=acme",
        default_scopes=["openid", "profile"],
        redirect_uri="https://app.example.com/cb",
    )

    resp = generic_oauth_login(
        provider="custom",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    url = _location(resp)

    # Only one '?' allowed in the whole URL — this is the regression.
    assert url.count("?") == 1, f"second '?' leaked into URL: {url}"

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    qs = parse_qs(parsed.query)

    assert base == "https://example.com/oauth/authorize"
    assert qs["tenant"] == ["acme"], "pre-existing query param dropped"
    assert qs["client_id"] == ["test-client-id"]
    assert qs["redirect_uri"] == ["https://app.example.com/cb"]
    assert qs["response_type"] == ["code"]
    assert "state" in qs


def test_auth_url_without_query_uses_question_mark(db_session):
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://example.com/oauth/authorize",
        default_scopes=["openid"],
        redirect_uri="https://app.example.com/cb",
    )

    resp = generic_oauth_login(
        provider="custom",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    url = _location(resp)

    assert url.count("?") == 1
    assert url.startswith("https://example.com/oauth/authorize?")


def test_zoom_provider_sets_prompt_login(db_session):
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://zoom.us/oauth/authorize",
        default_scopes=["user:read"],
        redirect_uri="https://app.example.com/cb",
    )

    resp = generic_oauth_login(
        provider="zoom",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    url = _location(resp)
    qs = parse_qs(urlparse(url).query)

    assert qs.get("prompt") == ["login"], f"zoom prompt missing: {url}"


def test_non_zoom_provider_does_not_set_prompt_login(db_session):
    """Sanity: only Zoom gets prompt=login (Google gets prompt=consent, others none)."""
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://example.com/oauth/authorize",
        default_scopes=["openid"],
        redirect_uri="https://app.example.com/cb",
    )

    resp = generic_oauth_login(
        provider="custom",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)
    assert "prompt" not in qs


def test_meta_login_uses_comma_separated_canonical_scopes_for_builtin_app(
    db_session, monkeypatch
):
    db, user = db_session
    token = _token_for(user)
    monkeypatch.delenv("META_CONFIG_ID", raising=False)
    monkeypatch.delenv("META_LOGIN_CONFIG_ID", raising=False)
    db.add(
        PublicMCPApp(
            app_id="facebook",
            name="Facebook Pages",
            description="Facebook connector",
            transport="oauth",
            provider_name="meta",
            category="Marketing",
            oauth_scopes=["pages_show_list", "pages_manage_posts"],
            is_visible_in_connector=True,
            launch_config={},
        )
    )
    db.commit()

    provider = _provider(
        auth_url="https://www.facebook.com/v25.0/dialog/oauth",
        default_scopes=["public_profile"],
        redirect_uri="https://app.example.com/api/auth/meta/callback",
    )

    resp = generic_oauth_login(
        provider="meta",
        token=token,
        app_id="facebook",
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)

    assert qs["scope"] == [
        "public_profile,pages_manage_posts,pages_read_engagement,"
        "pages_read_user_content,pages_show_list"
    ]


def test_github_login_requests_exact_canonical_scope(db_session):
    """The requested scope must be exactly the provider's default_scopes
    ("read:user") merged with the github app row's canonical oauth_scopes
    ("repo", "user:email", sorted) -- read:org must NOT reappear even if a
    stale/incorrect DB row still lists it, since get_app_by_id overlays the
    canonical registry's oauth_scopes for a builtin app_id regardless of
    what is persisted."""
    db, user = db_session
    token = _token_for(user)
    db.add(
        PublicMCPApp(
            app_id="github",
            name="GitHub",
            description="GitHub connector",
            transport="oauth",
            provider_name="github",
            category="Development",
            # Deliberately stale/wrong to prove the registry, not this
            # row's oauth_scopes, is what actually gets requested.
            oauth_scopes=["repo", "read:org", "user:email"],
            is_visible_in_connector=True,
            launch_config={},
        )
    )
    db.commit()

    provider = _provider(
        auth_url="https://github.com/login/oauth/authorize",
        default_scopes=["read:user"],
        redirect_uri="https://app.example.com/api/auth/github/callback",
    )

    resp = generic_oauth_login(
        provider="github",
        token=token,
        app_id="github",
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)

    assert qs["scope"] == ["read:user repo user:email"]


def test_bare_github_login_config_error_takes_precedence_over_bare_route_guard(
    db_session,
):
    """A bare (app_id-less) login persists to the exact same
    UserOAuth.provider="github" key an app-scoped login uses, since
    github's app_id and provider name are the same string. Left unblocked,
    re-running this route would silently replace a fully-scoped
    connection's grant with an identity-only one on the next callback --
    that guard exists and still runs before any state token is minted.

    But the guard is checked AFTER config/auth resolution, not before: a
    misconfigured provider (missing client_id) must still get the
    actionable CLIENT_ID config error, not a 404 that would incorrectly
    read as "this route doesn't support a normal connect attempt" to an
    operator debugging their setup."""
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://github.com/login/oauth/authorize",
        default_scopes=["read:user"],
        redirect_uri="https://app.example.com/api/auth/github/callback",
        client_id="",  # deliberately unconfigured
    )

    resp = generic_oauth_login(
        provider="github",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )

    assert resp.status_code == 500
    assert "GITHUB_CLIENT_ID" in resp.body.decode()


def test_bare_github_login_unauthenticated_gets_401_not_404(db_session):
    """Same ordering point as the config-error case above: an
    unauthenticated bare-route request must get the generic 401, not a 404
    that implies the route itself is unsupported."""
    db, _user = db_session

    provider = _provider(
        auth_url="https://github.com/login/oauth/authorize",
        default_scopes=["read:user"],
        redirect_uri="https://app.example.com/api/auth/github/callback",
    )

    resp = generic_oauth_login(
        provider="github",
        token=None,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )

    assert resp.status_code == 401


def test_bare_github_login_is_rejected_once_authenticated_and_configured(db_session):
    """Once config and auth both resolve cleanly, the bare-route guard for
    providers requiring an app-scoped grant still fires -- before any
    state token is minted, which is the property it actually needs to
    hold."""
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://github.com/login/oauth/authorize",
        default_scopes=["read:user"],
        redirect_uri="https://app.example.com/api/auth/github/callback",
    )

    resp = generic_oauth_login(
        provider="github",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )

    assert resp.status_code == 404


def test_bare_login_for_unrestricted_provider_still_proceeds(db_session):
    """Sanity: the bare-route guard is scoped to
    APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT members only -- an ordinary
    provider's bare login (no collision risk) must be unaffected."""
    db, user = db_session
    token = _token_for(user)

    provider = _provider(
        auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        default_scopes=["User.Read"],
        redirect_uri="https://app.example.com/cb",
    )

    resp = generic_oauth_login(
        provider="microsoft",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )

    assert resp.status_code == 307


def test_hubspot_login_sends_tier_gated_scopes_as_optional(db_session):
    """business-intelligence, marketing-email, and marketing.campaigns.read
    are all gated on a Marketing Hub tier above Free/CRM-only - requesting
    any of them as required scopes would block the whole authorization for
    portals below that tier. They must arrive via optional_scope, not
    merged into the required scope param."""
    db, user = db_session
    token = _token_for(user)
    db.add(
        PublicMCPApp(
            app_id="hubspot",
            name="HubSpot",
            description="HubSpot connector",
            transport="oauth",
            provider_name="hubspot",
            category="CRM",
            oauth_scopes=["crm.objects.contacts.read"],
            is_visible_in_connector=True,
            launch_config={},
        )
    )
    db.commit()

    provider = _provider(
        auth_url="https://app.hubspot.com/oauth/authorize",
        default_scopes=["oauth"],
        redirect_uri="https://app.example.com/api/auth/hubspot/callback",
    )

    resp = generic_oauth_login(
        provider="hubspot",
        token=token,
        app_id="hubspot",
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)

    assert "business-intelligence" not in qs["scope"][0]
    assert "marketing-email" not in qs["scope"][0]
    assert "marketing.campaigns.read" not in qs["scope"][0]
    assert qs["optional_scope"] == [
        "business-intelligence marketing-email marketing.campaigns.read"
    ]


def test_non_hubspot_app_sends_no_optional_scope_param(db_session, monkeypatch):
    """optional_oauth_scopes is a HubSpot-specific registry field today; a
    builtin app that doesn't set it must not get a stray optional_scope
    param on a provider whose authorize endpoint doesn't expect one."""
    db, user = db_session
    token = _token_for(user)
    monkeypatch.delenv("META_CONFIG_ID", raising=False)
    monkeypatch.delenv("META_LOGIN_CONFIG_ID", raising=False)
    db.add(
        PublicMCPApp(
            app_id="facebook",
            name="Facebook Pages",
            description="Facebook connector",
            transport="oauth",
            provider_name="meta",
            category="Marketing",
            oauth_scopes=["pages_show_list"],
            is_visible_in_connector=True,
            launch_config={},
        )
    )
    db.commit()

    provider = _provider(
        auth_url="https://www.facebook.com/v25.0/dialog/oauth",
        default_scopes=["public_profile"],
        redirect_uri="https://app.example.com/api/auth/meta/callback",
    )

    resp = generic_oauth_login(
        provider="meta",
        token=token,
        app_id="facebook",
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)

    assert "optional_scope" not in qs


def test_meta_login_uses_config_id_without_scope_when_configured(
    db_session, monkeypatch
):
    db, user = db_session
    token = _token_for(user)
    monkeypatch.setenv("META_CONFIG_ID", "1234567890")
    db.add(
        PublicMCPApp(
            app_id="instagram",
            name="Instagram",
            description="Instagram connector",
            transport="oauth",
            provider_name="meta",
            category="Marketing",
            oauth_scopes=["instagram_basic", "instagram_content_publish"],
            is_visible_in_connector=True,
            launch_config={},
        )
    )
    db.commit()

    provider = _provider(
        auth_url="https://www.facebook.com/v25.0/dialog/oauth",
        default_scopes=["public_profile"],
        redirect_uri="https://app.example.com/api/auth/meta/callback",
    )

    resp = generic_oauth_login(
        provider="meta",
        token=token,
        app_id="instagram",
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)

    assert qs["config_id"] == ["1234567890"]
    assert "scope" not in qs
    assert "optional_scope" not in qs


def test_meta_login_suppresses_optional_scope_under_config_id(db_session, monkeypatch):
    """config_id mode discards the registry's scope list entirely (the Meta
    Login Configuration is the sole source of truth) - optional_scope must
    be suppressed the same way scope already is, even for an app that DOES
    declare optional_oauth_scopes. No builtin meta-provider app declares
    optional_oauth_scopes today, so the registry lookup is patched to
    simulate one rather than asserting a coincidence of current data."""
    import xagent.web.mcp_apps as mcp_apps_module
    from xagent.web.builtin_mcp_registry import (
        get_builtin_execution_fields_and_optional_scopes,
    )

    def fake_lookup(app_id):
        execution_fields, _ = get_builtin_execution_fields_and_optional_scopes(app_id)
        optional_scopes = ["some_tier_gated_scope"] if app_id == "facebook" else []
        return execution_fields, optional_scopes

    db, user = db_session
    token = _token_for(user)
    monkeypatch.setenv("META_CONFIG_ID", "1234567890")
    monkeypatch.setattr(
        mcp_apps_module,
        "get_builtin_execution_fields_and_optional_scopes",
        fake_lookup,
    )
    db.add(
        PublicMCPApp(
            app_id="facebook",
            name="Facebook Pages",
            description="Facebook connector",
            transport="oauth",
            provider_name="meta",
            category="Marketing",
            oauth_scopes=["pages_show_list"],
            is_visible_in_connector=True,
            launch_config={},
        )
    )
    db.commit()

    provider = _provider(
        auth_url="https://www.facebook.com/v25.0/dialog/oauth",
        default_scopes=["public_profile"],
        redirect_uri="https://app.example.com/api/auth/meta/callback",
    )

    resp = generic_oauth_login(
        provider="meta",
        token=token,
        app_id="facebook",
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)

    assert qs["config_id"] == ["1234567890"]
    assert "scope" not in qs
    assert "optional_scope" not in qs


def test_meta_login_uses_config_id_without_scope_for_facebook(db_session, monkeypatch):
    """Facebook Pages via config_id: pages_read_user_content is NOT requested.

    In this mode the Login Configuration in the Meta App Dashboard is the sole
    source of truth for granted permissions (see META_CONFIG_ID in example.env)
    - our builtin registry scopes are not sent at all. This test documents that
    gap so a future change to the request-scope path doesn't silently mask it.
    """
    db, user = db_session
    token = _token_for(user)
    monkeypatch.setenv("META_CONFIG_ID", "1234567890")
    db.add(
        PublicMCPApp(
            app_id="facebook",
            name="Facebook Pages",
            description="Facebook connector",
            transport="oauth",
            provider_name="meta",
            category="Marketing",
            oauth_scopes=[
                "pages_show_list",
                "pages_read_engagement",
                "pages_manage_posts",
                "pages_read_user_content",
            ],
            is_visible_in_connector=True,
            launch_config={},
        )
    )
    db.commit()

    provider = _provider(
        auth_url="https://www.facebook.com/v25.0/dialog/oauth",
        default_scopes=["public_profile"],
        redirect_uri="https://app.example.com/api/auth/meta/callback",
    )

    resp = generic_oauth_login(
        provider="meta",
        token=token,
        app_id="facebook",
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)

    assert qs["config_id"] == ["1234567890"]
    assert "scope" not in qs
    assert "optional_scope" not in qs


def test_meta_login_ignores_undocumented_legacy_config_id_alias(
    db_session, monkeypatch
):
    db, user = db_session
    token = _token_for(user)
    monkeypatch.delenv("META_CONFIG_ID", raising=False)
    monkeypatch.setenv("META_LOGIN_CONFIG_ID", "legacy-config-id")

    provider = _provider(
        auth_url="https://www.facebook.com/v25.0/dialog/oauth",
        default_scopes=["public_profile"],
        redirect_uri="https://app.example.com/api/auth/meta/callback",
    )

    resp = generic_oauth_login(
        provider="meta",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)

    assert "config_id" not in qs
    assert qs["scope"] == ["public_profile"]


def test_login_uses_env_client_id_when_provider_client_id_is_empty(
    db_session, monkeypatch
):
    db, user = db_session
    token = _token_for(user)
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", "env-client-id")

    provider = _provider(
        auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        default_scopes=["User.Read"],
        redirect_uri="https://app.example.com/cb",
        client_id="",
    )

    resp = generic_oauth_login(
        provider="microsoft",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )
    qs = parse_qs(urlparse(_location(resp)).query)

    assert qs["client_id"] == ["env-client-id"]


@pytest.mark.parametrize("encrypted_value", [None, ""])
def test_resolve_oauth_secret_uses_env_without_decrypting_empty_values(
    encrypted_value, monkeypatch
):
    from xagent.core.utils import encryption

    def fail_on_empty_decrypt(value: str) -> str:
        if not value:
            raise ValueError("empty values should not be decrypted")
        return "db-secret"

    monkeypatch.setattr(encryption, "decrypt_value", fail_on_empty_decrypt)
    monkeypatch.setenv("MICROSOFT_CLIENT_SECRET", "env-secret")

    assert (
        _resolve_oauth_secret("microsoft", encrypted_value, "CLIENT_SECRET")
        == "env-secret"
    )


def test_login_fails_locally_when_client_id_is_missing(db_session, monkeypatch):
    db, user = db_session
    token = _token_for(user)
    monkeypatch.delenv("MICROSOFT_CLIENT_ID", raising=False)

    provider = _provider(
        auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        default_scopes=["User.Read"],
        redirect_uri="https://app.example.com/cb",
        client_id="",
    )

    resp = generic_oauth_login(
        provider="microsoft",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )

    assert resp.status_code == 500
    assert "MICROSOFT_CLIENT_ID" in resp.body.decode()
