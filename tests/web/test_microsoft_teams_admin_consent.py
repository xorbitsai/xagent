from __future__ import annotations

import html
import logging
import re
import time
from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api.auth import (
    create_access_token,
    generic_oauth_callback,
    verify_token,
)
from xagent.web.models.database import Base
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User

# Entra ID blocks a non-admin user outright when the requested delegated
# scopes include any that are classified as requiring org admin approval --
# the Teams connector requests TeamMember.Read.All and ChannelMessage.Read.All,
# among other scopes. The user never sees a consent screen at all; the redirect
# back to us carries error=access_denied&error_subcode=cancel with no `code`.
ADMIN_CONSENT_LINK_RE = re.compile(r'href="([^"]+)"')


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.add(
        PublicMCPApp(
            app_id="teams",
            name="Teams",
            description=(
                "Connect to Microsoft Teams to list teams, read channels and "
                "chats, and send messages."
            ),
            icon="https://www.google.com/s2/favicons?domain=teams.microsoft.com&sz=128",
            transport="oauth",
            provider_name="microsoft",
            category="Communication",
            oauth_scopes=[
                "Team.ReadBasic.All",
                "Channel.ReadBasic.All",
                "TeamMember.Read.All",
                "ChannelMessage.Read.All",
                "ChannelMessage.Send",
                "Chat.ReadWrite",
            ],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.teams"],
                "env_mapping": {"AUTH_TOKEN": "access_token"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _microsoft_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="microsoft",
        client_id=encrypt_value("microsoft-client-id"),
        client_secret=encrypt_value("microsoft-client-secret"),
        auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        redirect_uri="https://app.example.com/api/auth/microsoft/callback",
        userinfo_url="https://graph.microsoft.com/v1.0/me",
        user_id_path="id",
        email_path="userPrincipalName",
        default_scopes=["User.Read"],
    )


def _oauth_state(user, *, app_id: str = "teams") -> str:
    return create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "microsoft",
            "app_id": app_id,
            "redirect": None,
        },
        expires_delta=timedelta(minutes=10),
    )


def _extract_admin_consent_url(body: str) -> str:
    match = ADMIN_CONSENT_LINK_RE.search(body)
    assert match, f"no admin consent link found in response body: {body}"
    # The href attribute is HTML-escaped (`&` -> `&amp;`); undo that before
    # treating it as a URL, or parse_qs silently mis-splits every param
    # after the first.
    return html.unescape(match.group(1))


def test_teams_admin_consent_required_surfaces_a_forwardable_link(db_session):
    """A non-admin blocked by Entra ID must get an actionable link, not the
    generic "Error: access_denied" page every other provider/error still
    gets -- the whole point of this feature is that the user has no way to
    self-serve past this screen and needs to hand something to their IT
    admin."""
    db, user = db_session
    request = SimpleNamespace(
        query_params={
            "error": "access_denied",
            "error_subcode": "cancel",
            "state": _oauth_state(user),
        }
    )

    response = generic_oauth_callback("microsoft", request, db, _microsoft_provider())

    assert response.status_code == 400
    body = response.body.decode()
    assert "administrator" in body.lower()

    admin_consent_url = _extract_admin_consent_url(body)
    parsed = urlparse(admin_consent_url)
    assert parsed.netloc == "login.microsoftonline.com"
    assert parsed.path == "/organizations/v2.0/adminconsent"

    qs = parse_qs(parsed.query)
    assert qs["client_id"] == ["microsoft-client-id"]
    assert qs["redirect_uri"] == ["https://app.example.com/api/auth/microsoft/callback"]
    # The teams app's own scopes, merged with the provider's default_scopes
    # -- an admin approving a narrower/wrong scope set would leave ordinary
    # users blocked again on their very next connect attempt.
    assert qs["scope"] == [
        "User.Read Channel.ReadBasic.All ChannelMessage.Read.All ChannelMessage.Send "
        "Chat.ReadWrite Team.ReadBasic.All TeamMember.Read.All"
    ]

    state_payload = verify_token(qs["state"][0])
    assert state_payload["type"] == "admin_consent_state"
    assert state_payload["app_id"] == "teams"


def test_non_cancel_microsoft_error_keeps_the_generic_page(db_session):
    """Only the specific access_denied/cancel shape Entra ID uses for the
    admin-consent block should get the special page -- any other Microsoft
    error (e.g. the user's own invalid_grant on a later step) must not be
    reinterpreted as an admin-consent situation."""
    db, user = db_session
    request = SimpleNamespace(
        query_params={
            "error": "access_denied",
            "error_subcode": "something_else",
            "state": _oauth_state(user),
        }
    )

    response = generic_oauth_callback("microsoft", request, db, _microsoft_provider())

    assert response.status_code == 400
    assert "Error: access_denied" in response.body.decode()


def test_other_providers_are_not_affected(db_session):
    """The admin-consent branch is gated on provider == "microsoft" -- a
    same-shaped access_denied/cancel from an unrelated provider must not be
    misrouted into a Microsoft-specific admin consent link."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "slack",
            "app_id": "slack",
            "redirect": None,
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(
        query_params={
            "error": "access_denied",
            "error_subcode": "cancel",
            "state": state,
        }
    )

    response = generic_oauth_callback("slack", request, db, _microsoft_provider())

    assert response.status_code == 400
    assert "Error: access_denied" in response.body.decode()


def test_bare_microsoft_cancellation_keeps_the_generic_page(db_session):
    """error=access_denied&error_subcode=cancel is Entra ID's generic signal
    for a cancelled sign-in/consent step -- it fires just as much for a user
    simply declining an ordinary, self-consentable request as it does for a
    genuine admin-required block, and Microsoft does not distinguish the two
    in this redirect. A bare login (no app_id) only ever requests the
    provider's own default_scopes (User.Read), which is not in
    _MICROSOFT_ADMIN_CONSENT_SCOPES, so this must not be misread as
    "needs an admin" and must not hand an admin a request to grant broader,
    unrelated org-wide access nobody asked for."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "microsoft",
            "app_id": None,
            "redirect": None,
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(
        query_params={
            "error": "access_denied",
            "error_subcode": "cancel",
            "state": state,
        }
    )

    response = generic_oauth_callback("microsoft", request, db, _microsoft_provider())

    assert response.status_code == 400
    body = response.body.decode()
    assert "Error: access_denied" in body
    assert "adminconsent" not in body


def test_hidden_connector_blocks_the_admin_consent_link(db_session):
    """The admin-consent branch returns earlier in generic_oauth_callback
    than the existing hidden-app release gate (_reject_hidden_catalog_app),
    so it must apply that same gate itself -- otherwise a connector an
    admin just took offline could still have a tenant-wide admin-consent
    URL minted and shown for it, and Microsoft could receive (and an admin
    could approve) an org-wide grant request for an app Xagent has
    intentionally made unavailable."""
    db, user = db_session
    db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "teams").update(
        {"is_visible_in_connector": False}
    )
    db.commit()

    request = SimpleNamespace(
        query_params={
            "error": "access_denied",
            "error_subcode": "cancel",
            "state": _oauth_state(user),
        }
    )

    response = generic_oauth_callback("microsoft", request, db, _microsoft_provider())

    assert response.status_code == 404
    body = response.body.decode()
    assert "not currently available" in body
    assert "adminconsent" not in body


def test_admin_consent_return_trip_does_not_assert_a_verified_grant(db_session):
    """Entra ID's /adminconsent redirect carries admin_consent=True and the
    original `state`, never a `code` -- this must not fall into the ordinary
    "Missing code or state" branch.

    The `state` here is exactly what the admin-consent page displays and
    tells the user to forward -- so this same request (mint state, call the
    callback with admin_consent=True) is also what anyone holding that link
    could send directly, without ever visiting Microsoft. The response must
    therefore never claim a verified grant occurred (no "granted"/"approved"
    language), only that a response was received -- see the docstring on
    _handle_microsoft_admin_consent_return for why nothing here can prove a
    tenant grant actually happened."""
    db, _user = db_session
    state = create_access_token(
        data={"type": "admin_consent_state", "app_id": "teams"},
        expires_delta=timedelta(minutes=30),
    )
    request = SimpleNamespace(
        query_params={
            "admin_consent": "True",
            "state": state,
            "tenant": "11111111-2222-3333-4444-555555555555",
        }
    )

    response = generic_oauth_callback("microsoft", request, db, _microsoft_provider())

    assert response.status_code == 200
    body = response.body.decode()
    body_lower = body.lower()
    assert "Teams" in body
    # The old wording asserted the grant as settled fact; the fix must not
    # merely reword it while keeping the same affirmative claim.
    assert "<h1>admin consent granted</h1>" not in body_lower
    assert "your organization has approved" not in body_lower
    assert "no way to independently confirm" in body_lower


def test_admin_consent_return_trip_denied(db_session):
    db, _user = db_session
    state = create_access_token(
        data={"type": "admin_consent_state", "app_id": "teams"},
        expires_delta=timedelta(minutes=30),
    )
    request = SimpleNamespace(query_params={"admin_consent": "False", "state": state})

    response = generic_oauth_callback("microsoft", request, db, _microsoft_provider())

    assert response.status_code == 400
    assert "not" in response.body.decode().lower()


def test_admin_consent_return_trip_error_is_not_reported_as_success(db_session, caplog):
    """Microsoft documents admin-consent failures that still carry
    admin_consent=True, so the error parameter must take precedence."""
    db, _user = db_session
    state = create_access_token(
        data={"type": "admin_consent_state", "app_id": "teams"},
        expires_delta=timedelta(minutes=30),
    )
    request = SimpleNamespace(
        query_params={
            "admin_consent": "True",
            "error": "consent_required",
            "error_description": "The resource owner denied the request.",
            "state": state,
        }
    )

    with caplog.at_level(logging.WARNING, logger="xagent.web.api.auth"):
        response = generic_oauth_callback(
            "microsoft", request, db, _microsoft_provider()
        )

    assert response.status_code == 400
    body = response.body.decode().lower()
    assert "not granted" in body
    assert "consent granted" not in body
    assert "Microsoft admin consent callback failed" in caplog.text
    assert "consent_required" in caplog.text
    assert "The resource owner denied the request." in caplog.text


@pytest.mark.parametrize("state", [None, "tampered-state"])
def test_admin_consent_return_trip_rejects_invalid_state(db_session, state):
    db, _user = db_session
    query_params = {"admin_consent": "True"}
    if state is not None:
        query_params["state"] = state
    request = SimpleNamespace(query_params=query_params)

    response = generic_oauth_callback("microsoft", request, db, _microsoft_provider())

    assert response.status_code == 400
    body = response.body.decode().lower()
    assert "expired" in body
    # A stale/tampered state must not roll back or contradict a real
    # Microsoft-side grant that may already have happened -- the page must
    # not claim the approval itself failed, only that Xagent's own link did.
    assert "consent granted" not in body
    assert "still stands" in body


def test_admin_consent_state_lifetime_suits_asynchronous_admin_approval(db_session):
    """The handoff link is meant to be forwarded to an org admin who may act
    on it hours later (an IT approval queue), not the 10-minute lifetime an
    ordinary oauth_state uses for a same-session redirect round trip."""
    db, user = db_session
    request = SimpleNamespace(
        query_params={
            "error": "access_denied",
            "error_subcode": "cancel",
            "state": _oauth_state(user),
        }
    )

    response = generic_oauth_callback("microsoft", request, db, _microsoft_provider())
    admin_consent_url = _extract_admin_consent_url(response.body.decode())
    state_token = parse_qs(urlparse(admin_consent_url).query)["state"][0]

    payload = verify_token(state_token)
    minted_lifetime = payload["exp"] - int(time.time())
    assert minted_lifetime > timedelta(hours=1).total_seconds()
