"""Actor OAuth callback pages tell the member what to do next.

When an actor flow's callback carries a provider error, fails an actor check,
or comes back after its sign-in window was used or expired, the page must be
actionable (what happened, what to do, a way to close the window) and must
keep the single-use flow semantics. Every callback that does not connect
before the token exchange logs a reason code without any value of the flow
itself.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.oauth_browser_fixture import (
    create_access_token,
    generic_oauth_callback,
)
from xagent.core.utils.encryption import encrypt_value
from xagent.web import jwt_codec as jwt
from xagent.web import mcp_apps
from xagent.web.api import auth as auth_api
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth

OWNER = "actor:workspace-41:member-alice"
OLD_ACTOR_ERROR = "Invalid or expired actor OAuth flow"
CLOSE_BUTTON = '<button type="button" onclick="window.close()">'
AUTH_LOGGER = "xagent.web.api.auth"
CALLBACK_LOG_PREFIX = "OAuth callback did not connect"
APPS = {
    "outlook": {
        "name": "Outlook",
        "transport": "oauth",
        "provider_name": "microsoft",
        "oauth_scopes": ["Mail.ReadWrite"],
        "launch_config": {"command": "outlook"},
    },
    "calendar": {
        "name": "Calendar",
        "transport": "oauth",
        "provider_name": "custom",
        "oauth_scopes": [],
        "launch_config": {"command": "calendar"},
    },
}
DECLINED_DESCRIPTION = (
    "AADSTS65004: User declined to consent to access the app.\r\n"
    "Trace ID: 0b8c2f3e-0000-0000-0000-000000000000\r\n"
    "Correlation ID: 5d1e9a7c-0000-0000-0000-000000000000"
)
DECLINED_URI = "https://login.microsoftonline.com/error?code=65004"
REJECTED_TITLE = "This sign-in window can't finish connecting"
# What a failed actor check says happened, and what to do next.
OTHER_BROWSER_PAGE = (
    "This sign-in could not be matched to this browser, so the app was not "
    "connected. This can happen when it was started in a different browser or "
    "browser profile.",
    "Close this window and select Connect again from the same browser.",
)
CONNECTION_CHANGED_PAGE = (
    "The connection this sign-in was started for was turned off, removed or "
    "changed while you were signing in, so the app was not connected.",
    "Close this window. If the app is still available to you, select Connect "
    "again. Otherwise, contact your administrator.",
)
UNVERIFIED_PAGE = (
    "This sign-in could not be verified, so the app was not connected.",
    "Close this window and select Connect again. If it keeps happening, "
    "contact your administrator.",
)


class _ProviderResponse:
    def __init__(self, data: dict[str, object], status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._data


@pytest.fixture
def oauth_db(tmp_path, monkeypatch):
    registry_lookup = mcp_apps.get_builtin_execution_fields_and_optional_scopes

    def test_registry(app_id: str):
        if app_id in APPS:
            return APPS[app_id], []
        return registry_lookup(app_id)

    monkeypatch.setattr(
        mcp_apps, "get_builtin_execution_fields_and_optional_scopes", test_registry
    )

    engine = create_engine(f"sqlite:///{tmp_path / 'actor-oauth-feedback.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        db.add(user)
        db.flush()
        for app_id, execution in APPS.items():
            db.add(
                PublicMCPApp(
                    app_id=app_id,
                    name=execution["name"],
                    description=execution["name"],
                    transport="oauth",
                    provider_name=execution["provider_name"],
                    oauth_scopes=list(execution["oauth_scopes"]),
                    launch_config=dict(execution["launch_config"]),
                    is_visible_in_connector=True,
                )
            )
            server = MCPServer(
                name=execution["name"],
                description=execution["name"],
                managed="external",
                transport="oauth",
                auth={"app_id": app_id, "provider": execution["provider_name"]},
            )
            db.add(server)
            db.flush()
            db.add(
                UserMCPServer(
                    user_id=int(user.id),
                    mcpserver_id=int(server.id),
                    is_owner=False,
                    is_active=True,
                )
            )
        db.commit()
        db.refresh(user)
        yield db, user
    engine.dispose()


@pytest.fixture
def token_endpoint(monkeypatch) -> Mock:
    post = Mock(
        return_value=_ProviderResponse(
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=_ProviderResponse({"id": "account", "mail": "a@x.test"})),
    )
    return post


@pytest.fixture
def callback_logs(caplog):
    caplog.set_level(logging.WARNING, logger=AUTH_LOGGER)

    def messages() -> list[str]:
        return [
            record.getMessage()
            for record in caplog.records
            if record.name == AUTH_LOGGER
            and record.getMessage().startswith(CALLBACK_LOG_PREFIX)
        ]

    return messages


def _db_provider(provider: str) -> SimpleNamespace:
    if provider == "microsoft":
        return SimpleNamespace(
            client_id=encrypt_value("microsoft-client-id"),
            client_secret=encrypt_value("microsoft-client-secret"),
            auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
            userinfo_url="https://graph.microsoft.com/v1.0/me",
            redirect_uri="https://xagent.example/api/auth/microsoft/callback",
            default_scopes=["User.Read"],
            user_id_path="id",
            email_path="mail",
        )
    return SimpleNamespace(
        client_id=encrypt_value("client-id"),
        client_secret=encrypt_value("client-secret"),
        auth_url="https://provider.example/authorize",
        token_url="https://provider.example/token",
        userinfo_url="https://provider.example/me",
        redirect_uri="https://xagent.example/api/auth/custom/callback",
        default_scopes=["profile.read"],
        user_id_path="id",
        email_path="mail",
    )


class _Flow(NamedTuple):
    provider: str
    state: str
    cookie: tuple[str, str]
    nonce: str


def _start(db: Session, user: User, app_id: str = "outlook") -> _Flow:
    provider = APPS[app_id]["provider_name"]
    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider=provider,
        app_id=app_id,
        user=user,
        resource_owner_key=OWNER,
        redirect="https://app.example/connectors",
        db=db,
        db_provider=_db_provider(provider),
    )
    db.commit()
    state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
    jar = SimpleCookie()
    jar.load(response.headers["set-cookie"])
    ((cookie_name, morsel),) = [
        (name, morsel)
        for name, morsel in jar.items()
        if name.startswith("xagent_actor_oauth_")
    ]
    payload = auth_api.verify_token(state)
    assert payload is not None
    return _Flow(
        provider=provider,
        state=state,
        cookie=(cookie_name, morsel.value),
        nonce=payload["actor_flow_nonce"],
    )


def _callback(
    db: Session,
    flow: _Flow,
    *,
    state: str | None = None,
    cookie: tuple[str, str] | None = None,
    without_cookie: bool = False,
    **query: str,
):
    params = {"state": state or flow.state}
    if "error" not in query:
        params["code"] = "authorization-code-value"
    params.update(query)
    name, value = cookie or flow.cookie
    request = SimpleNamespace(
        query_params=params,
        cookies={} if without_cookie else {name: value},
    )
    return generic_oauth_callback(
        flow.provider, request, db, _db_provider(flow.provider)
    )


def _resign(
    flow: _Flow, *, lifetime: timedelta = timedelta(minutes=10), **changes: object
) -> str:
    payload = auth_api.verify_token(flow.state)
    assert payload is not None
    payload.update(changes)
    return create_access_token(data=payload, expires_delta=lifetime)


def _expire(db: Session, flow: _Flow, **changes: object) -> str:
    """Let the whole flow lapse, as it does in a browser after 10 minutes.

    The flow row, the state token and the browser cookie share one lifetime,
    so the returned state is past its expiry and callers drop the cookie.
    """
    db.query(ActorOAuthFlowState).update(
        {
            ActorOAuthFlowState.expires_at: datetime.now(timezone.utc)
            - timedelta(seconds=5)
        }
    )
    db.commit()
    state = _resign(flow, lifetime=timedelta(seconds=-5), **changes)
    assert auth_api.verify_token(state) is None
    return state


def _claims(flow: _Flow) -> dict[str, object]:
    payload = auth_api.verify_token(flow.state)
    assert payload is not None
    return payload


def _sign(claims: dict[str, object]) -> str:
    """Sign ``claims`` as given, without the expiry create_access_token adds."""
    return jwt.encode(claims, auth_api.JWT_SECRET_KEY, algorithm=auth_api.JWT_ALGORITHM)


def _open_flows(db: Session) -> int:
    return db.query(ActorOAuthFlowState).count()


def _actor_grants(db: Session) -> int:
    return db.query(UserOAuth).filter(UserOAuth.resource_owner_key == OWNER).count()


def _body(response) -> str:
    return response.body.decode()


def _assert_actionable(response) -> str:
    body = _body(response)
    assert response.status_code == 400
    assert response.headers["cache-control"] == "no-store"
    assert CLOSE_BUTTON in body
    assert "select Connect again" in body
    assert OLD_ACTOR_ERROR not in body
    return body


def _assert_rejected(response, page: tuple[str, str]) -> str:
    """Assert the page of a failed actor check, and only that one."""
    body = _assert_actionable(response)
    assert f"<title>{REJECTED_TITLE}</title>" in body
    assert (
        f"<body><h1>{REJECTED_TITLE}</h1>"
        + "".join(f"<p>{paragraph}</p>" for paragraph in page)
        + f"<p>{CLOSE_BUTTON}Close this window</button></p>"
        + "<p>If this window doesn't close, close it from your browser.</p>"
        + "</body>"
    ) in body
    for other in (OTHER_BROWSER_PAGE, CONNECTION_CHANGED_PAGE, UNVERIFIED_PAGE):
        if other != page:
            assert not any(paragraph in body for paragraph in other)
    if page != OTHER_BROWSER_PAGE:
        assert "same browser" not in body
        assert "different browser" not in body
    return body


def _assert_no_flow_values(text: str, flow: _Flow) -> None:
    for secret in (
        flow.state,
        flow.cookie[1],
        flow.nonce,
        flow.nonce[:24],
        OWNER,
        "authorization-code-value",
        "new-access",
        "new-refresh",
    ):
        assert secret not in text


def test_declined_microsoft_consent_explains_how_to_retry(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(
        db,
        flow,
        error="consent_required",
        error_description=DECLINED_DESCRIPTION,
        error_uri=DECLINED_URI,
    )

    body = _assert_actionable(response)
    assert "Permission was not granted" in body
    assert "choose Accept" in body
    assert "Microsoft 365 admin" in body
    assert "<code>consent_required, AADSTS65004</code>" in body
    for raw in ("User declined", "Trace ID", "Correlation ID", DECLINED_URI):
        assert raw not in body
    _assert_no_flow_values(body, flow)
    assert _open_flows(db) == 0
    token_endpoint.assert_not_called()
    assert _actor_grants(db) == 0

    (log,) = callback_logs()
    assert "provider=microsoft app_id=outlook actor_flow=true" in log
    assert "reason=provider_error" in log
    assert "provider_error=consent_required" in log
    assert "provider_error_code=AADSTS65004" in log
    assert "User declined" not in log
    _assert_no_flow_values(log, flow)


@pytest.mark.parametrize(
    ("description", "codes"),
    [
        (None, "access_denied"),
        (
            "AADSTS90094: The grant requires admin permission.",
            "access_denied, AADSTS90094",
        ),
    ],
)
def test_cancelled_microsoft_sign_in_asks_to_accept_without_admin_link(
    oauth_db, token_endpoint, description, codes
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    query = {"error": "access_denied", "error_subcode": "cancel"}
    if description is not None:
        query["error_description"] = description

    response = _callback(db, flow, **query)

    body = _assert_actionable(response)
    assert "Permission was not granted" in body
    assert "Microsoft 365 admin" in body
    assert f"<code>{codes}</code>" in body
    assert "href=" not in body
    assert "adminconsent" not in body
    assert _open_flows(db) == 0
    token_endpoint.assert_not_called()


def test_other_providers_get_neutral_not_granted_guidance(
    oauth_db, token_endpoint
) -> None:
    db, user = oauth_db
    flow = _start(db, user, app_id="calendar")

    response = _callback(
        db, flow, error="access_denied", error_description="AADSTS65004: declined"
    )

    body = _assert_actionable(response)
    assert "Permission was not granted" in body
    assert "approve the requested access" in body
    assert "Microsoft" not in body
    assert "<code>access_denied</code>" in body


def test_rejected_microsoft_request_shows_category_and_code(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(
        db,
        flow,
        error="invalid_client",
        error_description=(
            "AADSTS650051: The application needs to be added to the tenant "
            "first. Trace ID: 1234"
        ),
    )

    body = _assert_actionable(response)
    assert "The sign-in request was rejected" in body
    assert "<code>invalid_client, AADSTS650051</code>" in body
    assert "contact your Microsoft 365 admin or support" in body
    assert "added to the tenant" not in body
    (log,) = callback_logs()
    assert "provider_error=invalid_client" in log
    assert "provider_error_code=AADSTS650051" in log
    assert "added to the tenant" not in log


def test_temporarily_unavailable_provider_asks_to_retry_later(
    oauth_db, token_endpoint
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(db, flow, error="temporarily_unavailable")

    body = _assert_actionable(response)
    assert "temporarily unavailable" in body
    assert "in a few minutes" in body
    assert "<code>temporarily_unavailable</code>" in body


@pytest.mark.parametrize(
    ("error", "shown"),
    [
        ("interaction_required", "interaction_required"),
        ("<script>alert(1)</script>", "unknown_error"),
        ("consent required", "unknown_error"),
        ("x" * 65, "unknown_error"),
    ],
)
def test_unrecognized_provider_error_is_generic_and_never_echoed(
    oauth_db, token_endpoint, callback_logs, error, shown
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(db, flow, error=error)

    body = _assert_actionable(response)
    assert "The sign-in did not complete" in body
    assert f"<code>{shown}</code>" in body
    (log,) = callback_logs()
    assert f"provider_error={shown} " in log
    if shown == "unknown_error":
        assert error not in body
        assert error not in log
        assert "<script" not in body


@pytest.mark.parametrize(
    ("provider", "app_id"),
    [
        (
            "microsoft\r\nOAuth callback did not connect: provider=microsoft",
            "outlook\nreason=connected",
        ),
        ("micro soft", "out look"),
        ("<script>", "outlook;drop"),
        ("x" * 65, "x" * 65),
        ("microsoft\t", ["outlook"]),
        ("microsoft/../custom", 42),
    ],
)
def test_unsafe_provider_and_app_id_are_logged_as_invalid(
    oauth_db, token_endpoint, callback_logs, provider, app_id
) -> None:
    # `provider` is a URL path value and `app_id` a state claim: neither may
    # add a line or a field to the log.
    db, user = oauth_db
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": provider,
            "app_id": app_id,
            "redirect": None,
        },
        expires_delta=timedelta(minutes=10),
    )
    request = _request({"state": state, "error": "access_denied"})

    response = generic_oauth_callback(provider, request, db, _db_provider("custom"))

    assert response.status_code == 400
    assert _body(response) == "<h1>Error: access_denied</h1>"
    token_endpoint.assert_not_called()
    (log,) = callback_logs()
    assert log == (
        f"{CALLBACK_LOG_PREFIX}: provider=invalid app_id=invalid actor_flow=false "
        "reason=provider_error provider_error=access_denied provider_error_code=-"
    )


@pytest.mark.parametrize(
    ("error", "description", "title", "codes"),
    [
        # Entra ID redirects carry AADSTS codes in error_description; any
        # code it sends that is not a known category is a rejected request.
        (
            "interaction_required",
            "AADSTS50076: Due to a configuration change made by your "
            "administrator, you must use multi-factor authentication.",
            "The sign-in request was rejected",
            "interaction_required, AADSTS50076",
        ),
        # The not-granted codes decide the category whatever the error is.
        (
            "invalid_request",
            "AADSTS65004: User declined to consent to access the app.",
            "Permission was not granted",
            "invalid_request, AADSTS65004",
        ),
        (
            "invalid_request",
            "AADSTS90095: Admin consent is required for the permissions.",
            "Permission was not granted",
            "invalid_request, AADSTS90095",
        ),
        # A temporary error stays temporary even with a code attached.
        (
            "server_error",
            "AADSTS90033: A transient error has occurred. Please try again.",
            "The sign-in service is temporarily unavailable",
            "server_error, AADSTS90033",
        ),
        (
            "server_error",
            None,
            "The sign-in service is temporarily unavailable",
            "server_error",
        ),
    ],
)
def test_microsoft_error_category_follows_its_codes(
    oauth_db, token_endpoint, error, description, title, codes
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    query = {"error": error}
    if description is not None:
        query["error_description"] = description

    response = _callback(db, flow, **query)

    body = _assert_actionable(response)
    assert title in body
    assert f"<code>{codes}</code>" in body
    if description is not None:
        assert description.split(": ", 1)[1] not in body
    token_endpoint.assert_not_called()


def test_rejected_request_from_other_providers_names_no_microsoft_admin(
    oauth_db, token_endpoint
) -> None:
    db, user = oauth_db
    flow = _start(db, user, app_id="calendar")

    response = _callback(db, flow, error="invalid_client")

    body = _assert_actionable(response)
    assert "The sign-in request was rejected" in body
    assert "contact your administrator or support" in body
    assert "Microsoft" not in body
    assert "<code>invalid_client</code>" in body


def test_only_the_first_three_distinct_microsoft_codes_are_shown(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(
        db,
        flow,
        error="invalid_client",
        error_description=(
            "AADSTS700016 and aadsts700016, then AADSTS7000215, AADSTS50011 "
            "and AADSTS50020."
        ),
    )

    body = _assert_actionable(response)
    assert "<code>invalid_client, AADSTS700016, AADSTS7000215, AADSTS50011</code>" in (
        body
    )
    assert "AADSTS50020" not in body
    (log,) = callback_logs()
    assert "provider_error_code=AADSTS700016,AADSTS7000215,AADSTS50011" in log
    assert "AADSTS50020" not in log


def test_microsoft_codes_are_only_ascii_digits(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    description = "AADSTS\u0666\u0665\u0660\u0660\u0664 and AADSTS\uff11\uff12"

    response = _callback(
        db, flow, error="invalid_client", error_description=description
    )
    unauthenticated = _callback(
        db, flow, state="not-a-state", error="x", error_description=description
    )

    assert "<code>invalid_client</code>" in _assert_actionable(response)
    assert unauthenticated.status_code == 400
    first, second = callback_logs()
    assert "provider_error=invalid_client provider_error_code=-" in first
    assert "reason=state_invalid provider_error=x provider_error_code=-" in second


def test_reloading_a_declined_sign_in_window_says_it_was_used(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    query = {
        "error": "consent_required",
        "error_description": DECLINED_DESCRIPTION,
        "error_uri": DECLINED_URI,
    }

    first = _callback(db, flow, **query)
    reload = _callback(db, flow, **query)

    assert "Permission was not granted" in _assert_actionable(first)
    body = _assert_actionable(reload)
    assert "This sign-in window has already been used" in body
    assert "already shows as connected, there is nothing else to do" in body
    token_endpoint.assert_not_called()
    first_log, reload_log = callback_logs()
    assert "reason=provider_error" in first_log
    assert "reason=flow_used_or_expired" in reload_log
    assert "provider_error_code=AADSTS65004" in reload_log
    _assert_no_flow_values(reload_log, flow)


def test_replaying_a_completed_sign_in_keeps_the_grant(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    first = _callback(db, flow)
    replay = _callback(db, flow)

    assert first.status_code == 200
    body = _assert_actionable(replay)
    assert "This sign-in window has already been used" in body
    assert "Connected Successfully" not in body
    assert token_endpoint.call_count == 1
    assert _actor_grants(db) == 1
    (log,) = callback_logs()
    assert "reason=flow_used_or_expired provider_error=-" in log
    _assert_no_flow_values(log, flow)


def test_flow_row_expired_before_its_state_says_it_was_used(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    db.query(ActorOAuthFlowState).update(
        {
            ActorOAuthFlowState.expires_at: datetime.now(timezone.utc)
            - timedelta(seconds=1)
        }
    )
    db.commit()

    response = _callback(db, flow)

    assert "This sign-in window has already been used" in _assert_actionable(response)
    token_endpoint.assert_not_called()
    (log,) = callback_logs()
    assert "reason=flow_used_or_expired" in log


def test_sign_in_window_loaded_after_it_expired_says_it_was_used(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    state = _expire(db, flow)

    response = _callback(db, flow, state=state, without_cookie=True)

    body = _assert_actionable(response)
    assert "This sign-in window has already been used" in body
    _assert_no_flow_values(body, flow)
    token_endpoint.assert_not_called()
    assert _open_flows(db) == 1
    assert _actor_grants(db) == 0
    (log,) = callback_logs()
    assert (
        "provider=microsoft app_id=outlook actor_flow=true reason=state_expired "
        "provider_error=- provider_error_code=-"
    ) in log
    _assert_no_flow_values(log, flow)
    assert state not in log


def test_consent_declined_after_the_window_expired_explains_how_to_retry(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    state = _expire(db, flow)

    response = _callback(
        db,
        flow,
        state=state,
        without_cookie=True,
        error="consent_required",
        error_description=DECLINED_DESCRIPTION,
        error_uri=DECLINED_URI,
    )

    body = _assert_actionable(response)
    assert "Permission was not granted" in body
    assert "choose Accept" in body
    assert "Microsoft 365 admin" in body
    assert "<code>consent_required, AADSTS65004</code>" in body
    for raw in ("User declined", "Trace ID", DECLINED_URI):
        assert raw not in body
    token_endpoint.assert_not_called()
    assert _open_flows(db) == 1
    (log,) = callback_logs()
    assert "actor_flow=true reason=state_expired" in log
    assert "provider_error=consent_required provider_error_code=AADSTS65004" in log
    assert state not in log


@pytest.mark.parametrize(
    "mismatch",
    [
        "forged",
        "other_provider",
        "ordinary_flow",
        "wrong_type",
        "non_numeric_exp",
        "numeric_string_exp",
    ],
)
def test_expired_state_that_is_not_this_actor_flow_keeps_its_pages(
    oauth_db, token_endpoint, callback_logs, mismatch
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    if mismatch == "forged":
        header, payload, signature = _expire(db, flow).split(".")
        state = ".".join(
            (header, payload, ("a" if signature[0] != "a" else "b") + signature[1:])
        )
    elif mismatch == "other_provider":
        state = _expire(db, flow, provider="custom")
    elif mismatch == "ordinary_flow":
        state = _expire(db, flow, actor_flow_nonce=None, resource_owner_key=None)
    elif mismatch == "wrong_type":
        state = _expire(db, flow, type="access")
    else:
        # Both fail verify_token: one is not a number, the other is past.
        _expire(db, flow)
        exp = "soon" if mismatch == "non_numeric_exp" else "1700000000"
        state = _sign({**_claims(flow), "exp": exp})
        assert auth_api.verify_token(state) is None

    success = _callback(db, flow, state=state, without_cookie=True)
    declined = _callback(
        db, flow, state=state, without_cookie=True, error="consent_required"
    )

    assert success.status_code == 400
    assert _body(success) == "<h1>Error: Invalid or expired state</h1>"
    assert declined.status_code == 400
    assert _body(declined) == "<h1>Error: consent_required</h1>"
    token_endpoint.assert_not_called()
    logs = callback_logs()
    assert len(logs) == 2
    for log in logs:
        assert "actor_flow=unknown reason=state_invalid" in log


@pytest.mark.parametrize("exp", ["missing", None, "soon"])
def test_expired_actor_state_requires_a_numeric_expiry(oauth_db, exp) -> None:
    # verify_token accepts a state without exp as live, so the callback never
    # asks whether one has expired; the check itself must still refuse it.
    db, user = oauth_db
    claims = _claims(_start(db, user))
    claims.pop("exp")
    if exp != "missing":
        claims["exp"] = exp
    lapsed = _sign({**claims, "exp": int(time.time()) - 5})

    assert auth_api._expired_actor_oauth_state(lapsed, "microsoft") is not None
    assert auth_api._expired_actor_oauth_state(_sign(claims), "microsoft") is None


@pytest.mark.parametrize(
    ("cookie_mode", "reason"),
    [("missing", "cookie_missing"), ("wrong", "cookie_mismatch")],
)
def test_sign_in_from_another_browser_cannot_finish(
    oauth_db, token_endpoint, callback_logs, cookie_mode, reason
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    if cookie_mode == "missing":
        response = _callback(db, flow, without_cookie=True)
    else:
        response = _callback(db, flow, cookie=(flow.cookie[0], "other-browser"))

    _assert_rejected(response, OTHER_BROWSER_PAGE)
    assert _open_flows(db) == 1
    token_endpoint.assert_not_called()
    (log,) = callback_logs()
    assert f"reason={reason} " in log
    assert "other-browser" not in log
    _assert_no_flow_values(log, flow)


def test_provider_error_with_failed_actor_check_logs_both(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(
        db,
        flow,
        without_cookie=True,
        error="consent_required",
        error_description=DECLINED_DESCRIPTION,
    )

    _assert_rejected(response, OTHER_BROWSER_PAGE)
    assert _open_flows(db) == 1
    (log,) = callback_logs()
    assert "reason=cookie_missing" in log
    assert "provider_error=consent_required" in log
    assert "provider_error_code=AADSTS65004" in log


def _bad_claims(db: Session, user: User, flow: _Flow) -> str:
    return _resign(flow, actor_flow_nonce="not-a-nonce")


def _bad_owner(db: Session, user: User, flow: _Flow) -> str:
    return _resign(flow, resource_owner_key=OWNER)


def _missing_user(db: Session, user: User, flow: _Flow) -> str:
    return _resign(flow, user_id=int(user.id) + 1000)


def _inactive_link(db: Session, user: User, flow: _Flow) -> str:
    db.query(UserMCPServer).update({UserMCPServer.is_active: False})
    db.commit()
    return flow.state


def _catalog_drift(db: Session, user: User, flow: _Flow) -> str:
    db.query(PublicMCPApp).filter_by(app_id="outlook").one().provider_name = "other"
    db.commit()
    return flow.state


@pytest.mark.parametrize(
    ("break_flow", "reason", "page"),
    [
        (_bad_claims, "claims_invalid", UNVERIFIED_PAGE),
        (_bad_owner, "owner_invalid", UNVERIFIED_PAGE),
        (_missing_user, "user_missing", CONNECTION_CHANGED_PAGE),
        (_inactive_link, "link_invalid", CONNECTION_CHANGED_PAGE),
        (_catalog_drift, "catalog_invalid", CONNECTION_CHANGED_PAGE),
    ],
)
def test_each_actor_check_logs_its_reason_and_explains_it(
    oauth_db, token_endpoint, callback_logs, break_flow, reason, page
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    state = break_flow(db, user, flow)

    response = _callback(db, flow, state=state)

    _assert_rejected(response, page)
    assert _open_flows(db) == 1
    token_endpoint.assert_not_called()
    assert _actor_grants(db) == 0
    (log,) = callback_logs()
    assert f"reason={reason} " in log
    _assert_no_flow_values(log, flow)


def test_link_removed_during_exchange_cannot_finish(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    exchange = token_endpoint.return_value

    def disconnect_during_exchange(*args, **kwargs):
        db.query(UserMCPServer).update({UserMCPServer.is_active: False})
        db.commit()
        return exchange

    token_endpoint.side_effect = disconnect_during_exchange

    response = _callback(db, flow)

    _assert_rejected(response, CONNECTION_CHANGED_PAGE)
    assert token_endpoint.call_count == 1
    assert _actor_grants(db) == 0
    (log,) = callback_logs()
    assert "reason=link_changed_after_exchange" in log
    _assert_no_flow_values(log, flow)


@pytest.mark.parametrize("first_attempt", ["declined", "other-browser"])
def test_selecting_connect_again_after_a_failed_attempt_connects(
    oauth_db, token_endpoint, first_attempt
) -> None:
    db, user = oauth_db
    failed = _start(db, user)
    if first_attempt == "declined":
        _assert_actionable(_callback(db, failed, error="consent_required"))
    else:
        _assert_actionable(_callback(db, failed, without_cookie=True))

    retry = _start(db, user)
    response = _callback(db, retry)

    assert response.status_code == 200
    assert "Connected Successfully" in _body(response)
    assert _actor_grants(db) == 1


def _ordinary_state(user: User) -> str:
    return create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "microsoft",
            "app_id": "outlook",
            "redirect": None,
        },
        expires_delta=timedelta(minutes=10),
    )


def _ordinary_callback(db: Session, state: str, **query: str):
    request = SimpleNamespace(query_params={"state": state, **query}, cookies={})
    return generic_oauth_callback("microsoft", request, db, _db_provider("microsoft"))


def test_ordinary_flow_provider_error_pages_are_unchanged(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    state = _ordinary_state(user)

    denied = _ordinary_callback(db, state, error="access_denied")
    admin = _ordinary_callback(
        db,
        _ordinary_state(user),
        error="access_denied",
        error_subcode="cancel",
        error_description="AADSTS90094: The grant requires admin permission.",
    )

    assert denied.status_code == 400
    assert _body(denied) == "<h1>Error: access_denied</h1>"
    assert "cache-control" not in denied.headers
    assert admin.status_code == 400
    assert "Admin approval required" in _body(admin)
    assert "adminconsent" in _body(admin)
    token_endpoint.assert_not_called()
    denied_log, admin_log = callback_logs()
    assert "actor_flow=false reason=provider_error" in denied_log
    assert "provider_error_code=AADSTS90094" in admin_log


def test_invalid_state_keeps_its_page_and_logs_reason(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    header, payload, signature = flow.state.split(".")
    tampered = ".".join(
        (header, payload, ("a" if signature[0] != "a" else "b") + signature[1:])
    )

    response = _callback(db, flow, state=tampered)

    assert response.status_code == 400
    assert _body(response) == "<h1>Error: Invalid or expired state</h1>"
    assert _open_flows(db) == 1
    (log,) = callback_logs()
    assert "actor_flow=unknown reason=state_invalid" in log
    assert tampered not in log


def test_malformed_user_claim_on_an_actor_state_logs_its_reason(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    state = _resign(flow, user_id=2**31)

    response = _callback(db, flow, state=state)

    assert response.status_code == 400
    assert _body(response) == "<h1>Error: Invalid or expired state</h1>"
    assert _open_flows(db) == 1
    token_endpoint.assert_not_called()
    (log,) = callback_logs()
    assert "app_id=outlook actor_flow=true reason=state_invalid" in log
    assert state not in log
    _assert_no_flow_values(log, flow)


def _request(query: dict[str, str], cookies: dict[str, str] | None = None):
    return SimpleNamespace(query_params=query, cookies=cookies or {})


def _no_state_provider_error(db, user, monkeypatch):
    return "microsoft", _request({"error": "access_denied"}), _db_provider("microsoft")


def _no_state(db, user, monkeypatch):
    return "microsoft", _request({"code": "x"}), _db_provider("microsoft")


def _no_code(db, user, monkeypatch):
    flow = _start(db, user)
    request = _request({"state": flow.state}, dict([flow.cookie]))
    return "microsoft", request, _db_provider("microsoft")


def _restricted_gmail(db, user, monkeypatch):
    monkeypatch.setattr(auth_api, "get_google_restricted_scopes", lambda: False)
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": "gmail",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = _request({"state": state, "code": "x"})
    return "google", request, _db_provider("custom")


def _unreadable_verifier(db, user, monkeypatch):
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "microsoft",
            "app_id": "outlook",
            # Token-shaped, but sealed under a key this deployment never had.
            "code_verifier": Fernet(Fernet.generate_key()).encrypt(b"v").decode(),
        },
        expires_delta=timedelta(minutes=10),
    )
    request = _request({"state": state, "code": "x"})
    return "microsoft", request, _db_provider("microsoft")


def _bare_scoped_grant(db, user, monkeypatch):
    state = create_access_token(
        data={"type": "oauth_state", "user_id": user.id, "provider": "github"},
        expires_delta=timedelta(minutes=10),
    )
    return "github", _request({"state": state, "code": "x"}), _db_provider("custom")


def _actor_flow_with(db, user, db_provider):
    flow = _start(db, user)
    request = _request({"state": flow.state, "code": "x"}, dict([flow.cookie]))
    return "microsoft", request, db_provider


def _hidden_app(db, user, monkeypatch):
    # An actor flow already stops at its catalog check for a hidden app.
    app = db.query(PublicMCPApp).filter_by(app_id="outlook").one()
    app.is_visible_in_connector = False
    db.commit()
    request = _request({"state": _ordinary_state(user), "code": "x"})
    return "microsoft", request, _db_provider("microsoft")


def _unconfigured_provider(db, user, monkeypatch):
    return _actor_flow_with(db, user, None)


def _missing_client_config(db, user, monkeypatch):
    monkeypatch.delenv("MICROSOFT_CLIENT_ID", raising=False)
    monkeypatch.delenv("MICROSOFT_CLIENT_SECRET", raising=False)
    db_provider = _db_provider("microsoft")
    db_provider.client_id = None
    db_provider.client_secret = None
    return _actor_flow_with(db, user, db_provider)


def _missing_business_id(db, user, monkeypatch):
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "myob",
            "app_id": "myob",
        },
        expires_delta=timedelta(minutes=10),
    )
    return "myob", _request({"state": state, "code": "x"}), _db_provider("custom")


@pytest.mark.parametrize(
    ("callback", "status_code", "page", "logged"),
    [
        (
            _no_state_provider_error,
            400,
            "Error: access_denied",
            "app_id=- actor_flow=unknown reason=provider_error "
            "provider_error=access_denied",
        ),
        (
            _no_state,
            400,
            "Missing code or state",
            "app_id=- actor_flow=unknown reason=state_missing",
        ),
        (
            _no_code,
            400,
            "Missing code or state",
            "app_id=- actor_flow=unknown reason=code_missing",
        ),
        (
            _restricted_gmail,
            404,
            "This app is not currently available",
            "app_id=gmail actor_flow=false reason=app_restricted",
        ),
        (
            _unreadable_verifier,
            400,
            "Session expired",
            "app_id=outlook actor_flow=false reason=verifier_invalid",
        ),
        (
            _bare_scoped_grant,
            404,
            "must be started from its catalog entry",
            "app_id=- actor_flow=false reason=app_id_missing",
        ),
        (
            _hidden_app,
            404,
            "This app is not currently available",
            "app_id=outlook actor_flow=false reason=app_hidden",
        ),
        (
            _unconfigured_provider,
            500,
            "Provider not configured",
            "app_id=outlook actor_flow=true reason=provider_not_configured",
        ),
        (
            _missing_client_config,
            500,
            "OAuth provider not configured",
            "app_id=outlook actor_flow=true reason=provider_config_missing",
        ),
        (
            _missing_business_id,
            400,
            "did not return a businessId",
            "app_id=myob actor_flow=false reason=business_id_missing",
        ),
    ],
)
def test_callbacks_stopped_before_the_token_exchange_log_their_reason(
    oauth_db,
    token_endpoint,
    callback_logs,
    monkeypatch,
    callback,
    status_code,
    page,
    logged,
) -> None:
    db, user = oauth_db
    provider, request, db_provider = callback(db, user, monkeypatch)

    response = generic_oauth_callback(provider, request, db, db_provider)

    assert response.status_code == status_code
    assert page in _body(response)
    token_endpoint.assert_not_called()
    (log,) = callback_logs()
    assert f"provider={provider} {logged}" in log
    for value in (request.query_params.get("state"), *request.cookies.values()):
        if value:
            assert value not in log
