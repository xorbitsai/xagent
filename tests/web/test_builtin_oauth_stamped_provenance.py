"""Builtin OAuth rows created by the catalog callback stay canonical.

The regular catalog OAuth callback (``_ensure_user_mcp_server``) creates the
shared ``MCPServer`` row with the catalog's ``builtin_provenance`` marker in
``auth`` so seed migrations can recognize it as the official row. The trusted
actor paths validate the same row with ``_validate_canonical_builtin_oauth_server``
and must accept that marker, otherwise every actor connect for the app fails
whenever a regular user happened to create the row first. The marker is
accepted only when it names the app's own builtin identity.
"""

from __future__ import annotations

from copy import deepcopy
from http.cookies import SimpleCookie
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Query, Session, sessionmaker
from sqlalchemy.orm.attributes import flag_modified

from tests.shared.oauth_browser_fixture import generic_oauth_callback
from xagent.core.utils.encryption import encrypt_value
from xagent.web import mcp_apps
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import (
    _ensure_user_mcp_server,
    _require_actor_oauth_personal_link,
    start_builtin_oauth_for_resource_owner,
)
from xagent.web.auth_sessions import lock_current_session_epoch
from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth

# Builtin OAuth apps whose catalog rows declare a builtin_provenance marker,
# mapped to their OAuth provider.
STAMPED_BUILTIN_OAUTH_APPS = {
    "word": "microsoft",
    "excel": "microsoft",
    "powerpoint": "microsoft",
    "planner": "microsoft",
    "sharepoint": "microsoft",
    "whatsapp": "meta",
}
# Stamped too, but hidden until verified, so the actor paths refuse them.
HIDDEN_STAMPED_BUILTIN_OAUTH_APPS = {"zendesk"}


@pytest.fixture
def seeded_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'stamped-provenance.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        seed_builtin_oauth_and_public_mcp_apps(connection)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        web_user = User(username="web-user", password_hash="hash")
        account = User(username="workspace-account", password_hash="hash")
        db.add_all([web_user, account])
        db.commit()
        db.refresh(web_user)
        db.refresh(account)
        yield db, web_user, account
    engine.dispose()


def _catalog_app(db: Session, app_id: str) -> dict[str, Any]:
    app_info = mcp_apps.get_app_by_id(db, app_id)
    assert app_info is not None
    return app_info


def _connect_from_catalog(db: Session, user: User, app_id: str) -> MCPServer:
    """Create the shared row the way the regular catalog OAuth callback does."""
    app_info = _catalog_app(db, app_id)
    _ensure_user_mcp_server(db, int(user.id), app_info)
    db.commit()
    return db.query(MCPServer).filter(MCPServer.name == app_info["name"]).one()


@pytest.mark.parametrize("autoflush", [False, True])
def test_catalog_insert_collision_preserves_credential_transaction_and_epoch_lock(
    seeded_db, monkeypatch, autoflush
):
    db, web_user, account = seeded_db
    winner = _connect_from_catalog(db, account, "word")
    app_info = _catalog_app(db, "word")
    user_id, winner_id = int(web_user.id), int(winner.id)
    db.autoflush = autoflush
    assert lock_current_session_epoch(db, user_id, None)
    outer = db.get_transaction()
    db.add(UserOAuth(user_id=user_id, provider="word", access_token="new-credential"))
    original_first = Query.first
    missed = False

    def stale_catalog_lookup(query):
        nonlocal missed
        if query.column_descriptions[0].get("entity") is MCPServer and not missed:
            # Model a competing user's insert just after our initial SELECT.
            # The persisted winner makes the subsequent INSERT hit the real
            # unique constraint without needing concurrent SQLite writers.
            missed = True
            return None
        return original_first(query)

    monkeypatch.setattr(Query, "first", stale_catalog_lookup)
    _ensure_user_mcp_server(db, user_id, app_info)
    assert missed
    assert db.get_transaction() is outer
    assert outer.is_active
    db.commit()
    assert (
        db.query(UserOAuth)
        .filter_by(user_id=user_id, provider="word")
        .one()
        .access_token
        == "new-credential"
    )
    assert (
        db.query(UserMCPServer).filter_by(user_id=user_id).one().mcpserver_id
        == winner_id
    )
    assert db.query(MCPServer).filter_by(name=app_info["name"]).count() == 1


def test_catalog_savepoint_does_not_commit_a_new_server_before_the_caller(seeded_db):
    db, web_user, _account = seeded_db
    app_info = _catalog_app(db, "word")
    user_id = int(web_user.id)
    _ensure_user_mcp_server(db, user_id, app_info)
    db.rollback()
    assert db.query(MCPServer).filter_by(name=app_info["name"]).count() == 0
    assert db.query(UserMCPServer).filter_by(user_id=user_id).count() == 0


def _catalog_marker(db: Session, app_id: str) -> dict[str, Any]:
    marker = _catalog_app(db, app_id)["launch_config"]["builtin_provenance"]
    assert isinstance(marker, dict)
    return dict(marker)


def _set_auth(db: Session, server: MCPServer, auth: Any) -> None:
    server.auth = deepcopy(auth)
    db.commit()
    db.refresh(server)


_WORD_MARKER = {"registry": "xagent", "app_id": "word", "version": 1}


def _word_auth_with_marker(marker: Any) -> dict[str, Any]:
    return {"app_id": "word", "provider": "microsoft", "builtin_provenance": marker}


def _rejects_auth_only():
    """Expect the canonical check to fail on ``auth`` and on no other field."""
    return pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError, match=r"definition \(auth\)$"
    )


@pytest.mark.parametrize("app_id", sorted(STAMPED_BUILTIN_OAUTH_APPS))
def test_actor_paths_accept_row_created_by_catalog_callback(seeded_db, app_id) -> None:
    db, web_user, account = seeded_db
    provider = STAMPED_BUILTIN_OAUTH_APPS[app_id]
    server = _connect_from_catalog(db, web_user, app_id)
    stored_auth = deepcopy(server.auth)
    assert stored_auth == {
        "app_id": app_id,
        "provider": provider,
        "builtin_provenance": _catalog_marker(db, app_id),
    }

    ensured = mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
        db, user_id=int(account.id), app_id=app_id
    )
    db.commit()
    required = mcp_apps.require_builtin_oauth_server_definition(
        db, app_id=app_id, provider=provider
    )
    classified = mcp_apps.classify_actor_builtin_oauth_server(db, server)
    _require_actor_oauth_personal_link(
        db, user_id=int(account.id), provider=provider, app_id=app_id
    )

    assert ensured.id == server.id
    assert required.id == server.id
    assert classified is not None and classified["id"] == app_id
    db.refresh(server)
    assert server.auth == stored_auth
    assert db.query(MCPServer).count() == 1
    links = db.query(UserMCPServer).order_by(UserMCPServer.user_id).all()
    assert [(link.user_id, link.is_owner) for link in links] == [
        (int(web_user.id), True),
        (int(account.id), False),
    ]


def test_catalog_callback_auth_is_canonical_for_every_builtin_oauth_app(
    seeded_db,
) -> None:
    """The callback and the canonical check agree on the whole catalog.

    Both take the app's own marker from ``owned_catalog_marker``. The apps that
    get one are exactly the ones the actor path cases above cover.
    """
    db, web_user, _account = seeded_db
    stamped: set[str] = set()
    for app in db.query(PublicMCPApp).order_by(PublicMCPApp.app_id).all():
        app_info = _catalog_app(db, app.app_id)
        if app_info["auth_type"] != "builtin_oauth":
            continue
        server = _connect_from_catalog(db, web_user, app.app_id)
        assert mcp_apps._builtin_auth_is_canonical(server.auth, app_info), app.app_id
        if "builtin_provenance" in server.auth:
            stamped.add(app.app_id)

    assert (
        stamped == set(STAMPED_BUILTIN_OAUTH_APPS) | HIDDEN_STAMPED_BUILTIN_OAUTH_APPS
    )


@pytest.mark.parametrize("marker_persisted", [False, True])
def test_actor_created_row_stays_canonical_after_catalog_connect(
    seeded_db, marker_persisted
) -> None:
    """A catalog connect on an actor-created row keeps the actor paths working.

    The catalog callback merges its metadata, marker included, into the
    existing ``auth`` dict in place. The first case pins only the identity
    fields, so it holds whether or not that merge reaches the stored row; the
    second persists the merged metadata explicitly.
    """
    db, web_user, account = seeded_db
    server = mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
        db, user_id=int(account.id), app_id="word"
    )
    db.commit()
    assert server.auth == {"app_id": "word", "provider": "microsoft"}

    app_info = _catalog_app(db, "word")
    _ensure_user_mcp_server(db, int(web_user.id), app_info)
    if marker_persisted:
        flag_modified(server, "auth")
    db.commit()
    db.expire_all()

    if marker_persisted:
        assert server.auth == _word_auth_with_marker(_catalog_marker(db, "word"))
    else:
        assert (server.auth["app_id"], server.auth["provider"]) == ("word", "microsoft")
    assert (
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id="word", provider="microsoft"
        ).id
        == server.id
    )
    classified = mcp_apps.classify_actor_builtin_oauth_server(db, server)
    assert classified is not None and classified["id"] == "word"
    _require_actor_oauth_personal_link(
        db, user_id=int(account.id), provider="microsoft", app_id="word"
    )


def test_stamped_marker_with_other_version_is_accepted(seeded_db) -> None:
    db, web_user, _account = seeded_db
    server = _connect_from_catalog(db, web_user, "word")
    marker = _catalog_marker(db, "word")
    marker["version"] = int(marker["version"]) + 1
    _set_auth(db, server, _word_auth_with_marker(marker))

    required = mcp_apps.require_builtin_oauth_server_definition(
        db, app_id="word", provider="microsoft"
    )

    assert required.id == server.id


@pytest.mark.parametrize(
    "auth",
    [
        pytest.param(
            _word_auth_with_marker({**_WORD_MARKER, "app_id": "excel"}),
            id="other-app-marker",
        ),
        pytest.param(
            _word_auth_with_marker({**_WORD_MARKER, "registry": "custom"}),
            id="other-registry-marker",
        ),
        pytest.param(
            _word_auth_with_marker({"app_id": "word", "version": 1}),
            id="marker-without-registry",
        ),
        pytest.param(_word_auth_with_marker("xagent:word"), id="string-marker"),
        pytest.param(_word_auth_with_marker(["xagent", "word"]), id="list-marker"),
        pytest.param(_word_auth_with_marker(None), id="null-marker"),
        pytest.param(
            _word_auth_with_marker({**_WORD_MARKER, "command": "/bin/foreign"}),
            id="marker-with-extra-key",
        ),
        pytest.param(
            {**_word_auth_with_marker(_WORD_MARKER), "client_secret": "foreign"},
            id="extra-auth-key",
        ),
        pytest.param(
            {"app_id": "word", "builtin_provenance": _WORD_MARKER},
            id="marker-without-provider",
        ),
        pytest.param(
            {**_word_auth_with_marker(_WORD_MARKER), "provider": "google"},
            id="marker-with-wrong-provider",
        ),
    ],
)
def test_noncanonical_stamped_auth_is_rejected_and_not_repaired(
    seeded_db, auth
) -> None:
    db, web_user, account = seeded_db
    server = _connect_from_catalog(db, web_user, "word")
    _set_auth(db, server, auth)

    with _rejects_auth_only():
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id="word", provider="microsoft"
        )
    with _rejects_auth_only():
        mcp_apps.classify_actor_builtin_oauth_server(db, server)
    with _rejects_auth_only():
        mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
            db, user_id=int(account.id), app_id="word"
        )
    db.rollback()
    db.refresh(server)

    assert server.auth == auth
    assert (
        db.query(UserMCPServer).filter(UserMCPServer.user_id == int(account.id)).count()
        == 0
    )


@pytest.mark.parametrize("app_id", ["outlook", "onedrive", "teams"])
def test_marker_is_rejected_for_app_without_catalog_marker(seeded_db, app_id) -> None:
    db, web_user, _account = seeded_db
    server = _connect_from_catalog(db, web_user, app_id)
    assert server.auth == {"app_id": app_id, "provider": "microsoft"}
    marker = {**_WORD_MARKER, "app_id": app_id}
    _set_auth(db, server, {**server.auth, "builtin_provenance": marker})

    with _rejects_auth_only():
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id=app_id, provider="microsoft"
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("managed", "internal"),
        ("command", "/bin/foreign"),
        ("url", "https://foreign.example"),
        ("env", {"FOREIGN": "value"}),
        ("allow_delegated_authorization", True),
    ],
)
def test_stamped_row_keeps_other_canonical_checks(seeded_db, field_name, value) -> None:
    db, web_user, _account = seeded_db
    server = _connect_from_catalog(db, web_user, "word")
    setattr(server, field_name, value)
    db.commit()

    with pytest.raises(mcp_apps.BuiltinOAuthServerDefinitionError, match=field_name):
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id="word", provider="microsoft"
        )


ACTOR_OWNER = "workspace:member:actor-one"


class _ProviderResponse:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data
        self.status_code = 200

    def json(self) -> dict[str, object]:
        return self._data


def _oauth_provider(provider: str) -> SimpleNamespace:
    return SimpleNamespace(
        client_id=encrypt_value("client-id"),
        client_secret=encrypt_value("client-secret"),
        auth_url="https://provider.example/authorize",
        token_url="https://provider.example/token",
        userinfo_url="https://provider.example/me",
        redirect_uri=f"https://xagent.example/api/auth/{provider}/callback",
        default_scopes=["profile.read"],
        user_id_path="id",
        email_path="email",
    )


def _actor_flow_cookie(response) -> tuple[str, str]:
    parsed = SimpleCookie()
    parsed.load(response.headers["set-cookie"])
    (cookie,) = [
        (name, morsel.value)
        for name, morsel in parsed.items()
        if name.startswith("xagent_actor_oauth_")
    ]
    return cookie


@pytest.mark.parametrize(
    "app_id",
    sorted(
        app_id
        for app_id, provider in STAMPED_BUILTIN_OAUTH_APPS.items()
        if provider == "microsoft"
    ),
)
def test_actor_oauth_completes_on_row_created_by_catalog_callback(
    seeded_db, monkeypatch, app_id
) -> None:
    """Start and callback of an actor OAuth flow both accept the stamped row."""
    db, web_user, account = seeded_db
    provider = STAMPED_BUILTIN_OAUTH_APPS[app_id]
    server = _connect_from_catalog(db, web_user, app_id)
    stored_auth = deepcopy(server.auth)
    assert "builtin_provenance" in stored_auth
    mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
        db, user_id=int(account.id), app_id=app_id
    )
    db.commit()

    start = start_builtin_oauth_for_resource_owner(
        provider=provider,
        app_id=app_id,
        user=account,
        resource_owner_key=ACTOR_OWNER,
        redirect="https://actor.example/settings",
        db=db,
        db_provider=_oauth_provider(provider),
    )
    db.commit()
    assert start.status_code == 307
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    cookie_name, cookie_value = _actor_flow_cookie(start)

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=_ProviderResponse(
                {
                    "access_token": "actor-access",
                    "refresh_token": "actor-refresh",
                    "expires_in": 3600,
                    "scope": "profile.read",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=_ProviderResponse(
                {"id": "member-account", "email": "member@example.com"}
            )
        ),
    )
    response = generic_oauth_callback(
        provider,
        SimpleNamespace(
            query_params={"state": state, "code": "code"},
            cookies={cookie_name: cookie_value},
        ),
        db,
        _oauth_provider(provider),
    )

    assert response.status_code == 200
    (credential,) = (
        db.query(UserOAuth).filter(UserOAuth.resource_owner_key == ACTOR_OWNER).all()
    )
    assert (credential.user_id, credential.provider) == (int(account.id), app_id)
    assert credential.access_token
    db.refresh(server)
    assert server.auth == stored_auth
    assert db.query(MCPServer).count() == 1
