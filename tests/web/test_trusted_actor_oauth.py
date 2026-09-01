"""Security contract for trusted actor-owned builtin OAuth flows."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.utils.encryption import decrypt_value_strict, encrypt_value
from xagent.web import mcp_apps
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import (
    create_access_token,
    generic_oauth_callback,
    generic_oauth_login,
)
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth

ACTOR_ALICE = "toby:slack:41:UALICE"
ACTOR_BOB = "toby:slack:41:UBOB"
TEST_BUILTIN_APP_ID = "calendar"
TEST_BUILTIN_EXECUTION = {
    "name": "Google Calendar",
    "transport": "oauth",
    "provider_name": "custom",
    "oauth_scopes": [],
    "launch_config": {"command": "calendar"},
}


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
        if app_id == TEST_BUILTIN_APP_ID:
            return TEST_BUILTIN_EXECUTION, []
        return registry_lookup(app_id)

    monkeypatch.setattr(
        mcp_apps, "get_builtin_execution_fields_and_optional_scopes", test_registry
    )

    engine = create_engine(f"sqlite:///{tmp_path / 'actor-oauth.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)
        yield db, user
    engine.dispose()


def _provider() -> SimpleNamespace:
    return SimpleNamespace(
        client_id=encrypt_value("client-id"),
        client_secret=encrypt_value("client-secret"),
        auth_url="https://provider.example/authorize",
        token_url="https://provider.example/token",
        userinfo_url="https://provider.example/me",
        redirect_uri="https://xagent.example/api/auth/custom/callback",
        default_scopes=["profile.read"],
        user_id_path="id",
        email_path="email",
    )


def _catalog_link(
    db: Session, user: User, *, server_name: str = "Google Calendar"
) -> tuple[MCPServer, UserMCPServer]:
    app = PublicMCPApp(
        app_id="calendar",
        name="Google Calendar",
        description="Calendar",
        transport="oauth",
        provider_name="custom",
        launch_config={"command": "calendar"},
        is_visible_in_connector=True,
    )
    server = MCPServer(
        name=server_name,
        description="Calendar",
        managed="external",
        transport="oauth",
        auth={"app_id": "calendar", "provider": "custom"},
    )
    db.add_all([app, server])
    db.flush()
    link = UserMCPServer(
        user_id=int(user.id),
        mcpserver_id=int(server.id),
        is_owner=False,
        is_active=True,
    )
    db.add(link)
    db.commit()
    return server, link


def _state(response) -> str:
    return parse_qs(urlparse(response.headers["location"]).query)["state"][0]


def _flow_cookie(response) -> tuple[str, str, SimpleCookie]:
    parsed = SimpleCookie()
    parsed.load(response.headers["set-cookie"])
    matches = [
        (name, morsel.value)
        for name, morsel in parsed.items()
        if name.startswith("xagent_actor_oauth_")
    ]
    assert len(matches) == 1
    return matches[0][0], matches[0][1], parsed


def _request(state: str, *, cookie: tuple[str, str] | None = None, code: str = "code"):
    return SimpleNamespace(
        query_params={"state": state, "code": code},
        cookies={} if cookie is None else {cookie[0]: cookie[1]},
    )


def _error_request(state: str, cookie: tuple[str, str]):
    return SimpleNamespace(
        query_params={"state": state, "error": "access_denied"},
        cookies={cookie[0]: cookie[1]},
    )


def _start(
    db: Session,
    user: User,
    owner: str = ACTOR_ALICE,
    *,
    db_provider: SimpleNamespace | None = None,
    commit: bool = True,
):
    start = getattr(auth_api, "start_builtin_oauth_for_resource_owner")
    response = start(
        provider="custom",
        app_id="calendar",
        user=user,
        resource_owner_key=owner,
        redirect="https://toby.example/settings",
        db=db,
        db_provider=db_provider or _provider(),
    )
    if commit:
        db.commit()
    return response


def _mock_exchange(monkeypatch) -> Mock:
    post = Mock(
        return_value=_ProviderResponse(
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
                "scope": "profile.read",
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=_ProviderResponse({"id": "account", "email": "a@example.com"})
        ),
    )
    return post


def test_actor_start_uses_browser_bound_cookie_and_minimal_nonce(oauth_db) -> None:
    db, user = oauth_db
    _catalog_link(db, user)

    response = _start(db, user)

    state = auth_api.verify_token(_state(response))
    assert state is not None
    assert state["user_id"] == user.id
    assert state["provider"] == "custom"
    assert state["app_id"] == "calendar"
    encrypted_owner = state["resource_owner_key"]
    assert encrypted_owner != ACTOR_ALICE
    assert ACTOR_ALICE not in str(state)
    assert json.loads(decrypt_value_strict(encrypted_owner)) == {
        "owner": ACTOR_ALICE,
        "version": 1,
    }
    assert "governing_team_id" not in state
    cookie_name, cookie_value, parsed = _flow_cookie(response)
    morsel = parsed[cookie_name]
    assert cookie_value not in _state(response)
    assert morsel["secure"]
    assert morsel["httponly"]
    assert morsel["samesite"].lower() == "lax"
    flow_model = getattr(
        __import__("xagent.web.models", fromlist=["ActorOAuthFlowState"]),
        "ActorOAuthFlowState",
    )
    assert {column.name for column in flow_model.__table__.columns} == {
        "nonce",
        "expires_at",
    }
    assert db.query(flow_model).count() == 1


def test_actor_cookie_allows_http_callback(oauth_db) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    provider = _provider()
    provider.redirect_uri = "http://xagent.example/api/auth/custom/callback"

    response = _start(db, user, db_provider=provider)

    cookie_name, _cookie_value, parsed = _flow_cookie(response)
    assert not parsed[cookie_name]["secure"]


def test_actor_cookie_uses_callback_path(oauth_db) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    provider = _provider()
    provider.redirect_uri = "https://xagent.example/proxy/api/auth/custom/callback"

    response = _start(db, user, db_provider=provider)

    cookie_name, _cookie_value, parsed = _flow_cookie(response)
    assert parsed[cookie_name]["path"] == "/proxy/api/auth/custom/callback"


def test_actor_cookie_header_check() -> None:
    actor_cookie = f"xagent_actor_oauth_{'a' * 24}=proof; HttpOnly; Secure"

    assert auth_api.is_actor_oauth_cookie_header(actor_cookie)
    assert not auth_api.is_actor_oauth_cookie_header("session=proof; HttpOnly; Secure")


def test_actor_start_leaves_commit_to_caller(oauth_db) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    pending = User(username="unrelated", password_hash="hash")
    db.add(pending)

    response = _start(db, user, commit=False)

    assert response.status_code == 307
    assert pending.id is None

    db.commit()

    assert pending.id is not None


def test_actor_start_leaves_failed_flow_to_caller(oauth_db) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    pending = User(username="unrelated", password_hash="hash")
    db.add(pending)

    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider="custom",
        app_id="calendar",
        user=user,
        resource_owner_key=ACTOR_ALICE,
        db=db,
        db_provider=None,
    )

    assert response.status_code == 500
    assert pending in db

    db.commit()

    assert pending.id is not None


@pytest.mark.parametrize("active", [False, None])
def test_actor_start_requires_exact_active_personal_link(oauth_db, active) -> None:
    db, user = oauth_db
    _server, link = _catalog_link(db, user)
    if active is None:
        db.delete(link)
    else:
        link.is_active = active
    db.commit()

    with pytest.raises(ValueError, match="active personal"):
        _start(db, user)


def test_builtin_snapshot_avoids_repeat_queries(oauth_db, monkeypatch) -> None:
    db, user = oauth_db
    server, _link = _catalog_link(db, user)

    snapshot = mcp_apps.load_mcp_app_snapshot(db)
    monkeypatch.setattr(
        db,
        "query",
        lambda *_args, **_kwargs: pytest.fail(
            "snapshot validation queried the database"
        ),
    )

    resolved = mcp_apps.require_builtin_oauth_server_definition(
        db,
        app_id="calendar",
        provider="custom",
        snapshot=snapshot,
    )

    assert resolved is server


def test_actor_start_rejects_provider_catalog_mismatch(oauth_db) -> None:
    db, user = oauth_db
    _catalog_link(db, user)

    with pytest.raises(ValueError, match="provider"):
        getattr(auth_api, "start_builtin_oauth_for_resource_owner")(
            provider="wrong",
            app_id="calendar",
            user=user,
            resource_owner_key=ACTOR_ALICE,
            db=db,
            db_provider=_provider(),
        )


def test_actor_callback_claims_nonce_before_exchange_and_persists_exact_owner(
    oauth_db, monkeypatch
) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    db.add_all(
        [
            UserOAuth(
                user_id=user.id,
                provider="calendar",
                resource_owner_key=None,
                access_token="ordinary",
            ),
            UserOAuth(
                user_id=user.id,
                provider="calendar",
                resource_owner_key=ACTOR_BOB,
                access_token="bob",
            ),
        ]
    )
    db.commit()
    start = _start(db, user)
    cookie = _flow_cookie(start)[:2]
    post = _mock_exchange(monkeypatch)

    def assert_claimed(*args, **kwargs):
        flow_model = getattr(
            __import__("xagent.web.models", fromlist=["ActorOAuthFlowState"]),
            "ActorOAuthFlowState",
        )
        assert db.query(flow_model).count() == 0
        return _ProviderResponse(
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "scope": "profile.read",
            }
        )

    post.side_effect = assert_claimed
    side_effects = Mock(
        side_effect=AssertionError("ordinary callback work must be skipped")
    )
    monkeypatch.setattr(auth_api, "_run_post_commit_oauth_side_effects", side_effects)

    response = generic_oauth_callback(
        "custom", _request(_state(start), cookie=cookie), db, _provider()
    )

    assert response.status_code == 200
    rows = {
        (row.resource_owner_key, row.access_token) for row in db.query(UserOAuth).all()
    }
    assert rows == {(None, "ordinary"), (ACTOR_BOB, "bob"), (ACTOR_ALICE, "new-access")}
    side_effects.assert_not_called()


def test_actor_callback_keeps_canonical_nonowning_link(oauth_db, monkeypatch) -> None:
    db, user = oauth_db
    server, link = _catalog_link(db, user, server_name="calendar")
    start = _start(db, user)
    _mock_exchange(monkeypatch)

    response = generic_oauth_callback(
        "custom",
        _request(_state(start), cookie=_flow_cookie(start)[:2]),
        db,
        _provider(),
    )

    assert response.status_code == 200
    assert db.query(MCPServer).all() == [server]
    assert db.query(UserMCPServer).all() == [link]
    assert link.is_owner is False


def test_actor_callback_locks_personal_link_before_persist(
    oauth_db, monkeypatch
) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    start = _start(db, user)
    _mock_exchange(monkeypatch)
    require_link = auth_api._require_actor_oauth_personal_link
    lock_link = auth_api._lock_actor_link
    checks: list[str] = []

    def record_require(*args, **kwargs) -> None:
        checks.append("require")
        require_link(*args, **kwargs)

    def record_lock(*args, **kwargs) -> None:
        checks.append("lock")
        lock_link(*args, **kwargs)

    monkeypatch.setattr(auth_api, "_require_actor_oauth_personal_link", record_require)
    monkeypatch.setattr(auth_api, "_lock_actor_link", record_lock)

    response = generic_oauth_callback(
        "custom",
        _request(_state(start), cookie=_flow_cookie(start)[:2]),
        db,
        _provider(),
    )

    assert response.status_code == 200
    assert checks == ["require", "lock"]


@pytest.mark.parametrize("cookie_mode", ["missing", "wrong"])
def test_actor_callback_rejects_missing_or_wrong_cookie_before_exchange(
    oauth_db, monkeypatch, cookie_mode
) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    start = _start(db, user)
    cookie_name, cookie_value = _flow_cookie(start)[:2]
    cookie = None if cookie_mode == "missing" else (cookie_name, cookie_value + "wrong")
    post = Mock()
    monkeypatch.setattr(auth_api.requests, "post", post)

    response = generic_oauth_callback(
        "custom", _request(_state(start), cookie=cookie), db, _provider()
    )

    assert response.status_code == 400
    post.assert_not_called()


def test_actor_provider_error_consumes_flow_without_exchange(
    oauth_db, monkeypatch
) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    start = _start(db, user)
    state = _state(start)
    cookie = _flow_cookie(start)[:2]
    post = Mock()
    monkeypatch.setattr(auth_api.requests, "post", post)

    denied = generic_oauth_callback(
        "custom", _error_request(state, cookie), db, _provider()
    )
    replay = generic_oauth_callback(
        "custom", _request(state, cookie=cookie), db, _provider()
    )

    assert denied.status_code == 400
    assert replay.status_code == 400
    post.assert_not_called()
    flow_model = getattr(
        __import__("xagent.web.models", fromlist=["ActorOAuthFlowState"]),
        "ActorOAuthFlowState",
    )
    assert db.query(flow_model).count() == 0
    assert db.query(UserOAuth).count() == 0


def test_actor_callback_replay_fails_before_second_exchange(
    oauth_db, monkeypatch
) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    start = _start(db, user)
    request = _request(_state(start), cookie=_flow_cookie(start)[:2])
    post = _mock_exchange(monkeypatch)

    first = generic_oauth_callback("custom", request, db, _provider())
    second = generic_oauth_callback("custom", request, db, _provider())

    assert first.status_code == 200
    assert second.status_code == 400
    assert post.call_count == 1


def test_actor_callback_rejects_expired_nonce_before_exchange(
    oauth_db, monkeypatch
) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    start = _start(db, user)
    flow_model = getattr(
        __import__("xagent.web.models", fromlist=["ActorOAuthFlowState"]),
        "ActorOAuthFlowState",
    )
    db.query(flow_model).update(
        {flow_model.expires_at: datetime.now(timezone.utc) - timedelta(seconds=1)}
    )
    db.commit()
    post = Mock()
    monkeypatch.setattr(auth_api.requests, "post", post)

    response = generic_oauth_callback(
        "custom",
        _request(_state(start), cookie=_flow_cookie(start)[:2]),
        db,
        _provider(),
    )

    assert response.status_code == 400
    post.assert_not_called()


@pytest.mark.parametrize("drift", ["removed-link", "catalog-provider", "server-auth"])
def test_actor_callback_revalidates_link_and_catalog_before_exchange(
    oauth_db, monkeypatch, drift
) -> None:
    db, user = oauth_db
    server, link = _catalog_link(db, user)
    start = _start(db, user)
    if drift == "removed-link":
        db.delete(link)
    elif drift == "catalog-provider":
        db.query(PublicMCPApp).filter_by(
            app_id="calendar"
        ).one().provider_name = "wrong"
    else:
        server.auth = {"app_id": "other", "provider": "custom"}
    db.commit()
    post = Mock()
    monkeypatch.setattr(auth_api.requests, "post", post)

    response = generic_oauth_callback(
        "custom",
        _request(_state(start), cookie=_flow_cookie(start)[:2]),
        db,
        _provider(),
    )

    assert response.status_code == 400
    post.assert_not_called()
    assert db.query(UserOAuth).count() == 0


def test_actor_owner_ciphertext_round_trips_exact_namespace(
    oauth_db, monkeypatch
) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    ciphertext_owner = encrypt_value(ACTOR_BOB)
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="calendar",
            resource_owner_key=ACTOR_BOB,
            access_token="bob",
        )
    )
    db.commit()

    start = _start(db, user, owner=ciphertext_owner)
    payload = auth_api.verify_token(_state(start))
    assert payload is not None
    assert payload["resource_owner_key"] != ciphertext_owner
    _mock_exchange(monkeypatch)

    response = generic_oauth_callback(
        "custom",
        _request(_state(start), cookie=_flow_cookie(start)[:2]),
        db,
        _provider(),
    )

    assert response.status_code == 200
    credentials = {
        row.resource_owner_key: row.access_token for row in db.query(UserOAuth).all()
    }
    assert credentials == {
        ACTOR_BOB: "bob",
        ciphertext_owner: "new-access",
    }


def test_actor_callback_rejects_tampered_state_before_exchange(
    oauth_db, monkeypatch
) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    start = _start(db, user)
    state = _state(start)
    header, payload, signature = state.split(".")
    tampered_signature = ("a" if signature[0] != "a" else "b") + signature[1:]
    tampered = ".".join((header, payload, tampered_signature))
    assert auth_api.verify_token(tampered) is None
    post = Mock()
    monkeypatch.setattr(auth_api.requests, "post", post)

    response = generic_oauth_callback(
        "custom", _request(tampered, cookie=_flow_cookie(start)[:2]), db, _provider()
    )

    assert response.status_code == 400
    post.assert_not_called()


@pytest.mark.parametrize(
    "owner_claim_kind", ["plaintext", "plaintext-envelope", "foreign-ciphertext"]
)
def test_actor_callback_rejects_unreadable_owner_claim_before_exchange(
    oauth_db, monkeypatch, owner_claim_kind: str
) -> None:
    db, user = oauth_db
    _catalog_link(db, user)
    start = _start(db, user)
    payload = auth_api.verify_token(_state(start))
    assert payload is not None
    owner_claim = ACTOR_ALICE
    if owner_claim_kind == "plaintext-envelope":
        owner_claim = json.dumps({"owner": ACTOR_ALICE, "version": 1})
    elif owner_claim_kind == "foreign-ciphertext":
        owner_claim = (
            Fernet(Fernet.generate_key()).encrypt(ACTOR_ALICE.encode()).decode()
        )
    payload["resource_owner_key"] = owner_claim
    state = create_access_token(data=payload, expires_delta=timedelta(minutes=10))
    post = Mock()
    monkeypatch.setattr(auth_api.requests, "post", post)

    response = generic_oauth_callback(
        "custom", _request(state, cookie=_flow_cookie(start)[:2]), db, _provider()
    )

    assert response.status_code == 400
    post.assert_not_called()


def test_ordinary_oauth_flow_keeps_cookie_free_state_and_provisioning(
    oauth_db, monkeypatch
) -> None:
    db, user = oauth_db
    token = create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )
    start = generic_oauth_login(
        "custom", token=token, app_id="calendar", db=db, db_provider=_provider()
    )
    payload = auth_api.verify_token(_state(start))
    assert payload is not None
    assert "resource_owner_key" not in payload
    assert "actor_flow_nonce" not in payload
    assert "set-cookie" not in start.headers
    _mock_exchange(monkeypatch)
    side_effects = Mock()
    monkeypatch.setattr(auth_api, "_run_post_commit_oauth_side_effects", side_effects)

    response = generic_oauth_callback(
        "custom", _request(_state(start)), db, _provider()
    )

    assert response.status_code == 200
    assert db.query(UserOAuth).one().resource_owner_key is None
    side_effects.assert_called_once_with(db, user_id=user.id, connector_key="calendar")
