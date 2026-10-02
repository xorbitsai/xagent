"""Environment policy for Google review scopes, without provider requests."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from xagent.web import mcp_apps
from xagent.web.api import auth as auth_api
from xagent.web.api.mcp import list_mcp_apps
from xagent.web.builtin_mcp_registry import (
    get_builtin_public_mcp_app,
    sync_google_scope_policy,
)
from xagent.web.models.database import (
    Base,
    _initialize_database_schema,
    get_engine,
    init_db,
)
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth

FLAG = "XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED"
DRIVE = "https://www.googleapis.com/auth/drive"
DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"
GMAIL = "https://www.googleapis.com/auth/gmail.modify"
ACTOR_OWNER = "toby:slack:41:UALICE"


@pytest.fixture
def google_db(tmp_path, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv(
        "GOOGLE_REDIRECT_URI", "https://app.example/api/auth/google/callback"
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'google.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(username="alice", password_hash="hash")
        db.add(user)
        db.flush()
        for app_id in ("gmail", "google-drive", "google-calendar", "google-docs"):
            row = get_builtin_public_mcp_app(app_id)
            assert row is not None
            db.add(PublicMCPApp(**row))
            server = MCPServer(
                name=row["name"],
                managed="external",
                transport="oauth",
                auth={"app_id": app_id, "provider": "google"},
            )
            db.add(server)
            db.flush()
            db.add(
                UserMCPServer(
                    user_id=user.id,
                    mcpserver_id=server.id,
                    is_active=True,
                    is_owner=False,
                )
            )
        db.commit()
        yield db, user
    engine.dispose()


def _provider():
    return SimpleNamespace(
        client_id="test-client",
        client_secret="test-secret",
        auth_url="https://accounts.google.com/o/oauth2/auth",
        token_url="https://oauth2.googleapis.com/token",
        redirect_uri="https://app.example/api/auth/google/callback",
        default_scopes=["openid", "email"],
    )


def _login(db, user, app_id, flow, provider=None):
    provider = provider or _provider()
    if flow == "actor":
        return auth_api.start_builtin_oauth_for_resource_owner(
            provider="google",
            app_id=app_id,
            user=user,
            resource_owner_key=ACTOR_OWNER,
            redirect=None,
            db=db,
            db_provider=provider,
        )
    return auth_api.generic_oauth_login(
        "google",
        token=auth_api.create_access_token(data={"sub": user.username}),
        app_id=app_id,
        db=db,
        db_provider=provider,
    )


@pytest.mark.parametrize("value", [None, "false", "true"])
def test_registry_scope_policy(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    enabled = value == "true"
    drive = get_builtin_public_mcp_app("google-drive")
    gmail = get_builtin_public_mcp_app("gmail")
    assert drive["oauth_scopes"] == [DRIVE if enabled else DRIVE_FILE]
    assert gmail["is_visible_in_connector"] is enabled
    assert gmail["oauth_scopes"] == ([GMAIL] if enabled else [])
    assert drive["is_visible_in_connector"] is enabled


def test_production_catalog_closed(monkeypatch):
    monkeypatch.setenv(FLAG, "false")
    gmail = get_builtin_public_mcp_app("gmail")
    drive = get_builtin_public_mcp_app("google-drive")
    assert gmail["oauth_scopes"] == []
    assert gmail["is_visible_in_connector"] is False
    assert drive["oauth_scopes"] == [DRIVE_FILE]
    assert drive["is_visible_in_connector"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_sync_google_catalog_rows(google_db, monkeypatch, enabled):
    db, _user = google_db
    gmail = db.query(PublicMCPApp).filter_by(app_id="gmail").one()
    drive = db.query(PublicMCPApp).filter_by(app_id="google-drive").one()
    # Model both the narrowed main rows and a previous enabled deployment.
    gmail.oauth_scopes = [] if enabled else [GMAIL]
    gmail.is_visible_in_connector = not enabled
    drive.oauth_scopes = [DRIVE_FILE if enabled else DRIVE]
    drive.is_visible_in_connector = not enabled
    db.commit()
    monkeypatch.setenv(FLAG, str(enabled))

    with db.get_bind().begin() as connection:
        sync_google_scope_policy(connection)
    db.expire_all()

    assert gmail.oauth_scopes == ([GMAIL] if enabled else [])
    assert gmail.is_visible_in_connector is enabled
    assert drive.oauth_scopes == [DRIVE if enabled else DRIVE_FILE]
    assert drive.is_visible_in_connector is True


@pytest.mark.parametrize("value", [None, "false"])
@pytest.mark.parametrize("visible", [False, True])
def test_keep_drive_visibility(tmp_path, monkeypatch, value, visible):
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    init_db(db_url=f"sqlite:///{tmp_path / 'existing.db'}")
    engine = get_engine()
    with Session(engine) as db:
        drive = db.query(PublicMCPApp).filter_by(app_id="google-drive").one()
        drive.is_visible_in_connector = visible
        db.commit()

    # SG exposes drive.file today; startup must not hide that connector.
    assert _initialize_database_schema(engine) == []
    with Session(engine) as db:
        drive = db.query(PublicMCPApp).filter_by(app_id="google-drive").one()
        assert drive.is_visible_in_connector is visible
        assert drive.oauth_scopes == [DRIVE_FILE]


@pytest.mark.parametrize("flow", ["ordinary", "actor"])
def test_sg_drive_connect(google_db, flow):
    db, user = google_db
    drive = db.query(PublicMCPApp).filter_by(app_id="google-drive").one()
    drive.is_visible_in_connector = True
    oauth = UserOAuth(
        user_id=user.id,
        provider="google-drive",
        access_token="existing-access",
        refresh_token="existing-refresh",
        scope=DRIVE,
        resource_owner_key=ACTOR_OWNER if flow == "actor" else None,
    )
    db.add(oauth)
    db.commit()
    connections = db.query(UserMCPServer).count()

    with db.get_bind().begin() as connection:
        sync_google_scope_policy(connection)
    db.expire_all()

    apps = list_mcp_apps(
        search=None,
        category="All",
        location="remote",
        status="all",
        current_user=user,
        db=db,
    )
    assert any(app["id"] == "google-drive" for app in apps)
    response = _login(db, user, "google-drive", flow)
    assert response.status_code == 307
    params = parse_qs(urlparse(response.headers["location"]).query)
    assert set(params["scope"][0].split()) == {"openid", "email", DRIVE_FILE}
    assert params["include_granted_scopes"] == ["false"]
    assert oauth.access_token == "existing-access"
    assert oauth.refresh_token == "existing-refresh"
    assert oauth.scope == DRIVE
    assert db.query(UserMCPServer).count() == connections


def test_gmail_overlay_blocks_db(google_db):
    db, _user = google_db
    gmail = db.query(PublicMCPApp).filter_by(app_id="gmail").one()
    gmail.is_visible_in_connector = True
    db.commit()
    assert mcp_apps.get_app_by_id(db, "gmail")["is_visible_in_connector"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_gmail_connector_list(google_db, monkeypatch, enabled):
    db, user = google_db
    monkeypatch.setenv(FLAG, str(enabled))
    gmail = db.query(PublicMCPApp).filter_by(app_id="gmail").one()
    gmail.is_visible_in_connector = True
    db.commit()

    apps = list_mcp_apps(
        search=None,
        category="All",
        location="remote",
        status="all",
        current_user=user,
        db=db,
    )
    assert any(app["id"] == "gmail" for app in apps) is enabled


def test_gmail_login_missing_row(google_db):
    db, user = google_db
    db.delete(db.query(PublicMCPApp).filter_by(app_id="gmail").one())
    db.commit()
    response = _login(db, user, "gmail", "ordinary")
    assert response.status_code == 404
    assert "location" not in response.headers


@pytest.mark.parametrize("scope", [DRIVE, GMAIL])
def test_reject_optional_scopes(google_db, monkeypatch, scope):
    db, user = google_db
    lookup = mcp_apps.get_builtin_execution_fields_and_optional_scopes

    def with_optional(app_id):
        fields, optional = lookup(app_id)
        return fields, [scope] if app_id == "google-calendar" else optional

    monkeypatch.setattr(
        mcp_apps, "get_builtin_execution_fields_and_optional_scopes", with_optional
    )
    response = _login(db, user, "google-calendar", "ordinary")
    assert response.status_code == 403
    assert "location" not in response.headers


@pytest.mark.parametrize("flow", ["ordinary", "actor"])
@pytest.mark.parametrize("enabled", [False, True])
def test_drive_redirect_scopes(google_db, monkeypatch, flow, enabled):
    db, user = google_db
    monkeypatch.setenv(FLAG, str(enabled))
    expected_scope = DRIVE if enabled else DRIVE_FILE
    drive = db.query(PublicMCPApp).filter_by(app_id="google-drive").one()
    drive.oauth_scopes = [expected_scope]
    # A manually exposed narrow connector must still request only drive.file.
    drive.is_visible_in_connector = True
    db.commit()
    response = _login(db, user, "google-drive", flow)
    assert response.status_code == 307
    params = parse_qs(urlparse(response.headers["location"]).query)
    assert set(params["scope"][0].split()) == {"openid", "email", expected_scope}
    assert params["include_granted_scopes"] == ["true" if enabled else "false"]


@pytest.mark.parametrize("flow", ["ordinary", "actor"])
def test_gmail_login_gate(google_db, monkeypatch, flow):
    db, user = google_db
    gmail = db.query(PublicMCPApp).filter_by(app_id="gmail").one()
    gmail.is_visible_in_connector = True
    db.commit()
    response = _login(db, user, "gmail", flow)
    assert response.status_code == 404
    assert "location" not in response.headers

    monkeypatch.setenv(FLAG, "true")
    gmail.oauth_scopes = [GMAIL]
    db.commit()
    response = _login(db, user, "gmail", flow)
    assert response.status_code == 307
    params = parse_qs(urlparse(response.headers["location"]).query)
    assert GMAIL in params["scope"][0].split()


@pytest.mark.parametrize("scope", [DRIVE, GMAIL])
@pytest.mark.parametrize("app_id", [None, "google-calendar", "custom-google"])
def test_reject_provider_scopes(google_db, scope, app_id):
    db, user = google_db
    provider = _provider()
    provider.default_scopes.append(scope)
    response = _login(db, user, app_id, "ordinary", provider)
    assert response.status_code == 403
    assert "location" not in response.headers


@pytest.mark.parametrize("scope", [DRIVE, GMAIL])
def test_reject_custom_scopes(google_db, scope):
    db, user = google_db
    db.add(
        PublicMCPApp(
            app_id="custom-google",
            name="Custom Google",
            transport="oauth",
            provider_name="google",
            oauth_scopes=[scope],
            is_visible_in_connector=True,
        )
    )
    db.commit()
    response = _login(db, user, "custom-google", "ordinary")
    assert response.status_code == 403
    assert "location" not in response.headers


@pytest.mark.parametrize("app_id", [None, "google-calendar", "google-docs"])
def test_other_google_scopes(google_db, app_id):
    db, user = google_db
    response = _login(db, user, app_id, "ordinary")
    assert response.status_code == 307
    params = parse_qs(urlparse(response.headers["location"]).query)
    scopes = set(params["scope"][0].split())
    assert not scopes.intersection({DRIVE, GMAIL})
    assert {"openid", "email"}.issubset(scopes)


@pytest.mark.parametrize("has_catalog_row", [False, True])
def test_gmail_callback_gate(google_db, monkeypatch, has_catalog_row):
    db, user = google_db
    gmail = db.query(PublicMCPApp).filter_by(app_id="gmail").one()
    if has_catalog_row:
        gmail.is_visible_in_connector = True
    else:
        db.delete(gmail)
    db.commit()
    state = auth_api.create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": "gmail",
        },
        expires_delta=timedelta(minutes=10),
    )
    post = Mock(side_effect=AssertionError("Google must not be contacted"))
    monkeypatch.setattr(auth_api.requests, "post", post)
    request = SimpleNamespace(query_params={"code": "code", "state": state})
    response = auth_api.generic_oauth_callback("google", request, db, _provider())
    assert response.status_code == 404
    post.assert_not_called()
    assert db.query(UserOAuth).count() == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_startup_sync_and_revert(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv(FLAG, str(enabled))
    init_db(db_url=f"sqlite:///{tmp_path / 'startup.db'}")
    engine = get_engine()
    with Session(engine) as db:
        drive = db.query(PublicMCPApp).filter_by(app_id="google-drive").one()
        gmail = db.query(PublicMCPApp).filter_by(app_id="gmail").one()
        assert drive.oauth_scopes == [DRIVE if enabled else DRIVE_FILE]
        assert drive.is_visible_in_connector is enabled
        assert gmail.oauth_scopes == ([GMAIL] if enabled else [])
        assert gmail.is_visible_in_connector is enabled
        drive.description = "Keep this description"
        drive.is_visible_in_connector = False
        db.add(PublicMCPApp(app_id="custom-app", name="Custom", oauth_scopes=[DRIVE]))
        db.commit()
        generation = drive.generation

    monkeypatch.setenv(FLAG, str(not enabled))
    assert _initialize_database_schema(engine) == []
    with Session(engine) as db:
        drive = db.query(PublicMCPApp).filter_by(app_id="google-drive").one()
        gmail = db.query(PublicMCPApp).filter_by(app_id="gmail").one()
        assert drive.oauth_scopes == [DRIVE_FILE if enabled else DRIVE]
        assert gmail.is_visible_in_connector is not enabled
        assert gmail.oauth_scopes == ([] if enabled else [GMAIL])
        assert drive.description == "Keep this description"
        assert drive.is_visible_in_connector is not enabled
        assert drive.generation == generation
        assert db.query(PublicMCPApp).filter_by(
            app_id="custom-app"
        ).one().oauth_scopes == [DRIVE]

    updates = []

    def record_updates(_conn, _cursor, statement, _params, _context, _many):
        if statement.upper().startswith("UPDATE PUBLIC_MCP_APPS"):
            updates.append(statement)

    event.listen(engine, "before_cursor_execute", record_updates)
    try:
        assert _initialize_database_schema(engine) == []
    finally:
        event.remove(engine, "before_cursor_execute", record_updates)
    assert updates == []
