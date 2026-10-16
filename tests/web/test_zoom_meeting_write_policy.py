"""Environment policy for the Zoom meeting:write:meeting scope, without Zoom
requests."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from xagent.web.api import auth as auth_api
from xagent.web.builtin_mcp_registry import (
    ZOOM_MEETING_WRITE_DESCRIPTION,
    ZOOM_READ_ONLY_DESCRIPTION,
    get_builtin_public_mcp_app,
    sync_zoom_meeting_write_policy,
)
from xagent.web.models.database import (
    Base,
    _initialize_database_schema,
    get_engine,
    init_db,
)
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User

FLAG = "XAGENT_ZOOM_MEETING_WRITE_ENABLED"
WRITE_SCOPE = "meeting:write:meeting"
READ_SCOPES = [
    "meeting:read:meeting",
    "meeting:read:list_meetings",
    "meeting:read:past_meeting",
    "cloud_recording:read:list_recording_files",
    "cloud_recording:read:meeting_transcript",
    "user:read:user",
]


@pytest.fixture(autouse=True)
def _flag_unset(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


def _set_flag(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)


def _zoom_row(monkeypatch, enabled):
    """The registry's Zoom row as it is with the flag on or off."""
    _set_flag(monkeypatch, "true" if enabled else "false")
    row = get_builtin_public_mcp_app("zoom")
    assert row is not None
    return row


def _load_seed_migration():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260730_seed_zoom_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location("seed_zoom_migration", migration_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", [None, "false", "", "invalid", "true", "1", "ON"])
def test_registry_zoom_policy(monkeypatch, value):
    _set_flag(monkeypatch, value)
    enabled = value in {"true", "1", "ON"}

    zoom = get_builtin_public_mcp_app("zoom")

    assert (WRITE_SCOPE in zoom["oauth_scopes"]) is enabled
    assert [scope for scope in zoom["oauth_scopes"] if scope != WRITE_SCOPE] == (
        READ_SCOPES
    )
    assert "meeting:write:meeting:admin" not in zoom["oauth_scopes"]
    assert zoom["description"] == (
        ZOOM_MEETING_WRITE_DESCRIPTION if enabled else ZOOM_READ_ONLY_DESCRIPTION
    )
    # Only an enabled deployment forwards the flag to the MCP subprocess,
    # which is what makes it register zoom_create_meeting.
    assert zoom["launch_config"].get("static_env") == (
        {FLAG: FLAG} if enabled else None
    )


@pytest.mark.parametrize("value", [None, "false"])
def test_disabled_zoom_row_is_the_frozen_seed_row(monkeypatch, value):
    """With the flag off, the catalog row is exactly what it was before
    meeting creation existed, so an upgrade changes nothing for Zoom."""
    _set_flag(monkeypatch, value)

    assert _load_seed_migration()._zoom_app_row() == get_builtin_public_mcp_app("zoom")


@pytest.fixture
def zoom_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'zoom.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(username="alice", password_hash="hash")
        db.add(user)
        db.commit()
        yield db, user
    engine.dispose()


def _add_zoom_row(db, row, **overrides):
    app = PublicMCPApp(**{**row, **overrides})
    db.add(app)
    db.commit()
    return app


@pytest.mark.parametrize("enabled", [False, True])
def test_sync_zoom_catalog_row(zoom_db, monkeypatch, enabled):
    db, _user = zoom_db
    # A row written under the other setting, as after flipping the flag.
    app = _add_zoom_row(db, _zoom_row(monkeypatch, not enabled))
    expected = _zoom_row(monkeypatch, enabled)

    with db.get_bind().begin() as connection:
        sync_zoom_meeting_write_policy(connection)
    db.expire_all()

    assert app.oauth_scopes == expected["oauth_scopes"]
    assert app.launch_config == expected["launch_config"]
    assert app.description == expected["description"]


@pytest.mark.parametrize("enabled", [False, True])
def test_sync_keeps_a_customized_zoom_description(zoom_db, monkeypatch, enabled):
    db, _user = zoom_db
    app = _add_zoom_row(
        db, _zoom_row(monkeypatch, not enabled), description="Our Zoom workspace"
    )
    expected = _zoom_row(monkeypatch, enabled)

    with db.get_bind().begin() as connection:
        sync_zoom_meeting_write_policy(connection)
    db.expire_all()

    assert app.description == "Our Zoom workspace"
    assert app.oauth_scopes == expected["oauth_scopes"]


def test_sync_without_a_zoom_row_is_a_no_op(zoom_db, monkeypatch):
    db, _user = zoom_db
    monkeypatch.setenv(FLAG, "true")

    with db.get_bind().begin() as connection:
        sync_zoom_meeting_write_policy(connection)

    assert db.query(PublicMCPApp).count() == 0


def _zoom_provider():
    return SimpleNamespace(
        client_id="test-client",
        client_secret="test-secret",
        auth_url="https://zoom.us/oauth/authorize",
        token_url="https://zoom.us/oauth/token",
        redirect_uri="https://app.example/api/auth/zoom/callback",
        default_scopes=[],
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_zoom_authorize_requests_write_scope_only_when_enabled(
    zoom_db, monkeypatch, enabled
):
    """The authorize request follows the flag, not a stale persisted row."""
    db, user = zoom_db
    _add_zoom_row(db, _zoom_row(monkeypatch, not enabled))
    _set_flag(monkeypatch, "true" if enabled else "false")

    response = auth_api.generic_oauth_login(
        "zoom",
        token=auth_api.create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id,
                "session_epoch": user.session_epoch,
            }
        ),
        app_id="zoom",
        db=db,
        db_provider=_zoom_provider(),
    )

    assert response.status_code == 307
    params = parse_qs(urlparse(response.headers["location"]).query)
    expected = set(READ_SCOPES) | ({WRITE_SCOPE} if enabled else set())
    assert set(params["scope"][0].split()) == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_startup_sync_and_revert(tmp_path, monkeypatch, enabled):
    _set_flag(monkeypatch, str(enabled).lower())
    init_db(db_url=f"sqlite:///{tmp_path / 'startup.db'}")
    engine = get_engine()
    with Session(engine) as db:
        zoom = db.query(PublicMCPApp).filter_by(app_id="zoom").one()
        assert (WRITE_SCOPE in zoom.oauth_scopes) is enabled
        assert zoom.description == (
            ZOOM_MEETING_WRITE_DESCRIPTION if enabled else ZOOM_READ_ONLY_DESCRIPTION
        )
        generation = zoom.generation

    _set_flag(monkeypatch, str(not enabled).lower())
    # No drift report: the synced row matches the registry for the new value.
    assert _initialize_database_schema(engine) == []
    with Session(engine) as db:
        zoom = db.query(PublicMCPApp).filter_by(app_id="zoom").one()
        expected = get_builtin_public_mcp_app("zoom")
        assert (WRITE_SCOPE in zoom.oauth_scopes) is not enabled
        assert zoom.oauth_scopes == expected["oauth_scopes"]
        assert zoom.launch_config == expected["launch_config"]
        assert zoom.description == expected["description"]
        assert zoom.generation == generation

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
