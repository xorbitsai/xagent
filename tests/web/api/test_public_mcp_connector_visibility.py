import os
import re
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import xagent.web.api.mcp as mcp_api
from xagent.web.api.admin_mcp import (
    PublicMCPAppCreate,
    PublicMCPAppUpdate,
    _commit_public_mcp_app_write,
    admin_mcp_router,
)
from xagent.web.api.auth import (
    AppNotOAuthError,
    _ensure_user_mcp_server,
    auth_router,
)
from xagent.web.api.mcp import mcp_router
from xagent.web.models.database import Base, get_db, get_engine
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp, PublicMCPAppAudit
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth


def override_get_db():
    db = None
    try:
        db = next(get_db())
        yield db
    finally:
        if db is not None:
            db.close()


app_for_tests = FastAPI()
app_for_tests.include_router(auth_router)
app_for_tests.include_router(mcp_router)
app_for_tests.include_router(admin_mcp_router)
app_for_tests.dependency_overrides[get_db] = override_get_db
client = TestClient(app_for_tests)


def test_admin_catalog_openapi_documents_launch_config_and_update_semantics() -> None:
    create_launch_config = PublicMCPAppCreate.model_json_schema()["properties"][
        "launch_config"
    ]
    update_launch_config = PublicMCPAppUpdate.model_json_schema()["properties"][
        "launch_config"
    ]

    for field_schema in (create_launch_config, update_launch_config):
        description = field_schema.get("description", "").lower()
        assert "credentials or secret values" in description
        assert "connector credential flow" in description

    operations = app_for_tests.openapi()["paths"]["/api/admin/mcp/apps/{app_id}"]
    assert "full replacement" in operations["put"]["description"].lower()
    assert "use patch" in operations["put"]["description"].lower()
    assert "partial update" in operations["patch"]["description"].lower()
    assert "presentation fields" in operations["patch"]["description"].lower()


def _setup_test_db() -> str:
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{temp_db_path}"
    init_db(db_url=db_url)
    return temp_dir


def _setup_admin() -> None:
    status_response = client.get("/api/auth/setup-status")
    assert status_response.status_code == 200
    if status_response.json().get("needs_setup", True):
        setup_response = client.post(
            "/api/auth/setup-admin",
            json={
                "username": "admin",
                "email": "admin@example.com",
                "password": "admin123",
            },
        )
        assert setup_response.status_code == 200


def _login(username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_public_mcp_write_commit_rolls_back_and_preserves_failure() -> None:
    db = Mock()
    failure = RuntimeError("commit failed")
    db.commit.side_effect = failure

    with pytest.raises(RuntimeError) as caught:
        _commit_public_mcp_app_write(db)

    assert caught.value is failure
    db.rollback.assert_called_once_with()


def test_public_mcp_create_commit_maps_integrity_error_after_rollback() -> None:
    db = Mock()
    db.commit.side_effect = IntegrityError(
        "INSERT INTO public_mcp_apps ...",
        {},
        RuntimeError("duplicate app_id"),
    )

    with pytest.raises(HTTPException) as caught:
        _commit_public_mcp_app_write(db, integrity_error_detail="App already exists")

    assert caught.value.status_code == 400
    assert caught.value.detail == "App already exists"
    db.rollback.assert_called_once_with()


def _create_public_app(
    headers: dict[str, str],
    app_id: str,
    name: str,
    is_visible_in_connector: bool,
    transport: str = "oauth",
) -> None:
    response = client.post(
        "/api/admin/mcp/apps",
        headers=headers,
        json={
            "app_id": app_id,
            "name": name,
            "description": f"{name} description",
            "icon": "",
            "transport": transport,
            "provider_name": None,
            "category": "Communication",
            "oauth_scopes": [],
            "is_visible_in_connector": is_visible_in_connector,
            "launch_config": {},
        },
    )
    assert response.status_code == 200


def _connect_app_for_user(username: str, server_name: str) -> None:
    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None

        server = MCPServer(
            name=server_name,
            description="connected hidden app",
            managed="external",
            transport="oauth",
        )
        db.add(server)
        db.flush()

        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.commit()
    finally:
        db.close()


def _connect_custom_stdio_mcp_for_user(username: str, server_name: str) -> None:
    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None

        server = MCPServer(
            name=server_name,
            description="custom stdio connector",
            managed="external",
            transport="stdio",
            command="npx",
            args=["-y", "@floriscornel/teams-mcp@latest"],
        )
        db.add(server)
        db.flush()

        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.commit()
    finally:
        db.close()


def _connect_oauth_account_for_user(username: str, provider: str) -> None:
    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None

        db.add(
            UserOAuth(
                user_id=user.id,
                provider=provider,
                access_token="access-token",
                provider_user_id=f"{provider}-user",
                email=f"{provider}@example.com",
            )
        )
        db.commit()
    finally:
        db.close()


def test_connected_non_oauth_public_app_is_marked_connected() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        _create_public_app(
            admin_headers,
            "public-stdio",
            "Public Stdio",
            True,
            transport="stdio",
        )
        _connect_custom_stdio_mcp_for_user("regular", "Public Stdio")

        response = client.get(
            "/api/mcp/apps?location=remote&search=public",
            headers=regular_headers,
        )
        assert response.status_code == 200

        public_app = next(app for app in response.json() if app["id"] == "public-stdio")
        assert public_app["is_connected"] is True
        assert isinstance(public_app["server_id"], int)
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_local_custom_mcp_app_is_a_self_contained_actor_scoped_edit_reference() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")
        _connect_custom_stdio_mcp_for_user("regular", "Private Local MCP")

        response = client.get(
            "/api/mcp/apps?location=local&search=private",
            headers=regular_headers,
        )
        assert response.status_code == 200, response.text
        local_app = next(
            app for app in response.json() if app["id"] == "Private Local MCP"
        )
        assert isinstance(local_app["server_id"], int)
        assert local_app["transport"] == "stdio"
        assert local_app["is_custom"] is True
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_non_oauth_public_app_matches_space_hyphen_name_variant() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        _create_public_app(
            admin_headers,
            "space-hyphen",
            "Different Display Name",
            True,
            transport="stdio",
        )
        _connect_custom_stdio_mcp_for_user("regular", "Space Hyphen")

        response = client.get(
            "/api/mcp/apps?location=remote&search=different",
            headers=regular_headers,
        )
        assert response.status_code == 200

        public_app = next(app for app in response.json() if app["id"] == "space-hyphen")
        assert public_app["is_connected"] is True
        assert isinstance(public_app["server_id"], int)
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_remote_connector_builds_oauth_connectability_once_per_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")
        _connect_oauth_account_for_user("regular", "microsoft")

        checked_providers: list[str] = []

        def count_connectability_check(oauth_account: object) -> bool:
            checked_providers.append(str(getattr(oauth_account, "provider")))
            return True

        monkeypatch.setattr(
            mcp_api, "_oauth_account_can_connect", count_connectability_check
        )

        response = client.get("/api/mcp/apps?location=remote", headers=regular_headers)
        assert response.status_code == 200

        assert checked_providers == ["microsoft"]
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_oauth_account_can_connect_with_sqlite_naive_utc_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is timezone.utc:
                return cls(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
            return cls(2026, 1, 1, 16, 0)

    monkeypatch.setattr(mcp_api, "datetime", FixedDateTime)

    oauth_account = SimpleNamespace(
        access_token="access-token",
        refresh_token=None,
        expires_at=FixedDateTime(2026, 1, 1, 9, 0),
    )

    assert mcp_api._oauth_account_can_connect(oauth_account) is True


def test_hidden_public_mcp_app_is_excluded_from_remote_connector_list() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        _create_public_app(admin_headers, "visible-app", "Visible App", True)
        _create_public_app(admin_headers, "hidden-app", "Hidden App", False)

        response = client.get("/api/mcp/apps?location=remote", headers=regular_headers)
        assert response.status_code == 200

        app_ids = {app["id"] for app in response.json()}
        assert "visible-app" in app_ids
        assert "hidden-app" not in app_ids
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_custom_stdio_mcp_with_same_name_does_not_mark_builtin_oauth_app_connected() -> (
    None
):
    temp_dir = _setup_test_db()
    try:
        _setup_admin()

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        _connect_custom_stdio_mcp_for_user("regular", "Teams")

        response = client.get(
            "/api/mcp/apps?location=remote&search=teams",
            headers=regular_headers,
        )
        assert response.status_code == 200

        teams_app = next(app for app in response.json() if app["id"] == "teams")
        assert teams_app["is_connected"] is False
        assert "server_id" not in teams_app
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_oauth_connection_does_not_reuse_same_name_custom_stdio_mcp() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        _connect_custom_stdio_mcp_for_user("regular", "Teams")

        db = next(get_db())
        try:
            user = db.query(User).filter(User.username == "regular").first()
            assert user is not None

            with pytest.raises(
                ValueError, match="conflicts with an existing MCP server"
            ) as exc:
                _ensure_user_mcp_server(
                    db,
                    str(user.id),
                    {
                        "id": "teams",
                        "name": "Teams",
                        "description": "Connect to Microsoft Teams.",
                        "provider": "microsoft",
                        "auth_type": "builtin_oauth",
                    },
                )
            # A genuine metadata conflict on a real OAuth app is a plain
            # ValueError, NOT AppNotOAuthError — so the batch loop's narrowed
            # except surfaces it instead of misreporting it as "non-oauth".
            assert not isinstance(exc.value, AppNotOAuthError)
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_init_db_seeds_builtin_oauth_and_microsoft_graph_public_apps() -> None:
    temp_dir = _setup_test_db()
    db = next(get_db())
    try:
        provider_names = {row.provider_name for row in db.query(OAuthProvider).all()}
        assert {"google", "linkedin", "microsoft", "meta", "hubspot"}.issubset(
            provider_names
        )

        microsoft_provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == "microsoft")
            .first()
        )
        assert microsoft_provider is not None
        assert microsoft_provider.default_scopes == ["User.Read"]
        meta_provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == "meta")
            .first()
        )
        assert meta_provider is not None
        assert meta_provider.default_scopes == ["public_profile"]

        app_ids = {row.app_id for row in db.query(PublicMCPApp).all()}
        assert {
            "linkedin",
            "gmail",
            "google-drive",
            "google-calendar",
            "teams",
            "outlook",
            "onedrive",
            "facebook",
            "instagram",
        }.issubset(app_ids)

        teams_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "teams").first()
        )
        outlook_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "outlook").first()
        )
        onedrive_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "onedrive").first()
        )
        facebook_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "facebook").first()
        )
        instagram_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "instagram").first()
        )

        assert teams_app is not None
        assert teams_app.provider_name == "microsoft"
        assert teams_app.oauth_scopes == [
            "Team.ReadBasic.All",
            "Channel.ReadBasic.All",
            "TeamMember.Read.All",
            "ChannelMessage.Read.All",
            "ChannelMessage.Send",
            "Chat.ReadWrite",
        ]
        assert teams_app.launch_config == {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.teams"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        }

        assert outlook_app is not None
        assert outlook_app.provider_name == "microsoft"
        assert outlook_app.oauth_scopes == [
            "Mail.Read",
            "Mail.Send",
            "Calendars.ReadWrite",
            "Contacts.Read",
        ]
        assert outlook_app.launch_config == {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.outlook"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        }

        assert onedrive_app is not None
        assert onedrive_app.provider_name == "microsoft"
        assert onedrive_app.oauth_scopes == ["Files.ReadWrite"]
        assert onedrive_app.launch_config == {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.onedrive"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        }

        assert facebook_app is not None
        assert facebook_app.provider_name == "meta"
        assert facebook_app.category == "Marketing"
        assert facebook_app.oauth_scopes == [
            "pages_show_list",
            "pages_read_engagement",
            "pages_manage_posts",
        ]
        assert facebook_app.launch_config == {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.facebook"],
            "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
        }

        assert instagram_app is not None
        assert instagram_app.provider_name == "meta"
        assert instagram_app.category == "Marketing"
        assert instagram_app.oauth_scopes == [
            "pages_show_list",
            "pages_read_engagement",
            "instagram_basic",
            "instagram_content_publish",
        ]
        assert instagram_app.launch_config == {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.instagram"],
            "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
        }
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_builtin_registry_uses_runtime_available_launch_commands() -> None:
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    expected_python_apps = {
        "linkedin": (
            "xagent.web.tools.mcp.linkedin",
            {"LINKEDIN_ACCESS_TOKEN": "access_token"},
        ),
        "gmail": (
            "xagent.web.tools.mcp.gmail",
            {"GOOGLE_ACCESS_TOKEN": "access_token"},
        ),
        "google-drive": (
            "xagent.web.tools.mcp.google_drive",
            {"GOOGLE_ACCESS_TOKEN": "access_token"},
        ),
        "google-calendar": (
            "xagent.web.tools.mcp.calendar",
            {"GOOGLE_ACCESS_TOKEN": "access_token"},
        ),
        "google-docs": (
            "xagent.web.tools.mcp.google_docs",
            {"GOOGLE_ACCESS_TOKEN": "access_token"},
        ),
        "google-slides": (
            "xagent.web.tools.mcp.google_slides",
            {"GOOGLE_ACCESS_TOKEN": "access_token"},
        ),
        "hubspot": (
            "xagent.web.tools.mcp.hubspot",
            {"HUBSPOT_ACCESS_TOKEN": "access_token"},
        ),
        "teams": (
            "xagent.web.tools.mcp.teams",
            {"AUTH_TOKEN": "access_token"},
        ),
        "outlook": (
            "xagent.web.tools.mcp.outlook",
            {"AUTH_TOKEN": "access_token"},
        ),
        "onedrive": (
            "xagent.web.tools.mcp.onedrive",
            {"AUTH_TOKEN": "access_token"},
        ),
        "facebook": (
            "xagent.web.tools.mcp.facebook",
            {"META_ACCESS_TOKEN": "access_token"},
        ),
        "instagram": (
            "xagent.web.tools.mcp.instagram",
            {"META_ACCESS_TOKEN": "access_token"},
        ),
    }
    rows_by_app_id = {row["app_id"]: row for row in get_builtin_public_mcp_app_rows()}

    for app_id, (module_name, env_mapping) in expected_python_apps.items():
        assert rows_by_app_id[app_id]["launch_config"] == {
            "command": "python",
            "args": ["-m", module_name],
            "env_mapping": env_mapping,
        }

    assert rows_by_app_id["google-maps"]["launch_config"] == {
        "command": "npx",
        "args": ["-y", "@cablate/mcp-google-map", "--stdio"],
        "required_env": ["GOOGLE_MAPS_API_KEY"],
    }


def test_builtin_registry_helpers_use_exact_ids_and_return_defensive_copies() -> None:
    from xagent.web.builtin_mcp_registry import (
        get_builtin_execution_fields,
        get_builtin_public_mcp_app,
        is_builtin_public_mcp_app,
    )

    expected_execution_fields = {
        "name": "Gmail",
        "transport": "oauth",
        "provider_name": "google",
        "oauth_scopes": ["https://www.googleapis.com/auth/gmail.modify"],
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.gmail"],
            "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
        },
    }

    app = get_builtin_public_mcp_app("gmail")
    execution_fields = get_builtin_execution_fields("gmail")

    assert app is not None
    assert app["app_id"] == "gmail"
    assert execution_fields == expected_execution_fields
    assert is_builtin_public_mcp_app("gmail") is True
    assert get_builtin_public_mcp_app("Gmail") is None
    assert get_builtin_public_mcp_app("gmail ") is None
    assert get_builtin_public_mcp_app("unknown-app") is None
    assert get_builtin_execution_fields("unknown-app") is None
    assert is_builtin_public_mcp_app("unknown-app") is False

    app["name"] = "Mutated Gmail"
    app["oauth_scopes"].append("mutated-scope")
    app["launch_config"]["args"].append("mutated-arg")
    assert execution_fields is not None
    execution_fields["oauth_scopes"].append("another-mutated-scope")
    execution_fields["launch_config"]["env_mapping"]["MUTATED"] = "token"

    fresh_app = get_builtin_public_mcp_app("gmail")
    assert fresh_app is not None
    assert fresh_app["name"] == "Gmail"
    assert fresh_app["oauth_scopes"] == expected_execution_fields["oauth_scopes"]
    assert fresh_app["launch_config"] == expected_execution_fields["launch_config"]
    assert get_builtin_execution_fields("gmail") == expected_execution_fields


def test_builtin_registry_drift_validation_reports_safe_read_only_summaries() -> None:
    from xagent.web.builtin_mcp_registry import validate_builtin_public_mcp_apps

    secret_marker = "secret-value-must-not-appear"
    temp_dir = _setup_test_db()
    db = next(get_db())
    try:
        gmail_app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
        gmail_app.name = "Stale Gmail"
        gmail_app.launch_config = {
            "command": secret_marker,
            "args": ["--token", secret_marker],
        }
        deleted_builtin_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "google-drive").one()
        )
        db.delete(deleted_builtin_app)
        custom_app = PublicMCPApp(
            app_id="Gmail",
            name="Custom Gmail Lookalike",
            transport="stdio",
            launch_config={"command": "custom-command"},
        )
        db.add(custom_app)
        db.commit()

        with get_engine().begin() as connection:
            mismatches = validate_builtin_public_mcp_apps(connection)

        assert len(mismatches) == 1
        mismatch = mismatches[0]
        assert mismatch["app_id"] == "gmail"
        assert mismatch["mismatched_fields"] == ["name", "launch_config"]
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", mismatch["canonical_hash"])
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", mismatch["persisted_hash"])
        assert mismatch["canonical_hash"] != mismatch["persisted_hash"]
        assert secret_marker not in repr(mismatches)
        assert "Gmail" not in repr(mismatches)
        assert "google-drive" not in repr(mismatches)

        db.expire_all()
        persisted_gmail = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
        )
        persisted_custom = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "Gmail").one()
        )
        assert persisted_gmail.name == "Stale Gmail"
        assert persisted_gmail.launch_config == {
            "command": secret_marker,
            "args": ["--token", secret_marker],
        }
        assert persisted_custom.launch_config == {"command": "custom-command"}
        assert (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "google-drive").first()
            is None
        )
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_builtin_registry_drift_validation_accepts_canonical_rows() -> None:
    from xagent.web.builtin_mcp_registry import validate_builtin_public_mcp_apps

    temp_dir = _setup_test_db()
    try:
        with get_engine().begin() as connection:
            assert validate_builtin_public_mcp_apps(connection) == []
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_init_db_logs_safe_builtin_registry_drift_without_repairing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from xagent.web.models.database import init_db

    secret_marker = "secret-value-must-not-be-logged"
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{temp_db_path}"
    init_db(db_url=db_url)

    db = next(get_db())
    try:
        gmail_app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
        gmail_app.launch_config = {
            "command": secret_marker,
            "args": ["--token", secret_marker],
        }
        db.commit()
    finally:
        db.close()

    caplog.clear()
    with caplog.at_level("WARNING", logger="xagent.web.models.database"):
        init_db(db_url=db_url)

    drift_records = [
        record
        for record in caplog.records
        if record.name == "xagent.web.models.database"
        and "Built-in MCP catalog drift detected" in record.getMessage()
    ]
    assert len(drift_records) == 1
    drift_message = drift_records[0].getMessage()
    assert "app_id=gmail" in drift_message
    assert "mismatched_fields=launch_config" in drift_message
    assert "canonical_hash=sha256:" in drift_message
    assert "persisted_hash=sha256:" in drift_message
    assert secret_marker not in drift_message

    db = next(get_db())
    try:
        persisted_gmail = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
        )
        assert persisted_gmail.launch_config == {
            "command": secret_marker,
            "args": ["--token", secret_marker],
        }
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_init_db_does_not_warn_when_builtin_registry_is_canonical(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.clear()
    with caplog.at_level("WARNING", logger="xagent.web.models.database"):
        temp_dir = _setup_test_db()
    try:
        assert not any(
            record.name == "xagent.web.models.database"
            and "Built-in MCP catalog drift detected" in record.getMessage()
            for record in caplog.records
        )
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_init_db_does_not_reseed_deleted_builtin_app_on_existing_database() -> None:
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{temp_db_path}"

    init_db(db_url=db_url)

    db = next(get_db())
    try:
        gmail_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").first()
        )
        assert gmail_app is not None
        db.delete(gmail_app)
        db.commit()
    finally:
        db.close()

    init_db(db_url=db_url)

    db = next(get_db())
    try:
        recreated_gmail_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").first()
        )
        assert recreated_gmail_app is None
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_connected_hidden_public_mcp_app_is_excluded_in_strong_hide_mode() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        _create_public_app(admin_headers, "hidden-app", "Hidden App", False)
        _connect_app_for_user("regular", "Hidden App")

        response = client.get("/api/mcp/apps?location=remote", headers=regular_headers)
        assert response.status_code == 200

        app_ids = {app["id"] for app in response.json()}
        assert "hidden-app" not in app_ids
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_mixed_case_oauth_transport_app_is_marked_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the mixed-case transport fix: list_mcp_apps now routes on
    auth_type (which lowercases), so a "OAuth"-cased catalog entry is treated as
    builtin_oauth. Before, the exact-case `transport == "oauth"` branch stranded
    it in the non-oauth path, leaving is_connected always False."""
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        # Admin API accepts arbitrary transport casing.
        resp = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json={
                "app_id": "mixed-oauth",
                "name": "MixedOauth",
                "description": "",
                "icon": "",
                "transport": "OAuth",
                "provider_name": "microsoft",
                "category": "Communication",
                "oauth_scopes": [],
                "is_visible_in_connector": True,
                "launch_config": {},
            },
        )
        assert resp.status_code == 200

        _connect_oauth_account_for_user("regular", "microsoft")
        monkeypatch.setattr(mcp_api, "_oauth_account_can_connect", lambda _a: True)

        db = next(get_db())
        try:
            user = db.query(User).filter(User.username == "regular").first()
            assert user is not None
            server = MCPServer(
                name="MixedOauth",
                description="",
                managed="external",
                transport="oauth",
                auth={"app_id": "mixed-oauth"},
            )
            db.add(server)
            db.flush()
            db.add(
                UserMCPServer(
                    user_id=user.id,
                    mcpserver_id=server.id,
                    is_owner=True,
                    can_edit=True,
                    can_delete=True,
                    is_active=True,
                )
            )
            db.commit()
        finally:
            db.close()

        response = client.get("/api/mcp/apps?location=remote", headers=regular_headers)
        assert response.status_code == 200
        app = next(a for a in response.json() if a["id"] == "mixed-oauth")
        assert app["is_connected"] is True
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_create_app_rejects_keyless_command_entry() -> None:
    """Write-time constraint (#764): a non-oauth entry with a launch command but
    no required_env would classify as "unconnectable", so the admin API rejects
    it instead of silently persisting an unconnectable row."""
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        # Shape 1: command without required_env.
        resp = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json={
                "app_id": "bad-keyless",
                "name": "BadKeyless",
                "transport": "stdio",
                "launch_config": {"command": "npx", "args": ["-y", "x"]},
            },
        )
        assert resp.status_code == 422

        # Shape 2 (the reverse asymmetric shape): required_env without command.
        resp = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json={
                "app_id": "bad-nocommand",
                "name": "BadNoCommand",
                "transport": "stdio",
                "launch_config": {"required_env": ["KEY"]},
            },
        )
        assert resp.status_code == 422
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_update_app_enforces_auth_classification() -> None:
    """The write-time constraint fires on PUT too, not just POST (both use the
    same PublicMCPAppCreate model)."""
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        created = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json={
                "app_id": "good-keyed",
                "name": "GoodKeyed",
                "transport": "stdio",
                "launch_config": {"command": "npx", "required_env": ["KEY"]},
            },
        )
        assert created.status_code == 200
        app_pk = created.json()["id"]

        updated = client.put(
            f"/api/admin/mcp/apps/{app_pk}",
            headers=admin_headers,
            json={
                "app_id": "good-keyed",
                "name": "GoodKeyed",
                "transport": "stdio",
                "launch_config": {"command": "npx"},
            },
        )
        assert updated.status_code == 422
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_list_apps_does_not_500_on_partial_launch_config_row() -> None:
    """The write-time validator lives on the create model only, so listing must
    not re-validate on response serialization. A legacy/direct-DB row with a
    partial launch_config (classifies "unconnectable") must be returned, not turn
    the whole admin list into a 500."""
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        db = next(get_db())
        try:
            db.add(
                PublicMCPApp(
                    app_id="legacy-bad",
                    name="LegacyBad",
                    transport="stdio",
                    launch_config={"command": "npx"},
                )
            )
            db.commit()
        finally:
            db.close()

        resp = client.get("/api/admin/mcp/apps", headers=admin_headers)
        assert resp.status_code == 200
        assert any(a["app_id"] == "legacy-bad" for a in resp.json())
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_app_responses_derive_builtin_ownership() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        created = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json={
                "app_id": "custom-owned",
                "name": "Custom Owned",
                "transport": "stdio",
                "launch_config": {
                    "command": "custom-command",
                    "required_env": ["CUSTOM_TOKEN"],
                },
            },
        )
        assert created.status_code == 200
        assert created.json()["is_builtin"] is False

        listed = client.get("/api/admin/mcp/apps", headers=admin_headers)
        assert listed.status_code == 200
        apps = {app["app_id"]: app for app in listed.json()}
        assert apps["gmail"]["is_builtin"] is True
        assert apps["custom-owned"]["is_builtin"] is False
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_app_responses_overlay_builtin_execution_fields_without_persisting() -> (
    None
):
    from xagent.web.builtin_mcp_registry import get_builtin_execution_fields

    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")
        stale_launch_config = {
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.gmail"],
        }
        db = next(get_db())
        try:
            gmail_app = (
                db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
            )
            gmail_app.name = "Stale Gmail"
            gmail_app.oauth_scopes = ["stale-scope"]
            gmail_app.launch_config = stale_launch_config
            db.commit()
        finally:
            db.close()

        response = client.get("/api/admin/mcp/apps", headers=admin_headers)

        assert response.status_code == 200
        apps = {app["app_id"]: app for app in response.json()}
        execution_fields = get_builtin_execution_fields("gmail")
        assert execution_fields is not None
        assert {
            field: apps["gmail"][field] for field in execution_fields
        } == execution_fields
        assert apps["gmail"]["is_builtin"] is True

        db = next(get_db())
        try:
            persisted_gmail = (
                db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
            )
            assert persisted_gmail.name == "Stale Gmail"
            assert persisted_gmail.oauth_scopes == ["stale-scope"]
            assert persisted_gmail.launch_config == stale_launch_config
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("app_id", "renamed-gmail"),
        ("name", "Renamed Gmail"),
        ("transport", "stdio"),
        ("provider_name", "wrong-provider"),
        ("oauth_scopes", ["wrong-scope"]),
        (
            "launch_config",
            {
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.gmail"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        ),
    ],
)
def test_admin_patch_rejects_builtin_execution_field_changes(
    field: str, replacement: object
) -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")
        db = next(get_db())
        try:
            app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
            app_pk = app.id
            before = {
                column.name: getattr(app, column.name)
                for column in PublicMCPApp.__table__.columns
            }
        finally:
            db.close()

        response = client.patch(
            f"/api/admin/mcp/apps/{app_pk}",
            headers=admin_headers,
            json={field: replacement},
        )

        assert response.status_code == 409
        db = next(get_db())
        try:
            app = db.query(PublicMCPApp).filter(PublicMCPApp.id == app_pk).one()
            after = {
                column.name: getattr(app, column.name)
                for column in PublicMCPApp.__table__.columns
            }
            assert after == before
            assert db.query(PublicMCPAppAudit).count() == 0
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_put_rejects_changed_builtin_execution_field_without_audit() -> None:
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")
        db = next(get_db())
        try:
            app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
            app_pk = app.id
            persisted_name = app.name
        finally:
            db.close()

        canonical = get_builtin_public_mcp_app("gmail")
        assert canonical is not None
        canonical["name"] = "Renamed Gmail"

        response = client.put(
            f"/api/admin/mcp/apps/{app_pk}",
            headers=admin_headers,
            json=canonical,
        )

        assert response.status_code == 409
        assert response.json()["detail"] == (
            "Built-in MCP app field 'name' is managed by code"
        )
        db = next(get_db())
        try:
            app = db.query(PublicMCPApp).filter(PublicMCPApp.id == app_pk).one()
            assert app.name == persisted_name
            assert db.query(PublicMCPAppAudit).count() == 0
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_patch_allows_builtin_presentation_fields() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")
        db = next(get_db())
        try:
            app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
            app_pk = app.id
            original_launch_config = app.launch_config
            original_description = app.description
            admin_user_id = db.query(User.id).filter(User.username == "admin").scalar()
        finally:
            db.close()

        response = client.patch(
            f"/api/admin/mcp/apps/{app_pk}",
            headers={**admin_headers, "X-Request-ID": "builtin-patch"},
            json={
                "description": "Managed Gmail description",
                "icon": "https://example.com/managed-gmail.png",
                "category": "Managed Communication",
                "is_visible_in_connector": False,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["is_builtin"] is True
        assert body["description"] == "Managed Gmail description"
        assert body["icon"] == "https://example.com/managed-gmail.png"
        assert body["category"] == "Managed Communication"
        assert body["is_visible_in_connector"] is False
        assert body["launch_config"] == original_launch_config

        db = next(get_db())
        try:
            audit = db.query(PublicMCPAppAudit).one()
            assert audit.action == "update"
            assert audit.app_id == "gmail"
            assert audit.actor_user_id == admin_user_id
            assert audit.request_id == "builtin-patch"
            assert audit.before_values["description"] == original_description
            assert audit.after_values["description"] == "Managed Gmail description"
            assert audit.after_values["launch_config"] == original_launch_config
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_legacy_put_accepts_canonical_builtin_execution_fields() -> None:
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")
        db = next(get_db())
        try:
            app_pk = (
                db.query(PublicMCPApp.id)
                .filter(PublicMCPApp.app_id == "gmail")
                .scalar()
            )
        finally:
            db.close()
        canonical = get_builtin_public_mcp_app("gmail")
        assert canonical is not None
        canonical["description"] = "Legacy client presentation update"

        response = client.put(
            f"/api/admin/mcp/apps/{app_pk}",
            headers={**admin_headers, "X-Request-ID": "builtin-put"},
            json=canonical,
        )

        assert response.status_code == 200
        assert response.json()["is_builtin"] is True
        assert response.json()["description"] == "Legacy client presentation update"
        db = next(get_db())
        try:
            audit = db.query(PublicMCPAppAudit).one()
            assert audit.action == "update"
            assert audit.app_id == "gmail"
            assert audit.request_id == "builtin-put"
            assert (
                audit.before_values["description"] != audit.after_values["description"]
            )
            assert (
                audit.after_values["description"] == "Legacy client presentation update"
            )
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_custom_patch_validates_merged_state_and_keeps_app_id_immutable() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")
        created = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json={
                "app_id": "custom-patch",
                "name": "Custom Patch",
                "transport": "stdio",
                "launch_config": {
                    "command": "old-command",
                    "required_env": ["CUSTOM_TOKEN"],
                },
            },
        )
        assert created.status_code == 200
        app_pk = created.json()["id"]

        updated = client.patch(
            f"/api/admin/mcp/apps/{app_pk}",
            headers=admin_headers,
            json={
                "launch_config": {
                    "command": "new-command",
                    "required_env": ["CUSTOM_TOKEN"],
                }
            },
        )
        assert updated.status_code == 200
        assert updated.json()["launch_config"]["command"] == "new-command"
        assert updated.json()["is_builtin"] is False

        invalid = client.patch(
            f"/api/admin/mcp/apps/{app_pk}",
            headers=admin_headers,
            json={"launch_config": {"command": "incomplete-command"}},
        )
        assert invalid.status_code == 422

        renamed = client.patch(
            f"/api/admin/mcp/apps/{app_pk}",
            headers=admin_headers,
            json={"app_id": "custom-patch-renamed"},
        )
        assert renamed.status_code == 409
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_create_rejects_reserved_builtin_id_after_deletion() -> None:
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")
        db = next(get_db())
        try:
            app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
            db.delete(app)
            db.commit()
        finally:
            db.close()
        canonical = get_builtin_public_mcp_app("gmail")
        assert canonical is not None

        response = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json=canonical,
        )

        assert response.status_code == 409
        db = next(get_db())
        try:
            assert (
                db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").first()
                is None
            )
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_delete_rejects_builtin_catalog_app() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")
        db = next(get_db())
        try:
            app_pk = (
                db.query(PublicMCPApp.id)
                .filter(PublicMCPApp.app_id == "gmail")
                .scalar()
            )
        finally:
            db.close()

        response = client.delete(
            f"/api/admin/mcp/apps/{app_pk}",
            headers=admin_headers,
        )

        assert response.status_code == 409
        db = next(get_db())
        try:
            assert db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").one()
            assert db.query(PublicMCPAppAudit).count() == 0
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_failed_custom_create_rolls_back_staged_audit() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        def fail_commit_after_flush(session: Session) -> None:
            session.flush()
            raise RuntimeError("commit failed after flush")

        with patch.object(Session, "commit", fail_commit_after_flush):
            with pytest.raises(RuntimeError, match="commit failed after flush"):
                client.post(
                    "/api/admin/mcp/apps",
                    headers=admin_headers,
                    json={
                        "app_id": "custom-failed-audit",
                        "name": "Custom Failed Audit",
                        "transport": "stdio",
                        "launch_config": {
                            "command": "custom-command",
                            "required_env": ["CUSTOM_TOKEN"],
                        },
                    },
                )

        db = next(get_db())
        try:
            assert (
                db.query(PublicMCPApp)
                .filter(PublicMCPApp.app_id == "custom-failed-audit")
                .first()
                is None
            )
            assert (
                db.query(PublicMCPAppAudit)
                .filter(PublicMCPAppAudit.app_id == "custom-failed-audit")
                .count()
                == 0
            )
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_custom_catalog_writes_record_before_after_audits() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")
        db = next(get_db())
        try:
            admin_user_id = db.query(User.id).filter(User.username == "admin").scalar()
        finally:
            db.close()

        created = client.post(
            "/api/admin/mcp/apps",
            headers={**admin_headers, "X-Request-ID": "catalog-create"},
            json={
                "app_id": "custom-audited",
                "name": "Custom Audited",
                "transport": "stdio",
                "launch_config": {
                    "command": "old-command",
                    "required_env": ["CUSTOM_TOKEN"],
                },
            },
        )
        assert created.status_code == 200
        app_pk = created.json()["id"]

        replaced = client.put(
            f"/api/admin/mcp/apps/{app_pk}",
            headers={**admin_headers, "X-Request-ID": "catalog-put"},
            json={
                "app_id": "custom-audited",
                "name": "Custom Audited Put",
                "transport": "stdio",
                "launch_config": {
                    "command": "put-command",
                    "required_env": ["CUSTOM_TOKEN"],
                },
            },
        )
        assert replaced.status_code == 200

        updated = client.patch(
            f"/api/admin/mcp/apps/{app_pk}",
            headers={**admin_headers, "X-Request-ID": "catalog-update"},
            json={
                "launch_config": {
                    "command": "new-command",
                    "required_env": ["CUSTOM_TOKEN"],
                }
            },
        )
        assert updated.status_code == 200

        deleted = client.delete(
            f"/api/admin/mcp/apps/{app_pk}",
            headers={**admin_headers, "X-Request-ID": "catalog-delete"},
        )
        assert deleted.status_code == 200

        db = next(get_db())
        try:
            audits = (
                db.query(PublicMCPAppAudit)
                .filter(PublicMCPAppAudit.app_id == "custom-audited")
                .order_by(PublicMCPAppAudit.id)
                .all()
            )
            assert [audit.action for audit in audits] == [
                "create",
                "update",
                "update",
                "delete",
            ]
            assert [audit.request_id for audit in audits] == [
                "catalog-create",
                "catalog-put",
                "catalog-update",
                "catalog-delete",
            ]
            assert {audit.actor_user_id for audit in audits} == {admin_user_id}
            assert audits[0].before_values is None
            assert audits[0].after_values["launch_config"]["command"] == "old-command"
            assert audits[1].before_values["launch_config"]["command"] == "old-command"
            assert audits[1].after_values["launch_config"]["command"] == "put-command"
            assert audits[2].before_values["launch_config"]["command"] == "put-command"
            assert audits[2].after_values["launch_config"]["command"] == "new-command"
            assert audits[3].before_values["launch_config"]["command"] == "new-command"
            assert audits[3].after_values is None
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass
