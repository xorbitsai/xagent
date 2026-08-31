"""Contract tests for canonical builtin OAuth server visibility."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web import mcp_apps
from xagent.web.builtin_mcp_registry import (
    get_builtin_execution_fields_and_optional_scopes,
)
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth

TEST_BUILTIN_APP_ID = "calendar"
TEST_BUILTIN_EXECUTION = {
    "name": "Google Calendar",
    "transport": "oauth",
    "provider_name": "custom",
    "oauth_scopes": [],
    "launch_config": {"command": "calendar"},
}
NON_EXECUTION_SERVER_FIELDS = {
    "id",
    "description",
    "created_at",
    "updated_at",
}
DIRECTLY_VALIDATED_SERVER_FIELDS = {
    "name",
    "managed",
    "transport",
    "auth",
    "concurrency_safe",
    "concurrent_tools",
    "allow_delegated_authorization",
    "restart_policy",
}
NONCANONICAL_SERVER_VALUES = {
    "managed": "internal",
    "command": "foreign-command",
    "args": ["foreign-argument"],
    "url": "https://foreign.example",
    "env": {"FOREIGN": "value"},
    "cwd": "/foreign",
    "headers": {"X-Foreign": "value"},
    "timeout": 1,
    "runtime_input_schema": {"type": "object"},
    "runtime_bindings": {"token": "foreign"},
    "docker_url": "unix:///foreign.sock",
    "docker_image": "foreign/image",
    "docker_environment": {"FOREIGN": "value"},
    "docker_working_dir": "/foreign",
    "volumes": ["/foreign:/foreign"],
    "bind_ports": {"8080": 8080},
    "auto_start": True,
    "container_id": "foreign-container",
    "container_name": "foreign-container",
    "container_logs": ["foreign-log"],
    "concurrency_safe": True,
    "concurrent_tools": ["foreign-tool"],
    "allow_delegated_authorization": True,
    "restart_policy": "always",
}


@pytest.fixture
def catalog_db(tmp_path, monkeypatch):
    registry_lookup = mcp_apps.get_builtin_execution_fields_and_optional_scopes

    def test_registry(app_id: str):
        if app_id == TEST_BUILTIN_APP_ID:
            return TEST_BUILTIN_EXECUTION, []
        return registry_lookup(app_id)

    monkeypatch.setattr(
        mcp_apps, "get_builtin_execution_fields_and_optional_scopes", test_registry
    )

    engine = create_engine(f"sqlite:///{tmp_path / 'builtin-oauth-catalog.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)
        yield db, user
    engine.dispose()


def _catalog_link(db: Session, user: User) -> tuple[MCPServer, UserMCPServer]:
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
        name="Google Calendar",
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


def test_canonical_validation_covers_mcp_server_schema() -> None:
    covered_fields = (
        NON_EXECUTION_SERVER_FIELDS
        | DIRECTLY_VALIDATED_SERVER_FIELDS
        | set(mcp_apps._CANONICAL_EMPTY_SERVER_FIELDS)
    )

    assert set(MCPServer.__table__.columns.keys()) == covered_fields


@pytest.mark.parametrize(
    ("field_name", "value"),
    NONCANONICAL_SERVER_VALUES.items(),
)
def test_definition_rejects_noncanonical_server_fields(
    catalog_db, field_name, value
) -> None:
    db, user = catalog_db
    server, _link = _catalog_link(db, user)
    setattr(server, field_name, value)
    db.commit()

    with pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError,
        match=field_name,
    ):
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id="calendar", provider="custom"
        )


def test_visibility_creates_canonical_nonowning_personal_link(catalog_db) -> None:
    db, user = catalog_db
    db.add(
        PublicMCPApp(
            app_id="calendar",
            name="Google Calendar",
            description="Calendar",
            transport="oauth",
            provider_name="custom",
            launch_config={"command": "calendar"},
            is_visible_in_connector=True,
        )
    )
    db.commit()

    first = mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
        db, user_id=int(user.id), app_id="calendar"
    )
    second = mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
        db, user_id=int(user.id), app_id="calendar"
    )

    assert first.id == second.id
    assert first.auth == {"app_id": "calendar", "provider": "custom"}
    links = db.query(UserMCPServer).all()
    assert len(links) == 1
    assert links[0].is_owner is False
    assert links[0].can_edit is False
    assert links[0].can_delete is False
    assert links[0].is_shared is False
    assert links[0].is_active is True
    assert db.query(UserOAuth).count() == 0


def test_visibility_rejects_invalid_user_id_with_plain_value_error(
    catalog_db,
) -> None:
    db, _user = catalog_db

    with pytest.raises(ValueError, match="persisted positive integer") as exc_info:
        mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
            db, user_id=0, app_id="calendar"
        )

    assert type(exc_info.value) is ValueError


@pytest.mark.parametrize("server_name", ["calendar", "Google Calendar"])
@pytest.mark.parametrize(
    "auth",
    [
        None,
        {},
        {"provider": "custom"},
        {"app_id": "calendar"},
    ],
)
def test_definition_requires_exact_auth_metadata(catalog_db, server_name, auth) -> None:
    db, user = catalog_db
    server, _link = _catalog_link(db, user)
    server.name = server_name
    server.auth = auth
    db.commit()

    with pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError,
        match="auth",
    ):
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id="calendar", provider="custom"
        )


@pytest.mark.parametrize(
    "auth",
    [
        None,
        {},
        {"provider": "custom"},
        {"app_id": "calendar"},
    ],
)
def test_visibility_repairs_incomplete_auth_metadata(catalog_db, auth) -> None:
    db, user = catalog_db
    server, _link = _catalog_link(db, user)
    server.auth = auth
    db.commit()

    mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
        db, user_id=int(user.id), app_id="calendar"
    )

    db.refresh(server)
    assert server.auth == {"app_id": "calendar", "provider": "custom"}


def test_definition_rejects_mixed_case_transport(catalog_db) -> None:
    db, user = catalog_db
    server, _link = _catalog_link(db, user)
    server.transport = "OAUTH"
    db.commit()

    with pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError,
        match="transport",
    ):
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id="calendar", provider="custom"
        )


def test_definition_rejects_duplicate_builtin_servers(catalog_db) -> None:
    db, user = catalog_db
    _catalog_link(db, user)
    db.add(
        MCPServer(
            name="calendar",
            managed="external",
            transport="oauth",
            auth={"app_id": "calendar", "provider": "custom"},
        )
    )
    db.commit()

    with pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError,
        match="exactly one|multiple",
    ):
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id="calendar", provider="custom"
        )


def test_visibility_rejects_normalized_reserved_alias(catalog_db) -> None:
    db, user = catalog_db
    execution, _optional_scopes = get_builtin_execution_fields_and_optional_scopes(
        "google-calendar"
    )
    assert execution is not None
    db.add_all(
        [
            PublicMCPApp(
                app_id="google-calendar",
                name=str(execution["name"]),
                transport=str(execution["transport"]),
                provider_name=str(execution["provider_name"]),
                oauth_scopes=list(execution["oauth_scopes"]),
                launch_config=dict(execution["launch_config"]),
                is_visible_in_connector=True,
            ),
            MCPServer(
                name=" google   calendar ",
                managed="external",
                transport="oauth",
            ),
        ]
    )
    db.commit()

    with pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError,
        match="ambiguous reserved",
    ):
        mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
            db, user_id=int(user.id), app_id="google-calendar"
        )


def test_classification_ignores_missing_normalized_identity(catalog_db) -> None:
    db, _user = catalog_db
    app = PublicMCPApp(
        app_id="blank-catalog",
        name="",
        transport="oauth",
        provider_name="custom",
        is_visible_in_connector=True,
    )
    server = MCPServer(
        name="   ",
        managed="external",
        transport="stdio",
    )
    db.add_all([app, server])
    db.commit()

    assert mcp_apps.classify_actor_builtin_oauth_server(db, server) is None


def test_visibility_rejects_normalized_reserved_auth_alias(catalog_db) -> None:
    db, user = catalog_db
    execution, _optional_scopes = get_builtin_execution_fields_and_optional_scopes(
        "google-calendar"
    )
    assert execution is not None
    db.add_all(
        [
            PublicMCPApp(
                app_id="google-calendar",
                name=str(execution["name"]),
                transport=str(execution["transport"]),
                provider_name=str(execution["provider_name"]),
                oauth_scopes=list(execution["oauth_scopes"]),
                launch_config=dict(execution["launch_config"]),
                is_visible_in_connector=True,
            ),
            MCPServer(
                name="Foreign server",
                managed="external",
                transport="oauth",
                auth={"app_id": " GOOGLE   CALENDAR "},
            ),
        ]
    )
    db.commit()

    with pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError,
        match="ambiguous reserved",
    ):
        mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
            db, user_id=int(user.id), app_id="google-calendar"
        )

    assert db.query(MCPServer).count() == 1
    assert db.query(UserMCPServer).count() == 0


def test_visibility_rejects_padded_app_id(catalog_db) -> None:
    db, user = catalog_db
    _catalog_link(db, user)

    with pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError,
        match="exact|whitespace",
    ):
        mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
            db, user_id=int(user.id), app_id=" calendar "
        )


def test_definition_rejects_padded_app_id(catalog_db) -> None:
    db, user = catalog_db
    _catalog_link(db, user)

    with pytest.raises(mcp_apps.BuiltinOAuthServerDefinitionError):
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id=" calendar ", provider="custom"
        )


def test_definition_rejects_app_missing_from_builtin_registry(catalog_db) -> None:
    db, _user = catalog_db
    db.add_all(
        [
            PublicMCPApp(
                app_id="admin-calendar",
                name="Admin Calendar",
                transport="oauth",
                provider_name="custom",
                launch_config={"command": "calendar"},
                is_visible_in_connector=True,
            ),
            MCPServer(
                name="Admin Calendar",
                managed="external",
                transport="oauth",
                auth={"app_id": "admin-calendar", "provider": "custom"},
            ),
        ]
    )
    db.commit()

    with pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError,
        match="builtin registry",
    ):
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id="admin-calendar", provider="custom"
        )


def test_definition_rejects_seeded_catalog_execution_drift(catalog_db) -> None:
    db, _user = catalog_db
    execution, _optional_scopes = get_builtin_execution_fields_and_optional_scopes(
        "gmail"
    )
    assert execution is not None
    db.add_all(
        [
            PublicMCPApp(
                app_id="gmail",
                name=str(execution["name"]),
                transport=str(execution["transport"]),
                provider_name="evil",
                oauth_scopes=list(execution["oauth_scopes"]),
                launch_config=dict(execution["launch_config"]),
                is_visible_in_connector=True,
            ),
            MCPServer(
                name=str(execution["name"]),
                managed="external",
                transport="oauth",
                auth={
                    "app_id": "gmail",
                    "provider": execution["provider_name"],
                },
            ),
        ]
    )
    db.commit()

    with pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError,
        match="catalog.*drift|persisted",
    ):
        mcp_apps.require_builtin_oauth_server_definition(
            db,
            app_id="gmail",
            provider=str(execution["provider_name"]),
        )


def test_visibility_preserves_existing_owning_link(catalog_db) -> None:
    db, user = catalog_db
    _server, link = _catalog_link(db, user)
    link.is_owner = True
    link.can_edit = True
    link.can_delete = True
    link.is_shared = True
    link.is_active = False
    db.commit()

    mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
        db,
        user_id=int(user.id),
        app_id="calendar",
    )
    db.refresh(link)

    assert link.is_owner is True
    assert link.can_edit is True
    assert link.can_delete is True
    assert link.is_shared is True
    assert link.is_active is True


def test_owner_drift_fails_closed_at_strict_definition(catalog_db) -> None:
    db, user = catalog_db
    server, link = _catalog_link(db, user)
    link.is_owner = True
    link.can_edit = True
    link.can_delete = True
    server.transport = "stdio"
    server.command = "/bin/foreign-command"
    db.commit()

    with pytest.raises(
        mcp_apps.BuiltinOAuthServerDefinitionError,
        match="canonical builtin OAuth definition",
    ):
        mcp_apps.require_builtin_oauth_server_definition(
            db, app_id="calendar", provider="custom"
        )

    db.refresh(link)
    assert link.is_owner is True
    assert link.can_edit is True
    assert link.can_delete is True


def test_visibility_preserves_existing_nonowning_policy(catalog_db) -> None:
    db, user = catalog_db
    _server, link = _catalog_link(db, user)
    link.is_owner = False
    link.can_edit = True
    link.can_delete = True
    link.is_shared = True
    link.is_active = False
    db.commit()

    mcp_apps.ensure_builtin_oauth_server_visibility_for_user(
        db,
        user_id=int(user.id),
        app_id="calendar",
    )
    db.refresh(link)

    assert link.is_owner is False
    assert link.can_edit is True
    assert link.can_delete is True
    assert link.is_shared is True
    assert link.is_active is True
