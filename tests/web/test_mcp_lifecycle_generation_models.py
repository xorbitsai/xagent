from __future__ import annotations

import ast
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateColumn

from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _catalog_app() -> PublicMCPApp:
    return PublicMCPApp(app_id="generation-test", name="Generation test")


def _user_and_server(db: Session) -> tuple[User, object]:
    user = User(username="generation-test", password_hash="x")
    server = MCPServer(
        name="generation-test-server",
        managed="external",
        transport="streamable_http",
        restart_policy="no",
    )
    db.add_all([user, server])
    db.flush()
    return user, server


def test_catalog_generation_is_random_unique_and_not_reused_with_sqlite_rowid() -> None:
    with _session() as db:
        original = _catalog_app()
        db.add(original)
        db.commit()
        original_id = original.id
        original_generation = original.generation

        db.delete(original)
        db.commit()

        replacement = _catalog_app()
        db.add(replacement)
        db.commit()

        assert replacement.id == original_id
        assert isinstance(replacement.generation, uuid.UUID)
        assert replacement.generation != original_generation


def test_association_generation_is_random_unique_and_not_reused() -> None:
    with _session() as db:
        user, server = _user_and_server(db)
        original = UserMCPServer(user_id=user.id, mcpserver_id=server.id)
        db.add(original)
        db.commit()
        original_generation = original.lifecycle_generation

        db.delete(original)
        db.commit()

        replacement = UserMCPServer(user_id=user.id, mcpserver_id=server.id)
        db.add(replacement)
        db.commit()

        assert isinstance(replacement.lifecycle_generation, uuid.UUID)
        assert replacement.lifecycle_generation != original_generation


@pytest.mark.parametrize(
    ("factory", "attribute"),
    [
        (_catalog_app, "generation"),
        (lambda: UserMCPServer(user_id=1, mcpserver_id=1), "lifecycle_generation"),
    ],
)
def test_generation_is_immutable_after_insert(factory, attribute: str) -> None:
    with _session() as db:
        if attribute == "lifecycle_generation":
            user, server = _user_and_server(db)
            row = UserMCPServer(user_id=user.id, mcpserver_id=server.id)
        else:
            row = factory()
        db.add(row)
        db.commit()
        original = getattr(row, attribute)

        setattr(row, attribute, uuid.uuid4())
        with pytest.raises(ValueError, match="immutable"):
            db.commit()
        db.rollback()

        assert getattr(row, attribute) == original


def test_generation_unique_constraints_reject_duplicate_values() -> None:
    with _session() as db:
        generation = uuid.uuid4()
        db.add_all(
            [
                PublicMCPApp(
                    app_id="generation-one",
                    name="Generation one",
                    generation=generation,
                ),
                PublicMCPApp(
                    app_id="generation-two",
                    name="Generation two",
                    generation=generation,
                ),
            ]
        )
        with pytest.raises(IntegrityError):
            db.commit()


def test_regular_updates_preserve_generations() -> None:
    with _session() as db:
        user, server = _user_and_server(db)
        app = _catalog_app()
        association = UserMCPServer(user_id=user.id, mcpserver_id=server.id)
        db.add_all([app, association])
        db.commit()
        catalog_generation = app.generation
        association_generation = association.lifecycle_generation

        app.name = "Updated generation test"
        association.is_active = False
        db.commit()

        assert app.generation == catalog_generation
        assert association.lifecycle_generation == association_generation


def test_fresh_schema_raw_inserts_receive_database_generations() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user, server = _user_and_server(db)
        db.commit()
        user_id = user.id
        server_id = server.id

    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO public_mcp_apps (app_id, name, transport) "
                "VALUES ('raw-catalog', 'Raw catalog', 'oauth')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO user_mcpservers "
                "(user_id, mcpserver_id, is_owner, can_edit, can_delete, "
                "is_shared, is_active, is_default) "
                "VALUES (:user_id, :server_id, 0, 0, 0, 0, 1, 0)"
            ),
            {"user_id": user_id, "server_id": server_id},
        )
        catalog_generation = uuid.UUID(
            str(
                connection.scalar(
                    sa.text(
                        "SELECT generation FROM public_mcp_apps "
                        "WHERE app_id = 'raw-catalog'"
                    )
                )
            )
        )
        association_generation = uuid.UUID(
            str(
                connection.scalar(
                    sa.text(
                        "SELECT lifecycle_generation FROM user_mcpservers "
                        "WHERE user_id = :user_id AND mcpserver_id = :server_id"
                    ),
                    {"user_id": user_id, "server_id": server_id},
                )
            )
        )

    assert catalog_generation.version == 4
    assert association_generation.version == 4
    assert catalog_generation != association_generation


@pytest.mark.parametrize(
    "column",
    [PublicMCPApp.generation, UserMCPServer.lifecycle_generation],
)
def test_postgresql_fresh_schema_uses_database_uuid_default(column) -> None:
    ddl = str(CreateColumn(column).compile(dialect=postgresql.dialect()))
    assert "DEFAULT gen_random_uuid()" in ddl


def test_production_association_creation_paths_share_the_model_default() -> None:
    """Pin every production constructor that inherits the generation default."""
    repository_root = Path(__file__).parents[2]
    source_root = repository_root / "src" / "xagent"
    constructors: dict[str, set[str]] = {
        "PublicMCPApp": set(),
        "UserMCPServer": set(),
    }
    for path in source_root.rglob("*.py"):
        if "migrations" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name) and function.id in constructors:
                generation_field = (
                    "generation"
                    if function.id == "PublicMCPApp"
                    else "lifecycle_generation"
                )
                assert all(keyword.arg != generation_field for keyword in node.keywords)
                constructors[function.id].add(str(path.relative_to(repository_root)))

    assert constructors == {
        "PublicMCPApp": {"src/xagent/web/api/admin_mcp.py"},
        "UserMCPServer": {
            "src/xagent/web/api/auth.py",
            "src/xagent/web/api/mcp.py",
            "src/xagent/web/mcp_apps.py",
        },
    }
