from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command

from xagent.db.config import create_alembic_config
from xagent.web.models.generation import RandomUUID

REVISION = "20260909_actor_mcp_connections"
DOWN_REVISION = "20260901_seed_zendesk_mcp_app"
TABLE = "actor_mcp_server_connections"
MIGRATIONS_DIR = Path(__file__).parents[2] / "src" / "xagent" / "migrations"


def _legacy_schema(connection: sa.Connection) -> None:
    connection.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY NOT NULL)"))
    connection.execute(
        sa.text(
            "CREATE TABLE public_mcp_apps ("
            "id INTEGER PRIMARY KEY NOT NULL, app_id VARCHAR(100) NOT NULL UNIQUE, "
            "generation CHAR(32) NOT NULL UNIQUE)"
        )
    )
    connection.execute(
        sa.text(
            "CREATE TABLE alembic_version ("
            "version_num VARCHAR(255) NOT NULL PRIMARY KEY)"
        )
    )
    connection.execute(
        sa.text("INSERT INTO alembic_version VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def _create_actor_table(
    connection: sa.Connection,
    *,
    encrypted_env_type: sa.types.TypeEngine | None = None,
    lifecycle_default: sa.sql.ClauseElement | None = None,
    check_expression: str = "CAST(lifecycle_generation AS VARCHAR) <> ''",
    extra_user_unique: bool = False,
) -> None:
    metadata = sa.MetaData()
    sa.Table("users", metadata, autoload_with=connection)
    sa.Table("public_mcp_apps", metadata, autoload_with=connection)
    constraints: list[sa.Constraint] = [
        sa.CheckConstraint(
            check_expression,
            name="ck_actor_mcp_server_connections_generation_nonempty",
        ),
        sa.UniqueConstraint(
            "lifecycle_generation",
            name="uq_actor_mcp_server_connections_lifecycle_generation",
        ),
        sa.UniqueConstraint(
            "user_id",
            "resource_owner_key",
            "app_id",
            name="uq_actor_mcp_server_connections_actor_app",
        ),
        sa.UniqueConstraint(
            "user_id",
            "resource_owner_key",
            "catalog_app_generation",
            name="uq_actor_mcp_server_connections_actor_catalog_generation",
        ),
    ]
    if extra_user_unique:
        constraints.append(
            sa.UniqueConstraint(
                "user_id", name="uq_actor_mcp_server_connections_user_id_drift"
            )
        )
    table = sa.Table(
        TABLE,
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "lifecycle_generation",
            sa.Uuid(),
            server_default=(
                lifecycle_default if lifecycle_default is not None else RandomUUID()
            ),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("resource_owner_key", sa.String(512), nullable=False),
        sa.Column("app_id", sa.String(100), nullable=False),
        sa.Column(
            "catalog_app_generation",
            sa.Uuid(),
            sa.ForeignKey("public_mcp_apps.generation", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "encrypted_env",
            encrypted_env_type if encrypted_env_type is not None else sa.JSON(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        *constraints,
    )
    table.create(connection)
    sa.Index("ix_actor_mcp_server_connections_id", table.c.id).create(connection)


def test_sqlite_upgrade_constraints_and_downgrade() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    with engine.connect() as connection:
        connection.execute(sa.text("PRAGMA foreign_keys=ON"))
        _legacy_schema(connection)
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)

        inspector = sa.inspect(connection)
        columns = {item["name"]: item for item in inspector.get_columns(TABLE)}
        assert columns["resource_owner_key"]["type"].length == 512
        assert columns["app_id"]["type"].length == 100
        assert columns["encrypted_env"]["nullable"] is True
        assert columns["lifecycle_generation"]["nullable"] is False
        assert columns["lifecycle_generation"]["default"] is not None
        assert {
            tuple(item["column_names"])
            for item in inspector.get_unique_constraints(TABLE)
        } >= {
            ("lifecycle_generation",),
            ("user_id", "resource_owner_key", "app_id"),
            ("user_id", "resource_owner_key", "catalog_app_generation"),
        }
        foreign_keys = {
            tuple(item["constrained_columns"]): (
                item["referred_table"],
                item["options"].get("ondelete"),
            )
            for item in inspector.get_foreign_keys(TABLE)
        }
        assert foreign_keys == {
            ("catalog_app_generation",): ("public_mcp_apps", "CASCADE"),
            ("user_id",): ("users", "CASCADE"),
        }

        connection.execute(sa.text("INSERT INTO users (id) VALUES (1)"))
        app_generation = uuid.uuid4()
        connection.execute(
            sa.text(
                "INSERT INTO public_mcp_apps (id, app_id, generation) "
                "VALUES (1, 'test-app', :generation)"
            ),
            {"generation": app_generation.hex},
        )
        connection.execute(
            sa.text(
                f"INSERT INTO {TABLE} "
                "(user_id, resource_owner_key, app_id, catalog_app_generation) "
                "VALUES (1, 'toby:test', 'test-app', :generation)"
            ),
            {"generation": app_generation.hex},
        )
        generation = uuid.UUID(
            str(connection.scalar(sa.text(f"SELECT lifecycle_generation FROM {TABLE}")))
        )
        assert generation.version == 4
        connection.execute(sa.text("DELETE FROM public_mcp_apps WHERE id = 1"))
        assert connection.scalar(sa.text(f"SELECT count(*) FROM {TABLE}")) == 0

        command.downgrade(config, DOWN_REVISION)
        assert TABLE not in sa.inspect(connection).get_table_names()

    engine.dispose()


def test_sqlite_upgrade_adopts_exact_metadata_table() -> None:
    from xagent.web.models.actor_mcp_connection import ActorMCPServerConnection

    engine = sa.create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    with engine.connect() as connection:
        connection.execute(sa.text("PRAGMA foreign_keys=ON"))
        _legacy_schema(connection)
        ActorMCPServerConnection.__table__.create(bind=connection)
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        assert sa.inspect(connection).get_table_names().count(TABLE) == 1
        assert (
            connection.scalar(sa.text("SELECT version_num FROM alembic_version"))
            == REVISION
        )

    engine.dispose()


def test_sqlite_upgrade_rejects_partial_preexisting_table() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    with engine.connect() as connection:
        _legacy_schema(connection)
        connection.execute(
            sa.text(
                f"CREATE TABLE {TABLE} (id INTEGER NOT NULL PRIMARY KEY, "
                "user_id INTEGER NOT NULL)"
            )
        )
        config.attributes["connection"] = connection

        with pytest.raises(RuntimeError, match="incompatible schema"):
            command.upgrade(config, REVISION)

        assert (
            connection.scalar(sa.text("SELECT version_num FROM alembic_version"))
            == DOWN_REVISION
        )

    engine.dispose()


@pytest.mark.parametrize(
    "drift",
    ["encrypted-type", "fixed-uuid-default", "same-name-check", "extra-unique"],
)
def test_sqlite_upgrade_rejects_semantic_schema_drift(drift: str) -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    with engine.connect() as connection:
        connection.execute(sa.text("PRAGMA foreign_keys=ON"))
        _legacy_schema(connection)
        kwargs: dict[str, object] = {}
        if drift == "encrypted-type":
            kwargs["encrypted_env_type"] = sa.Text()
        elif drift == "fixed-uuid-default":
            kwargs["lifecycle_default"] = sa.text("'00000000000000000000000000000000'")
        elif drift == "same-name-check":
            kwargs["check_expression"] = "1 = 1"
        elif drift == "extra-unique":
            kwargs["extra_user_unique"] = True
        _create_actor_table(connection, **kwargs)  # type: ignore[arg-type]
        config.attributes["connection"] = connection

        with pytest.raises(RuntimeError, match="incompatible schema"):
            command.upgrade(config, REVISION)

        assert (
            connection.scalar(sa.text("SELECT version_num FROM alembic_version"))
            == DOWN_REVISION
        )

    engine.dispose()
