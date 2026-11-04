"""Tests for the public MCP app audit table migration."""

import importlib.util
from io import StringIO
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260715_add_public_mcp_app_audits.py"
)
REVISION = "20260715_add_public_mcp_app_audits"
DOWN_REVISION = "20260715_normalize_builtin_mcp_launch"
TABLE = "public_mcp_app_audits"
INDEXES = {
    "ix_public_mcp_app_audits_actor_user_id",
    "ix_public_mcp_app_audits_action",
    "ix_public_mcp_app_audits_app_id",
    "ix_public_mcp_app_audits_request_id",
    "ix_public_mcp_app_audits_created_at",
}


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "add_public_mcp_app_audits_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _offline_sql(migration, dialect_name: str, operation: str) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        getattr(migration, operation)()

    return output.getvalue()


def test_upgrade_creates_schema_foreign_key_and_indexes() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")

        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()

        inspector = sa.inspect(connection)
        columns = {column["name"]: column for column in inspector.get_columns(TABLE)}
        assert set(columns) == {
            "id",
            "actor_user_id",
            "action",
            "app_id",
            "before_values",
            "after_values",
            "request_id",
            "created_at",
        }
        assert columns["id"]["primary_key"] == 1
        assert columns["actor_user_id"]["nullable"] is True
        assert columns["action"]["nullable"] is False
        assert columns["app_id"]["nullable"] is False
        assert columns["before_values"]["nullable"] is True
        assert columns["after_values"]["nullable"] is True
        assert columns["request_id"]["nullable"] is True
        assert columns["created_at"]["nullable"] is False
        assert columns["created_at"]["default"] is not None

        foreign_keys = inspector.get_foreign_keys(TABLE)
        assert len(foreign_keys) == 1
        assert foreign_keys[0]["constrained_columns"] == ["actor_user_id"]
        assert foreign_keys[0]["referred_table"] == "users"
        assert foreign_keys[0]["referred_columns"] == ["id"]
        assert foreign_keys[0]["options"]["ondelete"] == "SET NULL"
        assert {index["name"] for index in inspector.get_indexes(TABLE)} == INDEXES

        connection.execute(sa.text("INSERT INTO users (id) VALUES (1)"))
        connection.execute(
            sa.text(
                f"INSERT INTO {TABLE} (actor_user_id, action, app_id) "
                "VALUES (1, 'update', 'gmail')"
            )
        )
        connection.execute(sa.text("DELETE FROM users WHERE id = 1"))
        actor_user_id, created_at = connection.execute(
            sa.text(f"SELECT actor_user_id, created_at FROM {TABLE}")
        ).one()
        assert actor_user_id is None
        assert created_at is not None


def test_downgrade_drops_indexes_and_table() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


def test_online_upgrade_is_idempotent_when_table_already_exists() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert TABLE in inspector.get_table_names()
        assert {index["name"] for index in inspector.get_indexes(TABLE)} == INDEXES


def test_online_upgrade_skips_until_sqlalchemy_base_users_table_exists() -> None:
    from xagent.web import models as _models  # noqa: F401
    from xagent.web.models.database import Base

    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()

        Base.metadata.create_all(connection)
        inspector = sa.inspect(connection)
        assert TABLE in inspector.get_table_names()
        foreign_keys = inspector.get_foreign_keys(TABLE)
        assert len(foreign_keys) == 1
        assert foreign_keys[0]["referred_table"] == "users"

        with Operations.context(operations.get_context()):
            migration.downgrade()


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql", "mysql"])
def test_offline_upgrade_emits_table_fk_indexes_without_bind_parameters(
    dialect_name: str,
) -> None:
    migration = _load_migration_module()

    sql = _offline_sql(migration, dialect_name, "upgrade")

    assert f"CREATE TABLE {TABLE}" in sql
    assert "FOREIGN KEY(actor_user_id) REFERENCES users (id) ON DELETE SET NULL" in sql
    assert sql.count("CREATE INDEX ix_public_mcp_app_audits_") == len(INDEXES)
    assert "action VARCHAR(16) NOT NULL" in sql
    assert "app_id VARCHAR(100) NOT NULL" in sql
    assert "request_id VARCHAR(128)" in sql
    assert "created_at" in sql
    assert "DEFAULT" in sql
    assert "%(" not in sql
    assert ":param" not in sql
    assert "?" not in sql
    if dialect_name == "postgresql":
        assert "TIMESTAMP WITH TIME ZONE" in sql


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql", "mysql"])
def test_offline_downgrade_drops_indexes_before_table(dialect_name: str) -> None:
    migration = _load_migration_module()

    sql = _offline_sql(migration, dialect_name, "downgrade")

    for index_name in INDEXES:
        assert f"DROP INDEX {index_name}" in sql
        assert sql.index(f"DROP INDEX {index_name}") < sql.index(f"DROP TABLE {TABLE}")
    assert "%(" not in sql
    assert ":param" not in sql
    assert "?" not in sql


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == REVISION
    assert migration.down_revision == DOWN_REVISION
