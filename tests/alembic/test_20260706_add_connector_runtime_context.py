"""Tests for the connector runtime context migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from xagent.db.config import create_alembic_config


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260706_add_connector_runtime_context.py"
    )
    spec = importlib.util.spec_from_file_location(
        "connector_runtime_context_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    context = MigrationContext.configure(connection)
    return Operations(context)


def _create_parent_tables(connection) -> None:
    connection.execute(
        text(
            """
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title VARCHAR(200) NOT NULL
            )
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                managed VARCHAR(20) NOT NULL,
                transport VARCHAR(50) NOT NULL
            )
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE TABLE custom_apis (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL
            )
            """
        )
    )


def test_upgrade_adds_connector_runtime_columns_and_context_table(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()

    with engine.begin() as connection:
        _create_parent_tables(connection)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        assert "task_connector_runtime_contexts" in tables

        task_columns = {column["name"] for column in inspector.get_columns("tasks")}
        assert "connector_runtime_selected_refs" in task_columns

        for table_name in ("mcp_servers", "custom_apis"):
            columns = {column["name"] for column in inspector.get_columns(table_name)}
            assert {
                "runtime_input_schema",
                "runtime_bindings",
                "allow_delegated_authorization",
            }.issubset(columns)

        context_columns = {
            column["name"]
            for column in inspector.get_columns("task_connector_runtime_contexts")
        }
        assert {
            "id",
            "task_id",
            "connector_type",
            "connector_id",
            "context",
            "created_at",
        }.issubset(context_columns)

        unique_constraints = {
            constraint["name"]: tuple(constraint["column_names"])
            for constraint in inspector.get_unique_constraints(
                "task_connector_runtime_contexts"
            )
        }
        assert unique_constraints["uq_task_connector_runtime_contexts_ref"] == (
            "task_id",
            "connector_type",
            "connector_id",
        )

        indexes = {
            index["name"]
            for index in inspector.get_indexes("task_connector_runtime_contexts")
        }
        assert "ix_task_connector_runtime_contexts_task_id" in indexes

        foreign_keys = inspector.get_foreign_keys("task_connector_runtime_contexts")
        assert any(
            foreign_key["constrained_columns"] == ["task_id"]
            and foreign_key["referred_table"] == "tasks"
            for foreign_key in foreign_keys
        )


def test_downgrade_removes_connector_runtime_schema(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()

    with engine.begin() as connection:
        _create_parent_tables(connection)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        assert "task_connector_runtime_contexts" not in tables

        task_columns = {column["name"] for column in inspector.get_columns("tasks")}
        assert "connector_runtime_selected_refs" not in task_columns

        for table_name in ("mcp_servers", "custom_apis"):
            columns = {column["name"] for column in inspector.get_columns(table_name)}
            assert "runtime_input_schema" not in columns
            assert "runtime_bindings" not in columns
            assert "allow_delegated_authorization" not in columns


def test_empty_database_alembic_upgrade_fails_closed_at_owner_revision(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'unsupported-empty.db'}")
    cfg = create_alembic_config(engine)

    with engine.begin() as connection:
        cfg.attributes["connection"] = connection

        with pytest.raises(RuntimeError, match="requires the users table"):
            command.upgrade(cfg, "head")

        inspector = inspect(connection)
        assert "users" not in inspector.get_table_names()
        assert "resource_owner_key" not in {
            column["name"] for column in inspector.get_columns("user_oauth")
        }
        # Not scalar_one(): alembic_version can legitimately hold more than
        # one row here. Any other connector's migration branch that hasn't
        # yet been reunited with this one by a merge revision has no
        # dependency on the users table this owner-aware migration requires,
        # so it fully applies to its own independent tip while this branch
        # is blocked -- exactly one row (this branch's) is pinned at
        # "20260818_seed_stripe_mcp_app", not necessarily the only row.
        applied_revisions = {
            row[0]
            for row in connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).fetchall()
        }
        assert "20260818_seed_stripe_mcp_app" in applied_revisions
        assert (
            connection.execute(
                text("SELECT count(*) FROM public_mcp_apps WHERE app_id = 'stripe'")
            ).scalar_one()
            == 1
        )


def test_database_with_core_users_table_alembic_upgrade_to_head_completes(
    tmp_path,
) -> None:
    from xagent.web.models.user import User

    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    cfg = create_alembic_config(engine)

    with engine.begin() as connection:
        # Application startup creates metadata-owned core tables before
        # Alembic upgrades the migration-owned schema.
        User.__table__.create(bind=connection)
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")

        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        assert "alembic_version" in tables
