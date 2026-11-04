"""Tests for the workforces.manager_instructions column-drop migration."""

import importlib.util
from io import StringIO
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260721_drop_workforce_manager_instructions.py"
)
REVISION = "20260721_drop_workforce_manager_instr"
DOWN_REVISION = "20260715_add_public_mcp_app_audits"
TABLE = "workforces"
COLUMN = "manager_instructions"
INDEXES = {"ix_workforces_status", "ix_workforces_manager_agent_id"}


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "drop_workforce_manager_instructions_migration", MIGRATION_PATH
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


def _create_legacy_workforces_table(connection) -> None:
    connection.exec_driver_sql("CREATE TABLE agents (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql(
        f"""
        CREATE TABLE {TABLE} (
            id INTEGER PRIMARY KEY,
            owner_user_id INTEGER NOT NULL,
            scope_type VARCHAR(50) NOT NULL,
            scope_id VARCHAR(200) NOT NULL,
            name VARCHAR(200) NOT NULL,
            description TEXT,
            manager_agent_id INTEGER NOT NULL REFERENCES agents(id),
            manager_instructions TEXT,
            status VARCHAR(20) NOT NULL,
            canvas_layout JSON,
            created_at DATETIME,
            updated_at DATETIME
        )
        """
    )
    connection.exec_driver_sql(f"CREATE INDEX ix_workforces_status ON {TABLE}(status)")
    connection.exec_driver_sql(
        f"CREATE INDEX ix_workforces_manager_agent_id ON {TABLE}(manager_agent_id)"
    )
    connection.execute(sa.text("INSERT INTO agents (id) VALUES (1)"))
    connection.execute(
        sa.text(
            f"INSERT INTO {TABLE} "
            "(id, owner_user_id, scope_type, scope_id, name, manager_agent_id, "
            "manager_instructions, status) "
            "VALUES (1, 1, 'user', '1', 'Legacy Team', 1, 'Legacy stored text', "
            "'active')"
        )
    )


def _column_names(connection) -> set[str]:
    return {column["name"] for column in sa.inspect(connection).get_columns(TABLE)}


def test_upgrade_drops_column_and_preserves_rows_indexes_and_fk() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        _create_legacy_workforces_table(connection)

        with Operations.context(_operations(connection).get_context()):
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert COLUMN not in _column_names(connection)
        assert {index["name"] for index in inspector.get_indexes(TABLE)} == INDEXES
        foreign_keys = inspector.get_foreign_keys(TABLE)
        assert len(foreign_keys) == 1
        assert foreign_keys[0]["constrained_columns"] == ["manager_agent_id"]
        assert foreign_keys[0]["referred_table"] == "agents"

        row = connection.execute(sa.text(f"SELECT id, name, status FROM {TABLE}")).one()
        assert row == (1, "Legacy Team", "active")
        assert connection.execute(sa.text("PRAGMA foreign_key_check")).fetchall() == []


def test_downgrade_restores_nullable_column_without_data() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_workforces_table(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.downgrade()

        columns = {
            column["name"]: column
            for column in sa.inspect(connection).get_columns(TABLE)
        }
        assert COLUMN in columns
        assert columns[COLUMN]["nullable"] is True
        # Structure only: the dropped value is not recoverable.
        assert connection.execute(
            sa.text(f"SELECT {COLUMN} FROM {TABLE} WHERE id = 1")
        ).one() == (None,)


def test_online_upgrade_is_idempotent_when_column_already_dropped() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_workforces_table(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.upgrade()

        assert COLUMN not in _column_names(connection)


def test_online_downgrade_is_idempotent_when_column_already_present() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_legacy_workforces_table(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.downgrade()

        columns = _column_names(connection)
        assert COLUMN in columns
        # No duplicate column was added.
        assert len([name for name in columns if name == COLUMN]) == 1


def test_online_upgrade_and_downgrade_skip_when_table_missing() -> None:
    # Fresh installs create core tables via Base.metadata.create_all() after
    # Alembic runs, so the workforces table may legitimately not exist yet.
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


@pytest.mark.parametrize("dialect_name", ["postgresql", "mysql"])
def test_offline_upgrade_emits_drop_column_without_bind_parameters(
    dialect_name: str,
) -> None:
    migration = _load_migration_module()

    sql = _offline_sql(migration, dialect_name, "upgrade")

    assert f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}" in sql
    assert "%(" not in sql
    assert ":param" not in sql
    assert "?" not in sql


@pytest.mark.parametrize("dialect_name", ["postgresql", "mysql"])
def test_offline_downgrade_emits_add_column_without_bind_parameters(
    dialect_name: str,
) -> None:
    migration = _load_migration_module()

    sql = _offline_sql(migration, dialect_name, "downgrade")

    assert f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} TEXT" in sql
    assert "NOT NULL" not in sql
    assert "%(" not in sql
    assert ":param" not in sql
    assert "?" not in sql


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == REVISION
    assert migration.down_revision == DOWN_REVISION
