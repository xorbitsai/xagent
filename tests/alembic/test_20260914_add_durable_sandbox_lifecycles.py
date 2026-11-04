import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.exc import IntegrityError

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260914_add_durable_sandbox_lifecycles.py"
)
TABLE = "durable_sandbox_lifecycles"


def _migration_module():
    spec = importlib.util.spec_from_file_location(
        "durable_lifecycle_migration", MIGRATION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operations(connection: sa.Connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def test_sqlite_upgrade_builds_constraints_indexes_and_no_backfill() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        inspector = sa.inspect(connection)
        columns = {column["name"] for column in inspector.get_columns(TABLE)}
        assert columns == {
            "id",
            "scope_digest",
            "lifecycle_token",
            "backend_lifecycle_digest",
            "owner_token",
            "version",
            "state",
            "task_id",
            "run_id",
            "lease_attempt_id",
            "turn_digest",
            "eligible_at",
            "owner_lease_expires_at",
            "delete_claim_expires_at",
            "retry_at",
            "registered_at",
            "ready_at",
            "deleting_at",
            "updated_at",
            "delete_attempts",
        }
        assert {index["name"] for index in inspector.get_indexes(TABLE)} == {
            "ix_dsl_reclaim",
            "ix_dsl_task_attempt",
        }
        assert {item["name"] for item in inspector.get_unique_constraints(TABLE)} == {
            "uq_dsl_scope_digest",
            "uq_dsl_lifecycle_token",
            "uq_dsl_backend_lifecycle_digest",
        }
        assert (
            connection.execute(sa.text(f"SELECT count(*) FROM {TABLE}")).scalar() == 0
        )


def test_sqlite_checks_reject_invalid_state_shape_and_duplicate_scope() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    values = {
        "scope": "a" * 64,
        "lifecycle": "b" * 64,
        "backend": "c" * 64,
        "owner": "d" * 64,
    }
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        statement = sa.text(
            f"INSERT INTO {TABLE} "
            "(scope_digest, lifecycle_token, backend_lifecycle_digest, owner_token, "
            "version, state, task_id, run_id, lease_attempt_id, eligible_at, "
            "owner_lease_expires_at, delete_attempts) VALUES "
            "(:scope, :lifecycle, :backend, :owner, 1, 'registered', 1, 'r', 'a', "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)"
        )
        connection.execute(statement, values)
        with pytest.raises(IntegrityError):
            connection.execute(
                statement,
                {**values, "lifecycle": "e" * 64, "backend": "f" * 64},
            )


def test_upgrade_and_downgrade_are_idempotent() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
            migration.downgrade()
            migration.downgrade()
        assert TABLE not in sa.inspect(connection).get_table_names()
