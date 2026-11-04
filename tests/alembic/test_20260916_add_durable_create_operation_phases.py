import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.exc import IntegrityError

ROOT = Path(__file__).parent.parent.parent
BASE_PATH = (
    ROOT / "src/xagent/migrations/versions/20260914_add_durable_sandbox_lifecycles.py"
)
MIGRATION_PATH = (
    ROOT
    / "src/xagent/migrations/versions/20260916_add_durable_create_operation_phases.py"
)
TABLE = "durable_sandbox_lifecycles"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operations(connection: sa.Connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _create_old_schema(connection: sa.Connection) -> None:
    migration = _module(BASE_PATH, "durable_lifecycle_base_migration")
    with patch.object(migration, "op", _operations(connection)):
        migration.upgrade()


def _insert_old_row(connection: sa.Connection, *, scope: str = "a" * 64) -> None:
    connection.execute(
        sa.text(
            f"INSERT INTO {TABLE} "
            "(scope_digest, lifecycle_token, backend_lifecycle_digest, owner_token, "
            "version, state, task_id, run_id, lease_attempt_id, eligible_at, "
            "owner_lease_expires_at, delete_attempts) VALUES "
            "(:scope, :lifecycle, :backend, :owner, 1, 'registered', 1, 'r', 'a', "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)"
        ),
        {
            "scope": scope,
            "lifecycle": "b" * 64,
            "backend": "c" * 64,
            "owner": "d" * 64,
        },
    )


def _upgrade(connection: sa.Connection):
    migration = _module(MIGRATION_PATH, "durable_create_operation_migration")
    with patch.object(migration, "op", _operations(connection)):
        migration.upgrade()
    return migration


def test_sqlite_upgrade_backfills_opaque_operation_and_active_generation() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _create_old_schema(connection)
        _insert_old_row(connection)
        _upgrade(connection)
        row = connection.execute(
            sa.text(
                f"SELECT scope_digest, active_scope_digest, "
                f"create_operation_token, create_phase, "
                f"create_terminal_outcome FROM {TABLE}"
            )
        ).one()
        assert row.active_scope_digest == row.scope_digest == "a" * 64
        assert len(row.create_operation_token) == 64
        assert row.create_operation_token not in {"a" * 64, "b" * 64, "c" * 64}
        assert row.create_phase == "not_started"
        assert row.create_terminal_outcome is None

        uniques = {
            item["name"]
            for item in sa.inspect(connection).get_unique_constraints(TABLE)
        }
        assert "uq_dsl_scope_digest" not in uniques
        assert {
            "uq_dsl_active_scope_digest",
            "uq_dsl_create_operation_token",
            "uq_dsl_lifecycle_token",
            "uq_dsl_backend_lifecycle_digest",
        }.issubset(uniques)
        indexes = {item["name"] for item in sa.inspect(connection).get_indexes(TABLE)}
        assert "ix_dsl_scope_digest" in indexes
        plan = connection.execute(
            sa.text(
                f"EXPLAIN QUERY PLAN SELECT * FROM {TABLE} WHERE scope_digest = :scope"
            ),
            {"scope": "a" * 64},
        ).all()
        assert any("ix_dsl_scope_digest" in str(item) for item in plan)


def test_sqlite_upgrade_backfills_existing_tombstone_as_quarantined() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _create_old_schema(connection)
        _insert_old_row(connection)
        connection.execute(
            sa.text(
                f"UPDATE {TABLE} SET state = 'deleting', "
                "deleting_at = CURRENT_TIMESTAMP, "
                "delete_claim_expires_at = CURRENT_TIMESTAMP"
            )
        )
        _upgrade(connection)
        row = connection.execute(
            sa.text(f"SELECT active_scope_digest, create_phase FROM {TABLE}")
        ).one()
        assert row.active_scope_digest is None
        assert row.create_phase == "not_started"


def test_sqlite_one_active_plus_multiple_quarantined_generations() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _create_old_schema(connection)
        _upgrade(connection)
        statement = sa.text(
            f"INSERT INTO {TABLE} "
            "(scope_digest, active_scope_digest, lifecycle_token, "
            "backend_lifecycle_digest, owner_token, create_operation_token, "
            "create_phase, version, state, task_id, run_id, lease_attempt_id, "
            "eligible_at, owner_lease_expires_at, deleting_at, "
            "delete_claim_expires_at, delete_attempts) VALUES "
            "(:scope, :active, :lifecycle, :backend, :owner, :operation, "
            ":phase, 1, :state, 1, 'r', 'a', CURRENT_TIMESTAMP, "
            "CURRENT_TIMESTAMP, :deleting_at, :claim, 0)"
        )
        base = {
            "scope": "a" * 64,
            "active": "a" * 64,
            "lifecycle": "b" * 64,
            "backend": "c" * 64,
            "owner": "d" * 64,
            "operation": "e" * 64,
            "phase": "not_started",
            "state": "registered",
            "deleting_at": None,
            "claim": None,
        }
        connection.execute(statement, base)
        with pytest.raises(IntegrityError):
            connection.execute(
                statement,
                {
                    **base,
                    "lifecycle": "1" * 64,
                    "backend": "2" * 64,
                    "operation": "3" * 64,
                },
            )

        quarantine = {
            **base,
            "active": None,
            "lifecycle": "4" * 64,
            "backend": "5" * 64,
            "operation": "6" * 64,
            "phase": "may_publish",
            "state": "deleting",
            "deleting_at": "2026-09-16 00:00:00",
            "claim": "2026-09-16 00:01:00",
        }
        connection.execute(statement, quarantine)
        connection.execute(
            statement,
            {
                **quarantine,
                "lifecycle": "7" * 64,
                "backend": "8" * 64,
                "operation": "9" * 64,
            },
        )
        assert (
            connection.execute(sa.text(f"SELECT count(*) FROM {TABLE}")).scalar() == 3
        )


def test_sqlite_rejects_invalid_active_shape_and_phase() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _create_old_schema(connection)
        _insert_old_row(connection)
        _upgrade(connection)
        with pytest.raises(IntegrityError):
            connection.execute(
                sa.text(
                    f"UPDATE {TABLE} SET active_scope_digest = NULL "
                    "WHERE state = 'registered'"
                )
            )
        with pytest.raises(IntegrityError):
            connection.execute(sa.text(f"UPDATE {TABLE} SET create_phase = 'unknown'"))
        with pytest.raises(IntegrityError):
            connection.execute(sa.text(f"UPDATE {TABLE} SET create_phase = 'terminal'"))
        with pytest.raises(IntegrityError):
            connection.execute(
                sa.text(f"UPDATE {TABLE} SET create_terminal_outcome = 'success'")
            )


def test_upgrade_is_idempotent_and_downgrade_refuses_coexisting_generations() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _create_old_schema(connection)
        _insert_old_row(connection)
        migration = _upgrade(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        connection.execute(
            sa.text(
                f"UPDATE {TABLE} SET state = 'deleting', active_scope_digest = NULL, "
                "deleting_at = CURRENT_TIMESTAMP, "
                "delete_claim_expires_at = CURRENT_TIMESTAMP"
            )
        )
        connection.execute(
            sa.text(
                f"INSERT INTO {TABLE} "
                "(scope_digest, active_scope_digest, lifecycle_token, "
                "backend_lifecycle_digest, owner_token, create_operation_token, "
                "create_phase, version, state, task_id, run_id, lease_attempt_id, "
                "eligible_at, owner_lease_expires_at, delete_attempts) VALUES "
                "(:scope, :scope, :lifecycle, :backend, :owner, :operation, "
                "'not_started', 1, 'registered', 2, 'r2', 'a2', "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)"
            ),
            {
                "scope": "a" * 64,
                "lifecycle": "1" * 64,
                "backend": "2" * 64,
                "owner": "3" * 64,
                "operation": "4" * 64,
            },
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="successor generations coexist"):
                migration.downgrade()


@pytest.mark.parametrize("phase", ["may_publish", "observed"])
def test_downgrade_refuses_ambiguous_generation(phase: str) -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _create_old_schema(connection)
        _insert_old_row(connection)
        migration = _upgrade(connection)
        connection.execute(
            sa.text(
                f"UPDATE {TABLE} SET state = 'deleting', "
                "active_scope_digest = NULL, create_phase = :phase, "
                "deleting_at = CURRENT_TIMESTAMP, "
                "delete_claim_expires_at = CURRENT_TIMESTAMP"
            ),
            {"phase": phase},
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="ambiguous generation"):
                migration.downgrade()


def test_safe_downgrade_and_reupgrade_round_trip() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _create_old_schema(connection)
        _insert_old_row(connection)
        migration = _upgrade(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        columns = {item["name"] for item in sa.inspect(connection).get_columns(TABLE)}
        assert "create_phase" not in columns
        assert "create_terminal_outcome" not in columns
        assert "uq_dsl_scope_digest" in {
            item["name"]
            for item in sa.inspect(connection).get_unique_constraints(TABLE)
        }

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        row = connection.execute(
            sa.text(f"SELECT create_phase, create_terminal_outcome FROM {TABLE}")
        ).one()
        assert row.create_phase == "not_started"
        assert row.create_terminal_outcome is None
