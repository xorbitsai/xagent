"""Migration coverage for per-model OpenRouter endpoint allowlists."""

from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import disposable_database_factory

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261126_add_openrouter_provider_only.py"
)
TABLE = "models"


def _load_migration_module() -> ModuleType:
    spec = spec_from_file_location(
        "openrouter_provider_policy_migration", MIGRATION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _migration_context(engine):
    migration = _load_migration_module()
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            yield migration, connection


def _legacy_table(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        TABLE,
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False),
    )


def test_revision_follows_shopify_store_lock_migration() -> None:
    migration = _load_migration_module()
    assert migration.revision == "20261126_openrouter_routing"
    assert migration.down_revision == "20261119_shopify_store_locks"


def _offline_sql(dialect: str, operation: str) -> str:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


def test_postgresql_ddl_has_a_transaction_local_lock_timeout() -> None:
    for operation in ("upgrade", "downgrade"):
        sql = _offline_sql("postgresql", operation)
        assert sql.count("SET LOCAL lock_timeout = '5s'") == 1
        assert sql.index("SET LOCAL lock_timeout") < sql.index("ALTER TABLE models")
        assert sql.count("ALTER TABLE models") == 3


def test_sqlite_ddl_does_not_set_a_postgresql_lock_timeout() -> None:
    for operation in ("upgrade", "downgrade"):
        sql = _offline_sql("sqlite", operation)
        assert "lock_timeout" not in sql
        assert sql.count("ALTER TABLE models") == 3


@pytest.mark.postgresql
def test_postgresql_upgrade_and_downgrade_restore_the_lock_timeout() -> None:
    with disposable_database_factory("xagent_openrouter_routing") as make_database:
        engine = make_database("round_trip")
        metadata = sa.MetaData()
        table = _legacy_table(metadata)
        metadata.create_all(engine)
        with engine.begin() as connection:
            connection.execute(table.insert().values(id=1, model_name="deepseek/test"))

        with _migration_context(engine) as (migration, connection):
            previous = connection.execute(sa.text("SHOW lock_timeout")).scalar_one()
            migration.upgrade()
            assert connection.execute(sa.text("SHOW lock_timeout")).scalar_one() == "5s"
            columns = {
                column["name"] for column in sa.inspect(connection).get_columns(TABLE)
            }
            assert {
                "openrouter_provider_only",
                "openrouter_zdr",
                "openrouter_data_collection",
            } <= columns
            migration.downgrade()

        with engine.connect() as connection:
            assert (
                connection.execute(sa.text("SHOW lock_timeout")).scalar_one()
                == previous
            )
            columns = {
                column["name"] for column in sa.inspect(connection).get_columns(TABLE)
            }
            assert "openrouter_provider_only" not in columns
            assert "openrouter_zdr" not in columns
            assert "openrouter_data_collection" not in columns
            assert (
                connection.execute(
                    sa.text("SELECT model_name FROM models WHERE id = 1")
                ).scalar_one()
                == "deepseek/test"
            )


def test_sqlite_upgrade_round_trips_openrouter_allowlist(tmp_path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'openrouter-policy.db'}")
    metadata = sa.MetaData()
    table = _legacy_table(metadata)
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(table.insert().values(id=1, model_name="deepseek/test"))

    with _migration_context(engine) as (migration, connection):
        migration.upgrade()
        columns = {
            column["name"] for column in sa.inspect(connection).get_columns(TABLE)
        }
        assert {
            "openrouter_provider_only",
            "openrouter_zdr",
            "openrouter_data_collection",
        } <= columns

        allowlist = '["together/fp8", "inference-net/fp8"]'
        connection.execute(
            sa.text(
                "UPDATE models SET openrouter_provider_only = :allowlist, "
                "openrouter_zdr = :zdr, openrouter_data_collection = :collection "
                "WHERE id = 1"
            ),
            {"allowlist": allowlist, "zdr": True, "collection": "deny"},
        )
        row = connection.execute(
            sa.text(
                "SELECT openrouter_provider_only, openrouter_zdr, "
                "openrouter_data_collection FROM models WHERE id = 1"
            )
        ).one()
        assert tuple(row) == (allowlist, 1, "deny")

        migration.downgrade()
        columns = {
            column["name"] for column in sa.inspect(connection).get_columns(TABLE)
        }
        assert {
            "openrouter_provider_only",
            "openrouter_zdr",
            "openrouter_data_collection",
        }.isdisjoint(columns)
        assert (
            connection.execute(
                sa.text("SELECT model_name FROM models WHERE id = 1")
            ).scalar_one()
            == "deepseek/test"
        )
