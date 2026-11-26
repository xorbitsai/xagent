"""Migration coverage for per-model OpenRouter endpoint allowlists."""

from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

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
