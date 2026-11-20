"""The Shopify store lock table migrates both ways; upgrade is idempotent."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION = "xagent.migrations.versions.20261119_shopify_store_locks"


def test_store_lock_table_migrates_both_ways(tmp_path):
    migration = importlib.import_module(MIGRATION)
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.upgrade()
        migration.upgrade()
        inspector = sa.inspect(connection)
        assert inspector.has_table("shopify_store_locks")
        assert inspector.get_pk_constraint("shopify_store_locks")[
            "constrained_columns"
        ] == ["provider", "store"]
        migration.downgrade()
        assert not sa.inspect(connection).has_table("shopify_store_locks")
