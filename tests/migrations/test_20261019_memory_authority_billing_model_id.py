"""The memory-authority billing column migrates both ways and is idempotent."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from xagent.web.models.global_memory_embedding_authority import (
    GlobalMemoryEmbeddingAuthority,
)

MIGRATION = "xagent.migrations.versions.20261019_memory_authority_billing_model_id"


def _columns(connection) -> dict:
    return {
        column["name"]: column
        for column in sa.inspect(connection).get_columns(
            GlobalMemoryEmbeddingAuthority.__tablename__
        )
    }


def test_billing_model_id_migrates_both_ways_and_is_idempotent(tmp_path):
    migration = importlib.import_module(MIGRATION)
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    GlobalMemoryEmbeddingAuthority.__table__.create(engine)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.downgrade()
        migration.downgrade()
        assert "billing_model_id" not in _columns(connection)

        migration.upgrade()
        migration.upgrade()
        column = _columns(connection)["billing_model_id"]
        # Existing authorities keep recording, unpriced, until an operator
        # names a billing model; no default stamps one on them.
        assert column["nullable"] is True and column["default"] is None
        assert set(_columns(connection)) == set(
            GlobalMemoryEmbeddingAuthority.__table__.columns.keys()
        )


def test_missing_table_is_left_to_the_table_migration(tmp_path):
    migration = importlib.import_module(MIGRATION)
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.upgrade()
        migration.downgrade()
        assert not sa.inspect(connection).has_table(
            GlobalMemoryEmbeddingAuthority.__tablename__
        )
