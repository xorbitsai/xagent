"""Encrypted model credentials are unbounded text on supported databases."""

from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.dialects import postgresql

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.web.models.model import Model

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261009_expand_model_api_key_ciphertext.py"
)
TABLE = "models"
COLUMN = "_api_key_encrypted"


def _legacy_table(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        TABLE,
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(COLUMN, sa.String(500), nullable=False),
    )


@contextmanager
def _migration_context(engine):
    migration = load_migration_module(MIGRATION_PATH)
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            yield migration, connection


def test_revision_chain_and_postgresql_orm_type():
    migration = load_migration_module(MIGRATION_PATH)
    assert migration.revision == "20261009_model_api_key_text"
    assert migration.down_revision == "20261008_microsoft_offline_access"
    assert isinstance(Model.__table__.c[COLUMN].type, sa.Text)
    assert (
        Model.__table__.c[COLUMN].type.compile(dialect=postgresql.dialect()) == "TEXT"
    )


def test_sqlite_upgrade_preserves_ciphertext_and_safe_downgrade(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'model-key.db'}")
    metadata = sa.MetaData()
    table = _legacy_table(metadata)
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(table.insert().values(id=1, **{COLUMN: "short-ciphertext"}))

    with _migration_context(engine) as (migration, connection):
        migration.upgrade()
        reflected = sa.inspect(connection).get_columns(TABLE)
        encrypted = next(column for column in reflected if column["name"] == COLUMN)
        assert isinstance(encrypted["type"], sa.Text)
        assert (
            connection.execute(sa.text(f"SELECT {COLUMN} FROM {TABLE}")).scalar_one()
            == "short-ciphertext"
        )

        connection.execute(
            sa.text(f"UPDATE {TABLE} SET {COLUMN} = :value"),
            {"value": "x" * 1420},
        )
        with pytest.raises(RuntimeError, match="exceed 500 characters"):
            migration.downgrade()


@pytest.mark.postgresql
def test_postgresql_upgrade_accepts_long_ciphertext_and_preserves_it():
    with disposable_database_factory("xagent_model_api_key_text") as make:
        engine = make("upgrade")
        metadata = sa.MetaData()
        _legacy_table(metadata)
        metadata.create_all(engine)

        with _migration_context(engine) as (migration, connection):
            migration.upgrade()
            encrypted = next(
                column
                for column in sa.inspect(connection).get_columns(TABLE)
                if column["name"] == COLUMN
            )
            assert isinstance(encrypted["type"], sa.Text)

            ciphertext = "x" * 1420
            connection.execute(
                sa.text(f"INSERT INTO {TABLE} (id, {COLUMN}) VALUES (1, :value)"),
                {"value": ciphertext},
            )
            assert (
                connection.execute(
                    sa.text(f"SELECT {COLUMN} FROM {TABLE} WHERE id = 1")
                ).scalar_one()
                == ciphertext
            )
            with pytest.raises(RuntimeError, match="exceed 500 characters"):
                migration.downgrade()
