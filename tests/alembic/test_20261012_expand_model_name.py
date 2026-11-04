"""Provider model identifiers support Bedrock inference profile ARNs."""

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
    / "src/xagent/migrations/versions/20261012_expand_model_name.py"
)
TABLE = "models"
COLUMN = "model_name"
SYSTEM_PROFILE_ARN = "arn:aws:bedrock:us-west-2::inference-profile/" + "a" * 64
APPLICATION_PROFILE_ARN = (
    "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/" + "b" * 64
)


def _legacy_table(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        TABLE,
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(COLUMN, sa.String(100), nullable=False),
    )


@contextmanager
def _migration_context(engine):
    migration = load_migration_module(MIGRATION_PATH)
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            yield migration, connection


def _model_name_column(connection) -> dict:
    return next(
        column
        for column in sa.inspect(connection).get_columns(TABLE)
        if column["name"] == COLUMN
    )


def test_revision_chain_and_orm_width() -> None:
    migration = load_migration_module(MIGRATION_PATH)
    assert migration.revision == "20261012_model_name_2048"
    assert migration.down_revision == "20261010_bedrock_settings"
    assert Model.__table__.c[COLUMN].type.length == 2048
    assert (
        Model.__table__.c[COLUMN].type.compile(dialect=postgresql.dialect())
        == "VARCHAR(2048)"
    )


def test_sqlite_upgrade_round_trip_and_safe_downgrade(tmp_path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'model-name.db'}")
    metadata = sa.MetaData()
    table = _legacy_table(metadata)
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(table.insert().values(id=1, model_name="short-model"))

    with _migration_context(engine) as (migration, connection):
        migration.upgrade()
        assert _model_name_column(connection)["type"].length == 2048
        connection.execute(
            sa.text(f"UPDATE {TABLE} SET {COLUMN} = :value WHERE id = 1"),
            {"value": SYSTEM_PROFILE_ARN},
        )
        assert (
            connection.execute(
                sa.text(f"SELECT {COLUMN} FROM {TABLE} WHERE id = 1")
            ).scalar_one()
            == SYSTEM_PROFILE_ARN
        )
        with pytest.raises(RuntimeError, match="exceed 100 characters"):
            migration.downgrade()


@pytest.mark.postgresql
def test_postgresql_upgrade_supports_create_update_and_safe_downgrade() -> None:
    with disposable_database_factory("xagent_model_name_2048") as make:
        engine = make("round_trip")
        metadata = sa.MetaData()
        _legacy_table(metadata)
        metadata.create_all(engine)

        with _migration_context(engine) as (migration, connection):
            migration.upgrade()
            assert _model_name_column(connection)["type"].length == 2048
            connection.execute(
                sa.text(f"INSERT INTO {TABLE} (id, {COLUMN}) VALUES (1, :model_name)"),
                {"model_name": SYSTEM_PROFILE_ARN},
            )
            connection.execute(
                sa.text(f"UPDATE {TABLE} SET {COLUMN} = :model_name WHERE id = 1"),
                {"model_name": APPLICATION_PROFILE_ARN},
            )
            assert (
                connection.execute(
                    sa.text(f"SELECT {COLUMN} FROM {TABLE} WHERE id = 1")
                ).scalar_one()
                == APPLICATION_PROFILE_ARN
            )
            with pytest.raises(RuntimeError, match="exceed 100 characters"):
                migration.downgrade()


@pytest.mark.postgresql
def test_postgresql_short_name_survives_upgrade_and_downgrade() -> None:
    with disposable_database_factory("xagent_model_name_2048") as make:
        engine = make("downgrade")
        metadata = sa.MetaData()
        table = _legacy_table(metadata)
        metadata.create_all(engine)
        with engine.begin() as connection:
            connection.execute(table.insert().values(id=1, model_name="short-model"))

        with _migration_context(engine) as (migration, connection):
            migration.upgrade()
            migration.downgrade()
            assert _model_name_column(connection)["type"].length == 100
            assert (
                connection.execute(
                    sa.text(f"SELECT {COLUMN} FROM {TABLE} WHERE id = 1")
                ).scalar_one()
                == "short-model"
            )
