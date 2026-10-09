"""Migration coverage for native Amazon Bedrock model settings."""

from contextlib import contextmanager
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import load_migration_module

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261009_add_bedrock_model_settings.py"
)
TABLE = "models"


@contextmanager
def _migration_context(engine):
    migration = load_migration_module(MIGRATION_PATH)
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            yield migration, connection


def _legacy_table(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        TABLE,
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("_api_key_encrypted", sa.Text, nullable=False),
    )


def test_revision_follows_unified_model_token_migration() -> None:
    migration = load_migration_module(MIGRATION_PATH)
    assert migration.revision == "20261009_bedrock_settings"
    assert migration.down_revision == "20261009_model_api_key_text"


def test_sqlite_upgrade_preserves_long_token_and_bedrock_settings(tmp_path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'bedrock-settings.db'}")
    metadata = sa.MetaData()
    table = _legacy_table(metadata)
    metadata.create_all(engine)
    ciphertext = "x" * 1420
    with engine.begin() as connection:
        connection.execute(table.insert().values(id=1, _api_key_encrypted=ciphertext))

    with _migration_context(engine) as (migration, connection):
        migration.upgrade()
        columns = {
            column["name"] for column in sa.inspect(connection).get_columns(TABLE)
        }
        assert {"bedrock_region", "bedrock_auth_mode"} <= columns

        connection.execute(
            sa.text(
                "UPDATE models SET bedrock_region = :region, "
                "bedrock_auth_mode = :auth_mode WHERE id = 1"
            ),
            {"region": "us-west-2", "auth_mode": "api_key"},
        )
        row = connection.execute(
            sa.text(
                "SELECT _api_key_encrypted, bedrock_region, bedrock_auth_mode "
                "FROM models WHERE id = 1"
            )
        ).one()
        assert tuple(row) == (ciphertext, "us-west-2", "api_key")

        migration.downgrade()
        columns = {
            column["name"] for column in sa.inspect(connection).get_columns(TABLE)
        }
        assert "bedrock_region" not in columns
        assert "bedrock_auth_mode" not in columns
        assert (
            connection.execute(
                sa.text("SELECT _api_key_encrypted FROM models WHERE id = 1")
            ).scalar_one()
            == ciphertext
        )
