"""Existing model rows survive adding and removing speech settings."""

from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import load_migration_module

MIGRATION_PATH = (
    Path(__file__).parents[2]
    / "src/xagent/migrations/versions/20261031_add_speech_model_settings.py"
)


def test_speech_settings_migration_preserves_rows_and_handles_existing_columns():
    migration = load_migration_module(MIGRATION_PATH)
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    table = sa.Table(
        "models",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("voice", sa.String(255), nullable=True),
    )
    metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            connection.execute(
                table.insert().values(id=1, model_name="existing", voice="saved-voice")
            )
            with Operations.context(MigrationContext.configure(connection)):
                migration.upgrade()
                migration.upgrade()
                row = connection.execute(
                    sa.text(
                        "SELECT model_name, language, voice, format, sample_rate FROM models"
                    )
                ).one()
                assert tuple(row) == ("existing", None, "saved-voice", None, None)
                connection.execute(
                    sa.text(
                        "UPDATE models SET language='en', format='pcm', sample_rate=16000"
                    )
                )
                assert (
                    connection.execute(
                        sa.text("SELECT sample_rate FROM models")
                    ).scalar_one()
                    == 16000
                )
                migration.downgrade()
                assert (
                    connection.execute(
                        sa.text("SELECT model_name FROM models")
                    ).scalar_one()
                    == "existing"
                )
                assert not set(migration.COLUMNS) & {
                    column["name"]
                    for column in sa.inspect(connection).get_columns("models")
                }
    finally:
        engine.dispose()
