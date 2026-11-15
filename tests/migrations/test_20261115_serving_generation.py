from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from tests.shared.postgres_disposable import load_migration_module
from xagent.db.config import create_alembic_config
from xagent.web.models.database import Base

MIGRATION = load_migration_module(
    Path(__file__).parents[2]
    / "src/xagent/migrations/versions/20261115_serving_generation.py"
)


def _run(connection, fn) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        fn()


def test_upgrade_adds_only_the_single_row_table(tmp_path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    with engine.begin() as connection:
        _run(connection, MIGRATION.upgrade)
        _run(connection, MIGRATION.upgrade)  # idempotent
        inspector = sa.inspect(connection)
        assert inspector.get_table_names() == ["serving_generation"]
        model = Base.metadata.tables["serving_generation"]
        assert {c["name"] for c in inspector.get_columns("serving_generation")} == set(
            model.columns.keys()
        )
        connection.execute(
            sa.text("INSERT INTO serving_generation (id, generation) VALUES (1, 'a')")
        )
        try:
            connection.execute(
                sa.text(
                    "INSERT INTO serving_generation (id, generation) VALUES (2, 'b')"
                )
            )
        except sa.exc.IntegrityError:
            pass
        else:  # pragma: no cover
            raise AssertionError("a second row must be refused")
    with engine.begin() as connection:
        _run(connection, MIGRATION.downgrade)
        assert "serving_generation" not in sa.inspect(connection).get_table_names()


def test_revision_chains_onto_the_previous_head(tmp_path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'h.db'}")
    script = ScriptDirectory.from_config(create_alembic_config(engine))
    revision = script.get_revision("20261115_serving_generation")
    assert revision.down_revision == "20261114_oauth_revocation_fences"
    assert len(script.get_heads()) == 1


def test_offline_sql_generation_emits_the_create_table() -> None:
    from io import StringIO

    buffer = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buffer}
    )
    with Operations.context(context):
        MIGRATION.upgrade()
    sql = buffer.getvalue()
    assert "CREATE TABLE serving_generation" in sql
    assert "ck_serving_generation_single_row" in sql
    assert "ALTER" not in sql and "INSERT" not in sql
