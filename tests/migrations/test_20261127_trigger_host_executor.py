from __future__ import annotations

import importlib.util
from io import StringIO
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from xagent.db.config import create_alembic_config

_PATH = (
    Path(__file__).parents[2]
    / "src/xagent/migrations/versions/20261127_trigger_host_executor.py"
)
_SPEC = importlib.util.spec_from_file_location("trigger_host_executor_migration", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
MIGRATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MIGRATION)


def _run(connection, fn) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        fn()


def _columns(connection) -> set[str]:
    return {c["name"] for c in sa.inspect(connection).get_columns("agent_triggers")}


def test_upgrade_adds_nullable_executor_idempotently_and_downgrades(tmp_path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    with engine.begin() as connection:
        connection.execute(
            sa.text("CREATE TABLE agent_triggers (id INTEGER PRIMARY KEY)")
        )
        connection.execute(sa.text("INSERT INTO agent_triggers (id) VALUES (1)"))
        _run(connection, MIGRATION.upgrade)
        _run(connection, MIGRATION.upgrade)
        assert "executor" in _columns(connection)
        assert (
            connection.execute(
                sa.text("SELECT executor FROM agent_triggers WHERE id = 1")
            ).scalar_one()
            is None
        )
    with engine.begin() as connection:
        _run(connection, MIGRATION.downgrade)
        assert "executor" not in _columns(connection)


def test_revision_chains_onto_the_previous_head(tmp_path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'h.db'}")
    script = ScriptDirectory.from_config(create_alembic_config(engine))
    revision = script.get_revision("20261127_trigger_host_executor")
    assert revision.down_revision == "20261127_mixpanel_description"
    assert len(script.get_heads()) == 1


def test_postgres_ddl_sets_lock_timeout_offline() -> None:
    buffer = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buffer}
    )
    with Operations.context(context):
        MIGRATION.upgrade()
    sql = buffer.getvalue()
    assert "lock_timeout" in sql
    assert "ADD COLUMN executor VARCHAR(64)" in sql
