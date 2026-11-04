import importlib.util
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa


def _load_migration_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "src/xagent/migrations/versions/20260721_backfill_strip_agent_category.py"
    )
    spec = importlib.util.spec_from_file_location(
        "migration_20260721_backfill_strip_agent_category", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upgrade_strips_agent_from_existing_agents() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    agents = sa.Table(
        "agents",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tool_categories", sa.JSON, nullable=True),
    )
    metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            agents.insert(),
            [
                {"id": 1, "tool_categories": ["agent"]},
                {"id": 2, "tool_categories": ["basic", "agent", "mcp:my_server"]},
                {"id": 3, "tool_categories": ["basic"]},
                {"id": 4, "tool_categories": None},
            ],
        )

        with patch.object(migration.op, "get_bind", return_value=conn):
            migration.upgrade()

        rows = conn.execute(
            sa.select(agents.c.id, agents.c.tool_categories).order_by(agents.c.id)
        ).all()

    assert rows == [
        (1, []),
        (2, ["basic", "mcp:my_server"]),
        (3, ["basic"]),
        (4, None),
    ]


def test_upgrade_is_noop_when_agents_table_missing() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as conn:
        with patch.object(migration.op, "get_bind", return_value=conn):
            migration.upgrade()  # must not raise


def test_upgrade_is_noop_when_tool_categories_column_missing() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    agents = sa.Table(
        "agents",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
    )
    metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(agents.insert(), [{"id": 1}])

        with patch.object(migration.op, "get_bind", return_value=conn):
            migration.upgrade()  # must not raise

        rows = conn.execute(sa.select(agents.c.id)).all()

    assert rows == [(1,)]
