import importlib.util
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa


def _load_migration_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "src/xagent/migrations/versions/20260611_backfill_basic_agents_web_search.py"
    )
    spec = importlib.util.spec_from_file_location(
        "migration_20260611_backfill_basic_agents_web_search", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upgrade_backfills_web_search_for_existing_basic_agents() -> None:
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
                {"id": 1, "tool_categories": ["basic"]},
                {"id": 2, "tool_categories": ["basic", "web_search"]},
                {"id": 3, "tool_categories": ["file"]},
                {"id": 4, "tool_categories": None},
            ],
        )

        with patch.object(migration.op, "get_bind", return_value=conn):
            migration.upgrade()

        rows = conn.execute(
            sa.select(agents.c.id, agents.c.tool_categories).order_by(agents.c.id)
        ).all()

    assert rows == [
        (1, ["basic", "web_search"]),
        (2, ["basic", "web_search"]),
        (3, ["file"]),
        (4, None),
    ]
