"""Exercise the category data upgrade against actual stored JSON values."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa


def _migration():
    path = (
        Path(__file__).resolve().parents[2]
        / "src/xagent/migrations/versions/20261107_split_tool_categories.py"
    )
    spec = importlib.util.spec_from_file_location("split_tool_categories", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASES = [
    (["basic"], ["basic_tools", "code_execution", "web_tools"]),
    (["web_search"], ["web_tools"]),
    (["basic", "web_search"], ["basic_tools", "code_execution", "web_tools"]),
    (
        ["file", "web_search", "basic", "mcp:GitHub", "code_execution", "web_tools"],
        ["file", "web_tools", "basic_tools", "code_execution", "mcp:GitHub"],
    ),
    ([], []),
    (None, None),
    (
        ["basic_tools", "code_execution", "web_tools"],
        ["basic_tools", "code_execution", "web_tools"],
    ),
    (["knowledge", "mcp:Google", "ssh"], ["knowledge", "mcp:Google", "ssh"]),
    ("basic", "basic"),
]


def test_upgrade_migrates_agents_and_task_configs_in_batches(monkeypatch):
    migration = _migration()
    monkeypatch.setattr(migration, "BATCH_SIZE", 2)
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    agents = sa.Table(
        "agents",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tool_categories", sa.JSON),
    )
    tasks = sa.Table(
        "tasks",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("agent_config", sa.JSON),
    )
    metadata.create_all(engine)
    # Identically named fields in prompt data are not configuration grants.
    context = {"tool_categories": ["basic"], "message": "web_search"}
    with engine.begin() as connection:
        connection.execute(
            agents.insert(),
            [
                {"id": i * 3, "tool_categories": old}
                for i, (old, _) in enumerate(CASES, start=1)
            ],
        )
        connection.execute(
            tasks.insert(),
            [
                {
                    "id": i * 3,
                    "agent_config": {
                        "tool_categories": old,
                        "instructions": "Keep this",
                        "prompt_context": context,
                    },
                }
                for i, (old, _) in enumerate(CASES, start=1)
            ]
            + [
                {"id": 100, "agent_config": None},
                {"id": 101, "agent_config": {}},
            ],
        )
        with patch.object(migration.op, "get_bind", return_value=connection):
            migration.upgrade()
            # A second pass does not expand categories again.
            migration.upgrade()
            migration.downgrade()
        stored_agents = (
            connection.execute(
                sa.select(agents.c.tool_categories).order_by(agents.c.id)
            )
            .scalars()
            .all()
        )
        stored_tasks = (
            connection.execute(sa.select(tasks.c.agent_config).order_by(tasks.c.id))
            .scalars()
            .all()
        )
    assert stored_agents == [new for _, new in CASES]
    assert stored_tasks == [
        {"tool_categories": new, "instructions": "Keep this", "prompt_context": context}
        for _, new in CASES
    ] + [None, {}]


@pytest.mark.parametrize("table_name", [None, "agents", "tasks"])
def test_upgrade_tolerates_missing_tables_and_columns(table_name):
    migration = _migration()
    engine = sa.create_engine("sqlite:///:memory:")
    if table_name:
        metadata = sa.MetaData()
        sa.Table(table_name, metadata, sa.Column("id", sa.Integer, primary_key=True))
        metadata.create_all(engine)
    with engine.begin() as connection:
        with patch.object(migration.op, "get_bind", return_value=connection):
            migration.upgrade()


def test_bundled_templates_only_use_new_categories():
    import yaml

    root = Path(__file__).resolve().parents[2] / "src/xagent/templates"

    def check(value):
        if isinstance(value, dict):
            categories = value.get("tool_categories", [])
            assert "basic" not in categories
            assert "web_search" not in categories
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    for path in root.rglob("*.yaml"):
        check(yaml.safe_load(path.read_text()))
