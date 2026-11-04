"""Tests for migration 20261018_agent_prompt_context_fields.

Follows the repo migration-test convention (see
``test_20260728_add_agent_template_id_and_name_uniqueness.py``): create the
minimal pre-migration schema, stamp the parent revision, run only this
migration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

PARENT_REVISION = "20261019_zendesk_marketplace"
TARGET_REVISION = "20261018_agent_prompt_context_fields"

_CREATE_USERS_TABLE = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    is_admin BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME,
    updated_at DATETIME
)
"""

# All Agent columns before prompt_context_fields, whose addition is under test.
_CREATE_AGENTS_TABLE = """
CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    team_id INTEGER,
    visibility VARCHAR(20) NOT NULL DEFAULT 'team',
    name VARCHAR(200) NOT NULL,
    description TEXT,
    instructions TEXT,
    template_id VARCHAR(255),
    execution_mode VARCHAR(20) NOT NULL,
    models JSON,
    knowledge_bases JSON,
    skills JSON,
    tool_categories JSON,
    suggested_prompts JSON,
    logo_url VARCHAR(500),
    widget_enabled BOOLEAN NOT NULL,
    allowed_domains JSON,
    widget_key VARCHAR(255) UNIQUE,
    share_enabled BOOLEAN NOT NULL,
    share_token VARCHAR(255),
    share_updated_at DATETIME,
    origin VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    published_at DATETIME,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
)
"""


@pytest.fixture
def config_at_parent_revision(tmp_path: Path) -> Config:
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    config = Config()
    config.set_main_option("sqlalchemy.url", db_url)
    config.set_main_option("script_location", "src/xagent/migrations")

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        conn.execute(
            text(
                f"INSERT INTO alembic_version (version_num) VALUES ('{PARENT_REVISION}')"
            )
        )
        conn.execute(text(_CREATE_USERS_TABLE))
        conn.execute(text(_CREATE_AGENTS_TABLE))
        conn.execute(
            text(
                "INSERT INTO users (id, username, password_hash, is_admin) "
                "VALUES (1, 'owner', 'hash', 0)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO agents (id, user_id, name, execution_mode, "
                "widget_enabled, share_enabled, origin, status, created_at, "
                "updated_at) VALUES (1, 1, 'Agent', 'balanced', 1, 0, 'user', "
                "'draft', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
    engine.dispose()
    return config


def _columns(config: Config) -> set[str]:
    engine = create_engine(config.get_main_option("sqlalchemy.url"))
    try:
        return {column["name"] for column in inspect(engine).get_columns("agents")}
    finally:
        engine.dispose()


def test_upgrade_adds_nullable_column_and_preserves_rows(
    config_at_parent_revision: Config,
) -> None:
    command.upgrade(config_at_parent_revision, TARGET_REVISION)

    assert "prompt_context_fields" in _columns(config_at_parent_revision)
    engine = create_engine(config_at_parent_revision.get_main_option("sqlalchemy.url"))
    try:
        with engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT name, prompt_context_fields FROM agents WHERE id = 1")
            ).one()
    finally:
        engine.dispose()
    assert row.name == "Agent"
    assert row.prompt_context_fields is None


def test_downgrade_removes_column(config_at_parent_revision: Config) -> None:
    command.upgrade(config_at_parent_revision, TARGET_REVISION)
    command.downgrade(config_at_parent_revision, PARENT_REVISION)

    assert "prompt_context_fields" not in _columns(config_at_parent_revision)


def test_upgrade_keeps_column_already_present(
    config_at_parent_revision: Config,
) -> None:
    engine = create_engine(config_at_parent_revision.get_main_option("sqlalchemy.url"))
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE agents ADD COLUMN prompt_context_fields JSON"))
    engine.dispose()

    command.upgrade(config_at_parent_revision, TARGET_REVISION)

    assert "prompt_context_fields" in _columns(config_at_parent_revision)


def test_downgrade_succeeds_when_column_already_absent(
    config_at_parent_revision: Config,
) -> None:
    engine = create_engine(config_at_parent_revision.get_main_option("sqlalchemy.url"))
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE alembic_version SET version_num = :target"),
            {"target": TARGET_REVISION},
        )
    engine.dispose()

    command.downgrade(config_at_parent_revision, PARENT_REVISION)

    assert "prompt_context_fields" not in _columns(config_at_parent_revision)
