"""Tests for migration 20260728_add_agent_template_id_and_name_uniqueness.

Covers:
- template_id column + index added on upgrade
- the per-user (user_id, name) partial unique index, excluding
  workforce_generated_manager agents
- the pre-index dedupe rename pass, including the rename-target-collision
  guard (regression test for PR review finding F3)
- downgrade removing the column/indexes without attempting to restore
  renamed names (documented, not a bug - see F4)

Following this repo's existing migration-test convention (see
tests/migrations/test_custom_api_migration.py): rather than replaying the
full alembic history, the minimal pre-migration `users`/`agents` schema is
created directly with raw DDL, stamped to this migration's parent revision,
and only the migration under test is run against it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

PARENT_REVISION = "20260801_add_trigger_consecutive_prepare_failures"
TARGET_REVISION = "20260728_add_agent_template_id_and_name_uniqueness"

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

# Mirrors Agent's pre-migration columns (everything except template_id,
# which this migration under test adds).
_CREATE_AGENTS_TABLE = """
CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    team_id INTEGER,
    visibility VARCHAR(20) NOT NULL DEFAULT 'team',
    name VARCHAR(200) NOT NULL,
    description TEXT,
    instructions TEXT,
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
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def config_at_parent_revision(db_url: str) -> Config:
    """Alembic config for a database already at this migration's parent
    revision: `agents`/`users` exist with their pre-migration schema."""
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
    engine.dispose()

    return config


def _insert_user(conn: Any, user_id: int) -> None:
    conn.execute(
        sa.text(
            "INSERT INTO users (id, username, password_hash, is_admin) "
            "VALUES (:id, :username, 'hash', 0)"
        ),
        {"id": user_id, "username": f"user_{user_id}"},
    )


def _insert_agent(
    conn: Any,
    *,
    agent_id: int,
    user_id: int,
    name: str,
    origin: str = "user",
) -> None:
    conn.execute(
        sa.text(
            "INSERT INTO agents "
            "(id, user_id, name, execution_mode, widget_enabled, share_enabled, "
            "origin, status, created_at, updated_at) "
            "VALUES (:id, :user_id, :name, 'balanced', 1, 0, :origin, 'draft', "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"id": agent_id, "user_id": user_id, "name": name, "origin": origin},
    )


def _agent_names_by_id(engine: Any) -> dict[int, str]:
    with engine.begin() as conn:
        rows = conn.execute(sa.text("SELECT id, name FROM agents")).fetchall()
    return {row[0]: row[1] for row in rows}


def _insert_agent_with_template(
    conn: Any,
    *,
    agent_id: int,
    user_id: int,
    name: str,
    template_id: str,
    origin: str = "template_quick_access",
) -> None:
    """Only valid post-migration - template_id doesn't exist beforehand."""
    conn.execute(
        sa.text(
            "INSERT INTO agents "
            "(id, user_id, name, execution_mode, widget_enabled, share_enabled, "
            "origin, status, template_id, created_at, updated_at) "
            "VALUES (:id, :user_id, :name, 'balanced', 1, 0, :origin, 'draft', "
            ":template_id, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {
            "id": agent_id,
            "user_id": user_id,
            "name": name,
            "origin": origin,
            "template_id": template_id,
        },
    )


class TestUpgradeAddsSchema:
    def test_template_id_column_and_index_added(
        self, db_url: str, config_at_parent_revision: Config
    ) -> None:
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)

        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("agents")}
        assert "template_id" in columns

        index_names = {idx["name"] for idx in inspector.get_indexes("agents")}
        assert "ix_agents_template_id" in index_names
        assert "uq_agents_user_id_name_active" in index_names
        engine.dispose()

    def test_unique_index_rejects_same_user_duplicate_after_migration(
        self, db_url: str, config_at_parent_revision: Config
    ) -> None:
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)
            _insert_agent(conn, agent_id=1, user_id=1, name="Unique Agent")

        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                _insert_agent(conn, agent_id=2, user_id=1, name="Unique Agent")
        engine.dispose()

    def test_unique_index_allows_workforce_manager_duplicates_after_migration(
        self, db_url: str, config_at_parent_revision: Config
    ) -> None:
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)
            _insert_agent(
                conn,
                agent_id=1,
                user_id=1,
                name="Manager",
                origin="workforce_generated_manager",
            )

        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        with engine.begin() as conn:
            _insert_agent(
                conn,
                agent_id=2,
                user_id=1,
                name="Manager",
                origin="workforce_generated_manager",
            )
        with engine.begin() as conn:
            count = conn.execute(
                sa.text("SELECT COUNT(*) FROM agents WHERE name = 'Manager'")
            ).scalar()
        assert count == 2
        engine.dispose()


class TestTemplateQuickAccessUniqueIndex:
    """Covers AGENT_TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX, the partial unique
    index on (user_id, template_id) scoped to origin='template_quick_access'
    that backs the /task template quick-access resolve flow's atomicity
    (PR review findings B1/B2/D2/D3)."""

    def test_index_added(self, db_url: str, config_at_parent_revision: Config) -> None:
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)

        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        inspector = inspect(engine)
        index_names = {idx["name"] for idx in inspector.get_indexes("agents")}
        assert "uq_agents_user_id_template_id_quick_access" in index_names
        engine.dispose()

    def test_rejects_same_user_template_duplicate_within_quick_access_origin(
        self, db_url: str, config_at_parent_revision: Config
    ) -> None:
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)

        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        with engine.begin() as conn:
            _insert_agent_with_template(
                conn,
                agent_id=1,
                user_id=1,
                name="Quick Access Agent",
                template_id="tmpl-a",
            )
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                _insert_agent_with_template(
                    conn,
                    agent_id=2,
                    user_id=1,
                    name="Quick Access Agent 2",
                    template_id="tmpl-a",
                )
        engine.dispose()

    def test_allows_a_non_quick_access_duplicate_for_the_same_template(
        self, db_url: str, config_at_parent_revision: Config
    ) -> None:
        """A workforce-builder agent (origin='user') built from the same
        template as an existing quick-access agent must not collide - the
        index is scoped to the quick-access origin only (B4)."""
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)

        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        with engine.begin() as conn:
            _insert_agent_with_template(
                conn,
                agent_id=1,
                user_id=1,
                name="Quick Access Agent",
                template_id="tmpl-a",
                origin="template_quick_access",
            )
        with engine.begin() as conn:
            _insert_agent_with_template(
                conn,
                agent_id=2,
                user_id=1,
                name="Workforce Worker",
                template_id="tmpl-a",
                origin="user",
            )
        with engine.begin() as conn:
            count = conn.execute(
                sa.text("SELECT COUNT(*) FROM agents WHERE template_id = 'tmpl-a'")
            ).scalar()
        assert count == 2
        engine.dispose()


class TestDedupeRenamesLosingDuplicates:
    def test_lowest_id_keeps_name_later_rows_get_suffixed(
        self, db_url: str, config_at_parent_revision: Config
    ) -> None:
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)
            _insert_agent(conn, agent_id=10, user_id=1, name="Dup Agent")
            _insert_agent(conn, agent_id=11, user_id=1, name="Dup Agent")
            _insert_agent(conn, agent_id=12, user_id=1, name="Dup Agent")

        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        names = _agent_names_by_id(engine)
        assert names[10] == "Dup Agent"
        assert names[11] == "Dup Agent (11)"
        assert names[12] == "Dup Agent (12)"
        engine.dispose()

    def test_workforce_manager_rows_are_not_deduped(
        self, db_url: str, config_at_parent_revision: Config
    ) -> None:
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)
            _insert_agent(
                conn,
                agent_id=20,
                user_id=1,
                name="Manager Dup",
                origin="workforce_generated_manager",
            )
            _insert_agent(
                conn,
                agent_id=21,
                user_id=1,
                name="Manager Dup",
                origin="workforce_generated_manager",
            )

        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        names = _agent_names_by_id(engine)
        assert names[20] == "Manager Dup"
        assert names[21] == "Manager Dup"
        engine.dispose()

    def test_rename_target_collision_falls_back_to_a_free_name(
        self, db_url: str, config_at_parent_revision: Config
    ) -> None:
        """Regression test for F3: the naive rename target
        "{name} ({agent_id})" can itself already be taken by an unrelated
        row for the same user. The dedupe pass must keep searching for a
        genuinely free name instead of creating a *new* collision that would
        then make the unique-index build itself fail.
        """
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)
            _insert_agent(conn, agent_id=30, user_id=1, name="Foo")
            _insert_agent(conn, agent_id=31, user_id=1, name="Foo")
            # Pre-existing row that happens to already occupy the naive
            # rename target the dedupe pass would otherwise pick for id 31.
            _insert_agent(conn, agent_id=32, user_id=1, name="Foo (31)")

        # Must not raise - the migration should resolve the collision itself.
        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        names = _agent_names_by_id(engine)
        assert names[30] == "Foo"
        assert names[32] == "Foo (31)"
        assert names[31] not in ("Foo", "Foo (31)")
        assert names[31].startswith("Foo (31-")
        assert len(set(names.values())) == 3
        engine.dispose()

    def test_rename_candidate_is_truncated_to_fit_the_name_column(
        self, db_url: str, config_at_parent_revision: Config
    ) -> None:
        """Regression test for F5: a pre-existing name at or near
        Agent.name's String(200) limit must not overflow once the rename
        suffix is appended - that would raise StringDataRightTruncation on
        backends (e.g. Postgres) that enforce column length, aborting the
        exact migration written to survive messy data.
        """
        long_name = "x" * 200
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)
            _insert_agent(conn, agent_id=40, user_id=1, name=long_name)
            _insert_agent(conn, agent_id=41, user_id=1, name=long_name)

        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        names = _agent_names_by_id(engine)
        assert names[40] == long_name
        assert len(names[41]) <= 200
        assert names[41] != long_name
        assert names[41].endswith("(41)")
        engine.dispose()


class TestDowngrade:
    def test_downgrade_removes_column_and_indexes_but_keeps_renamed_names(
        self, db_url: str, config_at_parent_revision: Config
    ) -> None:
        with create_engine(db_url).begin() as conn:
            _insert_user(conn, 1)
            _insert_agent(conn, agent_id=40, user_id=1, name="Renamed Group")
            _insert_agent(conn, agent_id=41, user_id=1, name="Renamed Group")

        command.upgrade(config_at_parent_revision, TARGET_REVISION)

        engine = create_engine(db_url)
        names_after_upgrade = _agent_names_by_id(engine)
        assert names_after_upgrade[41] == "Renamed Group (41)"
        engine.dispose()

        command.downgrade(config_at_parent_revision, PARENT_REVISION)

        engine = create_engine(db_url)
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("agents")}
        assert "template_id" not in columns

        index_names = {idx["name"] for idx in inspector.get_indexes("agents")}
        assert "ix_agents_template_id" not in index_names
        assert "uq_agents_user_id_name_active" not in index_names

        # Renaming is a one-way disambiguation - downgrade does not attempt
        # to restore the pre-migration colliding name (see the migration's
        # module docstring).
        names_after_downgrade = _agent_names_by_id(engine)
        assert names_after_downgrade[41] == "Renamed Group (41)"
        engine.dispose()
