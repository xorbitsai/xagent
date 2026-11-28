"""The ``agent_versions`` migration on SQLite and PostgreSQL.

The PostgreSQL half comes from ``disposable_database_factory``, which reads
``XAGENT_TEST_POSTGRES_URL`` and skips when it is unset.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.models.agent_version import AgentVersion
from xagent.web.models.database import Base

REVISION = "20261119_agent_versions"
DOWN_REVISION = "20261118_seed_cosmos_mcp_app"
TABLE = "agent_versions"
MIGRATION = (
    Path(__file__).parents[2]
    / "src/xagent/migrations/versions/20261119_add_agent_versions.py"
)


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def engine(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[sa.Engine]:
    if request.param == "sqlite":
        sqlite_engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
        try:
            yield sqlite_engine
        finally:
            sqlite_engine.dispose()
        return
    with disposable_database_factory("xagent_agent_versions_mig") as make_database:
        yield make_database("m")


def _create_schema_before_revision(connection: sa.Connection) -> None:
    tables = [t for t in Base.metadata.sorted_tables if t.name != TABLE]
    Base.metadata.create_all(connection, tables=tables)


def _assert_table_shape(connection: sa.Connection) -> None:
    inspector = sa.inspect(connection)
    columns = {c["name"]: c for c in inspector.get_columns(TABLE)}
    assert set(columns) == {c.name for c in AgentVersion.__table__.columns}
    nullable = {name for name, column in columns.items() if column["nullable"]}
    assert nullable == {"actor_user_id", "actor_key_prefix", "actor_source"}

    foreign_keys = {
        tuple(fk["constrained_columns"]): (
            fk["referred_table"],
            (fk.get("options") or {}).get("ondelete"),
        )
        for fk in inspector.get_foreign_keys(TABLE)
    }
    assert foreign_keys == {
        ("agent_id",): ("agents", "CASCADE"),
        ("actor_user_id",): ("users", "SET NULL"),
    }
    uniques = {
        tuple(u["column_names"]) for u in inspector.get_unique_constraints(TABLE)
    }
    assert ("agent_id", "version_no") in uniques
    indexes = {tuple(i["column_names"]) for i in inspector.get_indexes(TABLE)}
    assert ("actor_user_id",) in indexes
    if connection.dialect.name == "sqlite":
        ddl = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE name = 'agent_versions'"
        ).scalar_one()
        assert "AUTOINCREMENT" in ddl


def _commit(connection: sa.Connection) -> None:
    # On PostgreSQL the environment runs each revision in its own
    # transaction, so the connection must not be inside one when it starts.
    if connection.in_transaction():
        connection.commit()


def test_upgrade_to_head_and_back(engine: sa.Engine) -> None:
    config = create_alembic_config(engine)
    with engine.connect() as connection:
        _create_schema_before_revision(connection)
        _commit(connection)
        config.attributes["connection"] = connection
        command.stamp(config, DOWN_REVISION)
        _commit(connection)

        command.upgrade(config, "head")
        _commit(connection)
        _assert_table_shape(connection)
        assert (
            connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
            == REVISION
        )

        _commit(connection)
        command.downgrade(config, DOWN_REVISION)
        _commit(connection)
        assert TABLE not in sa.inspect(connection).get_table_names()
        assert "agents" in sa.inspect(connection).get_table_names()

        _commit(connection)
        command.upgrade(config, "head")
        _commit(connection)
        _assert_table_shape(connection)


def test_upgrade_skips_a_table_created_from_the_models(engine: sa.Engine) -> None:
    migration = load_migration_module(MIGRATION)
    with engine.begin() as connection:
        Base.metadata.create_all(connection)
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        _assert_table_shape(connection)
        migration.downgrade()
        migration.downgrade()
        assert TABLE not in sa.inspect(connection).get_table_names()


def test_upgrade_skips_without_agents(tmp_path: Path) -> None:
    migration = load_migration_module(MIGRATION)
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE unrelated (id INTEGER)")
            migration.op = Operations(MigrationContext.configure(connection))
            migration.upgrade()
            migration.downgrade()
            assert sa.inspect(connection).get_table_names() == ["unrelated"]
    finally:
        engine.dispose()
