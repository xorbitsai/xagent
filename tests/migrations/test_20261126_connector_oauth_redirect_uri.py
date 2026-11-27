"""The connector callback URI snapshot columns migrate both ways."""

import importlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION = "xagent.migrations.versions.20261126_connector_oauth_redirect_uri"


def _columns(inspector: sa.Inspector, table: str) -> set[str]:
    if table not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table)}


@pytest.mark.parametrize(
    "tables",
    [
        pytest.param(("mcp_oauth_flow_states", "user_oauth"), id="both-relations"),
        pytest.param(("mcp_oauth_flow_states",), id="missing-user-oauth"),
        pytest.param(("user_oauth",), id="missing-flow-states"),
        pytest.param((), id="missing-both-relations"),
    ],
)
def test_flow_and_grant_redirect_uri_columns_migrate_both_ways(tmp_path, tables):
    migration = importlib.import_module(MIGRATION)
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    metadata = sa.MetaData()
    sa.Table(
        "mcp_oauth_flow_states", metadata, sa.Column("id", sa.Integer, primary_key=True)
    )
    sa.Table("user_oauth", metadata, sa.Column("id", sa.Integer, primary_key=True))
    if tables:
        metadata.create_all(engine, tables=[metadata.tables[name] for name in tables])

    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.upgrade()
        migration.upgrade()
        inspector = sa.inspect(connection)
        for table in tables:
            assert "redirect_uri" in _columns(inspector, table)
        migration.downgrade()
        inspector = sa.inspect(connection)
        assert set(inspector.get_table_names()) == set(tables)
        for table in tables:
            assert "redirect_uri" not in _columns(inspector, table)
    engine.dispose()
