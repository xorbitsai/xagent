"""The connector callback URI snapshot columns migrate both ways."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION = "xagent.migrations.versions.20261126_connector_oauth_redirect_uri"


def test_flow_and_grant_redirect_uri_columns_migrate_both_ways(tmp_path):
    migration = importlib.import_module(MIGRATION)
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    metadata = sa.MetaData()
    sa.Table(
        "mcp_oauth_flow_states", metadata, sa.Column("id", sa.Integer, primary_key=True)
    )
    sa.Table("user_oauth", metadata, sa.Column("id", sa.Integer, primary_key=True))
    metadata.create_all(engine)

    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.upgrade()
        migration.upgrade()
        inspector = sa.inspect(connection)
        assert "redirect_uri" in {
            column["name"] for column in inspector.get_columns("mcp_oauth_flow_states")
        }
        assert "redirect_uri" in {
            column["name"] for column in inspector.get_columns("user_oauth")
        }
        migration.downgrade()
        inspector = sa.inspect(connection)
        assert "redirect_uri" not in {
            column["name"] for column in inspector.get_columns("mcp_oauth_flow_states")
        }
        assert "redirect_uri" not in {
            column["name"] for column in inspector.get_columns("user_oauth")
        }
    engine.dispose()
