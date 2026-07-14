"""Add team_id scope column to agents

Revision ID: 20260713_add_team_id_to_agents
Revises: 20260713_mcp_oauth_dcr
Create Date: 2026-07-13
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision = "20260713_add_team_id_to_agents"
down_revision = "20260713_mcp_oauth_dcr"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    return {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    if "agents" not in inspector.get_table_names():
        return
    if "team_id" not in _column_names("agents"):
        op.add_column("agents", sa.Column("team_id", sa.Integer(), nullable=True))
        op.create_index("ix_agents_team_id", "agents", ["team_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    if "agents" not in inspector.get_table_names():
        return
    if "team_id" in _column_names("agents"):
        op.drop_index("ix_agents_team_id", table_name="agents")
        op.drop_column("agents", "team_id")
