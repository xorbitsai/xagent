"""Add visibility column to agents

Revision ID: 20260713_add_agent_visibility
Revises: 20260713_add_team_id_to_agents
Create Date: 2026-07-13
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision = "20260713_add_agent_visibility"
down_revision = "20260713_add_team_id_to_agents"
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
    if "visibility" not in _column_names("agents"):
        op.add_column(
            "agents",
            sa.Column(
                "visibility",
                sa.String(length=20),
                nullable=False,
                server_default="team",
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    if "agents" not in inspector.get_table_names():
        return
    if "visibility" in _column_names("agents"):
        op.drop_column("agents", "visibility")
