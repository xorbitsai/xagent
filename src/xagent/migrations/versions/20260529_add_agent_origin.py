"""add agent origin

Revision ID: 20260529_add_agent_origin
Revises: 20260528_add_kb_ingest_targets
Create Date: 2026-05-29 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260529_add_agent_origin"
down_revision: Union[str, None] = "20260528_add_kb_ingest_targets"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "agents" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("agents")}
    if "origin" not in existing_columns:
        op.add_column(
            "agents",
            sa.Column(
                "origin",
                sa.String(length=50),
                server_default="user",
                nullable=False,
            ),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("agents")}
    if "ix_agents_origin" not in existing_indexes:
        op.create_index("ix_agents_origin", "agents", ["origin"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "agents" not in inspector.get_table_names():
        return

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("agents")}
    if "ix_agents_origin" in existing_indexes:
        op.drop_index("ix_agents_origin", table_name="agents")

    existing_columns = {col["name"] for col in inspector.get_columns("agents")}
    if "origin" in existing_columns:
        op.drop_column("agents", "origin")
