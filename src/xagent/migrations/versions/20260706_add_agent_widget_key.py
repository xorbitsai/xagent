"""add agent widget key

Revision ID: 20260706_add_agent_widget_key
Revises: 20260705_add_user_mcpserver_env_source
Create Date: 2026-07-06 00:00:00.000000

"""

import secrets
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260706_add_agent_widget_key"
down_revision: Union[str, tuple[str, str], None] = (
    "20260705_add_user_mcpserver_env_source"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "agents" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("agents")}
    if "widget_key" not in existing_columns:
        op.add_column(
            "agents",
            sa.Column("widget_key", sa.String(length=255), nullable=True),
        )

    # Backfill so enforcement can turn on without manual operator steps:
    # every widget-enabled agent gets an unguessable key.
    agents = sa.table(
        "agents",
        sa.column("id", sa.Integer),
        sa.column("widget_enabled", sa.Boolean),
        sa.column("widget_key", sa.String),
    )
    rows = bind.execute(
        sa.select(agents.c.id).where(
            agents.c.widget_enabled.is_(True), agents.c.widget_key.is_(None)
        )
    ).fetchall()
    for (agent_id,) in rows:
        bind.execute(
            agents.update()
            .where(agents.c.id == agent_id)
            .values(widget_key=secrets.token_urlsafe(32))
        )

    existing_indexes = _index_names("agents")
    if "ix_agents_widget_key" not in existing_indexes:
        op.create_index("ix_agents_widget_key", "agents", ["widget_key"], unique=True)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "agents" not in inspector.get_table_names():
        return

    existing_indexes = _index_names("agents")
    if "ix_agents_widget_key" in existing_indexes:
        op.drop_index("ix_agents_widget_key", table_name="agents")

    existing_columns = {col["name"] for col in inspector.get_columns("agents")}
    if "widget_key" in existing_columns:
        op.drop_column("agents", "widget_key")
