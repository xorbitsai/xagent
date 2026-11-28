"""add agent_versions for recorded states of an agent's versioned fields

Existing agents get no rows: having none is a normal state, and the first
change made through ``apply_agent_change`` records the state before it.

Revision ID: 20261119_agent_versions
Revises: 20261127_trigger_host_executor
Create Date: 2026-11-19

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261119_agent_versions"
down_revision: Union[str, None] = "20261127_trigger_host_executor"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "agent_versions"
UNIQUE = "uq_agent_versions_agent_id_version_no"
ACTOR_USER_INDEX = "ix_agent_versions_actor_user_id"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    # ``agents`` and ``users`` are metadata-owned in Alembic-only
    # installations; this table is then created from the models with them.
    if (
        not inspector.has_table("agents")
        or not inspector.has_table("users")
        or inspector.has_table(TABLE)
    ):
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "agent_id",
            sa.Integer(),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_no", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("actor_kind", sa.String(16), nullable=False),
        sa.Column(
            "actor_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("actor_key_prefix", sa.String(32), nullable=True),
        sa.Column("actor_source", sa.String(64), nullable=True),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("agent_id", "version_no", name=UNIQUE),
        sqlite_autoincrement=True,
    )
    op.create_index(ACTOR_USER_INDEX, TABLE, ["actor_user_id"])


def downgrade() -> None:
    if not sa.inspect(op.get_bind()).has_table(TABLE):
        return
    op.drop_index(ACTOR_USER_INDEX, table_name=TABLE)
    op.drop_table(TABLE)
