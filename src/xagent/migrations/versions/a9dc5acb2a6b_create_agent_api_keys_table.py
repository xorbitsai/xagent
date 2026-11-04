"""create_agent_api_keys_table

Revision ID: a9dc5acb2a6b
Revises: 20260509_add_delegate_agent_ids_to_tasks
Create Date: 2026-05-12 19:02:06.787360

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "a9dc5acb2a6b"
# Re-chained on top of upstream's latest head (was
# "20260509_add_delegate_agent_ids_to_tasks") after main merged the
# delegate-agents drop + task-execution-lease branches; without this
# alembic sees two heads and ``alembic-check`` pre-commit fails.
down_revision: Union[str, None] = "20260514_drop_delegate_agent_ids_from_tasks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = inspector.get_table_names()

    if "agent_api_keys" in existing_tables:
        return

    op.create_table(
        "agent_api_keys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=False),
        sa.Column("key_prefix", sa.String(length=12), nullable=False),
        sa.Column("key_hash", sa.String(length=128), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_agent_api_keys_id"), "agent_api_keys", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_agent_api_keys_agent_id"),
        "agent_api_keys",
        ["agent_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_api_keys_key_prefix"),
        "agent_api_keys",
        ["key_prefix"],
        unique=True,
    )

    # Partial unique index: at most one active (non-revoked) key per agent.
    # Both PostgreSQL and SQLite support partial indexes via the WHERE
    # clause; emitting raw DDL here makes the constraint identical across
    # both engines (so the SQLite-backed test suite exercises the same
    # rotation semantics as production). Using op.create_index with
    # ``postgresql_where`` would silently degrade SQLite to a plain
    # unique index, which would (wrongly) reject every key rotation.
    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX uq_agent_api_keys_agent_active "
            "ON agent_api_keys (agent_id) WHERE revoked_at IS NULL"
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = inspector.get_table_names()

    if "agent_api_keys" not in existing_tables:
        return

    op.drop_index("uq_agent_api_keys_agent_active", table_name="agent_api_keys")
    op.drop_index(op.f("ix_agent_api_keys_key_prefix"), table_name="agent_api_keys")
    op.drop_index(op.f("ix_agent_api_keys_agent_id"), table_name="agent_api_keys")
    op.drop_index(op.f("ix_agent_api_keys_id"), table_name="agent_api_keys")
    op.drop_table("agent_api_keys")
