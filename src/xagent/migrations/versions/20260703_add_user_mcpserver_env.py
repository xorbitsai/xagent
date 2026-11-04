"""add per-user env overrides to user_mcpservers

Revision ID: 20260703_add_user_mcpserver_env
Revises: 20260625_add_user_skills_tables
Create Date: 2026-07-03 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260703_add_user_mcpserver_env"
down_revision: Union[str, None] = "20260625_add_user_skills_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    if "user_mcpservers" not in sa.inspect(op.get_bind()).get_table_names():
        return

    columns = _columns("user_mcpservers")
    if "env" not in columns:
        op.add_column("user_mcpservers", sa.Column("env", sa.JSON(), nullable=True))

    # Backfill ownership. Historically the create endpoint left is_owner unset
    # (False) even though the creator is the owner; no code path ever creates an
    # intentional non-owner row, so every existing non-owner row is a regressed
    # owner. Without this, this deploy's can_edit_global gate would lock owners
    # out of editing their own servers' global config.
    if {"is_owner", "can_edit", "can_delete"} <= columns:
        ums = sa.table(
            "user_mcpservers",
            sa.column("is_owner", sa.Boolean),
            sa.column("can_edit", sa.Boolean),
            sa.column("can_delete", sa.Boolean),
        )
        op.execute(
            ums.update()
            .where(ums.c.is_owner == sa.false())
            .values(is_owner=True, can_edit=True, can_delete=True)
        )


def downgrade() -> None:
    if "env" in _columns("user_mcpservers"):
        op.drop_column("user_mcpservers", "env")
