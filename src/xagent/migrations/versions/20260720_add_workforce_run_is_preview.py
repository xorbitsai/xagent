"""add is_preview flag to workforce_runs

Revision ID: 20260720_add_workforce_run_is_preview
Revises: 20260715_add_public_mcp_app_audits
Create Date: 2026-07-20

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260720_add_workforce_run_is_preview"
down_revision: Union[str, None] = "20260715_add_public_mcp_app_audits"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "workforce_runs"
COLUMN = "is_preview"


def _existing_columns(inspector: Inspector) -> list[str]:
    return [col["name"] for col in inspector.get_columns(TABLE)]


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    if TABLE not in inspector.get_table_names():
        return

    if COLUMN not in _existing_columns(inspector):
        op.add_column(
            TABLE,
            sa.Column(
                COLUMN,
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    if TABLE not in inspector.get_table_names():
        return

    if COLUMN in _existing_columns(inspector):
        op.drop_column(TABLE, COLUMN)
