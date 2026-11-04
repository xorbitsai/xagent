"""drop workforces.manager_instructions column

Workforce-level manager instructions were removed (#800): the manager's
behaviour is fully defined by the Manager Agent's own instructions. This is
the contract step of the expand-contract sequence; the read/write paths were
removed first, so stored values are already ignored and can be discarded.

Revision ID: 20260721_drop_workforce_manager_instr
Revises: 20260715_add_public_mcp_app_audits
Create Date: 2026-07-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260721_drop_workforce_manager_instr"
down_revision: Union[str, None] = "20260715_add_public_mcp_app_audits"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _workforces_columns() -> set[str] | None:
    """Column names of the workforces table, or None if the table is absent.

    On fresh installations core tables are created by Base.metadata.create_all()
    after Alembic runs, so the table may legitimately not exist yet.
    """
    inspector = sa.inspect(op.get_bind())
    if "workforces" not in inspector.get_table_names():
        return None
    return {col["name"] for col in inspector.get_columns("workforces")}


def upgrade() -> None:
    if not op.get_context().as_sql:
        columns = _workforces_columns()
        if columns is None or "manager_instructions" not in columns:
            return
    # batch mode recreates the table on SQLite; on PostgreSQL it is a plain
    # ALTER TABLE ... DROP COLUMN.
    with op.batch_alter_table("workforces") as batch_op:
        batch_op.drop_column("manager_instructions")


def downgrade() -> None:
    if not op.get_context().as_sql:
        columns = _workforces_columns()
        if columns is None or "manager_instructions" in columns:
            return
    # Structure only: the dropped values are not recoverable.
    with op.batch_alter_table("workforces") as batch_op:
        batch_op.add_column(sa.Column("manager_instructions", sa.Text(), nullable=True))
