"""add delegate_agent_ids to tasks

Revision ID: 20260509_add_delegate_agent_ids_to_tasks
Revises: b2c517a02b3b
Create Date: 2026-05-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20260509_add_delegate_agent_ids_to_tasks"
down_revision: Union[str, None] = "b2c517a02b3b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    tables = inspector.get_table_names()
    if "tasks" not in tables:
        return

    existing_columns = [col["name"] for col in inspector.get_columns("tasks")]
    if "delegate_agent_ids" not in existing_columns:
        op.add_column(
            "tasks", sa.Column("delegate_agent_ids", sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    dialect_name = bind.dialect.name

    tables = inspector.get_table_names()
    if "tasks" not in tables:
        return

    existing_columns = [col["name"] for col in inspector.get_columns("tasks")]
    if "delegate_agent_ids" in existing_columns:
        if dialect_name == "sqlite":
            with op.batch_alter_table("tasks", recreate="auto") as batch_op:
                batch_op.drop_column("delegate_agent_ids")
        else:
            op.drop_column("tasks", "delegate_agent_ids")
