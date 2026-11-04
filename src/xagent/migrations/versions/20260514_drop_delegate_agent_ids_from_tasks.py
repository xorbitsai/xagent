"""drop delegate_agent_ids from tasks

Revision ID: 20260514_drop_delegate_agent_ids_from_tasks
Revises: 20260514_add_user_template_relations
Create Date: 2026-05-14 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20260514_drop_delegate_agent_ids_from_tasks"
down_revision: Union[str, None] = "20260514_add_user_template_relations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
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


def downgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)

    tables = inspector.get_table_names()
    if "tasks" not in tables:
        return

    existing_columns = [col["name"] for col in inspector.get_columns("tasks")]
    if "delegate_agent_ids" not in existing_columns:
        op.add_column(
            "tasks", sa.Column("delegate_agent_ids", sa.JSON(), nullable=True)
        )
