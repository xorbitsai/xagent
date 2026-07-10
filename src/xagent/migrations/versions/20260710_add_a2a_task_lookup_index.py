"""add A2A task lookup index

Revision ID: 20260710_a2a_task_index
Revises: 1c2ae61b5a6d
Create Date: 2026-07-10

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect

revision: str = "20260710_a2a_task_index"
down_revision: Union[str, None] = "1c2ae61b5a6d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "ix_tasks_agent_id_source"
INDEX_COLUMNS = {"agent_id", "source"}


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if "tasks" not in inspector.get_table_names():
        return
    existing_columns = {item["name"] for item in inspector.get_columns("tasks")}
    if not INDEX_COLUMNS.issubset(existing_columns):
        return
    existing_indexes = {item["name"] for item in inspector.get_indexes("tasks")}
    if INDEX_NAME not in existing_indexes:
        op.create_index(INDEX_NAME, "tasks", ["agent_id", "source"])


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if "tasks" not in inspector.get_table_names():
        return
    existing_indexes = {item["name"] for item in inspector.get_indexes("tasks")}
    if INDEX_NAME in existing_indexes:
        op.drop_index(INDEX_NAME, table_name="tasks")
