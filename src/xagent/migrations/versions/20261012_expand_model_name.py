"""expand persisted provider model names

Revision ID: 20261012_model_name_2048
Revises: 20261010_bedrock_settings
Create Date: 2026-10-12 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261012_model_name_2048"
down_revision: Union[str, None] = "20261010_bedrock_settings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "models"
COLUMN = "model_name"
PREVIOUS_LENGTH = 100
NEW_LENGTH = 2048


def _column_exists(bind: sa.engine.Connection) -> bool:
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return False
    return COLUMN in {column["name"] for column in inspector.get_columns(TABLE)}


def upgrade() -> None:
    bind = op.get_bind()
    if not _column_exists(bind):
        return
    with op.batch_alter_table(TABLE) as batch_op:
        batch_op.alter_column(
            COLUMN,
            existing_type=sa.String(PREVIOUS_LENGTH),
            type_=sa.String(NEW_LENGTH),
            existing_nullable=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    if not _column_exists(bind):
        return

    longest = bind.execute(
        sa.text(f'SELECT MAX(LENGTH("{COLUMN}")) FROM "{TABLE}"')
    ).scalar()
    if longest is not None and int(longest) > PREVIOUS_LENGTH:
        raise RuntimeError(
            "Cannot downgrade models.model_name to VARCHAR(100): "
            "stored model names exceed 100 characters"
        )

    with op.batch_alter_table(TABLE) as batch_op:
        batch_op.alter_column(
            COLUMN,
            existing_type=sa.String(NEW_LENGTH),
            type_=sa.String(PREVIOUS_LENGTH),
            existing_nullable=False,
        )
