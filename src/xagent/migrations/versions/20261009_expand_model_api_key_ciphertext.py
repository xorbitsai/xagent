"""store encrypted model credentials as text

Revision ID: 20261009_model_api_key_text
Revises: 20261008_microsoft_offline_access
Create Date: 2026-10-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261009_model_api_key_text"
down_revision: Union[str, None] = "20261008_microsoft_offline_access"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "models"
COLUMN = "_api_key_encrypted"
PREVIOUS_LENGTH = 500


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
            type_=sa.Text(),
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
            "Cannot downgrade models._api_key_encrypted to VARCHAR(500): "
            "stored encrypted credentials exceed 500 characters"
        )

    with op.batch_alter_table(TABLE) as batch_op:
        batch_op.alter_column(
            COLUMN,
            existing_type=sa.Text(),
            type_=sa.String(PREVIOUS_LENGTH),
            existing_nullable=False,
        )
