"""add Amazon Bedrock model settings

Revision ID: 20261010_bedrock_settings
Revises: 20261009_model_api_key_text
Create Date: 2026-10-10 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261010_bedrock_settings"
down_revision: Union[str, None] = "20261009_model_api_key_text"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "models"
REGION_COLUMN = "bedrock_region"
AUTH_MODE_COLUMN = "bedrock_auth_mode"


def _column_names(bind: sa.engine.Connection) -> set[str]:
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(TABLE)}


def upgrade() -> None:
    bind = op.get_bind()
    columns = _column_names(bind)
    if not columns:
        return
    with op.batch_alter_table(TABLE) as batch_op:
        if REGION_COLUMN not in columns:
            batch_op.add_column(sa.Column(REGION_COLUMN, sa.String(64), nullable=True))
        if AUTH_MODE_COLUMN not in columns:
            batch_op.add_column(
                sa.Column(AUTH_MODE_COLUMN, sa.String(32), nullable=True)
            )


def downgrade() -> None:
    bind = op.get_bind()
    columns = _column_names(bind)
    if not columns:
        return
    with op.batch_alter_table(TABLE) as batch_op:
        if AUTH_MODE_COLUMN in columns:
            batch_op.drop_column(AUTH_MODE_COLUMN)
        if REGION_COLUMN in columns:
            batch_op.drop_column(REGION_COLUMN)
