"""Add instance_url to user_oauth table

Salesforce (and no other provider) returns a per-org API host in the token
response instead of using a fixed API domain; every subsequent API call
must go through this URL, not a hardcoded one, so it needs its own
persisted column on the legacy UserOAuth grant.

Revision ID: 20260818_add_instance_url_to_user_oauth
Revises: 20260813_trace_json_columns_to_jsonb
Create Date: 2026-08-18 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20260818_add_instance_url_to_user_oauth"
down_revision: Union[str, None] = "20260813_trace_json_columns_to_jsonb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "user_oauth"
COLUMN_NAME = "instance_url"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)

    if TABLE_NAME not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns(TABLE_NAME)}
    if COLUMN_NAME not in existing_columns:
        op.add_column(TABLE_NAME, sa.Column(COLUMN_NAME, sa.String(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)

    if TABLE_NAME not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns(TABLE_NAME)}
    if COLUMN_NAME in existing_columns:
        op.drop_column(TABLE_NAME, COLUMN_NAME)
