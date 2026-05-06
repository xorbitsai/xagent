"""add body field to custom_api

Revision ID: b2c517a02b3b
Revises: dd20b4878cf1
Create Date: 2026-05-06 15:36:01.510171

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c517a02b3b"
down_revision: Union[str, None] = "dd20b4878cf1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context
    from sqlalchemy.engine.reflection import Inspector

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if column already exists before adding
    columns = [col["name"] for col in inspector.get_columns("custom_apis")]
    if "body" not in columns:
        op.add_column("custom_apis", sa.Column("body", sa.Text(), nullable=True))


def downgrade() -> None:
    from alembic import context
    from sqlalchemy.engine.reflection import Inspector

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    columns = [col["name"] for col in inspector.get_columns("custom_apis")]
    if "body" in columns:
        op.drop_column("custom_apis", "body")
