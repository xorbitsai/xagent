"""add connector visibility flag to public mcp apps

Revision ID: 20260519_add_public_mcp_visibility
Revises: fab71cf4b1ad
Create Date: 2026-05-19 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260519_add_public_mcp_visibility"
down_revision: Union[str, None] = "fab71cf4b1ad"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)

    existing_tables = inspector.get_table_names()
    if "public_mcp_apps" not in existing_tables:
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("public_mcp_apps")
    }
    if "is_visible_in_connector" in existing_columns:
        return

    bool_true = sa.text("true") if bind.dialect.name == "postgresql" else sa.text("1")

    op.add_column(
        "public_mcp_apps",
        sa.Column(
            "is_visible_in_connector",
            sa.Boolean(),
            nullable=False,
            server_default=bool_true,
        ),
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)

    existing_tables = inspector.get_table_names()
    if "public_mcp_apps" not in existing_tables:
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("public_mcp_apps")
    }
    if "is_visible_in_connector" in existing_columns:
        op.drop_column("public_mcp_apps", "is_visible_in_connector")
