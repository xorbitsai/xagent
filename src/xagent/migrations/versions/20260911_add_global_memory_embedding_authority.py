"""Add explicit global memory embedding authority.
Revision ID: 20260911_global_memory_authority
Revises: 20260911_assistant_source_event
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260911_global_memory_authority"
down_revision: Union[str, None] = "20260911_assistant_source_event"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "global_memory_embedding_authority"


def upgrade() -> None:
    bind = op.get_bind()
    if _TABLE in Inspector.from_engine(bind).get_table_names():
        return
    op.create_table(
        _TABLE,
        sa.Column("authority_key", sa.String(32), primary_key=True),
        sa.Column("model_provider", sa.String(50), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("base_url", sa.String(500), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("instruct", sa.Text(), nullable=True),
        sa.Column("max_retries", sa.Integer(), nullable=False),
        sa.Column("api_key_encrypted", sa.Text(), nullable=False),
        sa.Column("credential_digest", sa.String(64), nullable=False),
        sa.Column("configured_by_actor_subject", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "authority_key = 'global'", name="ck_global_memory_authority_key"
        ),
    )


def downgrade() -> None:
    bind = op.get_bind()
    if _TABLE in Inspector.from_engine(bind).get_table_names():
        op.drop_table(_TABLE)
