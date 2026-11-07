"""add api_idempotency_keys for the v1 Idempotency-Key header

Revision ID: 20261106_api_idempotency_keys
Revises: 20261105_update_atlassian_name_and_description
Create Date: 2026-11-06

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261106_api_idempotency_keys"
down_revision: Union[str, None] = "20261105_update_atlassian_name_and_description"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "api_idempotency_keys"
UNIQUE = "uq_api_idempotency_scope_operation_key"
EXPIRES_INDEX = "ix_api_idempotency_keys_expires_at"
TASK_INDEX = "ix_api_idempotency_keys_task_id"


def _has_table() -> bool:
    return TABLE in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    # ``tasks`` is metadata-owned in Alembic-only installations; this table is
    # then created from the models together with it.
    if not inspector.has_table("tasks") or inspector.has_table(TABLE):
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("scope_kind", sa.String(16), nullable=False),
        sa.Column("scope_id", sa.Integer(), nullable=False),
        sa.Column("operation", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("claim_token", sa.String(32), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "task_id",
            sa.Integer(),
            sa.ForeignKey("tasks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("response_body", sa.JSON(), nullable=True),
        sa.Column("response_status", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "scope_kind", "scope_id", "operation", "idempotency_key", name=UNIQUE
        ),
    )
    op.create_index(EXPIRES_INDEX, TABLE, ["expires_at"])
    op.create_index(TASK_INDEX, TABLE, ["task_id"])


def downgrade() -> None:
    if not _has_table():
        return
    op.drop_index(TASK_INDEX, table_name=TABLE)
    op.drop_index(EXPIRES_INDEX, table_name=TABLE)
    op.drop_table(TABLE)
