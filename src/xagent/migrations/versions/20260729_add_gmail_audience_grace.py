"""persist the previous Gmail callback audience during endpoint transitions

Revision ID: 20260729_add_gmail_audience_grace
Revises: 20260726_add_task_telegram_user_id
Create Date: 2026-07-29 00:00:00.000000

Gmail Pub/Sub callbacks already dispatched with the prior OIDC audience may
arrive after an endpoint reconciliation commits the new audience. Persist the
previous value and its expiry so callback verification can honor that bounded
delivery grace period across processes and restarts.
"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "20260729_add_gmail_audience_grace"
down_revision: Union[str, None] = "20260726_add_task_telegram_user_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "gmail_watch_states"
COLUMNS = (
    sa.Column("previous_push_audience", sa.Text(), nullable=True),
    sa.Column(
        "previous_push_audience_expires_at",
        sa.DateTime(timezone=True),
        nullable=True,
    ),
)


def _column_names() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {str(column["name"]) for column in inspector.get_columns(TABLE)}


def upgrade() -> None:
    """Add nullable grace fields without rewriting existing watch rows."""
    context = op.get_context()
    if context.as_sql:
        for column in COLUMNS:
            op.add_column(TABLE, column)
        return

    existing_columns = _column_names()
    if not existing_columns:
        # XAgent supports upgrading deliberately partial databases used by
        # migration recovery and compatibility tooling. Keep that path
        # non-fatal, but make the skipped schema change observable instead of
        # silently stamping the revision.
        logger.warning(
            "Required table %r is missing; skipping migration %r because "
            "this database has a partial schema",
            TABLE,
            revision,
        )
        return
    for column in COLUMNS:
        if column.name not in existing_columns:
            op.add_column(TABLE, column)


def downgrade() -> None:
    """Remove only grace fields present in the live schema."""
    context = op.get_context()
    if context.as_sql:
        for column in reversed(COLUMNS):
            op.drop_column(TABLE, column.name)
        return

    existing_columns = _column_names()
    for column in reversed(COLUMNS):
        if column.name in existing_columns:
            op.drop_column(TABLE, column.name)
