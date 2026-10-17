"""Stage account-recovery email changes until the new address is confirmed."""

import sqlalchemy as sa
from alembic import op

revision = "20261016_email_confirmation"
down_revision = "20261016_user_session_epoch"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("users"):
        return
    columns = {c["name"] for c in inspector.get_columns("users")}
    for column in (
        sa.Column("pending_email", sa.String(255), nullable=True),
        sa.Column("email_confirmation_hash", sa.String(64), nullable=True),
        sa.Column(
            "email_confirmation_expires_at", sa.DateTime(timezone=True), nullable=True
        ),
    ):
        if column.name not in columns:
            op.add_column("users", column)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("users"):
        return
    columns = {c["name"] for c in inspector.get_columns("users")}
    with op.batch_alter_table("users") as batch:
        for column in (
            "email_confirmation_expires_at",
            "email_confirmation_hash",
            "pending_email",
        ):
            if column in columns:
                batch.drop_column(column)
