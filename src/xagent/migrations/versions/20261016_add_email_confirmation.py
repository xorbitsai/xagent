"""Stage account-recovery email changes until the new address is confirmed."""

import sqlalchemy as sa
from alembic import op

revision = "20261016_email_confirmation"
down_revision = "20261016_user_session_epoch"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("users")}
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
    with op.batch_alter_table("users") as batch:
        batch.drop_column("email_confirmation_expires_at")
        batch.drop_column("email_confirmation_hash")
        batch.drop_column("pending_email")
