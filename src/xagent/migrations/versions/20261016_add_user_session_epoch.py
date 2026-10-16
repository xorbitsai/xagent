"""Add durable revocation generations for user sessions."""

import sqlalchemy as sa
from alembic import op

revision = "20261016_user_session_epoch"
down_revision = "20261015_task_cmd_claimable_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("users")}
    if "session_epoch" not in columns:
        op.add_column("users", sa.Column("session_epoch", sa.String(32), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_column("session_epoch")
