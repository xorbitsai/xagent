"""Track interactive browser logins for the idle-expiry policy.

``browser_sessions`` holds one row per login (its ``sid``, last activity and
revocation). ``users.browser_session_id`` records which login holds the user's
refresh token, so refresh tokens issued before this release can be counted.
Both start empty; the idle policy is off by default.

Revision ID: 20261113_browser_idle_session
Revises: 20261110_rename_atlassian_to_confluence
"""

import sqlalchemy as sa
from alembic import op

revision = "20261113_browser_idle_session"
down_revision = "20261110_rename_atlassian_to_confluence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("users"):
        return
    if "browser_session_id" not in {c["name"] for c in inspector.get_columns("users")}:
        op.add_column(
            "users", sa.Column("browser_session_id", sa.String(32), nullable=True)
        )
    if not inspector.has_table("browser_sessions"):
        op.create_table(
            "browser_sessions",
            sa.Column("sid", sa.String(32), primary_key=True),
            sa.Column(
                "user_id",
                sa.Integer(),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_reason", sa.String(16), nullable=True),
        )
        op.create_index("ix_browser_sessions_user_id", "browser_sessions", ["user_id"])
        op.create_index(
            "ix_browser_sessions_last_used_at", "browser_sessions", ["last_used_at"]
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("browser_sessions"):
        op.drop_table("browser_sessions")
    if inspector.has_table("users") and "browser_session_id" in {
        c["name"] for c in inspector.get_columns("users")
    }:
        with op.batch_alter_table("users") as batch:
            batch.drop_column("browser_session_id")
