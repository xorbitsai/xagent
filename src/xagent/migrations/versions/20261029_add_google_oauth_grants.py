"""Add shared Google OAuth grant metadata and app associations.

Revision ID: 20261029_google_oauth_grants
Revises: 20261027_wrap_legacy_password_hashes
Create Date: 2026-10-29
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261029_google_oauth_grants"
down_revision: Union[str, None] = "20261027_wrap_legacy_password_hashes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

GRANTS_TABLE = "google_oauth_grants"
ASSOCIATIONS_TABLE = "google_oauth_app_associations"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    # Metadata-owned test databases may create these tables before Alembic is
    # invoked. Keep the migration safe for that supported bootstrap shape.
    if not inspector.has_table("user_oauth") or not inspector.has_table("users"):
        return
    if not inspector.has_table(GRANTS_TABLE):
        op.create_table(
            GRANTS_TABLE,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "user_id",
                sa.Integer,
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("resource_owner_key", sa.String(512), nullable=True),
            sa.Column("google_subject", sa.String(255), nullable=False),
            sa.Column("oauth_client_fingerprint", sa.String(64), nullable=False),
            sa.Column(
                "oauth_account_id",
                sa.Integer,
                sa.ForeignKey("user_oauth.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.UniqueConstraint(
                "oauth_account_id", name="uq_google_oauth_grant_account"
            ),
        )
        op.create_index(
            "uq_google_oauth_grant_ordinary_identity",
            GRANTS_TABLE,
            ["user_id", "google_subject", "oauth_client_fingerprint"],
            unique=True,
            sqlite_where=sa.text("resource_owner_key IS NULL"),
            postgresql_where=sa.text("resource_owner_key IS NULL"),
        )
        op.create_index(
            "uq_google_oauth_grant_actor_identity",
            GRANTS_TABLE,
            [
                "user_id",
                "resource_owner_key",
                "google_subject",
                "oauth_client_fingerprint",
            ],
            unique=True,
            sqlite_where=sa.text("resource_owner_key IS NOT NULL"),
            postgresql_where=sa.text("resource_owner_key IS NOT NULL"),
        )

    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(ASSOCIATIONS_TABLE):
        op.create_table(
            ASSOCIATIONS_TABLE,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "grant_id",
                sa.Integer,
                sa.ForeignKey(f"{GRANTS_TABLE}.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("app_id", sa.String(255), nullable=False),
            sa.Column(
                "legacy_oauth_account_id",
                sa.Integer,
                sa.ForeignKey("user_oauth.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.UniqueConstraint(
                "grant_id", "app_id", name="uq_google_oauth_app_association"
            ),
            sa.UniqueConstraint(
                "legacy_oauth_account_id",
                name="uq_google_oauth_app_association_legacy_account",
            ),
        )
        op.create_index(
            "ix_google_oauth_app_associations_app_id",
            ASSOCIATIONS_TABLE,
            ["app_id"],
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table(ASSOCIATIONS_TABLE):
        op.drop_table(ASSOCIATIONS_TABLE)
    if inspector.has_table(GRANTS_TABLE):
        op.drop_table(GRANTS_TABLE)
