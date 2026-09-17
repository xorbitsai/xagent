"""Add the dormant durable sandbox lifecycle substrate.

Revision ID: 20260914_durable_sandbox_lifecycles
Revises: 20260914_add_hubspot_deals_write_scope

No rows are backfilled.  Existing sandboxes remain outside this protocol and
Chrome remains hidden/default-off until the separate consumer stack lands.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260914_durable_sandbox_lifecycles"
down_revision = "20260914_add_hubspot_deals_write_scope"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("durable_sandbox_lifecycles"):
        return
    op.create_table(
        "durable_sandbox_lifecycles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("scope_digest", sa.String(64), nullable=False),
        sa.Column("lifecycle_token", sa.String(64), nullable=False),
        sa.Column("backend_lifecycle_digest", sa.String(64), nullable=False),
        sa.Column("owner_token", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("lease_attempt_id", sa.String(64), nullable=False),
        sa.Column("turn_digest", sa.String(64), nullable=True),
        sa.Column("eligible_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("owner_lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delete_claim_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleting_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("delete_attempts", sa.Integer(), server_default="0", nullable=False),
        sa.CheckConstraint("length(scope_digest) = 64", name="ck_dsl_scope_digest"),
        sa.CheckConstraint(
            "length(lifecycle_token) = 64", name="ck_dsl_lifecycle_token"
        ),
        sa.CheckConstraint(
            "length(backend_lifecycle_digest) = 64",
            name="ck_dsl_backend_lifecycle_digest",
        ),
        sa.CheckConstraint("length(owner_token) = 64", name="ck_dsl_owner_token"),
        sa.CheckConstraint(
            "turn_digest IS NULL OR length(turn_digest) = 64",
            name="ck_dsl_turn_digest",
        ),
        sa.CheckConstraint("version >= 1", name="ck_dsl_version"),
        sa.CheckConstraint("delete_attempts >= 0", name="ck_dsl_delete_attempts"),
        sa.CheckConstraint(
            "state IN ('registered', 'ready', 'deleting')", name="ck_dsl_state"
        ),
        sa.CheckConstraint(
            "(state = 'registered' AND ready_at IS NULL AND deleting_at IS NULL "
            "AND delete_claim_expires_at IS NULL) OR "
            "(state = 'ready' AND ready_at IS NOT NULL AND deleting_at IS NULL "
            "AND delete_claim_expires_at IS NULL) OR "
            "(state = 'deleting' AND deleting_at IS NOT NULL "
            "AND delete_claim_expires_at IS NOT NULL)",
            name="ck_dsl_state_shape",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scope_digest", name="uq_dsl_scope_digest"),
        sa.UniqueConstraint("lifecycle_token", name="uq_dsl_lifecycle_token"),
        sa.UniqueConstraint(
            "backend_lifecycle_digest", name="uq_dsl_backend_lifecycle_digest"
        ),
    )
    op.create_index(
        "ix_dsl_reclaim",
        "durable_sandbox_lifecycles",
        [
            "state",
            "eligible_at",
            "retry_at",
            "delete_claim_expires_at",
            "id",
        ],
    )
    op.create_index(
        "ix_dsl_task_attempt",
        "durable_sandbox_lifecycles",
        ["task_id", "run_id", "lease_attempt_id"],
    )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("durable_sandbox_lifecycles"):
        return
    op.drop_index("ix_dsl_task_attempt", table_name="durable_sandbox_lifecycles")
    op.drop_index("ix_dsl_reclaim", table_name="durable_sandbox_lifecycles")
    op.drop_table("durable_sandbox_lifecycles")
