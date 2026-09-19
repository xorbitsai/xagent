"""Add durable create-operation phases and successor generations.

Revision ID: 20260916_durable_create_operations
Revises: 20260911_global_memory_authority

The substrate remains dormant: this migration adds ownership metadata and
cross-dialect generation fencing but wires no runtime consumer.
"""

import secrets

import sqlalchemy as sa
from alembic import op

revision = "20260916_durable_create_operations"
down_revision = "20260911_global_memory_authority"
branch_labels = None
depends_on = None

TABLE = "durable_sandbox_lifecycles"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(TABLE):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if "create_phase" in columns:
        return

    with op.batch_alter_table(TABLE) as batch:
        batch.add_column(sa.Column("active_scope_digest", sa.String(64)))
        batch.add_column(sa.Column("create_operation_token", sa.String(64)))
        batch.add_column(sa.Column("create_terminal_outcome", sa.String(16)))
        batch.add_column(
            sa.Column(
                "create_phase",
                sa.String(16),
                nullable=False,
                server_default="not_started",
            )
        )

    rows = bind.execute(sa.text(f"SELECT id, scope_digest, state FROM {TABLE}")).all()
    for row_id, scope_digest, state in rows:
        bind.execute(
            sa.text(
                f"UPDATE {TABLE} SET active_scope_digest = :scope, "
                "create_operation_token = :token WHERE id = :row_id"
            ),
            {
                "scope": scope_digest if state in ("registered", "ready") else None,
                "token": secrets.token_hex(32),
                "row_id": row_id,
            },
        )

    with op.batch_alter_table(TABLE) as batch:
        batch.alter_column("create_operation_token", nullable=False)
        batch.drop_constraint("uq_dsl_scope_digest", type_="unique")
        batch.create_unique_constraint(
            "uq_dsl_active_scope_digest", ["active_scope_digest"]
        )
        batch.create_unique_constraint(
            "uq_dsl_create_operation_token", ["create_operation_token"]
        )
        batch.create_index("ix_dsl_scope_digest", ["scope_digest"], unique=False)
        batch.create_check_constraint(
            "ck_dsl_active_scope_digest",
            "active_scope_digest IS NULL OR length(active_scope_digest) = 64",
        )
        batch.create_check_constraint(
            "ck_dsl_create_operation_token",
            "length(create_operation_token) = 64",
        )
        batch.create_check_constraint(
            "ck_dsl_create_phase",
            "create_phase IN ('not_started', 'may_publish', 'terminal', 'observed')",
        )
        batch.create_check_constraint(
            "ck_dsl_create_terminal_outcome",
            "(create_phase = 'terminal' AND create_terminal_outcome IS NOT NULL AND "
            "create_terminal_outcome IN ('success', 'terminal_absent')) OR "
            "(create_phase != 'terminal' AND create_terminal_outcome IS NULL)",
        )
        batch.create_check_constraint(
            "ck_dsl_active_generation",
            "((state = 'registered' OR state = 'ready') "
            "AND active_scope_digest IS NOT NULL "
            "AND active_scope_digest = scope_digest) OR "
            "(state = 'deleting' AND active_scope_digest IS NULL)",
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(TABLE):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if "create_phase" not in columns:
        return
    duplicate_scope = bind.execute(
        sa.text(
            f"SELECT scope_digest FROM {TABLE} GROUP BY scope_digest "
            "HAVING count(*) > 1 LIMIT 1"
        )
    ).first()
    if duplicate_scope is not None:
        raise RuntimeError(
            "cannot downgrade durable create operations while successor "
            "generations coexist"
        )
    ambiguous_create = bind.execute(
        sa.text(
            f"SELECT id FROM {TABLE} "
            "WHERE create_phase IN ('may_publish', 'observed') LIMIT 1"
        )
    ).first()
    if ambiguous_create is not None:
        raise RuntimeError(
            "cannot downgrade durable create operations while an ambiguous "
            "generation may still exist"
        )
    with op.batch_alter_table(TABLE) as batch:
        batch.drop_constraint("ck_dsl_active_generation", type_="check")
        batch.drop_constraint("ck_dsl_create_terminal_outcome", type_="check")
        batch.drop_constraint("ck_dsl_create_phase", type_="check")
        batch.drop_constraint("ck_dsl_create_operation_token", type_="check")
        batch.drop_constraint("ck_dsl_active_scope_digest", type_="check")
        batch.drop_constraint("uq_dsl_create_operation_token", type_="unique")
        batch.drop_constraint("uq_dsl_active_scope_digest", type_="unique")
        batch.drop_index("ix_dsl_scope_digest")
        batch.create_unique_constraint("uq_dsl_scope_digest", ["scope_digest"])
        batch.drop_column("create_terminal_outcome")
        batch.drop_column("create_phase")
        batch.drop_column("create_operation_token")
        batch.drop_column("active_scope_digest")
