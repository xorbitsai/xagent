"""add deployments table and workforce_runs idempotency_key

Foundation for exposing Workforce through external channels (#946 / #805):
- ``deployments``: shared per-(owner_type, owner_id) external-deployment
  config for agents and workforces (widget / share-link channels).
- ``workforce_runs.idempotency_key``: caller-supplied dedup token with a
  unique (workforce_id, idempotency_key) index.

Revision ID: 20260721_add_workforce_deployment_foundation
Revises: 20260721_backfill_strip_agent_category
Create Date: 2026-07-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260721_add_workforce_deployment_foundation"
down_revision: Union[str, None] = "20260721_backfill_strip_agent_category"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEPLOYMENTS_TABLE = "deployments"
RUNS_TABLE = "workforce_runs"
RUNS_COLUMN = "idempotency_key"
RUNS_UNIQUE_INDEX = "uq_workforce_run_idempotency"


def _existing_columns(inspector: Inspector, table: str) -> list[str]:
    return [col["name"] for col in inspector.get_columns(table)]


def _existing_indexes(inspector: Inspector, table: str) -> list[str]:
    return [str(index["name"]) for index in inspector.get_indexes(table)]


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    table_names = inspector.get_table_names()

    if DEPLOYMENTS_TABLE not in table_names:
        op.create_table(
            DEPLOYMENTS_TABLE,
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("owner_type", sa.String(length=20), nullable=False),
            sa.Column("owner_id", sa.Integer(), nullable=False),
            sa.Column(
                "widget_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
            sa.Column("allowed_domains", sa.JSON(), nullable=True),
            sa.Column("widget_key", sa.String(length=255), nullable=True),
            sa.Column(
                "share_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
            sa.Column("share_token", sa.String(length=255), nullable=True),
            sa.Column("share_updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.UniqueConstraint("owner_type", "owner_id", name="uq_deployment_owner"),
        )
        op.create_index("ix_deployments_id", DEPLOYMENTS_TABLE, ["id"])
        op.create_index("ix_deployments_owner_type", DEPLOYMENTS_TABLE, ["owner_type"])
        op.create_index("ix_deployments_owner_id", DEPLOYMENTS_TABLE, ["owner_id"])
        op.create_index(
            "ix_deployments_widget_key", DEPLOYMENTS_TABLE, ["widget_key"], unique=True
        )
        op.create_index(
            "ix_deployments_share_token",
            DEPLOYMENTS_TABLE,
            ["share_token"],
            unique=True,
        )

    if RUNS_TABLE in table_names:
        if RUNS_COLUMN not in _existing_columns(inspector, RUNS_TABLE):
            op.add_column(
                RUNS_TABLE,
                sa.Column(RUNS_COLUMN, sa.String(length=128), nullable=True),
            )
        if RUNS_UNIQUE_INDEX not in _existing_indexes(inspector, RUNS_TABLE):
            op.create_index(
                RUNS_UNIQUE_INDEX,
                RUNS_TABLE,
                ["workforce_id", RUNS_COLUMN],
                unique=True,
            )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    table_names = inspector.get_table_names()

    if RUNS_TABLE in table_names:
        if RUNS_UNIQUE_INDEX in _existing_indexes(inspector, RUNS_TABLE):
            op.drop_index(RUNS_UNIQUE_INDEX, table_name=RUNS_TABLE)
        if RUNS_COLUMN in _existing_columns(inspector, RUNS_TABLE):
            op.drop_column(RUNS_TABLE, RUNS_COLUMN)

    if DEPLOYMENTS_TABLE in table_names:
        op.drop_table(DEPLOYMENTS_TABLE)
