"""add per-user Auto model configuration

Revision ID: 20260904_add_auto_model_config
Revises: 20260905_task_execution_events
Create Date: 2026-09-04 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260904_add_auto_model_config"
down_revision: Union[str, None] = "20260905_task_execution_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    existing_tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "auto_model_configs" not in existing_tables:
        config_foreign_keys = []
        if "models" in existing_tables:
            config_foreign_keys.extend(
                [
                    sa.ForeignKeyConstraint(
                        ["fallback_model_id"], ["models.id"], ondelete="RESTRICT"
                    ),
                    sa.ForeignKeyConstraint(
                        ["router_model_id"], ["models.id"], ondelete="CASCADE"
                    ),
                ]
            )
        if "users" in existing_tables:
            config_foreign_keys.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )

        op.create_table(
            "auto_model_configs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("router_model_id", sa.Integer(), nullable=False),
            sa.Column("strategy", sa.String(length=20), nullable=False),
            sa.Column("fallback_model_id", sa.Integer(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            *config_foreign_keys,
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "router_model_id", name="uq_auto_model_config_router_model"
            ),
            sa.UniqueConstraint("user_id", name="uq_auto_model_config_user"),
        )
        op.create_index(
            op.f("ix_auto_model_configs_id"),
            "auto_model_configs",
            ["id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_auto_model_configs_user_id"),
            "auto_model_configs",
            ["user_id"],
            unique=False,
        )
    if "auto_model_candidates" not in existing_tables:
        candidate_foreign_keys = [
            sa.ForeignKeyConstraint(
                ["config_id"], ["auto_model_configs.id"], ondelete="CASCADE"
            )
        ]
        if "models" in existing_tables:
            candidate_foreign_keys.append(
                sa.ForeignKeyConstraint(
                    ["target_model_id"], ["models.id"], ondelete="RESTRICT"
                )
            )

        op.create_table(
            "auto_model_candidates",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("config_id", sa.Integer(), nullable=False),
            sa.Column("routing_model_id", sa.String(length=200), nullable=False),
            sa.Column("target_model_id", sa.Integer(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            *candidate_foreign_keys,
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "config_id",
                "routing_model_id",
                name="uq_auto_candidate_routing_model",
            ),
            sa.UniqueConstraint(
                "config_id",
                "target_model_id",
                name="uq_auto_candidate_target_model",
            ),
        )
        op.create_index(
            op.f("ix_auto_model_candidates_config_id"),
            "auto_model_candidates",
            ["config_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_auto_model_candidates_id"),
            "auto_model_candidates",
            ["id"],
            unique=False,
        )


def downgrade() -> None:
    existing_tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "auto_model_candidates" in existing_tables:
        op.drop_index(
            op.f("ix_auto_model_candidates_id"), table_name="auto_model_candidates"
        )
        op.drop_index(
            op.f("ix_auto_model_candidates_config_id"),
            table_name="auto_model_candidates",
        )
        op.drop_table("auto_model_candidates")
    if "auto_model_configs" in existing_tables:
        op.drop_index(
            op.f("ix_auto_model_configs_user_id"), table_name="auto_model_configs"
        )
        op.drop_index(op.f("ix_auto_model_configs_id"), table_name="auto_model_configs")
        op.drop_table("auto_model_configs")
