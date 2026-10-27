"""Persist Auto routing mode and candidate order.

Revision ID: 20261026_auto_sequential_failover
Revises: 20261023_add_mcp_signed_uploads
Create Date: 2026-10-26 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261026_auto_sequential_failover"
down_revision: Union[str, None] = "20261023_add_mcp_signed_uploads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _unique_constraints(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return set()
    return {
        constraint["name"]
        for constraint in inspector.get_unique_constraints(table_name)
        if constraint.get("name")
    }


def upgrade() -> None:
    if "auto_model_configs" not in sa.inspect(op.get_bind()).get_table_names():
        return

    if "mode" not in _columns("auto_model_configs"):
        op.add_column(
            "auto_model_configs",
            sa.Column(
                "mode",
                sa.String(length=32),
                nullable=False,
                server_default="routing",
            ),
        )

    old_profile_constraint = "uq_auto_candidate_routing_model"
    if old_profile_constraint in _unique_constraints("auto_model_candidates"):
        with op.batch_alter_table("auto_model_candidates") as batch_op:
            batch_op.drop_constraint(old_profile_constraint, type_="unique")

    if "position" not in _columns("auto_model_candidates"):
        op.add_column(
            "auto_model_candidates",
            sa.Column("position", sa.Integer(), nullable=True),
        )
        rows = op.get_bind().execute(
            sa.text(
                "SELECT id, config_id FROM auto_model_candidates ORDER BY config_id, id"
            )
        )
        positions: dict[int, int] = {}
        for candidate_id, config_id in rows:
            position = positions.get(config_id, 0)
            op.get_bind().execute(
                sa.text(
                    "UPDATE auto_model_candidates SET position = :position "
                    "WHERE id = :candidate_id"
                ),
                {"position": position, "candidate_id": candidate_id},
            )
            positions[config_id] = position + 1
        with op.batch_alter_table("auto_model_candidates") as batch_op:
            batch_op.alter_column(
                "position", existing_type=sa.Integer(), nullable=False
            )

    if "uq_auto_candidate_position" not in _unique_constraints("auto_model_candidates"):
        with op.batch_alter_table("auto_model_candidates") as batch_op:
            batch_op.create_unique_constraint(
                "uq_auto_candidate_position", ["config_id", "position"]
            )


def downgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "auto_model_candidates" in tables and "position" in _columns(
        "auto_model_candidates"
    ):
        # The old schema cannot represent repeated profiles. Refuse a lossy
        # downgrade instead of silently deleting endpoint bindings.
        duplicate = (
            op.get_bind()
            .execute(
                sa.text(
                    "SELECT config_id, routing_model_id, COUNT(*) AS binding_count "
                    "FROM auto_model_candidates "
                    "GROUP BY config_id, routing_model_id HAVING COUNT(*) > 1 "
                    "ORDER BY config_id, routing_model_id"
                )
            )
            .first()
        )
        if duplicate is not None:
            config_id, routing_model_id, binding_count = duplicate
            raise RuntimeError(
                "Cannot downgrade Auto sequential failover without losing "
                f"bindings: config {config_id} maps profile "
                f"{routing_model_id!r} {binding_count} times"
            )
        with op.batch_alter_table("auto_model_candidates") as batch_op:
            batch_op.drop_constraint("uq_auto_candidate_position", type_="unique")
            batch_op.drop_column("position")
            batch_op.create_unique_constraint(
                "uq_auto_candidate_routing_model",
                ["config_id", "routing_model_id"],
            )
    if "auto_model_configs" in tables and "mode" in _columns("auto_model_configs"):
        op.drop_column("auto_model_configs", "mode")
