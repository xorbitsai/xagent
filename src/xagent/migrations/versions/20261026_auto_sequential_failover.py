"""Persist Auto routing mode and candidate order.

Downgrade is lossless only while every config remains in ``routing`` mode and
each config maps a routing profile at most once. It refuses the downgrade
before any DDL when either condition is false. A successful downgrade removes
the explicit positions, so legacy readers return candidates in id order rather
than preserving a custom order, and removes the redundant ``routing`` mode.

Revision ID: 20261026_auto_sequential_failover
Revises: 20261026_task_requested_dispatch_class
Create Date: 2026-10-26 00:00:00.000000
"""

from collections import defaultdict
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261026_auto_sequential_failover"
down_revision: Union[str, None] = "20261026_task_requested_dispatch_class"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CONFIG_TABLE = "auto_model_configs"
CANDIDATE_TABLE = "auto_model_candidates"
PROFILE_CONSTRAINT = "uq_auto_candidate_routing_model"
POSITION_CONSTRAINT = "uq_auto_candidate_position"


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table_name: str) -> dict[str, dict]:
    if table_name not in _tables():
        return {}
    return {
        str(column["name"]): column
        for column in sa.inspect(op.get_bind()).get_columns(table_name)
    }


def _unique_constraints(table_name: str) -> set[str]:
    if table_name not in _tables():
        return set()
    return {
        str(constraint["name"])
        for constraint in sa.inspect(op.get_bind()).get_unique_constraints(table_name)
        if constraint.get("name")
    }


def _drop_unique_if_present(table_name: str, constraint_name: str) -> None:
    if constraint_name not in _unique_constraints(table_name):
        return
    with op.batch_alter_table(table_name) as batch_op:
        batch_op.drop_constraint(constraint_name, type_="unique")


def _create_unique_if_missing(
    table_name: str, constraint_name: str, columns: list[str]
) -> None:
    if constraint_name in _unique_constraints(table_name):
        return
    with op.batch_alter_table(table_name) as batch_op:
        batch_op.create_unique_constraint(constraint_name, columns)


def _repair_positions() -> None:
    rows = list(
        op.get_bind().execute(
            sa.text(
                f"SELECT id, config_id, position FROM {CANDIDATE_TABLE} "
                "ORDER BY config_id, id"
            )
        )
    )
    by_config: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
    for candidate_id, config_id, position in rows:
        by_config[int(config_id)].append((int(candidate_id), position))

    repair_config_ids = {
        config_id
        for config_id, candidates in by_config.items()
        if any(position is None for _, position in candidates)
        or len({position for _, position in candidates}) != len(candidates)
    }
    if not repair_config_ids:
        return

    # A partially applied migration may already have created this constraint
    # while leaving nullable rows. Drop it before deterministic reassignment so
    # intermediate updates cannot collide, then upgrade() recreates it below.
    _drop_unique_if_present(CANDIDATE_TABLE, POSITION_CONSTRAINT)
    for config_id in sorted(repair_config_ids):
        for position, (candidate_id, _) in enumerate(by_config[config_id]):
            op.get_bind().execute(
                sa.text(
                    f"UPDATE {CANDIDATE_TABLE} SET position = :position "
                    "WHERE id = :candidate_id"
                ),
                {"position": position, "candidate_id": candidate_id},
            )


def _duplicate_profile() -> tuple[int, str, int] | None:
    if CANDIDATE_TABLE not in _tables():
        return None
    row = (
        op.get_bind()
        .execute(
            sa.text(
                f"SELECT config_id, routing_model_id, COUNT(*) AS binding_count "
                f"FROM {CANDIDATE_TABLE} "
                "GROUP BY config_id, routing_model_id HAVING COUNT(*) > 1 "
                "ORDER BY config_id, routing_model_id"
            )
        )
        .first()
    )
    if row is None:
        return None
    return int(row[0]), str(row[1]), int(row[2])


def _non_routing_config() -> tuple[int, str] | None:
    if CONFIG_TABLE not in _tables() or "mode" not in _columns(CONFIG_TABLE):
        return None
    row = (
        op.get_bind()
        .execute(
            sa.text(
                f"SELECT id, mode FROM {CONFIG_TABLE} "
                "WHERE mode <> 'routing' ORDER BY id"
            )
        )
        .first()
    )
    if row is None:
        return None
    return int(row[0]), str(row[1])


def upgrade() -> None:
    if CONFIG_TABLE not in _tables():
        return

    if "mode" not in _columns(CONFIG_TABLE):
        op.add_column(
            CONFIG_TABLE,
            sa.Column(
                "mode",
                sa.String(length=32),
                nullable=False,
                server_default="routing",
            ),
        )

    if CANDIDATE_TABLE not in _tables():
        return

    _drop_unique_if_present(CANDIDATE_TABLE, PROFILE_CONSTRAINT)

    if "position" not in _columns(CANDIDATE_TABLE):
        op.add_column(
            CANDIDATE_TABLE,
            sa.Column("position", sa.Integer(), nullable=True),
        )

    _repair_positions()
    position_column = _columns(CANDIDATE_TABLE)["position"]
    if position_column.get("nullable", True):
        with op.batch_alter_table(CANDIDATE_TABLE) as batch_op:
            batch_op.alter_column(
                "position", existing_type=sa.Integer(), nullable=False
            )

    _create_unique_if_missing(
        CANDIDATE_TABLE,
        POSITION_CONSTRAINT,
        ["config_id", "position"],
    )


def downgrade() -> None:
    # Validate every lossy condition before the first DDL statement. A refused
    # downgrade must leave mode, positions, constraints, and bindings intact.
    non_routing = _non_routing_config()
    if non_routing is not None:
        config_id, mode = non_routing
        raise RuntimeError(
            "Cannot downgrade ordered Auto bindings without losing data: "
            f"config {config_id} uses mode {mode!r}"
        )

    duplicate = _duplicate_profile()
    if duplicate is not None:
        config_id, routing_model_id, binding_count = duplicate
        raise RuntimeError(
            "Cannot downgrade ordered Auto bindings without losing data: "
            f"config {config_id} maps profile {routing_model_id!r} "
            f"{binding_count} times"
        )

    if CANDIDATE_TABLE in _tables():
        _drop_unique_if_present(CANDIDATE_TABLE, POSITION_CONSTRAINT)
        if "position" in _columns(CANDIDATE_TABLE):
            with op.batch_alter_table(CANDIDATE_TABLE) as batch_op:
                batch_op.drop_column("position")
        _create_unique_if_missing(
            CANDIDATE_TABLE,
            PROFILE_CONSTRAINT,
            ["config_id", "routing_model_id"],
        )

    if CONFIG_TABLE in _tables() and "mode" in _columns(CONFIG_TABLE):
        op.drop_column(CONFIG_TABLE, "mode")
