"""add_multi_key_support_to_agent_api_keys

Revision ID: 20260703_add_multi_key_support_to_agent_api_keys
Revises: 20260705_add_user_mcpserver_env_source
Create Date: 2026-07-03 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "20260703_add_multi_key_support_to_agent_api_keys"
# Repeatedly re-chained as main kept growing new siblings off whatever
# this PR's migration was currently based on -- this is a long-lived
# branch and main's migrations/versions/ directory is a hot path other
# PRs also land in:
#   1. Originally based on 20260629_add_gmail_watch_states.
#   2. -> 20260625_add_user_skills_tables (merged two stray heads off
#      the same gmail-watch-states parent).
#   3. -> 20260704_merge_alembic_heads (merged two more heads off
#      20260625_add_user_skills_tables:
#      20260703_add_user_mcpserver_env,
#      20260703_backfill_strip_other_tool_category).
#   4. -> 20260705_add_user_mcpserver_env_source, current tip of the
#      20260703_seed_google_maps_mcp_app branch that also forked from
#      20260704_merge_alembic_heads. Chaining onto the latest tip each
#      time keeps a single linear history instead of re-merging.
down_revision: Union[str, None] = "20260705_add_user_mcpserver_env_source"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "agent_api_keys"
_UNIQUE_INDEX = "uq_agent_api_keys_agent_active"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if _TABLE not in inspector.get_table_names():
        return

    existing_columns = {c["name"] for c in inspector.get_columns(_TABLE)}

    if "label" not in existing_columns:
        op.add_column(_TABLE, sa.Column("label", sa.String(length=100), nullable=True))
    if "paused_at" not in existing_columns:
        op.add_column(
            _TABLE, sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True)
        )
    if "last_used_at" not in existing_columns:
        op.add_column(
            _TABLE,
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        )
    if "usage_month" not in existing_columns:
        op.add_column(
            _TABLE, sa.Column("usage_month", sa.String(length=7), nullable=True)
        )
    if "usage_month_calls" not in existing_columns:
        op.add_column(
            _TABLE,
            sa.Column(
                "usage_month_calls",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )

    # Drop the "at most one active key per agent" constraint -- an agent may
    # now hold any number of simultaneously-active keys. Guard on the index
    # actually existing so this migration is safe to re-run.
    existing_indexes = {idx["name"] for idx in inspector.get_indexes(_TABLE)}
    if _UNIQUE_INDEX in existing_indexes:
        op.drop_index(_UNIQUE_INDEX, table_name=_TABLE)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if _TABLE not in inspector.get_table_names():
        return

    # Best-effort: recreating this partial unique index fails if more than
    # one active key exists for any agent at downgrade time (expected once
    # the multi-key feature has been used) -- acceptable for a dev-history
    # downgrade.
    existing_indexes = {idx["name"] for idx in inspector.get_indexes(_TABLE)}
    if _UNIQUE_INDEX not in existing_indexes:
        op.execute(
            sa.text(
                f"CREATE UNIQUE INDEX {_UNIQUE_INDEX} "
                f"ON {_TABLE} (agent_id) WHERE revoked_at IS NULL"
            )
        )

    existing_columns = {c["name"] for c in inspector.get_columns(_TABLE)}
    for column in (
        "usage_month_calls",
        "usage_month",
        "last_used_at",
        "paused_at",
        "label",
    ):
        if column in existing_columns:
            op.drop_column(_TABLE, column)
