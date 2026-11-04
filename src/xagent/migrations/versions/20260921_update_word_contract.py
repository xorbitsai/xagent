"""align existing Word catalog rows with the shipped connector contract

Revision ID: 20260921_word_contract
Revises: 20260917_seed_word_mcp_app
Create Date: 2026-09-21 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import builtin_provenance_identity

# revision identifiers, used by Alembic.
revision: str = "20260921_word_contract"
down_revision: Union[str, None] = "20260917_seed_word_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ID = "word"
BUILTIN_PROVENANCE = {"registry": "xagent", "app_id": APP_ID, "version": 1}
OLD_DESCRIPTION = (
    "Connect to Word to read, create, and edit documents stored on OneDrive or "
    "SharePoint."
)
NEW_DESCRIPTION = (
    "Connect to Word to create documents and read or edit top-level main-body "
    "paragraphs stored on OneDrive or SharePoint. Tables, headers, footers, text "
    "boxes, notes, and tracked changes are excluded."
)
OUTPUT_LIMIT_ENV = "XAGENT_TOOL_MAX_OUTPUT_LENGTH"

PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("description", sa.Text),
    sa.column("launch_config", sa.JSON),
)


def _is_owned(launch_config: object) -> bool:
    return isinstance(launch_config, dict) and builtin_provenance_identity(
        launch_config.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _owned_launch_config(bind: sa.engine.Connection) -> dict | None:
    row = (
        bind.execute(
            sa.select(PUBLIC_MCP_APPS_TABLE.c.launch_config).where(
                PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
            )
        )
        .mappings()
        .first()
    )
    if row is None or not _is_owned(row["launch_config"]):
        return None
    return dict(row["launch_config"])


def _description_if_unchanged(expected_current: str, new_value: str) -> sa.Case:
    """A CASE expression re-checking ``description`` at UPDATE time.

    description is not a protected builtin field (admin_mcp.py's
    _BUILTIN_PROTECTED_FIELDS), so an operator may have legitimately
    customized it via the admin PATCH endpoint at any time. Evaluating the
    "is it still canonical" condition inside the UPDATE itself (rather than
    branching in Python on a value this function already SELECTed earlier)
    means the database re-reads the current value atomically at write time,
    so a customization landing between this migration's read and its write
    can't be silently clobbered by a decision made against stale state.
    Mirrors 20260911_update_onedrive_description.py's identical rationale,
    expressed atomically instead of read-then-write.
    """
    return sa.case(
        (PUBLIC_MCP_APPS_TABLE.c.description == expected_current, new_value),
        else_=PUBLIC_MCP_APPS_TABLE.c.description,
    )


def upgrade() -> None:
    bind = op.get_bind()
    if "public_mcp_apps" not in set(sa.inspect(bind).get_table_names()):
        return
    launch_config = _owned_launch_config(bind)
    if launch_config is None:
        return
    static_env = dict(launch_config.get("static_env") or {})
    static_env[OUTPUT_LIMIT_ENV] = OUTPUT_LIMIT_ENV
    launch_config["static_env"] = static_env
    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(
            description=_description_if_unchanged(OLD_DESCRIPTION, NEW_DESCRIPTION),
            launch_config=launch_config,
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    if "public_mcp_apps" not in set(sa.inspect(bind).get_table_names()):
        return
    launch_config = _owned_launch_config(bind)
    if launch_config is None:
        return
    static_env = dict(launch_config.get("static_env") or {})
    if static_env.get(OUTPUT_LIMIT_ENV) == OUTPUT_LIMIT_ENV:
        static_env.pop(OUTPUT_LIMIT_ENV)
    if static_env:
        launch_config["static_env"] = static_env
    else:
        launch_config.pop("static_env", None)
    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(
            description=_description_if_unchanged(NEW_DESCRIPTION, OLD_DESCRIPTION),
            launch_config=launch_config,
        )
    )
