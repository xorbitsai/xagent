"""rename the Atlassian connector display name to Confluence

Revision ID: 20261110_rename_atlassian_to_confluence
Revises: 20261109_seed_jira_ticket_mcp_app
Create Date: 2026-11-10

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import canonicalize_builtin_identity

logger = logging.getLogger(__name__)

revision: str = "20261110_rename_atlassian_to_confluence"
down_revision: Union[str, None] = "20261109_seed_jira_ticket_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
)

APP_ID = "atlassian"
# The value 20261105_update_atlassian_name_and_description.py left in place.
PREVIOUS_NAME = "Atlassian (Confluence, Bitbucket)"
CURRENT_NAME = "Confluence"
# New name for a pre-existing custom server that collides with CURRENT_NAME.
CUSTOM_SERVER_NAME = "Confluence (custom)"

MCP_SERVERS_TABLE = sa.table(
    "mcp_servers",
    sa.column("id", sa.Integer),
    sa.column("name", sa.String),
)


def _name_column_present(bind: sa.engine.Connection) -> bool:
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return False
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    return {"app_id", "name"}.issubset(columns)


def _rename_if_unchanged(
    bind: sa.engine.Connection, expected_current: str, new_value: str
) -> None:
    """Rename only while the row still holds the seeded value.

    name is code-owned (protected from admin PATCH and drift-checked by
    validate_builtin_public_mcp_apps), so only an unexpected value is left
    alone. The mcp_oauth connect path resolves this app's shared server row by
    app_id, so existing connections survive the rename; the connector listing,
    however, also matches servers by canonicalized display name, which is what
    _free_up_confluence_server_name guards.
    """
    if not _name_column_present(bind):
        return
    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID,
            PUBLIC_MCP_APPS_TABLE.c.name == expected_current,
        )
        .values(name=new_value)
    )


def _free_up_confluence_server_name(bind: sa.engine.Connection) -> None:
    """Rename custom mcp_servers rows that would collide with the new name.

    The catalog listing takes the display name from builtin_mcp_registry, so
    "confluence" becomes a catalog key for the atlassian app as soon as this
    code ships. A custom server that already carries that name would then be
    presented as the official card (and hidden from the local picker). This
    app's own shared row is named after its app_id ("atlassian"), never after
    its display name, so any row matching here is a custom one. Servers are
    referenced by id, so renaming is safe. The rename is not reverted on
    downgrade: the old name is no longer known.
    """
    inspector = sa.inspect(bind)
    if "mcp_servers" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("mcp_servers")}
    if not {"id", "name"}.issubset(columns):
        return

    rows = list(
        bind.execute(
            sa.select(MCP_SERVERS_TABLE.c.id, MCP_SERVERS_TABLE.c.name)
        ).mappings()
    )
    colliding = [
        row
        for row in rows
        if canonicalize_builtin_identity(row["name"])
        == canonicalize_builtin_identity(CURRENT_NAME)
    ]
    taken = {str(row["name"]) for row in rows}
    for row in colliding:
        # mcp_servers.name is unique, so pick the first free suffix.
        candidate = CUSTOM_SERVER_NAME
        suffix = 2
        while candidate in taken:
            candidate = f"{CUSTOM_SERVER_NAME} {suffix}"
            suffix += 1
        taken.add(candidate)
        logger.warning(
            "Renaming custom MCP server %s ('%s') to '%s': '%s' is now the "
            "built-in Atlassian connector's display name.",
            row["id"],
            row["name"],
            candidate,
            CURRENT_NAME,
        )
        bind.execute(
            sa.update(MCP_SERVERS_TABLE)
            .where(MCP_SERVERS_TABLE.c.id == row["id"])
            .values(name=candidate)
        )


def upgrade() -> None:
    bind = op.get_bind()
    # Independent of the public_mcp_apps row: the runtime name comes from the
    # registry, so the collision exists even if that row was left untouched.
    _free_up_confluence_server_name(bind)
    _rename_if_unchanged(bind, PREVIOUS_NAME, CURRENT_NAME)


def downgrade() -> None:
    _rename_if_unchanged(op.get_bind(), CURRENT_NAME, PREVIOUS_NAME)
