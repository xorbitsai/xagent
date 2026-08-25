"""add Slack channels:join OAuth scope

Revision ID: 20260825_add_slack_channels_join_scope
Revises: 20260824_seed_google_search_console_mcp_app
Create Date: 2026-08-25

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260825_add_slack_channels_join_scope"
down_revision: Union[str, None] = "20260824_seed_google_search_console_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("description", sa.Text),
    sa.column("oauth_scopes", sa.JSON),
)

OAUTH_PROVIDERS_TABLE = sa.table(
    "oauth_providers",
    sa.column("provider_name", sa.String),
    sa.column("default_scopes", sa.JSON),
)

APP_ID = "slack"
PROVIDER_NAME = "slack"

PREVIOUS_SCOPES = [
    "chat:write",
    "chat:write.public",
    "channels:read",
    "channels:history",
    "groups:read",
    "groups:history",
    "im:read",
    "im:history",
    "mpim:read",
    "mpim:history",
    "reactions:write",
    "files:write",
]
CURRENT_SCOPES = [
    "chat:write",
    "chat:write.public",
    "channels:read",
    "channels:history",
    "channels:join",
    "groups:read",
    "groups:history",
    "im:read",
    "im:history",
    "mpim:read",
    "mpim:history",
    "reactions:write",
    "files:write",
]

PREVIOUS_DESCRIPTION = (
    "Connect to Slack to search and read channel, thread, and DM history, "
    "post messages and replies, react to messages, and upload files, e.g. "
    "incident summaries and recommended fixes."
)
CURRENT_DESCRIPTION = (
    "Connect to Slack to search and read channel, thread, and DM history, "
    "post messages and replies, react to messages, upload files, and (with "
    "your approval) join public channels, e.g. incident summaries and "
    "recommended fixes."
)


def _columns_present(
    bind: sa.engine.Connection, table_name: str, required_columns: set[str]
) -> bool:
    """Whether ``table_name`` exists and has all of ``required_columns``.

    Shared by every guard below: this migration must be a no-op (not an
    error) against a database mid-way through a schema this old, or an
    admin's reduced-schema table, rather than assume a table shape that
    matches only the current model.
    """
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return False
    columns = {c["name"] for c in inspector.get_columns(table_name)}
    return required_columns.issubset(columns)


def _set_slack_scopes(bind: sa.engine.Connection, scopes: list[str]) -> None:
    """Keep the persisted row in sync with the code registry's canonical value.

    Mirrors 20260812_add_slack_history_reactions_files_scopes._set_slack_scopes:
    this does NOT drive what scope is actually requested at OAuth-authorize
    time for a builtin app (that's sourced live from the code registry), it
    only exists so validate_builtin_public_mcp_apps doesn't report drift
    between the registry and an already-seeded row. oauth_scopes is in
    admin_mcp's _BUILTIN_PROTECTED_FIELDS, so an operator can never have
    customized it via the admin PATCH endpoint — safe to overwrite
    unconditionally.
    """
    if not _columns_present(bind, "public_mcp_apps", {"app_id", "oauth_scopes"}):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=scopes)
    )


def _set_slack_provider_default_scopes_if_unchanged(
    bind: sa.engine.Connection, expected_current: list[str], new_value: list[str]
) -> None:
    """Keep oauth_providers.default_scopes for provider "slack" in sync too,
    without clobbering an operator customization.

    Mirrors 20260812's equivalent guard — see that migration for the full
    rationale on why this column (unlike oauth_scopes) drives live behavior
    for the app-id-less authorize path.
    """
    if not _columns_present(
        bind, "oauth_providers", {"provider_name", "default_scopes"}
    ):
        return

    current = bind.execute(
        sa.select(OAUTH_PROVIDERS_TABLE.c.default_scopes).where(
            OAUTH_PROVIDERS_TABLE.c.provider_name == PROVIDER_NAME
        )
    ).scalar()
    if isinstance(current, str):
        try:
            current = json.loads(current)
        except (TypeError, ValueError):
            pass
    if current is not None and current != expected_current:
        return

    bind.execute(
        sa.update(OAUTH_PROVIDERS_TABLE)
        .where(OAUTH_PROVIDERS_TABLE.c.provider_name == PROVIDER_NAME)
        .values(default_scopes=new_value)
    )


def _set_slack_description_if_unchanged(
    bind: sa.engine.Connection, expected_current: str, new_value: str
) -> None:
    """Refresh the stale default description, without clobbering a customization.

    Mirrors 20260812's equivalent guard: description isn't in admin_mcp's
    _BUILTIN_PROTECTED_FIELDS, so an operator can legitimately have edited
    it via the admin PATCH endpoint. Only overwrite when the persisted
    value still equals the last-known canonical description.
    """
    if not _columns_present(bind, "public_mcp_apps", {"app_id", "description"}):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID,
            PUBLIC_MCP_APPS_TABLE.c.description == expected_current,
        )
        .values(description=new_value)
    )


def upgrade() -> None:
    bind = op.get_bind()
    _set_slack_scopes(bind, CURRENT_SCOPES)
    _set_slack_provider_default_scopes_if_unchanged(
        bind, PREVIOUS_SCOPES, CURRENT_SCOPES
    )
    _set_slack_description_if_unchanged(bind, PREVIOUS_DESCRIPTION, CURRENT_DESCRIPTION)
    # Deliberately does NOT force-disconnect existing Slack grants: Slack
    # doesn't retroactively grant a newly-requested scope to an
    # already-issued token, but it also doesn't revoke or break what that
    # token could already do — an existing connection keeps working for
    # every tool it already supported. Only the brand-new
    # slack_join_channel tool needs channels:join, and it surfaces a clear
    # "reconnect" message on missing_scope (see slack.py) rather than this
    # migration forcing every existing user to reconnect just to unlock one
    # new, optional tool.


def downgrade() -> None:
    bind = op.get_bind()
    _set_slack_scopes(bind, PREVIOUS_SCOPES)
    _set_slack_provider_default_scopes_if_unchanged(
        bind, CURRENT_SCOPES, PREVIOUS_SCOPES
    )
    _set_slack_description_if_unchanged(bind, CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION)
