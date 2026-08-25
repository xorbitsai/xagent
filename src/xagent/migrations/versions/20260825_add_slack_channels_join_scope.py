"""add Slack channels:join OAuth scope

Revision ID: 20260825_add_slack_channels_join_scope
Revises: a3b70c638cc3
Create Date: 2026-08-25

"""

import json
import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "20260825_add_slack_channels_join_scope"
down_revision: Union[str, None] = "a3b70c638cc3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("oauth_scopes", sa.JSON),
)

OAUTH_PROVIDERS_TABLE = sa.table(
    "oauth_providers",
    sa.column("provider_name", sa.String),
    sa.column("default_scopes", sa.JSON),
)

USER_OAUTH_TABLE = sa.table(
    "user_oauth",
    sa.column("provider", sa.String),
    sa.column("access_token", sa.String),
    sa.column("refresh_token", sa.String),
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


def _slack_channels_join_scope_already_present(bind: sa.engine.Connection) -> bool:
    """Whether channels:join is already in the persisted scopes.

    Guards upgrade() against wiping already-reconnected grants a second
    time on a re-run against an already-current DB (a direct module
    invocation, an ops retry, or a stamp+reapply) — without this,
    _invalidate_existing_slack_grants would fire unconditionally on every
    call, even when this migration's own scope change was already applied.
    """
    if not _columns_present(bind, "public_mcp_apps", {"app_id", "oauth_scopes"}):
        return False

    current = bind.execute(
        sa.select(PUBLIC_MCP_APPS_TABLE.c.oauth_scopes).where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
        )
    ).scalar()
    if isinstance(current, str):
        try:
            current = json.loads(current)
        except (TypeError, ValueError):
            current = None
    return bool(current) and "channels:join" in current


def _invalidate_existing_slack_grants(bind: sa.engine.Connection) -> None:
    """Force reconnection so the user has a chance to grant channels:join.

    Same rationale as 20260812's equivalent function: an existing bot-token
    connection would keep showing "Connected" and only fail at call time with
    a raw Slack missing_scope error the first time the user agrees to add the
    bot to a public channel via slack_join_channel.
    """
    if not _columns_present(bind, "user_oauth", {"provider", "access_token"}):
        return

    values: dict[str, object] = {"access_token": ""}
    if _columns_present(bind, "user_oauth", {"refresh_token"}):
        values["refresh_token"] = None

    result = bind.execute(
        sa.update(USER_OAUTH_TABLE)
        .where(USER_OAUTH_TABLE.c.provider == APP_ID)
        .values(**values)
    )
    if result.rowcount:
        logger.warning(
            "Disconnected %d existing Slack grant(s) for the new "
            "channels:join scope. Affected users must reconnect the Slack "
            "connector.",
            result.rowcount,
        )


def upgrade() -> None:
    bind = op.get_bind()
    already_current = _slack_channels_join_scope_already_present(bind)
    _set_slack_scopes(bind, CURRENT_SCOPES)
    _set_slack_provider_default_scopes_if_unchanged(
        bind, PREVIOUS_SCOPES, CURRENT_SCOPES
    )
    if not already_current:
        _invalidate_existing_slack_grants(bind)


def downgrade() -> None:
    # The cleared access tokens are gone for good (that's the point — force a
    # reconnect); there is nothing meaningful to restore for user_oauth here.
    bind = op.get_bind()
    _set_slack_scopes(bind, PREVIOUS_SCOPES)
    _set_slack_provider_default_scopes_if_unchanged(
        bind, CURRENT_SCOPES, PREVIOUS_SCOPES
    )
