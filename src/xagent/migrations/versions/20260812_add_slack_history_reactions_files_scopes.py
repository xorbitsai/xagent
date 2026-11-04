"""add Slack history, reaction, and file-upload OAuth scopes

Revision ID: 20260812_add_slack_history_reactions_files_scopes
Revises: 20260810_add_hubspot_marketing_scopes
Create Date: 2026-08-12

"""

import json
import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "20260812_add_slack_history_reactions_files_scopes"
down_revision: Union[str, None] = "20260810_add_hubspot_marketing_scopes"
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

USER_OAUTH_TABLE = sa.table(
    "user_oauth",
    sa.column("provider", sa.String),
    sa.column("access_token", sa.String),
    sa.column("refresh_token", sa.String),
)

APP_ID = "slack"
PROVIDER_NAME = "slack"

PREVIOUS_SCOPES = ["chat:write", "chat:write.public", "channels:read"]
CURRENT_SCOPES = [
    *PREVIOUS_SCOPES,
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

PREVIOUS_DESCRIPTION = (
    "Connect to Slack to list channels and post messages, e.g. incident "
    "summaries and recommended fixes."
)
CURRENT_DESCRIPTION = (
    "Connect to Slack to search and read channel, thread, and DM history, "
    "post messages and replies, react to messages, and upload files, e.g. "
    "incident summaries and recommended fixes."
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

    This does NOT drive what scope is actually requested at OAuth-authorize
    time: for a builtin app, _app_to_dict sources oauth_scopes from
    get_builtin_execution_fields (the code registry), never from this DB row.
    The write here exists solely so validate_builtin_public_mcp_apps doesn't
    report drift between the registry and an already-seeded row. Like
    HubSpot's equivalent migration, oauth_scopes is in admin_mcp's
    _BUILTIN_PROTECTED_FIELDS, so an operator can never have customized it
    via the admin PATCH endpoint — safe to overwrite unconditionally.
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

    Unlike public_mcp_apps.oauth_scopes, this one DOES drive live behavior:
    the app-id-less authorize path (``GET /api/auth/{provider}/login`` with
    no ``app_id``) merges only ``db_provider.default_scopes`` (see
    ``_merge_oauth_scopes`` in api/auth.py) — it never reads the app's own
    oauth_scopes, which is sourced live from the code registry and would
    otherwise mask this exact staleness. Left unpatched, an already-seeded
    row stays pinned at the original 3 scopes forever and that path keeps
    minting under-scoped tokens after this migration. The app-scoped
    connect flow is unaffected either way, since it unions this value with
    the app's own (always-current) oauth_scopes.

    Unlike oauth_scopes, default_scopes has no admin_mcp builtin-protected-
    fields guard, so an operator can legitimately have edited it via the
    admin PATCH endpoint. Mirrors
    _set_slack_description_if_unchanged's only-overwrite-when-unchanged
    guard — except the comparison happens in Python after a SELECT rather
    than in the UPDATE's WHERE clause, since JSON-column equality operators
    are not guaranteed to compare consistently (key order, whitespace)
    across every SQLAlchemy-supported backend.

    No other app_id shares provider_name "slack" today, so this update
    cannot affect an unrelated app's authorize request.

    Known accepted gap: 20260801_seed_slack_mcp_app.py's seed now bakes in
    scopes this migration hasn't granted yet (backdated for
    20260825_add_slack_channels_join_scope's channels:join), so a fresh
    install's row never equals `expected_current` and this guard always
    skips its own write. Harmless today only because the seeded value is
    already a superset of what this migration would have written — see
    that migration's own docstring, which does a true delta-merge instead
    of this skip-if-changed shape precisely to avoid this failure mode for
    scopes added after it.
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
    # current is None both when no "slack" provider row exists (the update
    # below then affects zero rows, same as before this guard existed) and
    # when an existing row's default_scopes is genuinely NULL; either way
    # there is nothing customized to protect, so the write proceeds.
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

    Unlike ``oauth_scopes``, ``description`` is not in admin_mcp's
    _BUILTIN_PROTECTED_FIELDS, so an operator can legitimately have edited it
    via the admin PATCH endpoint. Only overwrite when the persisted value
    still equals the last-known canonical description (i.e. it was never
    customized); an edited value matches neither PREVIOUS_DESCRIPTION nor
    CURRENT_DESCRIPTION and is left alone in either direction.
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


def _invalidate_existing_slack_grants(bind: sa.engine.Connection) -> None:
    """Force reconnection so the user has a chance to grant the new scopes.

    Slack's oauth.v2.access response scope isn't re-verified after the fact
    by _oauth_account_can_connect (it only checks token presence/expiry), so
    an existing bot-token connection would keep showing "Connected" and only
    fail at call time with a raw Slack missing_scope error for every new
    history/reaction/file tool. Every stored grant was necessarily
    authorized before these scopes existed, so all of them are invalidated
    (same approach as the HubSpot Marketing Hub scope migration).

    refresh_token is cleared too as defense-in-depth, even though Slack bot
    tokens don't expire and this connector's token resolver never calls a
    refresh path for Slack today.
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
            "Disconnected %d existing Slack grant(s) for the new required "
            "history, reaction, and file-upload scopes. Affected users must "
            "reconnect the Slack connector.",
            result.rowcount,
        )


def upgrade() -> None:
    bind = op.get_bind()
    _set_slack_scopes(bind, CURRENT_SCOPES)
    _set_slack_provider_default_scopes_if_unchanged(
        bind, PREVIOUS_SCOPES, CURRENT_SCOPES
    )
    _set_slack_description_if_unchanged(bind, PREVIOUS_DESCRIPTION, CURRENT_DESCRIPTION)
    _invalidate_existing_slack_grants(bind)


def downgrade() -> None:
    # The cleared access tokens are gone for good (that's the point — force a
    # reconnect); there is nothing meaningful to restore for user_oauth here.
    bind = op.get_bind()
    _set_slack_scopes(bind, PREVIOUS_SCOPES)
    _set_slack_provider_default_scopes_if_unchanged(
        bind, CURRENT_SCOPES, PREVIOUS_SCOPES
    )
    _set_slack_description_if_unchanged(bind, CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION)
