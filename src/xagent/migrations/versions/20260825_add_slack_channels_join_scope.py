"""add Slack channels:join OAuth scope

Revision ID: 20260825_add_slack_channels_join_scope
Revises: 20260824_seed_google_search_console_mcp_app
Create Date: 2026-08-25

"""

import json
import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

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


def _merge_slack_provider_default_scopes(
    bind: sa.engine.Connection,
    add_scopes: list[str],
    remove_scopes: list[str],
    full_scopes_if_null: list[str],
) -> None:
    """Add/remove this migration's own scope delta from
    oauth_providers.default_scopes for provider "slack", preserving any
    operator customization instead of skipping the update entirely
    whenever the persisted value isn't an exact historical snapshot.

    The prior "skip unless it still equals PREVIOUS_SCOPES exactly" shape
    (mirrored from 20260812's equivalent guard) has a real bug: once an
    operator adds even one custom scope here, every later migration that
    only compares against its own PREVIOUS_SCOPES permanently no-ops
    forever, silently dropping any newly-required scope (channels:join,
    here) from the app-id-less authorize path (GET /api/auth/{provider}/
    login with no app_id) — see api/auth.py's _merge_oauth_scopes for why
    that path reads this column directly rather than unioning in the app's
    own (always-current) oauth_scopes. Merging by delta instead means a
    customization survives, and the scope this migration actually owns
    still gets added regardless of what else is in the list.

    add_scopes/remove_scopes are this migration's own delta (e.g.
    ["channels:join"], not the full CURRENT_SCOPES/PREVIOUS_SCOPES list) —
    so an operator's unrelated custom scope is never touched either
    direction.

    full_scopes_if_null is written instead of the delta specifically when
    default_scopes is column-NULL (never populated) — mirroring
    20260812_add_slack_history_reactions_files_scopes.py's
    _set_slack_provider_default_scopes_if_unchanged, which writes its own
    full new_value in that same case. A NULL row has no informative
    starting value to merge a delta into: writing only ["channels:join"]
    there would silently drop every other scope this connector needs from
    the app-id-less authorize path. This does NOT apply to a row that is
    an explicit, already-stored empty list ([]) rather than NULL — that is
    an operator customization (deliberately cleared via the admin PATCH
    endpoint) like any other, and gets the same delta-merge treatment as
    a non-empty customization.

    A default_scopes value that is a string Slack/JSON can't parse is left
    untouched (logged, not overwritten) rather than coerced to an empty
    list and then clobbered by a merge or a full write — either would
    silently discard whatever was actually stored, taking away the
    operator's ability to inspect what went wrong.

    Locks the row for the rest of this migration's transaction on
    PostgreSQL (SELECT ... FOR UPDATE) so a customization committed via
    the admin PATCH endpoint between this read and the write below can't
    be silently clobbered. SQLite has no real concurrent-writer scenario
    for a migration run and doesn't support FOR UPDATE, so it's skipped
    there (also consistent with SQLite migrations here assuming
    non-transactional DDL).
    """
    if not _columns_present(
        bind, "oauth_providers", {"provider_name", "default_scopes"}
    ):
        return

    select_stmt = sa.select(OAUTH_PROVIDERS_TABLE.c.default_scopes).where(
        OAUTH_PROVIDERS_TABLE.c.provider_name == PROVIDER_NAME
    )
    if bind.dialect.name == "postgresql":
        select_stmt = select_stmt.with_for_update()

    row = bind.execute(select_stmt).first()
    if row is None:
        # No "slack" provider row at all -- nothing to merge into.
        return

    current = row[0]
    if current is None:
        # Column-NULL, never populated -- there is no delta to merge into,
        # so seed the full canonical set for this migration direction
        # instead of leaving the row with only this migration's own scope.
        current_list: list[str] = []
        new_scopes = list(full_scopes_if_null)
    else:
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except (TypeError, ValueError):
                logger.warning(
                    "oauth_providers.default_scopes for provider %r is "
                    "not valid JSON (%r) -- leaving it untouched rather "
                    "than overwriting unparsable data.",
                    PROVIDER_NAME,
                    current,
                )
                return
        current_list = list(current) if isinstance(current, list) else []
        remove = set(remove_scopes)
        new_scopes = [scope for scope in current_list if scope not in remove]
        for scope in add_scopes:
            if scope not in new_scopes:
                new_scopes.append(scope)

    if new_scopes == current_list:
        return

    bind.execute(
        sa.update(OAUTH_PROVIDERS_TABLE)
        .where(OAUTH_PROVIDERS_TABLE.c.provider_name == PROVIDER_NAME)
        .values(default_scopes=new_scopes)
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


_ADDED_SCOPES = [scope for scope in CURRENT_SCOPES if scope not in PREVIOUS_SCOPES]


def upgrade() -> None:
    bind = op.get_bind()
    _set_slack_scopes(bind, CURRENT_SCOPES)
    _merge_slack_provider_default_scopes(
        bind,
        add_scopes=_ADDED_SCOPES,
        remove_scopes=[],
        full_scopes_if_null=CURRENT_SCOPES,
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
    _merge_slack_provider_default_scopes(
        bind,
        add_scopes=[],
        remove_scopes=_ADDED_SCOPES,
        full_scopes_if_null=PREVIOUS_SCOPES,
    )
    _set_slack_description_if_unchanged(bind, CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION)
