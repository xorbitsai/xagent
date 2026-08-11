"""add HubSpot Marketing Hub scopes (forms, analytics, marketing email, campaigns)

Revision ID: 20260810_add_hubspot_marketing_scopes
Revises: 20260809_add_task_interaction_requests
Create Date: 2026-08-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260810_add_hubspot_marketing_scopes"
down_revision: Union[str, None] = "20260809_add_task_interaction_requests"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("description", sa.Text),
    sa.column("oauth_scopes", sa.JSON),
)

USER_OAUTH_TABLE = sa.table(
    "user_oauth",
    sa.column("provider", sa.String),
    sa.column("access_token", sa.String),
    sa.column("refresh_token", sa.String),
)

APP_ID = "hubspot"

PREVIOUS_SCOPES = [
    "crm.objects.contacts.read",
    "crm.objects.contacts.write",
    "crm.objects.companies.read",
    "crm.objects.companies.write",
    "crm.objects.deals.read",
]
CURRENT_SCOPES = [
    *PREVIOUS_SCOPES,
    "forms",
    "marketing.campaigns.read",
]
# business-intelligence and marketing-email are requested separately via the
# authorize request's optional_scope parameter (see api/auth.py and
# get_builtin_optional_oauth_scopes) - not part of oauth_scopes at all, so
# not persisted here. Both are tier-gated (business-intelligence: Marketing
# Hub Basic+; marketing-email: Enterprise or the transactional email
# add-on); requesting them as required scopes would block reconnection
# entirely for portals below those tiers.

PREVIOUS_DESCRIPTION = (
    "Connect to HubSpot CRM to search, create, and update contacts and "
    "companies, read deals, and log notes."
)
CURRENT_DESCRIPTION = (
    "Connect to HubSpot CRM and Marketing Hub to search, create, and update "
    "contacts and companies, read deals, log notes, read forms and "
    "submissions, pull traffic analytics reports, and read marketing emails "
    "and campaigns."
)


def _set_hubspot_scopes(bind: sa.engine.Connection, scopes: list[str]) -> None:
    """Keep the persisted row in sync with the code registry's canonical value.

    This does NOT drive what scope is actually requested at OAuth-authorize
    time: for a builtin app, _app_to_dict sources oauth_scopes from
    get_builtin_execution_fields (the code registry), never from this DB row.
    The write here exists solely so validate_builtin_public_mcp_apps doesn't
    report drift between the registry and an already-seeded row. Unlike
    ``description`` below, ``oauth_scopes`` is in admin_mcp's
    _BUILTIN_PROTECTED_FIELDS, so an operator can never have customized it
    via the admin PATCH endpoint — safe to overwrite unconditionally.
    """
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    if not {"app_id", "oauth_scopes"}.issubset(columns):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=scopes)
    )


def _set_hubspot_description_if_unchanged(
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
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    if not {"app_id", "description"}.issubset(columns):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID,
            PUBLIC_MCP_APPS_TABLE.c.description == expected_current,
        )
        .values(description=new_value)
    )


def _invalidate_existing_hubspot_grants(bind: sa.engine.Connection) -> None:
    """Force reconnection so the user has a chance to grant the new scopes.

    HubSpot's token-exchange response carries no `scope` field (same
    limitation as Facebook's, see
    20260728_add_facebook_pages_read_user_content_scope.py), so a stored
    grant's actual permissions can't be inspected after the fact — every row
    was necessarily authorized before these scopes existed. Without this,
    _oauth_account_can_connect only checks token presence/expiry, so an
    existing connection keeps showing "Connected" and fails only at call time
    with a raw HubSpot missing-scopes error.

    Unlike the Facebook migration, refresh_token must be cleared too:
    refresh_oauth_token_if_needed (web/tools/config.py) refreshes purely off
    expires_at + refresh_token without ever looking at access_token, and
    HubSpot access tokens expire in ~30 minutes — so a cleared access_token
    with a surviving refresh_token would be silently re-minted (with the old
    scope set) on the next tool call. Meta has no such path; its refresh
    exchanges the access token itself, which clearing already breaks.
    """
    inspector = sa.inspect(bind)
    if "user_oauth" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("user_oauth")}
    if not {"provider", "access_token"}.issubset(columns):
        return

    values: dict[str, object] = {"access_token": ""}
    if "refresh_token" in columns:
        values["refresh_token"] = None

    bind.execute(
        sa.update(USER_OAUTH_TABLE)
        .where(USER_OAUTH_TABLE.c.provider == APP_ID)
        .values(**values)
    )


def upgrade() -> None:
    bind = op.get_bind()
    _set_hubspot_scopes(bind, CURRENT_SCOPES)
    _set_hubspot_description_if_unchanged(
        bind, PREVIOUS_DESCRIPTION, CURRENT_DESCRIPTION
    )
    _invalidate_existing_hubspot_grants(bind)


def downgrade() -> None:
    # The cleared access tokens are gone for good (that's the point — force a
    # reconnect); there is nothing meaningful to restore for user_oauth here.
    bind = op.get_bind()
    _set_hubspot_scopes(bind, PREVIOUS_SCOPES)
    _set_hubspot_description_if_unchanged(
        bind, CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION
    )
