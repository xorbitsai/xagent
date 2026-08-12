"""add HubSpot Marketing Hub scopes (forms, analytics, marketing email, campaigns)

Revision ID: 20260810_add_hubspot_marketing_scopes
Revises: 20260810_add_task_interaction_protocol_version
Create Date: 2026-08-10

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "20260810_add_hubspot_marketing_scopes"
down_revision: Union[str, None] = "20260810_add_task_interaction_protocol_version"
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
]
# business-intelligence, marketing-email, and marketing.campaigns.read are
# requested separately via the authorize request's optional_scope parameter
# (see api/auth.py and get_builtin_execution_fields_and_optional_scopes) -
# not part of oauth_scopes at all, so not persisted here. All three are
# tier-gated (business-intelligence: Marketing Hub Basic+; marketing-email:
# Enterprise or the transactional email add-on; marketing.campaigns.read:
# the Campaigns API itself requires Marketing Hub Professional+); requesting
# any of them as required scopes would block reconnection entirely for
# portals below those tiers.

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
    if not _columns_present(bind, "public_mcp_apps", {"app_id", "oauth_scopes"}):
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

    Unlike the Facebook migration, refresh_token is cleared too, as
    defense-in-depth: the current token resolver (web/tools/config.py)
    already short-circuits on a falsy access_token before ever reaching
    refresh_oauth_token_if_needed, so a surviving refresh_token cannot
    resurrect the old-scoped grant through that specific path today. It's
    cleared anyway so this migration doesn't rely on that resolver's
    current shape staying exactly as it is - a future resolver that checks
    refresh_token independently of access_token would otherwise silently
    re-mint the old-scoped access_token on its next refresh. Meta has no
    such path to defend against; its refresh exchanges the access token
    itself, which clearing already breaks.
    """
    if not _columns_present(bind, "user_oauth", {"provider", "access_token"}):
        return

    columns = {c["name"] for c in sa.inspect(bind).get_columns("user_oauth")}
    values: dict[str, object] = {"access_token": ""}
    if "refresh_token" in columns:
        values["refresh_token"] = None

    result = bind.execute(
        sa.update(USER_OAUTH_TABLE)
        .where(USER_OAUTH_TABLE.c.provider == APP_ID)
        .values(**values)
    )
    if result.rowcount:
        logger.warning(
            "Disconnected %d existing HubSpot grant(s) for the new required "
            "'forms' scope (business-intelligence, marketing-email, and "
            "marketing.campaigns.read are requested as optional and do not "
            "require reconnection on their own). Affected users must "
            "reconnect the HubSpot connector.",
            result.rowcount,
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
