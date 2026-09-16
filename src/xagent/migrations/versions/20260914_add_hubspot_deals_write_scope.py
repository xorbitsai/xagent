"""add HubSpot crm.objects.deals.write scope

Revision ID: 20260914_add_hubspot_deals_write_scope
Revises: 20260912_shared_task_execution
Create Date: 2026-09-14

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "20260914_add_hubspot_deals_write_scope"
down_revision: Union[str, None] = "20260912_shared_task_execution"
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
    "forms",
]
# Inserted right after deals.read (not appended) to match the registry's
# grouping of the crm.objects.deals.* scopes together - see
# builtin_mcp_registry.py's hubspot entry and
# test_migration_fields_match_registry, which asserts this list equals the
# registry's oauth_scopes verbatim, order included.
CURRENT_SCOPES = [
    "crm.objects.contacts.read",
    "crm.objects.contacts.write",
    "crm.objects.companies.read",
    "crm.objects.companies.write",
    "crm.objects.deals.read",
    "crm.objects.deals.write",
    "forms",
]

PREVIOUS_DESCRIPTION = (
    "Connect to HubSpot CRM and Marketing Hub to search, create, and update "
    "contacts and companies, read deals, log notes, read forms and "
    "submissions, pull traffic analytics reports, and read marketing emails "
    "and campaigns."
)
CURRENT_DESCRIPTION = (
    "Connect to HubSpot CRM and Marketing Hub to list, search, create, and "
    "update contacts, companies, and deals, log notes, read forms and "
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

    See 20260810_add_hubspot_marketing_scopes.py for why this write exists
    (drift avoidance against validate_builtin_public_mcp_apps, not the actual
    OAuth-authorize-time scope source).
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

    Only overwrite when the persisted value still equals the last-known
    canonical description (i.e. it was never customized via the admin PATCH
    endpoint); an edited value matches neither PREVIOUS_DESCRIPTION nor
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
    """Force reconnection so the user has a chance to grant the new scope.

    HubSpot's token-exchange response carries no `scope` field, so a stored
    grant's actual permissions can't be inspected after the fact - every row
    was necessarily authorized before crm.objects.deals.write existed.
    Without this, _oauth_account_can_connect only checks token
    presence/expiry, so an existing connection keeps showing "Connected" and
    deal writes keep failing at call time with a raw HubSpot 401 (the
    connector's read tools work throughout, since those scopes were already
    granted - see the incident this migration fixes: read-only deal access
    with a raw-401 fallback path for writes).

    refresh_token is cleared alongside access_token as defense-in-depth: the
    current token resolver (web/tools/config.py) already short-circuits on a
    falsy access_token before ever reaching refresh_oauth_token_if_needed, so
    a surviving refresh_token cannot resurrect the old-scoped grant through
    that specific path today. Cleared anyway so this migration doesn't rely
    on that resolver's current shape staying exactly as it is.
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
            "Disconnected %d existing HubSpot grant(s) for the new required "
            "'crm.objects.deals.write' scope. Affected users must reconnect "
            "the HubSpot connector to create or update deals.",
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
    # The cleared access tokens are gone for good (that's the point - force a
    # reconnect); there is nothing meaningful to restore for user_oauth here.
    bind = op.get_bind()
    _set_hubspot_scopes(bind, PREVIOUS_SCOPES)
    _set_hubspot_description_if_unchanged(
        bind, CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION
    )
