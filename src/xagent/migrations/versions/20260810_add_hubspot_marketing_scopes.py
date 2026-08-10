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
    "business-intelligence",
    "marketing-email",
    "marketing.campaigns.read",
]

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


def _set_hubspot_fields(
    bind: sa.engine.Connection, scopes: list[str], description: str
) -> None:
    """Keep the persisted row in sync with the code registry's canonical values.

    ``oauth_scopes`` does NOT drive what scope is actually requested at
    OAuth-authorize time: for a builtin app, _app_to_dict sources oauth_scopes
    from get_builtin_execution_fields (the code registry), never from this DB
    row. That write exists solely so validate_builtin_public_mcp_apps doesn't
    report drift between the registry and an already-seeded row.

    ``description`` is different: _app_to_dict reads it straight from this DB
    row (it is not a code-registry execution field), so without this write an
    already-seeded install would keep showing the old connector description
    forever.
    """
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    if not {"app_id", "oauth_scopes", "description"}.issubset(columns):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=scopes, description=description)
    )


def upgrade() -> None:
    _set_hubspot_fields(op.get_bind(), CURRENT_SCOPES, CURRENT_DESCRIPTION)


def downgrade() -> None:
    _set_hubspot_fields(op.get_bind(), PREVIOUS_SCOPES, PREVIOUS_DESCRIPTION)
