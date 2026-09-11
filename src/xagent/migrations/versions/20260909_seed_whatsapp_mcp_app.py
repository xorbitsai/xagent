"""seed built-in WhatsApp Business (Meta OAuth) MCP connector

Revision ID: 20260909_seed_whatsapp_mcp_app
Revises: 370740d9125a
Create Date: 2026-09-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260909_seed_whatsapp_mcp_app"
down_revision: Union[str, None] = "370740d9125a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
    sa.column("icon", sa.String),
    sa.column("transport", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("category", sa.String),
    sa.column("oauth_scopes", sa.JSON),
    sa.column("is_visible_in_connector", sa.Boolean),
    sa.column("launch_config", sa.JSON),
)

APP_ID = "whatsapp"

ROW = {
    "app_id": APP_ID,
    "name": "WhatsApp Business",
    "description": "Connect to the WhatsApp Business Platform to discover business accounts and phone numbers, browse message templates, and send text, template, and media messages to customers.",
    "icon": "https://www.google.com/s2/favicons?domain=whatsapp.com&sz=128",
    "transport": "oauth",
    "provider_name": "meta",
    "category": "Communication",
    "oauth_scopes": [
        "business_management",
        "whatsapp_business_management",
        "whatsapp_business_messaging",
    ],
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.whatsapp"],
        "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
    },
}

# Every non-env-dependent column of ROW; see downgrade() for why all of them
# (not just a structural few) take part in the "is this our row" match.
APP_ROW_GUARD_COLUMNS = {
    "name",
    "description",
    "icon",
    "transport",
    "provider_name",
    "category",
    "oauth_scopes",
    "is_visible_in_connector",
    "launch_config",
}

# The "meta" oauth_providers row is NOT seeded here, unlike a migration that
# introduces a brand-new provider (salesforce/deputy/myob): it already exists
# (20260627_seed_meta_connectors seeded it for Facebook Pages/Instagram, and
# this connector reuses the same provider for OAuth), and every other
# public_mcp_apps row that adds an app to an already-seeded provider
# (20260703_google_maps, 20260724_google_ads, 20260730_google_analytics,
# 20260806_google_sheets, 20260824_google_search_console, and the sibling
# meta-ads connector) leaves oauth_providers alone rather than re-seeding it
# defensively. Following that precedent instead of adding a second, drifting
# copy of the frozen provider row here.


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _row_matches_seeded_shape(
    row: sa.engine.Row, seeded: dict[str, object], compare_columns: set[str]
) -> bool:
    """Compare a fetched row against the seeded row dict in Python.

    Deliberately not pushed into the SQL WHERE clause: PostgreSQL's plain
    ``json`` column type (what oauth_scopes/launch_config actually are) has
    no ``=`` operator, so ``.where(json_column == python_value)`` compiles
    fine but raises ``UndefinedFunction: operator does not exist: json =
    json`` at execute time on Postgres. Comparing in Python after a plain
    SELECT works identically on every backend.
    """
    return all(row._mapping[column] == seeded[column] for column in compare_columns)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    app_columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    existing_app_ids = set(
        bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars()
    )
    # A pre-existing row with this app_id (e.g. hand-created by an operator
    # before this migration deployed) is left exactly as it is; the builtin
    # registry overlays the canonical execution fields onto any row sharing
    # a builtin app_id at read time anyway.
    if APP_ID not in existing_app_ids:
        bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), [_filter_row(ROW, app_columns)])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    # Only the catalog entry is removed. The meta oauth_providers row is
    # shared with the Facebook Pages and Instagram connectors and is owned by
    # 20260627_seed_meta_connectors, so this migration never touches it. Any
    # MCPServer/UserMCPServer rows created by users who already connected are
    # intentionally left in place -- connect-driven rows are not owned by
    # this migration and are cleaned up through the normal disconnect path.
    #
    # Only delete the catalog entry when it still matches the FULL static
    # shape this migration seeded -- an unconditional delete-by-app_id would
    # remove a pre-existing operator row that happened to already occupy
    # app_id "whatsapp" before this migration ever ran (upgrade()'s own
    # `if APP_ID not in existing_app_ids` check would have skipped inserting
    # over it, so upgrade and downgrade must agree on what "this migration's
    # row" means). Matching only a structural few (name/transport/
    # provider_name) isn't enough: description/is_visible_in_connector are
    # freely PATCHable by admins today, and a raw DB edit could diverge any
    # column -- so every non-env-dependent column is compared. A row that
    # doesn't match is left in place: restored rather than destroyed. Same
    # guard as the salesforce seed migration.
    #
    # A reduced-schema table missing one of the guard columns (mid-chain,
    # before the column-adding migration ran) would otherwise make the SELECT
    # reference a nonexistent column and raise; no-op instead rather than
    # fall back to a weaker match on whatever columns remain.
    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    if not APP_ROW_GUARD_COLUMNS.issubset(columns):
        return

    app_row = bind.execute(
        sa.select(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    ).first()
    if app_row is not None and _row_matches_seeded_shape(
        app_row, ROW, APP_ROW_GUARD_COLUMNS
    ):
        bind.execute(
            sa.delete(PUBLIC_MCP_APPS_TABLE).where(
                PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
            )
        )
