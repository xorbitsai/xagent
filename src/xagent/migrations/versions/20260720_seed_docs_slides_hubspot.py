"""seed built-in Google Docs, Google Slides, and HubSpot CRM MCP connectors

Revision ID: 20260720_seed_docs_slides_hubspot
Revises: 20260715_add_public_mcp_app_audits
Create Date: 2026-07-20 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260720_seed_docs_slides_hubspot"
down_revision: Union[str, None] = "20260715_add_public_mcp_app_audits"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FULL_OAUTH_PROVIDERS_TABLE = sa.table(
    "oauth_providers",
    sa.column("provider_name", sa.String),
    sa.column("name", sa.String),
    sa.column("client_id", sa.String),
    sa.column("client_secret", sa.String),
    sa.column("auth_url", sa.String),
    sa.column("token_url", sa.String),
    sa.column("redirect_uri", sa.String),
    sa.column("userinfo_url", sa.String),
    sa.column("user_id_path", sa.String),
    sa.column("email_path", sa.String),
    sa.column("default_scopes", sa.JSON),
)

OAUTH_PROVIDERS_TABLE = sa.table(
    "oauth_providers",
    sa.column("provider_name", sa.String),
)

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

NEW_APP_IDS = ("google-docs", "google-slides", "hubspot")


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _hubspot_provider_row() -> dict[str, object]:
    return {
        "provider_name": "hubspot",
        "name": "HubSpot",
        "client_id": os.environ.get("HUBSPOT_CLIENT_ID", ""),
        "client_secret": os.environ.get("HUBSPOT_CLIENT_SECRET", ""),
        "auth_url": "https://app.hubspot.com/oauth/authorize",
        "token_url": "https://api.hubapi.com/oauth/v1/token",
        "redirect_uri": os.environ.get("HUBSPOT_REDIRECT_URI", ""),
        "userinfo_url": "https://api.hubapi.com/oauth/v1/access-tokens/{{access_token}}",
        "user_id_path": "user_id",
        "email_path": "user",
        "default_scopes": ["oauth"],
    }


def _new_app_rows() -> list[dict[str, object]]:
    return [
        {
            "app_id": "google-docs",
            "name": "Google Docs",
            "description": "Connect to Google Docs to create documents from Markdown or text, read document content, and edit text.",
            "icon": "https://www.google.com/s2/favicons?domain=docs.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Productivity",
            "oauth_scopes": [
                "https://www.googleapis.com/auth/documents",
                "https://www.googleapis.com/auth/drive.file",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_docs"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "google-slides",
            "name": "Google Slides",
            "description": "Connect to Google Slides to create presentations, add slides, and read slide content.",
            "icon": "https://www.google.com/s2/favicons?domain=slides.google.com&sz=128",
            "transport": "oauth",
            "provider_name": "google",
            "category": "Productivity",
            "oauth_scopes": [
                "https://www.googleapis.com/auth/presentations",
                "https://www.googleapis.com/auth/drive.file",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.google_slides"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "hubspot",
            "name": "HubSpot",
            "description": "Connect to HubSpot CRM to search, create, and update contacts and companies, read deals, and log notes.",
            "icon": "https://www.google.com/s2/favicons?domain=hubspot.com&sz=128",
            "transport": "oauth",
            "provider_name": "hubspot",
            "category": "CRM",
            "oauth_scopes": [
                "crm.objects.contacts.read",
                "crm.objects.contacts.write",
                "crm.objects.companies.read",
                "crm.objects.companies.write",
                "crm.objects.deals.read",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.hubspot"],
                "env_mapping": {"HUBSPOT_ACCESS_TOKEN": "access_token"},
            },
        },
    ]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "oauth_providers" in existing_tables:
        oauth_columns = {
            column["name"] for column in inspector.get_columns("oauth_providers")
        }
        existing_provider_names = set(
            bind.execute(sa.select(OAUTH_PROVIDERS_TABLE.c.provider_name)).scalars()
        )
        if "hubspot" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_hubspot_provider_row(), oauth_columns)],
            )

    if "public_mcp_apps" in existing_tables:
        app_columns = {
            column["name"] for column in inspector.get_columns("public_mcp_apps")
        }
        existing_app_ids = set(
            bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars()
        )
        app_rows_to_insert = [
            _filter_row(row, app_columns)
            for row in _new_app_rows()
            if row["app_id"] not in existing_app_ids
        ]
        if app_rows_to_insert:
            bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), app_rows_to_insert)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "public_mcp_apps" in existing_tables:
        # Only the catalog entries are removed. Any user OAuth connections created
        # against these apps are not owned by this migration and are cleaned up
        # through the normal disconnect path.
        bind.execute(
            sa.delete(PUBLIC_MCP_APPS_TABLE).where(
                PUBLIC_MCP_APPS_TABLE.c.app_id.in_(NEW_APP_IDS)
            )
        )

    if "oauth_providers" not in existing_tables:
        return

    if "public_mcp_apps" in existing_tables:
        remaining_hubspot_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "hubspot")
        ).scalar()
        if remaining_hubspot_apps:
            return

    # Only delete the provider row when it still matches the static shape this
    # migration seeded, so an admin-created "hubspot" provider (via
    # POST /admin/mcp/providers) is preserved. client_id/client_secret are
    # env-dependent and intentionally not part of the guard.
    provider_columns = {
        column["name"] for column in inspector.get_columns("oauth_providers")
    }
    seeded_provider = _hubspot_provider_row()
    delete_stmt = sa.delete(FULL_OAUTH_PROVIDERS_TABLE).where(
        FULL_OAUTH_PROVIDERS_TABLE.c.provider_name == "hubspot"
    )
    for column in ("name", "auth_url", "token_url"):
        if column in provider_columns:
            delete_stmt = delete_stmt.where(
                FULL_OAUTH_PROVIDERS_TABLE.c[column] == seeded_provider[column]
            )
    bind.execute(delete_stmt)
