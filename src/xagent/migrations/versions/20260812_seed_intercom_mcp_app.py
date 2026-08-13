"""seed built-in Intercom (OAuth) MCP connector

Revision ID: 20260812_seed_intercom_mcp_app
Revises: 20260812_add_slack_history_reactions_files_scopes
Create Date: 2026-08-12 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260812_seed_intercom_mcp_app"
down_revision: Union[str, None] = "20260812_add_slack_history_reactions_files_scopes"
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

APP_ID = "intercom"


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _intercom_provider_row() -> dict[str, object]:
    return {
        "provider_name": "intercom",
        "name": "Intercom",
        "client_id": os.environ.get("INTERCOM_CLIENT_ID", ""),
        "client_secret": os.environ.get("INTERCOM_CLIENT_SECRET", ""),
        # Single global authorize host: per Intercom's community answer
        # (community.intercom.com/api-webhooks-23/do-we-need-to-create
        # -multiple-oauth-apps-per-data-region-2522), a public app is
        # replicated across US/EU/AU automatically once it clears Intercom
        # App Review. See builtin_mcp_registry.py's intercom provider row for
        # the full citation and the caveat that this is a community answer,
        # not primary docs, unverified against a live non-US workspace.
        "auth_url": "https://app.intercom.com/oauth",
        # No region prefix: api.intercom.io auto-routes to the workspace's
        # actual hosting region (US/EU/AU) -- this part is documented in
        # Intercom's primary REST API reference, unlike Intercom's hosted MCP
        # server (mcp.intercom.com / mcp.eu.intercom.com) which has separate
        # per-region endpoints and does not support AU workspaces at all.
        "token_url": "https://api.intercom.io/auth/eagle/token",
        "redirect_uri": os.environ.get("INTERCOM_REDIRECT_URI", ""),
        "userinfo_url": "https://api.intercom.io/me",
        "user_id_path": "id",
        "email_path": "email",
        # Intercom has no `scope` authorize-URL param; granted permissions
        # come entirely from the app's Authentication settings in the
        # Developer Hub, so there is nothing to list here.
        "default_scopes": [],
    }


def _intercom_app_row() -> dict[str, object]:
    return {
        "app_id": APP_ID,
        "name": "Intercom",
        "description": "Connect to Intercom to search contacts, review conversations, and reply to customers.",
        "icon": "https://www.google.com/s2/favicons?domain=intercom.com&sz=128",
        "transport": "oauth",
        "provider_name": "intercom",
        "category": "Support",
        "oauth_scopes": [],
        # Hidden until manually verified against a live workspace -- see
        # builtin_mcp_registry.py's intercom app row for the full rationale
        # (this ships customer-facing write tools: reply/note/close).
        "is_visible_in_connector": False,
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.intercom"],
            "env_mapping": {"INTERCOM_ACCESS_TOKEN": "access_token"},
        },
    }


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "oauth_providers" in existing_tables:
        oauth_columns = {
            column["name"] for column in inspector.get_columns("oauth_providers")
        }
        existing_provider_names = set(
            bind.execute(
                sa.select(FULL_OAUTH_PROVIDERS_TABLE.c.provider_name)
            ).scalars()
        )
        if "intercom" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_intercom_provider_row(), oauth_columns)],
            )

    if "public_mcp_apps" in existing_tables:
        app_columns = {
            column["name"] for column in inspector.get_columns("public_mcp_apps")
        }
        existing_app_ids = set(
            bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars()
        )
        if APP_ID not in existing_app_ids:
            bind.execute(
                sa.insert(PUBLIC_MCP_APPS_TABLE),
                [_filter_row(_intercom_app_row(), app_columns)],
            )
        else:
            # A row with this app_id already exists (e.g. hand-created by an
            # operator before this migration deployed). Such a row keeps its
            # own is_visible_in_connector, which defaults to TRUE for
            # hand-created rows -- and the builtin registry overlays the real
            # transport/launch_config onto ANY row sharing this app_id at read
            # time, so a visible pre-existing row would silently become a
            # working, one-click-connectable, unverified write-capable
            # connector, defeating the hidden-rollout gate with no further
            # action. Same guard, same reasoning, as the chrome seed
            # migration's collision branch (#1143).
            bind.execute(
                sa.update(PUBLIC_MCP_APPS_TABLE)
                .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
                .values(is_visible_in_connector=False)
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "public_mcp_apps" in existing_tables:
        # Only the catalog entry is removed. Any user OAuth connections created
        # against this app are not owned by this migration and are cleaned up
        # through the normal disconnect path.
        bind.execute(
            sa.delete(PUBLIC_MCP_APPS_TABLE).where(
                PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
            )
        )

    if "oauth_providers" not in existing_tables:
        return

    if "public_mcp_apps" in existing_tables:
        remaining_intercom_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "intercom")
        ).scalar()
        if remaining_intercom_apps:
            return

    # Only delete the provider row when it still matches the static shape this
    # migration seeded, so an admin-created "intercom" provider (via
    # POST /admin/mcp/providers) is preserved -- same guard as the zoom seed
    # migration, and for the same reason: unlike slack/meta/google-maps (whose
    # downgrades delete unconditionally), a differently-shaped custom row here
    # must survive. client_id/client_secret are env-dependent and
    # intentionally not part of the guard. name/auth_url/token_url are NOT
    # NULL core columns present since the table's creation, so they can be
    # matched unconditionally.
    seeded_provider = _intercom_provider_row()
    bind.execute(
        sa.delete(FULL_OAUTH_PROVIDERS_TABLE)
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.provider_name == "intercom")
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.name == seeded_provider["name"])
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.auth_url == seeded_provider["auth_url"])
        .where(FULL_OAUTH_PROVIDERS_TABLE.c.token_url == seeded_provider["token_url"])
    )
