"""seed built-in Meta connector registry data

Revision ID: 20260627_seed_meta_connectors
Revises: 20260624_add_mcp_concurrency_config
Create Date: 2026-06-27 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.migrations.seed_helpers import (
    OAUTH_PROVIDER_SEED_MATCH_COLUMNS,
    delete_unmodified_seeded_rows,
)

# revision identifiers, used by Alembic.
revision: str = "20260627_seed_meta_connectors"
down_revision: Union[str, None] = "20260624_add_mcp_concurrency_config"
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


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _meta_provider_row() -> dict[str, object]:
    return {
        "provider_name": "meta",
        "name": "Meta",
        "client_id": os.environ.get("META_CLIENT_ID", ""),
        "client_secret": os.environ.get("META_CLIENT_SECRET", ""),
        "auth_url": "https://www.facebook.com/v25.0/dialog/oauth",
        "token_url": "https://graph.facebook.com/v25.0/oauth/access_token",
        "redirect_uri": os.environ.get("META_REDIRECT_URI", ""),
        "userinfo_url": "https://graph.facebook.com/v25.0/me?fields=id,email",
        "user_id_path": "id",
        "email_path": "email",
        "default_scopes": ["public_profile"],
    }


def _meta_app_rows() -> list[dict[str, object]]:
    # launch_config.command is "python" here (not the original "uv") to match
    # what 20260715_normalize_builtin_mcp_launch.py's irreversible normalization
    # leaves behind - see that migration's CANONICAL_EXECUTION_FIELDS comment.
    # This narrows a downgrade-cleanup window in the other direction: a DB
    # seeded by this migration but downgraded *before* 20260715 ever ran will
    # now look "operator-modified" (command still "uv") and be preserved
    # instead of removed. That fails safe (no data loss, just a stale row)
    # and is a narrow, non-standard ordering to hit in practice.
    return [
        {
            "app_id": "facebook",
            "name": "Facebook Pages",
            "description": "Connect to Facebook Pages to discover managed pages and publish page posts.",
            "icon": "https://www.google.com/s2/favicons?domain=facebook.com&sz=128",
            "transport": "oauth",
            "provider_name": "meta",
            "category": "Marketing",
            "oauth_scopes": [
                "pages_show_list",
                "pages_read_engagement",
                "pages_manage_posts",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.facebook"],
                "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
            },
        },
        {
            "app_id": "instagram",
            "name": "Instagram",
            "description": "Connect to Instagram professional accounts through Meta Graph API.",
            "icon": "https://www.google.com/s2/favicons?domain=instagram.com&sz=128",
            "transport": "oauth",
            "provider_name": "meta",
            "category": "Marketing",
            "oauth_scopes": [
                "pages_show_list",
                "pages_read_engagement",
                "instagram_basic",
                "instagram_content_publish",
            ],
            "is_visible_in_connector": True,
            "launch_config": {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.instagram"],
                "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
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
        if "meta" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_meta_provider_row(), oauth_columns)],
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
            for row in _meta_app_rows()
            if row["app_id"] not in existing_app_ids
        ]
        if app_rows_to_insert:
            bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), app_rows_to_insert)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "public_mcp_apps" in existing_tables:
        # Only rows still matching this migration's seed snapshot are removed,
        # so an operator's pre-existing custom "facebook"/"instagram" app_id is
        # left in place.
        delete_unmodified_seeded_rows(bind, PUBLIC_MCP_APPS_TABLE, _meta_app_rows())

    if "oauth_providers" not in existing_tables:
        return

    if "public_mcp_apps" in existing_tables:
        remaining_meta_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "meta")
        ).scalar_one()
        if remaining_meta_apps:
            return

    # Only delete the provider row when it still matches the static shape this
    # migration seeded, so an admin-created/edited "meta" provider (via
    # POST /admin/mcp/providers) is preserved. client_id/client_secret/
    # redirect_uri are env-dependent and intentionally not part of the guard.
    delete_unmodified_seeded_rows(
        bind,
        FULL_OAUTH_PROVIDERS_TABLE,
        [_meta_provider_row()],
        match_columns=OAUTH_PROVIDER_SEED_MATCH_COLUMNS,
        id_column="provider_name",
    )
