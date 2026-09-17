"""seed built-in Employment Hero (OAuth) MCP connector

Revision ID: 20260826_seed_employment_hero_mcp_app
Revises: 20260903_model_management
Create Date: 2026-08-26 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260826_seed_employment_hero_mcp_app"
down_revision: Union[str, None] = "20260903_model_management"
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

APP_ID = "employment-hero"

EMPLOYMENT_HERO_SCOPES = [
    "organisations:list",
    "employees:list",
    "employees:show",
    "teams:list",
    "timesheet_entries:list",
]


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _employment_hero_provider_row() -> dict[str, object]:
    return {
        "provider_name": "employment-hero",
        "name": "Employment Hero",
        "client_id": os.environ.get("EMPLOYMENT_HERO_CLIENT_ID", ""),
        "client_secret": os.environ.get("EMPLOYMENT_HERO_CLIENT_SECRET", ""),
        "auth_url": "https://oauth.employmenthero.com/oauth2/authorize",
        "token_url": "https://oauth.employmenthero.com/oauth2/token",
        "redirect_uri": os.environ.get("EMPLOYMENT_HERO_REDIRECT_URI", ""),
        "userinfo_url": "",
        "user_id_path": "id",
        "email_path": "email",
        "default_scopes": [],
    }


def _employment_hero_app_row() -> dict[str, object]:
    return {
        "app_id": APP_ID,
        "name": "Employment Hero",
        "description": "Connect to Employment Hero to look up organisations, employees, teams, and timesheet entries.",
        "icon": "https://www.google.com/s2/favicons?domain=employmenthero.com&sz=128",
        "transport": "oauth",
        "provider_name": "employment-hero",
        "category": "HR",
        "oauth_scopes": EMPLOYMENT_HERO_SCOPES,
        "is_visible_in_connector": True,
        "launch_config": {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.employment_hero"],
            "env_mapping": {"EMPLOYMENT_HERO_ACCESS_TOKEN": "access_token"},
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
        if "employment-hero" not in existing_provider_names:
            bind.execute(
                sa.insert(FULL_OAUTH_PROVIDERS_TABLE),
                [_filter_row(_employment_hero_provider_row(), oauth_columns)],
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
                [_filter_row(_employment_hero_app_row(), app_columns)],
            )


def _row_matches_seeded_shape(
    row: sa.engine.Row, seeded: dict[str, object], compare_columns: set[str]
) -> bool:
    """Compare a fetched row against the seeded row dict in Python.

    Deliberately not pushed into the SQL WHERE clause: PostgreSQL's plain
    ``json`` column type (what oauth_scopes/launch_config/default_scopes
    actually are -- see the models, no ``.with_variant(JSONB(), ...)``
    escape hatch here) has no ``=`` operator at all, so
    ``.where(json_column == python_value)`` compiles fine but raises
    ``UndefinedFunction: operator does not exist: json = json`` at
    execute time on Postgres. Comparing in Python after a plain SELECT
    sidesteps that entirely and works identically on every backend.
    """
    return all(row._mapping[column] == seeded[column] for column in compare_columns)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "public_mcp_apps" in existing_tables:
        # Only delete the catalog entry when it still matches the FULL
        # static shape this migration seeded -- an unconditional
        # delete-by-app_id would remove a pre-existing operator row that
        # happened to already occupy app_id "employment-hero" before this
        # migration ever ran (upgrade()'s own `if APP_ID not in
        # existing_app_ids` check would have skipped inserting over it, so
        # upgrade and downgrade must agree on what "this migration's row"
        # means). Every one of this row's columns is compared, not just a
        # structural few, since description/is_visible_in_connector are
        # freely PATCHable today and a raw DB edit (or this app_id later
        # being dropped from the built-in registry while the row persists)
        # could diverge any of them. In Python, see
        # _row_matches_seeded_shape's docstring for why not in SQL.
        app_row = bind.execute(
            sa.select(PUBLIC_MCP_APPS_TABLE).where(
                PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
            )
        ).first()
        if app_row is not None and _row_matches_seeded_shape(
            app_row,
            _employment_hero_app_row(),
            {
                "name",
                "description",
                "icon",
                "transport",
                "provider_name",
                "category",
                "oauth_scopes",
                "is_visible_in_connector",
                "launch_config",
            },
        ):
            bind.execute(
                sa.delete(PUBLIC_MCP_APPS_TABLE).where(
                    PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
                )
            )

    if "oauth_providers" not in existing_tables:
        return

    if "public_mcp_apps" in existing_tables:
        remaining_employment_hero_apps = bind.execute(
            sa.select(sa.func.count())
            .select_from(PUBLIC_MCP_APPS_TABLE)
            .where(PUBLIC_MCP_APPS_TABLE.c.provider_name == "employment-hero")
        ).scalar()
        if remaining_employment_hero_apps:
            return

    # Only delete the provider row when it still matches the FULL static
    # shape this migration seeded, so an admin-created or admin-edited
    # "employment-hero" provider (via POST/PUT /admin/mcp/providers) is
    # preserved. client_id/client_secret/redirect_uri are env-dependent and
    # intentionally excluded from the guard; every other column is static
    # and matched, not just name/auth_url/token_url -- an admin who edited
    # userinfo_url/user_id_path/email_path/default_scopes without touching
    # those few fields would otherwise still match and get silently
    # deleted.
    provider_row = bind.execute(
        sa.select(FULL_OAUTH_PROVIDERS_TABLE).where(
            FULL_OAUTH_PROVIDERS_TABLE.c.provider_name == "employment-hero"
        )
    ).first()
    if provider_row is not None and _row_matches_seeded_shape(
        provider_row,
        _employment_hero_provider_row(),
        {
            "name",
            "auth_url",
            "token_url",
            "userinfo_url",
            "user_id_path",
            "email_path",
            "default_scopes",
        },
    ):
        bind.execute(
            sa.delete(FULL_OAUTH_PROVIDERS_TABLE).where(
                FULL_OAUTH_PROVIDERS_TABLE.c.provider_name == "employment-hero"
            )
        )
