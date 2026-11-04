"""forward the Zendesk Marketplace settings to the Zendesk connector

Revision ID: 20261019_zendesk_marketplace
Revises: 20261017_task_cmd_dispatch_class
Create Date: 2026-10-19

Zendesk Marketplace integration apps must send identifying headers on every
API request (#3222). The connector subprocess only receives the environment
variables its catalog row's launch_config names, so the Xagent-owned Zendesk
row gains a static_env entry per setting. The builtin registry seed only
inserts missing rows, so an existing row needs this update.
"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import builtin_provenance_identity

revision: str = "20261019_zendesk_marketplace"
down_revision: Union[str, None] = "20261017_task_cmd_dispatch_class"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ID = "zendesk"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}
# Subprocess variable -> host variable, the same names on both sides.
MARKETPLACE_STATIC_ENV = {
    name: name
    for name in (
        "ZENDESK_MARKETPLACE_NAME",
        "ZENDESK_MARKETPLACE_ORGANIZATION_ID",
        "ZENDESK_MARKETPLACE_APP_ID",
    )
}

PUBLIC_MCP_APPS = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("launch_config", sa.JSON),
)


def _owned_launch_config(bind: sa.engine.Connection) -> dict[str, Any] | None:
    """The Zendesk row's launch_config, if Xagent's registry owns it."""
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return None
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if not {"app_id", "launch_config"}.issubset(columns):
        return None
    launch_config = bind.execute(
        sa.select(PUBLIC_MCP_APPS.c.launch_config).where(
            PUBLIC_MCP_APPS.c.app_id == APP_ID
        )
    ).scalar_one_or_none()
    if not isinstance(launch_config, dict):
        return None
    # A legacy API-token row or one an operator repointed is left alone.
    if builtin_provenance_identity(
        launch_config.get("builtin_provenance")
    ) != builtin_provenance_identity(BUILTIN_PROVENANCE):
        return None
    return launch_config


def _static_env(launch_config: dict[str, Any]) -> dict[str, Any]:
    static_env = launch_config.get("static_env")
    return dict(static_env) if isinstance(static_env, dict) else {}


def _write(bind: sa.engine.Connection, launch_config: dict[str, Any]) -> None:
    bind.execute(
        sa.update(PUBLIC_MCP_APPS)
        .where(PUBLIC_MCP_APPS.c.app_id == APP_ID)
        .values(launch_config=launch_config)
    )


def upgrade() -> None:
    bind = op.get_bind()
    launch_config = _owned_launch_config(bind)
    if launch_config is None:
        return
    static_env = {**_static_env(launch_config), **MARKETPLACE_STATIC_ENV}
    if launch_config.get("static_env") == static_env:
        return
    _write(bind, {**launch_config, "static_env": static_env})


def downgrade() -> None:
    bind = op.get_bind()
    launch_config = _owned_launch_config(bind)
    if launch_config is None or not isinstance(launch_config.get("static_env"), dict):
        return
    static_env = {
        name: host_name
        for name, host_name in launch_config["static_env"].items()
        if name not in MARKETPLACE_STATIC_ENV
    }
    restored = {
        key: value for key, value in launch_config.items() if key != "static_env"
    }
    if static_env:
        restored["static_env"] = static_env
    _write(bind, restored)
