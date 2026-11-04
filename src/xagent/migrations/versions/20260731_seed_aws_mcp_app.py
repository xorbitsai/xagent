"""seed built-in AWS (key-based) MCP connector

Revision ID: 20260731_seed_aws_mcp_app
Revises: 20260728_add_agent_template_id_and_name_uniqueness
Create Date: 2026-07-31 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260731_seed_aws_mcp_app"
down_revision: Union[str, None] = "20260728_add_agent_template_id_and_name_uniqueness"
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

APP_ID = "aws"

ROW = {
    "app_id": APP_ID,
    "name": "AWS",
    "description": "Connect to AWS to check CloudWatch alarms/metrics/logs, DynamoDB health, and SQS queue depth.",
    "icon": "https://www.google.com/s2/favicons?domain=aws.amazon.com&sz=128",
    "transport": "stdio",
    "provider_name": None,
    "category": "Operations",
    "oauth_scopes": None,
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.aws"],
        "required_env": [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
        ],
    },
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return

    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    existing = set(bind.execute(sa.select(PUBLIC_MCP_APPS_TABLE.c.app_id)).scalars())
    if APP_ID in existing:
        return

    row = {k: v for k, v in ROW.items() if k in columns}
    bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), [row])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return
    # Only the catalog entry is removed. AWS has no oauth_providers row (it is
    # key-based). Any MCPServer/UserMCPServer rows created by users who already
    # connected are intentionally left in place — connect-driven rows are not
    # owned by this migration and are cleaned up through the normal disconnect
    # path.
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
