"""merge two alembic heads on main

20260703_add_multi_key_support_to_agent_api_keys (multi API keys per
agent) and 20260706_add_agent_widget_key (per-agent widget key) both
branch from 20260705_add_user_mcpserver_env_source, leaving the
migration graph with two heads. This is a no-op merge revision that
reunites them into a single head.

Revision ID: 20260707_merge_alembic_heads
Revises: 20260703_add_multi_key_support_to_agent_api_keys, 20260706_add_agent_widget_key
Create Date: 2026-07-07 00:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "20260707_merge_alembic_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20260703_add_multi_key_support_to_agent_api_keys",
    "20260706_add_agent_widget_key",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
