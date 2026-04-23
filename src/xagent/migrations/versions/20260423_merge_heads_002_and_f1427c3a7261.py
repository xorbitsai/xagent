"""Merge heads: 002_update_agent_execution_mode and f1427c3a7261

Revision ID: 20260423_merge_heads_002_f1427c3a7261
Revises: 002_update_agent_execution_mode, f1427c3a7261
Create Date: 2026-04-23

"""

from typing import Sequence, Union

revision: str = "20260423_merge_heads_002_f1427c3a7261"
down_revision: Union[str, None] = ("002_update_agent_execution_mode", "f1427c3a7261")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
