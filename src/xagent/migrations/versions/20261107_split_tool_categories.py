"""Replace old tool groups in agents and persisted task configurations.

Revision ID: 20261107_split_tool_categories
Revises: 20261106_hash_user_refresh_tokens
"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261107_split_tool_categories"
down_revision: Union[str, None] = "20261106_hash_user_refresh_tokens"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

BATCH_SIZE = 500

# Frozen here so future runtime category changes cannot change this migration.
_REPLACEMENTS = {
    "basic": ("basic_tools", "code_execution", "web_tools"),
    "web_search": ("web_tools",),
}


def _categories(value: Any) -> Any:
    if not isinstance(value, list) or not any(
        isinstance(item, str) and item in _REPLACEMENTS for item in value
    ):
        return value
    result: list[Any] = []
    for item in value:
        replacements = (
            _REPLACEMENTS.get(item, (item,)) if isinstance(item, str) else (item,)
        )
        for category in replacements:
            if category not in result:
                result.append(category)
    return result


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    for table_name, column_name in (
        ("agents", "tool_categories"),
        ("tasks", "agent_config"),
    ):
        if table_name not in tables or column_name not in {
            column["name"] for column in inspector.get_columns(table_name)
        }:
            continue
        table = sa.table(
            table_name,
            sa.column("id", sa.Integer),
            sa.column(column_name, sa.JSON),
        )
        column = table.c[column_name]
        last_id = None
        while True:
            query = sa.select(table.c.id, column).order_by(table.c.id).limit(BATCH_SIZE)
            if last_id is not None:
                query = query.where(table.c.id > last_id)
            rows = bind.execute(query).all()
            if not rows:
                break
            for row_id, value in rows:
                if column_name == "tool_categories":
                    updated = _categories(value)
                elif isinstance(value, dict) and "tool_categories" in value:
                    # Only this configuration field is a tool grant. Leave
                    # prompt context, input payloads and audit snapshots intact.
                    updated = {
                        **value,
                        "tool_categories": _categories(value["tool_categories"]),
                    }
                else:
                    continue
                if updated != value:
                    bind.execute(
                        table.update()
                        .where(table.c.id == row_id)
                        .values({column_name: updated})
                    )
            last_id = rows[-1][0]


def downgrade() -> None:
    # The merged network group cannot be split back unambiguously. Reversing
    # it would also overwrite categories deliberately chosen after upgrade.
    pass
