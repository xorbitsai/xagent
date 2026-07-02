"""backfill mcp server ownership flags

Revision ID: ac3599eb3f1c
Revises: 5bb3df522a7d
Create Date: 2026-07-02

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ac3599eb3f1c"
down_revision: Union[str, None] = "5bb3df522a7d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Backfill is_owner/can_edit/can_delete for pre-existing user_mcpservers rows.

    `update_mcp_server` now enforces edit permission using these flags, but
    rows created between the 222f2073c886 migration (which introduced the
    columns) and the introduction of that enforcement were never backfilled
    and default to false. This leaves every non-admin user locked out of
    editing MCP servers they created before this deploy.

    Only rows where ALL THREE flags are still false/null (i.e. untouched by
    the 222f2073c886 backfill, which set all three to True for migrated
    legacy servers, and not created by the fixed `create_mcp_server`, which
    also sets all three to True) are updated. Any row where at least one
    flag is already truthy is left untouched, so deliberate narrower grants
    are preserved.
    """
    from sqlalchemy.engine.reflection import Inspector

    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    if "user_mcpservers" not in inspector.get_table_names():
        # Table doesn't exist yet in this database (e.g. a synthetic test DB
        # stamped past the migration that creates it, or a from-scratch
        # replay that hasn't reached create_all() yet) -- nothing to backfill.
        return

    is_postgres = bind.dialect.name == "postgresql"
    true_literal = "true" if is_postgres else "1"
    false_literal = "false" if is_postgres else "0"

    bind.execute(
        sa.text(
            f"""
            UPDATE user_mcpservers
            SET is_owner = {true_literal},
                can_edit = {true_literal},
                can_delete = {true_literal}
            WHERE COALESCE(is_owner, {false_literal}) = {false_literal}
              AND COALESCE(can_edit, {false_literal}) = {false_literal}
              AND COALESCE(can_delete, {false_literal}) = {false_literal}
            """
        )
    )


def downgrade() -> None:
    """No-op: this is a data backfill and is intentionally irreversible.

    Reverting would re-break access for existing editors who relied on the
    pre-existing (unenforced) behavior that this migration restores. There
    is also no way to distinguish rows backfilled by this migration from
    rows that legitimately already had all three flags set to True.
    """
    pass
