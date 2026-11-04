"""fix_agents_created_at_default

Revision ID: aaca07b20ea9
Revises: f79da474c69d
Create Date: 2026-02-01 01:11:12.645015

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "aaca07b20ea9"
down_revision: Union[str, None] = "f79da474c69d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context
    from sqlalchemy.engine.reflection import Inspector

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if index already exists
    existing_indexes = [idx["name"] for idx in inspector.get_indexes("agents")]
    if "ix_agents_id" not in existing_indexes:
        # Index doesn't exist, create it
        try:
            op.create_index(op.f("ix_agents_id"), "agents", ["id"], unique=False)
        except Exception:
            pass  # Ignore errors, might be a race condition


def downgrade() -> None:
    from alembic import context
    from sqlalchemy.engine.reflection import Inspector

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if index exists before dropping
    existing_indexes = [idx["name"] for idx in inspector.get_indexes("agents")]
    if "ix_agents_id" in existing_indexes:
        try:
            op.drop_index(op.f("ix_agents_id"), table_name="agents")
        except Exception:
            pass  # Ignore errors
