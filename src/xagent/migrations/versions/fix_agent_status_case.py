"""Fix agent status case sensitivity

Revision ID: fix_agent_status_case
Revises: 9800a4c3abe5
Create Date: 2025-02-08

This migration fixes uppercase status values (PUBLISHED, DRAFT, ARCHIVED)
to lowercase (published, draft, archived) to match the enum definition.
"""

from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision = "fix_agent_status_case"
down_revision = "9800a4c3abe5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    dialect_name = bind.dialect.name

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        return

    # Check if status column exists
    columns = [col["name"] for col in inspector.get_columns("agents")]
    if "status" not in columns:
        return

    # This migration uses PostgreSQL-specific syntax
    if dialect_name == "postgresql":
        # Update uppercase status values to lowercase
        op.execute("""
            UPDATE agents
            SET status = LOWER(status::text)::agentstatus
            WHERE status::text IN ('PUBLISHED', 'DRAFT', 'ARCHIVED')
        """)
    else:
        # SQLite: use LOWER() function instead
        op.execute("""
            UPDATE agents
            SET status = LOWER(status)
            WHERE status IN ('PUBLISHED', 'DRAFT', 'ARCHIVED')
        """)


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    dialect_name = bind.dialect.name

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        return

    # Check if status column exists
    columns = [col["name"] for col in inspector.get_columns("agents")]
    if "status" not in columns:
        return

    # Revert to uppercase (not recommended)
    if dialect_name == "postgresql":
        # PostgreSQL: handle custom enum type
        op.execute("""
            UPDATE agents
            SET status = UPPER(status::text)::agentstatus
            WHERE status::text IN ('published', 'draft', 'archived')
        """)
    else:
        # SQLite: use UPPER() function
        op.execute("""
            UPDATE agents
            SET status = UPPER(status)
            WHERE status IN ('published', 'draft', 'archived')
        """)
