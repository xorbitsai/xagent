"""Fix agent status case sensitivity

Revision ID: fix_agent_status_case
Revises: 9800a4c3abe5
Create Date: 2025-02-08

This migration fixes uppercase status values (PUBLISHED, DRAFT, ARCHIVED)
to lowercase (published, draft, archived) to match the enum definition.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "fix_agent_status_case"
down_revision = "9800a4c3abe5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Update uppercase status values to lowercase
    op.execute("""
        UPDATE agents
        SET status = LOWER(status::text)::agentstatus
        WHERE status::text IN ('PUBLISHED', 'DRAFT', 'ARCHIVED')
    """)


def downgrade() -> None:
    # Revert to uppercase (not recommended)
    op.execute("""
        UPDATE agents
        SET status = UPPER(status)
        WHERE status IN ('published', 'draft', 'archived')
    """)
