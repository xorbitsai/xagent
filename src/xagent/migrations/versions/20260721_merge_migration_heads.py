"""merge parallel 2026-07-20/21 migration heads

Three migrations merged in parallel PRs (#933, #939 and the docs/slides/
hubspot seed) all branched off 20260715_add_public_mcp_app_audits, leaving
alembic with three heads and making every ``alembic upgrade`` (and
``init_db``) fail with "Multiple heads are present". This no-op merge
revision joins them back into a single head.

Revision ID: 20260721_merge_migration_heads
Revises: 20260720_add_workforce_run_is_preview, 20260720_seed_docs_slides_hubspot, 20260721_drop_workforce_manager_instr
Create Date: 2026-07-21

"""

from typing import Sequence, Union

revision: str = "20260721_merge_migration_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20260720_add_workforce_run_is_preview",
    "20260720_seed_docs_slides_hubspot",
    "20260721_drop_workforce_manager_instr",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
