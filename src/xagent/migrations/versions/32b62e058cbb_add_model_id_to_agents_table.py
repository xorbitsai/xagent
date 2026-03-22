"""add model_id to agents table

Revision ID: 32b62e058cbb
Revises: 9800a4c3abe5
Create Date: 2026-01-31 23:17:50.576086

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "32b62e058cbb"
down_revision: Union[str, None] = "9800a4c3abe5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check if column already exists
    existing_columns = [col["name"] for col in inspector.get_columns("agents")]
    if "model_id" in existing_columns:
        return  # Column already exists, skip migration

    dialect_name = bind.dialect.name
    if dialect_name == "sqlite":
        # Use batch mode for SQLite to add column with foreign key
        with op.batch_alter_table("agents", recreate="auto") as batch_op:
            batch_op.add_column(sa.Column("model_id", sa.Integer(), nullable=True))
            batch_op.create_foreign_key(
                "fk_agents_model_id_models", "models", ["model_id"], ["id"]
            )
    else:
        # For PostgreSQL and other databases, use native operations
        op.add_column("agents", sa.Column("model_id", sa.Integer(), nullable=True))
        op.create_foreign_key(
            "fk_agents_model_id_models", "agents", "models", ["model_id"], ["id"]
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check if column exists before dropping
    existing_columns = [col["name"] for col in inspector.get_columns("agents")]
    if "model_id" not in existing_columns:
        return  # Column doesn't exist, skip downgrade

    dialect_name = bind.dialect.name
    if dialect_name == "sqlite":
        # Use batch mode for SQLite to drop foreign key and column
        # Note: batch_alter_table will handle the constraint automatically
        with op.batch_alter_table("agents", recreate="auto") as batch_op:
            batch_op.drop_column("model_id")
    else:
        # For PostgreSQL and other databases, use native operations
        try:
            op.drop_constraint(
                "fk_agents_model_id_models", "agents", type_="foreignkey"
            )
        except Exception:
            pass  # Constraint doesn't exist, skip
        op.drop_column("agents", "model_id")
