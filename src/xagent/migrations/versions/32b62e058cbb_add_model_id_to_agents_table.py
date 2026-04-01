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
        # SQLite: Try batch_alter_table with auto recreate first
        # If reflection fails (users table doesn't exist), fall back to simple add_column
        try:
            with op.batch_alter_table("agents", recreate="auto") as batch_op:
                batch_op.add_column(sa.Column("model_id", sa.Integer(), nullable=True))
                # Only create foreign key if models table exists
                if "models" in tables:
                    batch_op.create_foreign_key(
                        "fk_agents_model_id_models", "models", ["model_id"], ["id"]
                    )
        except sa.exc.NoSuchTableError:
            # Reflection failed due to missing referenced table (users)
            # Fall back to simple column addition without batch mode
            op.add_column("agents", sa.Column("model_id", sa.Integer(), nullable=True))
            # Note: Foreign key not created in this case
            # SQLAlchemy will handle FK creation when it creates the full schema
    else:
        # For PostgreSQL and other databases, use native operations
        op.add_column("agents", sa.Column("model_id", sa.Integer(), nullable=True))
        if "models" in tables:
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
        # Check if FK constraint exists before dropping to avoid transaction error
        fks = inspector.get_foreign_keys("agents")
        fk_exists = any(fk["name"] == "fk_agents_model_id_models" for fk in fks)
        if fk_exists:
            op.drop_constraint(
                "fk_agents_model_id_models", "agents", type_="foreignkey"
            )
        op.drop_column("agents", "model_id")
