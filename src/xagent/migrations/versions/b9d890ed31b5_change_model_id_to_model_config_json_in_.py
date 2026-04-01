"""change model_id to model_config json in agents

Revision ID: b9d890ed31b5
Revises: 32b62e058cbb
Create Date: 2026-01-31 23:20:57.039344

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b9d890ed31b5"
down_revision: Union[str, None] = "32b62e058cbb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context
    from sqlalchemy.engine.reflection import Inspector

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    columns = [col["name"] for col in inspector.get_columns("agents")]
    dialect_name = bind.dialect.name

    # Drop model_id column if it exists
    if "model_id" in columns:
        if dialect_name == "sqlite":
            # SQLite batch mode handles FK constraints automatically when
            # dropping columns — named FK constraints may not be tracked
            with op.batch_alter_table("agents", recreate="auto") as batch_op:
                batch_op.drop_column("model_id")
        else:
            # PostgreSQL: check if FK constraint exists before dropping
            fks = inspector.get_foreign_keys("agents")
            fk_exists = any(fk["name"] == "fk_agents_model_id_models" for fk in fks)
            if fk_exists:
                op.drop_constraint(
                    "fk_agents_model_id_models", "agents", type_="foreignkey"
                )
            op.drop_column("agents", "model_id")

    # Add model_config column if it doesn't exist
    # But skip if models column already exists (will be renamed by next migration)
    if "model_config" not in columns and "models" not in columns:
        op.add_column("agents", sa.Column("model_config", sa.JSON(), nullable=True))


def downgrade() -> None:
    from alembic import context
    from sqlalchemy.engine.reflection import Inspector

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    columns = [col["name"] for col in inspector.get_columns("agents")]
    dialect_name = bind.dialect.name

    # Drop model_config column if it exists
    if "model_config" in columns:
        if dialect_name == "sqlite":
            with op.batch_alter_table("agents", recreate="auto") as batch_op:
                batch_op.drop_column("model_config")
        else:
            op.drop_column("agents", "model_config")

    # Add back model_id column if it doesn't exist
    if "model_id" not in columns:
        if dialect_name == "sqlite":
            # SQLite: use batch_alter_table
            with op.batch_alter_table("agents", recreate="auto") as batch_op:
                batch_op.add_column(sa.Column("model_id", sa.Integer(), nullable=True))
                # Don't create FK in SQLite batch mode as it can cause issues
                # The FK will be recreated by the next migration's upgrade
        else:
            # PostgreSQL
            op.add_column("agents", sa.Column("model_id", sa.Integer(), nullable=True))
            # Only create FK if models table exists
            if "models" in tables:
                op.create_foreign_key(
                    "fk_agents_model_id_models",
                    "agents",
                    ["model_id"],
                    "models",
                    ["id"],
                )
