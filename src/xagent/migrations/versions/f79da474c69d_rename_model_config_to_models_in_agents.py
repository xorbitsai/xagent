"""rename model_config to models in agents

Revision ID: f79da474c69d
Revises: b9d890ed31b5
Create Date: 2026-01-31 23:22:03.212451

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "f79da474c69d"
down_revision: Union[str, None] = "b9d890ed31b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if model_config column exists before renaming
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    columns = [col["name"] for col in inspector.get_columns("agents")]
    dialect_name = bind.dialect.name

    if "model_config" in columns and "models" not in columns:
        # Rename model_config to models
        if dialect_name == "sqlite":
            # SQLite requires batch mode for ALTER TABLE operations
            with op.batch_alter_table("agents", recreate="auto") as batch_op:
                batch_op.alter_column("model_config", new_column_name="models")
        else:
            # PostgreSQL and other databases support native ALTER TABLE
            op.execute("ALTER TABLE agents RENAME COLUMN model_config TO models")
    elif "models" in columns and "model_config" in columns:
        # Both columns exist - drop model_config as models is the newer one
        if dialect_name == "sqlite":
            with op.batch_alter_table("agents", recreate="auto") as batch_op:
                batch_op.drop_column("model_config")
        else:
            op.drop_column("agents", "model_config")
    elif "models" in columns:
        # Column already renamed, skip
        pass


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    dialect_name = bind.dialect.name

    # Revert column name from models to model_config
    if dialect_name == "sqlite":
        # SQLite requires batch mode for ALTER TABLE operations
        with op.batch_alter_table("agents", recreate="auto") as batch_op:
            batch_op.alter_column("models", new_column_name="model_config")
    else:
        # PostgreSQL and other databases support native ALTER TABLE
        op.execute("ALTER TABLE agents RENAME COLUMN models TO model_config")
