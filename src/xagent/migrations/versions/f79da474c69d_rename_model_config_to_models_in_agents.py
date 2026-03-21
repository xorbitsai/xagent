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
    columns = [col["name"] for col in inspector.get_columns("agents")]

    if "model_config" in columns and "models" not in columns:
        # Use batch mode for SQLite to rename column
        with op.batch_alter_table("agents", recreate="auto") as batch_op:
            batch_op.alter_column("model_config", new_column_name="models")
    elif "models" in columns and "model_config" in columns:
        # Both columns exist - drop model_config as models is the newer one
        with op.batch_alter_table("agents", recreate="auto") as batch_op:
            batch_op.drop_column("model_config")
    elif "models" in columns:
        # Column already renamed, skip
        pass


def downgrade() -> None:
    # Use batch mode for SQLite to revert column name
    with op.batch_alter_table("agents", recreate="auto") as batch_op:
        batch_op.alter_column("models", new_column_name="model_config")
