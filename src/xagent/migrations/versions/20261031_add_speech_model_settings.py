"""Persist speech model language, voice, and audio defaults."""

import sqlalchemy as sa
from alembic import op

revision = "20261031_speech_settings"
down_revision = "20261031_task_checkpoint_pointer_index"
branch_labels = None
depends_on = None

TABLE = "models"
COLUMNS = {
    "language": sa.String(64),
    "voice": sa.String(255),
    "format": sa.String(64),
    "sample_rate": sa.Integer(),
}


def _column_names() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(TABLE)}


def upgrade() -> None:
    columns = _column_names()
    if not columns:
        return
    with op.batch_alter_table(TABLE) as batch_op:
        for name, column_type in COLUMNS.items():
            if name not in columns:
                batch_op.add_column(sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    columns = _column_names()
    if not columns:
        return
    with op.batch_alter_table(TABLE) as batch_op:
        for name in reversed(COLUMNS):
            if name in columns:
                batch_op.drop_column(name)
