"""generalize agent_api_keys with a nullable workforce_id owner

REST API / SDK channel for Workforce (#949 / #805): an API key row now
binds to exactly one owner -- either an agent (``agent_id``) or a
workforce (``workforce_id``). ``agent_id`` becomes nullable so workforce
keys don't need a placeholder agent; exactly-one-owner is enforced at
the service layer (no CHECK constraint, so SQLite needs a rebuild only
for the nullability flip that batch mode already implies).

Revision ID: 20260722_add_workforce_id_to_agent_api_keys
Revises: 20260721_add_workforce_deployment_foundation
Create Date: 2026-07-22

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260722_add_workforce_id_to_agent_api_keys"
# Re-parented onto #950's workforce-triggers migration when the REST/SDK
# and webhook channels merged, linearizing what were two heads off
# 20260721_add_workforce_deployment_foundation. The two are additive and
# order-independent (this one adds agent_api_keys.workforce_id; the other
# adds the workforce trigger tables).
down_revision: Union[str, None] = "20260722_add_workforce_triggers"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "agent_api_keys"
COLUMN = "workforce_id"
INDEX = "ix_agent_api_keys_workforce_id"
FK_NAME = "fk_agent_api_keys_workforce_id"


def _existing_columns(inspector: Inspector, table: str) -> list[str]:
    return [col["name"] for col in inspector.get_columns(table)]


def _existing_indexes(inspector: Inspector, table: str) -> list[str]:
    return [str(index["name"]) for index in inspector.get_indexes(table)]


def _agent_id_nullable(inspector: Inspector, table: str) -> bool:
    for col in inspector.get_columns(table):
        if col["name"] == "agent_id":
            return bool(col["nullable"])
    return True


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if TABLE not in inspector.get_table_names():
        return

    # batch_alter_table so the agent_id nullability flip works on SQLite
    # (table rebuild) as well as PostgreSQL (plain ALTERs). Each op is
    # guarded so the migration is re-runnable on a partially-applied DB.
    needs_column = COLUMN not in _existing_columns(inspector, TABLE)
    needs_nullable = not _agent_id_nullable(inspector, TABLE)

    if needs_column or needs_nullable:
        with op.batch_alter_table(TABLE) as batch_op:
            if needs_column:
                batch_op.add_column(sa.Column(COLUMN, sa.Integer(), nullable=True))
                batch_op.create_foreign_key(
                    FK_NAME,
                    "workforces",
                    [COLUMN],
                    ["id"],
                    ondelete="CASCADE",
                )
            if needs_nullable:
                batch_op.alter_column(
                    "agent_id", existing_type=sa.Integer(), nullable=True
                )

    inspector = Inspector.from_engine(bind)
    if INDEX not in _existing_indexes(inspector, TABLE):
        op.create_index(INDEX, TABLE, [COLUMN])


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if TABLE not in inspector.get_table_names():
        return

    # Workforce-bound rows have agent_id NULL and cannot survive the
    # NOT NULL restore; drop them before flipping nullability back.
    if COLUMN in _existing_columns(inspector, TABLE):
        bind.execute(sa.text(f"DELETE FROM {TABLE} WHERE {COLUMN} IS NOT NULL"))
        bind.execute(sa.text(f"DELETE FROM {TABLE} WHERE agent_id IS NULL"))

        if INDEX in _existing_indexes(inspector, TABLE):
            op.drop_index(INDEX, table_name=TABLE)

        with op.batch_alter_table(TABLE) as batch_op:
            batch_op.alter_column(
                "agent_id", existing_type=sa.Integer(), nullable=False
            )
            batch_op.drop_column(COLUMN)
