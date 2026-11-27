"""Index the two preview Agent identifiers on hidden preview tasks."""

import sqlalchemy as sa
from alembic import op

from xagent.db.preview_lookup_ddl import (
    CREATE_PREVIEW_CONFIG_TEXT,
    PREVIEW_LOOKUP_KEYS,
    PREVIEW_LOOKUP_PREDICATE,
    preview_lookup_expression,
)

revision = "20261126_preview_lookup_indexes"
down_revision = "20261126_connector_oauth_redirect_uri"
branch_labels = None
depends_on = None

_INDEX = sa.text("""
    SELECT i.indisvalid, i.indrelid = pg_catalog.to_regclass('tasks') AS on_tasks,
           am.amname, i.indisunique, i.indnkeyatts, i.indnatts,
           pg_catalog.pg_get_indexdef(i.indexrelid, 1, false) AS first_key,
           pg_catalog.pg_get_indexdef(i.indexrelid, 2, false) AS second_key,
           pg_catalog.pg_get_indexdef(i.indexrelid, 3, false) AS third_key,
           i.indoption[2] AS last_key_options,
           pg_catalog.pg_get_expr(i.indpred, i.indrelid) AS predicate
    FROM pg_catalog.pg_index AS i
    JOIN pg_catalog.pg_class AS c ON c.oid = i.indexrelid
    JOIN pg_catalog.pg_am AS am ON am.oid = c.relam
    WHERE i.indexrelid = pg_catalog.to_regclass(:name)
""")


def _matches_definition(index, key: str) -> bool:
    return (
        index.on_tasks
        and index.amname == "btree"
        and not index.indisunique
        and index.indnkeyatts == index.indnatts == 3
        and index.first_key == "user_id"
        and index.second_key
        == f"md5(xagent_preview_config_text_v1(agent_config, '{key}'::text))"
        and index.third_key == "id"
        and index.last_key_options == 3  # DESC with the default NULLS FIRST
        and index.predicate
        == "((is_visible IS FALSE) AND (xagent_preview_config_text_v1(agent_config, 'is_preview'::text) = ANY (ARRAY['true'::text, '1'::text])))"
    )


def upgrade() -> None:
    context = op.get_context()
    if context.dialect.name != "postgresql":
        return

    # The migration-only chain can run before metadata creates tasks.
    if not context.as_sql and not sa.inspect(op.get_bind()).has_table("tasks"):
        return

    op.execute(CREATE_PREVIEW_CONFIG_TEXT)
    with context.autocommit_block():
        for key in PREVIEW_LOOKUP_KEYS:
            name = f"ix_tasks_{key}_lookup"
            if not context.as_sql:
                bind = op.get_bind()
                index = bind.execute(_INDEX, {"name": name}).one_or_none()
                if index is not None:
                    if not _matches_definition(index, key):
                        raise RuntimeError(f"Unexpected definition for index {name}")
                    if index.indisvalid:
                        continue
                    # An interrupted concurrent build can leave an invalid index.
                    op.drop_index(
                        name, table_name="tasks", postgresql_concurrently=True
                    )
                elif bind.execute(
                    sa.text("SELECT pg_catalog.to_regclass(:name) IS NOT NULL"),
                    {"name": name},
                ).scalar_one():
                    raise RuntimeError(f"Unexpected relation named {name}")
            op.create_index(
                name,
                "tasks",
                [
                    "user_id",
                    sa.text(preview_lookup_expression(key)),
                    sa.text("id DESC"),
                ],
                postgresql_where=sa.text(PREVIEW_LOOKUP_PREDICATE),
                postgresql_concurrently=True,
            )


def downgrade() -> None:
    context = op.get_context()
    if context.dialect.name != "postgresql":
        return

    with context.autocommit_block():
        for key in PREVIEW_LOOKUP_KEYS:
            op.drop_index(
                f"ix_tasks_{key}_lookup",
                table_name="tasks",
                postgresql_concurrently=True,
                if_exists=True,
            )

    op.execute("DROP FUNCTION IF EXISTS xagent_preview_config_text_v1(json, text)")
