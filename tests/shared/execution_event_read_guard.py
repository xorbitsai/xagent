"""Content-read tripwire for explicitly V2-only acceptance test scopes."""

import re
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy.sql import operators, visitors
from sqlalchemy.sql.elements import ClauseElement

_FORBIDDEN = {
    ("task_chat_messages", "content"),
    ("task_chat_messages", "attachments"),
    ("task_chat_messages", "interactions"),
    ("trace_events", "data"),
    ("dag_executions", "current_plan"),
    ("dag_executions", "skipped_steps"),
    ("trace_message_blobs", "message_data"),
    ("trace_checkpoint_blobs", "blob_data"),
}


def _legacy_control_json(expression, compiled, parameters):
    keys = []
    base = expression
    while getattr(base, "operator", None) is operators.json_getitem_op:
        keys.insert(
            0,
            parameters.get(
                compiled.bind_names.get(base.right), getattr(base.right, "value", None)
            ),
        )
        base = base.left
    if (
        getattr(getattr(base, "table", None), "name", None) == "trace_events"
        and getattr(base, "name", None) == "data"
        and tuple(keys)
        in {
            ("checkpoint_type",),
            ("turn_id",),
            ("_task_run_id",),
            ("root_execution_id",),
            ("execution_id",),
            ("snapshot", "execution_id"),
        }
    ):
        return sa.literal(None)
    return None


def reject_legacy_content(_conn, _cursor, statement, _params, context, _many):
    if context.isinsert or context.isupdate or context.isdelete:
        return
    compiled = context.compiled
    sql = statement.lower().replace('"', "")
    # Log queries share an eager relationship loader with V1. Its explicit
    # V1-only subquery cannot fetch V2 content, even for a V2 parent row.
    versions = (
        []
        if compiled is None
        else [
            value
            for parameters in context.compiled_parameters
            for key, value in parameters.items()
            if key.startswith("conversation_storage_version_")
        ]
    )
    if versions == [1] and re.search(
        r"task_chat_messages.task_id in \(select tasks.id\s+from tasks\s+where tasks.conversation_storage_version = (?:\?|%\(conversation_storage_version_\d+\)s)\)",
        sql,
    ):
        return
    # Metrics unions have exactly one legacy source, joined to its V1 task
    # before UNION ALL. Check actual bound values, including cached statements.
    # Only that source's data column is exempt; all other content stays blocked.
    legacy_metrics = (
        versions
        and versions[0] == 1
        and len(re.findall(r"\bfrom trace_events\b", sql)) == 1
        and re.search(
            r"from trace_events join tasks on tasks.id = trace_events.task_id\s+where tasks.conversation_storage_version = (?:\?|%\(conversation_storage_version_\d+\)s)",
            sql,
        )
    )
    forbidden = _FORBIDDEN - ({("trace_events", "data")} if legacy_metrics else set())
    # Inspect the compiled result map: ORM selected_columns still includes
    # explicitly deferred attributes that are absent from the emitted SQL.
    result_columns = getattr(compiled, "_result_columns", ())
    columns = [
        expression
        for result in result_columns
        for expression in result.objects
        if isinstance(expression, ClauseElement)
    ]
    predicate = getattr(getattr(compiled, "statement", None), "whereclause", None)
    if predicate is not None:
        # Projection deduplication/pruning may inspect only these control keys.
        # All other legacy content predicates are blocked, even without a
        # content column in the selected result.
        predicate = visitors.replacement_traverse(
            predicate,
            {},
            lambda node: _legacy_control_json(
                node, compiled, context.compiled_parameters[0]
            ),
        )
        for expression in visitors.iterate(predicate):
            for base in getattr(expression, "base_columns", ()):
                key = (
                    getattr(getattr(base, "table", None), "name", None),
                    getattr(base, "name", None),
                )
                assert key not in forbidden, f"Legacy content read in predicate: {key}"
    if columns:
        # Check selected expressions, including aliases and scalar subqueries.
        # JSON predicates used solely to prune old checkpoint control rows do
        # not fetch their bodies and remain allowed until stage 3.5.
        for column in columns:
            for expression in visitors.iterate(column):
                for base in getattr(expression, "base_columns", ()):
                    key = (
                        getattr(getattr(base, "table", None), "name", None),
                        getattr(base, "name", None),
                    )
                    assert key not in forbidden, f"Legacy content read: {'.'.join(key)}"
    else:
        # Production paths use compiled SQLAlchemy selects. Keep literal SQL
        # probes from bypassing the tripwire's self-tests.
        selected = re.split(r"\bfrom\b", sql, maxsplit=1)[0]
        for table, column in forbidden:
            assert f"{table}.{column}" not in selected, (
                f"Legacy content read: {table}.{column}"
            )


@contextmanager
def forbid_legacy_content(engine):
    sa.event.listen(engine, "before_cursor_execute", reject_legacy_content)
    try:
        yield
    finally:
        sa.event.remove(engine, "before_cursor_execute", reject_legacy_content)
