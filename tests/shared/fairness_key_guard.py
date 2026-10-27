"""Fail any test whose code rewrites the tenant key of a command row.

The key is written when a command is accepted and never updated afterwards, so
the only statement allowed to write it is the flush ``stage_task_command``
issues for the command it is accepting. The guard listens on every engine
through the ``Engine`` class and looks at each statement that mentions the key,
in any case:

* A compiled UPDATE of the command table, or of an alias of it, counts when the
  columns it writes include the key, so a statement that only reads the key
  (``SET x = (SELECT fairness_key ...)``) passes.
* A compiled INSERT into the command table counts when its ON CONFLICT DO UPDATE
  sets the key. It is never allowed: it filters on no command id. The key is
  matched in any case, since SQLite reads ``"FAIRNESS_KEY"`` as the same column.
* A statement without a compiled UPDATE (raw SQL) counts when, after leading
  comments, it is an UPDATE of the command table (optionally behind a WITH clause
  or as UPDATE OR <action>) and the key's name appears anywhere in it, in any
  case. Raw SQL is not parsed: an UPDATE of the table that names the key only in
  WHERE, in a string value or in a subquery is refused as well, so such a
  statement has to be written with the ORM.

A statement that counts is allowed only when ``stage_task_command``, found by
its code object on the stack of the thread that runs the statement, is
accepting a command whose id is the id every parameter set of the statement
filters on. Raw SQL is never allowed, since there is no compiled UPDATE to read
the id from. If the command, or its id, cannot be read, the statement is not
allowed. The listener runs synchronously in the thread that executes the
statement, so a ``stage_task_command`` running in a worker thread is found on
that thread's stack. The id is read from the local variable ``command`` of
``stage_task_command``; renaming it makes the guard refuse the write, which fails
a test loudly instead of letting a write through.

A violation is recorded and raised. Production code may catch the exception, so
every test also checks the record when it ends, and the session checks it once
more at the end. Taking the record removes only what was taken, so a violation
a background thread records meanwhile is reported by the next check.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.sql import operators
from sqlalchemy.sql.dml import Insert, Update
from sqlalchemy.sql.elements import BinaryExpression, BindParameter, BooleanClauseList

TABLE = "task_execution_commands"
KEY = "fairness_key"
STAGING_MODULE = "xagent.web.services.task_command_transport"

# (test node id, statement, parameters) of every statement that was refused.
violations: list[tuple[str, str, Any]] = []
current_test = ""

_RAW_UPDATE = re.compile(
    r"^(?:WITH\b.*?\bUPDATE|UPDATE)(?:\s+OR\s+\w+)?\s+(?:ONLY\s+)?"
    r'(?:"?\w+"?\.)?"?' + TABLE + r'"?\b',
    re.IGNORECASE | re.DOTALL,
)


def mentions_key(statement: str) -> bool:
    """Whether the key's name appears anywhere in the statement, in any case."""
    return KEY in statement.lower()


def _without_leading_comments(statement: str) -> str:
    while True:
        statement = statement.lstrip()
        if statement.startswith("/*") and "*/" in statement:
            statement = statement[statement.index("*/") + 2 :]
        elif statement.startswith("--") and "\n" in statement:
            statement = statement[statement.index("\n") + 1 :]
        else:
            return statement


def raw_update_mentions_key(statement: str) -> bool:
    """Whether raw SQL is an UPDATE of the command table naming the key anywhere."""
    return (
        mentions_key(statement)
        and _RAW_UPDATE.match(_without_leading_comments(statement)) is not None
    )


def _base_table(table: Any) -> Any:
    """The table behind an alias, or the table itself."""
    while hasattr(table, "element"):
        table = table.element
    return table


def insert_updates_key(compiled: Any) -> bool:
    """Whether a compiled INSERT into the command table sets the key on conflict.

    The name is compared in any case: SQLite reads ``"FAIRNESS_KEY"`` as the key
    column although SQLAlchemy only warns about a name that is not a column key.
    """
    statement = compiled.statement
    if not isinstance(statement, Insert) or _base_table(statement.table).name != TABLE:
        return False
    clause = getattr(statement, "_post_values_clause", None)
    return any(
        str(getattr(column, "key", column)).lower() == KEY
        for column, _ in getattr(clause, "update_values_to_set", None) or ()
    )


def compiled_update_writes_key(compiled: Any) -> bool:
    """Whether a compiled UPDATE of the command table writes the key column."""
    statement = compiled.statement
    if not isinstance(statement, Update) or _base_table(statement.table).name != TABLE:
        return False
    written = set(compiled.column_keys or ())
    state = getattr(compiled, "compile_state", None)
    for column in getattr(state, "_dict_parameters", None) or {}:
        written.add(getattr(column, "key", column))
    return KEY in written


def _conjuncts(clause: Any) -> list[Any]:
    if isinstance(clause, BooleanClauseList) and clause.operator is operators.and_:
        return [part for child in clause.clauses for part in _conjuncts(child)]
    return [clause]


def where_id_parameter(compiled: Any) -> str | None:
    """The bind name the UPDATE's WHERE compares the command id with, if any."""
    for criterion in compiled.statement._where_criteria:
        for clause in _conjuncts(criterion):
            if not (
                isinstance(clause, BinaryExpression) and clause.operator is operators.eq
            ):
                continue
            for column, other in (
                (clause.left, clause.right),
                (clause.right, clause.left),
            ):
                if (
                    getattr(column, "name", None) == "id"
                    and getattr(getattr(column, "table", None), "name", None) == TABLE
                    and isinstance(other, BindParameter)
                ):
                    return compiled.bind_names.get(other)
    return None


def every_parameter_set_targets(
    parameter_sets: Sequence[dict[str, Any]],
    id_parameter: str | None,
    command_id: int | None,
) -> bool:
    """Whether every parameter set filters on ``command_id``.

    Not allowed when the id parameter or the command id is unknown, when there
    is no parameter set, or when any one set (not only the first) names another
    id.
    """
    if id_parameter is None or command_id is None or not parameter_sets:
        return False
    return all(group.get(id_parameter) == command_id for group in parameter_sets)


def _command_being_accepted() -> int | None:
    """The id of the command ``stage_task_command`` is accepting on this stack."""
    module = sys.modules.get(STAGING_MODULE)
    if module is None:
        return None
    code = module.stage_task_command.__code__
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code is code:
            command = frame.f_locals.get("command")
            # Read the loaded attribute only: a refresh here would run a statement
            # inside this listener.
            values = getattr(command, "__dict__", {})
            command_id = values.get("id")
            return command_id if isinstance(command_id, int) else None
        frame = frame.f_back
    return None


def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    if not mentions_key(statement):
        return
    compiled = None if context is None else context.compiled
    target = None if compiled is None else compiled.statement
    if isinstance(target, Update):
        if not compiled_update_writes_key(compiled):
            return
        if every_parameter_set_targets(
            context.compiled_parameters,
            where_id_parameter(compiled),
            _command_being_accepted(),
        ):
            return
    elif isinstance(target, Insert):
        if not insert_updates_key(compiled):
            return
    elif not raw_update_mentions_key(statement):
        return
    violations.append((current_test, statement, parameters))
    raise AssertionError(
        f"{current_test or 'a test'} rewrote {TABLE}.{KEY}, which is written once "
        f"when a command is accepted and never updated: {statement!r} "
        f"with {parameters!r}"
    )


_installed = False


def install() -> None:
    global _installed
    if not _installed:
        event.listen(Engine, "before_cursor_execute", before_cursor_execute)
        _installed = True


def take_violations() -> list[tuple[str, str, Any]]:
    """Return the violations recorded so far and forget only those.

    Any thread may record a violation; only the pytest thread takes them. A
    violation recorded after the copy is taken stays in the list for the next
    call, because only as many leading items as were copied are deleted.
    """
    found = violations[:]
    del violations[: len(found)]
    return found
