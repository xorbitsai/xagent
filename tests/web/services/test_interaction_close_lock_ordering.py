"""Static guard: the legacy resume close's lock ordering against purge, plus
the shape and reach of the statements and functions that carry it out.

purge_task_rows closes an interaction row's task before deleting the row
itself, an ordering fixed by task_deletion.py. The legacy resume close
statements run in the opposite order (interaction row first, task marker
second), so each of the four call sites needs a lock read on the task row,
taken before either statement, to keep that reversed order from closing a
lock cycle with purge at any lock level purge itself might take.

The obligation splits into two halves because the four call sites satisfy
it two different ways (see task_interaction_close.py's module docstring,
_update_a2a_resume_input_sync's inline comment, and
_update_reply_input_sync's inline comment):

* The online and deferred injection sites (task_command_execution.py and
  task_execution.py) share one
  short-transaction helper, close_legacy_resume_interaction_sync, which
  takes one explicit FOR NO KEY UPDATE / key_share=True lock read before
  calling into the close-and-clear function -- one lock read, not two,
  because the obligation moved into the shared helper.
* The A2A and v1 reply resume-input sites (task_resume.py) each take no lock read of their own: the resume-input
  fence UPDATE each already issues, ahead of its close call, is the first
  statement that transaction directs at tasks or task_interaction_requests,
  and (being a non-key-column UPDATE) already carries the ordering and
  strength this obligation asks for.

This is a static, AST-based check of source order, not a live-database
lock test: it proves the statements appear in the required order in the
code, not that they behave correctly under real concurrent access. The real
concurrency case (purge racing a close, expected to deadlock without the
lock read) is out of scope here -- it needs the retirement finalizers in
place to be exercised meaningfully at all.

Two further guards close out what "the statements only live in one place,
and the unlocked function is only reachable through it" actually means:
the module issues exactly the three UPDATE statements the design calls
for, against exactly the tables named; and the unlocked
close_legacy_resume_interaction (no lock read of its own -- it relies on
its caller to have taken one, or to already hold an equivalent guarantee)
is called from nowhere outside this module except the two sites that have
proven they satisfy that obligation another way. A new caller added
without a lock read of its own, or without the fence-ordering argument the
A2A and v1 reply sites each carry, would defeat the whole point of
splitting the obligation into two halves above.

A third guard (below the two per-obligation halves) covers a different
question entirely: every web/ module that injects a message via
post_user_message, regardless of whether it satisfies the lock-ordering
obligation through the shared WebSocket helper or the fence-ordering
argument, must also wire in some close-family function -- otherwise a
legacy resume answers a question without ever retiring the marker that
names it. That guard is a static, module-level check, not a proof that
the close call runs on every code path.

A fourth guard covers when a close call may run at all: for every
production call to close_legacy_resume_interaction or
close_legacy_resume_interaction_sync outside task_interaction_close.py,
the innermost enclosing condition is an ``if <name> is
UserMessageInjectionOutcome.POSTED_FRESH:`` whose body holds the call, so
a replayed injection does not reach any of them. It is a static check of
source shape; what each site's guard buys is tested beside that site.
"""

from __future__ import annotations

import ast
from pathlib import Path

import xagent
from xagent.web.services import (
    task_command_execution,
    task_execution,
    task_interaction_close,
    task_question_cancel,
    task_resume,
)

CLOSE_MODULE_NAME = "task_interaction_close"
UNLOCKED_CLOSE_FUNCTION = "close_legacy_resume_interaction"
# The only callers allowed to reach the unlocked function directly, because
# each has its own argument for why no lock read is needed (see
# _update_a2a_resume_input_sync's and _update_reply_input_sync's inline
# comments): each site's ownership fence UPDATE is already the first
# statement the transaction directs at tasks or task_interaction_requests.
# Any other name added here must carry the same kind of argument, not just
# a passing test.
APPROVED_UNLOCKED_CALLERS = frozenset({"task_resume"})

POST_USER_MESSAGE_CALL_NAME = "post_user_message"
CLOSE_FAMILY_CALL_NAMES = frozenset(
    {
        "close_legacy_resume_interaction",
        "close_legacy_resume_interaction_sync",
        "clear_interaction_marker_if_unpaired",
    }
)
# web/ modules that call post_user_message but are exempt from the
# close-family wiring obligation checked below. Empty by construction:
# every known post_user_message call site today (task_resume.py,
# task_command_execution.py, task_execution.py) wires in a close-family call. A new call
# site should wire one in too, not be added here as an exception.
APPROVED_UNWIRED_POST_USER_MESSAGE_CALLERS: frozenset[str] = frozenset()


def _source(module) -> str:
    return Path(module.__file__).read_text()


def _find_function(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"function {name!r} not found")


def _callee_name(call: ast.Call) -> str | None:
    """The called attribute or bare name, or ``None`` for any other callee."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _call_lines(tree: ast.AST, *, attr: str) -> list[int]:
    """Line numbers of every Call whose callee name (attribute or bare
    name) matches ``attr``."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _callee_name(node) == attr
    ]


def _key_share_lock_read_lines(tree: ast.AST) -> list[int]:
    """Line numbers of every ``with_for_update(...)`` call that actually
    takes the FOR NO KEY UPDATE lock the ordering obligation requires.

    Verified against this repo's SQLAlchemy 2.0.47: ``key_share=True``
    alone renders FOR NO KEY UPDATE, the strength this obligation needs.
    ``key_share=False`` renders plain FOR UPDATE -- a stronger, deadlock-
    prone lock the obligation's argument does not sanction -- so a literal
    ``True`` is required, not just the keyword's presence. Adding
    ``read=True`` alongside ``key_share=True`` renders FOR KEY SHARE
    instead, a weaker lock that does not carry the ordering argument
    either, so any call with a ``read`` keyword is excluded regardless of
    its value.
    """
    lines = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "with_for_update"
        ):
            continue
        key_share_kw = next((kw for kw in node.keywords if kw.arg == "key_share"), None)
        if key_share_kw is None:
            continue
        is_literal_true = (
            isinstance(key_share_kw.value, ast.Constant)
            and key_share_kw.value.value is True
        )
        has_read_kw = any(kw.arg == "read" for kw in node.keywords)
        if is_literal_true and not has_read_kw:
            lines.append(node.lineno)
    return lines


def test_close_sync_takes_the_lock_read_before_the_close_call() -> None:
    """The two WebSocket injection sites' shared helper: one
    key_share=True lock read, ahead of the call into the close-and-clear
    function.

    Deleting the lock read turns this red.
    """
    fn = _find_function(
        ast.parse(_source(task_interaction_close)),
        "close_legacy_resume_interaction_sync",
    )
    lock_read_lines = _key_share_lock_read_lines(fn)
    assert len(lock_read_lines) == 1, (
        "expected exactly one key_share=True lock read in "
        f"close_legacy_resume_interaction_sync, found {len(lock_read_lines)}"
    )
    close_call_lines = _call_lines(fn, attr="close_legacy_resume_interaction")
    assert len(close_call_lines) == 1, (
        "expected exactly one call into close_legacy_resume_interaction from "
        f"close_legacy_resume_interaction_sync, found {len(close_call_lines)}"
    )
    assert lock_read_lines[0] < close_call_lines[0], (
        "the key_share lock read must precede the call into "
        "close_legacy_resume_interaction"
    )


def test_both_websocket_injection_sites_call_the_shared_close_helper() -> None:
    """The online and deferred injection handlers each call the shared
    helper exactly once, and it is called from nowhere else in either
    module."""
    command_tree = ast.parse(_source(task_command_execution))
    execution_tree = ast.parse(_source(task_execution))
    online = _find_function(command_tree, "handle_task_message")
    deferred = _find_function(execution_tree, "execute_resume_background")
    online_calls = _call_lines(online, attr="close_legacy_resume_interaction_sync")
    deferred_calls = _call_lines(deferred, attr="close_legacy_resume_interaction_sync")
    assert len(online_calls) == 1, (
        "expected exactly one close_legacy_resume_interaction_sync call in "
        f"the online injection handler, found {len(online_calls)}"
    )
    assert len(deferred_calls) == 1, (
        "expected exactly one close_legacy_resume_interaction_sync call in "
        f"the deferred injection handler, found {len(deferred_calls)}"
    )
    all_calls = _call_lines(
        command_tree, attr="close_legacy_resume_interaction_sync"
    ) + _call_lines(execution_tree, attr="close_legacy_resume_interaction_sync")
    assert len(all_calls) == 2, (
        "close_legacy_resume_interaction_sync must be called from exactly "
        f"the two websocket injection sites, found {len(all_calls)} call(s) "
        "across the command and execution services"
    )


def test_a2a_resume_input_fence_precedes_the_close_call() -> None:
    """The A2A resume-input fence UPDATE must precede the close call it
    satisfies the lock obligation for.

    This only checks source order between the fence UPDATE and the close
    call, not that the fence UPDATE is literally the first statement the
    transaction issues against ``tasks`` or ``task_interaction_requests``
    -- see the module docstring and ``_update_a2a_resume_input_sync``'s own
    inline comment for that broader claim, which this assertion does not
    prove.

    Moving the fence UPDATE after the close call turns this red.
    """
    fn = _find_function(
        ast.parse(_source(task_resume)), "_update_a2a_resume_input_sync"
    )
    fence_update_lines = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
    ]
    assert len(fence_update_lines) == 1, (
        "expected exactly one .update(...) call in "
        f"_update_a2a_resume_input_sync, found {len(fence_update_lines)}"
    )
    close_call_lines = _call_lines(fn, attr="close_legacy_resume_interaction")
    assert len(close_call_lines) == 1, (
        "expected exactly one call into close_legacy_resume_interaction from "
        f"_update_a2a_resume_input_sync, found {len(close_call_lines)}"
    )
    assert fence_update_lines[0] < close_call_lines[0], (
        "the resume-input fence UPDATE must precede the call into "
        "close_legacy_resume_interaction"
    )


TABLE_EXISTS_CHECK_NAME = "interaction_requests_table_exists"


def _table_exists_guard_line_for_close_call(
    fn: ast.AST, *, close_call_attr: str
) -> int | None:
    """Line number of an ``if interaction_requests_table_exists(...):``
    guard whose body contains a call to ``close_call_attr``, or ``None`` if
    no such guard wraps that call."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Call) and _callee_name(test) == TABLE_EXISTS_CHECK_NAME
        ):
            continue
        for inner in ast.walk(node):
            if inner is node:
                continue
            if not isinstance(inner, ast.Call):
                continue
            if _callee_name(inner) == close_call_attr:
                return node.lineno
    return None


def test_a2a_resume_input_close_is_gated_by_table_exists_check() -> None:
    """The A2A fence-and-close site must confirm
    task_interaction_requests exists before calling
    close_legacy_resume_interaction. Nothing else in this repo pins that
    gate down: every test database already has the table, so deleting the
    gate leaves the rest of the suite green.

    Deleting the ``if interaction_requests_table_exists(db):`` guard, or
    moving the close call outside its body, turns this red.
    """
    fn = _find_function(
        ast.parse(_source(task_resume)), "_update_a2a_resume_input_sync"
    )
    guard_line = _table_exists_guard_line_for_close_call(
        fn, close_call_attr="close_legacy_resume_interaction"
    )
    assert guard_line is not None, (
        "expected close_legacy_resume_interaction to be called inside an "
        "`if interaction_requests_table_exists(db):` guard in "
        "_update_a2a_resume_input_sync"
    )


def test_reply_resume_input_close_is_gated_by_table_exists_check() -> None:
    """Same obligation as the A2A site, above, for task_resume.py's
    fence-and-close pair.

    Deleting the ``if interaction_requests_table_exists(db):`` guard, or
    moving the close call outside its body, turns this red.
    """
    fn = _find_function(ast.parse(_source(task_resume)), "_update_reply_input_sync")
    guard_line = _table_exists_guard_line_for_close_call(
        fn, close_call_attr="close_legacy_resume_interaction"
    )
    assert guard_line is not None, (
        "expected close_legacy_resume_interaction to be called inside an "
        "`if interaction_requests_table_exists(db):` guard in "
        "_update_reply_input_sync"
    )


def _fence_update_value_columns(fn: ast.AST) -> set[str]:
    """Attribute names (``Task.<name>``) used as keys in the resume-input
    fence's ``.update({...})`` values dict."""
    for node in ast.walk(fn):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            continue
        return {key.attr for key in node.args[0].keys if isinstance(key, ast.Attribute)}
    raise AssertionError("no .update({...}) call found")


def test_a2a_resume_input_fence_writes_exactly_the_approved_columns() -> None:
    """The fence UPDATE's no-lock-read argument (see
    _update_a2a_resume_input_sync's inline comment) holds only because every
    column it writes is a non-key column. Adding a key column -- or any
    column covered by a unique index -- to the UPDATE's values without
    re-justifying that argument and widening
    task_resume.RESUME_INPUT_FENCE_UPDATE_COLUMNS to match must turn this red.
    """
    fn = _find_function(
        ast.parse(_source(task_resume)), "_update_a2a_resume_input_sync"
    )
    assert _fence_update_value_columns(fn) == set(
        task_resume.RESUME_INPUT_FENCE_UPDATE_COLUMNS
    )


def test_reply_resume_input_fence_precedes_the_close_call() -> None:
    """The v1 reply resume-input fence UPDATE must precede the close call it
    satisfies the lock obligation for -- the same argument as the A2A site,
    applied to task_resume.py's own fence-and-close pair.

    Moving the fence UPDATE after the close call turns this red.
    """
    fn = _find_function(ast.parse(_source(task_resume)), "_update_reply_input_sync")
    fence_update_lines = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
    ]
    assert len(fence_update_lines) == 1, (
        "expected exactly one .update(...) call in "
        f"_update_reply_input_sync, found {len(fence_update_lines)}"
    )
    close_call_lines = _call_lines(fn, attr="close_legacy_resume_interaction")
    assert len(close_call_lines) == 1, (
        "expected exactly one call into close_legacy_resume_interaction from "
        f"_update_reply_input_sync, found {len(close_call_lines)}"
    )
    assert fence_update_lines[0] < close_call_lines[0], (
        "the resume-input fence UPDATE must precede the call into "
        "close_legacy_resume_interaction"
    )


def test_reply_resume_input_fence_writes_exactly_the_approved_columns() -> None:
    """The v1 reply fence UPDATE's no-lock-read argument (see
    _update_reply_input_sync's inline comment) holds only because every
    column it writes is a non-key column, the same argument the A2A site
    makes. Adding a key column -- or any column covered by a unique index --
    to the UPDATE's values without re-justifying that argument and widening
    task_resume.RESUME_INPUT_FENCE_UPDATE_COLUMNS to match must turn this red.
    """
    fn = _find_function(ast.parse(_source(task_resume)), "_update_reply_input_sync")
    assert _fence_update_value_columns(fn) == set(
        task_resume.RESUME_INPUT_FENCE_UPDATE_COLUMNS
    )


def _sa_update_targets(tree: ast.AST) -> list[str]:
    """The target table name of every ``sa.update(<Table>)`` call, in
    source order. Matched on ``sa.update(...)`` specifically (the Core
    form this module uses throughout), not ORM-style ``.query().update()``
    calls, which this module never issues."""
    targets = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sa"
        ):
            continue
        if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            targets.append(node.args[0].id)
        else:
            targets.append("<unrecognized target shape>")
    return targets


def test_close_module_issues_exactly_the_expected_update_statements() -> None:
    """The whole module issues exactly four UPDATE statements: the legacy
    close (task_interaction_requests), its paired clear (tasks), the
    compensation paths' clear (tasks) -- both clears guarded by the same
    NOT EXISTS clause -- and the cancel-question termination
    (task_interaction_requests). Any other statement, or one against an
    unexpected table, means either a duplicated write path or a table this
    design never named."""
    targets = _sa_update_targets(ast.parse(_source(task_interaction_close)))
    assert targets == [
        "TaskInteractionRequest",
        "Task",
        "Task",
        "TaskInteractionRequest",
    ], targets


def test_cancel_claims_tasks_row_before_terminating_the_interaction() -> None:
    """``_cancel_waiting_task`` must lock/write the ``tasks`` row before the
    interaction termination UPDATE, matching every other close path's
    tasks-then-interaction order. Counting statements cannot see this; the
    source order of the two calls is what carries the lock obligation.
    """
    fn = _find_function(
        ast.parse(_source(task_question_cancel)), "_cancel_waiting_task"
    )

    def _queries_task(node: ast.AST) -> bool:
        return any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "query"
            and sub.args
            and isinstance(sub.args[0], ast.Name)
            and sub.args[0].id == "Task"
            for sub in ast.walk(node)
        )

    claim = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and _queries_task(node.func.value)
    )
    terminate_lines = _call_lines(
        fn, attr="terminate_active_interaction_for_question_cancel"
    )
    assert len(terminate_lines) == 1, terminate_lines
    assert claim.lineno < terminate_lines[0], (
        "the tasks-row claim UPDATE must precede the interaction termination"
    )


def _calls_to(tree: ast.AST, *, name: str) -> bool:
    return bool(_call_lines(tree, attr=name))


def test_unlocked_close_is_only_called_from_the_approved_sites() -> None:
    """close_legacy_resume_interaction takes no lock read of its own -- it
    relies on its caller to already satisfy that obligation another way
    (see the module docstring above). Scans every source file under the
    installed xagent package, so a new caller added anywhere without
    updating APPROVED_UNLOCKED_CALLERS is caught regardless of which
    package it lands in, not only task_resume.py and the execution modules.

    Known blind spot, not fixed here: this matches the call as written
    (``close_legacy_resume_interaction(...)`` or
    ``some_module.close_legacy_resume_interaction(...)``), not a name
    reached through a re-export or an alias assigned to another name --
    the same limitation the interaction-staging production gate documents
    for its own equivalent scan.
    """
    root = Path(next(iter(xagent.__path__)))
    callers: set[str] = set()
    for path in root.rglob("*.py"):
        if path.stem == CLOSE_MODULE_NAME:
            # This module's own close_legacy_resume_interaction_sync is the
            # approved lock-reading wrapper; it is expected to call the
            # unlocked function and is not an "other" caller.
            continue
        source = path.read_text(encoding="utf-8")
        if _calls_to(ast.parse(source), name=UNLOCKED_CLOSE_FUNCTION):
            callers.add(path.stem)
    assert callers == APPROVED_UNLOCKED_CALLERS, (
        f"unlocked close_legacy_resume_interaction is called from {callers}, "
        f"expected exactly {set(APPROVED_UNLOCKED_CALLERS)}"
    )


def test_every_post_user_message_caller_wires_the_close_family() -> None:
    """Every web/ module that injects a message via post_user_message must
    also call one of the close-family functions somewhere in the same
    module, or a legacy resume can answer a question through that module
    without ever retiring the marker that names it.

    This is a module-level check, not a call-site-level one: it proves a
    close-family call exists somewhere in the module, not that it runs on
    every code path that reaches post_user_message. Today's four call
    sites across three modules (task_resume.py, task_command_execution.py,
    task_execution.py)
    each carry their own per-site argument for why their wiring is
    complete; see the tests
    above and task_interaction_close.py's module docstring.

    Scan root is xagent/web, not the whole installed xagent package: the
    scan is looking for callers of post_user_message, but post_user_message
    is itself *defined* under xagent/core/agent (service.py, runner.py,
    registry.py, execution_adapter.py), where each layer's delegation to
    the next also matches the same AST call-name check. Widening the root
    to include core/ would catch those internal delegations as unwired
    callers and turn this permanently red for a reason unrelated to the
    wiring obligation being checked.

    Deleting a close-family call from a module that calls
    post_user_message, without adding the module to
    APPROVED_UNWIRED_POST_USER_MESSAGE_CALLERS, turns this red.
    """
    root = Path(next(iter(xagent.__path__))) / "web"
    unwired: set[str] = set()
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        if not _calls_to(tree, name=POST_USER_MESSAGE_CALL_NAME):
            continue
        if path.stem in APPROVED_UNWIRED_POST_USER_MESSAGE_CALLERS:
            continue
        if not any(_calls_to(tree, name=n) for n in CLOSE_FAMILY_CALL_NAMES):
            unwired.add(path.stem)
    assert not unwired, (
        f"modules {unwired} call post_user_message but wire in no "
        "close-family function (close_legacy_resume_interaction[_sync] / "
        "clear_interaction_marker_if_unpaired); either wire one in or add "
        "the module to APPROVED_UNWIRED_POST_USER_MESSAGE_CALLERS with a "
        "documented reason"
    )


INJECTION_OUTCOME_ENUM_NAME = "UserMessageInjectionOutcome"
FRESH_INJECTION_MEMBER = "POSTED_FRESH"
CLOSE_CALL_NAMES = frozenset(
    {"close_legacy_resume_interaction", "close_legacy_resume_interaction_sync"}
)
# Every production close call, as (module stem, enclosing function). Each
# is the one injection site of that function, and the innermost condition
# around each must be an `is POSTED_FRESH` branch.
EXPECTED_CLOSE_CALL_SITES = (
    ("task_command_execution", "handle_task_message"),
    ("task_execution", "execute_resume_background"),
    ("task_resume", "_update_a2a_resume_input_sync"),
    ("task_resume", "_update_reply_input_sync"),
)
# Close calls allowed to run without that guard. Empty by construction: a
# replayed injection must not retire a question at any site.
APPROVED_UNGUARDED_CLOSE_CALL_SITES: frozenset[tuple[str, str]] = frozenset()


def _is_fresh_injection_test(test: ast.expr) -> bool:
    """``<name> is UserMessageInjectionOutcome.POSTED_FRESH``. The left side
    must be a plain name, so an attribute (another enum member, or another
    object's outcome) or a constant is rejected; which local variable the
    name is cannot be seen here, and each site's own test covers that."""
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is)
        and len(test.comparators) == 1
    ):
        return False
    member = test.comparators[0]
    return (
        isinstance(member, ast.Attribute)
        and member.attr == FRESH_INJECTION_MEMBER
        and isinstance(member.value, ast.Name)
        and member.value.id == INJECTION_OUTCOME_ENUM_NAME
    )


def _close_calls_and_guards(tree: ast.AST) -> list[tuple[str, bool]]:
    """For each close call in ``tree``: the name of the function it is in,
    and whether its nearest enclosing condition (an ``if``, a conditional
    expression or a ``while``, looked for up to that function) is an
    ``if <name> is UserMessageInjectionOutcome.POSTED_FRESH:`` whose body
    holds the call."""
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _callee_name(node) not in CLOSE_CALL_NAMES:
            continue
        guarded: bool | None = None
        function_name = "<module>"
        child: ast.AST = node
        while child in parents:
            parent = parents[child]
            if guarded is None and isinstance(parent, (ast.If, ast.IfExp, ast.While)):
                guarded = (
                    isinstance(parent, ast.If)
                    and child in parent.body
                    and _is_fresh_injection_test(parent.test)
                )
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_name = parent.name
                break
            child = parent
        found.append((function_name, guarded is True))
    return found


def test_every_close_calls_innermost_condition_is_a_fresh_injection_guard() -> None:
    """Each production close call's innermost enclosing condition is
    ``if <name> is UserMessageInjectionOutcome.POSTED_FRESH:`` with the call
    in its body. Scans the installed xagent package except
    task_interaction_close.py, whose wrapper delegation is not a call site.
    A close passed as a value (for example through ``partial``) is not
    seen; each site's behavior is tested beside it."""
    root = Path(next(iter(xagent.__path__)))
    sites: list[tuple[str, str]] = []
    unguarded: set[tuple[str, str]] = set()
    for path in root.rglob("*.py"):
        if path.stem == CLOSE_MODULE_NAME:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function_name, guarded in _close_calls_and_guards(tree):
            site = (path.stem, function_name)
            sites.append(site)
            if not guarded:
                unguarded.add(site)
    assert sorted(sites) == sorted(EXPECTED_CLOSE_CALL_SITES)
    assert unguarded == set(APPROVED_UNGUARDED_CLOSE_CALL_SITES)
