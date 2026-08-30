"""Fail-closed AST analysis for recognized client-visible WebSocket shapes.

This models the listed direct producers, their known forwarding wrapper, and
the error-payload helpers and dict-spread grammar declared below. It is not
general interprocedural Python analysis or a claim about every possible egress.
"""

from __future__ import annotations

import ast
from typing import Iterator, NamedTuple

FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
TRY_NODES = (ast.Try, ast.TryStar)
LOOP_NODES = (ast.For, ast.AsyncFor, ast.While)
COMPREHENSION_NODES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

# arg name -> positional index of the client-visible message
PRODUCERS: dict[str, int | None] = {
    "finish_delivery_failure": 0,
    "finish_delivery": 1,
    "notify_deferred_delivery": 1,
    "send_message_delivery": None,  # keyword-only
}

# The one deliberate exception: agent RuntimeError text is passed through to
# the INITIATING SENDER - the rejection ack and the personal error bubble.
# Narrowing that wording is the product decision tracked in #1479.
#
# The broadcast half of the passthrough is closed (maintainer scope ruling on
# #1514): the task-wide broadcast reaches every connection under the task_id,
# anonymous widget and share visitors included, and DurableStorageOperation-
# Error subclasses RuntimeError with tenant-scope text in its message, so
# broadcasts carry CLIENT_SAFE_TASK_FAILURE instead - pinned by the two
# audience-boundary tests at the end of this file. Still #1479: whether the
# sender copy should also be narrowed when the initiator is an anonymous
# public connection.
#
# Anchored to the function and the exception handler that owns the expression.
# The raw wording is the deliberate #1479 RuntimeError contract; the same text
# in a validation or generic-exception branch is not curated and must fail.


class _RawMessageAllowance(NamedTuple):
    function: str
    handler: str
    expression: str


ALLOWED_RAW_MESSAGES = {
    _RawMessageAllowance(
        "_handle_chat_message_unserialized",
        "RuntimeError",
        "f'Runtime error: {str(e)}'",
    ),
    _RawMessageAllowance(
        "handle_execute_task", "RuntimeError", "f'Runtime error: {str(e)}'"
    ),
    _RawMessageAllowance(
        "handle_intervention", "RuntimeError", "f'Runtime error: {str(e)}'"
    ),
    _RawMessageAllowance(
        "_handle_pause_task_unserialized",
        "RuntimeError",
        "f'Runtime error: {str(e)}'",
    ),
    _RawMessageAllowance(
        "_handle_resume_task_unserialized",
        "RuntimeError",
        "f'Runtime error: {str(e)}'",
    ),
}


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_functions(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[ast.AST]:
    """Innermost-first chain of functions a node can read locals from."""
    chain: list[ast.AST] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, FUNCTION_NODES):
            chain.append(current)
        current = parents.get(current)
    return chain


def _message_expression(node: ast.Call, index: int | None) -> ast.expr | None:
    for keyword in node.keywords:
        if keyword.arg == "message":
            return keyword.value
    if index is not None and len(node.args) > index:
        return node.args[index]
    return None


# Text sinks take a serialized payload, so the dict sits one call deeper.
SERIALIZED_ERROR_PAYLOAD_SINKS = {
    "fanout_websocket_text",
    "send_text",
    "send_websocket_text",
}
ERROR_PAYLOAD_SINKS = {
    "send_personal_message",
    "broadcast_to_task",
    *SERIALIZED_ERROR_PAYLOAD_SINKS,
}

# Both render in the client's conversation, so both are the same disclosure
# surface. ``agent_error`` was missing until review found a producer using it.
ERROR_PAYLOAD_TYPES = {"error", "agent_error", "task_error"}
SENSITIVE_PAYLOAD_FIELDS = {"type", "message", "error"}
NON_ERROR_STREAM_EVENT_BUILDERS = {
    "_agent_outbound_event_type": None,
    "_waiting_or_paused_event_fields": 0,
}
DICT_ERROR_PAYLOAD_BUILDERS = {
    "_read_task_error_payload_offloop": "error",
    "_task_error_payload": "error",
    "_terminal_task_error_payload": "agent_error",
    "create_terminal_task_error_event": "task_error",
}

# The only functions allowed to mint client-visible text from an exception.
SAFE_MESSAGE_BUILDERS = {
    "client_safe_error_message",
    "client_safe_task_command_failure",
}
SAFE_MESSAGE_CONSTANTS = {
    "CLIENT_SAFE_TASK_FAILURE",
    "CLIENT_SAFE_VALIDATION_ERROR",
}


def _unwrap_serializer(expr: ast.expr, parents: dict[ast.AST, ast.AST]) -> ast.expr:
    """``json.dumps(payload)`` -> ``payload``; anything else unchanged."""
    if (
        isinstance(expr, ast.Call)
        and _called_name(expr, parents) == "dumps"
        and expr.args
    ):
        return expr.args[0]
    return expr


def _call_argument(
    node: ast.Call,
    position: int,
    keyword: str,
) -> ast.expr | None:
    """Resolve one argument passed either positionally or by exact keyword."""
    if len(node.args) > position:
        return node.args[position]
    return next(
        (candidate.value for candidate in node.keywords if candidate.arg == keyword),
        None,
    )


def _has_local_binding(
    node: ast.AST,
    name: str,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    scopes = _enclosing_functions(node, parents)
    return _is_parameter(scopes, name) or any(
        _local_assignments([scope], name) for scope in scopes
    )


def _is_known_non_error_event_type(
    expr: ast.expr | None,
    reference: ast.AST,
    parents: dict[ast.AST, ast.AST],
    module_helpers: set[str],
) -> bool:
    """Recognize only canonical helpers that return a non-error event type."""

    def trusted_builder(candidate: ast.expr, result_index: int | None) -> bool:
        return (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id in module_helpers
            and candidate.func.id in NON_ERROR_STREAM_EVENT_BUILDERS
            and NON_ERROR_STREAM_EVENT_BUILDERS[candidate.func.id] == result_index
            and not _has_local_binding(candidate, candidate.func.id, parents)
        )

    if not isinstance(expr, ast.Name):
        return isinstance(expr, ast.expr) and trusted_builder(expr, None)

    scopes = _enclosing_functions(reference, parents)
    if _is_parameter(scopes, expr.id):
        return False

    bindings: list[tuple[ast.expr, int | None]] = []
    assignment_stores: set[int] = set()
    nodes = [node for scope in scopes for node in _scope_nodes(scope)]
    for node in nodes:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == expr.id:
                assignment_stores.add(id(target))
                bindings.append((node.value, None))
            elif isinstance(target, ast.Tuple):
                for index, element in enumerate(target.elts):
                    if isinstance(element, ast.Name) and element.id == expr.id:
                        assignment_stores.add(id(element))
                        bindings.append((node.value, index))

    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == expr.id
        and id(node) not in assignment_stores
        for node in nodes
    ):
        return False
    return bool(bindings) and all(
        trusted_builder(value, result_index) for value, result_index in bindings
    )


def _dict_variants(
    expr: ast.expr,
    reference: ast.AST,
    parents: dict[ast.AST, ast.AST],
    module_helpers: set[str],
    resolving: frozenset[str] = frozenset(),
) -> list[tuple[dict[str, ast.expr], set[str]]]:
    """Resolve effective sensitive fields from dicts and local-name spreads."""
    if isinstance(expr, ast.Await):
        return _dict_variants(expr.value, reference, parents, module_helpers, resolving)
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "create_stream_event"
        and expr.func.id in module_helpers
        and not _has_local_binding(expr, expr.func.id, parents)
    ):
        event_type = _call_argument(expr, 0, "event_type")
        data = _call_argument(expr, 2, "data")
        if event_type is None or data is None:
            return [({}, SENSITIVE_PAYLOAD_FIELDS.copy())]
        variants = _dict_variants(data, reference, parents, module_helpers, resolving)
        for fields, unresolved_fields in variants:
            fields["type"] = event_type
            if isinstance(event_type, ast.Constant):
                unresolved_fields.discard("type")
            else:
                unresolved_fields.add("type")
        return variants
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id in DICT_ERROR_PAYLOAD_BUILDERS
        and expr.func.id in module_helpers
        and not _has_local_binding(expr, expr.func.id, parents)
    ):
        helper = expr.func.id
        message_position = 2 if helper == "_task_error_payload" else 1
        message = _call_argument(expr, message_position, "message")
        if message is None:
            return [({}, SENSITIVE_PAYLOAD_FIELDS.copy())]
        event_type = next(
            (keyword.value for keyword in expr.keywords if keyword.arg == "event_type"),
            ast.Constant(DICT_ERROR_PAYLOAD_BUILDERS[helper]),
        )
        fields = {"type": event_type, "message": message}
        if helper == "create_terminal_task_error_event":
            fields["error"] = message
        unresolved = set() if isinstance(event_type, ast.Constant) else {"type"}
        return [(fields, unresolved)]
    if isinstance(expr, ast.Name):
        if expr.id in resolving:
            return [({}, SENSITIVE_PAYLOAD_FIELDS.copy())]
        scopes = _enclosing_functions(expr, parents)
        assignments = _resolved_assignments(scopes, expr.id, expr, parents)
        if not assignments:
            return [({}, SENSITIVE_PAYLOAD_FIELDS.copy())]
        return [
            variant
            for assignment in assignments
            for variant in _dict_variants(
                assignment,
                reference,
                parents,
                module_helpers,
                resolving | {expr.id},
            )
        ]
    if not isinstance(expr, ast.Dict):
        return [({}, SENSITIVE_PAYLOAD_FIELDS.copy())]

    variants: list[tuple[dict[str, ast.expr], set[str]]] = [({}, set())]
    for key, value in zip(expr.keys, expr.values):
        if key is None:
            spread_variants = _dict_variants(
                value, reference, parents, module_helpers, resolving
            )
            merged_variants = []
            for fields, unresolved_fields in variants:
                for spread_fields, spread_unresolved in spread_variants:
                    merged_fields = fields.copy()
                    merged_unresolved = unresolved_fields.copy()
                    for field in spread_unresolved:
                        merged_fields.pop(field, None)
                    merged_unresolved.update(spread_unresolved)
                    merged_unresolved.difference_update(
                        spread_fields.keys() - spread_unresolved
                    )
                    merged_fields.update(spread_fields)
                    merged_variants.append((merged_fields, merged_unresolved))
            variants = merged_variants
        elif isinstance(key, ast.Constant) and isinstance(key.value, str):
            for fields, unresolved_fields in variants:
                fields[key.value] = value
                unresolved_fields.discard(key.value)
                if key.value == "type" and not isinstance(value, ast.Constant):
                    unresolved_fields.add("type")
        elif not isinstance(key, ast.Constant):
            for fields, unresolved_fields in variants:
                for field in SENSITIVE_PAYLOAD_FIELDS:
                    fields.pop(field, None)
                unresolved_fields.update(SENSITIVE_PAYLOAD_FIELDS)
    return variants


def _error_payload_messages(
    node: ast.Call,
    parents: dict[ast.AST, ast.AST],
    module_helpers: set[str],
) -> list[ast.expr]:
    """Client-visible text fields of a recognized error payload."""
    sink_name = _called_name(node, parents)
    if sink_name not in ERROR_PAYLOAD_SINKS:
        return []
    serialized_sink = sink_name in SERIALIZED_ERROR_PAYLOAD_SINKS
    payload_keyword = "data" if serialized_sink else "message"
    payload_position = (
        1 if sink_name in {"fanout_websocket_text", "send_websocket_text"} else 0
    )
    payload = _call_argument(node, payload_position, payload_keyword)
    if payload is None:
        return []
    argument = _unwrap_serializer(payload, parents)
    messages: list[ast.expr] = []
    if isinstance(argument, ast.Call) and isinstance(argument.func, ast.Name):
        helper = argument.func.id
        if helper == "create_terminal_task_error_event":
            helper_is_trusted = helper in module_helpers and not _has_local_binding(
                argument, helper, parents
            )
            if not helper_is_trusted:
                return [argument]
            message = _call_argument(argument, 1, "message")
            return [message if message is not None else argument]
        if helper != "create_stream_event":
            return []
    if not isinstance(argument, (ast.Call, ast.Dict, ast.Name, ast.Await)):
        return []
    if serialized_sink and isinstance(argument, (ast.Name, ast.Await)):
        # ``ConnectionManager`` serializes payloads already vetted at its
        # public send/broadcast boundary. Keep direct dict/json payloads
        # visible without treating that forwarding layer as a second
        # unresolved producer.
        return []
    for fields, unresolved_fields in _dict_variants(
        argument, node, parents, module_helpers
    ):
        kind = fields.get("type")
        is_error_payload = not _is_known_non_error_event_type(
            kind, node, parents, module_helpers
        ) and (
            (isinstance(kind, ast.Constant) and kind.value in ERROR_PAYLOAD_TYPES)
            or "type" in unresolved_fields
        )
        if not is_error_payload:
            continue
        messages.extend(
            value
            for field in ("message", "error")
            if (value := fields.get(field)) is not None
        )
        if unresolved_fields.intersection({"message", "error"}):
            messages.append(argument)
    return messages


ATTRIBUTE_CALL_RECEIVERS = {
    "broadcast_to_task": {"manager"},
    "dumps": {"json"},
    "send_personal_message": {"manager"},
    "send_text": {"connection", "self.ws", "websocket"},
}


def _attribute_path(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _attribute_path(node.value)
        if owner is not None:
            return f"{owner}.{node.attr}"
    return None


def _called_name(
    node: ast.Call, parents: dict[ast.AST, ast.AST] | None = None
) -> str | None:
    if isinstance(node.func, ast.Name):
        if parents is None:
            return node.func.id
        return _single_name_alias(node.func.id, node, parents)
    if isinstance(node.func, ast.Attribute):
        receivers = ATTRIBUTE_CALL_RECEIVERS.get(node.func.attr)
        if receivers is not None and _attribute_path(node.func.value) in receivers:
            return node.func.attr
    return None


def _is_parameter(scopes: list[ast.AST], name: str) -> bool:
    """A forwarded parameter is vetted at the wrapper's own call sites."""
    for scope in scopes:
        if not isinstance(scope, FUNCTION_NODES):
            continue
        arguments = scope.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *(argument for argument in (arguments.vararg, arguments.kwarg) if argument),
        ):
            if argument.arg == name:
                return True
    return False


def _resolved_assignments(
    scopes: list[ast.AST],
    name: str,
    reference: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> list[ast.expr]:
    """Prefer the active control branch, then the nearest lexical binding."""
    for scope in scopes:
        current = parents.get(reference)
        while current is not None and current is not scope:
            if isinstance(current, (ast.ExceptHandler, ast.If)):
                all_assignments = _local_assignments([current], name)
                if all_assignments:
                    assignments = [
                        value
                        for value in all_assignments
                        if _can_reach_reference(value, reference, parents)
                    ]
                    if not assignments:
                        return [_incoming_parameter()]
                    return _include_unassigned_parameter_path(
                        assignments, scopes, name, reference, current, parents
                    )
            current = parents.get(current)
        all_assignments = _local_assignments([scope], name)
        if all_assignments:
            assignments = [
                value
                for value in all_assignments
                if _can_reach_reference(value, reference, parents)
            ]
            if not assignments:
                return [_incoming_parameter()]
            return _include_unassigned_parameter_path(
                assignments, scopes, name, reference, scope, parents
            )
        if _is_parameter([scope], name):
            return []
    return []


def _include_unassigned_parameter_path(
    assignments: list[ast.expr],
    scopes: list[ast.AST],
    name: str,
    reference: ast.AST,
    binding_scope: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> list[ast.expr]:
    """Keep the incoming binding when every local assignment is conditional."""
    if not all(
        _is_conditional_before_reference(value, reference, binding_scope, parents)
        for value in assignments
    ):
        return assignments
    return [*assignments, _incoming_parameter()]


def _incoming_parameter() -> ast.Call:
    return ast.Call(
        func=ast.Name(id="_incoming_parameter", ctx=ast.Load()),
        args=[],
        keywords=[],
    )


def _unknown_binding(node: ast.AST) -> ast.Call:
    binding = ast.copy_location(
        ast.Call(
            func=ast.Name(id="_unknown_binding", ctx=ast.Load()),
            args=[],
            keywords=[],
        ),
        node,
    )
    binding._binding_node = node  # type: ignore[attr-defined]
    return binding


def _precedes(value: ast.expr, reference: ast.AST) -> bool:
    """Only bindings evaluated before the client-facing sink can reach it."""
    value_position = (getattr(value, "lineno", -1), getattr(value, "col_offset", -1))
    reference_position = (
        getattr(reference, "lineno", -1),
        getattr(reference, "col_offset", -1),
    )
    return value_position < reference_position


def _can_reach_reference(
    value: ast.expr, reference: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    """Whether a binding can reach a reference directly or on a loop back-edge."""
    if _is_in_mutually_exclusive_if_branch(value, reference, parents):
        return False
    return _precedes(value, reference) or _reaches_on_loop_backedge(
        value, reference, parents
    )


def _is_in_mutually_exclusive_if_branch(
    value: ast.expr,
    reference: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Whether a binding and sink sit in opposite branches of one ``if``."""
    current: ast.AST | None = value
    while current is not None:
        parent = parents.get(current)
        if isinstance(parent, ast.If):
            value_branch = _descendant_control_branch(value, parent, parents)
            reference_branch = _descendant_control_branch(reference, parent, parents)
            if (
                value_branch in {"body", "orelse"}
                and reference_branch in {"body", "orelse"}
                and value_branch != reference_branch
            ):
                return True
        current = parent
    return False


def _reaches_on_loop_backedge(
    value: ast.expr, reference: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    """A later binding in a repeated loop region reaches the next iteration."""
    if _precedes(value, reference):
        return False
    binding_node: ast.AST = getattr(value, "_binding_node", value)
    current: ast.AST | None = binding_node
    while current is not None:
        parent = parents.get(current)
        if isinstance(parent, LOOP_NODES):
            binding_branch = _descendant_control_branch(binding_node, parent, parents)
            reference_branch = _descendant_control_branch(reference, parent, parents)
            if (binding_branch == "body" and reference_branch == "body") or (
                isinstance(parent, ast.While)
                and binding_branch in {"body", "test"}
                and reference_branch == "test"
            ):
                return True
        current = parent
    return False


def _is_conditional_before_reference(
    value: ast.expr,
    reference: ast.AST,
    scope: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Whether an assignment can be skipped on the path to ``reference``."""
    if _reaches_on_loop_backedge(value, reference, parents):
        # The binding is unavailable on the first iteration.
        return True
    current: ast.AST | None = value
    while current is not None and current is not scope:
        parent = parents.get(current)
        if isinstance(parent, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            value_branch = _control_branch(current, parent)
            reference_branch = _descendant_control_branch(reference, parent, parents)
            if value_branch in {"body", "orelse"} and value_branch != reference_branch:
                return True
        elif isinstance(parent, ast.match_case):
            if not _is_descendant(reference, parent, parents):
                return True
        elif isinstance(parent, ast.BoolOp):
            if current is not parent.values[0] and not _is_descendant(
                reference, current, parents
            ):
                return True
        elif isinstance(parent, TRY_NODES):
            value_region = _try_region(current, parent, parents)
            reference_region = _try_region(reference, parent, parents)
            if isinstance(value_region, ast.ExceptHandler):
                if value_region is not reference_region:
                    return True
            elif value_region == "body":
                if isinstance(reference_region, ast.ExceptHandler) or (
                    reference_region == "finalbody"
                ):
                    return True
                if reference_region is None and parent.handlers:
                    return True
            elif value_region == "orelse" and reference_region != "orelse":
                return True
        elif isinstance(parent, ast.IfExp):
            if current in (parent.body, parent.orelse) and not _is_descendant(
                reference, current, parents
            ):
                return True
        current = parent
    return False


def _control_branch(
    node: ast.AST, conditional: ast.If | ast.For | ast.AsyncFor | ast.While
) -> str:
    if node in conditional.body:
        return "body"
    if node in conditional.orelse:
        return "orelse"
    return "test"


def _descendant_control_branch(
    node: ast.AST,
    conditional: ast.If | ast.For | ast.AsyncFor | ast.While,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    current = node
    while current is not conditional:
        parent = parents.get(current)
        if parent is conditional:
            return _control_branch(current, conditional)
        if parent is None:
            return None
        current = parent
    return None


def _try_region(
    node: ast.AST,
    conditional: ast.Try | ast.TryStar,
    parents: dict[ast.AST, ast.AST],
) -> str | ast.ExceptHandler | None:
    current = node
    while current is not conditional:
        parent = parents.get(current)
        if parent is conditional:
            if current in conditional.body:
                return "body"
            if current in conditional.orelse:
                return "orelse"
            if current in conditional.finalbody:
                return "finalbody"
            if isinstance(current, ast.ExceptHandler):
                return current
            return None
        if parent is None:
            return None
        current = parent
    return None


def _is_descendant(
    node: ast.AST, ancestor: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    current: ast.AST | None = node
    while current is not None:
        if current is ancestor:
            return True
        current = parents.get(current)
    return False


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Walk one lexical scope without borrowing bindings from its closures."""
    if isinstance(scope, (ast.Module, ast.ClassDef) + FUNCTION_NODES):
        children: Iterator[ast.AST] = iter(scope.body)
    else:
        children = ast.iter_child_nodes(scope)
    for child in children:
        yield from _scope_node(child)


def _scope_node(node: ast.AST) -> Iterator[ast.AST]:
    """Walk a node, entering only the outer-evaluated parts of new scopes."""
    if isinstance(node, ast.Lambda):
        for expression in _definition_time_expressions(node):
            yield from _scope_node(expression)
        return
    yield node
    if isinstance(node, FUNCTION_NODES + (ast.ClassDef,)):
        for expression in _definition_time_expressions(node):
            yield from _scope_node(expression)
        return
    for child in ast.iter_child_nodes(node):
        yield from _scope_node(child)


def _definition_time_expressions(node: ast.AST) -> Iterator[ast.expr]:
    if isinstance(node, FUNCTION_NODES + (ast.Lambda,)):
        if isinstance(node, FUNCTION_NODES):
            yield from node.decorator_list
        arguments = node.args
        yield from arguments.defaults
        yield from (default for default in arguments.kw_defaults if default is not None)
        if isinstance(node, FUNCTION_NODES):
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *(arg for arg in (arguments.vararg, arguments.kwarg) if arg),
            ):
                if argument.annotation is not None:
                    yield argument.annotation
            if node.returns is not None:
                yield node.returns
    elif isinstance(node, ast.ClassDef):
        yield from node.decorator_list
        yield from node.bases
        yield from (keyword.value for keyword in node.keywords)


def _target_values(target: ast.expr, value: ast.expr, name: str) -> list[ast.expr]:
    if isinstance(target, ast.Name):
        return [value] if target.id == name else []
    if isinstance(target, (ast.List, ast.Tuple)):
        if isinstance(value, (ast.List, ast.Tuple)) and len(target.elts) == len(
            value.elts
        ):
            values: list[ast.expr] = []
            for child_target, child_value in zip(target.elts, value.elts):
                values.extend(_target_values(child_target, child_value, name))
            return values
        if any(_target_contains_name(child, name) for child in target.elts):
            return [value]
    if isinstance(target, ast.Starred) and _target_contains_name(target.value, name):
        return [value]
    return []


def _target_contains_name(target: ast.expr, name: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, ast.Starred):
        return _target_contains_name(target.value, name)
    if isinstance(target, (ast.List, ast.Tuple)):
        return any(_target_contains_name(child, name) for child in target.elts)
    return False


def _local_assignments(scopes: list[ast.AST], name: str) -> list[ast.expr]:
    """Values bound to ``name`` in the given lexical scopes."""
    values: list[ast.expr] = []
    for scope in scopes:
        if isinstance(scope, ast.ExceptHandler) and scope.name == name:
            values.append(_unknown_binding(scope))
        for node in _scope_nodes(scope):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    values.extend(_target_values(target, node.value, name))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                values.extend(_target_values(node.target, node.value, name))
            elif isinstance(node, ast.AugAssign):
                if _target_contains_name(node.target, name):
                    binding = ast.copy_location(
                        ast.Call(
                            func=ast.Name(id="_augmented_assignment", ctx=ast.Load()),
                            args=[node.value],
                            keywords=[],
                        ),
                        node,
                    )
                    binding._binding_node = node  # type: ignore[attr-defined]
                    values.append(binding)
            elif isinstance(node, ast.NamedExpr):
                values.extend(_target_values(node.target, node.value, name))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                if _target_contains_name(node.target, name):
                    values.append(_unknown_binding(node.target))
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
                if node.name == name:
                    values.append(_unknown_binding(node))
            elif isinstance(node, ast.MatchMapping) and node.rest == name:
                values.append(_unknown_binding(node))
            elif (
                isinstance(node, ast.withitem)
                and node.optional_vars is not None
                and _target_contains_name(node.optional_vars, name)
            ):
                values.append(_unknown_binding(node.optional_vars))
            elif isinstance(node, ast.ExceptHandler) and node.name == name:
                values.append(_unknown_binding(node))
            elif isinstance(node, FUNCTION_NODES + (ast.ClassDef,)):
                if node.name == name:
                    values.append(_unknown_binding(node))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if any(
                    alias.name == "*"
                    or (alias.asname or alias.name.split(".")[0]) == name
                    for alias in node.names
                ):
                    values.append(_unknown_binding(node))
    return values


def _single_name_alias(
    name: str,
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    visited: frozenset[str] = frozenset(),
) -> str:
    """Resolve recursive aliases from definitions that can reach the call."""
    if name in visited:
        return name
    visited = visited | {name}
    scopes = _enclosing_functions(node, parents)
    for index, scope in enumerate(scopes):
        assignments = _reaching_alias_assignments(
            scope,
            name,
            node,
            parents,
            deferred=index > 0,
            prune_overwritten=index == 0,
        )
        if not assignments:
            if _is_parameter([scope], name):
                return name
            continue
        return _resolve_alias_values(name, assignments, parents, visited)
    current: ast.AST | None = node
    while current is not None and not isinstance(current, ast.Module):
        current = parents.get(current)
    if isinstance(current, ast.Module):
        assignments = _reaching_alias_assignments(
            current,
            name,
            node,
            parents,
            deferred=bool(scopes),
            prune_overwritten=True,
        )
        if assignments:
            return _resolve_alias_values(name, assignments, parents, visited)
    return name


def _reaching_alias_assignments(
    scope: ast.AST,
    name: str,
    reference: ast.AST,
    parents: dict[ast.AST, ast.AST],
    *,
    deferred: bool,
    prune_overwritten: bool,
) -> list[ast.expr]:
    assignments = _local_assignments([scope], name)
    if not deferred:
        assignments = [
            assignment
            for assignment in assignments
            if _can_reach_reference(assignment, reference, parents)
        ]
    direct = [
        assignment
        for assignment in assignments
        if _is_direct_scope_binding(assignment, scope, parents)
    ]
    direct_overwrites = [
        assignment
        for assignment in direct
        if not (isinstance(assignment, ast.Name) and assignment.id == name)
    ]
    if direct_overwrites and prune_overwritten:
        last_direct = max(
            direct_overwrites,
            key=lambda assignment: (
                getattr(assignment, "lineno", -1),
                getattr(assignment, "col_offset", -1),
            ),
        )
        assignments = [
            assignment
            for assignment in assignments
            if assignment is last_direct or _precedes(last_direct, assignment)
        ]
    return assignments


def _is_direct_scope_binding(
    binding: ast.expr,
    scope: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    current: ast.AST = getattr(binding, "_binding_node", binding)
    while current in parents and parents[current] is not scope:
        current = parents[current]
    return parents.get(current) is scope and isinstance(
        current,
        (
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Expr,
            ast.Import,
            ast.ImportFrom,
            ast.ClassDef,
        )
        + FUNCTION_NODES,
    )


def _resolve_alias_values(
    name: str,
    assignments: list[ast.expr],
    parents: dict[ast.AST, ast.AST],
    visited: frozenset[str],
) -> str:
    """Resolve direct single-name aliases, leaving ambiguous shapes unresolved.

    This is deliberately not a general alias engine: ambiguous or non-name
    values keep the original name and may remain outside the recognized grammar.
    """
    resolved: list[str] = []
    for assignment in assignments:
        if not isinstance(assignment, ast.Name) or assignment.id == name:
            continue
        alias = _single_name_alias(assignment.id, assignment, parents, visited)
        if alias in PRODUCERS:
            return alias
        resolved.append(alias)
    if len(assignments) == 1 and len(resolved) == 1:
        return resolved[0]
    return name


def _module_bindings(module: ast.Module, name: str) -> list[ast.AST]:
    bindings: list[ast.AST] = []
    for node in _scope_nodes(module):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bindings.extend(_target_values(target, node.value, name))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bindings.extend(_target_values(node.target, node.value, name))
        elif isinstance(node, ast.AugAssign) and _target_contains_name(
            node.target, name
        ):
            bindings.append(_unknown_binding(node))
        elif isinstance(node, ast.NamedExpr):
            bindings.extend(_target_values(node.target, node.value, name))
        elif isinstance(node, (ast.For, ast.AsyncFor)) and _target_contains_name(
            node.target, name
        ):
            bindings.append(_unknown_binding(node.target))
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name:
            bindings.append(_unknown_binding(node))
        elif isinstance(node, ast.MatchMapping) and node.rest == name:
            bindings.append(_unknown_binding(node))
        elif (
            isinstance(node, ast.withitem)
            and node.optional_vars is not None
            and _target_contains_name(node.optional_vars, name)
        ):
            bindings.append(_unknown_binding(node.optional_vars))
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            bindings.append(_unknown_binding(node))
        elif isinstance(node, FUNCTION_NODES + (ast.ClassDef,)) and node.name == name:
            bindings.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            alias.name == "*" or (alias.asname or alias.name.split(".")[0]) == name
            for alias in node.names
        ):
            bindings.append(node)
    return bindings


def _is_unshadowed_module_name(
    node: ast.Name,
    parents: dict[ast.AST, ast.AST],
    expected: type[ast.AST],
) -> bool:
    scopes = _enclosing_functions(node, parents)
    if (
        _is_comprehension_binding(node, node.id, parents)
        or _is_parameter(scopes, node.id)
        or any(_local_assignments([scope], node.id) for scope in scopes)
    ):
        return False
    module: ast.AST = node
    while module in parents:
        module = parents[module]
    bindings = (
        _module_bindings(module, node.id) if isinstance(module, ast.Module) else []
    )
    if not bindings:
        return True
    if (
        len(bindings) != 1
        or not isinstance(bindings[0], expected)
        or not _is_direct_module_binding(bindings[0], module, parents)
    ):
        return False
    if expected is ast.Dict:
        table = bindings[0]
        assert isinstance(table, ast.Dict)
        return all(
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            for key, value in zip(table.keys, table.values)
        )
    return True


def _is_direct_module_binding(
    binding: ast.AST,
    module: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    if isinstance(binding, FUNCTION_NODES + (ast.ClassDef,)):
        return parents.get(binding) is module
    statement = parents.get(binding)
    return isinstance(statement, (ast.Assign, ast.AnnAssign)) and (
        parents.get(statement) is module
    )


def _is_comprehension_binding(
    node: ast.AST, name: str, parents: dict[ast.AST, ast.AST]
) -> bool:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, COMPREHENSION_NODES) and any(
            _target_contains_name(generator.target, name)
            for generator in current.generators
        ):
            return True
        current = parents.get(current)
    return False


def _has_nested_global_rebinding(tree: ast.Module, name: str) -> bool:
    """Whether a function writes the module binding through ``global``."""
    for scope in ast.walk(tree):
        if not isinstance(scope, FUNCTION_NODES):
            continue
        nodes = list(_scope_nodes(scope))
        if any(
            isinstance(node, ast.Global) and name in node.names for node in nodes
        ) and _local_assignments([scope], name):
            return True
    return False


def _trusted_module_helpers(tree: ast.Module) -> set[str]:
    """Helpers with one real module implementation and no competing binding."""
    parents = _parents(tree)
    expected = {
        *DICT_ERROR_PAYLOAD_BUILDERS,
        *NON_ERROR_STREAM_EVENT_BUILDERS,
        "create_stream_event",
    }
    overload_bindings = [
        (node, alias)
        for node in _scope_nodes(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if (alias.asname or alias.name.split(".")[0]) == "overload"
    ]
    canonical_overload = (
        len(overload_bindings) == 1
        and isinstance(overload_bindings[0][0], ast.ImportFrom)
        and overload_bindings[0][0].level == 0
        and overload_bindings[0][0].module == "typing"
        and overload_bindings[0][1].name == "overload"
        and overload_bindings[0][1].asname in {None, "overload"}
    )
    trusted: set[str] = set()
    for name in expected:
        bindings = _module_bindings(tree, name)
        definitions = [node for node in bindings if isinstance(node, FUNCTION_NODES)]
        overload_stubs = [
            node
            for node in definitions
            if canonical_overload
            and len(node.decorator_list) == 1
            and isinstance(node.decorator_list[0], ast.Name)
            and node.decorator_list[0].id == "overload"
        ]
        implementations = [node for node in definitions if not node.decorator_list]
        has_other_binding = len(bindings) != len(definitions)
        has_unsafe_decorator = len(definitions) != len(overload_stubs) + len(
            implementations
        )
        if (
            len(implementations) == 1
            and not has_other_binding
            and not has_unsafe_decorator
            and all(
                _is_direct_module_binding(node, tree, parents) for node in definitions
            )
            and not _has_nested_global_rebinding(tree, name)
        ):
            trusted.add(name)
    return trusted


def _trusted_imported_safe_constants(tree: ast.Module) -> set[str]:
    """Constants with one exact relative import and no competing binding."""
    trusted: set[str] = set()
    for name in SAFE_MESSAGE_CONSTANTS:
        bindings = _module_bindings(tree, name)
        if len(bindings) != 1 or not isinstance(bindings[0], ast.ImportFrom):
            continue
        statement = bindings[0]
        imported_names = [
            alias.name
            for alias in statement.names
            if (alias.asname or alias.name) == name
        ]
        if (
            statement in tree.body
            and statement.level == 2
            and statement.module == "services.client_error_messages"
            and imported_names == [name]
            and not _has_nested_global_rebinding(tree, name)
        ):
            trusted.add(name)
    return trusted


def _is_client_safe(
    expr: ast.expr,
    parents: dict[ast.AST, ast.AST],
    imported_safe_constants: set[str] | frozenset[str] = frozenset(),
) -> bool:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return True
    if (
        isinstance(expr, ast.Name)
        and expr.id in imported_safe_constants
        and not _has_local_binding(expr, expr.id, parents)
    ):
        return True
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        if expr.func.id == "client_safe_error_message" and _is_unshadowed_module_name(
            expr.func, parents, ast.FunctionDef
        ):
            fallback = next(
                (
                    keyword.value
                    for keyword in expr.keywords
                    if keyword.arg == "fallback"
                ),
                None,
            )
            return fallback is None or _is_client_safe(
                fallback, parents, imported_safe_constants
            )
        if expr.func.id == "client_safe_task_command_failure" and (
            _is_unshadowed_module_name(expr.func, parents, ast.FunctionDef)
        ):
            # The prefix argument must be attribute access on server state
            # (``command.kind``), never a literal or a bare name a caller
            # could point at untrusted text.
            return bool(expr.args) and isinstance(expr.args[0], ast.Attribute)
        return False
    # `_TURN_REJECTION_MESSAGES.get(reason, "<literal>")` - a curated table.
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        return (
            expr.func.attr == "get"
            and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id == "_TURN_REJECTION_MESSAGES"
            and _is_unshadowed_module_name(expr.func.value, parents, ast.Dict)
            and len(expr.args) == 2
            and not expr.keywords
            and isinstance(expr.args[1], ast.Constant)
            and isinstance(expr.args[1].value, str)
        )
    if isinstance(expr, ast.BoolOp):
        return all(
            _is_client_safe(value, parents, imported_safe_constants)
            for value in expr.values
        )
    if isinstance(expr, ast.IfExp):
        return _is_client_safe(
            expr.body, parents, imported_safe_constants
        ) and _is_client_safe(expr.orelse, parents, imported_safe_constants)
    return False


class _ScanResult(NamedTuple):
    offenders: list[str]
    producers: int
    error_payloads: int
    used_allowlist: set[_RawMessageAllowance]


def _allowlist_key(
    candidate: ast.expr,
    sink: ast.Call,
    parents: dict[ast.AST, ast.AST],
) -> _RawMessageAllowance:
    functions = _enclosing_functions(candidate, parents)
    qualname = ".".join(
        reversed(
            [scope.name for scope in functions if isinstance(scope, FUNCTION_NODES)]
        )
    )
    candidate_handler = _enclosing_exception_handler(candidate, parents)
    sink_handler = _enclosing_exception_handler(sink, parents)
    handler_name = "<mismatched-except>"
    if candidate_handler is sink_handler and candidate_handler is not None:
        handler_name = (
            ast.unparse(candidate_handler.type)
            if candidate_handler.type is not None
            else "BaseException"
        )
        if not _handler_target_is_used(candidate_handler, candidate):
            handler_name = "<unbound-except>"
    return _RawMessageAllowance(
        qualname or "<module>", handler_name, ast.unparse(candidate)
    )


def _enclosing_exception_handler(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.ExceptHandler | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.ExceptHandler):
            return current
        current = parents.get(current)
    return None


def _handler_target_is_used(handler: ast.ExceptHandler, candidate: ast.expr) -> bool:
    return isinstance(handler.name, str) and any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == handler.name
        for node in ast.walk(candidate)
    )


def _inside_unsupported_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.Lambda, ast.ClassDef)):
            return True
        if isinstance(current, FUNCTION_NODES):
            return False
        current = parents.get(current)
    return False


def _is_non_none_test(test: ast.expr, name: str) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == name
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def _is_guarded_non_none(
    name: str,
    reference: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Whether a direct ``if``/``assert`` proves a sink value is not ``None``."""
    current = reference
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.If) and current in parent.body:
            if _is_non_none_test(parent.test, name):
                return True
        body = getattr(parent, "body", None)
        if isinstance(body, list) and current in body:
            index = body.index(current)
            if index > 0 and isinstance(body[index - 1], ast.Assert):
                if _is_non_none_test(body[index - 1].test, name):
                    return True
        current = parent
    return False


def _is_nullish_candidate(candidate: ast.expr) -> bool:
    return (
        isinstance(candidate, ast.Constant)
        and candidate.value is None
        or isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == "_incoming_parameter"
    )


def _scan(tree: ast.Module) -> _ScanResult:
    """The one copy of the sweep's recognition logic.

    Both the production sweep and the snippet-based regression tests run
    this same function, so a change to the analysis cannot pass the snippet
    tests while silently not applying to the real module (or vice versa).
    """
    parents = _parents(tree)
    module_helpers = _trusted_module_helpers(tree)
    imported_safe_constants = _trusted_imported_safe_constants(tree)
    producers = 0
    error_payloads = 0
    offenders: list[str] = []
    used_allowlist: set[_RawMessageAllowance] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node, parents)
        is_producer = name in PRODUCERS
        if is_producer:
            message = _message_expression(node, PRODUCERS[name])
            expressions = [message] if message is not None else []
        else:
            # The error bubble renders in the same conversation as the
            # rejection ack, so it is the same disclosure surface.
            expressions = _error_payload_messages(node, parents, module_helpers)
            name = f"{name}(error payload)"
        if not expressions:
            continue
        if is_producer:
            producers += 1
        else:
            error_payloads += len(expressions)
        if _inside_unsupported_scope(node, parents):
            offenders.append(f"{name}:{node.lineno} is inside an unsupported scope")
            continue
        scopes = _enclosing_functions(node, parents)
        for expr in expressions:
            if isinstance(expr, ast.Name):
                candidates = (
                    [_unknown_binding(expr)]
                    if _is_comprehension_binding(expr, expr.id, parents)
                    else _resolved_assignments(scopes, expr.id, node, parents)
                )
            else:
                candidates = [expr]
            if _is_client_safe(expr, parents, imported_safe_constants):
                continue
            if isinstance(expr, ast.Name) and _is_guarded_non_none(
                expr.id, node, parents
            ):
                non_null_candidates = [
                    candidate
                    for candidate in candidates
                    if not _is_nullish_candidate(candidate)
                ]
                if non_null_candidates and all(
                    _is_client_safe(candidate, parents, imported_safe_constants)
                    for candidate in non_null_candidates
                ):
                    candidates = non_null_candidates
            if not candidates:
                # Only a name nothing rebinds is a genuinely forwarded parameter,
                # vetted at the wrapper's own call sites. Every supported rebinding
                # form lands in candidates instead of taking this short-circuit.
                if isinstance(expr, ast.Name) and _is_parameter(scopes, expr.id):
                    continue
                offenders.append(f"{name}:{node.lineno} passes an unresolvable name")
                continue
            for candidate in candidates:
                if _is_client_safe(candidate, parents, imported_safe_constants):
                    continue
                key = _allowlist_key(candidate, node, parents)
                if key in ALLOWED_RAW_MESSAGES:
                    used_allowlist.add(key)
                    continue
                offenders.append(
                    f"{name}:{node.lineno} may send {ast.unparse(candidate)!r}"
                )
    return _ScanResult(offenders, producers, error_payloads, used_allowlist)


def guard_offenders(source: str) -> list[str]:
    return _scan(ast.parse(source)).offenders
