"""Static resolution of module-scope string constants and their aliases.

Shared by any architecture or static guard that needs to recognize a known
string value under whichever spelling a source file happens to use for it
-- a bare literal, a module-scope constant, an imported (and possibly
aliased) name, or an attribute access naming one of the values a caller
already knows about -- rather than only the inline ``ast.Constant``
spelling. Extracted from ``tests/architecture/test_architecture_guards.py``
so ``tests/web/services/test_supersede_static_guards.py`` could stop
importing these two functions as private names across a package boundary;
the logic and docstrings below are unchanged from that original, except
that the message-type-specific global they both closed over,
``_MESSAGE_TYPE_CONSTANTS``, is now a ``known_constants`` argument each
caller supplies -- this module has no built-in knowledge of any particular
constant's name or value.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator


def _module_level_statements(tree: ast.Module) -> Iterator[ast.AST]:
    """Statements in module scope, descending through module-level ``if``
    and ``try`` blocks -- a constant defined under ``if TYPE_CHECKING:``
    or in a ``try``/``except ImportError`` fallback is still a
    module-scope binding. Function and class bodies are deliberately not
    descended into: a local ``x = "question"`` in some unrelated function
    must not make every ``x`` in the file resolve to that value.
    """
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, ast.If):
            pending.extend(node.body)
            pending.extend(node.orelse)
        elif isinstance(node, (ast.Try, ast.TryStar)):
            pending.extend(node.body)
            pending.extend(node.orelse)
            pending.extend(node.finalbody)
            for handler in node.handlers:
                pending.extend(handler.body)


def string_constant_bindings(
    tree: ast.Module, known_constants: dict[str, str]
) -> dict[str, set[str]]:
    """Module-scope names mapped to every string value they are ever bound
    to, so a guard can recognize ``.update({message_type: SUPERSEDED})``
    as the same write as ``.update({message_type: "question_superseded"})``.

    Four carrier shapes are resolved:

    * a module-scope assignment to a string literal, plain or annotated
      (``S = "question_superseded"``, ``S: str = "question_superseded"``);
    * ``from ... import SUPERSEDED_MESSAGE_TYPE``, including
      ``... as S`` -- the imported name takes the value ``known_constants``
      says it holds, not a respelling;
    * a module-scope assignment from an attribute of one of those
      constants (``S = chat_history_service.SUPERSEDED_MESSAGE_TYPE``);
    * a chain of module-scope name-to-name assignments over any of the
      above (``A = S``; ``B = A``), resolved to a fixed point. Value sets
      only grow and the name set is finite, so a cycle (``A = B``;
      ``B = A``) terminates instead of looping.

    A name rebound at module scope keeps *every* value it was ever bound
    to rather than only the last one. "The last one" is not a defined
    notion here: ``_module_level_statements`` drains its work list with
    ``pop()``, so statements arrive last-in-first-out and not in source
    order. Keeping the union is the only well-defined answer, and it is
    also the safe one for something that feeds a ban.

    Shapes this does not resolve. Values built by anything other than a
    plain literal or one of the forms above -- an f-string,
    ``.format()``, ``"".join(...)``, an augmented assignment
    (``B += "..."``), ``getattr(module, "NAME")`` -- plus tuple-unpacking
    targets (``A, B = "x", "y"``), names arriving through
    ``from ... import *``, imports made inside a module-level ``with``
    block, constants defined in a class body, and a value reaching the
    write as a dict key or ``**`` carrier rather than as the mapped
    value. None of these is resolved; the table therefore stays a ban on
    the spellings it can see, not a proof that no other spelling exists.
    """
    bindings: dict[str, set[str]] = {}
    aliases: list[tuple[str, str]] = []

    def bind(name: str, value: str) -> None:
        bindings.setdefault(name, set()).add(value)

    for node in _module_level_statements(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in known_constants:
                    bind(alias.asname or alias.name, known_constants[alias.name])
            continue
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if node.value is None:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not names:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for name in names:
                bind(name, value.value)
        elif isinstance(value, ast.Name):
            aliases.extend((name, value.id) for name in names)
        elif isinstance(value, ast.Attribute) and value.attr in known_constants:
            for name in names:
                bind(name, known_constants[value.attr])

    changed = True
    while changed:
        changed = False
        for target, source in aliases:
            values = bindings.get(source)
            if values and not values <= bindings.get(target, set()):
                bindings.setdefault(target, set()).update(values)
                changed = True
    return bindings


def string_values(
    node: ast.AST, bindings: dict[str, set[str]], known_constants: dict[str, str]
) -> set[str]:
    """Every string value this expression can denote: an inline literal, a
    module-scope name bound to one, or ``<module>.NAME`` for one of
    ``known_constants``. Anything else resolves to nothing, so a guard
    built on this stays a ban on known values rather than a guess."""
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    if isinstance(node, ast.Name):
        return set(bindings.get(node.id, ()))
    if isinstance(node, ast.Attribute) and node.attr in known_constants:
        return {known_constants[node.attr]}
    return set()
