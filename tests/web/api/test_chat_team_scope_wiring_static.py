"""Static, source-level guards for the part of ``chat.py``'s team-governed
wiring that a fast unit test cannot reach without a real database session,
task row, and workforce runtime: the three places
``AgentServiceManager`` derives ``agent_creator_user_id`` for a
team-governed run, and the two ``create_default_tools`` call sites that
must forward both ``agent_creator_user_id`` and ``declared_knowledge_bases``
through.

These are not behavioural pins -- they cannot prove the values resolve to
the *correct* agent at runtime (the tests against the shared
``AgentRuntimeFields`` construction site and against
``create_default_tools``'s own forwarding into ``WebToolConfig`` do that,
see ``test_agent_runtime_fields_creator.py`` and
``test_create_default_tools_team_scope_forwarding.py``). What this file
closes is a narrower, cheaper gap: a derivation point quietly hardcoded to
``None``, or a call site dropping one of the two keywords, would not raise
anywhere at runtime, and no test that never drives this exact code path
would notice. Reading the source with the ``ast`` module is what lets this
be checked without constructing the session/task/workforce machinery that
driving the real code path would require.

This file's own blind spot -- a call site that swaps in a *wrong but
non-``None``* value (the running user's id instead of the agent's
creator, or a hardcoded empty list instead of the stored declaration) --
is closed behaviourally instead, in
``test_chat_team_scope_wiring_behavioral.py``: it drives both call sites
with a fake ``TaskSetupSnapshot`` and a spy on ``create_default_tools``,
and asserts the forwarded values equal the snapshot's own agent fields.
"""

from __future__ import annotations

import ast
import inspect

import xagent.web.services.agent_service_manager as chat_module


def _tree() -> ast.Module:
    return ast.parse(inspect.getsource(chat_module))


def _is_hardcoded_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _create_default_tools_call_sites() -> list[ast.Call]:
    calls = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == "create_default_tools":
            calls.append(node)
    return calls


def test_there_are_exactly_two_create_default_tools_call_sites() -> None:
    """A third call site would need this whole file's static guards
    revisited (it might not thread the two new values at all), the same
    reason the construction-site count is pinned elsewhere in this design.
    """
    calls = _create_default_tools_call_sites()
    assert len(calls) == 2, [c.lineno for c in calls]


def test_exactly_two_local_variable_derivations_of_agent_creator_user_id() -> None:
    """Two of the three places ``AgentServiceManager`` derives this value
    are plain local-variable assignments (the snapshot branch and the
    live-ORM-row branch inside ``_build_tools_for_task``); the third is
    folded directly into the second ``create_default_tools`` call's own
    keyword argument and is covered by the call-site check above instead.
    """
    assigns = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "agent_creator_user_id"
    ]
    assert len(assigns) == 2, [a.lineno for a in assigns]
    for assign in assigns:
        assert not _is_hardcoded_none(assign.value), (
            f"line {assign.lineno} hardcodes agent_creator_user_id to None"
        )
