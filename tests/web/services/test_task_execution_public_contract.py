"""Pin the cross-repo public contract of ``xagent.web.services.task_execution``.

Background:
    #2318 (``fe7ae92d``) moved ``execute_resume_background`` out of the
    WebSocket API module and replaced its ``delivery_websocket`` /
    ``delivery_client_message_id`` keyword parameters with a single
    ``delivery_notifier``. The downstream SaaS deployment
    (xorbitsai/xagent-saas, ``external_input_dispatch``) calls this function
    directly and still passed the removed keywords, so after the next
    submodule bump every widget/SDK form-answer resume failed on entry with a
    ``TypeError`` (issue #2390). Nothing in this repo exercised the keyword
    set: every existing test replaces the function with an ``AsyncMock`` or a
    ``**_kwargs`` stub.

What this test pins:

    * The module declares its cross-repo entrypoints in ``__all__``, and
      every declared name resolves.
    * The ordered parameter names, kinds, and defaults of
      ``execute_task_background`` and ``execute_resume_background``.
    * The exact keyword set the downstream resume call site passes binds to
      ``execute_resume_background``, and the two removed keywords do not.
    * The ``BackgroundTaskManager`` surface the downstream uses: the resume
      reservation methods it calls (``reserve_resume``,
      ``register_reserved_resume``, ``release_resume_reservation``), the
      ``running_tasks`` mapping it reads, and the module-level
      ``background_task_manager`` singleton.

Changing any of these is a deliberate, reviewed act: update the pinned
values here and the downstream consumer in the same release.
"""

import inspect
from collections.abc import Callable
from typing import Any

import pytest

from xagent.core.execution_scope import EXECUTION_SCOPE_NOT_PROVIDED
from xagent.web.services import task_execution
from xagent.web.services.task_execution import (
    BackgroundTaskManager,
    background_task_manager,
    execute_resume_background,
    execute_task_background,
)

_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
_KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
_KEYWORD_BINDABLE = (_POSITIONAL_OR_KEYWORD, _KEYWORD_ONLY)
_EMPTY = inspect.Parameter.empty

# Keyword set passed by xorbitsai/xagent-saas
# ``external_input_dispatch._execute_external_resume_background`` at both of
# its call sites (stranded-delivery resume and normal continuation).
DOWNSTREAM_RESUME_KWARGS: dict[str, Any] = {
    "task_id": 1,
    "agent_service": object(),
    "task_owner_user_id": 7,
    "previous_task": None,
    "pending_user_message": None,
    "delivery_turn_id": "turn-1",
    "delivery_already_dispatched": True,
    "expected_run_id": "run-1",
    "preacquired_lease": None,
    "preacquired_heartbeat_stop": None,
    "preacquired_heartbeat_task": None,
    "preacquired_prior_status": None,
}

# Keywords #2318 removed. Listed so the removal stays a documented decision
# rather than something a future reader has to reconstruct from git history.
# A compat shim that re-accepts these exact names is a deliberate contract
# change: relax ``test_removed_resume_kwargs_do_not_bind`` and update
# ``EXPECTED_RESUME_PARAMETERS`` in the same commit as the shim.
REMOVED_RESUME_KWARGS = ("delivery_websocket", "delivery_client_message_id")

EXPECTED_RESUME_PARAMETERS: list[tuple[str, Any]] = [
    ("task_id", _EMPTY),
    ("agent_service", _EMPTY),
    ("task_owner_user_id", _EMPTY),
    ("previous_task", None),
    ("pending_user_message", None),
    ("delivery_turn_id", None),
    ("delivery_already_dispatched", False),
    ("delivery_notifier", None),
    ("expected_run_id", None),
    ("resolved_execution_scope", EXECUTION_SCOPE_NOT_PROVIDED),
    ("preacquired_lease", None),
    ("preacquired_heartbeat_stop", None),
    ("preacquired_heartbeat_task", None),
    ("preacquired_prior_status", None),
]

EXPECTED_EXECUTE_PARAMETERS: list[tuple[str, Any]] = [
    ("task_id", _EMPTY),
    ("user_message", _EMPTY),
    ("context", _EMPTY),
    ("agent_manager", _EMPTY),
    ("task_owner_user_id", _EMPTY),
    ("before_message_id", None),
    ("llm_user_message", None),
    ("task_setup_snapshot", None),
    ("expected_run_id", None),
    ("task_lease", None),
    ("resolved_execution_scope", EXECUTION_SCOPE_NOT_PROVIDED),
    ("mcp_runtime_authorization_policy", None),
]

# ``BackgroundTaskManager`` methods the downstream calls, as
# ``(name, kind, default)`` triples. Kind matters here: the downstream passes
# ``task_id`` and ``task`` positionally and ``run_id`` by keyword.
EXPECTED_MANAGER_METHODS: dict[str, list[tuple[str, Any, Any]]] = {
    "reserve_resume": [("task_id", _POSITIONAL_OR_KEYWORD, _EMPTY)],
    "register_reserved_resume": [
        ("task_id", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("task", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("run_id", _KEYWORD_ONLY, _EMPTY),
    ],
    "release_resume_reservation": [("task_id", _POSITIONAL_OR_KEYWORD, _EMPTY)],
}

PUBLIC_ENTRYPOINTS = (
    "BackgroundTaskManager",
    "background_task_manager",
    "execute_resume_background",
    "execute_task_background",
)


def _signature(func: Callable[..., object]) -> inspect.Signature:
    # ``follow_wrapped=False``: a ``functools.wraps`` decorator that narrows the
    # accepted keywords would otherwise report the wrapped function's
    # signature and let a broken contract pass.
    return inspect.signature(func, follow_wrapped=False)


def _keyword_bindable_parameters(
    func: Callable[..., object],
) -> list[tuple[str, Any]]:
    """Return ordered ``(name, default)`` pairs.

    Downstream callers pass every argument by keyword, so a parameter may be
    positional-or-keyword or keyword-only, but never positional-only or
    ``*args`` / ``**kwargs``.
    """
    contract: list[tuple[str, Any]] = []
    for parameter in _signature(func).parameters.values():
        assert parameter.kind in _KEYWORD_BINDABLE, (
            f"{func.__name__}.{parameter.name} must stay keyword-bindable; "
            f"got {parameter.kind.name}"
        )
        contract.append((parameter.name, parameter.default))
    return contract


def test_module_declares_cross_repo_entrypoints() -> None:
    declared = getattr(task_execution, "__all__", None)
    assert declared is not None, "task_execution must declare __all__"
    # A superset check: adding an in-repo public name to ``__all__`` is not a
    # cross-repo contract change; dropping one of these is.
    assert set(PUBLIC_ENTRYPOINTS) <= set(declared)
    for name in declared:
        assert hasattr(task_execution, name), f"__all__ names missing attribute {name}"


def test_module_docstring_names_downstream_consumer() -> None:
    docstring = task_execution.__doc__ or ""
    assert "xorbitsai/xagent-saas" in docstring
    for name in PUBLIC_ENTRYPOINTS:
        assert name in docstring, f"module docstring must name {name}"


def test_execute_resume_background_parameter_contract() -> None:
    assert inspect.iscoroutinefunction(execute_resume_background)
    assert (
        _keyword_bindable_parameters(execute_resume_background)
        == EXPECTED_RESUME_PARAMETERS
    )


def test_execute_task_background_parameter_contract() -> None:
    assert inspect.iscoroutinefunction(execute_task_background)
    assert (
        _keyword_bindable_parameters(execute_task_background)
        == EXPECTED_EXECUTE_PARAMETERS
    )


def test_downstream_resume_call_shape_binds() -> None:
    """The downstream keyword set binds, and defaults cover every other parameter."""
    signature = _signature(execute_resume_background)
    bound = signature.bind(**DOWNSTREAM_RESUME_KWARGS)
    bound.apply_defaults()
    expected_names = [name for name, _default in EXPECTED_RESUME_PARAMETERS]
    assert list(bound.arguments) == expected_names
    for name, default in EXPECTED_RESUME_PARAMETERS:
        if name not in DOWNSTREAM_RESUME_KWARGS:
            assert bound.arguments[name] is default, name


@pytest.mark.parametrize("removed", REMOVED_RESUME_KWARGS)
def test_removed_resume_kwargs_do_not_bind(removed: str) -> None:
    signature = _signature(execute_resume_background)
    with pytest.raises(TypeError, match=removed):
        signature.bind(**DOWNSTREAM_RESUME_KWARGS, **{removed: None})


def test_background_task_manager_surface() -> None:
    assert isinstance(background_task_manager, BackgroundTaskManager)
    manager = BackgroundTaskManager()
    # The downstream reads ``running_tasks.get(task_id)`` for ``previous_task``.
    assert isinstance(manager.running_tasks, dict)
    assert manager.running_tasks == {}


@pytest.mark.parametrize("method_name", sorted(EXPECTED_MANAGER_METHODS))
def test_background_task_manager_method_contract(method_name: str) -> None:
    method = getattr(BackgroundTaskManager(), method_name)
    parameters = [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in _signature(method).parameters.values()
    ]
    assert parameters == EXPECTED_MANAGER_METHODS[method_name]


def test_downstream_register_reserved_resume_call_shape_binds() -> None:
    """Mirror of ``resume_manager.register_reserved_resume(task_id, bg_task, run_id=...)``."""
    signature = _signature(BackgroundTaskManager().register_reserved_resume)
    bound = signature.bind(1, object(), run_id="run-1")
    assert list(bound.arguments) == ["task_id", "task", "run_id"]
    with pytest.raises(TypeError):
        signature.bind(1, object(), "run-1")
