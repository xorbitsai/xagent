from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from xagent.core.agent.builtin import BuiltinAgentRunContext, BuiltinAgentSpec


def _spec(**overrides):
    values = {
        "name": "internal_worker",
        "version": "1",
        "system_prompt": "Perform one internal task.",
    }
    values.update(overrides)
    return BuiltinAgentSpec(**values)


def test_builtin_agent_spec_defaults_to_least_privilege() -> None:
    spec = _spec()

    assert spec.pattern == "single_call"
    assert spec.model_role == "general"
    assert spec.build_tools is None
    assert spec.memory_enabled is False
    assert spec.skills_enabled is False
    assert spec.workspace_enabled is False


def test_builtin_agent_spec_is_immutable() -> None:
    spec = _spec()

    with pytest.raises(FrozenInstanceError):
        setattr(spec, "name", "replacement")


def test_builtin_agent_name_accepts_the_bounded_maximum() -> None:
    assert _spec(name="a" * 64).name == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Internal Worker"),
        ("name", "a" * 65),
        ("version", " "),
        ("system_prompt", ""),
        ("model_role", " "),
        ("pattern", "unknown"),
    ],
)
def test_builtin_agent_spec_rejects_invalid_identity_and_runtime_fields(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        _spec(**{field: value})


@pytest.mark.parametrize("execution_id", [" ", "../escape", "nested/path", "run:1"])
def test_builtin_agent_run_context_rejects_unsafe_execution_id(
    execution_id: str,
) -> None:
    with pytest.raises(ValueError, match="execution_id"):
        BuiltinAgentRunContext(execution_id=execution_id)


def test_builtin_agent_run_context_copies_and_freezes_request_context() -> None:
    source = {"tenant": "alpha"}
    context = BuiltinAgentRunContext(
        execution_id="run-1",
        request_context=source,
    )
    source["tenant"] = "changed"

    assert context.request_context == {"tenant": "alpha"}
    with pytest.raises(TypeError):
        cast(dict[str, Any], context.request_context)["tenant"] = "mutated"
