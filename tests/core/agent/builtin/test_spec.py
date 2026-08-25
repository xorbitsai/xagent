from __future__ import annotations

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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Internal Worker"),
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


@pytest.mark.parametrize("execution_id", [" ", "../escape", "nested/path"])
def test_builtin_agent_run_context_rejects_unsafe_execution_id(
    execution_id: str,
) -> None:
    with pytest.raises(ValueError, match="execution_id"):
        BuiltinAgentRunContext(execution_id=execution_id)
