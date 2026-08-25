from __future__ import annotations

from typing import Any, cast

import pytest

from xagent.core.agent.builtin import (
    BuiltinAgentNotFoundError,
    BuiltinAgentRegistrationError,
    BuiltinAgentRegistry,
    BuiltinAgentSpec,
)


def _spec(name: str, version: str = "1") -> BuiltinAgentSpec:
    return BuiltinAgentSpec(
        name=name,
        version=version,
        system_prompt=f"Run {name}.",
    )


def test_registry_registers_and_lists_code_defined_specs() -> None:
    first = _spec("first")
    second = _spec("second")
    registry = BuiltinAgentRegistry([first])

    registry.register(second)

    assert registry.get("first") is first
    assert registry.require("second") is second
    assert registry.list_specs() == (first, second)
    assert "first" in registry


def test_registry_rejects_duplicate_names_even_across_versions() -> None:
    registry = BuiltinAgentRegistry([_spec("worker", version="1")])

    with pytest.raises(BuiltinAgentRegistrationError, match="already registered"):
        registry.register(_spec("worker", version="2"))


def test_registry_require_fails_for_unknown_agent() -> None:
    with pytest.raises(BuiltinAgentNotFoundError, match="not registered"):
        BuiltinAgentRegistry().require("missing")


def test_registry_rejects_non_spec_entries() -> None:
    with pytest.raises(BuiltinAgentRegistrationError, match="BuiltinAgentSpec"):
        BuiltinAgentRegistry().register(cast(Any, object()))
