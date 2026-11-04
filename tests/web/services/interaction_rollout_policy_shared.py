"""Install a frozen interaction rollout policy for one test."""

from __future__ import annotations

from typing import Literal

import pytest

import xagent.web.services.interaction_rollout as ir


def install_interaction_rollout_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: Literal["legacy", "read", "native"],
    shared_task_execution: bool,
    native_sources: frozenset[str] = frozenset(),
) -> None:
    """Set the policy singleton as validate_interaction_rollout_at_startup()
    would have frozen it; monkeypatch restores the previous value."""
    monkeypatch.setattr(
        ir,
        "_policy",
        ir.InteractionRolloutPolicy(
            mode=mode,
            native_sources=frozenset(native_sources),
            native_protocol_version=1,
            shared_task_execution=shared_task_execution,
        ),
    )
