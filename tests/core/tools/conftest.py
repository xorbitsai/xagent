"""Shared fixtures for the vibe agent-tool tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def no_resolvable_default_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the delegation model fallback yield nothing.

    ``get_configured_defaults`` ends in ``create_llm_from_env()``, so a test
    asserting the no-valid-model exit would otherwise pass only on a machine
    with no provider key exported -- and would issue a real request on one
    that has it. The two modules that assert that exit apply this module-wide
    via ``pytestmark``, so a new test there cannot reintroduce the hazard by
    forgetting to ask; a test wanting a resolvable default overrides it with
    its own monkeypatch.
    """
    from xagent.web.services.llm_utils import UserAwareModelStorage

    monkeypatch.setattr(
        UserAwareModelStorage,
        "get_configured_defaults",
        lambda self, *args, **kwargs: (None, None, None, None),
    )
