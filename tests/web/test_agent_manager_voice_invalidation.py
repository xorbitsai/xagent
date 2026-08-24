"""Unit tests for AgentServiceManager.invalidate_cached_agents_for_owner.

A cached AgentService bakes its system prompt in at construction time and
the per-turn cache-hit path only re-checks owner/scope invariants, not
preferences (see the end-to-end regression in
test_agent_manager_reconstruction.py::test_voice_change_invalidates_cached_agent_on_next_turn).
This method is the PATCH /api/auth/me/preferences endpoint's only way to
reach an already-warm task's cache; these tests cover its dict bookkeeping
in isolation.
"""

from xagent.web.api.chat import AgentServiceManager


def test_invalidate_evicts_only_the_matching_owners_tasks() -> None:
    manager = AgentServiceManager()
    manager._agents[1] = object()
    manager._agents[2] = object()
    manager._agent_owner_ids[1] = 7
    manager._agent_owner_ids[2] = 8
    manager._agent_sandbox_keys[1] = "user:7"
    manager._agent_sandbox_keys[2] = "user:8"
    manager._agent_sandbox_providers[1] = object()
    manager._agent_sandbox_providers[2] = object()
    manager._agent_scope_fingerprints[1] = None
    manager._agent_scope_fingerprints[2] = None

    manager.invalidate_cached_agents_for_owner(7)

    assert 1 not in manager._agents
    assert 1 not in manager._agent_owner_ids
    assert 1 not in manager._agent_sandbox_keys
    assert 1 not in manager._agent_sandbox_providers
    assert 1 not in manager._agent_scope_fingerprints
    # A different owner's cached task must survive untouched.
    assert 2 in manager._agents
    assert manager._agent_owner_ids[2] == 8


def test_invalidate_does_not_touch_workspace() -> None:
    """Unlike remove_agent, this must not call cleanup_workspace - the
    same owner's on-disk data must survive the rebuild (mirrors the
    scope-fingerprint-mismatch eviction's own workspace-preserving
    behavior in _get_agent_for_task_unlocked)."""
    manager = AgentServiceManager()

    class _RaisesIfCleaned:
        def cleanup_workspace(self) -> None:
            raise AssertionError("workspace must not be cleaned up")

    manager._agents[1] = _RaisesIfCleaned()
    manager._agent_owner_ids[1] = 7

    manager.invalidate_cached_agents_for_owner(7)

    assert 1 not in manager._agents


def test_invalidate_is_a_noop_for_an_owner_with_no_cached_tasks() -> None:
    manager = AgentServiceManager()
    manager._agents[1] = object()
    manager._agent_owner_ids[1] = 7

    manager.invalidate_cached_agents_for_owner(999)

    assert 1 in manager._agents
    assert manager._agent_owner_ids[1] == 7
