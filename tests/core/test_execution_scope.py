"""Unit tests for core/execution_scope.py.

Slice 1 of #757: the ExecutionScope skeleton carries no consumers yet, so
these tests cover the value type, the contextvar helpers, the resolver hook,
and the per-turn activation contract (including restart/resume re-resolution).
"""

import asyncio
import contextvars
import dataclasses

import pytest

from xagent.core.execution_scope import (
    ExecutionScope,
    ExecutionScopeContext,
    InvalidScopeComponentError,
    get_execution_scope,
    reset_execution_scope,
    resolve_execution_scope,
    set_execution_scope,
    set_execution_scope_resolver,
    turn_execution_scope,
    validate_scope_component,
)


@pytest.fixture(autouse=True)
def _clear_resolver():
    """Each test starts and ends with no registered resolver."""
    set_execution_scope_resolver(None)
    yield
    set_execution_scope_resolver(None)


class TestValidateScopeComponent:
    def test_accepts_valid_components(self):
        for value in ["a", "A-b_9", "x" * 63, "0", "_", "-"]:
            assert validate_scope_component(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "x" * 64,
            "a:b",
            "a/b",
            "..",
            "a b",
            "a\n",
            "café",
            "a.b",
            None,
            123,
            ["a"],
        ],
    )
    def test_rejects_invalid_components(self, value):
        with pytest.raises(InvalidScopeComponentError):
            validate_scope_component(value)

    def test_rejects_without_sanitizing(self, caplog):
        """Invalid input raises and logs; it is never rewritten to a valid form."""
        with caplog.at_level("ERROR"):
            with pytest.raises(InvalidScopeComponentError):
                validate_scope_component("bad:name", field_name="sandbox_key_suffix")
        assert any("sandbox_key_suffix" in r.message for r in caplog.records)


class TestExecutionScope:
    def test_defaults_are_unscoped_behavior(self):
        scope = ExecutionScope()
        assert scope.sandbox_key_suffix is None
        assert scope.workspace_segments == ()
        assert dict(scope.memory_dimensions) == {}
        assert scope.strict_memory_isolation is False
        assert scope.isolate_external_dirs is False

    def test_frozen(self):
        scope = ExecutionScope()
        with pytest.raises(dataclasses.FrozenInstanceError):
            scope.sandbox_key_suffix = "x"

    def test_memory_dimensions_are_read_only(self):
        scope = ExecutionScope(memory_dimensions={"tenant": "acme"})
        with pytest.raises(TypeError):
            scope.memory_dimensions["tenant"] = "other"

    def test_workspace_segments_normalized_to_tuple(self):
        scope = ExecutionScope(workspace_segments=["proj", "env"])
        assert scope.workspace_segments == ("proj", "env")

    def test_equality(self):
        a = ExecutionScope(
            sandbox_key_suffix="s",
            workspace_segments=("w",),
            memory_dimensions={"k": "v"},
        )
        b = ExecutionScope(
            sandbox_key_suffix="s",
            workspace_segments=["w"],
            memory_dimensions={"k": "v"},
        )
        assert a == b
        assert a != ExecutionScope(sandbox_key_suffix="other")

    def test_rejects_invalid_sandbox_key_suffix(self):
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(sandbox_key_suffix="a:b")

    def test_rejects_invalid_workspace_segment(self):
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(workspace_segments=("ok", "../escape"))

    def test_rejects_invalid_memory_dimension_key(self):
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(memory_dimensions={"bad key": "v"})

    @pytest.mark.parametrize("value", ["", None, 3])
    def test_rejects_invalid_memory_dimension_value(self, value):
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(memory_dimensions={"k": value})

    def test_none_containers_raise_descriptive_value_error(self):
        """None for the collection fields raises a descriptive ValueError
        instead of an opaque TypeError from tuple()/dict() conversion."""
        with pytest.raises(ValueError, match="workspace_segments cannot be None"):
            ExecutionScope(workspace_segments=None)
        with pytest.raises(ValueError, match="memory_dimensions cannot be None"):
            ExecutionScope(memory_dimensions=None)

    def test_boolean_flags_independent_of_other_fields(self):
        """Flags are consumable with an otherwise-empty scope (independent fields)."""
        scope = ExecutionScope(strict_memory_isolation=True)
        assert scope.strict_memory_isolation is True
        assert scope.sandbox_key_suffix is None
        assert scope.workspace_segments == ()


class TestSandboxMountSegments:
    """The mount-prefix field (#79-01): decouples the sandbox mount root from
    the full workspace_segments so scopes sharing a suffix + prefix share one
    container while deeper segments stay in disjoint subtrees."""

    def test_default_mount_covers_full_workspace_segments(self):
        """Unset prefix => mount root == workspace root (byte-identical)."""
        scope = ExecutionScope(
            sandbox_key_suffix="client-3",
            workspace_segments=("clients", "3", "end_users", "7"),
        )
        assert scope.sandbox_mount_segments is None
        assert scope.effective_mount_segments == ("clients", "3", "end_users", "7")

    def test_unscoped_scope_has_empty_effective_mount(self):
        assert ExecutionScope().effective_mount_segments == ()

    def test_prefix_mount_shared_across_deeper_segments(self):
        """Two end users of one CA share a mount prefix; only deeper differs."""
        a = ExecutionScope(
            sandbox_key_suffix="client-3",
            workspace_segments=("clients", "3", "end_users", "7"),
            sandbox_mount_segments=("clients", "3"),
        )
        b = ExecutionScope(
            sandbox_key_suffix="client-3",
            workspace_segments=("clients", "3", "end_users", "8"),
            sandbox_mount_segments=("clients", "3"),
        )
        assert a.effective_mount_segments == ("clients", "3")
        assert b.effective_mount_segments == ("clients", "3")
        assert a.workspace_segments != b.workspace_segments

    def test_mount_segments_normalized_to_tuple(self):
        scope = ExecutionScope(
            workspace_segments=["clients", "3", "end_users", "7"],
            sandbox_mount_segments=["clients", "3"],
        )
        assert scope.sandbox_mount_segments == ("clients", "3")

    def test_empty_prefix_mounts_at_user_root(self):
        """() is a valid prefix of any segments — mount at the user root."""
        scope = ExecutionScope(
            workspace_segments=("clients", "3"),
            sandbox_mount_segments=(),
        )
        assert scope.effective_mount_segments == ()

    def test_rejects_non_prefix_mount_segments(self):
        with pytest.raises(InvalidScopeComponentError, match="must be a prefix"):
            ExecutionScope(
                workspace_segments=("clients", "3", "end_users", "7"),
                sandbox_mount_segments=("clients", "4"),
            )

    def test_rejects_mount_longer_than_workspace_segments(self):
        with pytest.raises(InvalidScopeComponentError, match="must be a prefix"):
            ExecutionScope(
                workspace_segments=("clients", "3"),
                sandbox_mount_segments=("clients", "3", "end_users", "7"),
            )

    def test_rejects_invalid_mount_segment_component(self):
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(
                workspace_segments=("clients", "3"),
                sandbox_mount_segments=("clients", "../escape"),
            )

    def test_to_dict_from_dict_round_trips_prefix(self):
        scope = ExecutionScope(
            sandbox_key_suffix="client-3",
            workspace_segments=("clients", "3", "end_users", "7"),
            sandbox_mount_segments=("clients", "3"),
        )
        assert ExecutionScope.from_dict(scope.to_dict()) == scope
        assert scope.to_dict()["sandbox_mount_segments"] == ["clients", "3"]

    def test_to_dict_preserves_none_vs_empty_distinction(self):
        """None (mount == full segments) must not collapse into () (mount at
        user root) across a serialization round-trip."""
        default = ExecutionScope(workspace_segments=("clients", "3"))
        assert default.to_dict()["sandbox_mount_segments"] is None
        restored_default = ExecutionScope.from_dict(default.to_dict())
        assert restored_default.sandbox_mount_segments is None

        rooted = ExecutionScope(
            workspace_segments=("clients", "3"), sandbox_mount_segments=()
        )
        assert rooted.to_dict()["sandbox_mount_segments"] == []
        restored_rooted = ExecutionScope.from_dict(rooted.to_dict())
        assert restored_rooted.sandbox_mount_segments == ()


class TestContextvarHelpers:
    def test_default_is_none(self):
        assert get_execution_scope() is None

    def test_set_and_reset(self):
        scope = ExecutionScope(sandbox_key_suffix="s1")
        token = set_execution_scope(scope)
        try:
            assert get_execution_scope() is scope
        finally:
            reset_execution_scope(token)
        assert get_execution_scope() is None

    def test_context_manager_restores_previous(self):
        outer = ExecutionScope(sandbox_key_suffix="outer")
        inner = ExecutionScope(sandbox_key_suffix="inner")
        with ExecutionScopeContext(outer):
            assert get_execution_scope() is outer
            with ExecutionScopeContext(inner):
                assert get_execution_scope() is inner
            assert get_execution_scope() is outer
        assert get_execution_scope() is None

    def test_context_manager_restores_on_exception(self):
        scope = ExecutionScope(sandbox_key_suffix="s1")
        with pytest.raises(RuntimeError):
            with ExecutionScopeContext(scope):
                raise RuntimeError("boom")
        assert get_execution_scope() is None

    def test_explicit_none_overrides_outer_scope(self):
        """Setting None is explicitly-unscoped, shadowing any outer scope."""
        outer = ExecutionScope(sandbox_key_suffix="outer")
        with ExecutionScopeContext(outer):
            with ExecutionScopeContext(None):
                assert get_execution_scope() is None
            assert get_execution_scope() is outer


class TestResolverHook:
    def test_no_resolver_resolves_unscoped(self):
        assert resolve_execution_scope("42") is None

    def test_resolver_receives_task_id_as_str(self):
        seen = []

        def resolver(task_id):
            seen.append(task_id)
            return None

        set_execution_scope_resolver(resolver)
        resolve_execution_scope(42)
        assert seen == ["42"]

    def test_resolver_result_is_returned(self):
        scope = ExecutionScope(sandbox_key_suffix="s1")
        set_execution_scope_resolver(lambda task_id: scope)
        assert resolve_execution_scope("42") is scope

    def test_resolver_exception_propagates(self):
        """A resolver error fails the turn instead of silently running unscoped."""

        def resolver(task_id):
            raise RuntimeError("resolver down")

        set_execution_scope_resolver(resolver)
        with pytest.raises(RuntimeError, match="resolver down"):
            resolve_execution_scope("42")

    def test_none_task_id_raises_instead_of_resolving_the_string_none(self):
        """str(None) would silently query the resolver for "None"; a caller
        with no task identity must treat the execution as unscoped itself."""
        seen = []
        set_execution_scope_resolver(lambda task_id: seen.append(task_id))
        with pytest.raises(ValueError, match="task_id cannot be None"):
            resolve_execution_scope(None)
        assert seen == []

    def test_resolver_can_be_cleared(self):
        set_execution_scope_resolver(lambda task_id: ExecutionScope())
        set_execution_scope_resolver(None)
        assert resolve_execution_scope("42") is None


class TestTurnExecutionScope:
    def test_activates_resolved_scope_for_the_turn(self):
        scope = ExecutionScope(workspace_segments=("proj",))
        set_execution_scope_resolver(lambda task_id: scope if task_id == "7" else None)
        with turn_execution_scope(7) as active:
            assert active is scope
            assert get_execution_scope() is scope
        assert get_execution_scope() is None

    def test_unscoped_turn_activates_none(self):
        with turn_execution_scope("7") as active:
            assert active is None
            assert get_execution_scope() is None

    def test_resolver_called_once_per_turn(self):
        calls = []

        def resolver(task_id):
            calls.append(task_id)
            return ExecutionScope(sandbox_key_suffix=f"t{task_id}")

        set_execution_scope_resolver(resolver)
        with turn_execution_scope("7"):
            pass
        with turn_execution_scope("7"):
            pass
        assert calls == ["7", "7"]

    def test_scope_restored_on_exception(self):
        set_execution_scope_resolver(lambda task_id: ExecutionScope())
        with pytest.raises(RuntimeError):
            with turn_execution_scope("7"):
                raise RuntimeError("turn failed")
        assert get_execution_scope() is None

    def test_resume_after_restart_re_resolves_identical_scope(self):
        """A resumed task re-resolves and re-applies the identical scope.

        Simulates a process restart between turns: the contextvar starts
        empty in a fresh context and the embedder re-registers its resolver
        at startup; the scope is re-derived from the resolver's persistent
        mapping keyed by task_id, not from any prior in-process state.
        """
        resolver_calls = []

        def make_resolver():
            # The embedder derives the scope from its own persistent data;
            # a fresh resolver instance (new process) yields an equal scope.
            def resolver(task_id):
                resolver_calls.append(task_id)
                return ExecutionScope(
                    sandbox_key_suffix="tenant-a",
                    workspace_segments=("tenant-a",),
                    memory_dimensions={"tenant": "a"},
                )

            return resolver

        def run_turn():
            set_execution_scope_resolver(make_resolver())
            with turn_execution_scope("99") as scope:
                return scope, get_execution_scope()

        # Turn 1 and turn 2 run in independent contexts (as after a restart).
        first_scope, first_active = contextvars.copy_context().run(run_turn)
        set_execution_scope_resolver(None)
        second_scope, second_active = contextvars.copy_context().run(run_turn)

        assert resolver_calls == ["99", "99"]
        assert first_active is first_scope
        assert second_active is second_scope
        assert first_scope == second_scope

    def test_scope_visible_inside_async_turn(self):
        """The activated scope propagates into the turn's async execution."""
        scope = ExecutionScope(sandbox_key_suffix="s1")
        set_execution_scope_resolver(lambda task_id: scope)

        async def fake_agent_execution():
            await asyncio.sleep(0)
            return get_execution_scope()

        async def turn():
            with turn_execution_scope("7"):
                return await fake_agent_execution()

        assert asyncio.run(turn()) is scope
