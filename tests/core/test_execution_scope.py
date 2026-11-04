"""Unit tests for core/execution_scope.py.

Covers the value type, the contextvar helpers, the resolver hook, and the
per-turn activation contract (including restart/resume re-resolution). The
resolver is authoritative over a persisted snapshot rather than always
losing to it. The resolver fixtures below register through
``acknowledges_snapshot_candidate_contract=True`` -- see
``TestSnapshotCandidateAuthority`` for the full resolver/snapshot precedence
matrix and ``tests/web/test_execution_scope_delegation.py`` for the
web-layer snapshot loader wiring. The resolver/loader globals themselves are
reset by the root-level ``isolate_execution_scope_hooks`` autouse fixture in
``tests/conftest.py``, not by a fixture in this module.
"""

import asyncio
import contextvars
import dataclasses
import logging
from contextlib import contextmanager

import pytest

from xagent.core.execution_scope import (
    EXECUTION_SCOPE_AGENT_CONFIG_KEY,
    EXECUTION_SCOPE_SHAPE_VERSION,
    DeferToSnapshot,
    ExecutionScope,
    ExecutionScopeAbstentionMismatchError,
    ExecutionScopeAuthorityError,
    ExecutionScopeContext,
    ExecutionScopeResolverContractError,
    InvalidScopeComponentError,
    execution_scope_from_agent_config,
    execution_scope_resolver_registered,
    get_execution_scope,
    reset_execution_scope,
    resolve_execution_scope,
    resolve_execution_scope_off_turn,
    scope_fingerprint,
    set_execution_scope,
    set_execution_scope_resolver,
    set_execution_scope_snapshot_loader,
    turn_execution_scope,
    validate_scope_component,
)


@contextmanager
def scope_log_records(level: int = logging.WARNING):
    """Record what this module logs, without going through logging config.

    Asserting on log *content* through ``caplog`` or an attached handler makes
    the assertion depend on the level, the global disable threshold, the handler
    set and which module instance owns the logger -- none of which the code
    under test decides. Swapping the ``logger`` binding in the namespace the
    functions actually read it from observes the call itself, so what is
    asserted is that the code logs, not that a particular logging setup
    delivers it.

    ``level`` is the enablement the recorder itself reports, so code that asks
    ``logger.isEnabledFor(...)`` before building a message gets a deterministic
    answer from the recorder rather than from whatever the ambient logging
    configuration happens to be. Every call that survives that check is
    recorded verbatim; the caller filters.
    """

    class _Recorder:
        def __init__(self, wrapped: logging.Logger) -> None:
            self._wrapped = wrapped
            self.records: list[str] = []

        def isEnabledFor(self, asked: int) -> bool:  # noqa: N802 - logging API
            return asked >= level

        def info(self, msg: str, *args: object, **kwargs: object) -> None:
            self.records.append(msg % args if args else msg)
            self._wrapped.info(msg, *args, **kwargs)

        def warning(self, msg: str, *args: object, **kwargs: object) -> None:
            self.records.append(msg % args if args else msg)
            self._wrapped.warning(msg, *args, **kwargs)

        def error(self, msg: str, *args: object, **kwargs: object) -> None:
            self.records.append(msg % args if args else msg)
            self._wrapped.error(msg, *args, **kwargs)

        # There is deliberately no ``__getattr__`` delegating the rest to the
        # wrapped logger. ``isEnabledFor`` must not be delegated -- that would
        # put the ambient level and the process-wide disable threshold back in
        # charge of what the code under test does, which is the whole reason
        # this helper exists -- and modelling exactly the levels the module
        # logs at means a newly introduced level surfaces as a loud
        # ``AttributeError`` instead of a silently unrecorded call.

    # The function's own globals: the binding it resolves `logger` against at
    # call time, whichever module instance that turns out to be.
    namespace = resolve_execution_scope.__globals__
    original = namespace["logger"]
    recorder = _Recorder(original)
    namespace["logger"] = recorder
    try:
        yield recorder.records
    finally:
        namespace["logger"] = original


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

    def test_rejects_without_sanitizing(self):
        """Invalid input raises and logs; it is never rewritten to a valid form."""
        with scope_log_records(logging.ERROR) as records:
            with pytest.raises(InvalidScopeComponentError):
                validate_scope_component("bad:name", field_name="sandbox_key_suffix")
        assert any("sandbox_key_suffix" in r for r in records)


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


class TestDurableStorageSegments:
    """The durable-storage-handle field (#828): mirrors the filesystem
    external-dir allowlist — narrow the object-storage handle to the scope
    subtree only under ``isolate_external_dirs``."""

    def test_isolated_scope_yields_workspace_segments(self):
        scope = ExecutionScope(
            workspace_segments=("clients", "3", "end_users", "7"),
            isolate_external_dirs=True,
        )
        assert scope.durable_storage_segments == ("clients", "3", "end_users", "7")

    def test_non_isolated_scope_yields_empty(self):
        # Segments present but not isolated => owner-root handle (shared reads).
        scope = ExecutionScope(
            workspace_segments=("clients", "3", "end_users", "7"),
            isolate_external_dirs=False,
        )
        assert scope.durable_storage_segments == ()

    def test_unscoped_scope_yields_empty(self):
        assert ExecutionScope().durable_storage_segments == ()

    def test_isolated_without_segments_yields_empty(self):
        assert ExecutionScope(isolate_external_dirs=True).durable_storage_segments == ()


class TestContextvarHelpers:
    def test_not_provided_sentinel_is_a_shared_typed_value(self):
        from xagent.core.execution_scope import (
            EXECUTION_SCOPE_NOT_PROVIDED,
            ExecutionScopeNotProvided,
        )

        assert EXECUTION_SCOPE_NOT_PROVIDED is not None
        assert type(EXECUTION_SCOPE_NOT_PROVIDED) is ExecutionScopeNotProvided

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

        set_execution_scope_resolver(
            resolver, acknowledges_snapshot_candidate_contract=True
        )
        resolve_execution_scope(42)
        assert seen == ["42"]

    def test_resolver_result_is_returned(self):
        scope = ExecutionScope(sandbox_key_suffix="s1")
        set_execution_scope_resolver(
            lambda task_id: scope, acknowledges_snapshot_candidate_contract=True
        )
        assert resolve_execution_scope("42") is scope

    def test_resolver_exception_propagates(self):
        """A resolver error fails the turn instead of silently running unscoped."""

        def resolver(task_id):
            raise RuntimeError("resolver down")

        set_execution_scope_resolver(
            resolver, acknowledges_snapshot_candidate_contract=True
        )
        with pytest.raises(RuntimeError, match="resolver down"):
            resolve_execution_scope("42")

    def test_none_task_id_raises_instead_of_resolving_the_string_none(self):
        """str(None) would silently query the resolver for "None"; a caller
        with no task identity must treat the execution as unscoped itself."""
        seen = []
        set_execution_scope_resolver(
            lambda task_id: seen.append(task_id),
            acknowledges_snapshot_candidate_contract=True,
        )
        with pytest.raises(ValueError, match="task_id cannot be None"):
            resolve_execution_scope(None)
        assert seen == []

    def test_resolver_can_be_cleared(self):
        set_execution_scope_resolver(
            lambda task_id: ExecutionScope(),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_resolver(None)
        assert resolve_execution_scope("42") is None


class TestTurnExecutionScope:
    def test_activates_resolved_scope_for_the_turn(self):
        scope = ExecutionScope(workspace_segments=("proj",))
        set_execution_scope_resolver(
            lambda task_id: scope if task_id == "7" else None,
            acknowledges_snapshot_candidate_contract=True,
        )
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

        set_execution_scope_resolver(
            resolver, acknowledges_snapshot_candidate_contract=True
        )
        with turn_execution_scope("7"):
            pass
        with turn_execution_scope("7"):
            pass
        assert calls == ["7", "7"]

    def test_scope_restored_on_exception(self):
        set_execution_scope_resolver(
            lambda task_id: ExecutionScope(),
            acknowledges_snapshot_candidate_contract=True,
        )
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
            set_execution_scope_resolver(
                make_resolver(), acknowledges_snapshot_candidate_contract=True
            )
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
        set_execution_scope_resolver(
            lambda task_id: scope, acknowledges_snapshot_candidate_contract=True
        )

        async def fake_agent_execution():
            await asyncio.sleep(0)
            return get_execution_scope()

        async def turn():
            with turn_execution_scope("7"):
                return await fake_agent_execution()

        assert asyncio.run(turn()) is scope


class TestDeferToSnapshot:
    """The carrier for a resolver's explicit abstention."""

    def test_requires_an_execution_scope_fallback(self):
        with pytest.raises(TypeError):
            DeferToSnapshot("not-a-scope")

    def test_carrier_is_not_an_execution_scope(self):
        carrier = DeferToSnapshot(ExecutionScope())
        assert isinstance(carrier, DeferToSnapshot)
        assert isinstance(carrier, ExecutionScope) is False

    def test_carrier_has_no_public_bare_singleton(self):
        """The fallback-carrying class is the only way to abstain -- a bare
        module-level sentinel would let "defer" mean "unscoped on miss"
        implicitly instead of requiring an explicit fallback."""
        import xagent.core.execution_scope as scope_module

        assert not hasattr(scope_module, "DEFER_TO_SNAPSHOT")

    def test_no_passthrough_factory_duplicates_the_carrier(self):
        """One concept, one public name. The class holds the abstention
        contract and its own construction-time validation, so a module-level
        factory wrapping it would be a second spelling resolver authors have
        to choose between, and a second place validation could drift to."""
        import xagent.core.execution_scope as scope_module

        assert not hasattr(scope_module, "defer_to_snapshot")

    def test_carrier_exposes_the_fallback(self):
        fallback = ExecutionScope(sandbox_key_suffix="fallback")
        assert DeferToSnapshot(fallback).fallback == fallback


class TestSetExecutionScopeResolverAckToken:
    """The confirmation-token contract on ``set_execution_scope_resolver``."""

    def test_registering_a_resolver_without_ack_raises(self):
        with pytest.raises(TypeError, match="acknowledges_snapshot_candidate_contract"):
            set_execution_scope_resolver(lambda task_id: None)

    def test_clearing_the_resolver_never_needs_ack(self):
        # None never triggers the check: the ~20 cleanup call sites across
        # the test suite that reset the resolver to None stay unchanged.
        set_execution_scope_resolver(None)
        assert execution_scope_resolver_registered() is False

    def test_ack_true_registers_successfully(self):
        set_execution_scope_resolver(
            lambda task_id: None, acknowledges_snapshot_candidate_contract=True
        )
        assert execution_scope_resolver_registered() is True

    def test_ack_false_explicitly_still_raises(self):
        with pytest.raises(TypeError):
            set_execution_scope_resolver(
                lambda task_id: None,
                acknowledges_snapshot_candidate_contract=False,
            )


class TestExecutionScopeAuthorityErrorInheritance:
    """Pin: distinguishable from the two exceptions this module
    already raises, so no existing ``except RuntimeError``/``except
    ValueError`` clause silently folds an authority conflict into them."""

    def test_not_a_runtime_error(self):
        assert not issubclass(ExecutionScopeAuthorityError, RuntimeError)

    def test_not_a_value_error(self):
        assert not issubclass(ExecutionScopeAuthorityError, ValueError)

    def test_not_a_type_error(self):
        assert not issubclass(ExecutionScopeAuthorityError, TypeError)

    def test_not_a_key_error(self):
        assert not issubclass(ExecutionScopeAuthorityError, KeyError)

    def test_is_a_plain_exception(self):
        assert issubclass(ExecutionScopeAuthorityError, Exception)

    def test_str_carries_task_id_and_field_names_never_values(self):
        """``str()`` on this exception ends up in ``task.error_message`` and
        in the client's terminal error event (see this class's docstring for
        the exact surfacing sites), so it must never leak the scope values
        -- e.g. ``sandbox_key_suffix``/``workspace_segments``/
        ``memory_dimensions`` -- which can carry end-user/client
        identifiers. Only the task id and the mismatched field *names* are
        safe to include."""
        resolver_scope = ExecutionScope(
            sandbox_key_suffix="resolver-secret-suffix",
            workspace_segments=("resolver-secret-segment",),
        )
        snapshot_scope = ExecutionScope(
            sandbox_key_suffix="snapshot-secret-suffix",
            workspace_segments=("snapshot-secret-segment",),
        )
        exc = ExecutionScopeAuthorityError(
            "task-123",
            resolver_scope=resolver_scope,
            snapshot_scope=snapshot_scope,
            mismatched_fields={
                "sandbox_key_suffix": (
                    snapshot_scope.sandbox_key_suffix,
                    resolver_scope.sandbox_key_suffix,
                ),
                "workspace_segments": (
                    snapshot_scope.workspace_segments,
                    resolver_scope.workspace_segments,
                ),
            },
            resolver_scope_is_authoritative=True,
        )
        message = str(exc)
        assert "task-123" in message
        assert "sandbox_key_suffix" in message
        assert "workspace_segments" in message
        assert "resolver-secret-suffix" not in message
        assert "snapshot-secret-suffix" not in message
        assert "resolver-secret-segment" not in message
        assert "snapshot-secret-segment" not in message


class TestExecutionScopeResolverContractErrorInheritance:
    """Pin: the resolve-boundary "unknown return type" error
    must not be catchable by the ``except (ValueError, KeyError, TypeError)``
    clauses several websocket handlers wrap around the turn-execution path
    -- those fold anything in that tuple into a generic "client message
    format error" response, which would misreport a resolver author's bug
    as a malformed client message."""

    def test_not_a_runtime_error(self):
        assert not issubclass(ExecutionScopeResolverContractError, RuntimeError)

    def test_not_a_value_error(self):
        assert not issubclass(ExecutionScopeResolverContractError, ValueError)

    def test_not_a_type_error(self):
        assert not issubclass(ExecutionScopeResolverContractError, TypeError)

    def test_not_a_key_error(self):
        assert not issubclass(ExecutionScopeResolverContractError, KeyError)

    def test_is_a_plain_exception(self):
        assert issubclass(ExecutionScopeResolverContractError, Exception)


class TestExecutionScopeShapeVersionAlignment:
    """``to_dict``'s key set must track every dataclass field (precedent:
    test_runtime_spec.py:315) so a newly added field cannot silently miss
    persistence -- which would make a historical snapshot indistinguishable
    from a current one that simply left the field at its default."""

    def test_to_dict_keys_match_dataclass_fields(self):
        field_names = {f.name for f in dataclasses.fields(ExecutionScope)}
        assert set(ExecutionScope().to_dict().keys()) == field_names

    def test_fresh_scope_is_current_version(self):
        assert ExecutionScope().version == EXECUTION_SCOPE_SHAPE_VERSION

    def test_from_dict_missing_version_defaults_to_legacy_zero(self):
        data = ExecutionScope(sandbox_key_suffix="x").to_dict()
        del data["version"]
        assert ExecutionScope.from_dict(data).version == 0

    def test_from_dict_string_version_is_coerced_to_int(self):
        """A JSON-decoded snapshot carries ``version`` as whatever type the
        wire format gave it; ``from_dict`` must coerce it to ``int`` so a
        stringly-typed ``"1"`` still compares equal to
        ``EXECUTION_SCOPE_SHAPE_VERSION`` in ``resolve_execution_scope``
        instead of being silently treated as a shape mismatch."""
        data = ExecutionScope(sandbox_key_suffix="x").to_dict()
        data["version"] = str(EXECUTION_SCOPE_SHAPE_VERSION)
        decoded = ExecutionScope.from_dict(data)
        assert decoded.version == EXECUTION_SCOPE_SHAPE_VERSION
        assert isinstance(decoded.version, int)

    def test_version_is_excluded_from_equality(self):
        current = ExecutionScope(sandbox_key_suffix="x")
        legacy_data = current.to_dict()
        legacy_data["version"] = 0
        legacy = ExecutionScope.from_dict(legacy_data)
        assert legacy.version == 0
        assert legacy == current

    def test_to_dict_stamps_current_version_even_from_a_stale_scope(self):
        """``to_dict()`` always stamps :data:`EXECUTION_SCOPE_SHAPE_VERSION`,
        never ``self.version``: the dict it returns is being built *now*, in
        the current shape, regardless of whether ``self`` was itself decoded
        from an older snapshot (stale ``.version``). Propagating that stale
        value would let a decoded-then-re-persisted scope masquerade as
        pre-dating fields it actually has, permanently ignoring it as a
        candidate in ``resolve_execution_scope``."""
        stale_data = ExecutionScope(sandbox_key_suffix="x").to_dict()
        stale_data["version"] = 0
        stale_scope = ExecutionScope.from_dict(stale_data)
        assert stale_scope.version == 0
        assert stale_scope.to_dict()["version"] == EXECUTION_SCOPE_SHAPE_VERSION

    def test_shape_version_bump_requires_touching_this_pin(self):
        """Pins the exact field set ``EXECUTION_SCOPE_SHAPE_VERSION`` == 1
        was cut for. A namespace-affecting field can be added --
        wired into ``to_dict``/``from_dict``/
        ``_EXECUTION_SCOPE_NAMESPACE_FIELDS``/``scope_fingerprint`` -- and
        pass the rest of the suite without bumping the shape version, which
        would make every already-persisted snapshot compare against a wider
        shape it never had a chance to opt into, and
        ``resolve_execution_scope`` would fail every turn touching one until
        the version is bumped. This test forces the field set to be
        re-affirmed (and the version bumped) on the next field addition
        instead of drifting unnoticed."""
        field_names = {f.name for f in dataclasses.fields(ExecutionScope)}
        assert field_names == {
            "sandbox_key_suffix",
            "workspace_segments",
            "sandbox_mount_segments",
            "memory_dimensions",
            "strict_memory_isolation",
            "isolate_external_dirs",
            "version",
        }
        assert EXECUTION_SCOPE_SHAPE_VERSION == 1


class TestFromDictUntrustedFieldCoercion:
    """``from_dict`` decodes a mapping that can originate in a
    client-influenceable ``Task.agent_config`` (see
    :data:`EXECUTION_SCOPE_AGENT_CONFIG_KEY`), so every field it reads must
    be coerced defensively -- never a raw ``int()``/``tuple()``/``dict()``
    call that can raise a bare stdlib error (swallowed by generic ``except
    (ValueError, KeyError, TypeError)`` handlers elsewhere) or silently
    misread/truncate the value.
    """

    # --- version: readable, or a malformed-snapshot fault -----------------
    #
    # ``0`` is the marker for an authentic snapshot written before the field
    # existed, and that marker is trusted: the older half of the shape-version
    # gate is relaxed on the ``snapshot_defines_namespace=True`` branch, where
    # such a snapshot's namespace fields are used verbatim. An unreadable
    # value must therefore not decode to it.

    @pytest.mark.parametrize(
        "raw_version", ["abc", {"a": 1}, [1], object(), 1.5, True, -5]
    )
    def test_from_dict_unreadable_version_raises_instead_of_claiming_legacy(
        self, raw_version
    ):
        """Every shape that is not a readable version is a malformed persisted
        snapshot, not an old one. ``1.5`` and ``True`` are here because a bare
        ``int()`` turns them into ``1`` -- a claim to be the current shape --
        and ``-5`` because "older than every version that ever existed" would
        borrow the legacy marker's verbatim-namespace trust for a value no
        writer emits (``to_dict`` always stamps the current constant)."""
        data = ExecutionScope(sandbox_key_suffix="x").to_dict()
        data["version"] = raw_version
        with pytest.raises(ExecutionScopeResolverContractError):
            ExecutionScope.from_dict(data)

    def test_from_dict_unreadable_version_is_not_a_client_message_fault(self):
        """The error stays outside ``(ValueError, KeyError, TypeError)``: the
        websocket handlers that catch that tuple around the turn path would
        otherwise answer the client with a message-format validation error for
        what is a persisted-data fault."""
        data = ExecutionScope(sandbox_key_suffix="x").to_dict()
        data["version"] = "abc"
        with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
            ExecutionScope.from_dict(data)
        assert not isinstance(exc_info.value, _WEBSOCKET_VALIDATION_EXCEPT_TUPLE)

    def test_from_dict_unreadable_version_message_names_no_value(self):
        """The raised message reaches the task's error column and a
        client-facing event, so it names the offending type only; the value
        goes to the server-side log."""
        data = ExecutionScope(sandbox_key_suffix="x").to_dict()
        data["version"] = "9;DROP"
        with scope_log_records(logging.ERROR) as records:
            with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
                ExecutionScope.from_dict(data)
        assert "9;DROP" not in str(exc_info.value)
        assert "str" in str(exc_info.value)
        assert any("9;DROP" in r for r in records), records

    def test_from_dict_readable_versions_are_accepted(self):
        """The accepted domain: an absent key or ``None`` is the pre-versioning
        marker, a digit string is what a JSON-decoded version can arrive as,
        and a non-negative ``int`` passes through. Nothing here may raise --
        an old snapshot is a supported input, not a fault."""
        for raw, expected in ((None, 0), ("0", 0), ("1", 1), (0, 0), (1, 1), (9, 9)):
            data = ExecutionScope(sandbox_key_suffix="x").to_dict()
            data["version"] = raw
            decoded = ExecutionScope.from_dict(data)
            assert decoded.version == expected
            assert isinstance(decoded.version, int)

        missing = ExecutionScope(sandbox_key_suffix="x").to_dict()
        del missing["version"]
        assert ExecutionScope.from_dict(missing).version == 0

    # --- workspace_segments / sandbox_mount_segments: no silent str/dict
    # iteration, no bare TypeError on a non-iterable ----------------------

    def test_from_dict_string_workspace_segments_raises_instead_of_splitting_chars(
        self,
    ):
        """``tuple("ab")`` would silently become ``("a", "b")`` -- two valid-
        looking single-character segments that pass per-segment validation
        with no signal that the input was never a sequence."""
        data = ExecutionScope().to_dict()
        data["workspace_segments"] = "ab"
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope.from_dict(data)

    def test_from_dict_non_iterable_workspace_segments_raises_domain_error(self):
        """``tuple(5)`` raises a bare ``TypeError`` outside this module's
        error taxonomy; the coercion guard must raise
        ``InvalidScopeComponentError`` before reaching that call."""
        data = ExecutionScope().to_dict()
        data["workspace_segments"] = 5
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope.from_dict(data)

    def test_from_dict_falsy_workspace_segments_still_defaults_to_empty(self):
        """Falsy-but-absent-shaped values (``None``, ``""``, ``0``, ``False``)
        keep meaning "no segments supplied" rather than becoming a coercion
        error -- only a truthy non-sequence is a real contract violation."""
        for raw in (None, "", 0, False, []):
            data = ExecutionScope().to_dict()
            data["workspace_segments"] = raw
            assert ExecutionScope.from_dict(data).workspace_segments == ()

    def test_from_dict_string_sandbox_mount_segments_raises(self):
        data = ExecutionScope(workspace_segments=("a", "b")).to_dict()
        data["sandbox_mount_segments"] = "ab"
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope.from_dict(data)

    def test_from_dict_none_sandbox_mount_segments_stays_none_not_coerced(self):
        """``None`` is a load-bearing sentinel (mount == full
        workspace_segments), distinct from an explicit ``[]`` (rooted
        mount) -- the coercion guard must preserve that distinction rather
        than routing ``None`` through the same falsy-default path as
        ``workspace_segments``."""
        data = ExecutionScope(workspace_segments=("a", "b")).to_dict()
        assert data["sandbox_mount_segments"] is None
        assert ExecutionScope.from_dict(data).sandbox_mount_segments is None

    # --- memory_dimensions: no silent pair-splitting of a bad iterable ----

    def test_from_dict_string_memory_dimensions_raises_instead_of_dict_of_chars(self):
        """``dict(["ab"])`` would silently reinterpret ``"ab"`` as the
        key/value pair ``{"a": "b"}``."""
        data = ExecutionScope().to_dict()
        data["memory_dimensions"] = ["ab"]
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope.from_dict(data)

    def test_from_dict_non_mapping_memory_dimensions_raises_domain_error(self):
        data = ExecutionScope().to_dict()
        data["memory_dimensions"] = 5
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope.from_dict(data)

    def test_from_dict_falsy_memory_dimensions_still_defaults_to_empty(self):
        for raw in (None, "", 0, False, {}):
            data = ExecutionScope().to_dict()
            data["memory_dimensions"] = raw
            assert dict(ExecutionScope.from_dict(data).memory_dimensions) == {}


# The tuple several websocket handlers wrap around the turn-execution path,
# including its per-turn ``resolve_execution_scope`` call. Anything in it is
# answered to the client as a message-format validation error, so an exception
# that is *not* about the client's message must fall through it.
_WEBSOCKET_VALIDATION_EXCEPT_TUPLE = (ValueError, KeyError, TypeError)


class TestPersistedSnapshotDecodeIsNotAClientMessageFault:
    """A persisted snapshot that no longer decodes is a persisted-data fault.

    ``execution_scope_from_agent_config`` is where persisted snapshot data is
    read, and the registered snapshot loaders call it from inside
    ``resolve_execution_scope``. The field coercions it drives raise
    ``InvalidScopeComponentError``, which is a ``ValueError`` -- so without a
    boundary conversion the failure lands in the websocket handlers'
    ``except (ValueError, KeyError, TypeError)`` clause and the client is told
    its own message was malformed, for data it may not have sent and cannot
    fix. The ``version`` field reaches the same conclusion one step earlier:
    ``_coerce_snapshot_version`` raises the contract error itself, so an
    unreadable version leaves this boundary as the same class the field
    coercions are converted into here.

    ``InvalidScopeComponentError``'s own bases are deliberately left alone: it
    is also raised for live, caller-supplied components (``workspace.py``,
    ``sandbox_keys.py``), where being a ``ValueError`` is correct and is what
    surrounding handlers already expect.
    """

    MALFORMED_AGENT_CONFIG = {"execution_scope": {"workspace_segments": 5}}

    def _loader(self, task_id):
        return execution_scope_from_agent_config(self.MALFORMED_AGENT_CONFIG)

    def test_direct_decode_raises_the_contract_error(self):
        with pytest.raises(ExecutionScopeResolverContractError):
            execution_scope_from_agent_config(self.MALFORMED_AGENT_CONFIG)

    def test_decode_failure_chains_the_underlying_validation_error(self):
        """The conversion must not lose the diagnosis: the coercion error is
        kept as ``__cause__`` (and its field-level detail is logged where it
        was raised), so the boundary type costs nothing in debuggability."""
        with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
            execution_scope_from_agent_config(self.MALFORMED_AGENT_CONFIG)
        assert isinstance(exc_info.value.__cause__, InvalidScopeComponentError)

    def test_decode_failure_message_names_no_snapshot_value(self):
        """The message travels further than the log does, and a snapshot's
        segments can carry end-user identifiers."""
        agent_config = {
            "execution_scope": {"workspace_segments": "tenant-secret-identifier"}
        }
        with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
            execution_scope_from_agent_config(agent_config)
        assert "tenant-secret-identifier" not in str(exc_info.value)

    def test_not_folded_on_the_no_resolver_path(self):
        set_execution_scope_snapshot_loader(self._loader)
        with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
            resolve_execution_scope("1")
        assert not isinstance(exc_info.value, _WEBSOCKET_VALIDATION_EXCEPT_TUPLE)

    def test_not_folded_on_the_resolver_abstention_path(self):
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(ExecutionScope()),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(self._loader)
        with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
            resolve_execution_scope("1")
        assert not isinstance(exc_info.value, _WEBSOCKET_VALIDATION_EXCEPT_TUPLE)

    def test_unreadable_version_decode_is_not_folded_either(self):
        """The ``version`` field takes the shorter route -- it raises the
        contract error directly rather than being converted here -- so this
        pins that the boundary hands out the same class for it, and that an
        unreadable version fails the decode instead of being read as a legacy
        snapshot."""
        agent_config = {"execution_scope": {"version": "not-a-version"}}
        with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
            execution_scope_from_agent_config(agent_config)
        assert not isinstance(exc_info.value, _WEBSOCKET_VALIDATION_EXCEPT_TUPLE)

    def test_live_component_validation_is_still_a_value_error(self):
        """The counterpart the boundary conversion must not disturb: a
        component validated for a caller right now (not read back from
        storage) still raises the ``ValueError`` subclass its own callers
        catch."""
        assert issubclass(InvalidScopeComponentError, ValueError)
        with pytest.raises(InvalidScopeComponentError):
            ExecutionScope(sandbox_key_suffix="not:valid")


class TestResolverContractErrorMessageIsClientSafe:
    """``ExecutionScopeResolverContractError``'s message reaches a generic
    handler and can surface to the client, exactly like
    ``ExecutionScopeAuthorityError``'s (which is pinned to task id + field
    names by ``test_str_carries_task_id_and_field_names_never_values``). A
    resolver or loader bug that returns an internal object must therefore not
    publish that object's ``repr()``; the message names its *type* and the
    value goes only to the server-side log.
    """

    class _InternalConfig:
        """Stands in for whatever an embedder's resolver might wrongly return."""

        def __repr__(self) -> str:
            return "<InternalConfig token='resolver-secret-token'>"

    def test_resolver_return_type_error_names_the_type_not_the_value(self):
        offender = self._InternalConfig()
        set_execution_scope_resolver(
            lambda task_id: offender, acknowledges_snapshot_candidate_contract=True
        )
        with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
            resolve_execution_scope("1")
        message = str(exc_info.value)
        assert "resolver-secret-token" not in message
        assert "_InternalConfig" in message

    def test_snapshot_type_error_names_the_type_not_the_value(self):
        offender = self._InternalConfig()
        set_execution_scope_resolver(
            lambda task_id: ExecutionScope(sandbox_key_suffix="from-resolver"),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: offender)
        with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
            resolve_execution_scope("1")
        message = str(exc_info.value)
        assert "resolver-secret-token" not in message
        assert "_InternalConfig" in message


class TestSnapshotCandidateAuthority:
    """Full resolver x snapshot precedence matrix (#296).

    Axes: resolver registered x (ExecutionScope / None / DeferToSnapshot) x
    (snapshot equal / namespace-differing / policy-only-differing / absent /
    stale-version / loader-broken). Plus the "no resolver registered" golden
    (unchanged from the resolver-less contract) and the resolver-exception
    short-circuit.
    """

    RESOLVER_SCOPE = ExecutionScope(sandbox_key_suffix="from-resolver")

    def _register(self, resolver, loader=None):
        set_execution_scope_resolver(
            resolver, acknowledges_snapshot_candidate_contract=True
        )
        if loader is not None:
            set_execution_scope_snapshot_loader(loader)

    # --- resolver returns None: authoritative unscoped -------------------
    def test_resolver_none_is_authoritative_unscoped_even_with_snapshot(self):
        loader_calls = []

        def loader(task_id):
            loader_calls.append(task_id)
            return self.RESOLVER_SCOPE

        self._register(lambda task_id: None, loader)
        assert resolve_execution_scope("1") is None
        assert loader_calls == []  # None short-circuits before the snapshot

    def test_resolver_none_logs_that_it_de_scoped_the_task(self):
        """De-scoping a task must leave a trace, like every other branch that
        overrides or ignores a candidate. A resolver that does not yet
        recognise older tasks returns ``None`` for all of them, and without
        this line every one of those turns runs unscoped -- possibly past a
        persisted snapshot that named a namespace -- with nothing in the log
        to show it happened.

        ``INFO``, because the resolver is exercising its contract rather than
        failing, and the line says the snapshot was bypassed *unread*: this
        branch deliberately never calls the loader, which the assertion below
        pins so the wording cannot drift into claiming knowledge of whether a
        snapshot exists."""
        loader_calls = []

        def loader(task_id):
            loader_calls.append(task_id)
            return self.RESOLVER_SCOPE

        self._register(lambda task_id: None, loader)
        with scope_log_records(logging.INFO) as records:
            assert resolve_execution_scope("task-7") is None
        assert loader_calls == []
        assert any(
            "task-7" in r and "None" in r and "snapshot" in r for r in records
        ), records

    # --- resolver returns ExecutionScope -----------------------------------
    def test_resolver_scope_with_no_snapshot(self):
        self._register(lambda task_id: self.RESOLVER_SCOPE, lambda task_id: None)
        assert resolve_execution_scope("1") == self.RESOLVER_SCOPE

    def test_resolver_scope_with_equal_snapshot_corroborates_silently(self):
        self._register(
            lambda task_id: self.RESOLVER_SCOPE, lambda task_id: self.RESOLVER_SCOPE
        )
        assert resolve_execution_scope("1") == self.RESOLVER_SCOPE

    def test_mount_authority_compares_effective_value_not_raw_field(self):
        """A resolver-built scope that leaves the mount at its default
        (``None``, meaning the full ``workspace_segments``) and a snapshot
        that explicitly repeats those same segments as its mount select the
        identical mount -- only the raw ``sandbox_mount_segments`` attribute
        differs (``None`` vs. an explicit tuple), not the namespace value.
        Comparing the raw attribute here would be a false-positive authority
        conflict: a client-supplied snapshot that happens to spell out the
        workspace segments as its mount must not fail an otherwise-identical
        turn. A snapshot whose mount genuinely covers different segments
        must still raise."""
        resolver_scope = ExecutionScope(workspace_segments=("a",))
        same_effective_mount_snapshot = ExecutionScope(
            workspace_segments=("a",), sandbox_mount_segments=("a",)
        )
        self._register(
            lambda task_id: resolver_scope,
            lambda task_id: same_effective_mount_snapshot,
        )
        assert resolve_execution_scope("1") == resolver_scope

        different_effective_mount_snapshot = ExecutionScope(
            workspace_segments=("a", "b"), sandbox_mount_segments=("a",)
        )
        self._register(
            lambda task_id: ExecutionScope(workspace_segments=("a", "b")),
            lambda task_id: different_effective_mount_snapshot,
        )
        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope("1")
        assert "sandbox_mount_segments" in exc_info.value.mismatched_fields

    @pytest.mark.parametrize(
        "field_name,resolver_kwargs,snapshot_kwargs",
        [
            ("sandbox_key_suffix", {}, {"sandbox_key_suffix": "other"}),
            # workspace_segments alone would also move the *effective* mount
            # (``None`` means "the full segments"), reporting two fields; both
            # sides pin the mount at the user root so only the segments vary.
            (
                "workspace_segments",
                {"workspace_segments": ("a",), "sandbox_mount_segments": ()},
                {"workspace_segments": ("b",), "sandbox_mount_segments": ()},
            ),
            (
                "sandbox_mount_segments",
                {
                    "workspace_segments": ("a", "b"),
                    "sandbox_mount_segments": ("a", "b"),
                },
                {
                    "workspace_segments": ("a", "b"),
                    "sandbox_mount_segments": ("a",),
                },
            ),
            ("memory_dimensions", {}, {"memory_dimensions": {"k": "v"}}),
            ("isolate_external_dirs", {}, {"isolate_external_dirs": True}),
        ],
    )
    def test_namespace_field_mismatch_fails_the_turn(
        self, field_name, resolver_kwargs, snapshot_kwargs
    ):
        """Each case varies exactly one namespace field, so the reported
        mismatch set pins that field alone rather than merely containing it --
        a diff loop that dropped a field would otherwise still pass on a case
        whose second, incidentally-varied field kept the set non-empty."""
        resolver_scope = ExecutionScope(**resolver_kwargs)
        snapshot_scope = ExecutionScope(**snapshot_kwargs)
        self._register(lambda task_id: resolver_scope, lambda task_id: snapshot_scope)

        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope("1")
        assert set(exc_info.value.mismatched_fields) == {field_name}
        assert exc_info.value.resolver_scope == resolver_scope
        assert exc_info.value.snapshot_scope == snapshot_scope

    def test_namespace_mismatch_error_omits_a_concurrently_differing_policy_field(
        self,
    ):
        """The raise is gated on the namespace diff alone being non-empty,
        but strict_memory_isolation (policy) also differs here. The
        exception reaches task.error_message and a client-facing event, so
        mismatched_fields must carry only the namespace disagreement --
        listing the policy field there would misleadingly present it as
        part of the conflict that failed the turn."""
        resolver_scope = ExecutionScope(strict_memory_isolation=False)
        snapshot_scope = ExecutionScope(
            sandbox_key_suffix="other", strict_memory_isolation=True
        )
        self._register(lambda task_id: resolver_scope, lambda task_id: snapshot_scope)

        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope("1")
        assert set(exc_info.value.mismatched_fields) == {"sandbox_key_suffix"}
        assert "strict_memory_isolation" not in exc_info.value.mismatched_fields

    def test_policy_only_mismatch_does_not_fail_the_turn(self):
        """strict_memory_isolation is a policy field: a disagreement there
        does not change which key/path is touched, so the resolver's value
        wins without raising. The disagreement must still be observable --
        a silently-won policy mismatch would otherwise be undebuggable."""
        resolver_scope = ExecutionScope(strict_memory_isolation=False)
        snapshot_scope = ExecutionScope(strict_memory_isolation=True)
        self._register(lambda task_id: resolver_scope, lambda task_id: snapshot_scope)

        with scope_log_records() as records:
            assert resolve_execution_scope("1") == resolver_scope
        assert any(
            "policy-only mismatch" in r and "strict_memory_isolation" in r
            for r in records
        )

    def test_stale_version_snapshot_is_ignored_even_if_it_would_mismatch(self):
        resolver_scope = ExecutionScope()
        stale_data = ExecutionScope(sandbox_key_suffix="stale").to_dict()
        del stale_data["version"]  # predates EXECUTION_SCOPE_SHAPE_VERSION
        stale_snapshot = ExecutionScope.from_dict(stale_data)
        assert stale_snapshot.version == 0

        self._register(lambda task_id: resolver_scope, lambda task_id: stale_snapshot)
        with scope_log_records() as records:
            assert resolve_execution_scope("1") == resolver_scope
        assert any("shape" in r and "sandbox_key_suffix" in r for r in records)

    def test_newer_version_snapshot_is_ignored_too(self):
        """A mixed-version rollout can also see a snapshot stamped by a
        *newer* process than this one (e.g. during a rolling deploy where
        some workers already run the next shape). ``!=`` (not ``<``) covers
        this direction too: the snapshot's shape can't be safely compared
        field-by-field against a scope built under a different shape,
        regardless of which side is newer."""
        resolver_scope = ExecutionScope()
        newer_data = ExecutionScope(sandbox_key_suffix="from-the-future").to_dict()
        newer_data["version"] = EXECUTION_SCOPE_SHAPE_VERSION + 1
        newer_snapshot = ExecutionScope.from_dict(newer_data)

        self._register(lambda task_id: resolver_scope, lambda task_id: newer_snapshot)
        assert resolve_execution_scope("1") == resolver_scope

    def test_broken_snapshot_loader_does_not_veto_resolver_scope(self):
        def boom(task_id):
            raise RuntimeError("db down")

        self._register(lambda task_id: self.RESOLVER_SCOPE, boom)
        assert resolve_execution_scope("1") == self.RESOLVER_SCOPE

    def test_snapshot_wrong_type_raises_contract_error_not_attribute_error(self):
        """A snapshot loader is held to the same return-type discipline as
        the resolver. Without the shared validation funnel, a non-scope
        candidate would reach the ``.version`` comparison below and raise a
        bare, undiagnosable ``AttributeError`` instead."""
        self._register(
            lambda task_id: self.RESOLVER_SCOPE,
            lambda task_id: {"not": "a scope"},
        )
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope("1")

    # --- resolver returns DeferToSnapshot -----------------------------------
    def test_defer_uses_snapshot_when_present(self):
        """A snapshot that narrows a *scoped* fallback (not an all-default
        one -- see TestDeferSnapshotAllDefaultFallback below) is used as-is.
        Namespace and policy fields are varied independently elsewhere
        (test_defer_snapshot_policy_only_disagreement_fallback_wins) so this
        case stays a clean single-variable proof of "narrowing snapshot
        wins"."""
        fallback = ExecutionScope(workspace_segments=("tenant-a",))
        snapshot_scope = ExecutionScope(workspace_segments=("tenant-a", "sub"))
        self._register(
            lambda task_id: DeferToSnapshot(fallback),
            lambda task_id: snapshot_scope,
        )
        assert resolve_execution_scope("1") == snapshot_scope

    def test_defer_snapshot_policy_only_disagreement_fallback_wins(self):
        """On the abstention branch, the namespace fields agree (both
        all-default -- no narrowing violation), but
        strict_memory_isolation differs. The fallback plays the
        authoritative role here, so its policy value must win, logged, the
        same way the resolver's policy value wins on the authoritative
        branch (test_policy_only_mismatch_does_not_fail_the_turn above)."""
        fallback = ExecutionScope(strict_memory_isolation=True)
        snapshot_scope = ExecutionScope(strict_memory_isolation=False)
        self._register(
            lambda task_id: DeferToSnapshot(fallback),
            lambda task_id: snapshot_scope,
        )

        with scope_log_records() as records:
            result = resolve_execution_scope("1")
        assert result.strict_memory_isolation is True
        assert result.sandbox_key_suffix is None
        assert any(
            "policy-only mismatch" in r and "strict_memory_isolation" in r
            for r in records
        )

    def test_defer_policy_overlay_keeps_the_snapshot_narrowed_namespace(self):
        """The policy overlay must be applied *onto the snapshot*, not
        satisfied by returning the fallback. Here the snapshot legitimately
        narrows the fallback's namespace and also disagrees on policy: both
        halves have to survive, so the resolved scope is the snapshot's
        namespace carrying the fallback's policy value. Returning
        ``fallback`` instead would silently throw the narrowing away, which
        every all-default-namespace policy test is blind to."""
        fallback = ExecutionScope(
            workspace_segments=("tenant-a",), strict_memory_isolation=True
        )
        snapshot_scope = ExecutionScope(
            workspace_segments=("tenant-a", "sub"), strict_memory_isolation=False
        )
        self._register(
            lambda task_id: DeferToSnapshot(fallback),
            lambda task_id: snapshot_scope,
        )

        result = resolve_execution_scope("1")
        assert result.workspace_segments == ("tenant-a", "sub")
        assert result.strict_memory_isolation is True

    def test_defer_every_policy_field_is_taken_from_the_fallback(self, monkeypatch):
        """Generic over ``_EXECUTION_SCOPE_POLICY_FIELDS``: the abstention
        branch must compare and carry over every field the table lists, not a
        field it names literally. With one policy field in production those
        two implementations are indistinguishable, so this reclassifies a
        second real field into the policy bucket for the duration of the test
        and checks that it is honoured too.
        """
        from xagent.core import execution_scope as scope_module

        borrowed = "isolate_external_dirs"
        monkeypatch.setattr(
            scope_module,
            "_EXECUTION_SCOPE_NAMESPACE_FIELDS",
            tuple(
                name
                for name in scope_module._EXECUTION_SCOPE_NAMESPACE_FIELDS
                if name != borrowed
            ),
        )
        monkeypatch.setattr(
            scope_module,
            "_EXECUTION_SCOPE_POLICY_FIELDS",
            scope_module._EXECUTION_SCOPE_POLICY_FIELDS + (borrowed,),
        )
        policy_fields = scope_module._EXECUTION_SCOPE_POLICY_FIELDS
        assert len(policy_fields) > 1

        fallback = ExecutionScope(strict_memory_isolation=True, **{borrowed: True})
        snapshot_scope = ExecutionScope(
            strict_memory_isolation=False, **{borrowed: False}
        )
        self._register(
            lambda task_id: DeferToSnapshot(fallback),
            lambda task_id: snapshot_scope,
        )

        with scope_log_records() as records:
            result = resolve_execution_scope("1")
        for name in policy_fields:
            assert getattr(result, name) == getattr(fallback, name), name
            assert any("policy-only mismatch" in r and name in r for r in records), name

    def test_defer_current_shape_version_snapshot_is_used_not_shape_gated(self):
        """The positive mirror of
        test_defer_stale_version_snapshot_is_ignored_falls_back: a snapshot
        carrying the current shape version (as a real persisted snapshot
        would, round-tripped through to_dict/from_dict) is not shape-gated
        away and is used when it narrows the fallback."""
        fallback = ExecutionScope(workspace_segments=("tenant-a",))
        snapshot_data = ExecutionScope(workspace_segments=("tenant-a", "sub")).to_dict()
        assert snapshot_data["version"] == EXECUTION_SCOPE_SHAPE_VERSION
        snapshot_scope = ExecutionScope.from_dict(snapshot_data)

        self._register(
            lambda task_id: DeferToSnapshot(fallback),
            lambda task_id: snapshot_scope,
        )
        assert resolve_execution_scope("1") == snapshot_scope

    def test_defer_uses_fallback_when_snapshot_absent(self):
        fallback = ExecutionScope(strict_memory_isolation=True)
        self._register(lambda task_id: DeferToSnapshot(fallback), lambda task_id: None)
        assert resolve_execution_scope("1") == fallback

    def test_defer_loader_exception_propagates(self):
        """Unlike the authoritative branch (``test_broken_snapshot_loader_
        does_not_veto_resolver_scope`` above), a broken loader here must fail
        the turn: the resolver has just said it does not know this task's
        scope, so a broken candidate means nobody knows, and turn faces that
        reach this branch must fail closed rather than silently proceed
        under a possibly-wrong fallback."""

        def boom(task_id):
            raise RuntimeError("db down")

        fallback = ExecutionScope(strict_memory_isolation=True)
        self._register(lambda task_id: DeferToSnapshot(fallback), boom)
        with pytest.raises(RuntimeError, match="db down"):
            resolve_execution_scope("1")

    def test_defer_snapshot_wrong_type_raises_contract_error(self):
        """A snapshot loader is held to the same return-type discipline as
        the resolver: it must return an ExecutionScope or None."""
        fallback = ExecutionScope(strict_memory_isolation=True)
        self._register(
            lambda task_id: DeferToSnapshot(fallback),
            lambda task_id: {"not": "a scope"},
        )
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope("1")

    def test_defer_snapshot_carrier_from_loader_raises_contract_error(self):
        """A loader returning a DeferToSnapshot (instead of an actual
        snapshot) is exactly as invalid as a raw dict -- both fail the same
        isinstance(ExecutionScope) gate."""
        fallback = ExecutionScope(strict_memory_isolation=True)
        self._register(
            lambda task_id: DeferToSnapshot(fallback),
            lambda task_id: DeferToSnapshot(ExecutionScope()),
        )
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope("1")

    def test_defer_stale_version_snapshot_is_ignored_falls_back(self):
        fallback = ExecutionScope(sandbox_key_suffix="fallback")
        stale_data = ExecutionScope(sandbox_key_suffix="stale").to_dict()
        del stale_data["version"]
        stale_snapshot = ExecutionScope.from_dict(stale_data)
        assert stale_snapshot.version == 0

        self._register(
            lambda task_id: DeferToSnapshot(fallback),
            lambda task_id: stale_snapshot,
        )
        with scope_log_records() as records:
            assert resolve_execution_scope("1") == fallback
        assert any("shape" in r and "sandbox_key_suffix" in r for r in records)

    @pytest.mark.parametrize(
        "field_name,fallback_kwargs,snapshot_kwargs",
        [
            (
                "sandbox_key_suffix",
                {"sandbox_key_suffix": "fallback-suffix"},
                {"sandbox_key_suffix": "different-suffix"},
            ),
            (
                "workspace_segments",
                {"workspace_segments": ("tenant-a", "b")},
                {"workspace_segments": ("tenant-a",)},
            ),
            (
                "sandbox_mount_segments",
                {
                    "workspace_segments": ("tenant-a", "b", "c"),
                    "sandbox_mount_segments": ("tenant-a", "b", "c"),
                },
                {
                    "workspace_segments": ("tenant-a", "b", "c"),
                    "sandbox_mount_segments": ("tenant-a",),
                },
            ),
            (
                "isolate_external_dirs",
                {"isolate_external_dirs": True},
                {"isolate_external_dirs": False},
            ),
            (
                "memory_dimensions",
                {"memory_dimensions": {"k": "v"}},
                {"memory_dimensions": {}},
            ),
            # All-default fallback: the resolver has claimed no authority in
            # any dimension, so introducing scoping the fallback never
            # committed to is rejected even though, taken alone, it would
            # look like a narrowing (null -> set; False -> True). See
            # TestDeferSnapshotAllDefaultFallback for the case that varies
            # every field of such a fallback at once.
            ("sandbox_key_suffix", {}, {"sandbox_key_suffix": "attacker-chosen"}),
            ("isolate_external_dirs", {}, {"isolate_external_dirs": True}),
        ],
    )
    def test_defer_snapshot_widening_fallback_fails_the_turn(
        self, field_name, fallback_kwargs, snapshot_kwargs
    ):
        """A snapshot that is *wider* than the resolver's mandatory fallback
        on any namespace field must fail the turn: the snapshot remains
        untrusted input even though request bodies can no longer seed it --
        a runtime-extension provider holding a ``session_factory`` can write
        ``Task.agent_config`` directly, and a row persisted before
        ``execution_scope`` was reserved at the request boundary still
        carries whatever a request seeded at the time -- so an unchecked
        snapshot here would let a caller widen its own namespace past the
        resolver's own most conservative answer."""
        fallback = ExecutionScope(**fallback_kwargs)
        snapshot = ExecutionScope(**snapshot_kwargs)
        self._register(
            lambda task_id: DeferToSnapshot(fallback),
            lambda task_id: snapshot,
        )

        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope("1")
        assert field_name in exc_info.value.mismatched_fields
        assert exc_info.value.resolver_scope == fallback
        assert exc_info.value.snapshot_scope == snapshot

    @pytest.mark.parametrize(
        "fallback_kwargs,snapshot_kwargs",
        [
            (
                {"workspace_segments": ("tenant-a",)},
                {"workspace_segments": ("tenant-a", "sub")},
            ),
            (
                {
                    "workspace_segments": ("tenant-a", "b", "c"),
                    "sandbox_mount_segments": ("tenant-a",),
                },
                {
                    "workspace_segments": ("tenant-a", "b", "c"),
                    "sandbox_mount_segments": ("tenant-a", "b"),
                },
            ),
            (
                {"memory_dimensions": {"k": "v"}},
                {"memory_dimensions": {"k": "v", "k2": "v2"}},
            ),
            (
                {"sandbox_key_suffix": "same"},
                {"sandbox_key_suffix": "same"},
            ),
        ],
    )
    def test_defer_snapshot_narrowing_fallback_is_accepted(
        self, fallback_kwargs, snapshot_kwargs
    ):
        """The mirror of the widening cases above: a snapshot that only
        extends scoping the fallback already committed to (a deeper
        workspace path, a deeper mount prefix within one, or extra memory
        dimensions) is accepted and used as the resolved scope. Every case
        here has a non-default fallback for the field under test -- see
        TestDeferSnapshotAllDefaultFallback for why an all-default fallback
        accepts nothing but an equal snapshot instead."""
        fallback = ExecutionScope(**fallback_kwargs)
        snapshot = ExecutionScope(**snapshot_kwargs)
        self._register(
            lambda task_id: DeferToSnapshot(fallback),
            lambda task_id: snapshot,
        )
        assert resolve_execution_scope("1") == snapshot

    # --- boundary judgment / resolver misbehavior ---------------------------
    def test_resolver_returning_unexpected_type_raises_contract_error(self):
        self._register(lambda task_id: "not-a-scope")
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope("1")

    def test_resolver_exception_short_circuits_before_the_snapshot(self):
        loader_calls = []

        def boom(task_id):
            raise RuntimeError("resolver down")

        self._register(boom, lambda task_id: loader_calls.append(task_id))
        with pytest.raises(RuntimeError, match="resolver down"):
            resolve_execution_scope("1")
        assert loader_calls == []

    # --- no resolver registered: golden: unchanged resolver-less contract --------
    def test_no_resolver_registered_snapshot_alone_drives(self):
        set_execution_scope_snapshot_loader(lambda task_id: self.RESOLVER_SCOPE)
        assert resolve_execution_scope("1") == self.RESOLVER_SCOPE

    def test_no_resolver_registered_no_snapshot_is_unscoped(self):
        assert resolve_execution_scope("1") is None

    def test_no_resolver_registered_loader_exception_propagates(self):
        def boom(task_id):
            raise RuntimeError("db down")

        set_execution_scope_snapshot_loader(boom)
        with pytest.raises(RuntimeError, match="db down"):
            resolve_execution_scope("1")

    def test_no_resolver_registered_stale_version_snapshot_is_used_as_is(self):
        """The no-resolver branch returns whatever the loader returns
        directly (see resolve_execution_scope's ``if
        _execution_scope_resolver is None`` branch) -- it never routes
        through ``_validate_execution_scope_snapshot_candidate``, so a
        stale-shape snapshot is not shape-gated here the way it is on the
        other two branches. This pins that pre-existing, byte-identical-to-
        pre-authority behavior; it is not a new contract."""
        stale_data = ExecutionScope(sandbox_key_suffix="stale").to_dict()
        del stale_data["version"]
        stale_snapshot = ExecutionScope.from_dict(stale_data)
        assert stale_snapshot.version == 0

        set_execution_scope_snapshot_loader(lambda task_id: stale_snapshot)
        assert resolve_execution_scope("1") == stale_snapshot

    def test_no_resolver_registered_non_scope_loader_output_passes_through(self):
        """Same branch, same absence of validation: a loader returning
        something other than an ExecutionScope is not caught here (unlike
        the resolver-registered branches, which route every loaded
        candidate through ``_validate_execution_scope_snapshot_candidate``
        and raise ExecutionScopeResolverContractError for exactly this
        input). Pinned as a known, pre-existing gap on this branch rather
        than asserted correct."""
        set_execution_scope_snapshot_loader(lambda task_id: {"not": "a scope"})
        assert resolve_execution_scope("1") == {"not": "a scope"}


class TestDeferSnapshotAllDefaultFallback:
    """A namespace field sitting at its own identity default admits no
    narrowing, only an exact match: there is no scoping there for a snapshot
    to narrow into.

    This is what a purely relative per-field test cannot express. Evaluated
    only against the fallback's own value, every narrowing relation is
    vacuously satisfied once that value is the field's no-scoping default --
    so an all-default ``ExecutionScope()`` fallback, the natural value for a
    resolver with no opinion on a task, would accept any snapshot at all.
    """

    def test_all_default_fallback_rejects_a_snapshot_widening_every_field_at_once(
        self,
    ):
        """The worst case for a vacuous predicate: a fully unscoped fallback
        plus a snapshot that sets every namespace field simultaneously. Every
        field is out-of-authority for this fallback, so all five must be
        reported."""
        fallback = ExecutionScope()
        snapshot = ExecutionScope(
            sandbox_key_suffix="attacker-chosen",
            workspace_segments=("other", "tenant"),
            sandbox_mount_segments=("other", "tenant"),
            isolate_external_dirs=True,
            memory_dimensions={"tenant": "other"},
        )
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: snapshot)

        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope("1")
        assert set(exc_info.value.mismatched_fields) == {
            "sandbox_key_suffix",
            "workspace_segments",
            "sandbox_mount_segments",
            "isolate_external_dirs",
            "memory_dimensions",
        }

    def test_all_default_fallback_accepts_an_equal_snapshot(self):
        """The permissive half: an all-default snapshot introduces no scoping,
        so it passes the identity rule and is what gets returned.

        Asserted by identity, not equality: an all-default fallback and an
        all-default snapshot compare equal, so ``== fallback`` would hold
        whichever of the two the function returned -- and would keep holding
        if the snapshot were rejected or shape-gated away. ``is snapshot``
        distinguishes the accept path from both."""
        fallback = ExecutionScope()
        snapshot = ExecutionScope()
        assert snapshot is not fallback
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: snapshot)
        assert resolve_execution_scope("1") is snapshot


class TestDeferSnapshotDefinesNamespaceOptIn:
    """``DeferToSnapshot(fallback, snapshot_defines_namespace=True)``: the
    resolver states it does not own this task and the persisted snapshot is
    the intended namespace authority (the workforce/delegated-task shape).
    A conforming ``fallback`` claims no namespace of its own -- if the
    resolver knew the namespace it would return an authoritative
    ``ExecutionScope`` instead of deferring -- which is enforced at
    construction, so the strict narrowing rule in
    ``TestDeferSnapshotAllDefaultFallback`` above has nothing left to protect
    and is skipped for namespace fields here. Everything else still applies:
    the type check, the mandatory fallback, policy symmetry, and the
    newer-shape half of the version gate. Only the older-shape half is
    relaxed, since this branch compares nothing.
    """

    def test_opt_in_with_all_default_fallback_uses_snapshot_verbatim(self):
        """The workforce shape: the resolver has no opinion (all-default
        fallback) but explicitly hands namespace authority to the snapshot."""
        fallback = ExecutionScope()
        snapshot = ExecutionScope(
            sandbox_key_suffix="workforce-task",
            workspace_segments=("workforce", "task-7"),
            memory_dimensions={"tenant": "acme"},
        )
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback, snapshot_defines_namespace=True),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: snapshot)
        assert resolve_execution_scope("1") == snapshot

    def test_without_opt_in_the_same_snapshot_fails_closed(self):
        """The security half: the identical snapshot/fallback pair, without
        the opt-in, must still fail closed under the strict narrowing rule
        -- the opt-in is what changes the outcome, not the snapshot shape."""
        fallback = ExecutionScope()
        snapshot = ExecutionScope(
            sandbox_key_suffix="workforce-task",
            workspace_segments=("workforce", "task-7"),
            memory_dimensions={"tenant": "acme"},
        )
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: snapshot)

        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope("1")
        assert set(exc_info.value.mismatched_fields) == {
            "sandbox_key_suffix",
            "workspace_segments",
            "sandbox_mount_segments",
            "memory_dimensions",
        }

    def test_opt_in_accepts_an_older_shape_snapshot_and_keeps_its_data(self):
        """The older half of the asymmetric shape gate. Every field an
        older-shape snapshot carries is decodable here and the ones it lacks
        take current defaults, so the only thing its version breaks is a
        field-by-field comparison -- and this branch performs none. Gating it
        away would de-scope a workforce sub-task purely because its snapshot
        predates the version field."""
        fallback = ExecutionScope()
        older_data = ExecutionScope(
            sandbox_key_suffix="older", workspace_segments=("workforce", "task-7")
        ).to_dict()
        del older_data["version"]
        older_snapshot = ExecutionScope.from_dict(older_data)
        assert older_snapshot.version == 0

        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback, snapshot_defines_namespace=True),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: older_snapshot)
        assert resolve_execution_scope("1") == older_snapshot

    def test_opt_in_still_refuses_a_newer_shape_snapshot(self):
        """The newer half, which the opt-in does *not* relax.
        ``from_dict`` drops keys this shape does not know, so a snapshot
        written by a newer process arrives partially decoded -- possibly with
        a namespace-narrowing field gone -- and "used verbatim" would mean
        activating that truncated namespace. ``to_dict`` then stamps the
        current constant, so accepting it would also re-persist the
        truncation as if it were current."""
        fallback = ExecutionScope()
        newer_data = ExecutionScope(
            sandbox_key_suffix="newer", workspace_segments=("workforce", "task-7")
        ).to_dict()
        newer_data["version"] = EXECUTION_SCOPE_SHAPE_VERSION + 1
        newer_data["a_field_this_shape_does_not_know"] = "dropped-on-decode"
        newer_snapshot = ExecutionScope.from_dict(newer_data)
        assert newer_snapshot.version == EXECUTION_SCOPE_SHAPE_VERSION + 1

        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback, snapshot_defines_namespace=True),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: newer_snapshot)
        with scope_log_records() as records:
            assert resolve_execution_scope("1") == fallback
        assert any("shape" in r and "sandbox_key_suffix" in r for r in records)

    def test_default_abstention_path_refuses_a_newer_shape_snapshot(self):
        """The same newer-shape refusal on the comparing abstention path,
        where the fallback's own namespace is what the snapshot would have
        had to narrow."""
        fallback = ExecutionScope(workspace_segments=("tenant-a",))
        newer_data = ExecutionScope(workspace_segments=("tenant-a", "sub")).to_dict()
        newer_data["version"] = EXECUTION_SCOPE_SHAPE_VERSION + 1
        newer_snapshot = ExecutionScope.from_dict(newer_data)

        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: newer_snapshot)
        assert resolve_execution_scope("1") == fallback

    def test_opt_in_still_type_gates_a_non_scope_snapshot(self):
        """The older-shape relaxation does not extend to the type check -- a
        candidate that isn't even an ExecutionScope still raises, opt-in or
        not."""
        fallback = ExecutionScope()
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback, snapshot_defines_namespace=True),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: object())
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope("1")

    def test_opt_in_rejects_a_namespace_committed_fallback_at_construction(self):
        """The opt-in's precondition is machine-checked, not prose. This
        branch hands the namespace to the snapshot verbatim, so a fallback
        that already committed to one would have it *replaced* rather than
        treated as a floor -- silent de-tenanting. A resolver that knows the
        namespace must return an authoritative scope instead of deferring,
        so the carrier refuses to exist."""
        committed = ExecutionScope(
            sandbox_key_suffix="tenant-a",
            workspace_segments=("tenant-a",),
            memory_dimensions={"tenant": "a"},
        )
        with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
            DeferToSnapshot(committed, snapshot_defines_namespace=True)
        message = str(exc_info.value)
        for name in ("sandbox_key_suffix", "workspace_segments", "memory_dimensions"):
            assert name in message
        # The same pair is accepted without the opt-in, where the snapshot is
        # held to the narrowing rule instead: it is the flag that is refused,
        # not the fallback.
        assert DeferToSnapshot(committed).fallback == committed

    @pytest.mark.parametrize(
        "field_name,committed_kwargs",
        [
            ("sandbox_key_suffix", {"sandbox_key_suffix": "tenant-a"}),
            ("workspace_segments", {"workspace_segments": ("tenant-a",)}),
            (
                "sandbox_mount_segments",
                {
                    "workspace_segments": ("tenant-a",),
                    "sandbox_mount_segments": ("tenant-a",),
                },
            ),
            ("memory_dimensions", {"memory_dimensions": {"tenant": "a"}}),
            ("isolate_external_dirs", {"isolate_external_dirs": True}),
        ],
    )
    def test_opt_in_rejects_a_commitment_on_any_single_namespace_field(
        self, field_name, committed_kwargs
    ):
        """Per-field, so the precondition cannot be satisfied by checking only
        the obvious fields. Each case commits exactly one field of the
        fallback (the mount case necessarily carries the workspace segments
        it must prefix, and reports both)."""
        with pytest.raises(ExecutionScopeResolverContractError) as exc_info:
            DeferToSnapshot(
                ExecutionScope(**committed_kwargs), snapshot_defines_namespace=True
            )
        assert field_name in str(exc_info.value)

    def test_opt_in_with_a_policy_only_fallback_is_accepted(self):
        """Policy fields are not part of the precondition: the fallback still
        owns those on this branch (see
        test_opt_in_still_honours_policy_field_symmetry), so committing one
        must not be mistaken for claiming a namespace."""
        fallback = ExecutionScope(strict_memory_isolation=True)
        carrier = DeferToSnapshot(fallback, snapshot_defines_namespace=True)
        assert carrier.snapshot_defines_namespace is True
        assert carrier.fallback == fallback

    def test_opt_in_with_a_conforming_fallback_and_a_present_snapshot(self):
        """The whole conforming shape end to end: a fallback that claims no
        namespace but does set a policy field, plus a present snapshot that
        supplies the namespace. The snapshot's namespace is adopted and the
        fallback's policy value is kept."""
        fallback = ExecutionScope(strict_memory_isolation=True)
        snapshot = ExecutionScope(
            sandbox_key_suffix="workforce-task",
            workspace_segments=("workforce", "task-7"),
            memory_dimensions={"tenant": "acme"},
        )
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback, snapshot_defines_namespace=True),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: snapshot)

        result = resolve_execution_scope("1")
        assert result.sandbox_key_suffix == "workforce-task"
        assert result.workspace_segments == ("workforce", "task-7")
        assert dict(result.memory_dimensions) == {"tenant": "acme"}
        assert result.strict_memory_isolation is True

    def test_opt_in_still_honours_policy_field_symmetry(self):
        """Namespace fields agree (both all-default), only
        strict_memory_isolation differs: the fallback's policy value still
        wins and the disagreement is still logged, exactly as on the
        default (narrowing) branch."""
        fallback = ExecutionScope(strict_memory_isolation=True)
        snapshot = ExecutionScope(strict_memory_isolation=False)
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback, snapshot_defines_namespace=True),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: snapshot)

        with scope_log_records() as records:
            result = resolve_execution_scope("1")
        assert result.strict_memory_isolation is True
        assert any(
            "policy-only mismatch" in r and "strict_memory_isolation" in r
            for r in records
        )

    def test_flag_does_not_make_the_fallback_optional(self):
        """The opt-in changes what a present snapshot is checked against,
        not whether a fallback is required at all: ``DeferToSnapshot``
        still rejects a non-ExecutionScope fallback even with the flag set,
        and the mandatory fallback still applies on a snapshot miss.

        The fallback here commits no namespace (the opt-in's precondition) but
        does set a policy field, which keeps it distinguishable from the
        all-default scope an unscoped resolution would return."""
        with pytest.raises(TypeError):
            DeferToSnapshot("not-a-scope", snapshot_defines_namespace=True)

        fallback = ExecutionScope(strict_memory_isolation=True)
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback, snapshot_defines_namespace=True),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: None)
        assert resolve_execution_scope("1") == fallback


class TestShapeGateLogsFieldDiffLazily:
    """The shape-version gate's field diff exists only to populate its own
    log message -- unlike the narrowing/authoritative diffs, which are also
    needed for control flow, nothing downstream reads this one. It must
    therefore be computed only when that message will actually be emitted,
    since the gate runs on every turn for any snapshot that predates a
    shape bump."""

    def test_field_diff_not_computed_when_warning_is_disabled(self, monkeypatch):
        from xagent.core import execution_scope as scope_module

        def boom(*_args, **_kwargs):
            raise AssertionError(
                "field diff computed even though WARNING logging is disabled"
            )

        monkeypatch.setattr(scope_module, "_execution_scope_field_diff", boom)
        resolver_scope = ExecutionScope()
        stale_data = ExecutionScope(sandbox_key_suffix="stale").to_dict()
        del stale_data["version"]
        stale_snapshot = ExecutionScope.from_dict(stale_data)
        set_execution_scope_resolver(
            lambda task_id: resolver_scope,
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: stale_snapshot)
        with scope_log_records(level=logging.ERROR) as records:
            # Gate behavior is unaffected by the logging level: the stale
            # snapshot is still ignored and the resolver's scope wins.
            assert resolve_execution_scope("1") == resolver_scope
        assert records == []

    def test_field_diff_still_computed_and_logged_when_warning_is_enabled(self):
        """Positive mirror: with WARNING enabled (the default), the gate's
        existing observable behavior -- a log line naming the differing
        field -- is unchanged by the lazy guard."""
        resolver_scope = ExecutionScope()
        stale_data = ExecutionScope(sandbox_key_suffix="stale").to_dict()
        del stale_data["version"]
        stale_snapshot = ExecutionScope.from_dict(stale_data)
        set_execution_scope_resolver(
            lambda task_id: resolver_scope,
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: stale_snapshot)
        with scope_log_records() as records:
            assert resolve_execution_scope("1") == resolver_scope
        assert any("shape" in r and "sandbox_key_suffix" in r for r in records)


class TestResolveExecutionScopeOffTurn:
    """What decides whether a consumer may downgrade a mismatch is whether it
    is selecting a namespace for new bytes, not whether a turn is running.

    A consumer that selects one -- websocket ``_scope_segments_for_task``,
    whose caller composes the storage key for a brand-new durable object --
    resolves fail-closed through ``resolve_execution_scope`` even though it
    runs outside any turn: choosing where new bytes land is an authority
    decision that must not be downgraded to either side's guess. The
    consumers that come through this entry point instead read a namespace
    something else already committed to: ``ManagedFileRef``'s construction,
    the pause/resume handlers, and workspace cleanup. For them a
    resolver-authoritative namespace mismatch downgrades to the resolver's
    answer rather than failing an operation that has no turn left to fail.

    A mismatch whose ``resolver_scope`` is not an authoritative answer has
    nothing to downgrade to and stays fail-closed even here -- see
    ``ExecutionScopeAbstentionMismatchError``'s docstring."""

    def test_passthrough_when_no_mismatch(self):
        scope = ExecutionScope(sandbox_key_suffix="s")
        set_execution_scope_resolver(
            lambda task_id: scope, acknowledges_snapshot_candidate_contract=True
        )
        assert resolve_execution_scope_off_turn("1") == scope

    def test_authoritative_branch_mismatch_downgrades_to_resolver_value_with_warning(
        self,
    ):
        """The resolver returned a real ``ExecutionScope`` (authoritative
        branch): its value is a genuine answer, so a disagreeing snapshot's
        mismatch is downgraded to it off-turn instead of raised."""
        resolver_scope = ExecutionScope(sandbox_key_suffix="from-resolver")
        snapshot_scope = ExecutionScope(sandbox_key_suffix="from-snapshot")
        set_execution_scope_resolver(
            lambda task_id: resolver_scope,
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: snapshot_scope)

        # A real disagreement: the fail-closed entry point must reject it, or
        # the downgrade below would be indistinguishable from "nothing
        # disagreed" -- which also returns the resolver's scope.
        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope("1")
        assert not isinstance(exc_info.value, ExecutionScopeAbstentionMismatchError)

        with scope_log_records() as records:
            result = resolve_execution_scope_off_turn("1")

        assert result == resolver_scope
        assert any("authority mismatch" in r.lower() for r in records)

    def test_abstention_branch_mismatch_stays_fail_closed_off_turn(self):
        """The resolver deferred to the snapshot (abstention branch) and the
        snapshot widens the fallback: unlike the authoritative case above,
        ``resolved.fallback`` here is not an authoritative answer -- it is
        what the resolver supplied while declining to answer, all-default
        for this test. Downgrading to it off-turn would hand a caller like
        workspace cleanup an unscoped value indistinguishable from no scope
        at all, so this must propagate unchanged instead of being
        downgraded. No existing test covered this off-turn path before --
        every prior off-turn mismatch case used an authoritative resolver."""
        fallback = ExecutionScope()
        snapshot_scope = ExecutionScope(sandbox_key_suffix="attacker-chosen")
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: snapshot_scope)

        # Confirm the in-turn behavior this off-turn call must not soften.
        with pytest.raises(ExecutionScopeAbstentionMismatchError):
            resolve_execution_scope("1")

        with pytest.raises(ExecutionScopeAbstentionMismatchError):
            resolve_execution_scope_off_turn("1")

    def test_a_new_authority_error_subclass_is_not_downgraded_by_default(self):
        """The downgrade condition is positive -- "this raise site declared an
        authoritative value" -- rather than an exclusion list of the subclasses
        known to be unsafe. A subclass defined here, which no exclusion list in
        the module could name, must therefore propagate: an off-turn face that
        fails open for anything it has not been taught about would hand out
        ``resolver_scope`` values no branch ever sanctioned."""

        class _FutureMismatchError(ExecutionScopeAuthorityError):
            """A mismatch shape added after this predicate was written."""

        def resolver(task_id):
            raise _FutureMismatchError(
                str(task_id),
                resolver_scope=ExecutionScope(sandbox_key_suffix="not-an-authority"),
                snapshot_scope=ExecutionScope(sandbox_key_suffix="from-snapshot"),
                mismatched_fields={
                    "sandbox_key_suffix": ("from-snapshot", "not-an-authority")
                },
                resolver_scope_is_authoritative=False,
            )

        set_execution_scope_resolver(
            resolver, acknowledges_snapshot_candidate_contract=True
        )
        with pytest.raises(_FutureMismatchError):
            resolve_execution_scope_off_turn("1")

    def test_a_subclass_declaring_an_authority_is_still_downgraded(self):
        """The other half: the predicate keys on the declaration, not on the
        class, so a subclass that does have a sanctioned value is downgraded
        without the function needing to know the type."""

        class _FutureAuthoritativeMismatchError(ExecutionScopeAuthorityError):
            """A mismatch shape that does carry an authoritative answer."""

        authoritative = ExecutionScope(sandbox_key_suffix="from-resolver")

        def resolver(task_id):
            raise _FutureAuthoritativeMismatchError(
                str(task_id),
                resolver_scope=authoritative,
                snapshot_scope=ExecutionScope(sandbox_key_suffix="from-snapshot"),
                mismatched_fields={
                    "sandbox_key_suffix": ("from-snapshot", "from-resolver")
                },
                resolver_scope_is_authoritative=True,
            )

        set_execution_scope_resolver(
            resolver, acknowledges_snapshot_candidate_contract=True
        )
        with scope_log_records() as records:
            assert resolve_execution_scope_off_turn("1") == authoritative
        assert any("authority mismatch" in r.lower() for r in records)

    def test_other_exceptions_still_propagate(self):
        def boom(task_id):
            raise RuntimeError("resolver down")

        set_execution_scope_resolver(
            boom, acknowledges_snapshot_candidate_contract=True
        )
        with pytest.raises(RuntimeError, match="resolver down"):
            resolve_execution_scope_off_turn("1")


class TestExecutionScopeAbstentionMismatchErrorIsDistinguishable:
    """Pin: the abstention-branch error is a distinct type a caller must
    explicitly widen its ``except`` clause to catch, so an existing
    ``except ExecutionScopeAuthorityError: return exc.resolver_scope``
    downgrade path (written for the authoritative branch) cannot silently
    also catch and downgrade an abstention mismatch."""

    def test_is_a_subclass_of_the_base_authority_error(self):
        assert issubclass(
            ExecutionScopeAbstentionMismatchError, ExecutionScopeAuthorityError
        )

    def test_abstention_raise_site_declares_no_authoritative_value(self):
        """The attribute, not the class, is what blocks the off-turn
        downgrade (see ``resolve_execution_scope_off_turn``), so the
        abstention branch's raise site must declare ``False`` for it."""
        fallback = ExecutionScope()
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback),
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(
            lambda task_id: ExecutionScope(sandbox_key_suffix="attacker-chosen")
        )
        with pytest.raises(ExecutionScopeAbstentionMismatchError) as exc_info:
            resolve_execution_scope("1")
        assert exc_info.value.resolver_scope_is_authoritative is False

    def test_authoritative_raise_site_declares_an_authoritative_value(self):
        """The mirror: the branch that really has an answer says so, which is
        what licenses the off-turn downgrade."""
        resolver_scope = ExecutionScope(sandbox_key_suffix="from-resolver")
        set_execution_scope_resolver(
            lambda task_id: resolver_scope,
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(
            lambda task_id: ExecutionScope(sandbox_key_suffix="from-snapshot")
        )
        with pytest.raises(ExecutionScopeAuthorityError) as exc_info:
            resolve_execution_scope("1")
        assert exc_info.value.resolver_scope_is_authoritative is True


class TestDeferCarrierNeverActivated:
    """The carrier is never mistaken for a real scope, and the turn
    contextvar only ever holds the resolved fallback/snapshot, never the
    carrier itself."""

    def test_turn_activates_the_fallback_not_the_carrier(self):
        fallback = ExecutionScope(sandbox_key_suffix="fallback")
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback),
            acknowledges_snapshot_candidate_contract=True,
        )
        with turn_execution_scope("1") as active:
            assert active == fallback
            current = get_execution_scope()
            assert isinstance(current, ExecutionScope)
            assert isinstance(current, DeferToSnapshot) is False


class TestExecutionScopeFieldClassificationCompleteness:
    """Every dataclass field must be
    explicitly bucketed as namespace-affecting or policy-only, so a newly
    added field can't silently miss ``resolve_execution_scope``'s
    authority-mismatch classification. A field landing in neither bucket
    would never be compared at all (not even logged), which is worse than
    being misclassified into either one.
    """

    def test_namespace_field_classification_is_complete_and_disjoint_from_policy(
        self,
    ):
        """Static classification-completeness check only -- this does not
        call ``scope_fingerprint`` at all, so it claims no fingerprint
        coverage: a set-vs-set comparison cannot exercise one. Real per-field
        fingerprint behavior is covered by
        ``TestScopeFingerprintFieldCoverage`` below, which does call it."""
        from xagent.core import execution_scope as scope_module

        namespace_fields = set(scope_module._EXECUTION_SCOPE_NAMESPACE_FIELDS)
        policy_fields = set(scope_module._EXECUTION_SCOPE_POLICY_FIELDS)
        assert namespace_fields.isdisjoint(policy_fields)
        assert namespace_fields | policy_fields | {"version"} == {
            f.name for f in dataclasses.fields(ExecutionScope)
        }


class TestNarrowingViolationsFieldCoverage:
    """``_execution_scope_narrowing_violations`` must classify every
    namespace field through the shared
    ``_EXECUTION_SCOPE_NAMESPACE_FIELD_NARROWING`` table rather than a
    hardcoded per-field check, so a namespace field added to
    ``_EXECUTION_SCOPE_NAMESPACE_FIELDS`` cannot be silently skipped by this
    one consumer while the diff (``_execution_scope_field_diff``) and the
    fingerprint (``scope_fingerprint``), which already iterate their own
    tables, pick it up automatically.
    """

    def test_narrowing_table_tracks_the_namespace_fields_exactly(self):
        from xagent.core import execution_scope as scope_module

        assert set(scope_module._EXECUTION_SCOPE_NAMESPACE_FIELD_NARROWING) == set(
            scope_module._EXECUTION_SCOPE_NAMESPACE_FIELDS
        )


# One (base_kwargs, changed_kwargs) probe per namespace field: constructing
# ExecutionScope(**base_kwargs) and ExecutionScope(**changed_kwargs) changes
# only that field's contribution to scope_fingerprint(). Keyed by field name
# so TestScopeFingerprintFieldCoverage can assert this dict's keys track
# _EXECUTION_SCOPE_NAMESPACE_FIELDS exactly -- a namespace field added
# without a probe here fails loudly instead of being silently skipped.
# The ``sandbox_mount_segments`` probe holds ``workspace_segments`` fixed
# across both sides and varies only the mount, so the fingerprint difference
# it exercises comes from the mount slot alone. It does not, by itself, prove
# ``scope_fingerprint`` reads the *derived* mount value rather than the raw
# field -- see ``test_mount_fingerprint_reads_the_effective_value_not_the_raw_field``
# below for the probe that actually pins that.
_NAMESPACE_FIELD_FINGERPRINT_PROBES: dict[str, tuple[dict, dict]] = {
    "sandbox_key_suffix": ({}, {"sandbox_key_suffix": "probe-suffix"}),
    "workspace_segments": ({}, {"workspace_segments": ("probe",)}),
    "sandbox_mount_segments": (
        {
            "workspace_segments": ("probe", "deep"),
            "sandbox_mount_segments": ("probe",),
        },
        {
            "workspace_segments": ("probe", "deep"),
            "sandbox_mount_segments": None,
        },
    ),
    "memory_dimensions": ({}, {"memory_dimensions": {"k": "v"}}),
    "isolate_external_dirs": ({}, {"isolate_external_dirs": True}),
}

# Same idea for the one field the fingerprint deliberately excludes.
_POLICY_FIELD_FINGERPRINT_PROBES: dict[str, tuple[dict, dict]] = {
    "strict_memory_isolation": ({}, {"strict_memory_isolation": True}),
}


class TestScopeFingerprintFieldCoverage:
    """Behavior-derived counterpart to the set-comparison-only test above:
    calls ``scope_fingerprint()`` itself, per field, rather than comparing
    two hand-maintained collections against each other. Reverting
    ``scope_fingerprint()`` to drop any namespace field's contribution fails
    the corresponding parametrized case here directly.

    Why each bucket matters to the fingerprint: a namespace field is baked
    into per-task cached state at build time -- ``isolate_external_dirs``, for
    instance, into the cached ``AgentService``'s
    ``Workspace.allowed_external_dirs`` via
    ``AgentServiceManager.get_agent_for_task`` ->
    ``_build_allowed_external_dirs`` -- so a change there must evict the cache
    or the stale value keeps being enforced. A policy field like
    ``strict_memory_isolation`` is read fresh from the contextvar on every
    operation (``UserIsolatedMemoryStore``), so nothing cached goes stale and
    including it would only cause needless rebuilds.
    """

    def test_probes_cover_exactly_the_classified_fields(self):
        """Fixture completeness: every namespace/policy field must have a
        probe pair here, so a newly added field can't be silently skipped.
        Which bucket a field belongs to is separately pinned by
        TestExecutionScopeFieldClassificationCompleteness above; this only
        checks the probe tables track that bucketing."""
        from xagent.core import execution_scope as scope_module

        assert set(_NAMESPACE_FIELD_FINGERPRINT_PROBES) == set(
            scope_module._EXECUTION_SCOPE_NAMESPACE_FIELDS
        )
        assert set(_POLICY_FIELD_FINGERPRINT_PROBES) == set(
            scope_module._EXECUTION_SCOPE_POLICY_FIELDS
        )

    @pytest.mark.parametrize("field_name", list(_NAMESPACE_FIELD_FINGERPRINT_PROBES))
    def test_namespace_field_change_changes_the_fingerprint(self, field_name):
        base_kwargs, changed_kwargs = _NAMESPACE_FIELD_FINGERPRINT_PROBES[field_name]
        base = ExecutionScope(**base_kwargs)
        changed = ExecutionScope(**changed_kwargs)
        assert scope_fingerprint(base) != scope_fingerprint(changed)

    @pytest.mark.parametrize("field_name", list(_POLICY_FIELD_FINGERPRINT_PROBES))
    def test_policy_field_change_does_not_change_the_fingerprint(self, field_name):
        base_kwargs, changed_kwargs = _POLICY_FIELD_FINGERPRINT_PROBES[field_name]
        base = ExecutionScope(**base_kwargs)
        changed = ExecutionScope(**changed_kwargs)
        assert scope_fingerprint(base) == scope_fingerprint(changed)

    def test_mount_fingerprint_reads_the_effective_value_not_the_raw_field(self):
        """The ``!=`` probe above cannot tell a correct (derived) mount read
        apart from a regression to the raw ``sandbox_mount_segments``
        attribute: whenever the two probe scopes' raw mount values differ --
        as they must, to make the mount the only varying input -- a
        raw-reading ``scope_fingerprint`` also reports a difference, so that
        assertion passes under either implementation. This test uses the
        one pair that actually depends on which value is read: a scope left
        at the mount's default (``None``, meaning the full
        ``workspace_segments``) and a scope whose mount explicitly repeats
        those same segments have equal *effective* mounts but different
        *raw* ones, so their fingerprints must match only if
        ``scope_fingerprint`` reads ``effective_mount_segments``."""
        default_mount = ExecutionScope(workspace_segments=("a",))
        explicit_full_mount = ExecutionScope(
            workspace_segments=("a",), sandbox_mount_segments=("a",)
        )
        assert (
            default_mount.sandbox_mount_segments
            != explicit_full_mount.sandbox_mount_segments
        )
        assert (
            default_mount.effective_mount_segments
            == explicit_full_mount.effective_mount_segments
        )
        assert scope_fingerprint(default_mount) == scope_fingerprint(
            explicit_full_mount
        )


class TestAuthorityMismatchNamesItsRemediation:
    """The mismatch is stable, so every turn of the task fails the same way
    until the persisted snapshot is removed or corrected. The operator-facing
    log has to say that, because the snapshot sits in a column no request path
    offers to clear and a bare repeating failure states no way out."""

    def test_error_log_names_the_key_to_clear(self):
        resolver_scope = ExecutionScope(sandbox_key_suffix="from-resolver")
        snapshot_scope = ExecutionScope(sandbox_key_suffix="from-snapshot")
        set_execution_scope_resolver(
            lambda task_id: resolver_scope,
            acknowledges_snapshot_candidate_contract=True,
        )
        set_execution_scope_snapshot_loader(lambda task_id: snapshot_scope)

        with scope_log_records() as records:
            with pytest.raises(ExecutionScopeAuthorityError):
                resolve_execution_scope("1")

        remediation = [r for r in records if EXECUTION_SCOPE_AGENT_CONFIG_KEY in r]
        assert remediation, records
        assert "every turn" in remediation[0].lower()


class TestPersistedAgentConfigRejectsADecodedScope:
    """The override takes the task's raw ``agent_config``. An already-decoded
    scope is not a Mapping, so decoding it would yield "no snapshot" -- which
    on the abstention and no-resolver branches is indistinguishable from an
    authoritative unscoped answer. It has to be refused, not dropped."""

    def test_passing_a_scope_is_a_contract_error_not_a_silent_miss(self):
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope(
                "1", persisted_agent_config=ExecutionScope(sandbox_key_suffix="s")
            )

    def test_raw_agent_config_still_resolves(self):
        scope = ExecutionScope(sandbox_key_suffix="s")
        assert (
            resolve_execution_scope(
                "1",
                persisted_agent_config={
                    EXECUTION_SCOPE_AGENT_CONFIG_KEY: scope.to_dict()
                },
            )
            == scope
        )


class TestWrongTypedSnapshotIsCorruptNotAbsent:
    """A present ``execution_scope`` that is not a mapping cannot be decoded,
    exactly like a mapping with an invalid field. Reporting "no candidate" for
    one shape and raising for the other would make the same accidental
    namespace decision on every branch where the snapshot is the only
    authority."""

    @pytest.mark.parametrize("value", ["garbage-string", ["a"], 5, True])
    def test_present_but_wrong_typed_raises(self, value):
        with pytest.raises(ExecutionScopeResolverContractError):
            execution_scope_from_agent_config({EXECUTION_SCOPE_AGENT_CONFIG_KEY: value})

    def test_absent_key_is_still_absent(self):
        assert execution_scope_from_agent_config({"other": 1}) is None

    def test_explicit_null_is_still_absent(self):
        assert (
            execution_scope_from_agent_config({EXECUTION_SCOPE_AGENT_CONFIG_KEY: None})
            is None
        )

    def test_wrong_typed_snapshot_fails_the_abstention_branch(self):
        """The branch where "absent" would resolve to the abstention's
        fallback -- the widest value the narrowing check would ever admit."""
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback=ExecutionScope()),
            acknowledges_snapshot_candidate_contract=True,
        )
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope(
                "1",
                persisted_agent_config={
                    EXECUTION_SCOPE_AGENT_CONFIG_KEY: "garbage-string"
                },
            )


class TestDecodedScopeGuardRunsBeforeBranchDispatch:
    """The guard rejects a caller bug, so which resolver happens to be
    registered must not decide whether it is heard. The authoritative branch
    wraps its snapshot load in a tolerant except -- so a bad candidate cannot
    veto a real answer -- and a raise from inside that wrapper would be
    absorbed and misreported as a loader failure."""

    def test_raises_with_an_authoritative_resolver_registered(self):
        set_execution_scope_resolver(
            lambda task_id: ExecutionScope(sandbox_key_suffix="from-resolver"),
            acknowledges_snapshot_candidate_contract=True,
        )
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope(
                "1", persisted_agent_config=ExecutionScope(sandbox_key_suffix="s")
            )

    def test_raises_when_the_resolver_abstains(self):
        set_execution_scope_resolver(
            lambda task_id: DeferToSnapshot(fallback=ExecutionScope()),
            acknowledges_snapshot_candidate_contract=True,
        )
        with pytest.raises(ExecutionScopeResolverContractError):
            resolve_execution_scope(
                "1", persisted_agent_config=ExecutionScope(sandbox_key_suffix="s")
            )
