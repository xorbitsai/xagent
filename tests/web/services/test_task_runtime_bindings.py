"""Per-task provider binding filter for ``on_task_deleted`` dispatch."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest

from xagent.core.execution_scope import (
    EXECUTION_SCOPE_AGENT_CONFIG_KEY,
    execution_scope_from_agent_config,
)
from xagent.core.task_runtime import TaskRuntimeContext, TaskRuntimeContribution
from xagent.web.services.task_runtime import (
    CLIENT_RESERVED_AGENT_CONFIG_KEYS,
    SELECTED_FILE_IDS_AGENT_CONFIG_KEY,
    TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY,
    TaskRuntimeExtensionError,
    agent_config_with_task_extension_bindings,
    delete_task_extensions,
    register_task_extension,
    sanitize_client_agent_config,
    task_extension_bindings_from_agent_config,
    unregister_task_extension,
)


@pytest.fixture
def registered_names() -> Iterator[list[str]]:
    names: list[str] = []
    yield names
    for name in names:
        unregister_task_extension(name)


def _context() -> TaskRuntimeContext:
    return TaskRuntimeContext(
        task_id=42,
        user_id=7,
        source="internal",
        session_factory=lambda: object(),
    )


class _Provider:
    def __init__(self, name: str, *, fail_delete: BaseException | None = None) -> None:
        self.name = name
        self.fail_delete = fail_delete
        self.deleted: list[int] = []

    async def on_task_created(self, context: Any, configuration: Any) -> None:
        return None

    async def build_runtime(self, context: Any) -> TaskRuntimeContribution:
        return TaskRuntimeContribution()

    async def public_metadata(self, context: Any) -> dict[str, Any]:
        return {}

    async def on_task_deleted(self, context: TaskRuntimeContext) -> None:
        self.deleted.append(context.task_id)
        if self.fail_delete is not None:
            raise self.fail_delete


def _register(name: str, provider: Any, registered_names: list[str]) -> _Provider:
    register_task_extension(name, provider)
    registered_names.append(name)
    return provider


# ----- agent_config encoding -----


def test_bindings_roundtrip_through_agent_config() -> None:
    encoded = agent_config_with_task_extension_bindings(
        {"existing": 1}, ["b_ext", "a_ext", "b_ext"]
    )

    assert encoded["existing"] == 1
    # Deduplicated and sorted so the persisted JSON is stable.
    assert encoded[TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY] == ["a_ext", "b_ext"]
    assert task_extension_bindings_from_agent_config(encoded) == ("a_ext", "b_ext")


@pytest.mark.parametrize(
    "agent_config",
    [None, {}, {"other": 1}, "not-a-mapping", {"runtime_extension_bindings": "nope"}],
)
def test_bindings_decode_tolerates_missing_and_malformed_records(
    agent_config: Any,
) -> None:
    assert task_extension_bindings_from_agent_config(agent_config) == ()


def test_bindings_encode_drops_the_key_when_empty() -> None:
    encoded = agent_config_with_task_extension_bindings(
        {TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY: ["gone"], "keep": 2}, []
    )

    assert encoded == {"keep": 2}


# ----- client input sanitizing -----


def test_sanitize_drops_reserved_keys_and_keeps_client_keys() -> None:
    """Every reserved key is stripped from a client dict, in one place, so a
    future addition to the reserved set covers all task-create boundaries."""
    sanitized = sanitize_client_agent_config(
        {
            "keep": 1,
            **{key: ["forged"] for key in CLIENT_RESERVED_AGENT_CONFIG_KEYS},
        }
    )

    assert sanitized == {"keep": 1}
    assert TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY in CLIENT_RESERVED_AGENT_CONFIG_KEYS


@pytest.mark.parametrize("agent_config", [None, {}, "not-a-mapping", 7])
def test_sanitize_returns_a_fresh_dict_for_non_mappings(agent_config: Any) -> None:
    assert sanitize_client_agent_config(agent_config) == {}


def test_sanitize_does_not_mutate_the_caller_dict() -> None:
    original = {TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY: ["forged"], "keep": 1}

    sanitized = sanitize_client_agent_config(original)

    assert sanitized == {"keep": 1}
    assert original == {TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY: ["forged"], "keep": 1}


# ----- dispatch filtering -----


@pytest.mark.asyncio
async def test_delete_skips_providers_the_task_never_bound_to(
    registered_names: list[str],
) -> None:
    """A broken provider that owns nothing must not block the deletion."""

    broken = _register(
        "broken_ext",
        _Provider("broken_ext", fail_delete=RuntimeError("provider is down")),
        registered_names,
    )
    owning = _register("owning_ext", _Provider("owning_ext"), registered_names)

    unreleased = await delete_task_extensions(
        _context(), bound_extensions=("owning_ext",)
    )

    assert unreleased == ()
    assert owning.deleted == [42]
    assert broken.deleted == []


@pytest.mark.asyncio
async def test_delete_dispatches_to_nobody_when_the_task_has_no_bindings(
    registered_names: list[str],
) -> None:
    broken = _register(
        "broken_ext",
        _Provider("broken_ext", fail_delete=RuntimeError("provider is down")),
        registered_names,
    )

    assert await delete_task_extensions(_context(), bound_extensions=()) == ()
    assert broken.deleted == []


@pytest.mark.asyncio
async def test_delete_remains_fail_closed_for_a_bound_provider(
    registered_names: list[str],
) -> None:
    """M3 regression guard: an owning provider's failure preserves the task."""

    _register(
        "owner_ext",
        _Provider("owner_ext", fail_delete=RuntimeError("release failed")),
        registered_names,
    )

    with pytest.raises(TaskRuntimeExtensionError) as exc_info:
        await delete_task_extensions(_context(), bound_extensions=("owner_ext",))

    assert exc_info.value.extension == "owner_ext"
    assert exc_info.value.operation == "on_task_deleted"
    assert exc_info.value.unreleased_extensions == ("owner_ext",)


@pytest.mark.asyncio
async def test_force_delete_reports_failures_without_raising(
    registered_names: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    failing = _register(
        "owner_ext",
        _Provider("owner_ext", fail_delete=RuntimeError("release failed")),
        registered_names,
    )
    healthy = _register("other_ext", _Provider("other_ext"), registered_names)

    with caplog.at_level(logging.ERROR, logger="xagent.web.services.task_runtime"):
        unreleased = await delete_task_extensions(
            _context(),
            bound_extensions=("owner_ext", "other_ext"),
            force=True,
        )

    assert unreleased == ("owner_ext",)
    # Every bound provider is still attempted, not just the ones before failure.
    assert failing.deleted == [42]
    assert healthy.deleted == [42]
    assert any("force" in record.getMessage().lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_delete_reports_bindings_whose_provider_is_no_longer_registered(
    registered_names: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A binding with no live provider is logged, not turned into a deadlock."""

    healthy = _register("live_ext", _Provider("live_ext"), registered_names)

    with caplog.at_level(logging.ERROR, logger="xagent.web.services.task_runtime"):
        unreleased = await delete_task_extensions(
            _context(), bound_extensions=("live_ext", "ghost_ext")
        )

    assert unreleased == ("ghost_ext",)
    assert healthy.deleted == [42]
    assert any("ghost_ext" in record.getMessage() for record in caplog.records)


def test_execution_scope_is_reserved_so_a_request_cannot_choose_a_namespace() -> None:
    """A forged scope in a client dict is stripped, leaving no candidate for
    resolution to read. Why the key is refused rather than judged afterwards is
    on CLIENT_RESERVED_AGENT_CONFIG_KEYS.
    """
    assert EXECUTION_SCOPE_AGENT_CONFIG_KEY in CLIENT_RESERVED_AGENT_CONFIG_KEYS

    forged = {
        "sandbox_key_suffix": "victim",
        "workspace_segments": ["victim"],
        "memory_dimensions": {"tenant": "victim"},
    }
    sanitized = sanitize_client_agent_config(
        {EXECUTION_SCOPE_AGENT_CONFIG_KEY: forged, "keep": 1}
    )

    assert sanitized == {"keep": 1}
    # The column value a task is created with therefore carries no scope, so the
    # registered loader reports "no candidate" and the resolver answers alone.
    assert execution_scope_from_agent_config(sanitized) is None


def test_the_reserved_key_set_has_exactly_the_audited_members() -> None:
    """Pin the reserved set to literal strings, not the constants that name
    them. ``test_sanitize_drops_reserved_keys_and_keeps_client_keys`` builds
    its forged input by iterating ``CLIENT_RESERVED_AGENT_CONFIG_KEYS``, so it
    cannot detect a wrongly-added member -- whatever the set holds, that
    test's forged payload holds too. This test covers that direction instead.
    Literals are deliberate for a second reason: comparing against them also
    fails if either constant's *value* changes, which would silently move
    which requests get stripped at every task-create boundary.
    """
    assert CLIENT_RESERVED_AGENT_CONFIG_KEYS == frozenset(
        {
            "runtime_extension_bindings",
            "execution_scope",
            "selected_file_ids",
            # Public-channel identity/quota markers (#1108).
            "auth_mode",
            "guest_id",
            "widget_agent_id",
            "widget_workforce_id",
            "widget_client_ip",
            "share_agent_id",
            "share_workforce_id",
            "share_token",
        }
    )


def test_agent_builder_preview_keys_are_not_reserved() -> None:
    """``websocket.py``'s ``handle_build_preview_execution`` assembles a
    config from the WS message (``preview_agent_id`` included) and passes it
    through ``create_task`` -- and therefore through the sanitizer, layered
    below it rather than above. ``chat.py``'s re-layer of ``is_preview`` only
    fires when ``TaskCreateRequest.is_preview`` is ``True``, which this WS
    path never sets, so reserving either key here would strip it from that
    config with nothing to restore it, breaking agent-builder preview.
    """
    assert {"is_preview", "preview_agent_id"}.isdisjoint(
        CLIENT_RESERVED_AGENT_CONFIG_KEYS
    )


def test_selected_file_ids_is_reserved_so_a_request_cannot_name_files() -> None:
    """The bound file list is server-validated: the chat boundary substitutes a
    list filtered by owner and unbound-task before persisting it, and both
    readers re-check ownership in their own query. Reserving the key moves that
    enforcement to the one place every boundary shares, so the widget and share
    paths -- which never set the key themselves -- stop persisting whatever a
    request body carried.
    """
    assert SELECTED_FILE_IDS_AGENT_CONFIG_KEY in CLIENT_RESERVED_AGENT_CONFIG_KEYS

    sanitized = sanitize_client_agent_config(
        {SELECTED_FILE_IDS_AGENT_CONFIG_KEY: ["forged-file-id"], "keep": 1}
    )

    assert sanitized == {"keep": 1}
