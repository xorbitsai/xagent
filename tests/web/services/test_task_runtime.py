from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.task_runtime import (
    MAX_TASK_RUNTIME_ENVIRONMENT_BYTES,
    MAX_TASK_RUNTIME_EXTENSIONS,
    MAX_TASK_RUNTIME_JSON_BYTES,
    MAX_TASK_RUNTIME_PUBLIC_METADATA_BYTES,
    MAX_TASK_RUNTIME_REQUEST_BYTES,
    MAX_TASK_RUNTIME_TOOLS,
    TaskRuntimeContext,
    TaskRuntimeContribution,
    normalize_input_modalities,
    normalize_task_runtime_contribution,
)
from xagent.web.services.task_runtime import (
    TaskRuntimeExtensionError,
    build_task_runtime,
    create_task_extensions,
    delete_task_extensions,
    get_task_runtime_public_metadata,
    register_task_extension,
    registered_task_extensions,
    shutdown_task_runtime_hook_executor,
    unregister_task_extension,
    validate_task_extension_requests,
)


def test_public_task_runtime_facade_exports_provider_contract() -> None:
    from xagent.task_runtime import TaskRuntimeContext as PublicTaskRuntimeContext
    from xagent.task_runtime import (
        TaskRuntimeContribution as PublicTaskRuntimeContribution,
    )
    from xagent.task_runtime import (
        register_task_extension as public_register_task_extension,
    )

    assert PublicTaskRuntimeContext is TaskRuntimeContext
    assert PublicTaskRuntimeContribution is TaskRuntimeContribution
    assert public_register_task_extension is register_task_extension


@pytest.fixture
def registered_names() -> Iterator[list[str]]:
    names: list[str] = []
    yield names
    for name in names:
        unregister_task_extension(name)


def _context(*, workspace: Any = None) -> TaskRuntimeContext:
    return TaskRuntimeContext(
        task_id=42,
        user_id=7,
        source="internal",
        session_factory=lambda: object(),
        workspace=workspace,
    )


def test_task_runtime_executor_shutdown_allows_lazy_recreation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xagent.web.services.task_runtime as task_runtime

    shutdown_task_runtime_hook_executor()
    monkeypatch.setattr(task_runtime, "get_task_runtime_hook_max_workers", lambda: 2)

    first = task_runtime._get_task_runtime_hook_executor()
    shutdown_task_runtime_hook_executor()
    second = task_runtime._get_task_runtime_hook_executor()
    shutdown_task_runtime_hook_executor()

    assert first is not second
    assert first._shutdown is True
    assert second._shutdown is True


class _Provider:
    def __init__(
        self,
        name: str,
        *,
        contribution: TaskRuntimeContribution | None = None,
        metadata: dict[str, Any] | None = None,
        fail_create: Exception | None = None,
        fail_delete: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.name = name
        self.contribution = contribution
        self.metadata = metadata
        self.fail_create = fail_create
        self.fail_delete = fail_delete
        self.events = events if events is not None else []
        self.created_configuration: dict[str, Any] | None = None

    async def on_task_created(
        self,
        context: TaskRuntimeContext,
        configuration: dict[str, Any],
    ) -> None:
        assert context.task_id == 42
        self.created_configuration = configuration
        self.events.append(f"create:{self.name}")
        if self.fail_create is not None:
            raise self.fail_create

    async def build_runtime(
        self,
        context: TaskRuntimeContext,
    ) -> TaskRuntimeContribution | None:
        self.events.append(f"build:{self.name}")
        return self.contribution

    def public_metadata(self, context: TaskRuntimeContext) -> dict[str, Any] | None:
        self.events.append(f"metadata:{self.name}")
        return self.metadata

    async def on_task_deleted(self, context: TaskRuntimeContext) -> None:
        self.events.append(f"delete:{self.name}")
        if self.fail_delete is not None:
            raise self.fail_delete


def _register(
    name: str,
    provider: Any,
    registered_names: list[str],
) -> None:
    register_task_extension(name, provider)
    registered_names.append(name)


def test_normalize_input_modalities_ignores_none() -> None:
    assert normalize_input_modalities((None, " IMAGE ", None, "image")) == ("image",)


@pytest.mark.parametrize(
    "value",
    [
        b"image",
        {"image": True},
        ("image", 7),
    ],
)
def test_normalize_input_modalities_rejects_malformed_values(value: Any) -> None:
    with pytest.raises(TypeError, match="must be"):
        normalize_input_modalities(value)


@pytest.mark.asyncio
async def test_provider_lifecycle_merges_detached_runtime_contributions(
    registered_names: list[str],
) -> None:
    tool_a = SimpleNamespace(name="tool_a")
    tool_b = SimpleNamespace(name="tool_b")
    events: list[str] = []
    first = _Provider(
        "first",
        contribution=TaskRuntimeContribution(
            tools=(tool_a,),
            environment="First environment",
            preferred_input_modalities=("IMAGE",),
        ),
        metadata={"target": "one"},
        events=events,
    )
    second = _Provider(
        "second",
        contribution=TaskRuntimeContribution(
            tools=(tool_b,),
            environment="Second environment",
            preferred_input_modalities=("image", "audio"),
        ),
        metadata={},
        events=events,
    )
    _register("first_runtime", first, registered_names)
    _register("second_runtime", second, registered_names)

    requests = validate_task_extension_requests({"first_runtime": {"target": "one"}})
    await create_task_extensions(_context(), requests)
    contribution = await build_task_runtime(
        _context(workspace=SimpleNamespace(id="workspace"))
    )
    metadata = await get_task_runtime_public_metadata(_context())
    await delete_task_extensions(
        _context(),
        bound_extensions=("first_runtime", "second_runtime"),
    )

    assert first.created_configuration == {"target": "one"}
    assert second.created_configuration is None
    assert contribution.tools == (tool_a, tool_b)
    assert contribution.environment == "First environment\n\nSecond environment"
    assert contribution.preferred_input_modalities == ("image", "audio")
    assert contribution.tool_origins == (
        ("tool_a", "first_runtime"),
        ("tool_b", "second_runtime"),
    )
    assert metadata.extensions == {"first_runtime": {"target": "one"}}
    assert metadata.status == "complete"
    assert events[-2:] == ["delete:second", "delete:first"]


@pytest.mark.asyncio
async def test_create_failure_cleans_up_completed_and_failing_providers(
    registered_names: list[str],
) -> None:
    events: list[str] = []
    first = _Provider("first", events=events)
    second = _Provider(
        "second",
        fail_create=ValueError("invalid binding"),
        events=events,
    )
    _register("first_runtime", first, registered_names)
    _register("second_runtime", second, registered_names)

    with pytest.raises(TaskRuntimeExtensionError) as exc_info:
        await create_task_extensions(
            _context(),
            {
                "first_runtime": {},
                "second_runtime": {},
            },
        )

    assert exc_info.value.extension == "second_runtime"
    assert exc_info.value.operation == "on_task_created"
    assert events == [
        "create:first",
        "create:second",
        "delete:second",
        "delete:first",
    ]


@pytest.mark.asyncio
async def test_create_failure_logs_cleanup_error_and_preserves_create_error(
    registered_names: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = _Provider("first", fail_delete=RuntimeError("cleanup unavailable"))
    second = _Provider("second", fail_create=ValueError("invalid binding"))
    _register("first_runtime", first, registered_names)
    _register("second_runtime", second, registered_names)

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(TaskRuntimeExtensionError) as exc_info,
    ):
        await create_task_extensions(
            _context(),
            {
                "first_runtime": {},
                "second_runtime": {},
            },
        )

    assert isinstance(exc_info.value.cause, ValueError)
    assert "Cleanup failed for task runtime extension 'first_runtime'" in caplog.text
    assert "cleanup unavailable" in caplog.text


@pytest.mark.asyncio
async def test_create_failure_cleanup_does_not_swallow_process_control_exception(
    registered_names: list[str],
) -> None:
    provider = _Provider(
        "interrupting",
        fail_create=ValueError("invalid binding"),
        fail_delete=SystemExit("stop"),
    )
    _register("interrupting_runtime", provider, registered_names)

    with pytest.raises(SystemExit, match="stop"):
        await create_task_extensions(
            _context(),
            {"interrupting_runtime": {}},
        )


@pytest.mark.asyncio
async def test_create_wraps_registry_change_during_validation(
    registered_names: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xagent.web.services.task_runtime as task_runtime

    provider = _Provider("racy")
    _register("racy_runtime", provider, registered_names)
    original_validate = task_runtime.validate_task_extension_requests

    def validate_and_unregister(value: Any) -> dict[str, dict[str, Any]]:
        result = original_validate(value)
        unregister_task_extension("racy_runtime")
        return result

    monkeypatch.setattr(
        task_runtime,
        "validate_task_extension_requests",
        validate_and_unregister,
    )

    with pytest.raises(TaskRuntimeExtensionError) as exc_info:
        await create_task_extensions(_context(), {"racy_runtime": {}})

    assert exc_info.value.extension == "registry"
    assert exc_info.value.operation == "validate_requests"


@pytest.mark.asyncio
async def test_build_dispatch_uses_registry_snapshot_across_await(
    registered_names: list[str],
) -> None:
    events: list[str] = []
    second = _Provider(
        "second",
        contribution=TaskRuntimeContribution(environment="second"),
        events=events,
    )

    class _UnregisteringProvider(_Provider):
        async def build_runtime(
            self,
            context: TaskRuntimeContext,
        ) -> TaskRuntimeContribution:
            events.append("build:first")
            unregister_task_extension("second_runtime")
            return TaskRuntimeContribution(environment="first")

    first = _UnregisteringProvider("first", events=events)
    _register("first_runtime", first, registered_names)
    _register("second_runtime", second, registered_names)

    contribution = await build_task_runtime(_context())

    assert contribution.environment == "first\n\nsecond"
    assert events == ["build:first", "build:second"]


@pytest.mark.asyncio
async def test_metadata_dispatch_uses_registry_snapshot_across_await(
    registered_names: list[str],
) -> None:
    events: list[str] = []
    second = _Provider("second", metadata={"second": True}, events=events)

    class _UnregisteringProvider(_Provider):
        def public_metadata(
            self,
            context: TaskRuntimeContext,
        ) -> dict[str, Any]:
            events.append("metadata:first")
            unregister_task_extension("second_metadata")
            return {"first": True}

    first = _UnregisteringProvider("first", events=events)
    _register("first_metadata", first, registered_names)
    _register("second_metadata", second, registered_names)

    metadata = await get_task_runtime_public_metadata(_context())

    assert metadata.extensions == {
        "first_metadata": {"first": True},
        "second_metadata": {"second": True},
    }
    assert events == ["metadata:first", "metadata:second"]


def test_registry_rejects_unknown_invalid_and_incomplete_providers(
    registered_names: list[str],
) -> None:
    with pytest.raises(ValueError, match="not registered"):
        validate_task_extension_requests({"missing_runtime": {}})
    with pytest.raises(ValueError, match="lowercase"):
        register_task_extension("Computer", _Provider("bad"))
    with pytest.raises(TypeError, match="missing callable"):
        register_task_extension("incomplete", object())

    provider = _Provider("valid")
    _register("valid_runtime", provider, registered_names)
    assert "valid_runtime" in registered_task_extensions()
    with pytest.raises(ValueError, match="already registered"):
        register_task_extension("valid_runtime", provider)


def test_runtime_extension_requests_are_size_bounded(
    registered_names: list[str],
) -> None:
    provider = _Provider("bounded")
    _register("bounded_runtime", provider, registered_names)

    with pytest.raises(TypeError, match="JSON-compatible"):
        validate_task_extension_requests({"bounded_runtime": {"value": float("nan")}})
    with pytest.raises(ValueError, match="byte limit"):
        validate_task_extension_requests(
            {"bounded_runtime": {"value": "x" * MAX_TASK_RUNTIME_JSON_BYTES}}
        )
    with pytest.raises(ValueError, match="at most"):
        validate_task_extension_requests(
            {f"runtime_{index}": {} for index in range(MAX_TASK_RUNTIME_EXTENSIONS + 1)}
        )


@pytest.mark.asyncio
async def test_public_metadata_must_be_json_compatible(
    registered_names: list[str],
) -> None:
    provider = _Provider("invalid", metadata={"secret": object()})
    _register("invalid_metadata", provider, registered_names)

    with pytest.raises(TaskRuntimeExtensionError) as exc_info:
        await get_task_runtime_public_metadata(_context())

    assert exc_info.value.extension == "invalid_metadata"
    assert exc_info.value.operation == "public_metadata"


@pytest.mark.asyncio
async def test_provider_hook_timeout_is_attributed_and_bounded(
    registered_names: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import xagent.web.services.task_runtime as task_runtime

    class _HangingProvider(_Provider):
        async def build_runtime(
            self,
            context: TaskRuntimeContext,
        ) -> TaskRuntimeContribution:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    _register("hanging_runtime", _HangingProvider("hanging"), registered_names)
    monkeypatch.setitem(
        task_runtime._TASK_RUNTIME_HOOK_TIMEOUT_SECONDS,
        "build_runtime",
        0.01,
    )

    with caplog.at_level(logging.ERROR):
        contribution = await build_task_runtime(_context())

    assert contribution == TaskRuntimeContribution()
    assert "hanging_runtime" in caplog.text
    assert "0.01-second timeout" in caplog.text


@pytest.mark.asyncio
async def test_blocking_provider_timeout_does_not_consume_default_executor(
    registered_names: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xagent.web.services.task_runtime as task_runtime

    release_hook = threading.Event()

    class _BlockingProvider(_Provider):
        def build_runtime(
            self,
            context: TaskRuntimeContext,
        ) -> TaskRuntimeContribution:
            release_hook.wait()
            return TaskRuntimeContribution(environment="released")

    _register(
        "blocking_runtime",
        _BlockingProvider("blocking"),
        registered_names,
    )
    monkeypatch.setitem(
        task_runtime._TASK_RUNTIME_HOOK_TIMEOUT_SECONDS,
        "build_runtime",
        0.01,
    )

    try:
        contribution = await build_task_runtime(_context())
        default_executor_result = await asyncio.wait_for(
            asyncio.to_thread(lambda: "available"),
            timeout=0.5,
        )
    finally:
        release_hook.set()

    assert contribution == TaskRuntimeContribution()
    assert default_executor_result == "available"


@pytest.mark.asyncio
async def test_provider_execution_timeout_excludes_executor_queue_wait(
    registered_names: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xagent.web.services.task_runtime as task_runtime

    release_blocker = threading.Event()

    class _BlockingProvider(_Provider):
        def build_runtime(
            self,
            context: TaskRuntimeContext,
        ) -> TaskRuntimeContribution:
            release_blocker.wait()
            return TaskRuntimeContribution(environment="too late")

    shutdown_task_runtime_hook_executor()
    monkeypatch.setattr(task_runtime, "get_task_runtime_hook_max_workers", lambda: 1)
    monkeypatch.setattr(
        task_runtime,
        "get_task_runtime_hook_queue_timeout_seconds",
        lambda: 1,
    )
    monkeypatch.setitem(
        task_runtime._TASK_RUNTIME_HOOK_TIMEOUT_SECONDS,
        "build_runtime",
        0.05,
    )
    _register(
        "blocking_runtime",
        _BlockingProvider("blocking"),
        registered_names,
    )
    _register(
        "healthy_runtime",
        _Provider(
            "healthy",
            contribution=TaskRuntimeContribution(environment="healthy"),
        ),
        registered_names,
    )
    asyncio.get_running_loop().call_later(0.15, release_blocker.set)

    try:
        contribution = await build_task_runtime(_context())
    finally:
        release_blocker.set()
        shutdown_task_runtime_hook_executor()

    assert contribution.environment == "healthy"


@pytest.mark.asyncio
async def test_build_runtime_keeps_successful_provider_when_another_fails(
    registered_names: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    good_tool = SimpleNamespace(name="good_tool")

    class _FailingProvider(_Provider):
        async def build_runtime(
            self,
            context: TaskRuntimeContext,
        ) -> TaskRuntimeContribution:
            raise RuntimeError("provider unavailable")

    _register(
        "good_runtime",
        _Provider(
            "good",
            contribution=TaskRuntimeContribution(
                tools=(good_tool,),
                environment="Good environment",
            ),
        ),
        registered_names,
    )
    _register(
        "failing_runtime",
        _FailingProvider("failing"),
        registered_names,
    )

    with caplog.at_level(logging.ERROR):
        contribution = await build_task_runtime(_context())

    assert contribution.tools == (good_tool,)
    assert contribution.environment == "Good environment"
    assert "failing_runtime" in caplog.text


@pytest.mark.asyncio
async def test_runtime_contribution_environment_and_tools_are_bounded(
    registered_names: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    oversized_environment = _Provider(
        "environment",
        contribution=TaskRuntimeContribution(
            environment="x" * (MAX_TASK_RUNTIME_ENVIRONMENT_BYTES + 1)
        ),
    )
    _register("environment_runtime", oversized_environment, registered_names)

    with caplog.at_level(logging.ERROR):
        assert await build_task_runtime(_context()) == TaskRuntimeContribution()
    assert "environment_runtime" in caplog.text
    unregister_task_extension("environment_runtime")
    registered_names.remove("environment_runtime")

    oversized_tools = _Provider(
        "tools",
        contribution=TaskRuntimeContribution(
            tools=tuple(
                SimpleNamespace(name=f"tool_{index}")
                for index in range(MAX_TASK_RUNTIME_TOOLS + 1)
            )
        ),
    )
    _register("tools_runtime", oversized_tools, registered_names)

    with caplog.at_level(logging.ERROR):
        assert await build_task_runtime(_context()) == TaskRuntimeContribution()
    assert "tools_runtime" in caplog.text


@pytest.mark.asyncio
async def test_aggregate_tool_limit_logs_dropped_provider_and_reason(
    registered_names: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _register(
        "large_runtime",
        _Provider(
            "large",
            contribution=TaskRuntimeContribution(
                tools=tuple(
                    SimpleNamespace(name=f"large_{index}") for index in range(60)
                )
            ),
        ),
        registered_names,
    )
    _register(
        "small_runtime",
        _Provider(
            "small",
            contribution=TaskRuntimeContribution(
                tools=tuple(
                    SimpleNamespace(name=f"small_{index}") for index in range(5)
                )
            ),
        ),
        registered_names,
    )

    with caplog.at_level(logging.ERROR):
        contribution = await build_task_runtime(_context())

    assert len(contribution.tools) == 60
    assert "small_runtime" in caplog.text
    assert f"{MAX_TASK_RUNTIME_TOOLS}-tool limit" in caplog.text


@pytest.mark.asyncio
async def test_public_metadata_aggregate_is_bounded_with_top_level_status(
    registered_names: list[str],
) -> None:
    for index in range(5):
        _register(
            f"metadata_{index}",
            _Provider(
                f"metadata-{index}",
                metadata={"payload": "x" * (60 * 1024)},
            ),
            registered_names,
        )

    metadata = await get_task_runtime_public_metadata(_context())

    assert metadata.status == "truncated"
    assert metadata.omitted_extensions
    assert "_runtime" not in metadata.extensions
    assert (
        len(str(metadata.extensions).encode("utf-8"))
        < MAX_TASK_RUNTIME_PUBLIC_METADATA_BYTES
    )


@pytest.mark.asyncio
async def test_delete_dispatch_treats_provider_cancel_as_one_failure(
    registered_names: list[str],
) -> None:
    events: list[str] = []
    _register(
        "healthy_runtime",
        _Provider("healthy", events=events),
        registered_names,
    )
    _register(
        "cancelled_runtime",
        _Provider(
            "cancelled",
            fail_delete=asyncio.CancelledError("provider cancelled itself"),
            events=events,
        ),
        registered_names,
    )

    with pytest.raises(TaskRuntimeExtensionError) as exc_info:
        await delete_task_extensions(
            _context(),
            bound_extensions=("healthy_runtime", "cancelled_runtime"),
        )

    assert events == ["delete:cancelled", "delete:healthy"]
    assert exc_info.value.extension == "cancelled_runtime"
    assert isinstance(exc_info.value.cause, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_build_dispatch_propagates_surrounding_task_cancellation(
    registered_names: list[str],
) -> None:
    started = asyncio.Event()

    class _WaitingProvider(_Provider):
        async def build_runtime(
            self,
            context: TaskRuntimeContext,
        ) -> TaskRuntimeContribution:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    _register(
        "waiting_runtime",
        _WaitingProvider("waiting"),
        registered_names,
    )

    build_task = asyncio.create_task(build_task_runtime(_context()))
    await started.wait()
    build_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await build_task


@pytest.mark.asyncio
async def test_delete_dispatch_aggregates_multiple_provider_failures(
    registered_names: list[str],
) -> None:
    events: list[str] = []
    _register(
        "first_failure",
        _Provider(
            "first",
            fail_delete=RuntimeError("first failed"),
            events=events,
        ),
        registered_names,
    )
    _register(
        "second_failure",
        _Provider(
            "second",
            fail_delete=ValueError("second failed"),
            events=events,
        ),
        registered_names,
    )

    with pytest.raises(TaskRuntimeExtensionError) as exc_info:
        await delete_task_extensions(
            _context(),
            bound_extensions=("first_failure", "second_failure"),
        )

    assert events == ["delete:second", "delete:first"]
    assert exc_info.value.extension == "second_failure"
    assert "second_failure: second failed" in str(exc_info.value.cause)
    assert "first_failure: first failed" in str(exc_info.value.cause)


def test_normalize_drops_provider_supplied_registry_bookkeeping() -> None:
    """A provider cannot pre-populate the registry's own attribution fields.

    ``tool_origins``, ``provider_contributions`` and ``source_contribution``
    are registry-internal. ``frozen=True`` only blocks post-init mutation, so
    normalization -- not the dataclass -- has to be the gate that strips
    whatever a provider passed to the constructor.
    """

    evil_tool = SimpleNamespace(name="evil_tool")
    fabricated = TaskRuntimeContribution(
        tools=(evil_tool,),
        environment="evil",
        tool_origins=(("evil_tool", "honest_runtime"),),
        provider_contributions=(
            (
                "honest_runtime",
                TaskRuntimeContribution(tools=(evil_tool,)),
            ),
        ),
        source_contribution=TaskRuntimeContribution(
            tools=(SimpleNamespace(name="smuggled_tool"),),
        ),
    )

    normalized = normalize_task_runtime_contribution(fabricated)

    assert normalized.tools == (evil_tool,)
    assert normalized.environment == "evil"
    assert normalized.tool_origins == ()
    assert normalized.provider_contributions == ()
    assert normalized.source_contribution is None


@pytest.mark.asyncio
async def test_provider_cannot_fabricate_attribution_to_another_provider(
    registered_names: list[str],
) -> None:
    honest_tool = SimpleNamespace(name="honest_tool")
    evil_tool = SimpleNamespace(name="evil_tool")
    smuggled_tool = SimpleNamespace(name="smuggled_tool")

    _register(
        "evil_runtime",
        _Provider(
            "evil",
            contribution=TaskRuntimeContribution(
                tools=(evil_tool,),
                environment="Evil environment",
                # Fabricated: claims its tool belongs to the honest provider,
                # invents a provider record for a peer, and hides an extra tool
                # behind the registry-internal back-reference.
                tool_origins=(("evil_tool", "honest_runtime"),),
                provider_contributions=(
                    (
                        "honest_runtime",
                        TaskRuntimeContribution(tools=(evil_tool,)),
                    ),
                ),
                source_contribution=TaskRuntimeContribution(
                    tools=(smuggled_tool,),
                ),
            ),
        ),
        registered_names,
    )
    _register(
        "honest_runtime",
        _Provider(
            "honest",
            contribution=TaskRuntimeContribution(
                tools=(honest_tool,),
                environment="Honest environment",
            ),
        ),
        registered_names,
    )

    merged = await build_task_runtime(_context())

    assert dict(merged.tool_origins) == {
        "evil_tool": "evil_runtime",
        "honest_tool": "honest_runtime",
    }
    assert [name for name, _ in merged.provider_contributions] == [
        "evil_runtime",
        "honest_runtime",
    ]
    per_provider = dict(merged.provider_contributions)
    assert per_provider["evil_runtime"].tool_origins == ()
    assert per_provider["evil_runtime"].provider_contributions == ()
    assert per_provider["evil_runtime"].source_contribution is None
    assert per_provider["honest_runtime"].tools == (honest_tool,)
    # The smuggled tool never reaches the merged contribution, and the merged
    # view is not silently treated as a policy-narrowed view of one.
    assert merged.tools == (evil_tool, honest_tool)
    assert merged.source_contribution is None


def test_runtime_extension_requests_are_aggregate_size_bounded(
    registered_names: list[str],
) -> None:
    """Per-extension caps alone leave the aggregate payload unbounded.

    ``MAX_TASK_RUNTIME_EXTENSIONS`` entries each just under the per-extension
    ``MAX_TASK_RUNTIME_JSON_BYTES`` cap add up to roughly 1 MiB. The read path
    caps the aggregate too; the create path has to as well.
    """

    # Comfortably under the per-extension cap, so only the aggregate rule can
    # reject this payload.
    per_extension_payload = "x" * (MAX_TASK_RUNTIME_JSON_BYTES // 2)
    requests: dict[str, dict[str, Any]] = {}
    for index in range(MAX_TASK_RUNTIME_EXTENSIONS):
        name = f"bulk_runtime_{index}"
        _register(name, _Provider(name), registered_names)
        requests[name] = {"value": per_extension_payload}

    assert len(str(requests).encode("utf-8")) > MAX_TASK_RUNTIME_REQUEST_BYTES, (
        "fixture must exceed the aggregate cap"
    )

    with pytest.raises(ValueError, match="byte limit"):
        validate_task_extension_requests(requests)

    # A payload of the same shape that fits stays accepted.
    accepted = {
        name: {"value": "x" * 16}
        for name in list(requests)[:MAX_TASK_RUNTIME_EXTENSIONS]
    }
    assert validate_task_extension_requests(accepted) == accepted


@pytest.mark.asyncio
async def test_saturated_hook_pool_fails_fast_on_queue_wait(
    registered_names: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every dedicated hook worker hung: the queue-wait timeout must fire.

    The per-operation execution timeout cannot help here -- the hook never
    starts running -- so the only thing bounding the request is the queue-wait
    deadline in ``_invoke_provider_hook``.
    """

    import xagent.web.services.task_runtime as task_runtime

    max_workers = 2
    release_pool = threading.Event()
    occupied = threading.Barrier(max_workers + 1)

    shutdown_task_runtime_hook_executor()
    monkeypatch.setattr(
        task_runtime,
        "get_task_runtime_hook_max_workers",
        lambda: max_workers,
    )
    monkeypatch.setattr(
        task_runtime,
        "get_task_runtime_hook_queue_timeout_seconds",
        lambda: 0.05,
    )
    # Long enough that a fired execution timeout would be unmistakable: only
    # the queue-wait deadline can end this call quickly.
    monkeypatch.setitem(
        task_runtime._TASK_RUNTIME_HOOK_TIMEOUT_SECONDS,
        "build_runtime",
        30.0,
    )

    executor = task_runtime._get_task_runtime_hook_executor()

    def _hang() -> None:
        occupied.wait(timeout=5)
        release_pool.wait(timeout=5)

    hung = [executor.submit(_hang) for _ in range(max_workers)]
    # Every worker thread is now inside ``_hang``; nothing else can start.
    occupied.wait(timeout=5)

    provider = _Provider(
        "starved",
        contribution=TaskRuntimeContribution(environment="never runs"),
    )
    _register("starved_runtime", provider, registered_names)

    try:
        with caplog.at_level(logging.ERROR, logger=task_runtime.__name__):
            contribution = await build_task_runtime(_context())
    finally:
        release_pool.set()
        for future in hung:
            future.result(timeout=5)
        shutdown_task_runtime_hook_executor()

    # The hook never got a worker, so the provider is dropped, not applied.
    assert contribution == TaskRuntimeContribution()
    assert "build:starved" not in provider.events
    assert "Dropping task runtime contribution" in caplog.text
    assert "queue wait exceeded" in caplog.text
