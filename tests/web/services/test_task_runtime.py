from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.task_runtime import (
    TaskRuntimeContext,
    TaskRuntimeContribution,
    normalize_input_modalities,
)
from xagent.web.services.task_runtime import (
    TaskRuntimeExtensionError,
    build_task_runtime,
    create_task_extensions,
    delete_task_extensions,
    get_task_runtime_public_metadata,
    register_task_extension,
    registered_task_extensions,
    unregister_task_extension,
    validate_task_extension_requests,
)


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


class _Provider:
    def __init__(
        self,
        name: str,
        *,
        contribution: TaskRuntimeContribution | None = None,
        metadata: dict[str, Any] | None = None,
        fail_create: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.name = name
        self.contribution = contribution
        self.metadata = metadata
        self.fail_create = fail_create
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


def _register(
    name: str,
    provider: Any,
    registered_names: list[str],
) -> None:
    register_task_extension(name, provider)
    registered_names.append(name)


def test_normalize_input_modalities_ignores_none() -> None:
    assert normalize_input_modalities((None, " IMAGE ", None, "image")) == ("image",)


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
    await delete_task_extensions(_context())

    assert first.created_configuration == {"target": "one"}
    assert second.created_configuration is None
    assert contribution.tools == (tool_a, tool_b)
    assert contribution.environment == "First environment\n\nSecond environment"
    assert contribution.preferred_input_modalities == ("image", "audio")
    assert metadata == {"first_runtime": {"target": "one"}}
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
