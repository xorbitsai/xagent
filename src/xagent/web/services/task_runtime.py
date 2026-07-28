"""Process-wide registry for task-scoped runtime extensions.

Closed-source distributions can register providers at application startup.
The open-source task lifecycle then dispatches provider hooks without knowing
which runtime kinds, binding tables, or transports those providers implement.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Mapping
from typing import Any

from ...core.task_runtime import (
    EMPTY_TASK_RUNTIME_CONTRIBUTION,
    TaskRuntimeContext,
    TaskRuntimeContribution,
    TaskRuntimeExtensionProvider,
    merge_task_runtime_contributions,
    normalize_task_runtime_contribution,
)

_EXTENSION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PROVIDER_METHODS = (
    "on_task_created",
    "build_runtime",
    "public_metadata",
    "on_task_deleted",
)
_task_runtime_extensions: dict[str, TaskRuntimeExtensionProvider] = {}


class TaskRuntimeExtensionError(RuntimeError):
    """One registered provider failed a lifecycle operation."""

    def __init__(self, extension: str, operation: str, cause: BaseException):
        super().__init__(
            f"Task runtime extension '{extension}' failed during {operation}: {cause}"
        )
        self.extension = extension
        self.operation = operation
        self.cause = cause


def register_task_extension(
    name: str,
    provider: TaskRuntimeExtensionProvider,
    *,
    replace: bool = False,
) -> None:
    """Register one process-wide task runtime provider."""

    normalized = _normalize_extension_name(name)
    missing = [
        method
        for method in _PROVIDER_METHODS
        if not callable(getattr(provider, method, None))
    ]
    if missing:
        raise TypeError(
            "Task runtime extension provider is missing callable method(s): "
            + ", ".join(missing)
        )
    if normalized in _task_runtime_extensions and not replace:
        raise ValueError(f"Task runtime extension '{normalized}' is already registered")
    _task_runtime_extensions[normalized] = provider


def unregister_task_extension(name: str) -> TaskRuntimeExtensionProvider | None:
    """Remove and return one provider, if registered."""

    return _task_runtime_extensions.pop(_normalize_extension_name(name), None)


def registered_task_extensions() -> tuple[str, ...]:
    """Return registered names in deterministic dispatch order."""

    return tuple(_task_runtime_extensions)


def validate_task_extension_requests(value: Any) -> dict[str, dict[str, Any]]:
    """Validate create-time provider configurations without invoking hooks."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("runtime_extensions must be an object")

    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_configuration in value.items():
        name = _normalize_extension_name(str(raw_name))
        if name not in _task_runtime_extensions:
            raise ValueError(f"Task runtime extension '{name}' is not registered")
        if raw_configuration is None:
            configuration: dict[str, Any] = {}
        elif isinstance(raw_configuration, Mapping):
            configuration = dict(raw_configuration)
        else:
            raise TypeError(
                f"Task runtime extension '{name}' configuration must be an object"
            )
        _ensure_json_compatible(configuration, label=f"'{name}' configuration")
        normalized[name] = configuration
    return normalized


async def create_task_extensions(
    context: TaskRuntimeContext,
    requests: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Validate and persist requested task bindings through registered providers.

    If one provider fails, cleanup is dispatched to that provider and every
    provider that already completed, in reverse order. The caller remains
    responsible for compensating the newly committed core ``Task`` row.
    """

    normalized = validate_task_extension_requests(requests)
    completed: list[str] = []
    for name, configuration in normalized.items():
        provider = _task_runtime_extensions[name]
        try:
            await _maybe_await(provider.on_task_created(context, configuration))
        except Exception as exc:
            await _cleanup_after_create_failure(
                context,
                names=(*completed, name),
            )
            raise TaskRuntimeExtensionError(name, "on_task_created", exc) from exc
        completed.append(name)


async def build_task_runtime(
    context: TaskRuntimeContext,
) -> TaskRuntimeContribution:
    """Build and merge every registered provider's contribution for one task."""

    contributions: dict[str, TaskRuntimeContribution] = {}
    for name, provider in _task_runtime_extensions.items():
        try:
            contribution = await _maybe_await(provider.build_runtime(context))
            contributions[name] = normalize_task_runtime_contribution(contribution)
        except Exception as exc:
            raise TaskRuntimeExtensionError(name, "build_runtime", exc) from exc
    if not contributions:
        return EMPTY_TASK_RUNTIME_CONTRIBUTION
    return merge_task_runtime_contributions(contributions)


async def get_task_runtime_public_metadata(
    context: TaskRuntimeContext,
) -> dict[str, dict[str, Any]]:
    """Return provider-selected, JSON-safe metadata suitable for clients."""

    result: dict[str, dict[str, Any]] = {}
    for name, provider in _task_runtime_extensions.items():
        try:
            metadata = await _maybe_await(provider.public_metadata(context))
            if metadata is None:
                continue
            if not isinstance(metadata, Mapping):
                raise TypeError("public_metadata must return an object or None")
            detached = dict(metadata)
            _ensure_json_compatible(detached, label=f"'{name}' public metadata")
            if detached:
                result[name] = detached
        except Exception as exc:
            raise TaskRuntimeExtensionError(name, "public_metadata", exc) from exc
    return result


async def delete_task_extensions(context: TaskRuntimeContext) -> None:
    """Release provider-owned state for a deleted task.

    All providers are attempted even when one fails. A combined extension error
    is raised afterwards so callers can log cleanup failures without hiding a
    successful core task deletion.
    """

    failures: list[tuple[str, BaseException]] = []
    for name, provider in reversed(tuple(_task_runtime_extensions.items())):
        try:
            await _maybe_await(provider.on_task_deleted(context))
        except Exception as exc:
            failures.append((name, exc))
    if failures:
        failure_name, first_cause = failures[0]
        reported_cause: BaseException = first_cause
        if len(failures) > 1:
            reported_cause = RuntimeError(
                "; ".join(f"{failed_name}: {error}" for failed_name, error in failures)
            )
        raise TaskRuntimeExtensionError(
            failure_name,
            "on_task_deleted",
            reported_cause,
        ) from first_cause


async def _cleanup_after_create_failure(
    context: TaskRuntimeContext,
    *,
    names: tuple[str, ...],
) -> None:
    for name in reversed(names):
        provider = _task_runtime_extensions.get(name)
        if provider is None:
            continue
        try:
            await _maybe_await(provider.on_task_deleted(context))
        except Exception:
            # The original create failure remains primary. Providers should make
            # deletion idempotent so operators can retry cleanup safely.
            continue


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_extension_name(name: str) -> str:
    normalized = name.strip()
    if not _EXTENSION_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "Task runtime extension names must start with a lowercase letter and "
            "contain only lowercase letters, digits, or underscores"
        )
    return normalized


def _ensure_json_compatible(value: Any, *, label: str) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-compatible") from exc
