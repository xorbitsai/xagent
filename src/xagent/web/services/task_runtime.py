"""Process-wide registry for task-scoped runtime extensions.

Closed-source distributions can register providers at application startup.
The open-source task lifecycle then dispatches provider hooks without knowing
which runtime kinds, binding tables, or transports those providers implement.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal

from ...config import (
    get_task_runtime_hook_max_workers,
    get_task_runtime_hook_queue_timeout_seconds,
)
from ...core.task_runtime import (
    EMPTY_TASK_RUNTIME_CONTRIBUTION,
    MAX_TASK_RUNTIME_EXTENSIONS,
    MAX_TASK_RUNTIME_JSON_BYTES,
    MAX_TASK_RUNTIME_PUBLIC_METADATA_BYTES,
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
_task_runtime_extensions_lock = RLock()
_TASK_RUNTIME_HOOK_TIMEOUT_SECONDS = {
    "on_task_created": 30.0,
    "build_runtime": 10.0,
    "public_metadata": 10.0,
    "on_task_deleted": 30.0,
}
_task_runtime_hook_executor: ThreadPoolExecutor | None = None
_task_runtime_hook_executor_lock = RLock()
_PUBLIC_METADATA_STATUS_RESERVE_BYTES = 2 * 1024
logger = logging.getLogger(__name__)


class _ProviderHookRaised:
    """Carry a provider BaseException through the wait_for task boundary."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


class TaskRuntimeExtensionError(RuntimeError):
    """One registered provider failed a lifecycle operation."""

    def __init__(self, extension: str, operation: str, cause: BaseException):
        super().__init__(
            f"Task runtime extension '{extension}' failed during {operation}: {cause}"
        )
        self.extension = extension
        self.operation = operation
        self.cause = cause


@dataclass(frozen=True)
class TaskRuntimePublicMetadata:
    """Provider metadata plus aggregate delivery status for API callers."""

    extensions: dict[str, dict[str, Any]]
    status: Literal["complete", "truncated"] = "complete"
    omitted_extensions: tuple[str, ...] = ()


def shutdown_task_runtime_hook_executor() -> None:
    """Stop accepting provider hooks and cancel work that has not started."""

    global _task_runtime_hook_executor
    with _task_runtime_hook_executor_lock:
        executor = _task_runtime_hook_executor
        _task_runtime_hook_executor = None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


def _get_task_runtime_hook_executor() -> ThreadPoolExecutor:
    global _task_runtime_hook_executor
    with _task_runtime_hook_executor_lock:
        if _task_runtime_hook_executor is None:
            _task_runtime_hook_executor = ThreadPoolExecutor(
                max_workers=get_task_runtime_hook_max_workers(),
                thread_name_prefix="xagent-task-runtime",
            )
        return _task_runtime_hook_executor


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
    with _task_runtime_extensions_lock:
        if normalized in _task_runtime_extensions and not replace:
            raise ValueError(
                f"Task runtime extension '{normalized}' is already registered"
            )
        _task_runtime_extensions[normalized] = provider


def unregister_task_extension(name: str) -> TaskRuntimeExtensionProvider | None:
    """Remove and return one provider, if registered.

    Raises:
        ValueError: If ``name`` is not a valid extension name.
    """

    normalized = _normalize_extension_name(name)
    with _task_runtime_extensions_lock:
        return _task_runtime_extensions.pop(normalized, None)


def registered_task_extensions() -> tuple[str, ...]:
    """Return registered names in deterministic dispatch order."""

    return tuple(name for name, _provider in _registered_extension_items())


def validate_task_extension_requests(value: Any) -> dict[str, dict[str, Any]]:
    """Validate create-time provider configurations without invoking hooks."""

    if value is None:
        return {}
    # Pydantic rejects these two shape limits on the HTTP path. Keep the checks
    # here as defense-in-depth for SDK and internal callers that invoke the
    # service directly.
    if not isinstance(value, Mapping):
        raise TypeError("runtime_extensions must be an object")
    if len(value) > MAX_TASK_RUNTIME_EXTENSIONS:
        raise ValueError(
            f"runtime_extensions supports at most {MAX_TASK_RUNTIME_EXTENSIONS} entries"
        )

    registered_names = {name for name, _provider in _registered_extension_items()}
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_configuration in value.items():
        name = _normalize_extension_name(str(raw_name))
        if name not in registered_names:
            raise ValueError(f"Task runtime extension '{name}' is not registered")
        if not isinstance(raw_configuration, Mapping):
            raise TypeError(
                f"Task runtime extension '{name}' configuration must be an object"
            )
        configuration = dict(raw_configuration)
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

    try:
        normalized = validate_task_extension_requests(requests)
        with _task_runtime_extensions_lock:
            providers = tuple(
                (name, _task_runtime_extensions[name], configuration)
                for name, configuration in normalized.items()
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskRuntimeExtensionError(
            "registry",
            "validate_requests",
            exc,
        ) from exc

    completed: list[tuple[str, TaskRuntimeExtensionProvider]] = []
    for name, provider, configuration in providers:
        try:
            await _invoke_provider_hook(
                provider,
                "on_task_created",
                context,
                configuration,
            )
        except BaseException as exc:
            _propagate_control_flow_exception(exc)
            await _cleanup_after_create_failure(
                context,
                providers=(*completed, (name, provider)),
            )
            raise TaskRuntimeExtensionError(name, "on_task_created", exc) from exc
        completed.append((name, provider))


async def build_task_runtime(
    context: TaskRuntimeContext,
) -> TaskRuntimeContribution:
    """Build providers independently and merge every successful contribution."""

    contributions: dict[str, TaskRuntimeContribution | None] = {}
    merged = EMPTY_TASK_RUNTIME_CONTRIBUTION
    for name, provider in _registered_extension_items():
        try:
            contribution = await _invoke_provider_hook(
                provider,
                "build_runtime",
                context,
            )
            normalized = normalize_task_runtime_contribution(contribution)
            candidate = {**contributions, name: normalized}
            merged = merge_task_runtime_contributions(candidate)
        except BaseException as exc:
            _propagate_control_flow_exception(exc)
            logger.error(
                "Dropping task runtime contribution from extension '%s': %s",
                name,
                exc,
                exc_info=True,
            )
            continue
        contributions[name] = normalized
    return merged


async def get_task_runtime_public_metadata(
    context: TaskRuntimeContext,
) -> TaskRuntimePublicMetadata:
    """Return provider-selected, JSON-safe metadata suitable for clients."""

    result: dict[str, dict[str, Any]] = {}
    omitted_extensions: list[str] = []
    for name, provider in _registered_extension_items():
        try:
            metadata = await _invoke_provider_hook(
                provider,
                "public_metadata",
                context,
            )
            if metadata is None:
                continue
            if not isinstance(metadata, Mapping):
                raise TypeError("public_metadata must return an object or None")
            detached = dict(metadata)
            _ensure_json_compatible(detached, label=f"'{name}' public metadata")
            if detached:
                candidate = {**result, name: detached}
                if _json_encoded_size(candidate) > (
                    MAX_TASK_RUNTIME_PUBLIC_METADATA_BYTES
                    - _PUBLIC_METADATA_STATUS_RESERVE_BYTES
                ):
                    omitted_extensions.append(name)
                    logger.warning(
                        "Omitting public metadata from task runtime extension "
                        "'%s' because the aggregate response reached %d bytes",
                        name,
                        MAX_TASK_RUNTIME_PUBLIC_METADATA_BYTES,
                    )
                else:
                    result[name] = detached
        except BaseException as exc:
            _propagate_control_flow_exception(exc)
            raise TaskRuntimeExtensionError(name, "public_metadata", exc) from exc
    return TaskRuntimePublicMetadata(
        extensions=result,
        status="truncated" if omitted_extensions else "complete",
        omitted_extensions=tuple(omitted_extensions),
    )


async def delete_task_extensions(context: TaskRuntimeContext) -> None:
    """Release provider-owned state before the core task row is deleted.

    All providers are attempted even when one fails. A combined extension error
    is raised afterwards so callers can log cleanup failures and decide whether
    to continue deleting the core task.
    """

    failures: list[tuple[str, BaseException]] = []
    for name, provider in reversed(_registered_extension_items()):
        try:
            await _invoke_provider_hook(
                provider,
                "on_task_deleted",
                context,
            )
        except BaseException as exc:
            _propagate_control_flow_exception(exc)
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
    providers: tuple[tuple[str, TaskRuntimeExtensionProvider], ...],
) -> None:
    for name, provider in reversed(providers):
        try:
            await _invoke_provider_hook(
                provider,
                "on_task_deleted",
                context,
            )
        except BaseException as exc:
            _propagate_control_flow_exception(exc)
            # The original create failure remains primary. Providers should make
            # deletion idempotent so operators can retry cleanup safely.
            logger.warning(
                "Cleanup failed for task runtime extension '%s' after task "
                "creation failure",
                name,
                exc_info=True,
            )
            continue


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _invoke_provider_hook(
    provider: TaskRuntimeExtensionProvider,
    operation: str,
    *args: Any,
) -> Any:
    """Invoke one untrusted provider hook without blocking the event loop forever.

    The initial call runs in a worker so a synchronous provider cannot block the
    event loop. Cancelling a timed-out worker cannot stop Python code already
    running in that thread, but it does bound the request path and async hooks
    receive normal cancellation.
    """

    hook = getattr(provider, operation)
    timeout = _TASK_RUNTIME_HOOK_TIMEOUT_SECONDS[operation]
    loop = asyncio.get_running_loop()
    started: Future[None] = Future()

    def _call_hook() -> Any:
        started.set_result(None)
        return hook(*args)

    async def _invoke() -> Any:
        try:
            value = await execution_future
            return await _maybe_await(value)
        except asyncio.CancelledError as exc:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            return _ProviderHookRaised(exc)
        except BaseException as exc:
            # ``SystemExit``/``KeyboardInterrupt`` raised inside a child Task can
            # otherwise terminate the event loop before the awaiting caller can
            # observe them. Carry then re-raise them outside ``wait_for``.
            return _ProviderHookRaised(exc)

    execution_future = loop.run_in_executor(
        _get_task_runtime_hook_executor(),
        _call_hook,
    )
    started_future = asyncio.wrap_future(started)
    queue_timeout = get_task_runtime_hook_queue_timeout_seconds()
    try:
        await asyncio.wait_for(
            asyncio.shield(started_future),
            timeout=queue_timeout,
        )
    except TimeoutError as exc:
        execution_future.cancel()
        raise TimeoutError(
            f"Provider hook queue wait exceeded the {queue_timeout:g}-second timeout"
        ) from exc
    except asyncio.CancelledError:
        execution_future.cancel()
        raise

    try:
        result = await asyncio.wait_for(
            _invoke(),
            timeout=timeout,
        )
    except TimeoutError as exc:
        raise TimeoutError(
            f"Provider hook execution exceeded the {timeout:g}-second timeout"
        ) from exc
    if isinstance(result, _ProviderHookRaised):
        raise result.error
    return result


def _propagate_control_flow_exception(error: BaseException) -> None:
    """Preserve process control and cancellation of the surrounding task."""

    if isinstance(error, (SystemExit, KeyboardInterrupt)):
        raise error
    if isinstance(error, asyncio.CancelledError):
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise error


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
        encoded_size = _json_encoded_size(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-compatible") from exc
    if encoded_size > MAX_TASK_RUNTIME_JSON_BYTES:
        raise ValueError(
            f"{label} exceeds the {MAX_TASK_RUNTIME_JSON_BYTES}-byte limit"
        )


def _json_encoded_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _registered_extension_items() -> tuple[
    tuple[str, TaskRuntimeExtensionProvider], ...
]:
    """Snapshot the registry so no dispatch iterates a live dict across await."""

    with _task_runtime_extensions_lock:
        return tuple(_task_runtime_extensions.items())
