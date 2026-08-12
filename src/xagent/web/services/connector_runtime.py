"""Connector runtime context validation and task binding.

The web layer owns invocation trust, connector visibility, selected-ref
snapshots, and task-bound non-secret context persistence. Tool adapters only
consume the resolved runtime view later in the execution path.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Collection
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterable, cast

from sqlalchemy.orm import Session

from ...core.tools.adapters.vibe.connector_runtime import (
    CONNECTOR_TYPE_CUSTOM_API,
    CONNECTOR_TYPE_MCP,
    ERROR_CONNECTOR_NOT_FOUND,
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ERROR_INVALID_RUNTIME_CONTEXT,
    ERROR_MISSING_RUNTIME_CONTEXT,
    ERROR_RUNTIME_CONTEXT_IMMUTABLE,
    ERROR_RUNTIME_SECRET_NOT_ALLOWED,
    ERROR_RUNTIME_SECRET_UNAVAILABLE,
    ERROR_SCHEDULED_SECRET_UNAVAILABLE,
    RUNTIME_INPUT_AUTH_SELECTOR,
    RUNTIME_INPUT_CONTEXT,
    RUNTIME_INPUT_SECRETS,
    RUNTIME_SECRET_REASON_NOT_PROVIDED,
    RUNTIME_SECRET_REASON_STORE_LOST,
    ConnectorRef,
    ConnectorRuntimeError,
    ConnectorType,
    validate_runtime_source_key,
)
from ...core.tools.adapters.vibe.selection_spec import (
    ToolSelectionSpec,
    normalize_mcp_server_name,
)
from ..models.agent import Agent
from ..models.custom_api import CustomApi, UserCustomApi
from ..models.mcp import MCPServer, UserMCPServer
from ..models.task import Task, TaskConnectorRuntimeContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConnectorRuntimePayload:
    ref: ConnectorRef
    context: dict[str, Any]
    secrets: dict[str, Any]
    auth_selector: dict[str, Any]


@dataclass(frozen=True)
class ConnectorRuntimeCreatePlan:
    task_source: str | None
    connector_user_id: int
    selected_refs: tuple[ConnectorRef, ...]
    context_by_ref: dict[ConnectorRef, dict[str, Any]]
    ephemeral_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class ConnectorRuntimeAppendPlan:
    ephemeral_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class ConnectorRuntimeValues:
    context: dict[str, Any]
    secrets: dict[str, Any]
    auth_selector: dict[str, Any]

    def to_runtime_config(self) -> dict[str, Any]:
        return {
            RUNTIME_INPUT_CONTEXT: dict(self.context),
            RUNTIME_INPUT_SECRETS: dict(self.secrets),
            RUNTIME_INPUT_AUTH_SELECTOR: dict(self.auth_selector),
        }


@dataclass(frozen=True)
class ConnectorRuntimeRequest:
    task_id: int
    turn_id: str | None
    user_id: int | None
    connector_ref: ConnectorRef
    values: ConnectorRuntimeValues
    task_source: str | None = None


ConnectorRuntimeResolver = Callable[
    [ConnectorRuntimeRequest], ConnectorRuntimeValues | None
]


@dataclass(frozen=True)
class _ConnectorRuntimeResolverRegistration:
    resolver: ConnectorRuntimeResolver
    task_sources: frozenset[str] | None


# The default OSS store is process-local and single-turn: it is only reliable
# when the create/append request and the worker that consumes the turn run in
# the same process. Multi-worker deployments should provide ephemeral secrets
# through the resolver hook or a deployment-owned distributed secret store.
_EPHEMERAL_RUNTIME_VALUES: dict[str, dict[str, Any]] = {}
_EPHEMERAL_RUNTIME_MANIFESTS: dict[str, dict[str, dict[str, set[str]]]] = {}
_EPHEMERAL_RUNTIME_VALUES_LOCK = RLock()
_RUNTIME_RESOLVER_REGISTRATION: _ConnectorRuntimeResolverRegistration | None = None


def set_connector_runtime_resolver(
    resolver: ConnectorRuntimeResolver | None,
    *,
    task_sources: Collection[str] | None = None,
) -> None:
    """Install the server-side hook that can supply runtime values.

    ``task_sources=None`` preserves the legacy global behavior. A supplied
    non-empty collection limits the hook to exact ``Task.source`` matches.
    """

    global _RUNTIME_RESOLVER_REGISTRATION
    if resolver is None:
        if task_sources is not None:
            raise ValueError("task_sources requires a resolver")
        _RUNTIME_RESOLVER_REGISTRATION = None
        return
    _RUNTIME_RESOLVER_REGISTRATION = _ConnectorRuntimeResolverRegistration(
        resolver=resolver,
        task_sources=_normalize_resolver_task_sources(task_sources),
    )


def set_connector_runtime_resolver_for_testing(
    resolver: ConnectorRuntimeResolver | None,
    *,
    task_sources: Collection[str] | None = None,
) -> None:
    set_connector_runtime_resolver(resolver, task_sources=task_sources)


def _normalize_resolver_task_sources(
    task_sources: Collection[str] | None,
) -> frozenset[str] | None:
    if task_sources is None:
        return None
    if isinstance(task_sources, (str, bytes)):
        raise TypeError("task_sources must be a collection of strings")
    normalized = frozenset(task_sources)
    if not normalized:
        raise ValueError("task_sources must contain at least one source")
    if any(not isinstance(source, str) or not source for source in normalized):
        raise ValueError("task_sources must contain non-empty strings")
    if any(source != source.strip() for source in normalized):
        raise ValueError("task_sources must not contain surrounding whitespace")
    return normalized


def _runtime_resolver_for_task_source(
    task_source: str | None,
) -> ConnectorRuntimeResolver | None:
    registration = _RUNTIME_RESOLVER_REGISTRATION
    if registration is None:
        return None
    if (
        registration.task_sources is not None
        and task_source not in registration.task_sources
    ):
        return None
    return registration.resolver


def store_ephemeral_runtime_values(
    turn_id: str, values_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]]
) -> None:
    """Store per-turn secrets/auth selectors by turn id."""

    if not values_by_ref:
        return
    encoded = {
        ref.storage_key: {
            section: dict(values) for section, values in sections.items() if values
        }
        for ref, sections in values_by_ref.items()
    }
    manifest = {
        ref.storage_key: {
            section: set(values)
            for section, values in sections.items()
            if isinstance(values, dict) and values
        }
        for ref, sections in values_by_ref.items()
    }
    with _EPHEMERAL_RUNTIME_VALUES_LOCK:
        _EPHEMERAL_RUNTIME_VALUES[turn_id] = encoded
        _EPHEMERAL_RUNTIME_MANIFESTS[turn_id] = manifest


def pop_ephemeral_runtime_values(turn_id: str) -> dict[str, Any] | None:
    with _EPHEMERAL_RUNTIME_VALUES_LOCK:
        _EPHEMERAL_RUNTIME_MANIFESTS.pop(turn_id, None)
        return _EPHEMERAL_RUNTIME_VALUES.pop(turn_id, None)


def drop_ephemeral_runtime_values_for_testing(turn_id: str) -> None:
    """Simulate losing the secret values while keeping safe provenance."""

    with _EPHEMERAL_RUNTIME_VALUES_LOCK:
        _EPHEMERAL_RUNTIME_VALUES.pop(turn_id, None)


def get_ephemeral_runtime_values(turn_id: str) -> dict[str, Any] | None:
    with _EPHEMERAL_RUNTIME_VALUES_LOCK:
        values = _EPHEMERAL_RUNTIME_VALUES.get(turn_id)
        return dict(values) if isinstance(values, dict) else None


def get_ephemeral_runtime_manifest(
    turn_id: str,
) -> dict[str, dict[str, set[str]]] | None:
    with _EPHEMERAL_RUNTIME_VALUES_LOCK:
        manifest = _EPHEMERAL_RUNTIME_MANIFESTS.get(turn_id)
        if not isinstance(manifest, dict):
            return None
        return {
            ref_key: {
                section: set(keys)
                for section, keys in sections.items()
                if isinstance(keys, set)
            }
            for ref_key, sections in manifest.items()
            if isinstance(sections, dict)
        }


def load_connector_runtime_view(
    *,
    db: Session,
    task_id: int,
    turn_id: str | None,
    user_id: int | None,
    agent_team_id: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve task-bound and per-turn runtime values for tool creation."""

    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        return {}
    task_owner_user_id = _require_task_runtime_owner(task, expected_user_id=user_id)
    task_source = _task_source(task)

    selected_refs = _load_task_selected_refs(task)
    if not selected_refs:
        return {}

    persisted_context = _load_task_context_rows(db, task_id=task_id)
    ephemeral_by_ref = (
        get_ephemeral_runtime_values(turn_id) if isinstance(turn_id, str) else None
    )
    ephemeral_manifest = (
        get_ephemeral_runtime_manifest(turn_id) if isinstance(turn_id, str) else None
    )
    visible = _load_visible_runtime_connectors(
        db, user_id=task_owner_user_id, agent_team_id=agent_team_id
    )

    runtime_view: dict[str, dict[str, Any]] = {}
    for ref in selected_refs:
        connector = visible.get(ref)
        if connector is None:
            # Tool loading applies the same visibility filter when the
            # team-scope hook is installed, for both connector kinds, so a
            # ref absent here has no runtime tool either way. With only the
            # legacy hook installed, this view is user-keyed and unions a
            # legacy-granted team-shared custom API, while both tool
            # loaders resolve personal-only -- they consult the team-keyed
            # hook exclusively, never the legacy one -- so such a ref is
            # selectable and persistable into a task's connector selection
            # but builds no runtime tool.
            continue
        raw_ephemeral = (
            ephemeral_by_ref.get(ref.storage_key, {})
            if isinstance(ephemeral_by_ref, dict)
            else {}
        )
        values = ConnectorRuntimeValues(
            context=dict(persisted_context.get(ref, {})),
            secrets=dict(
                raw_ephemeral.get(RUNTIME_INPUT_SECRETS, {})
                if isinstance(raw_ephemeral, dict)
                else {}
            ),
            auth_selector=dict(
                raw_ephemeral.get(RUNTIME_INPUT_AUTH_SELECTOR, {})
                if isinstance(raw_ephemeral, dict)
                else {}
            ),
        )
        values = _resolve_runtime_values(
            task_id=task_id,
            turn_id=turn_id,
            task_source=task_source,
            user_id=task_owner_user_id,
            ref=ref,
            values=values,
        )
        _require_context_values(ref, connector, values.context)
        _require_ephemeral_values_at_binding(
            ref,
            connector,
            values,
            ephemeral_manifest=ephemeral_manifest,
            error_code=_binding_missing_ephemeral_error_code(task),
        )
        runtime_view[ref.storage_key] = values.to_runtime_config()

    return runtime_view


def prepare_create_connector_runtime(
    *,
    db: Session,
    agent: Agent,
    task_source: str | None,
    connector_user_id: int,
    payload_items: Iterable[Any] | None,
    allow_ephemeral: bool = True,
    missing_ephemeral_error_code: str = ERROR_RUNTIME_SECRET_UNAVAILABLE,
) -> ConnectorRuntimeCreatePlan:
    """Prepare runtime values for a task identity chosen by the caller.

    ``agent`` supplies tool-selection policy. ``connector_user_id`` supplies
    connector visibility and must be the same owner later persisted on Task.
    """

    visible = _load_visible_runtime_connectors(
        db,
        user_id=int(connector_user_id),
        agent_team_id=int(agent.team_id) if agent.team_id is not None else None,
    )
    selected_refs = _plan_selected_refs(agent, visible)
    payload_by_ref = _parse_payload_items(payload_items)
    if not allow_ephemeral:
        _reject_ephemeral_payload_values(payload_by_ref)
    _validate_payload_refs(payload_by_ref, visible=visible, selected_refs=selected_refs)

    context_by_ref: dict[ConnectorRef, dict[str, Any]] = {}
    ephemeral_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]] = {}

    for ref in selected_refs:
        connector = visible[ref]
        payload = payload_by_ref.get(ref)
        context = dict(payload.context) if payload is not None else {}
        secrets = dict(payload.secrets) if payload is not None else {}
        auth_selector = dict(payload.auth_selector) if payload is not None else {}
        _validate_values_against_schema(ref, connector, context, secrets, auth_selector)
        _require_context_values(ref, connector, context)
        if _runtime_resolver_for_task_source(task_source) is None:
            _require_ephemeral_values(
                ref,
                connector,
                secrets,
                auth_selector,
                error_code=missing_ephemeral_error_code,
            )
        if context:
            context_by_ref[ref] = context
        if secrets or auth_selector:
            ephemeral_by_ref[ref] = {
                RUNTIME_INPUT_SECRETS: secrets,
                RUNTIME_INPUT_AUTH_SELECTOR: auth_selector,
            }

    return ConnectorRuntimeCreatePlan(
        task_source=task_source,
        connector_user_id=int(connector_user_id),
        selected_refs=selected_refs,
        context_by_ref=context_by_ref,
        ephemeral_by_ref=ephemeral_by_ref,
    )


def prepare_connector_runtime_selection_snapshot(
    *,
    db: Session,
    agent: Agent | None,
    connector_user_id: int | None,
) -> tuple[ConnectorRef, ...]:
    """Return the connector-runtime closed set for a newly created task.

    This helper is intentionally selection-only: non-/v1 task creation paths do
    not accept per-invocation runtime payloads in this phase. ``agent`` supplies
    the agent's tool-selection policy, while ``connector_user_id`` supplies the
    same connector visibility scope used by normal web tool loading:
    ``connector_user_id``'s personal MCP/Custom API links, unioned with
    ``agent``'s owning team's connectors for both connector kinds when a
    team-keyed hook is installed. For published-agent chats, personal
    visibility still follows the task owner rather than the published
    agent's owner; team visibility follows the agent regardless of who is
    running it.
    """

    if agent is None or connector_user_id is None:
        return ()
    selected = resolve_agent_selected_connectors(
        db=db,
        agent=agent,
        connector_user_id=int(connector_user_id),
    )
    return _runtime_declared_refs(selected)


def bind_connector_runtime_selection_snapshot(
    *, task: Task, selected_refs: Iterable[ConnectorRef]
) -> None:
    """Attach a connector-runtime selection snapshot to a new task."""

    cast(Any, task).connector_runtime_selected_refs = [
        ref.to_wire() for ref in _sort_connector_refs(selected_refs)
    ]


def bind_create_connector_runtime_plan(
    *, task: Task, plan: ConnectorRuntimeCreatePlan
) -> None:
    """Validate and bind a create plan before the Task is persisted."""

    # Keep this service-boundary assertion even when a caller derives the Task
    # and plan from the same owner/source values.
    if (
        int(task.user_id) != plan.connector_user_id
        or _task_source(task) != plan.task_source
    ):
        raise ConnectorRuntimeError(
            ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
            "Connector runtime context is unavailable.",
            details={"reason": "runtime_task_identity_mismatch"},
            status_code=503,
        )
    bind_connector_runtime_selection_snapshot(
        task=task, selected_refs=plan.selected_refs
    )


def reject_ephemeral_connector_runtime_payload(
    payload_items: Iterable[Any] | None,
) -> None:
    """Validate that persisted runtime payload contains no ephemeral values."""

    _reject_ephemeral_payload_values(_parse_payload_items(payload_items))


def persist_create_connector_runtime_context(
    *, db: Session, task_id: int, plan: ConnectorRuntimeCreatePlan
) -> None:
    for ref, context in plan.context_by_ref.items():
        db.add(
            TaskConnectorRuntimeContext(
                task_id=task_id,
                connector_type=ref.connector_type,
                connector_id=ref.connector_id,
                context=_canonical_json_value(context),
            )
        )


def prepare_append_connector_runtime(
    *,
    db: Session,
    agent: Agent,
    task: Task,
    connector_user_id: int,
    payload_items: Iterable[Any] | None,
) -> ConnectorRuntimeAppendPlan:
    task_owner_user_id = _require_task_runtime_owner(
        task, expected_user_id=connector_user_id
    )
    task_source = _task_source(task)
    selected_refs = _load_task_selected_refs(task)
    payload_by_ref = _parse_payload_items(payload_items)
    visible = _load_visible_runtime_connectors(
        db,
        user_id=task_owner_user_id,
        agent_team_id=int(agent.team_id) if agent.team_id is not None else None,
    )
    _validate_payload_refs(payload_by_ref, visible=visible, selected_refs=selected_refs)

    persisted_context = _load_task_context_rows(db, task_id=int(task.id))
    ephemeral_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]] = {}

    for ref in selected_refs:
        connector = visible.get(ref)
        if connector is None:
            # Payload refs were already checked against current visibility above.
            # A historical selected ref that was later disabled/deleted should not
            # permanently block appends that do not try to supply values for it.
            continue
        payload = payload_by_ref.get(ref)
        context = dict(payload.context) if payload is not None else {}
        secrets = dict(payload.secrets) if payload is not None else {}
        auth_selector = dict(payload.auth_selector) if payload is not None else {}
        _validate_values_against_schema(ref, connector, context, secrets, auth_selector)
        if _runtime_resolver_for_task_source(task_source) is None:
            _require_ephemeral_values(ref, connector, secrets, auth_selector)
        stored = persisted_context.get(ref, {})
        _require_context_values(ref, connector, stored)
        if context and _canonical_json_value(context) != _canonical_json_value(stored):
            _raise_runtime_error(ERROR_RUNTIME_CONTEXT_IMMUTABLE, ref)
        if secrets or auth_selector:
            ephemeral_by_ref[ref] = {
                RUNTIME_INPUT_SECRETS: secrets,
                RUNTIME_INPUT_AUTH_SELECTOR: auth_selector,
            }

    return ConnectorRuntimeAppendPlan(ephemeral_by_ref=ephemeral_by_ref)


def _parse_payload_items(
    payload_items: Iterable[Any] | None,
) -> dict[ConnectorRef, ConnectorRuntimePayload]:
    result: dict[ConnectorRef, ConnectorRuntimePayload] = {}
    for item in payload_items or ():
        raw_ref = _read_field(item, "connector_ref")
        if hasattr(raw_ref, "model_dump"):
            raw_ref = raw_ref.model_dump()
        try:
            ref = ConnectorRef.from_wire(raw_ref)
        except ValueError as exc:
            raise ConnectorRuntimeError(
                ERROR_INVALID_RUNTIME_CONTEXT,
                "Invalid connector runtime context.",
                details={"reason": str(exc)},
            ) from exc
        if ref in result:
            _raise_runtime_error(
                ERROR_INVALID_RUNTIME_CONTEXT, ref, reason="duplicate_ref"
            )
        context = _optional_mapping(_read_field(item, RUNTIME_INPUT_CONTEXT))
        secrets = _optional_mapping(_read_field(item, RUNTIME_INPUT_SECRETS))
        auth_selector = _optional_mapping(
            _read_field(item, RUNTIME_INPUT_AUTH_SELECTOR)
        )
        if ref.connector_type == CONNECTOR_TYPE_CUSTOM_API and auth_selector:
            _raise_runtime_error(
                ERROR_INVALID_RUNTIME_CONTEXT, ref, reason="auth_selector_not_supported"
            )
        result[ref] = ConnectorRuntimePayload(
            ref=ref,
            context=context,
            secrets=secrets,
            auth_selector=auth_selector,
        )
    return result


def _reject_ephemeral_payload_values(
    payload_by_ref: dict[ConnectorRef, ConnectorRuntimePayload],
) -> None:
    for ref, payload in payload_by_ref.items():
        if payload.secrets or payload.auth_selector:
            _raise_runtime_error(ERROR_RUNTIME_SECRET_NOT_ALLOWED, ref)


def _read_field(item: Any, field: str) -> Any:
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)


def _optional_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConnectorRuntimeError(
            ERROR_INVALID_RUNTIME_CONTEXT,
            "Invalid connector runtime context.",
            details={"reason": "runtime section must be an object"},
        )
    return dict(value)


def _load_visible_runtime_connectors(
    db: Session, *, user_id: int, agent_team_id: int | None = None
) -> dict[ConnectorRef, Any]:
    from .connector_team_scope import (
        resolve_team_connector_ids_or_raise,
        team_connector_hook_installed,
        visible_team_connector_ids,
    )

    visible: dict[ConnectorRef, Any] = {}
    if team_connector_hook_installed():
        # Team ownership is resolved from the team that owns the governing
        # agent. A run with no governing agent resolves personal links only.
        team_ids = resolve_team_connector_ids_or_raise(
            db, team_id=agent_team_id, log_subject=user_id
        )
        # Both connector kinds are team-keyed together: the tool-build
        # loaders (WebToolConfig's MCP and custom-API paths) both consult
        # the same team scope now, so a team-owned custom API selected into
        # a task's runtime snapshot here always has a matching tool-loader
        # consumer to build it. Consuming the "custom_api" half without a
        # matching consumer would let a selected connector fail at
        # runtime-view resolution while never building a tool either --
        # that hazard is why the two sides move together, not separately.
    else:
        # No team-keyed hook: keep the legacy user-keyed overlay exactly as
        # it is, so an installation that adopts this revision without
        # installing the new hook behaves as it did before. Selection is on
        # hook presence, never on an empty result -- an installed hook
        # answers with empty sets for a team that owns nothing, and that
        # answer is authoritative.
        team_ids = visible_team_connector_ids(db, int(user_id))

    own_mcp = (
        db.query(MCPServer)
        .join(UserMCPServer, MCPServer.id == UserMCPServer.mcpserver_id)
        .filter(UserMCPServer.user_id == user_id, UserMCPServer.is_active)
        .all()
    )
    mcp_by_id = {int(s.id): s for s in own_mcp}
    missing = [sid for sid in team_ids["mcp"] if sid not in mcp_by_id]
    if missing:
        for server in db.query(MCPServer).filter(MCPServer.id.in_(missing)).all():
            mcp_by_id[int(server.id)] = server
    for sid, server in mcp_by_id.items():
        visible[ConnectorRef(cast(ConnectorType, CONNECTOR_TYPE_MCP), sid)] = server

    own_api = (
        db.query(CustomApi)
        .join(UserCustomApi, CustomApi.id == UserCustomApi.custom_api_id)
        .filter(UserCustomApi.user_id == user_id, UserCustomApi.is_active)
        .all()
    )
    api_by_id = {int(a.id): a for a in own_api}
    missing = [aid for aid in team_ids["custom_api"] if aid not in api_by_id]
    if missing:
        for api in db.query(CustomApi).filter(CustomApi.id.in_(missing)).all():
            api_by_id[int(api.id)] = api
    for aid, api in api_by_id.items():
        visible[ConnectorRef(cast(ConnectorType, CONNECTOR_TYPE_CUSTOM_API), aid)] = api
    return visible


def resolve_agent_selected_connectors(
    *, db: Session, agent: Agent, connector_user_id: int
) -> dict[ConnectorRef, Any]:
    """Return visible connectors selected by ``agent`` for the supplied owner.

    Selection derives only from ``Agent.tool_categories``; task-level runtime
    overlays are not applied. Callers that need the effective per-task tool set
    must apply those policies at the task-selection boundary.

    Runtime-declaration filtering is intentionally left to consumers. Results
    iterate deterministically by ``(connector_type, connector_id)``.
    """

    visible = _load_visible_runtime_connectors(
        db,
        user_id=int(connector_user_id),
        agent_team_id=int(agent.team_id) if agent.team_id is not None else None,
    )
    return _select_agent_visible_connectors(agent, visible)


def _plan_selected_refs(
    agent: Agent, visible: dict[ConnectorRef, Any]
) -> tuple[ConnectorRef, ...]:
    selected = _select_agent_visible_connectors(agent, visible)
    return _runtime_declared_refs(selected)


def _runtime_declared_refs(
    selected: dict[ConnectorRef, Any],
) -> tuple[ConnectorRef, ...]:
    """Project declared refs while preserving canonical selection order.

    Callers provide the mapping ordered by the connector selection owner.
    """

    return tuple(
        ref
        for ref, connector in selected.items()
        if _has_runtime_declaration(connector)
    )


def _select_agent_visible_connectors(
    agent: Agent, visible: dict[ConnectorRef, Any]
) -> dict[ConnectorRef, Any]:
    """Select visible connectors and establish canonical ref iteration order."""

    tool_categories = (
        list(agent.tool_categories) if isinstance(agent.tool_categories, list) else None
    )
    spec = ToolSelectionSpec.from_raw(tool_categories=tool_categories)
    selected: dict[ConnectorRef, Any] = {}
    for ref, connector in visible.items():
        if _is_agent_selected_connector(spec, ref, connector):
            selected[ref] = connector
    return {ref: selected[ref] for ref in _sort_connector_refs(selected)}


def _is_agent_selected_connector(
    spec: ToolSelectionSpec, ref: ConnectorRef, connector: Any
) -> bool:
    if ref.connector_type == CONNECTOR_TYPE_MCP:
        if not spec.includes_mcp():
            return False
    elif ref.connector_type == CONNECTOR_TYPE_CUSTOM_API:
        if not spec.includes_custom_api():
            return False
    else:
        return False

    scoped_mcp_servers = spec.scoped_mcp_servers()
    if scoped_mcp_servers is None:
        return True
    name_key = normalize_mcp_server_name(connector.name)
    return name_key in scoped_mcp_servers


def _sort_connector_refs(refs: Iterable[ConnectorRef]) -> tuple[ConnectorRef, ...]:
    return tuple(sorted(refs, key=lambda ref: (ref.connector_type, ref.connector_id)))


def _has_runtime_declaration(connector: Any) -> bool:
    return bool(getattr(connector, "runtime_input_schema", None)) or bool(
        getattr(connector, "runtime_bindings", None)
    )


def _validate_payload_refs(
    payload_by_ref: dict[ConnectorRef, ConnectorRuntimePayload],
    *,
    visible: dict[ConnectorRef, Any],
    selected_refs: tuple[ConnectorRef, ...],
) -> None:
    selected = set(selected_refs)
    for ref in payload_by_ref:
        if ref not in visible:
            _raise_runtime_error(ERROR_CONNECTOR_NOT_FOUND, ref)
        if ref not in selected:
            _raise_runtime_error(
                ERROR_INVALID_RUNTIME_CONTEXT, ref, reason="connector_not_selected"
            )


def _load_task_selected_refs(task: Task) -> tuple[ConnectorRef, ...]:
    raw_refs = task.connector_runtime_selected_refs
    if raw_refs is None:
        return ()
    if not isinstance(raw_refs, list):
        raise ConnectorRuntimeError(
            ERROR_INVALID_RUNTIME_CONTEXT,
            "Invalid connector runtime context.",
            details={"reason": "stored selected refs must be a list"},
        )
    try:
        return tuple(sorted(ConnectorRef.from_wire(raw_ref) for raw_ref in raw_refs))
    except ValueError as exc:
        raise ConnectorRuntimeError(
            ERROR_INVALID_RUNTIME_CONTEXT,
            "Invalid connector runtime context.",
            details={"reason": str(exc)},
        ) from exc


def _load_task_context_rows(
    db: Session, *, task_id: int
) -> dict[ConnectorRef, dict[str, Any]]:
    rows = (
        db.query(TaskConnectorRuntimeContext)
        .filter(TaskConnectorRuntimeContext.task_id == task_id)
        .all()
    )
    result: dict[ConnectorRef, dict[str, Any]] = {}
    for row in rows:
        connector_type = cast(ConnectorType, str(row.connector_type))
        ref = ConnectorRef(connector_type, int(row.connector_id))
        context: dict[str, Any] = row.context if isinstance(row.context, dict) else {}
        result[ref] = dict(context)
    return result


def _validate_values_against_schema(
    ref: ConnectorRef,
    connector: Any,
    context: dict[str, Any],
    secrets: dict[str, Any],
    auth_selector: dict[str, Any],
) -> None:
    schema = _runtime_input_schema(connector)
    for section_name, values in (
        (RUNTIME_INPUT_CONTEXT, context),
        (RUNTIME_INPUT_SECRETS, secrets),
        (RUNTIME_INPUT_AUTH_SELECTOR, auth_selector),
    ):
        declarations = _schema_section(schema, section_name)
        if (
            section_name == RUNTIME_INPUT_AUTH_SELECTOR
            and ref.connector_type != CONNECTOR_TYPE_MCP
        ):
            if values:
                _raise_runtime_error(
                    ERROR_INVALID_RUNTIME_CONTEXT,
                    ref,
                    reason="auth_selector_not_supported",
                )
            continue
        for key in values:
            try:
                validate_runtime_source_key(key)
            except ValueError as exc:
                _raise_runtime_error(
                    ERROR_INVALID_RUNTIME_CONTEXT, ref, reason=str(exc)
                )
            if key not in declarations:
                _raise_runtime_error(
                    ERROR_INVALID_RUNTIME_CONTEXT,
                    ref,
                    reason=f"undeclared_{section_name}_key",
                )


def _runtime_input_schema(connector: Any) -> dict[str, Any]:
    schema = getattr(connector, "runtime_input_schema", None)
    return schema if isinstance(schema, dict) else {}


def _schema_section(schema: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = schema.get(section_name)
    return section if isinstance(section, dict) else {}


def _require_context_values(
    ref: ConnectorRef, connector: Any, context: dict[str, Any]
) -> None:
    declarations = _schema_section(
        _runtime_input_schema(connector), RUNTIME_INPUT_CONTEXT
    )
    for key, declaration in declarations.items():
        try:
            validate_runtime_source_key(key)
        except ValueError as exc:
            _raise_runtime_error(ERROR_INVALID_RUNTIME_CONTEXT, ref, reason=str(exc))
        if _is_required(declaration) and key not in context:
            _raise_runtime_error(
                ERROR_MISSING_RUNTIME_CONTEXT, ref, reason=f"missing_context.{key}"
            )


def _require_ephemeral_values(
    ref: ConnectorRef,
    connector: Any,
    secrets: dict[str, Any],
    auth_selector: dict[str, Any],
    *,
    error_code: str = ERROR_RUNTIME_SECRET_UNAVAILABLE,
) -> None:
    schema = _runtime_input_schema(connector)
    for section_name, values in (
        (RUNTIME_INPUT_SECRETS, secrets),
        (RUNTIME_INPUT_AUTH_SELECTOR, auth_selector),
    ):
        declarations = _schema_section(schema, section_name)
        for key, declaration in declarations.items():
            try:
                validate_runtime_source_key(key)
            except ValueError as exc:
                _raise_runtime_error(
                    ERROR_INVALID_RUNTIME_CONTEXT, ref, reason=str(exc)
                )
            if _is_required(declaration) and key not in values:
                _raise_runtime_error(
                    error_code,
                    ref,
                    reason=RUNTIME_SECRET_REASON_NOT_PROVIDED,
                )


def _require_ephemeral_values_at_binding(
    ref: ConnectorRef,
    connector: Any,
    values: ConnectorRuntimeValues,
    *,
    ephemeral_manifest: dict[str, dict[str, set[str]]] | None,
    error_code: str = ERROR_RUNTIME_SECRET_UNAVAILABLE,
) -> None:
    schema = _runtime_input_schema(connector)
    for section_name, section_values in (
        (RUNTIME_INPUT_SECRETS, values.secrets),
        (RUNTIME_INPUT_AUTH_SELECTOR, values.auth_selector),
    ):
        declarations = _schema_section(schema, section_name)
        for key, declaration in declarations.items():
            if _is_required(declaration) and key not in section_values:
                reason = (
                    RUNTIME_SECRET_REASON_STORE_LOST
                    if _manifest_has_ephemeral_key(
                        ephemeral_manifest, ref, section_name, key
                    )
                    else RUNTIME_SECRET_REASON_NOT_PROVIDED
                )
                _raise_runtime_error(
                    error_code,
                    ref,
                    reason=reason,
                )


def _binding_missing_ephemeral_error_code(task: Task) -> str:
    if str(getattr(task, "source", "")) != "trigger":
        return ERROR_RUNTIME_SECRET_UNAVAILABLE
    config = getattr(task, "agent_config", None)
    if not isinstance(config, dict):
        return ERROR_RUNTIME_SECRET_UNAVAILABLE
    if str(config.get("trigger_type")) == "scheduled":
        return ERROR_SCHEDULED_SECRET_UNAVAILABLE
    return ERROR_RUNTIME_SECRET_UNAVAILABLE


def _require_task_runtime_owner(task: Task, *, expected_user_id: int | None) -> int:
    task_owner_user_id = int(task.user_id)
    # ``Task.user_id`` remains the owner SSOT; the optional expected value is a
    # defensive assertion for service callers that carry an independent owner.
    if expected_user_id is not None and int(expected_user_id) != task_owner_user_id:
        raise ConnectorRuntimeError(
            ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
            "Connector runtime context is unavailable.",
            details={"reason": "runtime_owner_mismatch"},
            status_code=503,
        )
    return task_owner_user_id


def _task_source(task: Task) -> str | None:
    source = task.source
    return source if isinstance(source, str) else None


def _resolve_runtime_values(
    *,
    task_id: int,
    turn_id: str | None,
    task_source: str | None,
    user_id: int,
    ref: ConnectorRef,
    values: ConnectorRuntimeValues,
) -> ConnectorRuntimeValues:
    resolver = _runtime_resolver_for_task_source(task_source)
    if resolver is None:
        return values
    resolved = resolver(
        ConnectorRuntimeRequest(
            task_id=task_id,
            turn_id=turn_id,
            user_id=user_id,
            connector_ref=ref,
            values=values,
            task_source=task_source,
        )
    )
    return resolved if resolved is not None else values


def _manifest_has_ephemeral_key(
    manifest: dict[str, dict[str, set[str]]] | None,
    ref: ConnectorRef,
    section_name: str,
    key: str,
) -> bool:
    if not isinstance(manifest, dict):
        return False
    ref_manifest = manifest.get(ref.storage_key)
    if not isinstance(ref_manifest, dict):
        return False
    keys = ref_manifest.get(section_name)
    return isinstance(keys, set) and key in keys


def _is_required(declaration: Any) -> bool:
    return isinstance(declaration, dict) and bool(declaration.get("required"))


def _canonical_json_value(value: dict[str, Any]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"))),
    )


def _raise_runtime_error(
    code: str, ref: ConnectorRef, *, reason: str | None = None
) -> None:
    details = {}
    if reason is not None:
        details["reason"] = reason
    raise ConnectorRuntimeError(
        code,
        _message_for_code(code),
        connector_ref=ref,
        details=details,
        status_code=_status_for_code(code),
    )


def _message_for_code(code: str) -> str:
    return {
        ERROR_CONNECTOR_NOT_FOUND: "Connector not found or not accessible.",
        ERROR_INVALID_RUNTIME_CONTEXT: "Invalid connector runtime context.",
        ERROR_MISSING_RUNTIME_CONTEXT: "Required connector runtime context is missing.",
        ERROR_RUNTIME_CONTEXT_IMMUTABLE: "Connector runtime context cannot change after task creation.",
        ERROR_RUNTIME_SECRET_UNAVAILABLE: "Required runtime secret is unavailable.",
        ERROR_SCHEDULED_SECRET_UNAVAILABLE: "Required scheduled runtime secret is unavailable.",
        ERROR_RUNTIME_SECRET_NOT_ALLOWED: "Runtime secret is not allowed for this entrypoint.",
    }.get(code, "Invalid connector runtime context.")


def _status_for_code(code: str) -> int:
    if code == ERROR_CONNECTOR_NOT_FOUND:
        return 404
    if code == ERROR_RUNTIME_CONTEXT_IMMUTABLE:
        return 409
    return 400
