from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Protocol, final
from uuid import UUID

from sqlalchemy.orm import Session

from ... import config as xagent_config
from ...builtin_identity import builtin_provenance_identity
from ...core.tools.adapters.vibe.config import ACTOR_STDIO_SHADOWED_REASON
from ...core.tools.adapters.vibe.selection_spec import normalize_mcp_server_name
from ..builtin_mcp_registry import (
    get_builtin_execution_fields,
    get_builtin_public_mcp_app_rows,
    get_builtin_stdio_session_scope,
)
from ..models.public_mcp import PublicMCPApp
from .actor_mcp_connections import (
    ActorMCPConnectionCredentialCorruptionError,
    ActorMCPConnectionNotFoundOrStaleError,
    ActorMCPConnectionValidationError,
    get_actor_mcp_connection_credentials_internal,
    list_actor_mcp_connection_metadata,
)
from .mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPActorExecutionIdentity,
    caller_id_env,
)
from .user_oauth import normalize_user_oauth_resource_owner_key

logger = logging.getLogger(__name__)


_ACTOR_MCP_STORAGE_ERRORS = (
    ActorMCPConnectionValidationError,
    ActorMCPConnectionNotFoundOrStaleError,
    ActorMCPConnectionCredentialCorruptionError,
)


class ActorMCPRuntimeDefinitionError(ValueError):
    """An actor stdio identity or its canonical definition is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        drift_fields: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.drift_fields = drift_fields


@dataclass(frozen=True)
class ActorMCPStdioConnectionIdentity:
    """Non-secret identity required to resolve one actor stdio connection."""

    user_id: int
    resource_owner_key: str = field(repr=False)
    app_id: str
    catalog_app_generation: UUID
    lifecycle_generation: UUID

    def __post_init__(self) -> None:
        if (
            isinstance(self.user_id, bool)
            or not isinstance(self.user_id, int)
            or self.user_id <= 0
        ):
            raise ActorMCPRuntimeDefinitionError(
                "actor stdio identity requires a persisted user_id"
            )
        owner_key = normalize_user_oauth_resource_owner_key(self.resource_owner_key)
        if owner_key != self.resource_owner_key:
            raise ActorMCPRuntimeDefinitionError(
                "actor stdio identity requires an exact resource_owner_key"
            )
        if (
            not isinstance(self.app_id, str)
            or not self.app_id
            or self.app_id != self.app_id.strip()
        ):
            raise ActorMCPRuntimeDefinitionError(
                "actor stdio identity requires an exact app_id"
            )
        if not isinstance(self.catalog_app_generation, UUID):
            raise ActorMCPRuntimeDefinitionError(
                "actor stdio identity requires catalog_app_generation"
            )
        if not isinstance(self.lifecycle_generation, UUID):
            raise ActorMCPRuntimeDefinitionError(
                "actor stdio identity requires lifecycle_generation"
            )


@dataclass(frozen=True)
class ActorMCPStdioSessionIdentity:
    """Complete key for one execution-scoped actor stdio child session."""

    execution: MCPActorExecutionIdentity
    connection: ActorMCPStdioConnectionIdentity

    @property
    def key(self) -> tuple[int, str, str, str, int, str, str, UUID, UUID]:
        return (
            self.execution.task_id,
            self.execution.run_id,
            self.execution.turn_id,
            self.execution.lease_attempt_id,
            self.connection.user_id,
            self.connection.resource_owner_key,
            self.connection.app_id,
            self.connection.catalog_app_generation,
            self.connection.lifecycle_generation,
        )


class ActorMCPStdioConnectionAdapter(Protocol):
    """Narrow secret boundary implemented by actor connection storage."""

    def list_connection_identities(
        self,
        db: Session,
        *,
        user_id: int,
        resource_owner_key: str,
    ) -> Sequence[ActorMCPStdioConnectionIdentity]: ...

    def get_connection_credentials(
        self,
        db: Session,
        *,
        user_id: int,
        resource_owner_key: str,
        app_id: str,
        catalog_app_generation: UUID,
        expected_lifecycle_generation: UUID,
    ) -> Mapping[str, str] | None: ...


@final
class ActorMCPConnectionServiceAdapter:
    """Stateless adapter over the lifecycle-fenced actor connection service."""

    def list_connection_identities(
        self,
        db: Session,
        *,
        user_id: int,
        resource_owner_key: str,
    ) -> Sequence[ActorMCPStdioConnectionIdentity]:
        metadata = list_actor_mcp_connection_metadata(
            db,
            user_id=user_id,
            resource_owner_key=resource_owner_key,
        )
        return tuple(
            ActorMCPStdioConnectionIdentity(
                user_id=item.user_id,
                resource_owner_key=item.resource_owner_key,
                app_id=item.app_id,
                catalog_app_generation=item.catalog_app_generation,
                lifecycle_generation=item.lifecycle_generation,
            )
            for item in metadata
        )

    def get_connection_credentials(
        self,
        db: Session,
        *,
        user_id: int,
        resource_owner_key: str,
        app_id: str,
        catalog_app_generation: UUID,
        expected_lifecycle_generation: UUID,
    ) -> Mapping[str, str] | None:
        snapshot = get_actor_mcp_connection_credentials_internal(
            db,
            user_id=user_id,
            resource_owner_key=resource_owner_key,
            app_id=app_id,
            catalog_app_generation=catalog_app_generation,
            expected_lifecycle_generation=expected_lifecycle_generation,
        )
        return snapshot.credentials


_ACTOR_MCP_CONNECTION_SERVICE_ADAPTER = ActorMCPConnectionServiceAdapter()


def production_actor_mcp_stdio_connection_adapter() -> ActorMCPStdioConnectionAdapter:
    """Return the immutable production storage adapter."""

    return _ACTOR_MCP_CONNECTION_SERVICE_ADAPTER


@dataclass(frozen=True)
class ActorMCPStdioResolution:
    configs: tuple[dict[str, Any], ...]
    blocked_server_ids: frozenset[int]
    blocked_server_reasons: tuple[tuple[int, str], ...] = ()
    session_identities: tuple[tuple[str, ActorMCPStdioSessionIdentity], ...] = ()


@cache
def _builtin_stdio_rows() -> tuple[dict[str, Any], ...]:
    return tuple(
        row
        for row in get_builtin_public_mcp_app_rows()
        if str(row.get("transport") or "").lower() == "stdio"
    )


@cache
def _reserved_stdio_keys() -> frozenset[str]:
    return frozenset(
        key
        for row in _builtin_stdio_rows()
        for key in (
            _dispatch_server_key(row.get("app_id")),
            _dispatch_server_key(row.get("name")),
        )
        if key is not None
    )


def _dispatch_server_key(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return normalize_mcp_server_name(value)


def _blocked_visible_server_ids(visible_servers: Sequence[Any]) -> frozenset[int]:
    reserved = _reserved_stdio_keys()
    return frozenset(
        int(server.id)
        for server in visible_servers
        if _dispatch_server_key(getattr(server, "name", None)) in reserved
    )


def _cached_builtin_stdio_execution(app_id: str) -> dict[str, Any] | None:
    return get_builtin_execution_fields(app_id)


def _canonical_oauth_scopes(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if (
        not isinstance(value, list)
        or any(not isinstance(scope, str) or not scope for scope in value)
        or len(value) != len(set(value))
    ):
        raise ActorMCPRuntimeDefinitionError(
            "actor stdio OAuth scope definition is invalid"
        )
    return tuple(sorted(value))


def _trusted_launch_fingerprint(value: object) -> dict[str, Any] | None:
    """Normalize only provenance version while retaining all trusted fields."""

    if not isinstance(value, Mapping):
        return None
    fingerprint = dict(value)
    if "builtin_provenance" in fingerprint:
        identity = builtin_provenance_identity(fingerprint["builtin_provenance"])
        if identity is None:
            return None
        fingerprint["builtin_provenance"] = identity
    return fingerprint


def _canonical_stdio_execution(
    db: Session,
    identity: ActorMCPStdioConnectionIdentity,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    execution = _cached_builtin_stdio_execution(identity.app_id)
    if execution is None or execution.get("transport") != "stdio":
        raise ActorMCPRuntimeDefinitionError(
            "actor stdio app is absent from the builtin registry"
        )

    app = (
        db.query(PublicMCPApp)
        .filter(PublicMCPApp.app_id == identity.app_id)
        .one_or_none()
    )
    normalized_id = _dispatch_server_key(identity.app_id)
    collision_count = sum(
        _dispatch_server_key(app_id) == normalized_id
        for (app_id,) in db.query(PublicMCPApp.app_id).all()
    )
    if app is None or collision_count != 1:
        raise ActorMCPRuntimeDefinitionError(
            "actor stdio catalog identity is unavailable or ambiguous"
        )
    if (
        app.generation != identity.catalog_app_generation
        or not app.is_visible_in_connector
    ):
        raise ActorMCPRuntimeDefinitionError(
            "actor stdio catalog lifecycle is stale or hidden"
        )

    persisted_execution = {
        "name": app.name,
        "transport": app.transport,
        "provider_name": app.provider_name,
        "oauth_scopes": _canonical_oauth_scopes(app.oauth_scopes),
        "launch_config": _trusted_launch_fingerprint(app.launch_config or {}),
    }
    expected_execution = {
        "name": execution["name"],
        "transport": execution["transport"],
        "provider_name": execution["provider_name"],
        "oauth_scopes": _canonical_oauth_scopes(execution["oauth_scopes"]),
        "launch_config": _trusted_launch_fingerprint(execution["launch_config"]),
    }
    if persisted_execution != expected_execution:
        drift_fields = tuple(
            field_name
            for field_name in (
                "name",
                "transport",
                "provider_name",
                "oauth_scopes",
                "launch_config",
            )
            if persisted_execution[field_name] != expected_execution[field_name]
        )
        raise ActorMCPRuntimeDefinitionError(
            "actor stdio persisted catalog definition has drifted",
            drift_fields=drift_fields,
        )

    launch = execution.get("launch_config")
    if not isinstance(launch, Mapping) or not isinstance(launch.get("command"), str):
        raise ActorMCPRuntimeDefinitionError(
            "actor stdio builtin execution definition is invalid"
        )
    args = launch.get("args", [])
    fields = launch.get("required_env", [])
    if (
        not isinstance(args, list)
        or any(not isinstance(arg, str) for arg in args)
        or not isinstance(fields, list)
        or any(not isinstance(name, str) or not name for name in fields)
        or len(fields) != len(set(fields))
    ):
        raise ActorMCPRuntimeDefinitionError(
            "actor stdio builtin execution definition is invalid"
        )
    return execution, tuple(fields)


def _validated_runtime_credentials(
    credentials: Mapping[str, str] | None,
    *,
    required_fields: tuple[str, ...],
) -> dict[str, str]:
    if not required_fields and credentials is None:
        return {}
    if not isinstance(credentials, Mapping) or set(credentials) != set(required_fields):
        raise ActorMCPRuntimeDefinitionError("actor stdio credentials are incomplete")
    values = dict(credentials)
    if any(
        not isinstance(value, str) or not value.strip() for value in values.values()
    ):
        raise ActorMCPRuntimeDefinitionError("actor stdio credentials are incomplete")
    return values


def resolve_actor_mcp_stdio_configs(
    db: Session,
    *,
    user_id: int,
    policy: MCPActorAuthorizationPolicy | None,
    adapter: ActorMCPStdioConnectionAdapter | None,
    visible_servers: Sequence[Any],
    execution_identity: MCPActorExecutionIdentity | None = None,
) -> ActorMCPStdioResolution:
    """Build actor configs without consulting persisted server definitions.

    Reserved dispatch identities remain blocked when the rollout feature is off.
    The flag gates synthetic execution, not protection against custom-server
    shadowing of a trusted builtin identity.
    """

    if policy is None or not policy.allow_builtin_stdio:
        return ActorMCPStdioResolution((), frozenset())

    blocked_server_ids = _blocked_visible_server_ids(visible_servers)
    blocked_server_reasons = tuple(
        (server_id, ACTOR_STDIO_SHADOWED_REASON)
        for server_id in sorted(blocked_server_ids)
    )
    if (
        isinstance(user_id, bool)
        or not isinstance(user_id, int)
        or user_id <= 0
        or not xagent_config.get_toby_personal_stdio_enabled()
        or adapter is None
    ):
        return ActorMCPStdioResolution((), blocked_server_ids, blocked_server_reasons)

    try:
        identities = tuple(
            adapter.list_connection_identities(
                db,
                user_id=user_id,
                resource_owner_key=policy.resource_owner_key,
            )
        )
    except _ACTOR_MCP_STORAGE_ERRORS as exc:
        logger.info("Actor stdio connection list unavailable (%s)", type(exc).__name__)
        return ActorMCPStdioResolution((), blocked_server_ids, blocked_server_reasons)
    except Exception as exc:
        logger.warning(
            "Actor stdio connection list failed unexpectedly (%s)",
            type(exc).__name__,
        )
        return ActorMCPStdioResolution((), blocked_server_ids, blocked_server_reasons)
    configs: list[dict[str, Any]] = []
    session_identities: list[tuple[str, ActorMCPStdioSessionIdentity]] = []
    for identity in identities:
        if (
            not isinstance(identity, ActorMCPStdioConnectionIdentity)
            or identity.user_id != user_id
            or identity.resource_owner_key != policy.resource_owner_key
        ):
            continue
        try:
            execution, required_fields = _canonical_stdio_execution(db, identity)
            session_identity: ActorMCPStdioSessionIdentity | None = None
            if get_builtin_stdio_session_scope(identity.app_id) == "execution":
                if execution_identity is None:
                    continue
                session_identity = ActorMCPStdioSessionIdentity(
                    execution=execution_identity,
                    connection=identity,
                )
            app_keys = {
                _dispatch_server_key(identity.app_id),
                _dispatch_server_key(execution.get("name")),
            }
            if any(
                _dispatch_server_key(getattr(server, "name", None)) in app_keys
                for server in visible_servers
            ):
                continue
        except ActorMCPRuntimeDefinitionError as exc:
            logger.info(
                "Actor stdio definition unavailable (%s; drift_fields=%s)",
                type(exc).__name__,
                ",".join(exc.drift_fields) or "none",
            )
            continue
        except Exception as exc:
            logger.warning(
                "Actor stdio definition failed unexpectedly (%s)",
                type(exc).__name__,
            )
            continue

        try:
            credentials = adapter.get_connection_credentials(
                db,
                user_id=user_id,
                resource_owner_key=policy.resource_owner_key,
                app_id=identity.app_id,
                catalog_app_generation=identity.catalog_app_generation,
                expected_lifecycle_generation=identity.lifecycle_generation,
            )
            env = _validated_runtime_credentials(
                credentials,
                required_fields=required_fields,
            )
            fixed_runtime_env = caller_id_env(user_id)
            if set(env).intersection(fixed_runtime_env):
                raise ActorMCPRuntimeDefinitionError(
                    "actor stdio credentials overlap trusted runtime env"
                )
        except (*_ACTOR_MCP_STORAGE_ERRORS, ActorMCPRuntimeDefinitionError) as exc:
            logger.info(
                "Actor stdio credential read unavailable (%s)", type(exc).__name__
            )
            continue
        except Exception as exc:
            logger.warning(
                "Actor stdio credential read failed unexpectedly (%s)",
                type(exc).__name__,
            )
            continue

        launch = execution["launch_config"]
        trusted_env = {**env, **fixed_runtime_env}
        config: dict[str, Any] = {
            "name": identity.app_id,
            "transport": "stdio",
            "description": execution.get("name"),
            "config": {
                "command": launch["command"],
                "args": list(launch.get("args") or []),
                "env": trusted_env,
                "concurrency_safe": False,
                "concurrent_tools": [],
            },
            "user_id": str(user_id),
        }
        if session_identity is not None:
            session_identities.append((identity.app_id, session_identity))
        configs.append(config)

    return ActorMCPStdioResolution(
        tuple(configs),
        blocked_server_ids,
        blocked_server_reasons,
        tuple(session_identities),
    )
