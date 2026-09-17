from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...core.utils.encryption import (
    EncryptionDecodeError,
    _is_encrypted,
    decrypt_env_dict_strict,
    get_cipher,
)
from ..builtin_mcp_registry import get_builtin_execution_fields
from ..models.actor_mcp_connection import (
    ACTOR_MCP_RESOURCE_OWNER_KEY_MAX_LENGTH,
    ActorMCPServerConnection,
)
from ..models.public_mcp import PublicMCPApp
from ..repositories.actor_mcp_connections import (
    actor_mcp_connection_identity_exists_for_create,
    add_actor_mcp_connection,
    get_actor_mcp_connection,
    get_actor_mcp_connection_metadata_row,
    get_current_actor_mcp_connection_for_create,
    hard_delete_actor_mcp_connection,
    list_actor_mcp_connections,
    update_actor_mcp_connection_env,
)

ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH = 4096


class ActorMCPConnectionValidationError(ValueError):
    """The actor connection request violates the internal storage contract."""


class ActorMCPConnectionConflictError(RuntimeError):
    """Creation conflicts with an existing connection's credentials."""


class ActorMCPConnectionCredentialCorruptionError(RuntimeError):
    """Stored credentials cannot be safely encoded or decoded."""


class ActorMCPConnectionNotFoundOrStaleError(LookupError):
    """The exact actor connection lifecycle is absent or catalog-stale."""


@dataclass(frozen=True)
class ActorMCPConnectionSnapshot:
    id: int
    lifecycle_generation: UUID
    user_id: int
    resource_owner_key: str
    app_id: str
    catalog_app_generation: UUID
    credentials: dict[str, str] | None = dataclass_field(repr=False)


@dataclass(frozen=True)
class ActorMCPConnectionMetadata:
    id: int
    lifecycle_generation: UUID
    user_id: int
    resource_owner_key: str
    app_id: str
    catalog_app_generation: UUID
    configured_field_names: frozenset[str]


def _positive_id(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActorMCPConnectionValidationError(
            f"{field_name} must be a persisted positive integer"
        )
    return int(value)


def _owner_key(value: Any) -> str:
    if not isinstance(value, str):
        raise ActorMCPConnectionValidationError("resource_owner_key must be a string")
    if not value or value != value.strip():
        raise ActorMCPConnectionValidationError(
            "resource_owner_key must be an exact non-blank string"
        )
    if len(value) > ACTOR_MCP_RESOURCE_OWNER_KEY_MAX_LENGTH:
        raise ActorMCPConnectionValidationError(
            "resource_owner_key exceeds "
            f"{ACTOR_MCP_RESOURCE_OWNER_KEY_MAX_LENGTH} characters"
        )
    return value


def _exact_app_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ActorMCPConnectionValidationError(
            "app_id must be an exact non-blank canonical app id"
        )
    if len(value) > 100:
        raise ActorMCPConnectionValidationError("app_id exceeds 100 characters")
    return value


def _generation(value: Any, *, field_name: str) -> UUID:
    if not isinstance(value, UUID):
        raise ActorMCPConnectionValidationError(f"{field_name} must be a UUID")
    return value


def _builtin_stdio_definition(
    db: Session, *, app_id: str
) -> tuple[PublicMCPApp, frozenset[str]]:
    app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == app_id).one_or_none()
    execution = get_builtin_execution_fields(app_id)
    if app is None or execution is None or execution.get("transport") != "stdio":
        raise ActorMCPConnectionValidationError(
            "app_id must identify a code-defined built-in stdio app"
        )
    launch = execution.get("launch_config")
    if not isinstance(launch, dict) or not launch.get("command"):
        raise ActorMCPConnectionValidationError(
            "built-in stdio app has an invalid execution definition"
        )
    raw_fields = launch.get("required_env")
    if raw_fields is None:
        fields: list[Any] = []
    elif isinstance(raw_fields, list):
        fields = raw_fields
    else:
        raise ActorMCPConnectionValidationError(
            "canonical stdio app has an invalid credential schema"
        )
    if any(
        not isinstance(field, str) or not field or field != field.strip()
        for field in fields
    ) or len(fields) != len(set(fields)):
        raise ActorMCPConnectionValidationError(
            "canonical stdio app has an invalid credential schema"
        )
    return app, frozenset(fields)


def _validated_credentials(
    credentials: Mapping[str, Any] | None,
    *,
    allowed_fields: frozenset[str],
    require_complete: bool,
) -> dict[str, str] | None:
    if credentials is None:
        supplied: Mapping[str, Any] = {}
    elif not isinstance(credentials, Mapping):
        raise ActorMCPConnectionValidationError("credentials must be an object")
    else:
        supplied = credentials

    unknown = set(supplied) - allowed_fields
    if unknown:
        raise ActorMCPConnectionValidationError(
            "credentials contain fields outside the canonical app schema"
        )
    if require_complete and set(supplied) != allowed_fields:
        raise ActorMCPConnectionValidationError(
            "connection creation requires every canonical credential field"
        )
    validated: dict[str, str] = {}
    for field, value in supplied.items():
        if not isinstance(field, str):
            raise ActorMCPConnectionValidationError(
                "credential field names must be strings"
            )
        if value is None:
            raise ActorMCPConnectionValidationError(
                "credential values must not be null"
            )
        if not isinstance(value, str):
            raise ActorMCPConnectionValidationError("credential values must be strings")
        if not value.strip():
            raise ActorMCPConnectionValidationError(
                "credential values must not be blank"
            )
        if len(value) > ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH:
            raise ActorMCPConnectionValidationError(
                "credential value exceeds "
                f"{ACTOR_MCP_CREDENTIAL_VALUE_MAX_LENGTH} characters"
            )
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ActorMCPConnectionValidationError(
                "credential values must be valid UTF-8 text"
            ) from None
        validated[field] = value
    return validated or None


def _credentials_from_row(
    row: ActorMCPServerConnection,
) -> dict[str, str] | None:
    try:
        encrypted = row.encrypted_env
        if encrypted is not None and (
            not isinstance(encrypted, dict)
            or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or not _is_encrypted(value)
                for key, value in encrypted.items()
            )
        ):
            raise ActorMCPConnectionCredentialCorruptionError(
                "stored actor MCP credentials are unreadable"
            )
        decrypted = decrypt_env_dict_strict(row.encrypted_env)
        if decrypted is not None and (
            not isinstance(decrypted, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in decrypted.items()
            )
        ):
            raise ActorMCPConnectionCredentialCorruptionError(
                "stored actor MCP credentials are unreadable"
            )
    except (EncryptionDecodeError, UnicodeError, ValueError, TypeError):
        raise ActorMCPConnectionCredentialCorruptionError(
            "stored actor MCP credentials are unreadable"
        ) from None
    return dict(decrypted) if decrypted is not None else None


def _encrypt_credentials(credentials: dict[str, str] | None) -> dict[str, str] | None:
    if credentials is None:
        return None
    try:
        cipher = get_cipher()
        encrypted = {
            field: cipher.encrypt(value.encode("utf-8")).decode("ascii")
            for field, value in credentials.items()
        }
    except (UnicodeError, ValueError, TypeError):
        raise ActorMCPConnectionCredentialCorruptionError(
            "actor MCP credentials could not be encrypted"
        ) from None
    return encrypted


def _snapshot(row: ActorMCPServerConnection) -> ActorMCPConnectionSnapshot:
    return ActorMCPConnectionSnapshot(
        id=row.id,
        lifecycle_generation=row.lifecycle_generation,
        user_id=row.user_id,
        resource_owner_key=row.resource_owner_key,
        app_id=row.app_id,
        catalog_app_generation=row.catalog_app_generation,
        credentials=_credentials_from_row(row),
    )


def _metadata(row: ActorMCPServerConnection) -> ActorMCPConnectionMetadata:
    encrypted = row.encrypted_env
    if encrypted is None:
        configured_field_names: frozenset[str] = frozenset()
    elif isinstance(encrypted, dict) and all(
        isinstance(field, str) for field in encrypted
    ):
        configured_field_names = frozenset(encrypted)
    else:
        raise ActorMCPConnectionCredentialCorruptionError(
            "stored actor MCP credential metadata is unreadable"
        )
    return ActorMCPConnectionMetadata(
        id=row.id,
        lifecycle_generation=row.lifecycle_generation,
        user_id=row.user_id,
        resource_owner_key=row.resource_owner_key,
        app_id=row.app_id,
        catalog_app_generation=row.catalog_app_generation,
        configured_field_names=configured_field_names,
    )


def create_actor_mcp_connection(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    credentials: Mapping[str, Any] | None,
) -> ActorMCPConnectionMetadata:
    user_id = _positive_id(user_id, field_name="user_id")
    owner_key = _owner_key(resource_owner_key)
    app_id = _exact_app_id(app_id)
    app, allowed_fields = _builtin_stdio_definition(db, app_id=app_id)
    validated = _validated_credentials(
        credentials, allowed_fields=allowed_fields, require_complete=True
    )
    existing = get_current_actor_mcp_connection_for_create(
        db,
        user_id=user_id,
        resource_owner_key=owner_key,
        app_id=app_id,
        catalog_app_generation=app.generation,
    )
    if existing is not None:
        if _credentials_from_row(existing) == validated:
            return _metadata(existing)
        raise ActorMCPConnectionConflictError(
            "actor MCP connection already exists; use generation-protected update"
        )

    encrypted = _encrypt_credentials(validated)
    try:
        with db.begin_nested():
            row = add_actor_mcp_connection(
                db,
                ActorMCPServerConnection(
                    user_id=user_id,
                    resource_owner_key=owner_key,
                    app_id=app_id,
                    catalog_app_generation=app.generation,
                    encrypted_env=encrypted,
                ),
            )
    except IntegrityError:
        winner = get_current_actor_mcp_connection_for_create(
            db,
            user_id=user_id,
            resource_owner_key=owner_key,
            app_id=app_id,
            catalog_app_generation=app.generation,
        )
        if winner is not None and _credentials_from_row(winner) == validated:
            return _metadata(winner)
        if winner is not None or actor_mcp_connection_identity_exists_for_create(
            db,
            user_id=user_id,
            resource_owner_key=owner_key,
            app_id=app_id,
        ):
            raise ActorMCPConnectionConflictError(
                "actor MCP connection already exists; use generation-protected update"
            ) from None
        raise
    return _metadata(row)


def get_actor_mcp_connection_metadata(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
) -> ActorMCPConnectionMetadata:
    app_id = _exact_app_id(app_id)
    row = get_actor_mcp_connection_metadata_row(
        db,
        user_id=_positive_id(user_id, field_name="user_id"),
        resource_owner_key=_owner_key(resource_owner_key),
        app_id=app_id,
    )
    if row is None:
        raise ActorMCPConnectionNotFoundOrStaleError(
            "actor MCP connection was not found or is stale"
        )
    return _metadata(row)


def list_actor_mcp_connection_metadata(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
) -> list[ActorMCPConnectionMetadata]:
    rows = list_actor_mcp_connections(
        db,
        user_id=_positive_id(user_id, field_name="user_id"),
        resource_owner_key=_owner_key(resource_owner_key),
    )
    return [_metadata(row) for row in rows]


def get_actor_mcp_connection_credentials_internal(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    catalog_app_generation: UUID,
    expected_lifecycle_generation: UUID,
) -> ActorMCPConnectionSnapshot:
    row = get_actor_mcp_connection(
        db,
        user_id=_positive_id(user_id, field_name="user_id"),
        resource_owner_key=_owner_key(resource_owner_key),
        app_id=_exact_app_id(app_id),
        catalog_app_generation=_generation(
            catalog_app_generation, field_name="catalog_app_generation"
        ),
        expected_lifecycle_generation=_generation(
            expected_lifecycle_generation,
            field_name="expected_lifecycle_generation",
        ),
    )
    if row is None:
        raise ActorMCPConnectionNotFoundOrStaleError(
            "actor MCP connection was not found or is stale"
        )
    return _snapshot(row)


def update_actor_mcp_connection_credentials(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    expected_lifecycle_generation: UUID,
    credentials: Mapping[str, Any],
) -> ActorMCPConnectionMetadata:
    user_id = _positive_id(user_id, field_name="user_id")
    owner_key = _owner_key(resource_owner_key)
    app_id = _exact_app_id(app_id)
    expected_generation = _generation(
        expected_lifecycle_generation,
        field_name="expected_lifecycle_generation",
    )
    row = get_actor_mcp_connection(
        db,
        user_id=user_id,
        resource_owner_key=owner_key,
        app_id=app_id,
        expected_lifecycle_generation=expected_generation,
        for_update=True,
    )
    if row is None:
        raise ActorMCPConnectionNotFoundOrStaleError(
            "actor MCP connection was not found or is stale"
        )
    app, allowed_fields = _builtin_stdio_definition(db, app_id=app_id)
    if app.generation != row.catalog_app_generation:
        raise ActorMCPConnectionValidationError(
            "actor connection belongs to a stale catalog app lifecycle"
        )
    updates = _validated_credentials(
        credentials, allowed_fields=allowed_fields, require_complete=False
    )
    existing = _credentials_from_row(row) or {}
    if updates:
        existing.update(updates)
    encrypted = _encrypt_credentials(existing) or None
    if not update_actor_mcp_connection_env(
        db,
        user_id=user_id,
        resource_owner_key=owner_key,
        app_id=app_id,
        expected_lifecycle_generation=expected_generation,
        encrypted_env=encrypted,
    ):
        raise ActorMCPConnectionNotFoundOrStaleError(
            "actor MCP connection was not found or is stale"
        )
    db.expire(row)
    updated = get_actor_mcp_connection(
        db,
        user_id=user_id,
        resource_owner_key=owner_key,
        app_id=app_id,
        expected_lifecycle_generation=expected_generation,
    )
    if updated is None:
        raise ActorMCPConnectionNotFoundOrStaleError(
            "actor MCP connection was not found or is stale"
        )
    return _metadata(updated)


def delete_actor_mcp_connection(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    expected_lifecycle_generation: UUID,
) -> None:
    deleted = hard_delete_actor_mcp_connection(
        db,
        user_id=_positive_id(user_id, field_name="user_id"),
        resource_owner_key=_owner_key(resource_owner_key),
        app_id=_exact_app_id(app_id),
        expected_lifecycle_generation=_generation(
            expected_lifecycle_generation,
            field_name="expected_lifecycle_generation",
        ),
    )
    if not deleted:
        raise ActorMCPConnectionNotFoundOrStaleError(
            "actor MCP connection was not found or is stale"
        )
