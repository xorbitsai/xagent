"""Shared connector runtime context contracts.

This module is intentionally free of web/ORM imports so both tool adapters and
web runtime services can use the same connector identity and error shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Mapping

ConnectorType = Literal["mcp", "custom_api"]

CONNECTOR_TYPE_MCP = "mcp"
CONNECTOR_TYPE_CUSTOM_API = "custom_api"
ALLOWED_CONNECTOR_TYPES = frozenset({CONNECTOR_TYPE_MCP, CONNECTOR_TYPE_CUSTOM_API})

RUNTIME_INPUT_CONTEXT = "context"
RUNTIME_INPUT_SECRETS = "secrets"
RUNTIME_INPUT_AUTH_SELECTOR = "auth_selector"

TARGET_MCP_META = "mcp_meta"
TARGET_TRANSPORT_HEADERS = "transport_headers"
TARGET_TOOL_ARGUMENTS = "tool_arguments"
TARGET_HEADERS = "headers"
TARGET_BODY_FIELD = "body_field"

RUNTIME_SOURCE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
REDACTED_RUNTIME_SECRET = "[REDACTED_RUNTIME_SECRET]"
MISSING_RUNTIME_VALUE = object()
_SENSITIVE_RUNTIME_KEY_PARTS = frozenset(
    {
        "access_key",
        "apikey",
        "authorization",
        "auth_selector",
        "bearer",
        "api_key",
        "credential",
        "credentials",
        "private_key",
        "password",
        "secret_key",
    }
)

ERROR_CONNECTOR_NOT_FOUND = "connector_not_found"
ERROR_INVALID_RUNTIME_CONTEXT = "invalid_runtime_context"
ERROR_MISSING_RUNTIME_CONTEXT = "missing_runtime_context"
ERROR_RUNTIME_CONTEXT_IMMUTABLE = "runtime_context_immutable"
ERROR_RUNTIME_SECRET_NOT_ALLOWED = "runtime_secret_not_allowed"
ERROR_RUNTIME_SECRET_UNAVAILABLE = "runtime_secret_unavailable"
ERROR_SCHEDULED_SECRET_UNAVAILABLE = "scheduled_secret_unavailable"
ERROR_CONNECTOR_RUNTIME_UNAVAILABLE = "connector_runtime_unavailable"
ERROR_MCP_OAUTH_AUTHORIZATION_FAILED = "mcp_oauth_authorization_failed"
ERROR_DELEGATED_AUTHORIZATION_FAILED = "delegated_authorization_failed"

RUNTIME_SECRET_REASON_NOT_PROVIDED = "not_provided"
RUNTIME_SECRET_REASON_STORE_LOST = "store_lost"


@dataclass(frozen=True, order=True)
class ConnectorRef:
    """Stable runtime identity for a connector selected by a task."""

    connector_type: ConnectorType
    connector_id: int

    _WIRE_KEYS: ClassVar[frozenset[str]] = frozenset({"connector_type", "connector_id"})

    def __post_init__(self) -> None:
        if self.connector_type not in ALLOWED_CONNECTOR_TYPES:
            raise ValueError(f"unsupported connector_type: {self.connector_type!r}")
        if not isinstance(self.connector_id, int) or self.connector_id <= 0:
            raise ValueError("connector_id must be a positive integer")

    @classmethod
    def from_wire(cls, value: Any) -> "ConnectorRef":
        if not isinstance(value, dict):
            raise ValueError("connector ref must be an object")
        extra = set(value) - cls._WIRE_KEYS
        if extra:
            raise ValueError(f"connector ref has unknown field(s): {sorted(extra)!r}")
        connector_type = value.get("connector_type")
        connector_id = value.get("connector_id")
        if connector_type not in ALLOWED_CONNECTOR_TYPES:
            raise ValueError("connector_type must be 'mcp' or 'custom_api'")
        if not isinstance(connector_id, int):
            raise ValueError("connector_id must be an integer")
        return cls(connector_type=connector_type, connector_id=connector_id)

    def to_wire(self) -> dict[str, Any]:
        return {
            "connector_type": self.connector_type,
            "connector_id": self.connector_id,
        }

    @property
    def storage_key(self) -> str:
        return f"{self.connector_type}:{self.connector_id}"


class ConnectorRuntimeError(RuntimeError):
    """Public-safe runtime context error.

    ``message`` and ``details`` must be safe to return to API callers and logs
    after normal structured redaction. Raw secret/auth selector values should
    never be attached to this exception.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        connector_ref: ConnectorRef | None = None,
        details: dict[str, Any] | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.connector_ref = connector_ref
        self.details = dict(details or {})
        self.status_code = status_code
        if connector_ref is not None:
            self.details.setdefault("connector_ref", connector_ref.to_wire())

    def __str__(self) -> str:
        return f"{self.code}: {self.safe_message}"

    def to_public_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.safe_message,
            "details": dict(self.details),
        }


def validate_runtime_source_key(key: str) -> str:
    if not isinstance(key, str) or not RUNTIME_SOURCE_KEY_RE.fullmatch(key):
        raise ValueError("runtime input key must match [A-Za-z0-9_-]+")
    return key


def redact_runtime_value(value: Any) -> Any:
    """Redact runtime secrets/auth selectors without preserving raw structure."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): REDACTED_RUNTIME_SECRET for key in value}
    if isinstance(value, list):
        return [REDACTED_RUNTIME_SECRET for _ in value]
    return REDACTED_RUNTIME_SECRET


def redact_runtime_sensitive_payload(value: Any) -> Any:
    """Recursively redact runtime credential fields from public payloads."""

    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if _is_sensitive_runtime_key(key):
                redacted[key] = redact_runtime_value(item)
            else:
                redacted[key] = redact_runtime_sensitive_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_runtime_sensitive_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_runtime_sensitive_payload(item) for item in value)
    return value


def is_runtime_header_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list, tuple, set))


def _is_sensitive_runtime_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    if any(part in normalized for part in _SENSITIVE_RUNTIME_KEY_PARTS):
        return True
    return (
        normalized in {"token", "secret", "secrets"}
        or normalized.endswith("_token")
        or normalized.endswith("_secret")
    )


def runtime_bindings_from_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    bindings = config.get("runtime_bindings")
    if not isinstance(bindings, list):
        return []
    return [binding for binding in bindings if isinstance(binding, dict)]


def connector_runtime_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config.get("connector_runtime")
    return runtime if isinstance(runtime, dict) else {}


def binding_source(binding: Mapping[str, Any]) -> dict[str, Any]:
    source = binding.get("source")
    if isinstance(source, dict):
        return source
    if isinstance(source, str) and "." in source:
        input_type, key = source.split(".", 1)
        return {"input_type": input_type, "key": key}
    return {}


def binding_target(binding: Mapping[str, Any]) -> dict[str, Any]:
    target = binding.get("target")
    return target if isinstance(target, dict) else {}


def binding_source_value(
    binding: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    allowed_input_types: set[str],
) -> Any:
    source = binding_source(binding)
    input_type = source.get("input_type") or source.get("type") or source.get("section")
    key = source.get("key")
    if input_type not in allowed_input_types or not isinstance(key, str):
        return MISSING_RUNTIME_VALUE
    section = runtime.get(input_type)
    if not isinstance(section, dict) or key not in section:
        return MISSING_RUNTIME_VALUE
    return section[key]


def validate_runtime_config_declaration(
    *,
    connector_type: ConnectorType,
    runtime_input_schema: Any,
    runtime_bindings: Any,
    allow_delegated_authorization: bool,
    static_headers: Mapping[str, Any] | None = None,
) -> None:
    """Validate connector runtime declarations at config-save time."""

    if connector_type not in ALLOWED_CONNECTOR_TYPES:
        raise ValueError(f"unsupported connector_type: {connector_type!r}")

    schema = _runtime_schema_or_empty(runtime_input_schema)
    bindings = _runtime_bindings_or_empty(runtime_bindings)
    declarations = _validate_runtime_schema(connector_type, schema)
    static_header_keys = {
        str(key).lower() for key in (static_headers or {}) if isinstance(key, str)
    }

    for binding in bindings:
        source = binding_source(binding)
        target = binding_target(binding)
        input_type = source.get("input_type") or source.get("type")
        source_key = source.get("key")
        target_type = target.get("target_type") or target.get("type")

        if not isinstance(input_type, str) or not isinstance(source_key, str):
            raise ValueError("runtime binding source must include input_type and key")
        validate_runtime_source_key(source_key)
        declaration = declarations.get(input_type, {}).get(source_key)
        if declaration is None:
            raise ValueError(
                f"runtime binding source {input_type}.{source_key} is not declared"
            )
        if not isinstance(target_type, str):
            raise ValueError("runtime binding target must include target_type")

        target_key = _validate_runtime_target(connector_type, target_type, target)
        _validate_runtime_source_target(
            connector_type=connector_type,
            input_type=input_type,
            source_key=source_key,
            source_declaration=declaration,
            target_type=target_type,
            target_key=target_key,
            allow_delegated_authorization=allow_delegated_authorization,
            static_header_keys=static_header_keys,
        )


def _runtime_schema_or_empty(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("runtime_input_schema must be an object")
    return value


def _runtime_bindings_or_empty(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("runtime_bindings must be an array")
    if not all(isinstance(binding, dict) for binding in value):
        raise ValueError("runtime_bindings entries must be objects")
    return list(value)


def _validate_runtime_schema(
    connector_type: ConnectorType, schema: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    allowed_sections = {RUNTIME_INPUT_CONTEXT, RUNTIME_INPUT_SECRETS}
    if connector_type == CONNECTOR_TYPE_MCP:
        allowed_sections.add(RUNTIME_INPUT_AUTH_SELECTOR)

    declarations: dict[str, dict[str, Any]] = {}
    for section_name, section in schema.items():
        if section_name not in {
            RUNTIME_INPUT_CONTEXT,
            RUNTIME_INPUT_SECRETS,
            RUNTIME_INPUT_AUTH_SELECTOR,
        }:
            raise ValueError(f"unsupported runtime input section: {section_name}")
        if section_name not in allowed_sections:
            raise ValueError(f"{section_name} is not supported for {connector_type}")
        if not isinstance(section, dict):
            raise ValueError(f"runtime_input_schema.{section_name} must be an object")
        declarations[section_name] = {}
        for key, declaration in section.items():
            validate_runtime_source_key(key)
            if declaration is not None and not isinstance(declaration, dict):
                raise ValueError(
                    f"runtime_input_schema.{section_name}.{key} must be an object"
                )
            declarations[section_name][key] = declaration or {}
    return declarations


def _validate_runtime_target(
    connector_type: ConnectorType, target_type: str, target: Mapping[str, Any]
) -> str:
    allowed_targets = (
        {TARGET_MCP_META, TARGET_TRANSPORT_HEADERS, TARGET_TOOL_ARGUMENTS}
        if connector_type == CONNECTOR_TYPE_MCP
        else {TARGET_HEADERS, TARGET_BODY_FIELD}
    )
    if target_type not in allowed_targets:
        raise ValueError(f"{target_type} is not supported for {connector_type}")

    if target_type in {
        TARGET_MCP_META,
        TARGET_TRANSPORT_HEADERS,
        TARGET_TOOL_ARGUMENTS,
        TARGET_HEADERS,
    }:
        key = target.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{target_type} target requires key")
        return key

    path = target.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{target_type} target requires path")
    _validate_simple_body_path(path)
    return path


def _validate_simple_body_path(path: str) -> None:
    if any(part == "" for part in path.split(".")):
        raise ValueError("body_field path must be a simple dot path")
    for part in path.split("."):
        validate_runtime_source_key(part)


def _validate_runtime_source_target(
    *,
    connector_type: ConnectorType,
    input_type: str,
    source_key: str,
    source_declaration: Mapping[str, Any],
    target_type: str,
    target_key: str,
    allow_delegated_authorization: bool,
    static_header_keys: set[str],
) -> None:
    if input_type == RUNTIME_INPUT_AUTH_SELECTOR:
        raise ValueError("auth_selector cannot be bound to connector targets")

    if input_type == RUNTIME_INPUT_SECRETS:
        allowed_secret_targets = {
            CONNECTOR_TYPE_MCP: {TARGET_TRANSPORT_HEADERS},
            CONNECTOR_TYPE_CUSTOM_API: {TARGET_HEADERS},
        }[connector_type]
        if target_type not in allowed_secret_targets:
            raise ValueError("secrets can only bind to connector headers")
    elif input_type == RUNTIME_INPUT_CONTEXT:
        allowed_context_targets = {
            CONNECTOR_TYPE_MCP: {TARGET_MCP_META, TARGET_TOOL_ARGUMENTS},
            CONNECTOR_TYPE_CUSTOM_API: {TARGET_HEADERS, TARGET_BODY_FIELD},
        }[connector_type]
        if target_type not in allowed_context_targets:
            raise ValueError(f"context cannot bind to {target_type}")
    else:
        raise ValueError(f"unsupported runtime binding source section: {input_type}")

    if target_type in {TARGET_HEADERS, TARGET_TRANSPORT_HEADERS}:
        if str(source_declaration.get("type") or "string") == "object":
            raise ValueError("object runtime values cannot bind to headers")
        if target_key.lower() == "authorization" and not allow_delegated_authorization:
            raise ValueError(
                "Authorization runtime binding requires delegated authorization"
            )
        if target_key.lower() in static_header_keys:
            raise ValueError(
                f"runtime binding conflicts with static header {target_key!r}"
            )
