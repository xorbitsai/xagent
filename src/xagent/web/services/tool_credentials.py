from __future__ import annotations

import base64
import hashlib
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from ..init_tool_configs import get_default_tool_configs
from ..models.tool_config import ScopedToolCredential, ToolConfig

ScopeType = str
ApiScopeType = Literal["user", "instance"]
ToolFieldSpec = dict[str, Any]

# Hook signature: (db: Session, user: Any) -> dict[str, {"enabled": bool|None}]
# Returns per-user overrides for tool enable/disable.
_get_user_tool_overrides_hook: Callable[[Session, Any], dict] | None = None


@dataclass(frozen=True)
class CredentialScopeRef:
    """A non-core credential fallback scope supplied by an embedding app."""

    scope_type: str
    scope_id: int
    source_label: str | None = None


# Hook signature: (db: Session, user: Any) -> Iterable[CredentialScopeRef].
# Embedding apps install this so core xagent can resolve shared credentials
# without importing application-specific tenancy concepts.
_credential_fallback_scopes_hook: (
    Callable[[Session, Any], Iterable[CredentialScopeRef]] | None
) = None


def set_user_tool_overrides_hook(hook: Callable[[Session, Any], dict] | None) -> None:
    global _get_user_tool_overrides_hook
    _get_user_tool_overrides_hook = hook


def get_user_tool_overrides(db: Session, user: Any) -> dict:
    if _get_user_tool_overrides_hook is not None:
        return _get_user_tool_overrides_hook(db, user)
    return {}


def set_credential_fallback_scopes_hook(
    hook: Callable[[Session, Any], Iterable[CredentialScopeRef]] | None,
) -> None:
    global _credential_fallback_scopes_hook
    _credential_fallback_scopes_hook = hook


def get_credential_fallback_scopes(db: Session, user: Any) -> list[CredentialScopeRef]:
    if _credential_fallback_scopes_hook is None:
        return []
    scopes: list[CredentialScopeRef] = []
    for scope in _credential_fallback_scopes_hook(db, user):
        if scope.scope_id is None:
            continue
        scopes.append(
            CredentialScopeRef(
                scope_type=str(scope.scope_type),
                scope_id=int(scope.scope_id),
                source_label=scope.source_label,
            )
        )
    return scopes


def set_instance_credentials_enabled(enabled: bool) -> None:
    """Deprecated compatibility shim.

    Instance credentials are a core standalone scope. SaaS should control
    runtime fallback order by installing credential fallback scopes instead.
    """


def instance_credentials_enabled() -> bool:
    return True


TOOL_CREDENTIAL_SPECS: dict[str, dict[str, ToolFieldSpec]] = {
    "exa_web_search": {
        "api_key": {
            "secret": True,
            "env": ["EXA_API_KEY"],
            "required": True,
            "label": "API Key",
        }
    },
    "zhipu_web_search": {
        "api_key": {
            "secret": True,
            "env": ["ZHIPU_API_KEY", "BIGMODEL_API_KEY"],
            "required": True,
            "label": "API Key",
        },
        "base_url": {
            "secret": False,
            "env": ["ZHIPU_BASE_URL"],
            "required": False,
            "label": "Base URL",
        },
    },
    "tavily_web_search": {
        "api_key": {
            "secret": True,
            "env": ["TAVILY_API_KEY"],
            "required": True,
            "label": "API Key",
        }
    },
    "web_search": {
        "api_key": {
            "secret": True,
            "env": ["GOOGLE_API_KEY"],
            "required": True,
            "label": "Google API Key",
        },
        "cse_id": {
            "secret": False,
            "env": ["GOOGLE_CSE_ID"],
            "required": True,
            "label": "Google CSE ID",
        },
    },
}

SQL_TOOL_NAME = "sql_query"
SQL_CONNECTION_ENV_PREFIX = "XAGENT_EXTERNAL_DB_"
ALLOWED_SQL_SCHEMES = {"duckdb", "postgresql", "mysql", "mariadb", "mssql", "sqlite"}


def list_configurable_tool_names() -> list[str]:
    return [*TOOL_CREDENTIAL_SPECS.keys(), SQL_TOOL_NAME]


def _build_fernet_key() -> bytes:
    raw = (
        os.getenv("XAGENT_SECRET_ENCRYPTION_KEY")
        or os.getenv("SECRET_KEY")
        or "xagent-dev-key"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _fernet() -> Fernet:
    return Fernet(_build_fernet_key())


def _encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def _decrypt(ciphertext: str) -> str | None:
    try:
        return _fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError):
        return None


def _mask_value(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{'*' * (len(value) - 4)}{value[-4:]}"


def _read_env(env_names: Iterable[str]) -> str | None:
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _sanitize_sql_connection_name(name: str) -> str:
    return name.strip().upper()


def _sql_url_mask(url: str) -> str:
    try:
        parsed = make_url(url)
        return parsed.render_as_string(hide_password=True)
    except Exception:
        return _mask_value(url)


def _validate_scope(scope_type: str, scope_id: int | None) -> ScopeType:
    if not scope_type:
        raise ValueError("Credential scope is required")
    if scope_type == "instance" and scope_id is not None:
        raise ValueError("Instance credential scope must not have a scope id")
    if scope_type != "instance" and scope_id is None:
        raise ValueError(f"{scope_type.title()} credential scope requires a scope id")
    return scope_type


def _runtime_scope_entries(
    *,
    user_id: int | None,
    user: Any | None,
    db: Session,
    include_instance: bool,
) -> tuple[tuple[ScopeType, int | None], ...]:
    entries: list[tuple[ScopeType, int | None]] = []
    if user_id is not None:
        entries.append(("user", user_id))
    shared_scopes: list[CredentialScopeRef] = []
    if user is not None:
        shared_scopes = get_credential_fallback_scopes(db, user)
        entries.extend((scope.scope_type, scope.scope_id) for scope in shared_scopes)
    if include_instance:
        entries.append(("instance", None))
    return tuple(entries)


def _query_scoped_credential(
    db: Session,
    *,
    scope_type: ScopeType,
    scope_id: int | None,
    tool_name: str,
    field_name: str,
) -> ScopedToolCredential | None:
    query = db.query(ScopedToolCredential).filter(
        ScopedToolCredential.scope_type == scope_type,
        ScopedToolCredential.tool_name == tool_name,
        ScopedToolCredential.field_name == field_name,
    )
    if scope_id is None:
        query = query.filter(ScopedToolCredential.scope_id.is_(None))
    else:
        query = query.filter(ScopedToolCredential.scope_id == scope_id)
    return query.first()


def _row_field_name(row: ScopedToolCredential) -> str:
    return str(row.field_name)


def _get_display_name(db: Session, tool_name: str) -> str:
    if tool_name == SQL_TOOL_NAME:
        return "SQL Query"
    config = db.query(ToolConfig).filter(ToolConfig.tool_name == tool_name).first()
    if config is not None and isinstance(getattr(config, "display_name", None), str):
        return str(config.display_name)
    defaults = {item["tool_name"]: item for item in get_default_tool_configs()}
    return str(defaults.get(tool_name, {}).get("display_name") or tool_name)


def _credential_specs_for_tool(tool_name: str) -> dict[str, ToolFieldSpec]:
    specs = TOOL_CREDENTIAL_SPECS.get(tool_name)
    if specs is None:
        raise ValueError(f"Tool '{tool_name}' is not configurable")
    return specs


def _resolve_plain_field(
    db: Session,
    *,
    tool_name: str,
    field_name: str,
    user_id: int | None,
    user: Any | None,
    include_instance: bool,
) -> tuple[str | None, str]:
    scope_entries = _runtime_scope_entries(
        user_id=user_id,
        user=user,
        db=db,
        include_instance=include_instance,
    )
    for scope_type, scope_id in scope_entries:
        if scope_type == "instance":
            if not include_instance:
                continue
            credential_scope_id = None
        else:
            if scope_id is None:
                continue
            credential_scope_id = scope_id
        row = _query_scoped_credential(
            db,
            scope_type=scope_type,
            scope_id=credential_scope_id,
            tool_name=tool_name,
            field_name=field_name,
        )
        if row is None:
            continue
        decrypted = _decrypt(str(row.encrypted_value))
        if decrypted:
            return decrypted, scope_type

    spec = TOOL_CREDENTIAL_SPECS.get(tool_name, {}).get(field_name)
    env_names = spec.get("env", []) if spec else []
    if isinstance(env_names, list):
        env_value = _read_env(env_names)
        if env_value:
            return env_value, "env"
    return None, "none"


def resolve_tool_credential(
    db: Session,
    tool_name: str,
    field_name: str,
    *,
    user_id: int | None = None,
    user: Any | None = None,
    include_instance: bool | None = None,
) -> str | None:
    if tool_name == SQL_TOOL_NAME:
        return resolve_sql_connection(
            db,
            field_name,
            user_id=user_id,
            user=user,
            include_instance=include_instance,
        )
    if field_name not in TOOL_CREDENTIAL_SPECS.get(tool_name, {}):
        return None
    resolved_include_instance = (
        instance_credentials_enabled() if include_instance is None else include_instance
    )
    value, _source = _resolve_plain_field(
        db,
        tool_name=tool_name,
        field_name=field_name,
        user_id=user_id,
        user=user,
        include_instance=resolved_include_instance,
    )
    return value


def set_scoped_tool_credentials(
    db: Session,
    *,
    scope_type: str,
    scope_id: int | None,
    tool_name: str,
    values: dict[str, str],
) -> None:
    normalized_scope = _validate_scope(scope_type, scope_id)
    if tool_name != SQL_TOOL_NAME:
        specs = _credential_specs_for_tool(tool_name)
        allowed_fields = set(specs)
    else:
        allowed_fields = set(values)

    now = datetime.now(UTC)
    for raw_field_name, raw_value in values.items():
        field_name = (
            _sanitize_sql_connection_name(raw_field_name)
            if tool_name == SQL_TOOL_NAME
            else raw_field_name
        )
        if field_name not in allowed_fields and tool_name != SQL_TOOL_NAME:
            continue
        normalized = raw_value.strip()
        if not normalized:
            continue
        if tool_name == SQL_TOOL_NAME:
            _validate_sql_connection_url(normalized)
            masked = _sql_url_mask(normalized)
        else:
            masked = _mask_value(normalized)

        row = _query_scoped_credential(
            db,
            scope_type=normalized_scope,
            scope_id=scope_id,
            tool_name=tool_name,
            field_name=field_name,
        )
        if row is None:
            row = ScopedToolCredential(
                scope_type=normalized_scope,
                scope_id=scope_id,
                tool_name=tool_name,
                field_name=field_name,
                encrypted_value=_encrypt(normalized),
                masked_value=masked,
            )
            db.add(row)
        else:
            row.encrypted_value = _encrypt(normalized)  # type: ignore[assignment]
            row.masked_value = masked  # type: ignore[assignment]
            row.updated_at = now  # type: ignore[assignment]
            db.add(row)
    db.commit()


def clear_scoped_tool_credential(
    db: Session,
    *,
    scope_type: str,
    scope_id: int | None,
    tool_name: str,
    field_name: str,
) -> None:
    normalized_scope = _validate_scope(scope_type, scope_id)
    normalized_field = (
        _sanitize_sql_connection_name(field_name)
        if tool_name == SQL_TOOL_NAME
        else field_name
    )
    row = _query_scoped_credential(
        db,
        scope_type=normalized_scope,
        scope_id=scope_id,
        tool_name=tool_name,
        field_name=normalized_field,
    )
    if row is not None:
        db.delete(row)
        db.commit()


def _validate_sql_connection_url(connection_url: str) -> None:
    try:
        parsed = make_url(connection_url)
    except Exception as exc:
        raise ValueError("Invalid SQLAlchemy connection URL") from exc

    base_scheme = str(parsed.drivername).split("+", 1)[0].lower()
    if base_scheme not in ALLOWED_SQL_SCHEMES:
        allowed_schemes_text = ", ".join(sorted(ALLOWED_SQL_SCHEMES))
        raise ValueError(
            f"Unsupported SQLAlchemy URL scheme '{parsed.drivername}'. "
            f"Allowed schemes: {allowed_schemes_text}"
        )


def _rows_for_scope(
    db: Session,
    *,
    scope_type: ScopeType,
    scope_id: int | None,
    tool_name: str,
) -> list[ScopedToolCredential]:
    query = db.query(ScopedToolCredential).filter(
        ScopedToolCredential.scope_type == scope_type,
        ScopedToolCredential.tool_name == tool_name,
    )
    if scope_id is None:
        query = query.filter(ScopedToolCredential.scope_id.is_(None))
    else:
        query = query.filter(ScopedToolCredential.scope_id == scope_id)
    return list(query.all())


def get_tool_credential_view(
    db: Session,
    tool_name: str,
    *,
    scope_type: str,
    scope_id: int | None,
    user_id: int | None = None,
    user: Any | None = None,
    include_instance: bool | None = None,
) -> dict[str, Any]:
    normalized_scope = _validate_scope(scope_type, scope_id)
    if tool_name == SQL_TOOL_NAME:
        return get_sql_credential_view(
            db,
            scope_type=normalized_scope,
            scope_id=scope_id,
            user_id=user_id,
            user=user,
            include_instance=include_instance,
        )

    specs = _credential_specs_for_tool(tool_name)
    scoped_rows = {
        _row_field_name(row): row
        for row in _rows_for_scope(
            db,
            scope_type=normalized_scope,
            scope_id=scope_id,
            tool_name=tool_name,
        )
    }
    resolved_include_instance = (
        instance_credentials_enabled() if include_instance is None else include_instance
    )

    fields: dict[str, Any] = {}
    all_required_ok = True
    for field_name, spec in specs.items():
        scoped_row = scoped_rows.get(field_name)
        env_value = (
            _read_env(spec.get("env", []))
            if isinstance(spec.get("env"), list)
            else None
        )
        resolved, source = _resolve_plain_field(
            db,
            tool_name=tool_name,
            field_name=field_name,
            user_id=user_id,
            user=user,
            include_instance=resolved_include_instance,
        )
        if scoped_row is not None:
            source = normalized_scope
        required = bool(spec.get("required", False))
        is_configured = bool(resolved)
        if required and not is_configured:
            all_required_ok = False

        fields[field_name] = {
            "label": spec.get("label", field_name),
            "required": required,
            "secret": bool(spec.get("secret", False)),
            "source": source,
            "is_configured": is_configured,
            "masked": str(scoped_row.masked_value)
            if scoped_row is not None
            else (_mask_value(env_value) if env_value else ""),
            "env_names": spec.get("env", []),
        }

    return {
        "tool_name": tool_name,
        "display_name": _get_display_name(db, tool_name),
        "configured": all_required_ok,
        "fields": fields,
    }


def list_tool_credential_views(
    db: Session,
    *,
    scope_type: str,
    scope_id: int | None,
    user_id: int | None = None,
    user: Any | None = None,
    include_instance: bool | None = None,
) -> list[dict[str, Any]]:
    return [
        get_tool_credential_view(
            db,
            tool_name,
            scope_type=scope_type,
            scope_id=scope_id,
            user_id=user_id,
            user=user,
            include_instance=include_instance,
        )
        for tool_name in list_configurable_tool_names()
    ]


def resolve_sql_connection(
    db: Session,
    name: str,
    *,
    user_id: int | None = None,
    user: Any | None = None,
    include_instance: bool | None = None,
) -> str | None:
    normalized_name = _sanitize_sql_connection_name(name)
    resolved_include_instance = (
        instance_credentials_enabled() if include_instance is None else include_instance
    )
    value, _source = _resolve_sql_field(
        db,
        normalized_name,
        user_id=user_id,
        user=user,
        include_instance=resolved_include_instance,
    )
    return value


def _resolve_sql_field(
    db: Session,
    normalized_name: str,
    *,
    user_id: int | None,
    user: Any | None,
    include_instance: bool,
) -> tuple[str | None, str]:
    scope_entries = _runtime_scope_entries(
        user_id=user_id,
        user=user,
        db=db,
        include_instance=include_instance,
    )
    for scope_type, scope_id in scope_entries:
        if scope_type == "instance":
            if not include_instance:
                continue
            credential_scope_id = None
        else:
            if scope_id is None:
                continue
            credential_scope_id = scope_id
        row = _query_scoped_credential(
            db,
            scope_type=scope_type,
            scope_id=credential_scope_id,
            tool_name=SQL_TOOL_NAME,
            field_name=normalized_name,
        )
        if row is None:
            continue
        decrypted = _decrypt(str(row.encrypted_value))
        if decrypted:
            return decrypted, scope_type

    env_value = os.getenv(f"{SQL_CONNECTION_ENV_PREFIX}{normalized_name}")
    if env_value:
        return env_value, "env"
    return None, "none"


def get_sql_connection_map(
    db: Session,
    user_id: int | None,
    *,
    user: Any | None = None,
    include_instance: bool | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    names = {
        key[len(SQL_CONNECTION_ENV_PREFIX) :]
        for key, value in os.environ.items()
        if key.startswith(SQL_CONNECTION_ENV_PREFIX) and value
    }
    if user_id is not None:
        names.update(
            _sanitize_sql_connection_name(_row_field_name(row))
            for row in _rows_for_scope(
                db,
                scope_type="user",
                scope_id=user_id,
                tool_name=SQL_TOOL_NAME,
            )
        )
    shared_scopes = get_credential_fallback_scopes(db, user) if user is not None else []
    for scope in shared_scopes:
        names.update(
            _sanitize_sql_connection_name(_row_field_name(row))
            for row in _rows_for_scope(
                db,
                scope_type=scope.scope_type,
                scope_id=scope.scope_id,
                tool_name=SQL_TOOL_NAME,
            )
        )
    resolved_include_instance = (
        instance_credentials_enabled() if include_instance is None else include_instance
    )
    if resolved_include_instance:
        names.update(
            _sanitize_sql_connection_name(_row_field_name(row))
            for row in _rows_for_scope(
                db,
                scope_type="instance",
                scope_id=None,
                tool_name=SQL_TOOL_NAME,
            )
        )

    for name in sorted(names):
        value = resolve_sql_connection(
            db,
            name,
            user_id=user_id,
            user=user,
            include_instance=resolved_include_instance,
        )
        if value:
            result[name] = value
    return result


def get_sql_credential_view(
    db: Session,
    *,
    scope_type: ScopeType,
    scope_id: int | None,
    user_id: int | None,
    user: Any | None,
    include_instance: bool | None,
) -> dict[str, Any]:
    scoped_rows = {
        _sanitize_sql_connection_name(_row_field_name(row)): row
        for row in _rows_for_scope(
            db,
            scope_type=scope_type,
            scope_id=scope_id,
            tool_name=SQL_TOOL_NAME,
        )
    }
    names = set(scoped_rows)
    env_values = {
        key[len(SQL_CONNECTION_ENV_PREFIX) :]: value
        for key, value in os.environ.items()
        if key.startswith(SQL_CONNECTION_ENV_PREFIX) and value
    }
    names.update(env_values)
    resolved_include_instance = (
        instance_credentials_enabled() if include_instance is None else include_instance
    )
    if user_id is not None:
        names.update(
            _sanitize_sql_connection_name(_row_field_name(row))
            for row in _rows_for_scope(
                db,
                scope_type="user",
                scope_id=user_id,
                tool_name=SQL_TOOL_NAME,
            )
        )
    shared_scopes = get_credential_fallback_scopes(db, user) if user is not None else []
    for scope in shared_scopes:
        names.update(
            _sanitize_sql_connection_name(_row_field_name(row))
            for row in _rows_for_scope(
                db,
                scope_type=scope.scope_type,
                scope_id=scope.scope_id,
                tool_name=SQL_TOOL_NAME,
            )
        )
    if resolved_include_instance:
        names.update(
            _sanitize_sql_connection_name(_row_field_name(row))
            for row in _rows_for_scope(
                db,
                scope_type="instance",
                scope_id=None,
                tool_name=SQL_TOOL_NAME,
            )
        )
    fields: dict[str, Any] = {}
    for name in sorted(names):
        scoped_row = scoped_rows.get(name)
        resolved, source = _resolve_sql_field(
            db,
            name,
            user_id=user_id,
            user=user,
            include_instance=resolved_include_instance,
        )
        if scoped_row is not None:
            source = scope_type
        env_value = env_values.get(name)
        fields[name] = {
            "label": name,
            "required": False,
            "secret": True,
            "source": source,
            "is_configured": bool(resolved),
            "masked": str(scoped_row.masked_value)
            if scoped_row is not None
            else (_sql_url_mask(env_value) if env_value else ""),
            "env_names": [f"{SQL_CONNECTION_ENV_PREFIX}{name}"],
        }

    return {
        "tool_name": SQL_TOOL_NAME,
        "display_name": _get_display_name(db, SQL_TOOL_NAME),
        "configured": bool(
            get_sql_connection_map(
                db,
                user_id,
                user=user,
                include_instance=resolved_include_instance,
            )
        ),
        "fields": fields,
    }
