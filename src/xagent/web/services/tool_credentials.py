from __future__ import annotations

import base64
import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, cast

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from ..init_tool_configs import get_default_tool_configs
from ..models.tool_config import ToolConfig, UserToolConfig

ToolFieldSpec = Dict[str, Any]


TOOL_CREDENTIAL_SPECS: Dict[str, Dict[str, ToolFieldSpec]] = {
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


def list_configurable_tool_names() -> list[str]:
    return list(TOOL_CREDENTIAL_SPECS.keys())


SQL_CONNECTION_ENV_PREFIX = "XAGENT_EXTERNAL_DB_"
ALLOWED_SQL_SCHEMES = {"postgresql", "mysql", "mariadb", "mssql", "sqlite"}


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


def _get_or_create_tool_config(db: Session, tool_name: str) -> ToolConfig:
    config = db.query(ToolConfig).filter(ToolConfig.tool_name == tool_name).first()
    if config:
        return config

    defaults = {item["tool_name"]: item for item in get_default_tool_configs()}
    default_data = defaults.get(tool_name)
    if default_data:
        config = ToolConfig(**default_data)
        db.add(config)
        db.flush()
        return config

    config = ToolConfig(
        tool_name=tool_name,
        tool_type="builtin",
        category="search",
        display_name=tool_name,
        description="",
        enabled=True,
        config={},
        dependencies=[],
    )
    db.add(config)
    db.flush()
    return config


def _get_or_create_user_tool_config(
    db: Session, user_id: int, tool_name: str
) -> UserToolConfig:
    config = (
        db.query(UserToolConfig)
        .filter(
            UserToolConfig.user_id == user_id,
            UserToolConfig.tool_name == tool_name,
        )
        .first()
    )
    if config:
        return config

    config = UserToolConfig(user_id=user_id, tool_name=tool_name, config={})
    db.add(config)
    db.flush()
    return config


def _get_storage(config: ToolConfig) -> dict[str, Any]:
    raw_config = cast(Any, getattr(config, "config", None))
    payload: dict[str, Any] = {}
    if isinstance(raw_config, dict):
        for key, value in raw_config.items():
            payload[str(key)] = value
    credentials = payload.get("credentials")
    if not isinstance(credentials, dict):
        credentials = {}
    payload["credentials"] = credentials
    cast(Any, config).config = payload
    return credentials


def _get_user_tool_payload(config: UserToolConfig) -> dict[str, Any]:
    raw_config = cast(Any, getattr(config, "config", None))
    payload: dict[str, Any] = {}
    if isinstance(raw_config, dict):
        for key, value in raw_config.items():
            payload[str(key)] = value
    cast(Any, config).config = payload
    return payload


def _read_env(env_names: Iterable[str]) -> str | None:
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _sanitize_sql_connection_name(name: str) -> str:
    return name.strip().upper()


def _get_user_sql_connection_store(config: UserToolConfig) -> dict[str, Any]:
    payload = _get_user_tool_payload(config)
    sql_connections = payload.get("sql_connections")
    if not isinstance(sql_connections, dict):
        sql_connections = {}
    payload["sql_connections"] = sql_connections
    cast(Any, config).config = payload
    return sql_connections


def _sql_url_mask(url: str) -> str:
    try:
        parsed = make_url(url)
        return parsed.render_as_string(hide_password=True)
    except Exception:
        return _mask_value(url)


def _get_user_sql_tool_config(db: Session, user_id: int) -> UserToolConfig:
    return _get_or_create_user_tool_config(db, user_id, "sql_query")


def set_sql_connection(
    db: Session, user_id: int, name: str, connection_url: str
) -> None:
    normalized_name = _sanitize_sql_connection_name(name)
    if not normalized_name:
        raise ValueError("Connection name is required")
    normalized_url = connection_url.strip()
    if not normalized_url:
        raise ValueError("Connection URL is required")
    try:
        parsed = make_url(normalized_url)
    except Exception as exc:
        raise ValueError("Invalid SQLAlchemy connection URL") from exc

    base_scheme = str(parsed.drivername).split("+", 1)[0].lower()
    if base_scheme not in ALLOWED_SQL_SCHEMES:
        allowed_schemes_text = ", ".join(sorted(ALLOWED_SQL_SCHEMES))
        raise ValueError(
            f"Unsupported SQLAlchemy URL scheme '{parsed.drivername}'. "
            f"Allowed schemes: {allowed_schemes_text}"
        )

    config = _get_user_sql_tool_config(db, user_id)
    storage = _get_user_sql_connection_store(config)
    now = datetime.now(timezone.utc).isoformat()
    storage[normalized_name] = {
        "ciphertext": _encrypt(normalized_url),
        "masked": _sql_url_mask(normalized_url),
        "updated_at": now,
    }
    db.add(config)
    flag_modified(config, "config")
    db.commit()


def delete_sql_connection(db: Session, user_id: int, name: str) -> None:
    normalized_name = _sanitize_sql_connection_name(name)
    config = (
        db.query(UserToolConfig)
        .filter(
            UserToolConfig.user_id == user_id,
            UserToolConfig.tool_name == "sql_query",
        )
        .first()
    )
    if not config:
        return
    payload = cast(Any, getattr(config, "config", None))
    if not isinstance(payload, dict):
        return
    sql_connections = payload.get("sql_connections")
    if not isinstance(sql_connections, dict):
        return
    removed = False
    for key in list(sql_connections.keys()):
        if (
            isinstance(key, str)
            and _sanitize_sql_connection_name(key) == normalized_name
        ):
            del sql_connections[key]
            removed = True

    if removed:
        cast(Any, config).config = payload
        db.add(config)
        flag_modified(config, "config")
        db.commit()


def resolve_sql_connection(db: Session, user_id: int | None, name: str) -> str | None:
    normalized_name = _sanitize_sql_connection_name(name)
    config = None
    if user_id is not None:
        config = (
            db.query(UserToolConfig)
            .filter(
                UserToolConfig.user_id == user_id,
                UserToolConfig.tool_name == "sql_query",
            )
            .first()
        )
    if config and isinstance(config.config, dict):
        sql_connections = config.config.get("sql_connections")
        if isinstance(sql_connections, dict):
            item = sql_connections.get(normalized_name)
            if isinstance(item, dict) and isinstance(item.get("ciphertext"), str):
                decrypted = _decrypt(item["ciphertext"])
                if decrypted:
                    return decrypted

    return os.getenv(f"{SQL_CONNECTION_ENV_PREFIX}{normalized_name}")


def get_sql_connection_map(db: Session, user_id: int | None) -> dict[str, str]:
    result: dict[str, str] = {}

    for key, value in os.environ.items():
        if key.startswith(SQL_CONNECTION_ENV_PREFIX) and value:
            name = key[len(SQL_CONNECTION_ENV_PREFIX) :]
            if name:
                result[name] = value

    if user_id is None:
        return result

    config = (
        db.query(UserToolConfig)
        .filter(
            UserToolConfig.user_id == user_id,
            UserToolConfig.tool_name == "sql_query",
        )
        .first()
    )
    if config and isinstance(config.config, dict):
        sql_connections = config.config.get("sql_connections")
        if isinstance(sql_connections, dict):
            for raw_name, item in sql_connections.items():
                if not isinstance(raw_name, str) or not isinstance(item, dict):
                    continue
                ciphertext = item.get("ciphertext")
                if isinstance(ciphertext, str):
                    decrypted = _decrypt(ciphertext)
                    if decrypted:
                        result[_sanitize_sql_connection_name(raw_name)] = decrypted

    return result


def list_sql_connections(db: Session, user_id: int | None) -> list[dict[str, Any]]:
    env_names = {
        key[len(SQL_CONNECTION_ENV_PREFIX) :]: value
        for key, value in os.environ.items()
        if key.startswith(SQL_CONNECTION_ENV_PREFIX) and value
    }

    db_entries: dict[str, dict[str, Any]] = {}
    config = None
    if user_id is not None:
        config = (
            db.query(UserToolConfig)
            .filter(
                UserToolConfig.user_id == user_id,
                UserToolConfig.tool_name == "sql_query",
            )
            .first()
        )
    if config and isinstance(config.config, dict):
        sql_connections = config.config.get("sql_connections")
        if isinstance(sql_connections, dict):
            for raw_name, item in sql_connections.items():
                if isinstance(raw_name, str) and isinstance(item, dict):
                    db_entries[_sanitize_sql_connection_name(raw_name)] = item

    all_names = sorted(set(env_names.keys()) | set(db_entries.keys()))
    output: list[dict[str, Any]] = []
    for name in all_names:
        db_item = db_entries.get(name)
        if db_item:
            masked = str(db_item.get("masked") or "")
            source = "db"
        else:
            env_value = env_names.get(name, "")
            masked = _sql_url_mask(env_value)
            source = "env" if env_value else "none"

        output.append(
            {
                "name": name,
                "source": source,
                "masked": masked,
                "configured": bool(masked),
            }
        )

    return output


def resolve_tool_credential(db: Session, tool_name: str, field_name: str) -> str | None:
    spec = TOOL_CREDENTIAL_SPECS.get(tool_name, {}).get(field_name)
    if not spec:
        return None

    config = db.query(ToolConfig).filter(ToolConfig.tool_name == tool_name).first()
    if config and isinstance(config.config, dict):
        credentials = config.config.get("credentials")
        if isinstance(credentials, dict):
            stored = credentials.get(field_name)
            if isinstance(stored, dict):
                if stored.get("secret") and isinstance(stored.get("ciphertext"), str):
                    decrypted = _decrypt(stored["ciphertext"])
                    if decrypted:
                        return decrypted
                if not stored.get("secret") and isinstance(stored.get("value"), str):
                    return stored["value"]

    env_names = spec.get("env", [])
    if isinstance(env_names, list):
        return _read_env(env_names)
    return None


def set_tool_credentials(db: Session, tool_name: str, values: dict[str, str]) -> None:
    specs = TOOL_CREDENTIAL_SPECS.get(tool_name)
    if not specs:
        raise ValueError(f"Tool '{tool_name}' is not configurable")

    config = _get_or_create_tool_config(db, tool_name)
    credentials = _get_storage(config)
    now = datetime.now(timezone.utc).isoformat()

    for field_name, raw_value in values.items():
        if field_name not in specs:
            continue
        normalized = raw_value.strip()
        if not normalized:
            continue
        is_secret = bool(specs[field_name].get("secret", False))
        if is_secret:
            credentials[field_name] = {
                "secret": True,
                "ciphertext": _encrypt(normalized),
                "masked": _mask_value(normalized),
                "updated_at": now,
            }
        else:
            credentials[field_name] = {
                "secret": False,
                "value": normalized,
                "updated_at": now,
            }

    db.add(config)
    flag_modified(config, "config")
    db.commit()


def clear_tool_credential(db: Session, tool_name: str, field_name: str) -> None:
    config = db.query(ToolConfig).filter(ToolConfig.tool_name == tool_name).first()
    if not config or not isinstance(config.config, dict):
        return
    credentials = config.config.get("credentials")
    if not isinstance(credentials, dict):
        return
    if field_name in credentials:
        del credentials[field_name]
        db.add(config)
        flag_modified(config, "config")
        db.commit()


def get_tool_credential_view(db: Session, tool_name: str) -> dict[str, Any]:
    specs = TOOL_CREDENTIAL_SPECS.get(tool_name)
    if not specs:
        raise ValueError(f"Tool '{tool_name}' is not configurable")

    config = db.query(ToolConfig).filter(ToolConfig.tool_name == tool_name).first()
    display_name = tool_name
    if config is not None and isinstance(getattr(config, "display_name", None), str):
        display_name = str(getattr(config, "display_name"))
    else:
        defaults = {item["tool_name"]: item for item in get_default_tool_configs()}
        display_name = str(defaults.get(tool_name, {}).get("display_name") or tool_name)
    credentials_store: dict[str, Any] = {}
    if config and isinstance(config.config, dict):
        maybe = config.config.get("credentials")
        if isinstance(maybe, dict):
            credentials_store = maybe

    fields: dict[str, Any] = {}
    all_required_ok = True
    for field_name, spec in specs.items():
        stored = credentials_store.get(field_name)
        stored_dict = stored if isinstance(stored, dict) else None
        db_set = stored_dict is not None
        db_masked = ""
        if stored_dict is not None:
            if stored_dict.get("secret"):
                db_masked = str(stored_dict.get("masked") or "")
            else:
                db_masked = str(stored_dict.get("value") or "")

        env_value = (
            _read_env(spec.get("env", []))
            if isinstance(spec.get("env"), list)
            else None
        )
        source = "db" if db_set else ("env" if env_value else "none")
        resolved = resolve_tool_credential(db, tool_name, field_name)
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
            "masked": db_masked
            if db_set
            else (_mask_value(env_value) if env_value else ""),
            "env_names": spec.get("env", []),
        }

    return {
        "tool_name": tool_name,
        "display_name": display_name,
        "configured": all_required_ok,
        "fields": fields,
    }
