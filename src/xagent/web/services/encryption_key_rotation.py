"""Raw-column key readability checks and offline Fernet rotation.

Reflection deliberately bypasses ORM encryption decorators and property setters.
The catalog is restricted to credential-bearing paths, never arbitrary JSON text.
"""

import logging
import os
from collections.abc import Callable, Iterator
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import Engine, MetaData, Table, inspect, or_, select, update
from sqlalchemy.engine import Connection

from ...config import (
    ENCRYPTION_KEY,
    get_tool_credentials_encryption_key,
    requires_secure_secrets,
)
from ...core.tools.core.mcp.model import SENSITIVE_AUTH_FIELDS
from ...core.utils.encryption import _get_encryption_key, _looks_like_fernet_token
from ..models.database import get_engine
from ..models.user_channel import CHANNEL_SECRET_FIELDS
from .tool_credentials import _build_fernet_key

logger = logging.getLogger(__name__)
_ROW_BATCH_SIZE = 200

ENCRYPTED_COLUMNS = {
    "models": ("_api_key_encrypted",),
    "user_channels": ("config",),
    "user_oauth": ("access_token", "refresh_token"),
    "oauth_providers": ("client_id", "client_secret"),
    "mcp_servers": ("env", "auth"),
    "user_mcpservers": ("env",),
    "actor_mcp_server_connections": ("encrypted_env",),
    "custom_apis": ("env",),
    "mcp_oauth_clients": ("client_secret",),
    "mcp_oauth_grants": ("access_token", "refresh_token"),
    "mcp_oauth_flow_states": ("code_verifier",),
    "task_runtime_secrets": ("ciphertext",),
    "agent_triggers": ("secret_encrypted",),
    "trigger_runs": ("payload_snapshot",),
    "global_memory_embedding_authority": ("api_key_encrypted",),
}
_REQUIRED_CIPHERTEXT = {
    ("models", "_api_key_encrypted"),
    ("task_runtime_secrets", "ciphertext"),
    ("global_memory_embedding_authority", "api_key_encrypted"),
}
# Match the writers, not the shape of unrelated JSON strings. These columns
# contain plaintext metadata next to explicitly encrypted top-level fields.
_JSON_SECRET_FIELDS = {
    ("user_channels", "config"): CHANNEL_SECRET_FIELDS,
    ("mcp_servers", "auth"): SENSITIVE_AUTH_FIELDS,
    ("trigger_runs", "payload_snapshot"): ("encrypted_payload",),
}
# encrypt_env_dict and actor MCP credential storage encrypt direct string
# values only. Nested objects/lists are not ciphertext-bearing paths.
_ENV_COLUMNS = {
    ("mcp_servers", "env"),
    ("user_mcpservers", "env"),
    ("actor_mcp_server_connections", "encrypted_env"),
    ("custom_apis", "env"),
}
# These consumers already fail closed per account/flow/run. An old unusable
# token must not prevent unrelated accounts or admin recovery from starting.
_RECOVERABLE_TABLES = {
    "user_oauth",
    "mcp_oauth_flow_states",
    "task_runtime_secrets",
    "trigger_runs",
}


def credential_rows(
    connection: Connection, catalog: dict[str, tuple[str, ...]]
) -> Iterator[tuple[Table, Any, str, Any]]:
    available = set(inspect(connection).get_table_names())
    for name, names in catalog.items():
        if name not in available:
            continue
        table = Table(name, MetaData(), autoload_with=connection)
        columns = [table.c[name] for name in names if name in table.c]
        if not columns:
            continue
        primary = list(table.primary_key.columns)
        if len(primary) != 1:
            raise RuntimeError(f"Unsupported credential table identity: {name}")
        last_id = None
        while True:
            statement = (
                select(primary[0], *columns)
                .where(or_(*(column.is_not(None) for column in columns)))
                .order_by(primary[0])
                .limit(_ROW_BATCH_SIZE)
            )
            if last_id is not None:
                statement = statement.where(primary[0] > last_id)
            # Exhaust the SELECT before the caller issues UPDATEs on SQLite.
            # Keyset pagination bounds memory without skipping updated rows.
            rows = connection.execute(statement).mappings().all()
            if not rows:
                break
            for row in rows:
                for column in columns:
                    yield table, row[primary[0].name], column.name, row[column.name]
            last_id = rows[-1][primary[0].name]


def transform_credential_value(
    table_name: str, column: str, value: Any, transform: Callable[[str], str]
) -> Any:
    """Apply crypto only to paths encrypted by the corresponding writer."""
    identity = (table_name, column)

    def transform_token(item: Any) -> Any:
        if (
            isinstance(item, str)
            and item
            and (identity in _REQUIRED_CIPHERTEXT or _looks_like_fernet_token(item))
        ):
            return transform(item)
        return item

    if identity in _ENV_COLUMNS or identity in _JSON_SECRET_FIELDS:
        if not isinstance(value, dict):
            return value
        fields = (
            value.keys() if identity in _ENV_COLUMNS else _JSON_SECRET_FIELDS[identity]
        )
        return {
            key: transform_token(item) if key in fields else item
            for key, item in value.items()
        }
    return transform_token(value)


def tool_credential_items(value: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield the tool credential paths shared by admission and offline rotation."""
    if not isinstance(value, dict):
        return
    for group in ("credentials", "sql_connections"):
        fields = value.get(group)
        if isinstance(fields, dict):
            for field, item in fields.items():
                if isinstance(item, dict) and isinstance(item.get("ciphertext"), str):
                    yield f"{group}.{field!r}", item


def validate_stored_encryption_keys(engine: Engine | None = None) -> None:
    """Refuse deployed admission rather than silently losing old credentials."""
    if not requires_secure_secrets():
        return
    master = Fernet(os.environ[ENCRYPTION_KEY].encode())
    tool = Fernet(_build_fernet_key(get_tool_credentials_encryption_key()))

    def validate_token(token: str) -> str:
        master.decrypt(token.encode())
        return token

    recoverable_failures = 0
    with (engine if engine is not None else get_engine()).connect() as connection:
        for table, row_id, column, value in credential_rows(
            connection, ENCRYPTED_COLUMNS
        ):
            try:
                transform_credential_value(table.name, column, value, validate_token)
            except (InvalidToken, ValueError):
                if table.name in _RECOVERABLE_TABLES:
                    recoverable_failures += 1
                    if recoverable_failures <= 5:
                        logger.warning(
                            "Unreadable recoverable credential: %s row %s field %s; "
                            "reconnect the account or restart the affected flow/run; "
                            "ciphertext preserved; see docs/tool-credential-key-upgrade.md",
                            table.name,
                            row_id,
                            column,
                        )
                    continue
                raise RuntimeError(
                    f"Unreadable encrypted credential: {table.name} row {row_id} field {column}; see docs/tool-credential-key-upgrade.md"
                ) from None
        for table, row_id, column, value in credential_rows(
            connection, {"tool_configs": ("config",), "user_tool_configs": ("config",)}
        ):
            for field, item in tool_credential_items(value):
                try:
                    tool.decrypt(item["ciphertext"].encode())
                except (InvalidToken, ValueError):
                    raise RuntimeError(
                        f"Unreadable tool credential: {table.name} row {row_id} field {field}; see docs/tool-credential-key-upgrade.md"
                    ) from None
    if recoverable_failures > 5:
        logger.warning(
            "%s additional unreadable recoverable credential fields; "
            "ciphertext preserved; see docs/tool-credential-key-upgrade.md",
            recoverable_failures - 5,
        )


def rotate_encryption_key(connection: Connection, old: Fernet, new: Fernet) -> int:
    """Rotate inside the caller's transaction, failing without partial writes."""
    count = 0
    for table, row_id, column, value in credential_rows(connection, ENCRYPTED_COLUMNS):

        def rotate(token: str) -> str:
            nonlocal count
            try:
                new.decrypt(token.encode())
                return token
            except (InvalidToken, ValueError):
                pass
            try:
                plaintext = old.decrypt(token.encode())
            except (InvalidToken, ValueError):
                raise RuntimeError(
                    f"Unreadable {table.name} row {row_id} field {column}; migration rolled back"
                ) from None
            count += 1
            return new.encrypt(plaintext).decode()

        rotated = transform_credential_value(table.name, column, value, rotate)
        if rotated != value:
            values = {column: rotated}
            if table.name == "global_memory_embedding_authority":
                from .global_memory_embedding_authority import _credential_verifier

                # The verifier derives from the configured master key, not the
                # passed Fernet object. Refuse a mismatched caller atomically.
                try:
                    Fernet(_get_encryption_key().encode()).decrypt(rotated.encode())
                except (InvalidToken, ValueError):
                    raise RuntimeError(
                        "Rotation destination must match configured ENCRYPTION_KEY "
                        "for the global authority; migration rolled back"
                    ) from None
                values["credential_verifier"] = _credential_verifier(
                    new.decrypt(rotated.encode()).decode()
                )
            primary = next(iter(table.primary_key.columns))
            connection.execute(update(table).where(primary == row_id).values(**values))
    return count
