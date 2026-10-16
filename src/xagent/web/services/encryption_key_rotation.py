"""Raw-column key readability checks and offline Fernet rotation.

Reflection deliberately bypasses ORM encryption decorators and property setters.
The catalog is restricted to credential-bearing columns, never arbitrary text.
"""

import os
from collections.abc import Callable, Iterator
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import MetaData, Table, inspect, select, update
from sqlalchemy.engine import Connection

from ...config import (
    ENCRYPTION_KEY,
    get_tool_credentials_encryption_key,
    requires_secure_secrets,
)
from ...core.utils.encryption import _looks_like_fernet_token
from ..models.database import get_engine
from .tool_credentials import _build_fernet_key

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
        for row in connection.execute(select(primary[0], *columns)).mappings():
            for column in columns:
                yield table, row[primary[0].name], column.name, row[column.name]


def transform_tokens(
    value: Any, transform: Callable[[str], str], *, required: bool = False
) -> Any:
    if isinstance(value, dict):
        return {key: transform_tokens(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [transform_tokens(item, transform) for item in value]
    if (
        isinstance(value, str)
        and value
        and (required or _looks_like_fernet_token(value))
    ):
        return transform(value)
    return value


def validate_stored_encryption_keys() -> None:
    """Refuse deployed admission rather than silently losing old credentials."""
    if not requires_secure_secrets():
        return
    master = Fernet(os.environ[ENCRYPTION_KEY].encode())
    tool = Fernet(_build_fernet_key(get_tool_credentials_encryption_key()))
    with get_engine().connect() as connection:
        for table, row_id, column, value in credential_rows(
            connection, ENCRYPTED_COLUMNS
        ):
            try:
                transform_tokens(
                    value,
                    lambda token: (master.decrypt(token.encode()), token)[1],
                    required=(table.name, column) in _REQUIRED_CIPHERTEXT,
                )
            except (InvalidToken, ValueError):
                raise RuntimeError(
                    f"Unreadable encrypted credential: {table.name} row {row_id} field {column}; see docs/tool-credential-key-upgrade.md"
                ) from None
        for table, row_id, column, value in credential_rows(
            connection, {"tool_configs": ("config",), "user_tool_configs": ("config",)}
        ):
            if not isinstance(value, dict):
                continue
            for group in ("credentials", "sql_connections"):
                fields = value.get(group)
                if not isinstance(fields, dict):
                    continue
                for field, item in fields.items():
                    if isinstance(item, dict) and isinstance(
                        item.get("ciphertext"), str
                    ):
                        try:
                            tool.decrypt(item["ciphertext"].encode())
                        except (InvalidToken, ValueError):
                            raise RuntimeError(
                                f"Unreadable tool credential: {table.name} row {row_id} field {group}.{field!r}; see docs/tool-credential-key-upgrade.md"
                            ) from None


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

        rotated = transform_tokens(
            value, rotate, required=(table.name, column) in _REQUIRED_CIPHERTEXT
        )
        if rotated != value:
            values = {column: rotated}
            if table.name == "global_memory_embedding_authority":
                from .global_memory_embedding_authority import _credential_verifier

                values["credential_verifier"] = _credential_verifier(
                    new.decrypt(rotated.encode()).decode()
                )
            primary = next(iter(table.primary_key.columns))
            connection.execute(update(table).where(primary == row_id).values(**values))
    return count
