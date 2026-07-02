"""add scoped tool credentials

Revision ID: 20260602_add_scoped_tool_credentials
Revises: 20260529_merge_email_reset_and_agent_origin_heads
Create Date: 2026-06-02 00:00:00.000000

"""

import base64
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import and_, inspect, text

revision: str = "20260602_add_scoped_tool_credentials"
down_revision: str | None = "20260529_add_oidc_consumed_tokens"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TOOL_CREDENTIAL_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "exa_web_search": {"api_key": {"secret": True}},
    "zhipu_web_search": {
        "api_key": {"secret": True},
        "base_url": {"secret": False},
    },
    "tavily_web_search": {"api_key": {"secret": True}},
    "web_search": {
        "api_key": {"secret": True},
        "cse_id": {"secret": False},
    },
}


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


def _coerce_json_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _legacy_credential_value(stored: Any, *, secret: bool) -> str | None:
    if isinstance(stored, Mapping):
        if secret:
            ciphertext = stored.get("ciphertext")
            if isinstance(ciphertext, str):
                return _decrypt(ciphertext)
        else:
            value = stored.get("value")
            if isinstance(value, str):
                return value
        return None
    if isinstance(stored, str):
        return stored
    return None


def _legacy_credential_mask(stored: Any, value: str) -> str:
    if isinstance(stored, Mapping):
        masked = stored.get("masked")
        if isinstance(masked, str) and masked:
            return masked
    return _mask_value(value)


def _backfill_legacy_tool_config_credentials() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "tool_configs" not in inspector.get_table_names():
        return

    rows = bind.execute(
        text("SELECT tool_name, config FROM tool_configs WHERE config IS NOT NULL")
    ).mappings()
    for row in rows:
        tool_name = row["tool_name"]
        if not isinstance(tool_name, str):
            continue
        specs = TOOL_CREDENTIAL_SPECS.get(tool_name)
        if specs is None:
            continue

        config = _coerce_json_mapping(row["config"])
        credentials = _coerce_json_mapping(config.get("credentials"))
        if not credentials:
            continue

        for field_name, field_spec in specs.items():
            value = _legacy_credential_value(
                credentials.get(field_name),
                secret=bool(field_spec.get("secret", False)),
            )
            if not value:
                continue

            bind.execute(
                text("""
                INSERT INTO scoped_tool_credentials (
                    scope_type,
                    scope_id,
                    tool_name,
                    field_name,
                    encrypted_value,
                    masked_value
                )
                SELECT
                    :scope_type,
                    NULL,
                    :tool_name,
                    :field_name,
                    :encrypted_value,
                    :masked_value
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM scoped_tool_credentials
                    WHERE scope_type = :scope_type
                        AND scope_id IS NULL
                        AND tool_name = :tool_name
                        AND field_name = :field_name
                )
            """),
                {
                    "scope_type": "instance",
                    "tool_name": tool_name,
                    "field_name": field_name,
                    "encrypted_value": _encrypt(value),
                    "masked_value": _legacy_credential_mask(
                        credentials.get(field_name),
                        value,
                    ),
                },
            )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = inspector.get_table_names()
    if "scoped_tool_credentials" not in tables:
        op.create_table(
            "scoped_tool_credentials",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("scope_type", sa.String(length=20), nullable=False),
            sa.Column("scope_id", sa.Integer(), nullable=True),
            sa.Column("tool_name", sa.String(length=100), nullable=False),
            sa.Column("field_name", sa.String(length=100), nullable=False),
            sa.Column("encrypted_value", sa.Text(), nullable=False),
            sa.Column("masked_value", sa.String(length=500), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
            ),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    indexes = {idx["name"] for idx in inspector.get_indexes("scoped_tool_credentials")}
    for index_name, column_name in (
        ("ix_scoped_tool_credentials_id", "id"),
        ("ix_scoped_tool_credentials_scope_type", "scope_type"),
        ("ix_scoped_tool_credentials_scope_id", "scope_id"),
        ("ix_scoped_tool_credentials_tool_name", "tool_name"),
        ("ix_scoped_tool_credentials_field_name", "field_name"),
    ):
        if index_name not in indexes:
            op.create_index(index_name, "scoped_tool_credentials", [column_name])

    if "uq_scoped_tool_credential_scoped" not in indexes:
        op.create_index(
            "uq_scoped_tool_credential_scoped",
            "scoped_tool_credentials",
            ["scope_type", "scope_id", "tool_name", "field_name"],
            unique=True,
            sqlite_where=sa.column("scope_id").is_not(None),
            postgresql_where=sa.column("scope_id").is_not(None),
        )
    if "uq_scoped_tool_credential_instance" not in indexes:
        op.create_index(
            "uq_scoped_tool_credential_instance",
            "scoped_tool_credentials",
            ["tool_name", "field_name"],
            unique=True,
            sqlite_where=and_(
                sa.column("scope_type") == "instance",
                sa.column("scope_id").is_(None),
            ),
            postgresql_where=and_(
                sa.column("scope_type") == "instance",
                sa.column("scope_id").is_(None),
            ),
        )

    _backfill_legacy_tool_config_credentials()


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "scoped_tool_credentials" not in inspector.get_table_names():
        return

    indexes = {idx["name"] for idx in inspector.get_indexes("scoped_tool_credentials")}
    for index_name in (
        "uq_scoped_tool_credential_instance",
        "uq_scoped_tool_credential_scoped",
        "ix_scoped_tool_credentials_field_name",
        "ix_scoped_tool_credentials_tool_name",
        "ix_scoped_tool_credentials_scope_id",
        "ix_scoped_tool_credentials_scope_type",
        "ix_scoped_tool_credentials_id",
    ):
        if index_name in indexes:
            op.drop_index(index_name, table_name="scoped_tool_credentials")
    op.drop_table("scoped_tool_credentials")
