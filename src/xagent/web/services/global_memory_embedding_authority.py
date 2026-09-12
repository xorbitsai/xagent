from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ...core.model.providers import canonical_provider_name
from ...core.utils.encryption import (
    EncryptionDecodeError,
    decrypt_value_strict,
    get_cipher,
)
from ..models.global_memory_embedding_authority import GlobalMemoryEmbeddingAuthority

AUTHORITY_KEY = "global"
_DEFAULT_ENDPOINTS = {
    "dashscope": "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
    "openai": "https://api.openai.com/v1/embeddings",
    "xinference": "http://localhost:9997",
}


class AuthorityCredentialUnavailable(RuntimeError):
    pass


class AuthorityConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1, max_length=50)
    model_name: str = Field(min_length=1, max_length=100)
    endpoint: str | None = Field(default=None, max_length=500)
    dimension: int = Field(gt=0)
    instruct: str | None = None
    max_retries: int = Field(default=10, ge=0)
    api_key: SecretStr


@dataclass(frozen=True)
class GlobalMemoryEmbeddingAuthoritySnapshot:
    provider: str
    model_name: str
    endpoint: str
    dimension: int
    instruct: str | None
    max_retries: int
    api_key: SecretStr = field(repr=False, compare=False)
    credential_identity: str = field(repr=False)

    def semantic_fingerprint(self) -> str:
        payload = (
            self.provider,
            self.model_name,
            self.endpoint,
            self.dimension,
            self.instruct,
            self.max_retries,
            self.credential_identity,
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def _canonicalize(config: AuthorityConfiguration) -> AuthorityConfiguration:
    provider = canonical_provider_name(config.provider)
    provider = {
        "openai_embedding": "openai",
        "openai-compatible": "openai",
    }.get(provider, provider)
    if provider not in _DEFAULT_ENDPOINTS:
        raise ValueError("Unsupported global memory embedding provider")
    model_name = config.model_name.strip()
    endpoint = (config.endpoint or _DEFAULT_ENDPOINTS[provider]).strip().rstrip("/")
    if provider == "openai" and endpoint.endswith("/v1"):
        endpoint += "/embeddings"
    instruct = config.instruct.strip() if config.instruct else None
    if provider != "dashscope":
        instruct = None
    if not model_name or not endpoint:
        raise ValueError("Global memory embedding identity is incomplete")
    if not config.api_key.get_secret_value().strip():
        raise ValueError("Global memory embedding credential is required")
    return AuthorityConfiguration(
        provider=provider,
        model_name=model_name,
        endpoint=endpoint,
        dimension=config.dimension,
        instruct=instruct,
        max_retries=config.max_retries,
        api_key=config.api_key,
    )


def _credential_digest(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class GlobalMemoryEmbeddingAuthorityService:
    def __init__(self, db: Session):
        self.db = db

    def get_row(self) -> GlobalMemoryEmbeddingAuthority | None:
        return self.db.get(GlobalMemoryEmbeddingAuthority, AUTHORITY_KEY)

    def set(
        self, config: AuthorityConfiguration, *, actor_subject: str
    ) -> GlobalMemoryEmbeddingAuthority:
        normalized = _canonicalize(config)
        secret = normalized.api_key.get_secret_value()
        try:
            encrypted = get_cipher().encrypt(secret.encode()).decode()
        except ValueError:
            raise AuthorityCredentialUnavailable(
                "Global memory embedding credential is unavailable"
            ) from None
        values: dict[str, Any] = {
            "authority_key": AUTHORITY_KEY,
            "model_provider": normalized.provider,
            "model_name": normalized.model_name,
            "base_url": normalized.endpoint,
            "dimension": normalized.dimension,
            "instruct": normalized.instruct,
            "max_retries": normalized.max_retries,
            "api_key_encrypted": encrypted,
            "credential_digest": _credential_digest(secret),
            "configured_by_actor_subject": actor_subject,
        }
        bind = self.db.get_bind()
        dialect = bind.dialect.name if bind is not None else ""
        insert = (
            sqlite_insert
            if dialect == "sqlite"
            else pg_insert
            if dialect == "postgresql"
            else None
        )
        if insert is None:
            raise RuntimeError(
                "Global memory embedding authority requires an atomic upsert dialect"
            )
        stmt = insert(GlobalMemoryEmbeddingAuthority).values(**values)
        update_values = {
            key: value for key, value in values.items() if key != "authority_key"
        }
        update_values["updated_at"] = func.now()
        self.db.execute(
            stmt.on_conflict_do_update(
                index_elements=["authority_key"], set_=update_values
            )
        )
        self.db.commit()
        row = self.get_row()
        assert row is not None
        return row

    def delete(self) -> None:
        self.db.query(GlobalMemoryEmbeddingAuthority).delete()
        self.db.commit()

    def load_snapshot(self) -> GlobalMemoryEmbeddingAuthoritySnapshot | None:
        row = self.get_row()
        if row is None:
            return None
        try:
            secret = decrypt_value_strict(str(row.api_key_encrypted))
        except (EncryptionDecodeError, ValueError):
            raise AuthorityCredentialUnavailable(
                "Global memory embedding credential is unavailable"
            ) from None
        digest = _credential_digest(secret)
        if not secret or digest != row.credential_digest:
            raise AuthorityCredentialUnavailable(
                "Global memory embedding credential is unavailable"
            )
        return GlobalMemoryEmbeddingAuthoritySnapshot(
            provider=str(row.model_provider),
            model_name=str(row.model_name),
            endpoint=str(row.base_url),
            dimension=int(row.dimension),
            instruct=str(row.instruct) if row.instruct is not None else None,
            max_retries=int(row.max_retries),
            api_key=SecretStr(secret),
            credential_identity=digest,
        )

    def credential_status(self) -> str:
        try:
            return "configured" if self.load_snapshot() is not None else "unavailable"
        except AuthorityCredentialUnavailable:
            return "unavailable"
