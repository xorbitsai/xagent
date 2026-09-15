from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ...core.model.providers import canonical_provider_name
from ...core.utils.encryption import (
    EncryptionDecodeError,
    decrypt_value_strict,
    derive_secret_hmac,
    get_cipher,
)
from ..models.global_memory_embedding_authority import GlobalMemoryEmbeddingAuthority

AUTHORITY_KEY = "global"
_DEFAULT_ENDPOINTS = {
    "dashscope": "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
    "openai": "https://api.openai.com/v1/embeddings",
    "xinference": "http://localhost:9997",
}
_CREDENTIAL_IDENTITY_DOMAIN = b"xagent:global-memory-credential-identity:v1\0"
# This table is introduced by the current unmerged migration, so there is no
# production bare-digest backfill. Unknown/legacy versions deliberately fail closed.
_CREDENTIAL_VERIFIER_PREFIX = "hmac-sha256$v1$"
_CREDENTIAL_VERIFIER_PURPOSE = b"global-memory-embedding-authority:credential:v1"


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
    endpoint = _canonical_endpoint(config.endpoint or _DEFAULT_ENDPOINTS[provider])
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


def _canonical_endpoint(value: str) -> str:
    endpoint = value.strip()
    if any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in endpoint
    ):
        raise ValueError("Global memory embedding endpoint is invalid")
    try:
        parsed = urlsplit(endpoint)
        _ = parsed.port
    except ValueError:
        raise ValueError("Global memory embedding endpoint is invalid") from None
    if parsed.scheme.lower() not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("Global memory embedding endpoint must be absolute HTTP(S)")
    hostname = parsed.hostname
    try:
        if ":" in hostname:
            ipaddress.IPv6Address(hostname)
        elif hostname.replace(".", "").isdigit():
            ipaddress.IPv4Address(hostname)
        else:
            ascii_hostname = hostname.encode("idna").decode("ascii")
            labels = ascii_hostname.rstrip(".").split(".")
            if len(ascii_hostname) > 253 or any(
                not label
                or len(label) > 63
                or label.startswith("-")
                or label.endswith("-")
                or not all(
                    character.isalnum() or character == "-" for character in label
                )
                for label in labels
            ):
                raise ValueError
    except (UnicodeError, ValueError):
        raise ValueError("Global memory embedding endpoint is invalid") from None
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Global memory embedding endpoint must be credential-free")
    return endpoint.rstrip("/")


def _credential_identity(api_key: str) -> str:
    return hashlib.sha256(_CREDENTIAL_IDENTITY_DOMAIN + api_key.encode()).hexdigest()


def _credential_verifier(api_key: str) -> str:
    digest = derive_secret_hmac(api_key, purpose=_CREDENTIAL_VERIFIER_PURPOSE)
    return f"{_CREDENTIAL_VERIFIER_PREFIX}{digest}"


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
            "credential_verifier": _credential_verifier(secret),
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
        verifier = _credential_verifier(secret)
        stored_verifier = str(row.credential_verifier).encode(errors="surrogatepass")
        if not secret or not hmac.compare_digest(verifier.encode(), stored_verifier):
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
            credential_identity=_credential_identity(secret),
        )

    def credential_status(self) -> str:
        try:
            return "configured" if self.load_snapshot() is not None else "unavailable"
        except AuthorityCredentialUnavailable:
            return "unavailable"
