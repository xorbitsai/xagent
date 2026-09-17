from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, SecretStr
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
_DEFAULT_PORTS = {"http": 80, "https": 443}
CREDENTIAL_CONFIGURED = "configured"
CREDENTIAL_UNAVAILABLE = "unavailable"
_MAX_ENDPOINT_LENGTH = 500
_MAX_MODEL_NAME_LENGTH = 100
_MAX_DIMENSION = 65536
_MAX_RETRIES = 100
_HOSTNAME = re.compile(
    r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*"
)
_CREDENTIAL_IDENTITY_PURPOSE = (
    b"global-memory-embedding-authority:credential-identity:v1"
)
# This table is introduced by the current unmerged migration, so there is no
# production bare-digest backfill. Unknown/legacy versions deliberately fail closed.
_CREDENTIAL_VERIFIER_PREFIX = "hmac-sha256$v1$"
_CREDENTIAL_VERIFIER_PURPOSE = b"global-memory-embedding-authority:credential:v1"


class AuthorityCredentialUnavailable(RuntimeError):
    pass


class CredentialSource(str, Enum):
    """Who owns the credential: an explicit fact, never inferred from the
    configuring actor's admin rights or subject."""

    APPLICATION_OWNED = "application_owned"
    ORGANIZATION_OWNED = "organization_owned"
    PERSONAL = "personal"
    UNKNOWN = "unknown"


GLOBALLY_SHAREABLE_CREDENTIAL_SOURCES = frozenset(
    {CredentialSource.APPLICATION_OWNED, CredentialSource.ORGANIZATION_OWNED}
)


class AuthorityConfiguration(BaseModel):
    """Admin-supplied authority request, deliberately unconstrained here.

    A pydantic failure is rendered by the shared ``/api`` validation handler,
    which echoes the raw ``input`` into the 422 body and logs it -- and for a
    missing field that input is the whole body, credential included. Domain
    rules live in :func:`_canonicalize`, which raises only sanitized errors.
    """

    model_config = ConfigDict(frozen=True)

    provider: str = ""
    model_name: str = ""
    endpoint: str | None = None
    dimension: int = 0
    instruct: str | None = None
    max_retries: int = 10
    credential_source: CredentialSource = CredentialSource.UNKNOWN
    global_sharing_consent: bool = False
    api_key: SecretStr = SecretStr("")


@dataclass(frozen=True, kw_only=True)
class GlobalMemoryEmbeddingAuthorityRecord:
    """Secret-free view of the authority, materialized before ``COMMIT`` so a
    writer never re-reads a row a concurrent delete may already have taken."""

    provider: str
    model_name: str
    endpoint: str
    dimension: int
    instruct: str | None
    max_retries: int
    credential_source: CredentialSource
    global_sharing_consent: bool
    consented_by_actor_subject: str
    consented_at: datetime
    created_at: datetime | None = None
    updated_at: datetime | None = None
    credential_status: str = CREDENTIAL_UNAVAILABLE


@dataclass(frozen=True, kw_only=True)
class GlobalMemoryEmbeddingAuthoritySnapshot(GlobalMemoryEmbeddingAuthorityRecord):
    api_key: SecretStr = field(repr=False, compare=False)
    credential_identity: str = field(repr=False)

    def _vector_space_inputs(self) -> tuple[Any, ...]:
        return (
            self.provider,
            self.model_name,
            self.endpoint,
            self.dimension,
            self.instruct,
        )

    def authority_fingerprint(self) -> str:
        """Identity of the authority as a whole, including how it calls out.

        Rotating the credential or changing the retry budget moves this value
        and never moves :meth:`vector_space_fingerprint`; governance facts
        stay out, because a clerical re-approval is not a change."""
        return _fingerprint(
            (*self._vector_space_inputs(), self.max_retries, self.credential_identity)
        )

    def vector_space_fingerprint(self) -> str:
        """Identity of the embedding space alone: only the inputs that decide
        what a vector means. Key rotation and retry changes move
        :meth:`authority_fingerprint` but never this value, so vectors already
        stored under it stay comparable."""
        return _fingerprint(self._vector_space_inputs())


def _record_from_row(
    row: GlobalMemoryEmbeddingAuthority,
) -> GlobalMemoryEmbeddingAuthorityRecord:
    return GlobalMemoryEmbeddingAuthorityRecord(
        provider=str(row.model_provider),
        model_name=str(row.model_name),
        endpoint=str(row.base_url),
        dimension=int(row.dimension),
        instruct=str(row.instruct) if row.instruct is not None else None,
        max_retries=int(row.max_retries),
        credential_source=CredentialSource(str(row.credential_source)),
        global_sharing_consent=bool(row.global_sharing_consent),
        consented_by_actor_subject=str(row.consented_by_actor_subject),
        consented_at=cast(datetime, row.consented_at),
        created_at=cast("datetime | None", row.created_at),
        updated_at=cast("datetime | None", row.updated_at),
    )


def _fingerprint(payload: tuple[Any, ...]) -> str:
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
    # The rewrite happens after _canonical_endpoint's own bound, so the value
    # that actually reaches base_url = String(500) is re-checked here.
    if len(endpoint) > _MAX_ENDPOINT_LENGTH:
        raise ValueError("Global memory embedding endpoint is invalid")
    instruct = config.instruct.strip() if config.instruct else None
    if provider != "dashscope":
        instruct = None
    if not model_name or len(model_name) > _MAX_MODEL_NAME_LENGTH:
        raise ValueError("Global memory embedding identity is incomplete")
    if not 0 < config.dimension <= _MAX_DIMENSION:
        raise ValueError("Global memory embedding dimension is out of range")
    if not 0 <= config.max_retries <= _MAX_RETRIES:
        raise ValueError("Global memory embedding retry budget is out of range")
    if config.credential_source not in GLOBALLY_SHAREABLE_CREDENTIAL_SOURCES:
        raise ValueError("Global memory embedding credential is not shareable")
    if config.global_sharing_consent is not True:
        raise ValueError("Global memory embedding sharing consent is required")
    if not config.api_key.get_secret_value().strip():
        raise ValueError("Global memory embedding credential is required")
    return config.model_copy(
        update={
            "provider": provider,
            "model_name": model_name,
            "endpoint": endpoint,
            "instruct": instruct,
        }
    )


def _canonical_host(hostname: str) -> str:
    """Wire-canonical host: IDNA ASCII or IP literal, no terminal DNS dot."""
    host = hostname.rstrip(".")
    try:
        if ":" in host:
            return f"[{ipaddress.IPv6Address(host).compressed}]"
        if host.replace(".", "").isdigit():
            return ipaddress.IPv4Address(host).compressed
        ascii_host = host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        raise ValueError("Global memory embedding endpoint is invalid") from None
    if len(ascii_host) > 253 or not _HOSTNAME.fullmatch(ascii_host):
        raise ValueError("Global memory embedding endpoint is invalid")
    return ascii_host


def _canonical_endpoint(value: str) -> str:
    """Rebuild the endpoint from canonical components.

    The length bound lives here, not on the request field, so an overlong --
    possibly credential-bearing -- endpoint is rejected by a sanitized error
    instead of echoed into the 422 body and the server log by pydantic.
    """
    endpoint = value.strip()
    if len(endpoint) > _MAX_ENDPOINT_LENGTH:
        raise ValueError("Global memory embedding endpoint is invalid")
    if any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in endpoint
    ):
        raise ValueError("Global memory embedding endpoint is invalid")
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        raise ValueError("Global memory embedding endpoint is invalid") from None
    scheme = parsed.scheme.lower()
    if scheme not in _DEFAULT_PORTS or parsed.hostname is None:
        raise ValueError("Global memory embedding endpoint must be absolute HTTP(S)")
    if port == 0:
        raise ValueError("Global memory embedding endpoint is invalid")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Global memory embedding endpoint must be credential-free")
    host = _canonical_host(parsed.hostname)
    netloc = host if port in (None, _DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    canonical = f"{scheme}://{netloc}{parsed.path}".rstrip("/")
    if len(canonical) > _MAX_ENDPOINT_LENGTH:
        raise ValueError("Global memory embedding endpoint is invalid")
    return canonical


def _credential_identity(api_key: str) -> str:
    """Keyed, in-memory-only identity: not brute-forceable back to the key
    without the deployment secret, and never persisted or shown in a repr."""
    return derive_secret_hmac(api_key, purpose=_CREDENTIAL_IDENTITY_PURPOSE)


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
    ) -> GlobalMemoryEmbeddingAuthorityRecord:
        normalized = _canonicalize(config)
        secret = normalized.api_key.get_secret_value()
        try:
            encrypted = get_cipher().encrypt(secret.encode()).decode()
            # Decided in-transaction from the ciphertext actually written, so
            # the write's own result never has to guess or re-read the row.
            round_trips = decrypt_value_strict(encrypted) == secret
        except (EncryptionDecodeError, ValueError):
            raise AuthorityCredentialUnavailable(
                "Global memory embedding credential is unavailable"
            ) from None
        # Provenance comes from the authenticated actor and the server clock;
        # the request body never gets to state who consented.
        consented_at = datetime.now(timezone.utc)
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
            "credential_source": normalized.credential_source.value,
            "global_sharing_consent": True,
            "configured_by_actor_subject": actor_subject,
            "consented_by_actor_subject": actor_subject,
            "consented_at": consented_at,
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
        table = GlobalMemoryEmbeddingAuthority.__table__
        written = self.db.execute(
            stmt.on_conflict_do_update(
                index_elements=["authority_key"], set_=update_values
            ).returning(table.c.created_at, table.c.updated_at)
        ).one()
        # Built before COMMIT: a concurrent delete may win the row the moment
        # the transaction lands, and must not fail this durable write.
        record = GlobalMemoryEmbeddingAuthorityRecord(
            provider=normalized.provider,
            model_name=normalized.model_name,
            endpoint=normalized.endpoint or "",
            dimension=normalized.dimension,
            instruct=normalized.instruct,
            max_retries=normalized.max_retries,
            credential_source=normalized.credential_source,
            global_sharing_consent=True,
            consented_by_actor_subject=actor_subject,
            consented_at=consented_at,
            created_at=written.created_at,
            updated_at=written.updated_at,
            credential_status=CREDENTIAL_CONFIGURED
            if round_trips
            else CREDENTIAL_UNAVAILABLE,
        )
        self.db.commit()
        return record

    def delete(self) -> None:
        self.db.query(GlobalMemoryEmbeddingAuthority).delete()
        self.db.commit()

    def read_record(self) -> GlobalMemoryEmbeddingAuthorityRecord | None:
        """The stored authority as the API renders it, credential status included."""
        row = self.get_row()
        if row is None:
            return None
        record = _record_from_row(row)
        return replace(record, credential_status=self.credential_status())

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
            **dict(
                asdict(_record_from_row(row)), credential_status=CREDENTIAL_CONFIGURED
            ),
            api_key=SecretStr(secret),
            credential_identity=_credential_identity(secret),
        )

    def credential_status(self) -> str:
        try:
            snapshot = self.load_snapshot()
        except AuthorityCredentialUnavailable:
            return CREDENTIAL_UNAVAILABLE
        return CREDENTIAL_CONFIGURED if snapshot else CREDENTIAL_UNAVAILABLE
