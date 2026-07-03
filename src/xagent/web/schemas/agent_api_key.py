"""Pydantic response models for the agent API key admin endpoints.

These models shape the three responses returned by
``POST /api/agents/{agent_id}/api-key`` (generate / rotate),
``GET .../api-key`` (read active metadata), and
``DELETE .../api-key`` (soft revoke). They are response-only -- the
endpoints take no request body beyond the path parameter and the JWT
header, so no corresponding ``*Request`` models are needed here.

Design notes:

  - ``full_key`` is plaintext and appears exactly once across an agent's
    lifetime (per rotation). It must be displayed to the agent owner
    immediately on POST response and is never refetchable.

  - ``masked_key`` deliberately uses a fixed bullet count rather than
    masking the real secret length. The secret is 32 chars; rendering
    eight bullets keeps the display short and prevents leaking length
    metadata in screenshots.

  - ``APIKeyRevokeResponse.revoked`` is the wire signal of idempotency:
    True means this call actually flipped an active row, False means
    no active row existed and the call was a no-op. Both cases return
    HTTP 200; clients should not treat the distinction as an error.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# ===== Multi-key admin schemas (``/api/agent-api-keys``) =====
#
# These back the centralized "API Keys" page, which lists and manages
# keys across every agent the caller owns -- unlike the three models
# above, which are scoped to the legacy single-key-per-agent endpoints.


class AgentApiKeyCreateRequest(BaseModel):
    """Request body for ``POST /api/agent-api-keys``."""

    agent_id: int = Field(..., description="Agent to create a new key for.")
    label: Optional[str] = Field(
        None, max_length=100, description="Owner-facing display name for the key."
    )


class AgentApiKeyListItem(BaseModel):
    """One row in the centralized API Keys table."""

    id: int
    agent_id: int
    agent_name: str
    label: Optional[str] = None
    key_prefix: str
    masked_key: str
    status: str = Field(..., description="One of: active, paused, revoked.")
    last_used_at: Optional[datetime] = None
    created_at: datetime


class AgentApiKeyStats(BaseModel):
    """Aggregate counters for the API Keys page's stat cards."""

    total_keys: int
    active_keys: int
    calls_this_month: int
    last_api_call: Optional[datetime] = None


class APIKeyGenerateResponse(BaseModel):
    """Response model for ``POST /api/agents/{agent_id}/api-key``.

    The ``full_key`` is plaintext and returned exactly once per
    rotation. Clients (the web UI) must show it to the agent owner
    immediately and warn that it will not be retrievable later. The
    server only persists ``bcrypt(full_key)`` in ``agent_api_keys.key_hash``;
    the plaintext leaves this response and is never written to disk
    server-side.
    """

    full_key: str = Field(
        ...,
        description=(
            "Plaintext API key in the format xag_<6 chars>_<32 chars>. "
            "Returned exactly once; cannot be retrieved later."
        ),
    )
    key_prefix: str = Field(
        ...,
        description=(
            "Public-safe 6-char lookup handle. Same value embedded in "
            "the full_key middle segment; returned separately so the UI "
            "does not have to re-parse the full key."
        ),
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the new key was persisted.",
    )


class APIKeyMetadataResponse(BaseModel):
    """Response model for ``GET /api/agents/{agent_id}/api-key``.

    Returned only when an active (non-revoked) key exists for the agent;
    callers receive HTTP 404 with ``detail='no_active_key'`` otherwise.
    Crucially, ``full_key`` is **not** part of this shape -- the
    plaintext secret is unrecoverable post-generation by design.
    """

    key_prefix: str = Field(
        ...,
        description="Public-safe 6-char lookup handle of the active key.",
    )
    masked_key: str = Field(
        ...,
        description=(
            "Display form ``xag_<prefix>_••••••••`` with a fixed eight "
            "bullet characters. The bullet count does not reflect the "
            "secret's real length (32 chars) by design."
        ),
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the active key was created.",
    )


class APIKeyRevokeResponse(BaseModel):
    """Response model for ``DELETE /api/agents/{agent_id}/api-key``.

    The endpoint is idempotent and always returns HTTP 200 when the
    caller owns the agent. ``revoked`` distinguishes the two cases:

      - ``True``: there was an active key and this call flipped its
        ``revoked_at`` to now.
      - ``False``: no active key existed; the call was a safe no-op.
        ``revoked_at`` is ``None`` in this case.
    """

    revoked: bool = Field(
        ...,
        description=(
            "True if this call actually revoked an active key; False if "
            "no active key existed (idempotent no-op)."
        ),
    )
    revoked_at: Optional[datetime] = Field(
        None,
        description=(
            "Set to the UTC revocation timestamp when ``revoked`` is True; "
            "None when ``revoked`` is False."
        ),
    )
