from __future__ import annotations

import asyncio
import hashlib
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

BROWSER_RELAY_PROTOCOL_VERSION = 1
BROWSER_RELAY_MAX_MESSAGE_BYTES = 12 * 1024 * 1024
BROWSER_RELAY_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0

RelaySend = Callable[[dict[str, Any]], Awaitable[None]]
RelayClose = Callable[[int, str], Awaitable[None]]


class BrowserRelayError(RuntimeError):
    """Base error for the user-browser relay."""


class BrowserRelayAuthenticationError(BrowserRelayError):
    """Raised when an extension cannot authenticate."""


class BrowserRelayUnavailableError(BrowserRelayError):
    """Raised when the user's extension or approved tab is unavailable."""


class BrowserRelayProtocolError(BrowserRelayError):
    """Raised when a relay peer violates the versioned protocol."""


class BrowserRelayInUseError(BrowserRelayError):
    """Raised when another task currently owns the user's browser relay."""


class BrowserRelayHello(BaseModel):
    """First extension message on every relay WebSocket."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["hello"]
    protocol_version: int
    client_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    client_name: str = Field(default="Chrome", min_length=1, max_length=128)
    pairing_token: SecretStr | None = None
    session_token: SecretStr | None = None

    @model_validator(mode="after")
    def validate_one_credential(self) -> "BrowserRelayHello":
        if (self.pairing_token is None) == (self.session_token is None):
            raise ValueError("provide exactly one relay credential")
        return self


class BrowserRelayResponse(BaseModel):
    """Extension response to one server command."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["response"]
    protocol_version: int
    request_id: str = Field(min_length=1, max_length=128)
    success: bool
    result: dict[str, Any] | None = None
    error: str = Field(default="", max_length=2_000)


class BrowserRelayStatusMessage(BaseModel):
    """Extension-reported user authorization and tab state."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["status"]
    protocol_version: int
    attached: bool
    tab_id: int | None = Field(default=None, ge=0)
    title: str | None = Field(default=None, max_length=500)
    url: str | None = Field(default=None, max_length=4_096)


class BrowserRelayPing(BaseModel):
    """Keepalive message sent by the extension service worker."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ping"]
    protocol_version: int


@dataclass(frozen=True)
class BrowserRelayPairing:
    pairing_token: str
    expires_at: datetime


@dataclass(frozen=True)
class BrowserRelayAuthentication:
    user_id: int
    client_id: str
    client_name: str
    session_token: str | None
    paired: bool


@dataclass
class _StoredSecret:
    user_id: int
    expires_at: datetime
    client_id: str | None = None
    client_name: str | None = None


@dataclass
class _RelayClaim:
    owner_task_id: str
    last_activity: datetime


class BrowserRelayConnection:
    """One authenticated extension connection with request/response routing."""

    def __init__(
        self,
        *,
        user_id: int,
        client_id: str,
        client_name: str,
        send: RelaySend,
        close_transport: RelayClose | None = None,
    ) -> None:
        self.user_id = user_id
        self.client_id = client_id
        self.client_name = client_name
        self._send = send
        self._close_transport = close_transport
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._pending_lock = asyncio.Lock()
        self._closed = False
        self.attached = False
        self.tab_id: int | None = None
        self.title: str | None = None
        self.url: str | None = None
        self.connected_at = datetime.now(timezone.utc)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def update_status(self, status: BrowserRelayStatusMessage) -> None:
        self.attached = status.attached
        self.tab_id = status.tab_id if status.attached else None
        self.title = status.title if status.attached else None
        self.url = status.url if status.attached else None

    async def request(
        self,
        command: str,
        payload: dict[str, Any],
        *,
        timeout: float = BROWSER_RELAY_DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        if self._closed:
            raise BrowserRelayUnavailableError("Browser extension is disconnected.")
        if not self.attached:
            raise BrowserRelayUnavailableError(
                "No browser tab is attached. Ask the user to open the Xagent "
                "extension and approve the current tab."
            )
        request_id = secrets.token_urlsafe(18)
        future = asyncio.get_running_loop().create_future()
        async with self._pending_lock:
            self._pending[request_id] = future
        try:
            await self._send(
                {
                    "type": "command",
                    "protocol_version": BROWSER_RELAY_PROTOCOL_VERSION,
                    "request_id": request_id,
                    "command": command,
                    "payload": payload,
                }
            )
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            raise BrowserRelayUnavailableError(
                f"Browser extension did not answer {command!r} in time."
            ) from exc
        finally:
            async with self._pending_lock:
                self._pending.pop(request_id, None)

    async def resolve(self, response: BrowserRelayResponse) -> None:
        if response.protocol_version != BROWSER_RELAY_PROTOCOL_VERSION:
            raise BrowserRelayProtocolError("Browser relay protocol version mismatch.")
        async with self._pending_lock:
            future = self._pending.get(response.request_id)
        if future is None or future.done():
            return
        if response.success:
            future.set_result(response.result or {})
        else:
            future.set_exception(
                BrowserRelayError(response.error or "Browser extension command failed.")
            )

    async def close(
        self,
        *,
        code: int = 1000,
        reason: str = "Browser relay disconnected.",
    ) -> None:
        if self._closed:
            return
        self._closed = True
        async with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(BrowserRelayUnavailableError(reason))
        if self._close_transport is not None:
            try:
                await self._close_transport(code, reason)
            except Exception:
                pass

    def public_status(self) -> dict[str, Any]:
        return {
            "connected": not self._closed,
            "client_id": self.client_id,
            "client_name": self.client_name,
            "attached": self.attached,
            "tab_id": self.tab_id,
            "title": self.title,
            "url": self.url,
            "connected_at": self.connected_at.isoformat(),
        }


class BrowserRelayRegistry:
    """In-process pairing, connection, and task-ownership registry.

    Pairing and session tokens are held only as SHA-256 digests. The session
    token survives extension service-worker restarts, but intentionally becomes
    invalid when the Xagent process restarts.
    """

    def __init__(
        self,
        *,
        pairing_ttl: timedelta = timedelta(minutes=10),
        session_ttl: timedelta = timedelta(days=7),
        claim_ttl: timedelta = timedelta(minutes=30),
    ) -> None:
        self._pairing_ttl = pairing_ttl
        self._session_ttl = session_ttl
        self._claim_ttl = claim_ttl
        self._pairings: dict[str, _StoredSecret] = {}
        self._sessions: dict[str, _StoredSecret] = {}
        self._connections: dict[int, BrowserRelayConnection] = {}
        self._claims: dict[int, _RelayClaim] = {}
        self._lock = asyncio.Lock()

    async def create_pairing(self, user_id: int) -> BrowserRelayPairing:
        if user_id <= 0:
            raise ValueError("browser relay pairing requires an authenticated user")
        now = datetime.now(timezone.utc)
        token = secrets.token_urlsafe(32)
        async with self._lock:
            self._purge_expired(now)
            for digest, pairing in list(self._pairings.items()):
                if pairing.user_id == user_id:
                    self._pairings.pop(digest, None)
            expires_at = now + self._pairing_ttl
            self._pairings[self._digest(token)] = _StoredSecret(
                user_id=user_id,
                expires_at=expires_at,
            )
        return BrowserRelayPairing(pairing_token=token, expires_at=expires_at)

    async def authenticate(
        self,
        hello: BrowserRelayHello,
    ) -> BrowserRelayAuthentication:
        if hello.protocol_version != BROWSER_RELAY_PROTOCOL_VERSION:
            raise BrowserRelayProtocolError(
                f"Unsupported browser relay protocol {hello.protocol_version}; "
                f"expected {BROWSER_RELAY_PROTOCOL_VERSION}."
            )
        now = datetime.now(timezone.utc)
        async with self._lock:
            self._purge_expired(now)
            if hello.pairing_token is not None:
                raw_token = hello.pairing_token.get_secret_value()
                pairing = self._pairings.pop(self._digest(raw_token), None)
                if pairing is None or pairing.expires_at <= now:
                    raise BrowserRelayAuthenticationError(
                        "Pairing token is invalid, expired, or already used."
                    )
                session_token = secrets.token_urlsafe(32)
                self._sessions[self._digest(session_token)] = _StoredSecret(
                    user_id=pairing.user_id,
                    client_id=hello.client_id,
                    client_name=hello.client_name,
                    expires_at=now + self._session_ttl,
                )
                return BrowserRelayAuthentication(
                    user_id=pairing.user_id,
                    client_id=hello.client_id,
                    client_name=hello.client_name,
                    session_token=session_token,
                    paired=True,
                )

            assert hello.session_token is not None
            raw_token = hello.session_token.get_secret_value()
            session = self._sessions.get(self._digest(raw_token))
            if (
                session is None
                or session.expires_at <= now
                or session.client_id != hello.client_id
            ):
                raise BrowserRelayAuthenticationError(
                    "Browser relay session is invalid or expired."
                )
            return BrowserRelayAuthentication(
                user_id=session.user_id,
                client_id=hello.client_id,
                client_name=session.client_name or hello.client_name,
                session_token=None,
                paired=False,
            )

    async def register(self, connection: BrowserRelayConnection) -> None:
        old_connection: BrowserRelayConnection | None
        async with self._lock:
            old_connection = self._connections.get(connection.user_id)
            self._connections[connection.user_id] = connection
        if old_connection is not None and old_connection is not connection:
            await old_connection.close(
                code=4001,
                reason="A newer browser relay connection replaced this one.",
            )

    async def unregister(self, connection: BrowserRelayConnection) -> None:
        async with self._lock:
            if self._connections.get(connection.user_id) is connection:
                self._connections.pop(connection.user_id, None)
        await connection.close()

    async def acquire(
        self,
        *,
        user_id: int,
        owner_task_id: str,
    ) -> BrowserRelayConnection:
        owner = owner_task_id.strip()
        if user_id <= 0 or not owner:
            raise ValueError("browser relay requires authenticated user and task owner")
        now = datetime.now(timezone.utc)
        async with self._lock:
            self._purge_expired(now)
            connection = self._connections.get(user_id)
            if connection is None or connection.is_closed:
                raise BrowserRelayUnavailableError(
                    "Browser extension is not connected for this user."
                )
            claim = self._claims.get(user_id)
            if (
                claim is not None
                and claim.owner_task_id != owner
                and now - claim.last_activity <= self._claim_ttl
            ):
                raise BrowserRelayInUseError(
                    "The user browser is already controlled by another task."
                )
            self._claims[user_id] = _RelayClaim(
                owner_task_id=owner,
                last_activity=now,
            )
            return connection

    async def touch_claim(self, *, user_id: int, owner_task_id: str) -> None:
        now = datetime.now(timezone.utc)
        async with self._lock:
            claim = self._claims.get(user_id)
            if claim is not None and claim.owner_task_id == owner_task_id:
                claim.last_activity = now

    async def release(self, *, user_id: int, owner_task_id: str) -> None:
        async with self._lock:
            claim = self._claims.get(user_id)
            if claim is not None and claim.owner_task_id == owner_task_id:
                self._claims.pop(user_id, None)

    async def status(self, user_id: int) -> dict[str, Any]:
        async with self._lock:
            connection = self._connections.get(user_id)
            if connection is None or connection.is_closed:
                return {"connected": False, "attached": False}
            return connection.public_status()

    async def revoke_user(self, user_id: int) -> None:
        async with self._lock:
            for digest, secret in list(self._pairings.items()):
                if secret.user_id == user_id:
                    self._pairings.pop(digest, None)
            for digest, secret in list(self._sessions.items()):
                if secret.user_id == user_id:
                    self._sessions.pop(digest, None)
            connection = self._connections.pop(user_id, None)
            self._claims.pop(user_id, None)
        if connection is not None:
            await connection.close(
                code=4002,
                reason="Browser relay access was revoked.",
            )

    def _purge_expired(self, now: datetime) -> None:
        for collection in (self._pairings, self._sessions):
            for digest, secret in list(collection.items()):
                if secret.expires_at <= now:
                    collection.pop(digest, None)
        for user_id, claim in list(self._claims.items()):
            if now - claim.last_activity > self._claim_ttl:
                self._claims.pop(user_id, None)

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()


_browser_relay_registry: BrowserRelayRegistry | None = None


def get_browser_relay_registry() -> BrowserRelayRegistry:
    global _browser_relay_registry
    if _browser_relay_registry is None:
        _browser_relay_registry = BrowserRelayRegistry()
    return _browser_relay_registry


def reset_browser_relay_registry() -> None:
    """Reset the process singleton for isolated tests."""
    global _browser_relay_registry
    _browser_relay_registry = None
