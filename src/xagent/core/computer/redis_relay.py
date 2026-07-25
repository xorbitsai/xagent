from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import RedisError

from .relay import (
    BROWSER_RELAY_DEFAULT_REQUEST_TIMEOUT_SECONDS,
    BROWSER_RELAY_PROTOCOL_VERSION,
    BrowserRelayAuthentication,
    BrowserRelayAuthenticationError,
    BrowserRelayCommandConnection,
    BrowserRelayConnection,
    BrowserRelayError,
    BrowserRelayHello,
    BrowserRelayInUseError,
    BrowserRelayPairing,
    BrowserRelayProtocolError,
    BrowserRelayStatusMessage,
    BrowserRelayUnavailableError,
)

logger = logging.getLogger(__name__)

_DEFAULT_NAMESPACE = "xagent:browser-relay"
_CONNECTION_TTL_SECONDS = 60

_PAIR_AND_CREATE_SESSION_SCRIPT = """
local pairing = redis.call("GET", KEYS[1])
if not pairing then
  return nil
end
local ok, decoded = pcall(cjson.decode, pairing)
if not ok or not decoded["user_id"] then
  return redis.error_reply("invalid browser relay pairing state")
end
local user_id = tostring(decoded["user_id"])
local pairing_index = ARGV[2] .. user_id .. ":pairing"
local session_key = ARGV[3] .. ARGV[4]
local session_index = ARGV[2] .. user_id .. ":sessions"
redis.call("DEL", KEYS[1])
if redis.call("GET", pairing_index) == ARGV[1] then
  redis.call("DEL", pairing_index)
end
local old_sessions = redis.call("SMEMBERS", session_index)
for _, digest in ipairs(old_sessions) do
  redis.call("DEL", ARGV[3] .. digest)
end
redis.call("DEL", session_index)
redis.call(
  "SET",
  session_key,
  cjson.encode({
    user_id = decoded["user_id"],
    client_id = ARGV[5],
    client_name = ARGV[6]
  }),
  "EX",
  ARGV[7]
)
redis.call("SADD", session_index, ARGV[4])
redis.call("EXPIRE", session_index, ARGV[7])
return pairing
"""

_CREATE_PAIRING_SCRIPT = """
local old_digest = redis.call("GET", KEYS[1])
if old_digest then
  redis.call("DEL", ARGV[1] .. old_digest)
end
redis.call("SET", KEYS[2], ARGV[3], "EX", ARGV[4])
redis.call("SET", KEYS[1], ARGV[2], "EX", ARGV[4])
return 1
"""

_DELETE_IF_VALUE_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("DEL", KEYS[1])
end
return 0
"""

_EXCHANGE_CONNECTION_SCRIPT = """
if ARGV[3] == "1" then
  local session = redis.call("GET", KEYS[2])
  if not session then
    return "__XAGENT_RELAY_AUTH_MISSING__"
  end
  local ok, decoded = pcall(cjson.decode, session)
  if not ok or tostring(decoded["user_id"]) ~= ARGV[4] then
    return "__XAGENT_RELAY_AUTH_MISSING__"
  end
end
local old_value = redis.call("GET", KEYS[1])
redis.call("SET", KEYS[1], ARGV[1], "EX", ARGV[2])
return old_value
"""

_SET_CONNECTION_IF_CURRENT_SCRIPT = """
local current = redis.call("GET", KEYS[1])
if not current then
  return 0
end
local ok, decoded = pcall(cjson.decode, current)
if not ok or decoded["connection_id"] ~= ARGV[1] then
  return 0
end
redis.call("SET", KEYS[1], ARGV[2], "EX", ARGV[3])
return 1
"""

_TOUCH_CONNECTION_IF_CURRENT_SCRIPT = """
local current = redis.call("GET", KEYS[1])
if not current then
  return 0
end
local ok, decoded = pcall(cjson.decode, current)
if not ok or decoded["connection_id"] ~= ARGV[1] then
  return 0
end
redis.call("EXPIRE", KEYS[1], ARGV[2])
return 1
"""

_DELETE_CONNECTION_IF_CURRENT_SCRIPT = """
local current = redis.call("GET", KEYS[1])
if not current then
  return 0
end
local ok, decoded = pcall(cjson.decode, current)
if not ok or decoded["connection_id"] ~= ARGV[1] then
  return 0
end
return redis.call("DEL", KEYS[1])
"""

_ACQUIRE_CLAIM_SCRIPT = """
local current = redis.call("GET", KEYS[1])
if not current or current == ARGV[1] then
  redis.call("SET", KEYS[1], ARGV[1], "EX", ARGV[2])
  return ARGV[1]
end
return current
"""

_TOUCH_CLAIM_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  redis.call("EXPIRE", KEYS[1], ARGV[2])
  return 1
end
return 0
"""

_PUBLISH_IF_CURRENT_SCRIPT = """
local current = redis.call("GET", KEYS[1])
if not current then
  return 0
end
local ok, decoded = pcall(cjson.decode, current)
if not ok or decoded["connection_id"] ~= ARGV[1] then
  return 0
end
if not decoded["attached"] then
  return 2
end
if redis.call("GET", KEYS[3]) ~= ARGV[3] then
  return 3
end
local receivers = redis.call("PUBLISH", KEYS[2], ARGV[2])
if receivers == 0 then
  return 4
end
redis.call("EXPIRE", KEYS[3], ARGV[4])
return 1
"""

_REVOKE_USER_SCRIPT = """
local pairing_digest = redis.call("GET", KEYS[1])
if pairing_digest then
  redis.call("DEL", ARGV[1] .. pairing_digest)
end
local sessions = redis.call("SMEMBERS", KEYS[2])
for _, digest in ipairs(sessions) do
  redis.call("DEL", ARGV[2] .. digest)
end
local connection = redis.call("GET", KEYS[3])
redis.call("DEL", KEYS[1], KEYS[2], KEYS[3], KEYS[4])
return connection
"""


@dataclass
class _GatewayState:
    connection: BrowserRelayConnection
    connection_id: str
    command_channel: str
    pubsub: Any
    listener_task: asyncio.Task[None] | None = None
    request_tasks: set[asyncio.Task[None]] = field(default_factory=set)


class _RedisRelayCommandConnection(BrowserRelayCommandConnection):
    def __init__(
        self,
        *,
        registry: RedisBrowserRelayRegistry,
        user_id: int,
        owner_task_id: str,
        connection_id: str,
    ) -> None:
        self._registry = registry
        self._user_id = user_id
        self._owner_task_id = owner_task_id
        self._connection_id = connection_id

    async def request(
        self,
        command: str,
        payload: dict[str, Any],
        *,
        timeout: float = BROWSER_RELAY_DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        return await self._registry.request(
            user_id=self._user_id,
            owner_task_id=self._owner_task_id,
            connection_id=self._connection_id,
            command=command,
            payload=payload,
            timeout=timeout,
        )


class RedisBrowserRelayRegistry:
    """Redis-coordinated relay for multi-process and multi-replica Xagent."""

    def __init__(
        self,
        redis_url: str,
        *,
        namespace: str = _DEFAULT_NAMESPACE,
        pairing_ttl: timedelta = timedelta(minutes=10),
        session_ttl: timedelta = timedelta(days=7),
        claim_ttl: timedelta = timedelta(minutes=30),
        connection_ttl_seconds: int = _CONNECTION_TTL_SECONDS,
        redis_client: Any | None = None,
    ) -> None:
        if not redis_url.strip():
            raise ValueError("Redis browser relay requires a non-empty Redis URL.")
        self._namespace = namespace.strip().rstrip(":")
        if not self._namespace:
            raise ValueError("Redis browser relay namespace must not be empty.")
        self._pairing_ttl_seconds = self._ttl_seconds(pairing_ttl)
        self._session_ttl_seconds = self._ttl_seconds(session_ttl)
        self._claim_ttl_seconds = self._ttl_seconds(claim_ttl)
        self._connection_ttl_seconds = max(10, connection_ttl_seconds)
        self._instance_id = uuid4().hex
        self._redis: Any = (
            redis_client
            if redis_client is not None
            else Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                health_check_interval=30,
            )
        )
        self._local_gateways: dict[int, _GatewayState] = {}
        self._local_lock = asyncio.Lock()

    async def create_pairing(self, user_id: int) -> BrowserRelayPairing:
        if user_id <= 0:
            raise ValueError("browser relay pairing requires an authenticated user")
        token = secrets.token_urlsafe(32)
        digest = self._digest(token)
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=self._pairing_ttl_seconds
        )
        record = self._dump_json(
            {
                "user_id": user_id,
                "expires_at": expires_at.isoformat(),
            }
        )
        await self._redis.eval(
            _CREATE_PAIRING_SCRIPT,
            2,
            self._pairing_index_key(user_id),
            self._pairing_key(digest),
            self._pairing_prefix(),
            digest,
            record,
            self._pairing_ttl_seconds,
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
        if hello.pairing_token is not None:
            raw_token = hello.pairing_token.get_secret_value()
            digest = self._digest(raw_token)
            session_token = secrets.token_urlsafe(32)
            session_digest = self._digest(session_token)
            raw_record = await self._redis.eval(
                _PAIR_AND_CREATE_SESSION_SCRIPT,
                1,
                self._pairing_key(digest),
                digest,
                self._key("user:"),
                self._session_prefix(),
                session_digest,
                hello.client_id,
                hello.client_name,
                self._session_ttl_seconds,
            )
            if not isinstance(raw_record, str):
                raise BrowserRelayAuthenticationError(
                    "Pairing token is invalid, expired, or already used."
                )
            record = self._load_json(raw_record)
            user_id = self._required_positive_int(record.get("user_id"), "user_id")
            return BrowserRelayAuthentication(
                user_id=user_id,
                client_id=hello.client_id,
                client_name=hello.client_name,
                session_token=session_token,
                session_id=session_digest,
                paired=True,
            )

        assert hello.session_token is not None
        digest = self._digest(hello.session_token.get_secret_value())
        raw_record = await self._redis.get(self._session_key(digest))
        if not isinstance(raw_record, str):
            raise BrowserRelayAuthenticationError(
                "Browser relay session is invalid or expired."
            )
        record = self._load_json(raw_record)
        if record.get("client_id") != hello.client_id:
            raise BrowserRelayAuthenticationError(
                "Browser relay session is invalid or expired."
            )
        return BrowserRelayAuthentication(
            user_id=self._required_positive_int(record.get("user_id"), "user_id"),
            client_id=hello.client_id,
            client_name=str(record.get("client_name") or hello.client_name),
            session_token=None,
            session_id=digest,
            paired=False,
        )

    async def register(self, connection: BrowserRelayConnection) -> None:
        connection_id = uuid4().hex
        command_channel = self._command_channel(connection_id)
        pubsub = self._redis.pubsub()
        try:
            await self._subscribe(pubsub, command_channel)
        except asyncio.CancelledError:
            await pubsub.aclose()
            raise
        except Exception:
            await pubsub.aclose()
            raise
        state = _GatewayState(
            connection=connection,
            connection_id=connection_id,
            command_channel=command_channel,
            pubsub=pubsub,
        )
        state.listener_task = asyncio.create_task(
            self._listen_for_commands(state),
            name=f"browser-relay-{connection.user_id}-{connection_id[:8]}",
        )
        await asyncio.sleep(0)

        async with self._local_lock:
            previous_state = self._local_gateways.get(connection.user_id)
            self._local_gateways[connection.user_id] = state
        try:
            old_raw = await self._redis.eval(
                _EXCHANGE_CONNECTION_SCRIPT,
                2,
                self._connection_key(connection.user_id),
                self._session_key(connection.authorization_id or "__direct__"),
                self._connection_record(state),
                self._connection_ttl_seconds,
                "1" if connection.authorization_id is not None else "0",
                str(connection.user_id),
            )
            if old_raw == "__XAGENT_RELAY_AUTH_MISSING__":
                raise BrowserRelayAuthenticationError(
                    "Browser relay session was revoked before connection."
                )
        except (asyncio.CancelledError, Exception):
            async with self._local_lock:
                if self._local_gateways.get(connection.user_id) is state:
                    if previous_state is None:
                        self._local_gateways.pop(connection.user_id, None)
                    else:
                        self._local_gateways[connection.user_id] = previous_state
            await self._discard_gateway(state)
            raise

        old_status: dict[str, Any] | None = None
        if isinstance(old_raw, str):
            old_status = self._load_json(old_raw)
        previous_was_current = (
            previous_state is not None
            and old_status is not None
            and old_status.get("connection_id") == previous_state.connection_id
        )
        if previous_state is not None:
            if previous_state.connection is not connection:
                await previous_state.connection.close(
                    code=4001,
                    reason="A newer browser relay connection replaced this one.",
                )
            await self._discard_gateway(previous_state)
        if (
            old_status is not None
            and old_status.get("connection_id") != connection_id
            and not previous_was_current
        ):
            await self._send_close(
                old_status,
                code=4001,
                reason="A newer browser relay connection replaced this one.",
            )

    async def unregister(self, connection: BrowserRelayConnection) -> None:
        async with self._local_lock:
            state = self._local_gateways.get(connection.user_id)
            if state is not None and state.connection is connection:
                self._local_gateways.pop(connection.user_id, None)
            else:
                state = None
        try:
            if state is not None:
                await self._redis.eval(
                    _DELETE_CONNECTION_IF_CURRENT_SCRIPT,
                    1,
                    self._connection_key(connection.user_id),
                    state.connection_id,
                )
        finally:
            if state is not None:
                await self._discard_gateway(state)
            await connection.close()

    async def update_connection_status(
        self,
        connection: BrowserRelayConnection,
        status: BrowserRelayStatusMessage,
    ) -> None:
        state = await self._local_state(connection)
        if state is None:
            return
        connection.update_status(status)
        updated = await self._redis.eval(
            _SET_CONNECTION_IF_CURRENT_SCRIPT,
            1,
            self._connection_key(connection.user_id),
            state.connection_id,
            self._connection_record(state),
            self._connection_ttl_seconds,
        )
        if updated != 1:
            await connection.close(
                code=4001,
                reason="A newer browser relay connection replaced this one.",
            )

    async def touch_connection(self, connection: BrowserRelayConnection) -> None:
        state = await self._local_state(connection)
        if state is None:
            return
        touched = await self._redis.eval(
            _TOUCH_CONNECTION_IF_CURRENT_SCRIPT,
            1,
            self._connection_key(connection.user_id),
            state.connection_id,
            self._connection_ttl_seconds,
        )
        if touched != 1:
            await connection.close(
                code=4001,
                reason="Browser relay connection is no longer current.",
            )

    async def acquire(
        self,
        *,
        user_id: int,
        owner_task_id: str,
    ) -> BrowserRelayCommandConnection:
        owner = owner_task_id.strip()
        if user_id <= 0 or not owner:
            raise ValueError("browser relay requires authenticated user and task owner")
        status = await self._read_connection_status(user_id)
        if status is None:
            raise BrowserRelayUnavailableError(
                "Browser extension is not connected for this user."
            )
        current_owner = await self._redis.eval(
            _ACQUIRE_CLAIM_SCRIPT,
            1,
            self._claim_key(user_id),
            owner,
            self._claim_ttl_seconds,
        )
        if current_owner != owner:
            raise BrowserRelayInUseError(
                "The user browser is already controlled by another task."
            )
        connection_id = self._required_string(
            status.get("connection_id"),
            "connection_id",
        )
        return _RedisRelayCommandConnection(
            registry=self,
            user_id=user_id,
            owner_task_id=owner,
            connection_id=connection_id,
        )

    async def touch_claim(self, *, user_id: int, owner_task_id: str) -> None:
        await self._redis.eval(
            _TOUCH_CLAIM_SCRIPT,
            1,
            self._claim_key(user_id),
            owner_task_id,
            self._claim_ttl_seconds,
        )

    async def release(self, *, user_id: int, owner_task_id: str) -> None:
        await self._redis.eval(
            _DELETE_IF_VALUE_SCRIPT,
            1,
            self._claim_key(user_id),
            owner_task_id,
        )

    async def status(self, user_id: int) -> dict[str, Any]:
        status = await self._read_connection_status(user_id)
        if status is None:
            return {"connected": False, "attached": False}
        return {
            key: status.get(key)
            for key in (
                "connected",
                "client_id",
                "client_name",
                "attached",
                "tab_id",
                "title",
                "url",
                "connected_at",
            )
        }

    async def revoke_user(self, user_id: int) -> None:
        raw_connection = await self._redis.eval(
            _REVOKE_USER_SCRIPT,
            4,
            self._pairing_index_key(user_id),
            self._session_index_key(user_id),
            self._connection_key(user_id),
            self._claim_key(user_id),
            self._pairing_prefix(),
            self._session_prefix(),
        )
        if isinstance(raw_connection, str):
            await self._send_close(
                self._load_json(raw_connection),
                code=4002,
                reason="Browser relay access was revoked.",
            )

    async def request(
        self,
        *,
        user_id: int,
        owner_task_id: str,
        connection_id: str,
        command: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        request_id = secrets.token_urlsafe(18)
        response_channel = self._response_channel(request_id)
        command_channel = self._command_channel(connection_id)
        response_pubsub = self._redis.pubsub()
        raw_response: str | None = None
        command_data = self._dump_json(
            {
                "kind": "command",
                "protocol_version": BROWSER_RELAY_PROTOCOL_VERSION,
                "request_id": request_id,
                "connection_id": connection_id,
                "command": command,
                "payload": payload,
                "timeout": timeout,
                "response_channel": response_channel,
            }
        )
        try:
            await self._subscribe(response_pubsub, response_channel)
            publish_result = await self._redis.eval(
                _PUBLISH_IF_CURRENT_SCRIPT,
                3,
                self._connection_key(user_id),
                command_channel,
                self._claim_key(user_id),
                connection_id,
                command_data,
                owner_task_id,
                self._claim_ttl_seconds,
            )
            if publish_result == 2:
                raise BrowserRelayUnavailableError(
                    "No browser tab is attached. Ask the user to open the Xagent "
                    "extension and approve the current tab."
                )
            if publish_result != 1:
                if publish_result == 3:
                    raise BrowserRelayInUseError(
                        "The user browser is no longer controlled by this task."
                    )
                raise BrowserRelayUnavailableError(
                    "Browser extension disconnected or reconnected. Request a fresh "
                    "screenshot before continuing."
                )
            deadline = asyncio.get_running_loop().time() + timeout
            while raw_response is None:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                message = await response_pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=min(1.0, remaining),
                )
                if message is not None and isinstance(message.get("data"), str):
                    raw_response = message["data"]
        finally:
            try:
                await response_pubsub.unsubscribe(response_channel)
            except RedisError:
                logger.debug(
                    "Could not unsubscribe browser response channel",
                    exc_info=True,
                )
            finally:
                await response_pubsub.aclose()
        if raw_response is None:
            raise BrowserRelayUnavailableError(
                f"Browser extension did not answer {command!r} in time."
            )
        decoded = self._load_json(raw_response)
        if decoded.get("success") is True:
            result = decoded.get("result")
            if not isinstance(result, dict):
                raise BrowserRelayProtocolError(
                    "Browser relay response result must be an object."
                )
            return result
        error = str(decoded.get("error") or "Browser extension command failed.")
        if decoded.get("error_type") == "unavailable":
            raise BrowserRelayUnavailableError(error)
        raise BrowserRelayError(error)

    async def aclose(self) -> None:
        async with self._local_lock:
            states = list(self._local_gateways.values())
            self._local_gateways.clear()
        for state in states:
            await self._discard_gateway(state)
        await self._redis.aclose()

    async def _listen_for_commands(self, state: _GatewayState) -> None:
        while not state.connection.is_closed:
            try:
                incoming = await state.pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if incoming is None:
                    continue
                raw_data = incoming.get("data")
                if not isinstance(raw_data, str):
                    continue
                message = self._load_json(raw_data)
                if message.get("connection_id") != state.connection_id:
                    continue
                if message.get("kind") == "close":
                    await state.connection.close(
                        code=int(message.get("code") or 4001),
                        reason=str(
                            message.get("reason") or "Browser relay connection closed."
                        ),
                    )
                    return
                task = asyncio.create_task(self._handle_gateway_command(state, message))
                state.request_tasks.add(task)
                task.add_done_callback(partial(self._gateway_task_done, state))
            except asyncio.CancelledError:
                raise
            except RedisError as exc:
                logger.warning("Browser relay Redis command listener failed: %s", exc)
                await asyncio.sleep(0.5)
            except Exception:
                logger.exception("Browser relay command listener rejected a message")

    async def _handle_gateway_command(
        self,
        state: _GatewayState,
        message: dict[str, Any],
    ) -> None:
        response_channel = message.get("response_channel")
        request_id = message.get("request_id")
        if not isinstance(response_channel, str) or not isinstance(request_id, str):
            return
        try:
            command = self._required_string(message.get("command"), "command")
            payload = message.get("payload")
            if not isinstance(payload, dict):
                raise BrowserRelayProtocolError(
                    "Distributed browser command payload must be an object."
                )
            timeout = float(message.get("timeout") or 30.0)
            result = await state.connection.request(
                command,
                payload,
                timeout=max(0.1, min(timeout, 120.0)),
            )
            response = {
                "request_id": request_id,
                "success": True,
                "result": result,
            }
        except BrowserRelayUnavailableError as exc:
            response = {
                "request_id": request_id,
                "success": False,
                "error_type": "unavailable",
                "error": str(exc),
            }
        except BrowserRelayError as exc:
            response = {
                "request_id": request_id,
                "success": False,
                "error_type": "relay",
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - remote failures become responses.
            logger.exception("Distributed browser relay command failed")
            response = {
                "request_id": request_id,
                "success": False,
                "error_type": "relay",
                "error": str(exc)[:2_000],
            }
        try:
            await self._redis.publish(
                response_channel,
                self._dump_json(response),
            )
        except RedisError:
            logger.exception("Could not publish distributed browser response")

    async def _send_close(
        self,
        status: dict[str, Any],
        *,
        code: int,
        reason: str,
    ) -> None:
        connection_id = status.get("connection_id")
        command_channel = status.get("command_channel")
        if not isinstance(connection_id, str) or not isinstance(command_channel, str):
            return
        try:
            await self._redis.publish(
                command_channel,
                self._dump_json(
                    {
                        "kind": "close",
                        "connection_id": connection_id,
                        "code": code,
                        "reason": reason,
                    }
                ),
            )
        except RedisError:
            logger.warning(
                "Could not notify an old browser relay connection to close",
                exc_info=True,
            )

    async def _discard_gateway(self, state: _GatewayState) -> None:
        current = asyncio.current_task()
        tasks = [
            task
            for task in [state.listener_task, *state.request_tasks]
            if task is not None and task is not current
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await state.pubsub.unsubscribe(state.command_channel)
        except RedisError:
            logger.debug(
                "Could not unsubscribe browser command channel",
                exc_info=True,
            )
        finally:
            await state.pubsub.aclose()

    async def _subscribe(self, pubsub: Any, channel: str) -> None:
        await pubsub.subscribe(channel)
        deadline = asyncio.get_running_loop().time() + 2
        while asyncio.get_running_loop().time() < deadline:
            message = await pubsub.get_message(timeout=0.2)
            if (
                message is not None
                and message.get("type") == "subscribe"
                and message.get("channel") == channel
            ):
                return
        raise BrowserRelayUnavailableError(
            "Redis did not confirm the browser relay subscription."
        )

    @staticmethod
    def _gateway_task_done(
        state: _GatewayState,
        task: asyncio.Task[None],
    ) -> None:
        state.request_tasks.discard(task)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.error(
                "Browser relay gateway task failed",
                exc_info=(
                    type(exception),
                    exception,
                    exception.__traceback__,
                ),
            )

    async def _local_state(
        self,
        connection: BrowserRelayConnection,
    ) -> _GatewayState | None:
        async with self._local_lock:
            state = self._local_gateways.get(connection.user_id)
            if state is None or state.connection is not connection:
                return None
            return state

    async def _read_connection_status(
        self,
        user_id: int,
    ) -> dict[str, Any] | None:
        raw_status = await self._redis.get(self._connection_key(user_id))
        if not isinstance(raw_status, str):
            return None
        status = self._load_json(raw_status)
        if status.get("connected") is not True:
            return None
        return status

    def _connection_record(self, state: _GatewayState) -> str:
        public_status = state.connection.public_status()
        # URLs and page titles may contain sensitive query parameters or user
        # content. Keep only routing/liveness metadata in durable Redis keys.
        public_status["title"] = None
        public_status["url"] = None
        return self._dump_json(
            {
                **public_status,
                "connection_id": state.connection_id,
                "gateway_id": self._instance_id,
                "command_channel": state.command_channel,
            }
        )

    def _key(self, suffix: str) -> str:
        return f"{self._namespace}:{suffix}"

    def _pairing_prefix(self) -> str:
        return self._key("pairing:")

    def _pairing_key(self, digest: str) -> str:
        return f"{self._pairing_prefix()}{digest}"

    def _pairing_index_key(self, user_id: int) -> str:
        return self._key(f"user:{user_id}:pairing")

    def _session_prefix(self) -> str:
        return self._key("session:")

    def _session_key(self, digest: str) -> str:
        return f"{self._session_prefix()}{digest}"

    def _session_index_key(self, user_id: int) -> str:
        return self._key(f"user:{user_id}:sessions")

    def _connection_key(self, user_id: int) -> str:
        return self._key(f"user:{user_id}:connection")

    def _claim_key(self, user_id: int) -> str:
        return self._key(f"user:{user_id}:claim")

    def _command_channel(self, connection_id: str) -> str:
        return self._key(f"commands:{connection_id}")

    def _response_channel(self, request_id: str) -> str:
        return self._key(f"responses:{request_id}")

    @staticmethod
    def _dump_json(value: dict[str, Any]) -> str:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _load_json(raw_value: str) -> dict[str, Any]:
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise BrowserRelayProtocolError(
                "Browser relay Redis state is invalid."
            ) from exc
        if not isinstance(value, dict):
            raise BrowserRelayProtocolError(
                "Browser relay Redis state must be an object."
            )
        return value

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _ttl_seconds(value: timedelta) -> int:
        return max(1, math.ceil(value.total_seconds()))

    @staticmethod
    def _required_positive_int(value: Any, field: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise BrowserRelayProtocolError(
                f"Browser relay {field} is invalid."
            ) from exc
        if parsed <= 0:
            raise BrowserRelayProtocolError(f"Browser relay {field} is invalid.")
        return parsed

    @staticmethod
    def _required_string(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise BrowserRelayProtocolError(f"Browser relay {field} is invalid.")
        return value
