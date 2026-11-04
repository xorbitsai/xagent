from __future__ import annotations

import asyncio
import base64
import binascii
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ImageContent, TextContent

from ...config import (
    get_browser_cua_driver_command,
    get_browser_cua_driver_socket,
    get_browser_cua_driver_timeout_seconds,
)


class CuaDriverError(RuntimeError):
    """Raised when the local cua-driver process cannot satisfy a tool call."""


@dataclass(frozen=True)
class CuaDriverResult:
    """Provider-neutral subset of one cua-driver MCP tool response."""

    structured: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    image_bytes: bytes | None = None
    image_mime_type: str | None = None


class CuaDriverClientProtocol(Protocol):
    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> CuaDriverResult: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class _ToolCallRequest:
    name: str
    arguments: dict[str, Any]
    future: asyncio.Future[CallToolResult]


class CuaDriverMCPClient:
    """Lazy, task-scoped stdio MCP client for the native cua-driver runtime."""

    def __init__(
        self,
        *,
        command: str | None = None,
        socket: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.command = command or get_browser_cua_driver_command()
        self.socket = (
            get_browser_cua_driver_socket()
            if socket is None
            else socket.strip() or None
        )
        self.timeout_seconds = (
            get_browser_cua_driver_timeout_seconds()
            if timeout_seconds is None
            else timeout_seconds
        )
        if not self.command.strip():
            raise ValueError("cua-driver command must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("cua-driver timeout must be positive")
        self._worker: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[_ToolCallRequest | None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        # Sessions are process-local inside cua-driver. Keep only successfully
        # established registrations so a replacement worker can restore their
        # capture scope before it accepts another tool call.
        self._active_sessions: dict[str, dict[str, Any]] = {}

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> CuaDriverResult:
        if not name.strip():
            raise ValueError("cua-driver tool name must not be empty")
        queue = await self._ensure_worker()
        future: asyncio.Future[CallToolResult] = (
            asyncio.get_running_loop().create_future()
        )
        await queue.put(
            _ToolCallRequest(
                name=name,
                arguments=dict(arguments or {}),
                future=future,
            )
        )
        try:
            response = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            raise CuaDriverError(
                f"cua-driver tool {name!r} timed out after "
                f"{self.timeout_seconds:g} seconds"
            ) from exc
        except Exception as exc:
            raise CuaDriverError(f"cua-driver tool {name!r} failed: {exc}") from exc
        finally:
            if not future.done():
                future.cancel()

        text_parts = [
            item.text for item in response.content if isinstance(item, TextContent)
        ]
        image_parts = [
            item for item in response.content if isinstance(item, ImageContent)
        ]
        message = "\n".join(text_parts).strip()
        if response.isError:
            raise CuaDriverError(message or f"cua-driver tool {name!r} failed")

        await self._record_session_call(name, dict(arguments or {}))

        image_bytes: bytes | None = None
        image_mime_type: str | None = None
        if image_parts:
            image = image_parts[0]
            try:
                image_bytes = base64.b64decode(image.data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise CuaDriverError(
                    f"cua-driver tool {name!r} returned invalid image data"
                ) from exc
            image_mime_type = image.mimeType

        structured = response.structuredContent
        return CuaDriverResult(
            structured=dict(structured) if isinstance(structured, dict) else {},
            text=message,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        )

    async def close(self) -> None:
        await self._stop_worker(clear_sessions=True)

    async def _stop_worker(self, *, clear_sessions: bool) -> None:
        async with self._lifecycle_lock:
            worker = self._worker
            queue = self._queue
            self._worker = None
            self._queue = None
            self._ready = None
            if clear_sessions:
                self._active_sessions.clear()
        if worker is None:
            return
        if queue is not None and not worker.done():
            await queue.put(None)
        try:
            await asyncio.wait_for(worker, timeout=self.timeout_seconds)
        except TimeoutError:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        except Exception:
            # Callers already receive the specific transport failure from their
            # request future. Teardown remains best-effort and idempotent.
            pass

    async def _ensure_worker(
        self,
    ) -> asyncio.Queue[_ToolCallRequest | None]:
        async with self._lifecycle_lock:
            if self._worker is None or self._worker.done():
                loop = asyncio.get_running_loop()
                self._queue = asyncio.Queue()
                self._ready = loop.create_future()
                self._worker = asyncio.create_task(
                    self._run_worker(self._queue, self._ready),
                    name="xagent-cua-driver-mcp",
                )
            queue = self._queue
            ready = self._ready
        assert queue is not None and ready is not None
        try:
            await asyncio.wait_for(
                asyncio.shield(ready),
                timeout=self.timeout_seconds,
            )
        except BaseException as exc:
            # Initialization owns a partially started worker/subprocess. Reset
            # it before propagating timeouts, transport failures, or caller
            # cancellation so a later call can create a fresh worker.
            # Keep registrations across an initialization failure. A later
            # replacement must either restore every session or fail closed;
            # silently forgetting the window-scoped session is unsafe.
            await self._stop_worker(clear_sessions=False)
            if isinstance(exc, TimeoutError):
                raise CuaDriverError(
                    "cua-driver MCP initialization timed out after "
                    f"{self.timeout_seconds:g} seconds"
                ) from exc
            if isinstance(exc, Exception) and not isinstance(exc, CuaDriverError):
                raise CuaDriverError(
                    f"could not initialize cua-driver MCP: {exc}"
                ) from exc
            raise
        return queue

    async def _run_worker(
        self,
        queue: asyncio.Queue[_ToolCallRequest | None],
        ready: asyncio.Future[None],
    ) -> None:
        args = ["mcp"]
        if self.socket is not None:
            args.extend(["--socket", self.socket])
        parameters = StdioServerParameters(
            command=self.command,
            args=args,
            env={
                # The MCP SDK inherits only a safe environment allowlist. Keep
                # driver telemetry disabled unless the host explicitly opted in.
                "CUA_DRIVER_RS_TELEMETRY_ENABLED": os.getenv(
                    "CUA_DRIVER_RS_TELEMETRY_ENABLED",
                    "0",
                ),
            },
        )
        failure: BaseException | None = None
        try:
            with open(os.devnull, "w", encoding="utf-8") as errlog:
                async with (
                    stdio_client(parameters, errlog=errlog) as (
                        read_stream,
                        write_stream,
                    ),
                    ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
                    ) as session,
                ):
                    await session.initialize()
                    await self._restore_active_sessions(session)
                    if not ready.done():
                        ready.set_result(None)
                    while True:
                        request = await queue.get()
                        if request is None:
                            break
                        if request.future.done():
                            continue
                        try:
                            response = await session.call_tool(
                                request.name,
                                request.arguments,
                            )
                        except Exception as exc:
                            if not request.future.done():
                                request.future.set_exception(exc)
                        else:
                            if not request.future.done():
                                request.future.set_result(response)
        except FileNotFoundError as exc:
            failure = CuaDriverError(
                f"cua-driver executable was not found: {self.command!r}"
            )
            failure.__cause__ = exc
        except asyncio.CancelledError:
            failure = CuaDriverError("cua-driver MCP worker was cancelled")
            raise
        except Exception as exc:
            failure = CuaDriverError(f"could not initialize cua-driver MCP: {exc}")
            failure.__cause__ = exc
        finally:
            if failure is not None and not ready.done():
                ready.set_exception(failure)
            while not queue.empty():
                request = queue.get_nowait()
                if request is not None and not request.future.done():
                    request.future.set_exception(
                        failure or CuaDriverError("cua-driver MCP worker stopped")
                    )

    async def _record_session_call(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        session_id = arguments.get("session")
        if not isinstance(session_id, str) or not session_id:
            return
        async with self._lifecycle_lock:
            if name == "start_session":
                self._active_sessions[session_id] = arguments
            elif name == "end_session":
                self._active_sessions.pop(session_id, None)

    async def _restore_active_sessions(self, session: ClientSession) -> None:
        async with self._lifecycle_lock:
            registrations = tuple(self._active_sessions.values())
        for arguments in registrations:
            response = await session.call_tool("start_session", arguments)
            if response.isError:
                message = "\n".join(
                    item.text
                    for item in response.content
                    if isinstance(item, TextContent)
                ).strip()
                raise CuaDriverError(
                    message or "cua-driver could not restore an active session"
                )
