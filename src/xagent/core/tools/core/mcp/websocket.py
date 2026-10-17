"""Public-only legacy MCP WebSocket transport.

The SDK's legacy helper doesn't accept connection options. Keep its JSON-RPC
stream protocol, but pin the socket and reject handshake redirects/proxies.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage
from pydantic import ValidationError
from websockets.asyncio.client import connect
from websockets.typing import Subprotocol

from ....utils.security import build_ca_bundle_ssl_context, validate_public_http_url


class _NoRedirectConnect(connect):
    def process_redirect(self, exc: Exception) -> Exception | str:
        # A new URI must never inherit the old socket override or credential
        # query. Operators must configure the actual final WebSocket URL.
        return exc


@asynccontextmanager
async def public_websocket_client(
    url: str,
) -> AsyncIterator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ]
]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError("MCP WebSocket URL must use ws or wss")
    http_url = urlunsplit(
        parsed._replace(scheme="https" if parsed.scheme == "wss" else "http")
    )
    addresses = await validate_public_http_url(http_url)
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    read_writer, read_stream = anyio.create_memory_object_stream[
        SessionMessage | Exception
    ](0)
    write_stream, write_reader = anyio.create_memory_object_stream[SessionMessage](0)
    tls_kwargs: dict[str, Any] = (
        {"ssl": build_ca_bundle_ssl_context(), "server_hostname": parsed.hostname}
        if parsed.scheme == "wss"
        else {}
    )
    async with (
        read_writer,
        read_stream,
        write_stream,
        write_reader,
        _NoRedirectConnect(
            url,
            host=addresses[0],
            port=port,
            proxy=None,
            subprotocols=[Subprotocol("mcp")],
            **tls_kwargs,
        ) as websocket,
    ):

        async def reader() -> None:
            async for raw in websocket:
                try:
                    await read_writer.send(
                        SessionMessage(JSONRPCMessage.model_validate_json(raw))
                    )
                except ValidationError as exc:
                    await read_writer.send(exc)

        async def writer() -> None:
            async for message in write_reader:
                await websocket.send(
                    message.message.model_dump_json(by_alias=True, exclude_none=True)
                )

        async with anyio.create_task_group() as group:
            group.start_soon(reader)
            group.start_soon(writer)
            try:
                yield read_stream, write_stream
            finally:
                group.cancel_scope.cancel()
