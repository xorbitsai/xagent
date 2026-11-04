"""Public-only legacy MCP WebSocket transport.

The SDK's legacy helper doesn't accept connection options. Keep its JSON-RPC
stream protocol, but pin the socket and reject handshake redirects/proxies.
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage
from pydantic import ValidationError
from websockets.asyncio.client import connect
from websockets.typing import Subprotocol
from yarl import URL

from ....utils.public_network_proxy import PublicNetworkEgressProxy
from ....utils.security import (
    build_ca_bundle_ssl_context,
    public_http_destination,
    validate_public_http_url,
)
from ..web_content import get_trusted_proxy_url

WEBSOCKET_CONNECT_TIMEOUT = 10.0


class _NoRedirectConnect(connect):
    def process_redirect(self, exc: Exception) -> Exception | str:
        # A new URI must never inherit the old socket override or credential
        # query. Operators must configure the actual final WebSocket URL.
        return exc


@asynccontextmanager
async def _connect_public(
    url: str,
    addresses: list[str],
    port: int,
    tls_kwargs: dict[str, Any],
    *,
    deadline: float,
) -> AsyncIterator[Any]:
    proxy = get_trusted_proxy_url()
    async with AsyncExitStack() as stack:
        remaining = deadline - anyio.current_time()
        if remaining <= 0:
            raise TimeoutError("MCP WebSocket connect budget exhausted")
        # Bound gate startup and all handshakes, but not the live session.
        with anyio.fail_after(remaining):
            options: dict[str, Any] = {}
            targets: list[str | None]
            if proxy:
                # Reserve a per-address slice so a black-holed first IP
                # cannot consume the whole upstream gate handshake budget.
                gate = PublicNetworkEgressProxy(
                    upstream_proxy_url=proxy,
                    connect_timeout=remaining / max(1, len(addresses)),
                )
                stack.push_async_callback(gate.close)
                options["proxy"] = await gate.start()
                targets = [None]
            else:
                options["proxy"] = None
                targets = list(addresses)
            for index, address in enumerate(targets):
                remaining = deadline - anyio.current_time()
                if remaining <= 0:
                    raise TimeoutError("MCP WebSocket connect budget exhausted")
                if address is not None:
                    options.update(host=address, port=port)
                try:
                    websocket = await stack.enter_async_context(
                        _NoRedirectConnect(
                            url,
                            open_timeout=remaining / (len(targets) - index),
                            subprotocols=[Subprotocol("mcp")],
                            **options,
                            **tls_kwargs,
                        )
                    )
                    break
                except (OSError, TimeoutError):
                    if index == len(targets) - 1:
                        raise
        yield websocket


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
    hostname, port = public_http_destination(http_url)
    # websockets uses stdlib IDNA2003 for an IRI, whereas our DNS validator
    # uses IDNA/UTS46. Canonicalize before the handshake so Host, SNI, and
    # proxy CONNECT all refer to the same checked name (e.g. faß != fass).
    url = str(URL(url))
    deadline = anyio.current_time() + WEBSOCKET_CONNECT_TIMEOUT
    with anyio.fail_after(WEBSOCKET_CONNECT_TIMEOUT):
        addresses = await validate_public_http_url(http_url)
    read_writer, read_stream = anyio.create_memory_object_stream[
        SessionMessage | Exception
    ](0)
    write_stream, write_reader = anyio.create_memory_object_stream[SessionMessage](0)
    tls_kwargs: dict[str, Any] = (
        {"ssl": build_ca_bundle_ssl_context(), "server_hostname": hostname}
        if parsed.scheme == "wss"
        else {}
    )
    async with (
        read_writer,
        read_stream,
        write_stream,
        write_reader,
        _connect_public(
            url, addresses, port, tls_kwargs, deadline=deadline
        ) as websocket,
    ):

        async def reader() -> None:
            # EOF must close the incoming memory stream immediately, rather
            # than waiting for the outer context: ClientSession needs that
            # EOF to stop waiting for a response from a disconnected peer.
            async with read_writer:
                async for raw in websocket:
                    try:
                        await read_writer.send(
                            SessionMessage(JSONRPCMessage.model_validate_json(raw))
                        )
                    except ValidationError as exc:
                        await read_writer.send(exc)

        async def writer() -> None:
            async with write_reader:
                async for message in write_reader:
                    await websocket.send(
                        message.message.model_dump_json(
                            by_alias=True, exclude_none=True
                        )
                    )

        async with anyio.create_task_group() as group:
            group.start_soon(reader)
            group.start_soon(writer)
            try:
                yield read_stream, write_stream
            finally:
                group.cancel_scope.cancel()
