"""Public-network-only HTTP CONNECT/forward proxy shared by untrusted fetchers."""

import asyncio
import logging
from typing import Any
from urllib.parse import urlsplit

from .security import validate_public_http_url

logger = logging.getLogger(__name__)

_PROXY_HEADER_LIMIT = 64 * 1024
_PROXY_CONNECT_TIMEOUT = 10.0
_PROXY_RESPONSE_TIMEOUT = 10.0
_PROXY_MAX_REQUEST_BODY = 64 * 1024 * 1024


class PublicNetworkEgressProxy:
    """Small loopback proxy that pins each connection to validated DNS IPs.

    Chromium resolves hosts independently of Python.  A browser-wide CONNECT
    proxy closes that TOCTOU gap: the proxy validates the hostname, connects
    directly to one of the validated public IPs, and then tunnels the original
    TLS/WebSocket handshake without changing its Host header or SNI.
    """

    def __init__(self) -> None:
        self._server: asyncio.AbstractServer | None = None
        self._client_tasks: set[asyncio.Task[Any]] = set()

    async def _resolve(self, url: str) -> list[str]:
        return await validate_public_http_url(url)

    async def start(self) -> str:
        self._server = await asyncio.start_server(
            self._handle_client,
            host="127.0.0.1",
            port=0,
            limit=_PROXY_HEADER_LIMIT,
        )
        socket = self._server.sockets[0]
        port = socket.getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def close(self) -> None:
        tasks = list(self._client_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._client_tasks.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
        try:
            header = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), _PROXY_CONNECT_TIMEOUT
            )
            method, target, version, headers = self._parse_request(header)
            if method.upper() == "CONNECT":
                await self._handle_connect(reader, writer, target)
            else:
                await self._handle_forward(
                    reader, writer, method, target, version, headers
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Public-network egress proxy rejected request: %s", exc)
            await self._send_error(writer, 502, "Bad Gateway")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            if task is not None:
                self._client_tasks.discard(task)

    @staticmethod
    def _parse_request(
        header: bytes,
    ) -> tuple[str, str, str, dict[str, str]]:
        lines = header[:-4].split(b"\r\n")
        method, target, version = lines[0].decode("latin-1").split(" ", 2)
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, value = line.decode("latin-1").split(":", 1)
            headers[name.lower().strip()] = value.strip()
        return method, target, version, headers

    async def _handle_connect(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        target: str,
    ) -> None:
        parsed = urlsplit(f"//{target}")
        if (
            parsed.username
            or parsed.password
            or parsed.hostname is None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid CONNECT authority")
        port = parsed.port or 443
        resolved = await self._resolve(f"https://{target}")
        remote_reader, remote_writer = await self._open_validated(resolved, port)
        try:
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            await self._bridge(reader, writer, remote_reader, remote_writer)
        finally:
            if not remote_writer.is_closing():
                remote_writer.close()
            try:
                await remote_writer.wait_closed()
            except Exception:
                pass

    async def _handle_forward(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        method: str,
        target: str,
        version: str,
        headers: dict[str, str],
    ) -> None:
        if target.startswith("/"):
            host = headers.get("host")
            if not host:
                raise ValueError("origin-form request has no Host header")
            logical_url = f"http://{host}{target}"
        else:
            logical_url = target
        parsed = urlsplit(logical_url)
        scheme = parsed.scheme.lower()
        websocket = scheme == "ws"
        if scheme == "ws":
            scheme = "http"
        elif scheme == "wss":
            scheme = "https"
        if scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("proxy only accepts HTTP(S) targets")
        if scheme == "https":
            raise ValueError("HTTPS requests must use CONNECT")
        port = parsed.port or 80
        resolved = await self._resolve(parsed._replace(scheme=scheme).geturl())
        body = await self._read_forward_body(reader, headers)
        remote_reader, remote_writer = await self._open_validated(resolved, port)
        try:
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"
            output_headers = []
            saw_host = False
            chunked = headers.get("transfer-encoding", "").lower() == "chunked"
            for name, value in headers.items():
                if (
                    name == "proxy-connection"
                    or name == "expect"
                    or (not websocket and name == "connection")
                    or (chunked and name == "content-length")
                ):
                    continue
                if name == "host":
                    value = parsed.netloc
                    saw_host = True
                output_headers.append(f"{name}: {value}")
            if not saw_host:
                output_headers.append(f"Host: {parsed.netloc}")
            if not websocket:
                output_headers.append("Connection: close")
            request = (
                f"{method} {path} {version}\r\n"
                + "\r\n".join(output_headers)
                + "\r\n\r\n"
            ).encode("latin-1")
            remote_writer.write(request + body)
            await remote_writer.drain()
            if websocket:
                # A WebSocket stays on this one validated origin after the
                # HTTP upgrade, so it needs the normal bidirectional bridge.
                await self._bridge(reader, writer, remote_reader, remote_writer)
            else:
                # Do not bridge client_reader here. Once this request has been
                # sent, any additional bytes on Chromium's keep-alive socket
                # must never reach this origin's upstream connection.
                await self._relay_response(remote_reader, writer)
        finally:
            remote_writer.close()
            try:
                await remote_writer.wait_closed()
            except Exception:
                pass

    @staticmethod
    async def _read_forward_body(
        reader: asyncio.StreamReader, headers: dict[str, str]
    ) -> bytes:
        """Read exactly one bounded HTTP request body before opening upstream."""
        transfer_encoding = headers.get("transfer-encoding", "").lower()
        if transfer_encoding not in {"", "identity", "chunked"}:
            raise ValueError("unsupported proxy request transfer encoding")
        if transfer_encoding == "chunked":
            body = bytearray()
            total = 0
            while True:
                size_line = await asyncio.wait_for(
                    reader.readuntil(b"\r\n"), _PROXY_CONNECT_TIMEOUT
                )
                try:
                    size_text = size_line[:-2].split(b";", 1)[0].strip()
                    chunk_size = int(size_text, 16)
                except ValueError as exc:
                    raise ValueError("invalid chunked proxy request body") from exc
                body.extend(size_line)
                if chunk_size == 0:
                    while True:
                        trailer = await asyncio.wait_for(
                            reader.readuntil(b"\r\n"), _PROXY_CONNECT_TIMEOUT
                        )
                        body.extend(trailer)
                        if trailer == b"\r\n":
                            return bytes(body)
                total += chunk_size
                if total > _PROXY_MAX_REQUEST_BODY:
                    raise ValueError("proxy request body is too large")
                chunk = await asyncio.wait_for(
                    reader.readexactly(chunk_size + 2), _PROXY_CONNECT_TIMEOUT
                )
                if not chunk.endswith(b"\r\n"):
                    raise ValueError("invalid chunked proxy request body")
                body.extend(chunk)
        raw_length = headers.get("content-length", "0")
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if content_length < 0 or content_length > _PROXY_MAX_REQUEST_BODY:
            raise ValueError("proxy request body is too large")
        if content_length == 0:
            return b""
        try:
            return await asyncio.wait_for(
                reader.readexactly(content_length), _PROXY_CONNECT_TIMEOUT
            )
        except asyncio.IncompleteReadError as exc:
            raise ValueError("proxy request body ended early") from exc

    @staticmethod
    async def _relay_response(
        remote_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        header = await asyncio.wait_for(
            remote_reader.readuntil(b"\r\n\r\n"), _PROXY_RESPONSE_TIMEOUT
        )
        lines = header[:-4].split(b"\r\n")
        response_headers: list[bytes] = []
        content_length: int | None = None
        chunked = False
        for line in lines[1:]:
            name, _, value = line.partition(b":")
            normalized_name = name.lower().strip()
            if normalized_name in {b"connection", b"proxy-connection", b"keep-alive"}:
                continue
            if normalized_name == b"content-length":
                try:
                    content_length = int(value.strip())
                except ValueError as exc:
                    raise ValueError("invalid upstream Content-Length") from exc
                if content_length < 0:
                    raise ValueError("invalid upstream Content-Length")
            elif normalized_name == b"transfer-encoding":
                chunked = value.strip().lower() == b"chunked"
            response_headers.append(line)
        response_headers.append(b"Connection: close")
        client_writer.write(b"\r\n".join([lines[0], *response_headers, b"", b""]))
        await client_writer.drain()

        if content_length is not None:
            remaining = content_length
            while remaining:
                data = await asyncio.wait_for(
                    remote_reader.read(min(64 * 1024, remaining)),
                    _PROXY_RESPONSE_TIMEOUT,
                )
                if not data:
                    break
                client_writer.write(data)
                await client_writer.drain()
                remaining -= len(data)
            return

        if chunked:
            while True:
                size_line = await asyncio.wait_for(
                    remote_reader.readuntil(b"\r\n"), _PROXY_RESPONSE_TIMEOUT
                )
                client_writer.write(size_line)
                await client_writer.drain()
                try:
                    chunk_size = int(size_line[:-2].split(b";", 1)[0].strip(), 16)
                except ValueError as exc:
                    raise ValueError("invalid upstream chunked response") from exc
                chunk = await asyncio.wait_for(
                    remote_reader.readexactly(chunk_size + 2),
                    _PROXY_RESPONSE_TIMEOUT,
                )
                client_writer.write(chunk)
                await client_writer.drain()
                if chunk_size == 0:
                    while True:
                        trailer = await asyncio.wait_for(
                            remote_reader.readuntil(b"\r\n"),
                            _PROXY_RESPONSE_TIMEOUT,
                        )
                        client_writer.write(trailer)
                        await client_writer.drain()
                        if trailer == b"\r\n":
                            return

        try:
            while data := await asyncio.wait_for(
                remote_reader.read(64 * 1024), _PROXY_RESPONSE_TIMEOUT
            ):
                client_writer.write(data)
                await client_writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.TimeoutError):
            pass

    @staticmethod
    async def _open_validated(
        addresses: list[str], port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        last_error: OSError | None = None
        for address in addresses:
            try:
                return await asyncio.wait_for(
                    asyncio.open_connection(address, port), _PROXY_CONNECT_TIMEOUT
                )
            except (OSError, asyncio.TimeoutError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise OSError("validated host did not provide any addresses")

    @staticmethod
    async def _send_error(
        writer: asyncio.StreamWriter, status: int, reason: str
    ) -> None:
        if writer.is_closing():
            return
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\n"
            "Connection: close\r\n"
            "Content-Length: 0\r\n\r\n".encode("ascii")
        )
        try:
            await writer.drain()
        except Exception:
            pass

    @staticmethod
    async def _bridge(
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        remote_reader: asyncio.StreamReader,
        remote_writer: asyncio.StreamWriter,
    ) -> None:
        async def copy_stream(
            source: asyncio.StreamReader, destination: asyncio.StreamWriter
        ) -> None:
            try:
                while data := await source.read(64 * 1024):
                    destination.write(data)
                    await destination.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                pass

        tasks = {
            asyncio.create_task(copy_stream(client_reader, remote_writer)),
            asyncio.create_task(copy_stream(remote_reader, client_writer)),
        }
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            remote_writer.close()
            try:
                await remote_writer.wait_closed()
            except Exception:
                pass
