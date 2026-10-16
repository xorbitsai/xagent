"""Public-network-only HTTP CONNECT/forward proxy shared by untrusted fetchers."""

import asyncio
import base64
import logging
import secrets
from typing import Any
from urllib.parse import unquote, urlsplit

from .security import build_ca_bundle_ssl_context, validate_public_http_url

logger = logging.getLogger(__name__)

_PROXY_HEADER_LIMIT = 64 * 1024
_PROXY_CONNECT_TIMEOUT = 10.0
_PROXY_RESPONSE_TIMEOUT = 10.0
_PROXY_MAX_REQUEST_BODY = 64 * 1024 * 1024


class PublicNetworkPolicyError(ValueError):
    """A terminal target-policy refusal, not an origin/WAF failure."""


class PublicNetworkEgressProxy:
    """Small loopback proxy that pins each connection to validated DNS IPs.

    HTTP clients resolve hosts independently of Python. A shared CONNECT
    proxy closes that TOCTOU gap: the proxy validates the hostname, connects
    directly to one of the validated public IPs, and then tunnels the original
    TLS/WebSocket handshake without changing its Host header or SNI.
    Upstream-backed gates require per-instance authentication; keep the URL
    returned by start() private. Direct browser gates carry no upstream identity.
    """

    def __init__(
        self,
        *,
        connect_timeout: float = _PROXY_CONNECT_TIMEOUT,
        response_timeout: float = _PROXY_RESPONSE_TIMEOUT,
        upstream_proxy_url: str | None = None,
    ) -> None:
        self._server: asyncio.AbstractServer | None = None
        self._closing = False
        self._client_tasks: set[asyncio.Task[Any]] = set()
        self._responses_started: set[asyncio.StreamWriter] = set()
        self._username = secrets.token_urlsafe(16)
        self._password = secrets.token_urlsafe(16)
        self._auth_token = base64.b64encode(
            f"{self._username}:{self._password}".encode()
        )
        self.denied_hosts: set[str] = set()
        self.connect_timeout = connect_timeout
        self.response_timeout = response_timeout
        self._upstream = urlsplit(upstream_proxy_url) if upstream_proxy_url else None
        if self._upstream and (
            self._upstream.scheme not in {"http", "https"}
            or not self._upstream.hostname
            or self._upstream.path not in {"", "/"}
            or self._upstream.query
            or self._upstream.fragment
        ):
            raise ValueError("Trusted upstream proxy must be an HTTP(S) proxy URL")
        if self._upstream and self._upstream.port == 0:
            raise ValueError("Trusted upstream proxy port must be positive")
        self._upstream_ssl = (
            build_ca_bundle_ssl_context()
            if self._upstream and self._upstream.scheme == "https"
            else None
        )

    def is_denied(self, url: str) -> bool:
        """Report this gate's policy refusals, independent of origin headers."""
        return urlsplit(url).hostname in self.denied_hosts

    async def _resolve(self, url: str) -> list[str]:
        try:
            return await validate_public_http_url(url)
        except ValueError as exc:
            hostname = urlsplit(url).hostname
            if hostname:
                self.denied_hosts.add(hostname)
            logger.warning("Public-network egress policy denied host %r", hostname)
            raise PublicNetworkPolicyError(str(exc)) from exc

    async def start(self) -> str:
        if self._server is not None:
            raise RuntimeError("Egress proxy is already started")
        self._closing = False
        self._server = await asyncio.start_server(
            self._handle_client,
            host="127.0.0.1",
            port=0,
            limit=_PROXY_HEADER_LIMIT,
        )
        socket = self._server.sockets[0]
        port = socket.getsockname()[1]
        credentials = f"{self._username}:{self._password}@" if self._upstream else ""
        return f"http://{credentials}127.0.0.1:{port}"

    async def close(self) -> None:
        self._closing = True
        # Stop accepting before taking the task snapshot. A callback already
        # queued by the event loop must also see _closing and exit without I/O.
        if self._server is not None:
            self._server.close()
        tasks = list(self._client_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._client_tasks.clear()
        # Python 3.12+ waits for accepted connections too: cancel/close their
        # handlers before awaiting this, especially for long-lived CONNECTs.
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
        request_started = False
        try:
            if self._closing:
                return
            header = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), self.connect_timeout
            )
            method, target, version, headers = self._parse_request(header)
            scheme, _, token = headers.get("proxy-authorization", "").partition(" ")
            if self._upstream and (
                scheme.lower() != "basic"
                or not secrets.compare_digest(token.encode("latin-1"), self._auth_token)
            ):
                await self._send_error(writer, 407, "Proxy Authentication Required")
                return
            request_started = True
            if method.upper() == "CONNECT":
                await self._handle_connect(reader, writer, target)
            else:
                await self._handle_forward(
                    reader, writer, method, target, version, headers
                )
        except asyncio.CancelledError:
            raise
        except PublicNetworkPolicyError:
            if writer not in self._responses_started:
                await self._send_error(writer, 403, "Public Network Policy Denied")
        except Exception as exc:
            logger.log(
                logging.INFO if request_started else logging.DEBUG,
                "Public-network egress transport failed: %s",
                type(exc).__name__,
            )
            if writer not in self._responses_started:
                await self._send_error(writer, 502, "Bad Gateway")
        finally:
            self._responses_started.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            finally:
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
            self._responses_started.add(writer)
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
                    or name == "proxy-authorization"
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
                self._responses_started.add(writer)
                # A WebSocket stays on this one validated origin after the
                # HTTP upgrade, so it needs the normal bidirectional bridge.
                await self._bridge(reader, writer, remote_reader, remote_writer)
            else:
                # Do not bridge client_reader here. Once this request has been
                # sent, any additional bytes on a client's keep-alive socket
                # must never reach this origin's upstream connection.
                await self._relay_response(remote_reader, writer, method=method)
        finally:
            remote_writer.close()
            try:
                await remote_writer.wait_closed()
            except Exception:
                pass

    async def _read_forward_body(
        self, reader: asyncio.StreamReader, headers: dict[str, str]
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
                    reader.readuntil(b"\r\n"), self.connect_timeout
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
                            reader.readuntil(b"\r\n"), self.connect_timeout
                        )
                        body.extend(trailer)
                        if trailer == b"\r\n":
                            return bytes(body)
                total += chunk_size
                if total > _PROXY_MAX_REQUEST_BODY:
                    raise ValueError("proxy request body is too large")
                chunk = await asyncio.wait_for(
                    reader.readexactly(chunk_size + 2), self.connect_timeout
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
                reader.readexactly(content_length), self.connect_timeout
            )
        except asyncio.IncompleteReadError as exc:
            raise ValueError("proxy request body ended early") from exc

    async def _relay_response(
        self,
        remote_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        *,
        method: str = "GET",
    ) -> None:
        while True:
            header = await asyncio.wait_for(
                remote_reader.readuntil(b"\r\n\r\n"), self.response_timeout
            )
            lines = header[:-4].split(b"\r\n")
            status = int(lines[0].split(b" ", 2)[1])
            if not 100 <= status < 200 or status == 101:
                break
            # Early Hints/Continue precede, rather than replace, the final
            # response. Do not apply HEAD/304 body rules until that arrives.
            self._responses_started.add(client_writer)
            client_writer.write(header)
            await client_writer.drain()
        response_headers: list[bytes] = []
        content_length: int | None = None
        chunked = any(
            line.partition(b":")[0].lower().strip() == b"transfer-encoding"
            and line.partition(b":")[2].lower().split(b",")[-1].strip() == b"chunked"
            for line in lines[1:]
        )
        for line in lines[1:]:
            name, _, value = line.partition(b":")
            normalized_name = name.lower().strip()
            if normalized_name in {b"connection", b"proxy-connection", b"keep-alive"}:
                continue
            if normalized_name == b"content-length":
                if chunked:
                    continue
                try:
                    content_length = int(value.strip())
                except ValueError as exc:
                    raise ValueError("invalid upstream Content-Length") from exc
                if content_length < 0:
                    raise ValueError("invalid upstream Content-Length")
            response_headers.append(line)
        response_headers.append(b"Connection: close")
        self._responses_started.add(client_writer)
        client_writer.write(b"\r\n".join([lines[0], *response_headers, b"", b""]))
        await client_writer.drain()

        # Content-Length on HEAD/304 describes the selected representation,
        # not bytes on this connection. 204 likewise never carries a body.
        if method.upper() == "HEAD" or status in {204, 304}:
            return

        if content_length is not None:
            remaining = content_length
            while remaining:
                data = await asyncio.wait_for(
                    remote_reader.read(min(64 * 1024, remaining)),
                    self.response_timeout,
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
                    remote_reader.readuntil(b"\r\n"), self.response_timeout
                )
                client_writer.write(size_line)
                await client_writer.drain()
                try:
                    chunk_size = int(size_line[:-2].split(b";", 1)[0].strip(), 16)
                except ValueError as exc:
                    raise ValueError("invalid upstream chunked response") from exc
                if chunk_size == 0:
                    while True:
                        trailer = await asyncio.wait_for(
                            remote_reader.readuntil(b"\r\n"),
                            self.response_timeout,
                        )
                        client_writer.write(trailer)
                        await client_writer.drain()
                        if trailer == b"\r\n":
                            return
                chunk = await asyncio.wait_for(
                    remote_reader.readexactly(chunk_size + 2),
                    self.response_timeout,
                )
                client_writer.write(chunk)
                await client_writer.drain()

        while data := await asyncio.wait_for(
            remote_reader.read(64 * 1024), self.response_timeout
        ):
            client_writer.write(data)
            await client_writer.drain()

    async def _open_validated(
        self, addresses: list[str], port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        last_error: OSError | None = None
        for address in addresses:
            try:
                return await asyncio.wait_for(
                    self._connect_address(address, port), self.connect_timeout
                )
            except (OSError, asyncio.TimeoutError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise OSError("validated host did not provide any addresses")

    async def _connect_address(
        self, address: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Tunnel a validated literal IP, never delegate target DNS upstream."""
        upstream = self._upstream
        if upstream is None:
            return await asyncio.open_connection(address, port)
        reader, writer = await asyncio.open_connection(
            upstream.hostname,
            upstream.port or (443 if upstream.scheme == "https" else 80),
            ssl=self._upstream_ssl,
            limit=_PROXY_HEADER_LIMIT,
        )
        try:
            authority = f"[{address}]:{port}" if ":" in address else f"{address}:{port}"
            auth = ""
            if upstream.username is not None:
                credentials = (
                    f"{unquote(upstream.username)}:{unquote(upstream.password or '')}"
                )
                encoded = base64.b64encode(credentials.encode()).decode("ascii")
                auth = f"Proxy-Authorization: Basic {encoded}\r\n"
            writer.write(
                f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n{auth}\r\n".encode(
                    "ascii"
                )
            )
            await writer.drain()
            try:
                header = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
                raise OSError("Incomplete upstream CONNECT reply") from exc
            status = header.split(b"\r\n", 1)[0].split(b" ", 2)
            if (
                len(status) < 2
                or status[0] not in {b"HTTP/1.0", b"HTTP/1.1"}
                or len(status[1]) != 3
                or not status[1].isdigit()
            ):
                raise OSError("Malformed upstream CONNECT reply")
            if status[1] != b"200":
                raise OSError("Trusted upstream proxy refused CONNECT")
            return reader, writer
        except BaseException:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            raise

    @staticmethod
    async def _send_error(
        writer: asyncio.StreamWriter, status: int, reason: str
    ) -> None:
        if writer.is_closing():
            return
        authenticate = (
            'Proxy-Authenticate: Basic realm="xagent-egress"\r\n'
            if status == 407
            else ""
        )
        writer.write(
            (
                f"HTTP/1.1 {status} {reason}\r\n"
                "Connection: close\r\n"
                f"{authenticate}Content-Length: 0\r\n\r\n"
            ).encode("ascii")
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
