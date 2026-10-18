"""Connection-bound public policy for Bedrock's urllib3 transport."""

import time
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit

from botocore.exceptions import ConnectTimeoutError, EndpointConnectionError
from botocore.httpsession import URLLib3Session
from urllib3 import PoolManager
from urllib3.util import Timeout

from .public_http_transport import UnsafePublicHttpTarget
from .security import validate_public_http_url_sync


class _PublicPoolManager(PoolManager):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.destination: ContextVar[tuple[str, Any] | None] = ContextVar(
            "public_botocore_destination", default=None
        )

    def connection_from_host(
        self,
        host: str | None,
        port: int | None = None,
        scheme: str | None = "http",
        pool_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        scheme = scheme or "http"
        logical_host = (host or "").removeprefix("[").removesuffix("]")
        port = port if port is not None else (443 if scheme == "https" else 80)
        destination = self.destination.get()
        if destination is None:
            raise UnsafePublicHttpTarget("A checked request destination is required")
        options = dict(pool_kwargs or {})
        options["timeout"] = destination[1]
        if scheme == "https":
            options["server_hostname"] = logical_host
            options["assert_hostname"] = logical_host
        return super().connection_from_host(destination[0], port, scheme, options)


class PublicBotocoreSession(URLLib3Session):  # type: ignore[no-any-unimported]  # SDK is untyped
    _manager: Any

    def __init__(
        self,
        verify: bool | str = True,
        proxies: Any = None,
        timeout: Any = None,
        max_pool_connections: int = 10,
        socket_options: Any = None,
        client_cert: Any = None,
        proxies_config: Any = None,
    ) -> None:
        if any((proxies or {}).values()):
            raise UnsafePublicHttpTarget(
                "Public Botocore sessions do not support proxies"
            )
        super().__init__(
            proxies={},
            timeout=timeout,
            verify=verify,
            max_pool_connections=max_pool_connections,
            socket_options=socket_options,
            client_cert=client_cert,
            proxies_config=proxies_config,
        )
        self._manager.clear()
        self._manager = _PublicPoolManager(**self._get_pool_manager_kwargs())
        # Keep AWS's streaming/header connection classes, not urllib3 defaults.
        self._manager.pool_classes_by_scheme = self._pool_classes_by_scheme

    def send(self, request: Any) -> Any:
        started = time.monotonic()
        if request.method.upper() == "CONNECT":
            raise UnsafePublicHttpTarget("HTTP tunnels are not allowed")
        try:
            addresses = validate_public_http_url_sync(request.url)
        except ValueError as exc:
            # Outside botocore's catch-all: policy failures must not become
            # retryable HTTPClientError failures in the SDK retry handler.
            raise UnsafePublicHttpTarget(str(exc)) from exc
        except OSError as exc:
            raise EndpointConnectionError(endpoint_url=request.url, error=exc) from exc
        parsed = urlsplit(request.url)
        hostname = parsed.hostname or ""
        literal = f"[{hostname}]" if ":" in hostname else hostname
        default_port = 443 if parsed.scheme == "https" else 80
        port = parsed.port if parsed.port is not None else default_port
        request.headers["Host"] = (
            literal if port == default_port else f"{literal}:{port}"
        )
        request.headers.pop("Proxy-Authorization", None)
        request.headers.pop("Proxy-Connection", None)
        budget = (
            self._timeout
            if isinstance(self._timeout, Timeout)
            else Timeout.from_float(self._timeout)
        )
        for index, address in enumerate(addresses):
            attempt_timeout = budget
            if isinstance(budget.connect_timeout, (int, float)):
                remaining = budget.connect_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise ConnectTimeoutError(
                        endpoint_url=request.url, error="Connect budget exhausted"
                    )
                attempt_timeout = Timeout(
                    connect=remaining / (len(addresses) - index),
                    read=budget.read_timeout,
                )
            token = self._manager.destination.set((address, attempt_timeout))
            try:
                return super().send(request)
            except (EndpointConnectionError, ConnectTimeoutError):
                if index == len(addresses) - 1:
                    raise
            finally:
                self._manager.destination.reset(token)
        raise EndpointConnectionError(endpoint_url=request.url)
