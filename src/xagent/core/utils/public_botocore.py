"""Connection-bound public policy for Bedrock's urllib3 transport."""

from typing import Any
from urllib.parse import urlsplit

from botocore.httpsession import URLLib3Session
from urllib3 import PoolManager

from .security import validate_public_http_url_sync


class _PublicPoolManager(PoolManager):
    def connection_from_host(
        self,
        host: str | None,
        port: int | None = None,
        scheme: str | None = "http",
        pool_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        scheme = scheme or "http"
        logical_host = host or ""
        literal = f"[{logical_host}]" if ":" in logical_host else logical_host
        port = port if port is not None else (443 if scheme == "https" else 80)
        addresses = validate_public_http_url_sync(f"{scheme}://{literal}:{port}/")
        options = dict(pool_kwargs or {})
        if scheme == "https":
            options["server_hostname"] = logical_host
            options["assert_hostname"] = logical_host
        return super().connection_from_host(addresses[0], port, scheme, options)


class PublicBotocoreSession(URLLib3Session):  # type: ignore[no-any-unimported]  # SDK is untyped
    _manager: Any

    def __init__(self, *, timeout: float, verify: bool | str = True) -> None:
        super().__init__(proxies={}, timeout=timeout, verify=verify)
        self._manager.clear()
        self._manager = _PublicPoolManager(**self._get_pool_manager_kwargs())
        # Keep AWS's streaming/header connection classes, not urllib3 defaults.
        self._manager.pool_classes_by_scheme = self._pool_classes_by_scheme

    def send(self, request: Any) -> Any:
        parsed = urlsplit(request.url)
        if request.method == "CONNECT":
            raise ValueError("HTTP tunnels are not allowed")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Model URL must not contain embedded credentials")
        hostname = parsed.hostname or ""
        literal = f"[{hostname}]" if ":" in hostname else hostname
        default_port = 443 if parsed.scheme == "https" else 80
        port = parsed.port if parsed.port is not None else default_port
        request.headers["Host"] = (
            literal if port == default_port else f"{literal}:{port}"
        )
        request.headers.pop("Proxy-Authorization", None)
        request.headers.pop("Proxy-Connection", None)
        return super().send(request)
