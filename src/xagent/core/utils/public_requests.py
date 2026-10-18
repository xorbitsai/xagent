"""Requests-compatible public-only clients for synchronous integrations.

The original URL remains available for cookies, redirects, signing and error
metadata; only the connection pool target is rewritten. TLS SNI and certificate
hostname checks still use the logical hostname. No global monkey-patching.
"""

from __future__ import annotations

import os
import ssl
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import requests as _requests
from requests import ConnectionError as ConnectionError
from requests import ConnectTimeout as ConnectTimeout
from requests import HTTPError as HTTPError
from requests import (
    PreparedRequest,
)
from requests import Request as Request
from requests import RequestException as RequestException
from requests import (
    Response,
)
from requests import Timeout as Timeout
from requests import exceptions as exceptions
from requests.adapters import HTTPAdapter
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, NewConnectionError
from urllib3.util import Timeout as PoolTimeout

from ...config import get_trusted_egress_proxy_enabled

if TYPE_CHECKING:
    from requests.adapters import _HostParams, _PoolKwargs

from .public_policy import HttpDestinationPolicy
from .security import (
    build_ca_bundle_ssl_context,
    public_http_destination,
    validate_public_http_url_sync,
)


class _PublicAdapter(HTTPAdapter):
    def __init__(self, *, _policy: HttpDestinationPolicy | None = None) -> None:
        self._policy = _policy
        super().__init__()
        self._destination: ContextVar[tuple[str, bool] | None] = ContextVar(
            "public_requests_destination", default=None
        )
        self._cached_ca_context: (
            tuple[tuple[str | None, str | None], ssl.SSLContext] | None
        ) = None

    def _ca_context(self, cert: Any) -> ssl.SSLContext:
        # Never reuse contexts that urllib3 may mutate by loading a client
        # certificate. No-cert upload chunks can safely share CA-only state.
        if cert is not None:
            return build_ca_bundle_ssl_context()
        key = (os.getenv("SSL_CERT_FILE"), os.getenv("SSL_CERT_DIR"))
        cached = self._cached_ca_context
        if cached is not None and cached[0] == key:
            return cached[1]
        context = build_ca_bundle_ssl_context()
        self._cached_ca_context = (key, context)
        return context

    def build_connection_pool_key_attributes(
        self, request: PreparedRequest, verify: Any, cert: Any = None
    ) -> tuple[_HostParams, _PoolKwargs]:
        destination = self._destination.get()
        if destination is None:
            try:
                destination = (
                    self._resolve_addresses(request.url or "")[0],
                    False,
                )
            except OSError as exc:
                raise ConnectionError(
                    "Public API host could not be resolved", request=request
                ) from exc
        parsed = urlsplit(request.url or "")
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        if not destination[1]:
            host_params["host"] = destination[0]
        if parsed.scheme == "https":
            # Requests' stubs omit these supported urllib3 HTTPS pool keys.
            tls_kwargs = cast(dict[str, Any], pool_kwargs)
            if verify is True:
                # Preserve SSL_CERT_FILE/SSL_CERT_DIR as well as Requests'
                # explicit CA-path handling. Cache CA-only state for keep-alive,
                # while client-cert contexts stay isolated per request.
                tls_kwargs["ssl_context"] = self._ca_context(cert)
            tls_kwargs["server_hostname"] = parsed.hostname
            tls_kwargs["assert_hostname"] = parsed.hostname
        return host_params, pool_kwargs

    def _resolve_addresses(self, url: str) -> list[str]:
        return (
            self._policy.resolve_sync(url)
            if self._policy is not None
            else validate_public_http_url_sync(url)
        )

    def send(
        self,
        request: PreparedRequest,
        stream: bool = False,
        timeout: Any = None,
        verify: Any = True,
        cert: Any = None,
        proxies: Any = None,
    ) -> Response:
        if request.method == "CONNECT":
            raise ValueError("HTTP tunnels are not allowed")
        via_proxy = any((proxies or {}).values())
        trusted_proxy = _trusted_proxy_url()
        if via_proxy and any(
            value != trusted_proxy for value in proxies.values() if value
        ):
            raise ValueError(
                "Public Requests clients cannot bypass policy via an untrusted proxy"
            )
        started = time.monotonic()
        try:
            addresses = self._resolve_addresses(request.url or "")
        except OSError as exc:
            raise ConnectionError(
                "Public API host could not be resolved", request=request
            ) from exc
        except ValueError as exc:
            raise _requests.exceptions.InvalidURL(str(exc), request=request) from exc
        parsed = urlsplit(request.url or "")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        request.headers["Host"] = (
            host
            if port == (443 if parsed.scheme == "https" else 80)
            else f"{host}:{port}"
        )
        request.headers.pop("Proxy-Authorization", None)
        request.headers.pop("Proxy-Connection", None)
        budget = (
            timeout
            if isinstance(timeout, PoolTimeout)
            else (
                PoolTimeout(connect=timeout[0], read=timeout[1])
                if isinstance(timeout, tuple)
                else PoolTimeout.from_float(timeout)
            )
        )
        targets = addresses[:1] if via_proxy else addresses
        for index, address in enumerate(targets):
            attempt_timeout = budget
            if isinstance(budget.connect_timeout, (int, float)):
                remaining = budget.connect_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise ConnectTimeout(
                        "Public API connect budget exhausted", request=request
                    )
                attempt_timeout = PoolTimeout(
                    connect=remaining / (len(targets) - index), read=budget.read_timeout
                )
            token = self._destination.set((address, via_proxy))
            try:
                return super().send(
                    request,
                    stream=stream,
                    timeout=cast(
                        Any, attempt_timeout
                    ),  # Requests supports urllib3 Timeout; its stubs omit it.
                    verify=verify,
                    cert=cert,
                    proxies=proxies,
                )
            except ConnectionError as exc:
                # Only urllib3's pre-connect failures are safe to retry. A
                # generic ConnectionError may instead follow a body write.
                reason = exc.args[0] if exc.args else None
                if isinstance(reason, MaxRetryError):
                    reason = reason.reason
                if (
                    not isinstance(reason, (NewConnectionError, ConnectTimeoutError))
                    or index == len(targets) - 1
                ):
                    raise
            finally:
                self._destination.reset(token)
        raise ConnectionError("Public API host did not resolve", request=request)


def _trusted_proxy_url() -> str | None:
    if not get_trusted_egress_proxy_enabled():
        return None
    return (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )


class Session(_requests.Session):
    """Use the checked IP for each request, including every redirect."""

    def __init__(self, *, _policy: HttpDestinationPolicy | None = None) -> None:
        super().__init__()
        self._policy = _policy
        self.trust_env = False
        proxy = _trusted_proxy_url()
        if proxy:
            # The operator opt-in means this proxy enforces public-only DNS
            # at connect time. NO_PROXY cannot bypass that boundary.
            self.proxies = {"http": proxy, "https": proxy}
        self.max_redirects = 5
        self.mount("http://", _PublicAdapter(_policy=_policy))
        self.mount("https://", _PublicAdapter(_policy=_policy))

    def merge_environment_settings(
        self, url: str | bytes | None, proxies: Any, stream: Any, verify: Any, cert: Any
    ) -> Any:
        # Keep CA trust configuration without enabling ambient proxy or netrc
        # credentials. Explicit verify paths / False and Session.verify win.
        if verify in (None, True) and self.verify is True:
            verify = (
                os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("CURL_CA_BUNDLE") or verify
            )
        return super().merge_environment_settings(url, proxies, stream, verify, cert)

    def prepare_request(self, request: Request) -> PreparedRequest:
        # Requests/urllib3 may normalize a zero port or URL userinfo before
        # the adapter sees it. Reject those inputs before normalization.
        if self._policy is not None:
            self._policy.destination(request.url or "")
        else:
            public_http_destination(request.url or "")
        return super().prepare_request(request)

    def rebuild_auth(
        self, prepared_request: PreparedRequest, response: Response
    ) -> None:
        super().rebuild_auth(prepared_request, response)
        old, new = urlsplit(response.request.url), urlsplit(prepared_request.url)
        if (old.scheme, old.hostname, old.port) == (new.scheme, new.hostname, new.port):
            return
        if prepared_request.method not in {"GET", "HEAD"} or prepared_request.body:
            raise ValueError("Cross-origin redirects cannot replay integration bodies")
        prepared_request.headers.clear()
        prepared_request.headers["Accept"] = "application/json"
        prepared_request.headers["User-Agent"] = "Xagent-Public-Integration/1.0"


def request(
    method: str,
    url: str,
    *,
    _policy: HttpDestinationPolicy | None = None,
    **kwargs: Any,
) -> Response:
    session = Session(_policy=_policy)
    try:
        response = session.request(method, url, **kwargs)
    except BaseException:
        session.close()
        raise
    if kwargs.get("stream"):
        original_close = response.close

        def close() -> None:
            try:
                original_close()
            finally:
                session.close()

        response.close = close  # type: ignore[method-assign]  # own this streaming response's session
    else:
        session.close()
    return response


def get(url: str, **kwargs: Any) -> Response:
    kwargs.setdefault("allow_redirects", True)
    return request("GET", url, **kwargs)


def post(url: str, data: Any = None, json: Any = None, **kwargs: Any) -> Response:
    return request("POST", url, data=data, json=json, **kwargs)


def put(url: str, data: Any = None, **kwargs: Any) -> Response:
    return request("PUT", url, data=data, **kwargs)


def patch(url: str, data: Any = None, **kwargs: Any) -> Response:
    return request("PATCH", url, data=data, **kwargs)


def delete(url: str, **kwargs: Any) -> Response:
    return request("DELETE", url, **kwargs)


def head(url: str, **kwargs: Any) -> Response:
    kwargs.setdefault("allow_redirects", False)
    return request("HEAD", url, **kwargs)


def options(url: str, **kwargs: Any) -> Response:
    return request("OPTIONS", url, **kwargs)
