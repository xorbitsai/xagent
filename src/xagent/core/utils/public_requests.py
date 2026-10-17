"""Requests-compatible public-only clients for synchronous integrations.

The original URL remains available for cookies, redirects, signing and error
metadata; only the connection pool target is rewritten. TLS SNI and certificate
hostname checks still use the logical hostname. No global monkey-patching.
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from requests.adapters import _HostParams, _PoolKwargs

from .security import public_http_destination, validate_public_http_url_sync


class _PublicAdapter(HTTPAdapter):
    def build_connection_pool_key_attributes(
        self, request: PreparedRequest, verify: Any, cert: Any = None
    ) -> tuple[_HostParams, _PoolKwargs]:
        addresses = validate_public_http_url_sync(request.url or "")
        parsed = urlsplit(request.url or "")
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        host_params["host"] = addresses[0]
        if parsed.scheme == "https":
            # Requests' stubs omit these supported urllib3 HTTPS pool keys.
            tls_kwargs = cast(dict[str, Any], pool_kwargs)
            tls_kwargs["server_hostname"] = parsed.hostname
            tls_kwargs["assert_hostname"] = parsed.hostname
        return host_params, pool_kwargs

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
        if any((proxies or {}).values()):
            raise ValueError("Public Requests clients cannot bypass policy via a proxy")
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
        return super().send(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )


class Session(_requests.Session):
    """Use the checked IP for each request, including every redirect."""

    def __init__(self) -> None:
        super().__init__()
        self.trust_env = False
        self.max_redirects = 5
        self.mount("http://", _PublicAdapter())
        self.mount("https://", _PublicAdapter())

    def prepare_request(self, request: Request) -> PreparedRequest:
        # Requests/urllib3 may normalize a zero port or URL userinfo before
        # the adapter sees it. Reject those inputs before normalization.
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


def request(method: str, url: str, **kwargs: Any) -> Response:
    session = Session()
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
