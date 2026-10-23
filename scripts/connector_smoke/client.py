"""HTTP access to a deployed instance for the connector smoke checks.

Cookies are sent as an explicit header rather than through a cookie jar: the
deployment may route by cookie (Xagent SaaS selects the region with
``selected_region`` and authenticates the login with the ``login_route``
cookie minted by ``/api/auth/login/start``), and a jar would also collect the
OAuth state cookie every launch sets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


class SmokeError(Exception):
    """A request or response the check cannot use; the message is safe to print."""


class SmokeTimeout(SmokeError):
    """The request did not complete within its timeout."""


@dataclass(frozen=True)
class LaunchStart:
    status_code: int
    payload: Any


@dataclass(frozen=True)
class JSONResponse:
    status_code: int
    payload: Any


@dataclass(frozen=True)
class LaunchRedeem:
    status_code: int
    location: str | None
    body: str


class SmokeClient:
    def __init__(
        self,
        base_url: str,
        *,
        origin: str,
        cookies: Mapping[str, str] | None = None,
        timeout: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._cookies = dict(cookies or {})
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Accept": "application/json", "Origin": origin},
            timeout=timeout,
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> SmokeClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def login(
        self,
        login: str,
        password: str,
        *,
        login_field: str = "username",
        login_start: bool = False,
    ) -> None:
        """Authenticate and send the bearer token on every later request."""
        login_cookies: dict[str, str] = {}
        if login_start:
            start = self._request(
                "POST", "/api/auth/login/start", json={"email": login}
            )
            if start.status_code not in (200, 204):
                raise SmokeError(f"login start failed: HTTP {start.status_code}")
            login_cookies = dict(start.cookies.items())
        response = self._request(
            "POST",
            "/api/auth/login",
            json={login_field: login, "password": password},
            extra_cookies=login_cookies,
        )
        if response.status_code != 200:
            raise SmokeError(f"login failed: HTTP {response.status_code}")
        try:
            token = response.json().get("access_token")
        except (ValueError, AttributeError):
            token = None
        if not isinstance(token, str) or not token:
            raise SmokeError("login response has no access_token")
        self._http.headers["Authorization"] = f"Bearer {token}"

    def list_catalog_apps(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/api/mcp/apps", params={"location": "remote"})
        if response.status_code != 200:
            raise SmokeError(f"catalog request failed: HTTP {response.status_code}")
        try:
            apps = response.json()
        except ValueError:
            apps = None
        if not isinstance(apps, list):
            raise SmokeError("catalog response is not a list")
        return [app for app in apps if isinstance(app, dict)]

    def list_servers(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/api/mcp/servers")
        if response.status_code != 200:
            raise SmokeError(f"server list request failed: HTTP {response.status_code}")
        try:
            servers = response.json()
        except ValueError:
            servers = None
        if not isinstance(servers, list):
            raise SmokeError("server list response is not a list")
        return [server for server in servers if isinstance(server, dict)]

    def list_server_tools(self, server_id: int, *, timeout: float) -> JSONResponse:
        """Start one connected server and list its tools.

        The server may take its own load timeout, after the token refresh and
        (for npx) the sandbox lease, to answer, so this request carries its own,
        longer timeout.
        """
        response = self._request(
            "GET", f"/api/mcp/servers/{int(server_id)}/tools", timeout=timeout
        )
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        return JSONResponse(response.status_code, payload)

    def probe_server(self, server_id: int, *, timeout: float) -> JSONResponse:
        """Run the read-only probe the server declares for this connector.

        The route starts the connector, as ``list_server_tools`` does, and
        then calls one tool against the vendor, so this request carries its
        own timeout too.
        """
        response = self._request(
            "POST", f"/api/mcp/servers/{int(server_id)}/probe", timeout=timeout
        )
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        return JSONResponse(response.status_code, payload)

    def start_launch(
        self, provider: str, app_id: str, account: str | None = None
    ) -> LaunchStart:
        body: dict[str, str] = {"app_id": app_id}
        if account is not None:
            body["account"] = account
        response = self._request("POST", _launch_path(provider), json=body)
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        return LaunchStart(response.status_code, payload)

    def redeem_launch(self, provider: str, authorization_url: object) -> LaunchRedeem:
        """Redeem a launch without following the redirect to the provider.

        The URL comes from the server, so it must be this provider's own
        relative launch path: anything else would send the bearer token and
        cookies to a host the check does not control.
        """
        prefix = f"{_launch_path(provider)}?"
        if not isinstance(authorization_url, str) or not authorization_url.startswith(
            prefix
        ):
            raise SmokeError("launch start returned an unexpected authorization URL")
        response = self._request("GET", authorization_url)
        return LaunchRedeem(
            response.status_code, response.headers.get("location"), response.text
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        extra_cookies: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        cookies = {**self._cookies, **(extra_cookies or {})}
        headers = {"Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items())}
        try:
            response = self._http.request(
                method, path, headers=headers if cookies else None, **kwargs
            )
        except httpx.TimeoutException as exc:
            raise SmokeTimeout(
                f"{method} request failed: {type(exc).__name__}"
            ) from None
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            # The exception text can carry the request URL, which for a launch
            # redeem holds a ticket, or a header value. Report the class only.
            # ValueError covers a header httpx cannot encode.
            raise SmokeError(f"{method} request failed: {type(exc).__name__}") from None
        finally:
            self._http.cookies.clear()
        return response


def _launch_path(provider: str) -> str:
    return f"/api/auth/{quote(provider, safe='')}/login"
