"""Login credentials in WebSocket handshake headers, never navigation URLs."""

import re

from starlette.datastructures import Headers
from starlette.types import Scope

AUTH_PROTOCOL = "xagent-auth-v1"
AUTH_TOKEN_PREFIX = "xagent-auth-token."
_TOKEN = re.compile(r"[A-Za-z0-9._~-]{1,8192}\Z")


class InvalidWebSocketCredentials(ValueError):
    """Malformed or ambiguous credentials; never include their value."""


def websocket_header_token(scope: Scope) -> str | None:
    """Accept one Bearer header or one browser subprotocol credential.

    The credential subprotocol is only an input carrier. It must never be
    selected/echoed in the upgrade response. Other purpose-scoped protocols
    (for example public widget sessions) are not login credentials.
    """
    headers = Headers(scope=scope)
    # Gateways may attach a separate HTTP authentication scheme. Only Bearer
    # is an application login carrier; malformed Bearer values still fail.
    authorization = [
        value
        for value in headers.getlist("authorization")
        if value.split(maxsplit=1) and value.split(maxsplit=1)[0].lower() == "bearer"
    ]
    protocols = [
        value.strip()
        for header in headers.getlist("sec-websocket-protocol")
        for value in header.split(",")
    ]
    credentials = [
        p[len(AUTH_TOKEN_PREFIX) :]
        for p in protocols
        if p.startswith(AUTH_TOKEN_PREFIX)
    ]
    uses_protocol = AUTH_PROTOCOL in protocols or bool(credentials)
    if authorization and uses_protocol:
        raise InvalidWebSocketCredentials("Ambiguous WebSocket credentials")
    if authorization:
        if len(authorization) != 1:
            raise InvalidWebSocketCredentials("Ambiguous WebSocket credentials")
        scheme, separator, token = authorization[0].partition(" ")
        if not separator or scheme.lower() != "bearer" or not _TOKEN.fullmatch(token):
            raise InvalidWebSocketCredentials("Invalid WebSocket credentials")
        return token
    if uses_protocol:
        if (
            protocols.count(AUTH_PROTOCOL) != 1
            or len(credentials) != 1
            or not _TOKEN.fullmatch(credentials[0])
        ):
            raise InvalidWebSocketCredentials("Invalid WebSocket credentials")
        return credentials[0]
    return None
