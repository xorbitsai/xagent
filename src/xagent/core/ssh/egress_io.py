"""DNS-resolving egress authorization (Phase 3, the I/O half of egress).

``egress.check_ip`` is pure decision logic; this module adds the name
resolution around it. Resolving here (and rejecting if ANY address is denied)
gives the executor an early, fail-closed check before it connects. The runner
still re-checks the actual connected peer IP as the DNS-rebinding backstop.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable

from .egress import EgressPolicyConfig, check_ip
from .errors import SshError, SshErrorCode

Resolver = Callable[[str, int], Awaitable[list[str]]]


async def _default_resolver(hostname: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    # De-duplicate while preserving order; sockaddr[0] is the IP for v4 and v6.
    seen: dict[str, None] = {}
    for info in infos:
        seen.setdefault(str(info[4][0]), None)
    return list(seen)


async def resolve_and_authorize(
    hostname: str,
    port: int,
    config: EgressPolicyConfig,
    *,
    resolver: Resolver | None = None,
) -> list[str]:
    """Resolve ``hostname`` and authorize every address against the egress
    policy. Returns the resolved IPs, or raises EGRESS_DENIED if the name does
    not resolve or any resolved address is denied."""
    resolve = resolver or _default_resolver
    try:
        addresses = await resolve(hostname, port)
    except OSError as exc:
        # socket.gaierror (a subclass) on an unresolvable name would otherwise
        # bypass the executor's SshError handling and the audit sink.
        raise SshError(
            SshErrorCode.EGRESS_DENIED, "hostname did not resolve", cause=exc
        ) from exc
    if not addresses:
        raise SshError(SshErrorCode.EGRESS_DENIED, "hostname did not resolve")
    for ip in addresses:
        if not check_ip(ip, config).allowed:
            raise SshError(
                SshErrorCode.EGRESS_DENIED, "destination denied by egress policy"
            )
    return addresses
