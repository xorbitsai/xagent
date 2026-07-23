"""Egress policy for SSH destinations. Pure decision logic over a single IP.

DNS resolution and re-confirming the connected peer IP (DNS-rebinding defense)
are I/O and belong to the executor (Phase 3). This module only decides whether
an already-resolved IP is permitted, so it can be exhaustively unit tested.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

# Cloud instance metadata endpoints. 169.254.169.254 is link-local (already
# caught when deny_link_local is on) but is denied explicitly so it stays
# blocked even if a deployment disables the broad link-local rule.
#
# Cloud instance metadata endpoints, parsed so matching is numeric (a raw
# string compare would miss the IPv4-mapped form ::ffff:169.254.169.254).
_METADATA_ADDRESSES: frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address] = frozenset(
    ipaddress.ip_address(a) for a in ("169.254.169.254", "fd00:ec2::254")
)


@dataclass(frozen=True)
class EgressPolicyConfig:
    """Deployment-configurable egress rules.

    allow_cidrs wins over every deny rule, so operators can open a customer
    private network explicitly. default_allow_public=False turns the policy
    into deny-by-default (only allow_cidrs permitted).
    """

    deny_loopback: bool = True
    deny_link_local: bool = True
    deny_private: bool = True
    deny_metadata: bool = True
    default_allow_public: bool = True
    allow_cidrs: tuple[str, ...] = ()
    extra_denied_cidrs: tuple[str, ...] = ()


@dataclass(frozen=True)
class EgressDecision:
    """Outcome of an egress check."""

    allowed: bool
    reason: str


def _in_any(addr: ipaddress._BaseAddress, cidrs: tuple[str, ...]) -> bool:
    for cidr in cidrs:
        try:
            network = ipaddress.ip_network(cidr, strict=False)
        except ValueError:
            continue
        if addr.version == network.version and addr in network:
            return True
    return False


def check_ip(ip: str, config: EgressPolicyConfig) -> EgressDecision:
    """Decide whether ``ip`` may be connected to under ``config``."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return EgressDecision(False, "invalid ip address")

    # Normalize IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) to its IPv4 form so
    # dual-stack representations classify identically to the bare IPv4 address.
    # NOTE: tunnel-embedded IPv4 (NAT64 64:ff9b::/96, 6to4 2002::/16) is NOT
    # decoded here; the executor (Phase 3) re-checks the actual connected peer
    # IP, which closes that class of gap.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped

    # Explicit allowlist wins over all deny rules.
    if _in_any(addr, config.allow_cidrs):
        return EgressDecision(True, "allowlisted")

    if config.deny_metadata and addr in _METADATA_ADDRESSES:
        return EgressDecision(False, "cloud metadata address denied")
    if config.deny_loopback and addr.is_loopback:
        return EgressDecision(False, "loopback address denied")
    if config.deny_link_local and addr.is_link_local:
        return EgressDecision(False, "link-local address denied")
    if config.deny_private and addr.is_private:
        return EgressDecision(False, "private address denied")
    if _in_any(addr, config.extra_denied_cidrs):
        return EgressDecision(False, "denied by reserved cidr")

    if config.default_allow_public:
        return EgressDecision(True, "public address allowed")
    return EgressDecision(False, "not in allowlist (deny by default)")
