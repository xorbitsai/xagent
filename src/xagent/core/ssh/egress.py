"""Egress policy for SSH destinations. Pure decision logic over a single IP.

DNS resolution and re-confirming the connected peer IP (DNS-rebinding defense)
are I/O and belong to the executor (Phase 3). This module only decides whether
an already-resolved IP is permitted, so it can be exhaustively unit tested.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field

_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
_IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


def _parse_cidrs(cidrs: tuple[str, ...]) -> tuple[_IPNetwork, ...]:
    parsed: list[_IPNetwork] = []
    for cidr in cidrs:
        try:
            parsed.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            continue
    return tuple(parsed)


# Cloud instance metadata endpoints. 169.254.169.254 is link-local (already
# caught when deny_link_local is on) but is denied explicitly so it stays
# blocked even if a deployment disables the broad link-local rule.
#
# Cloud instance metadata endpoints, parsed so matching is numeric (a raw
# string compare would miss the IPv4-mapped form ::ffff:169.254.169.254).
_METADATA_ADDRESSES: frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address] = (
    frozenset(ipaddress.ip_address(a) for a in ("169.254.169.254", "fd00:ec2::254"))
)

# Tunnels that embed an IPv4 address inside an IPv6 one. Left undecoded, a
# 6to4/NAT64 address wrapping a private/loopback/metadata target classifies as
# an ordinary public IPv6 and slips past every deny rule; decode it so the
# embedded IPv4 is what the policy actually judges.
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_6TO4 = ipaddress.ip_network("2002::/16")


def _decode_tunneled_ipv4(addr: _IPAddress) -> ipaddress.IPv4Address | None:
    """Return the IPv4 embedded in a NAT64/6to4 IPv6 address, else None."""
    if not isinstance(addr, ipaddress.IPv6Address):
        return None
    packed = addr.packed
    if addr in _NAT64:
        return ipaddress.IPv4Address(packed[12:16])  # low 32 bits
    if addr in _6TO4:
        return ipaddress.IPv4Address(packed[2:6])  # bits 16..48
    return None


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
    # Pre-parsed once so check_ip doesn't re-parse CIDR strings on every call.
    _allow_networks: tuple[_IPNetwork, ...] = field(
        init=False, repr=False, compare=False
    )
    _denied_networks: tuple[_IPNetwork, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_allow_networks", _parse_cidrs(self.allow_cidrs))
        object.__setattr__(
            self, "_denied_networks", _parse_cidrs(self.extra_denied_cidrs)
        )


@dataclass(frozen=True)
class EgressDecision:
    """Outcome of an egress check."""

    allowed: bool
    reason: str


def _in_any(addr: _IPAddress, networks: tuple[_IPNetwork, ...]) -> bool:
    return any(
        addr.version == network.version and addr in network for network in networks
    )


def check_ip(ip: str, config: EgressPolicyConfig) -> EgressDecision:
    """Decide whether ``ip`` may be connected to under ``config``."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return EgressDecision(False, "invalid ip address")

    # Normalize IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) to its IPv4 form so
    # dual-stack representations classify identically to the bare IPv4 address.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped

    # Decode tunnel-embedded IPv4 (NAT64/6to4) so a wrapped private/metadata
    # target is judged as its real IPv4 rather than an "ordinary public IPv6".
    tunneled = _decode_tunneled_ipv4(addr)
    if tunneled is not None:
        addr = tunneled

    # The cloud metadata address is uniquely dangerous (SSRF into instance
    # credentials), so its deny sits ahead of the allowlist — a broad
    # allow_cidrs (e.g. 0.0.0.0/0) can open private ranges but never metadata.
    if config.deny_metadata and addr in _METADATA_ADDRESSES:
        return EgressDecision(False, "cloud metadata address denied")

    # Explicit allowlist wins over the remaining deny rules.
    if _in_any(addr, config._allow_networks):
        return EgressDecision(True, "allowlisted")

    if config.deny_loopback and addr.is_loopback:
        return EgressDecision(False, "loopback address denied")
    if config.deny_link_local and addr.is_link_local:
        return EgressDecision(False, "link-local address denied")
    if config.deny_private and addr.is_private:
        return EgressDecision(False, "private address denied")
    # Multicast (e.g. 224.0.0.0/4) is never a valid unicast SSH peer, so deny it
    # unconditionally — independent of deny_private, which is orthogonal: a
    # deny_private=False deployment must still not connect to multicast (m6).
    if addr.is_multicast:
        return EgressDecision(False, "multicast address denied")
    if _in_any(addr, config._denied_networks):
        return EgressDecision(False, "denied by reserved cidr")

    if config.default_allow_public:
        return EgressDecision(True, "public address allowed")
    return EgressDecision(False, "not in allowlist (deny by default)")
