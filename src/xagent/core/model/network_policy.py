"""Default-public model policy with explicit, deployment-owned private exceptions."""

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

from ...config import get_model_private_cidrs, get_model_private_endpoints
from ..utils.security import (
    PrivateNetworkHostError,
    reject_private_network_host,
    validate_public_http_url_shape,
)

_METADATA_HOSTS = frozenset(
    {
        "metadata",
        "metadata.google.internal",
        "instance-data",
        "instance-data.ec2.internal",
    }
)
_METADATA_IPS = frozenset(
    {"169.254.169.254", "100.100.100.200", "168.63.129.16", "fd00:ec2::254"}
)


def _address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _hard_deny(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        str(address) in _METADATA_IPS
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or (address.is_reserved and not address.is_loopback)
        or (
            isinstance(address, ipaddress.IPv4Address)
            and address in ipaddress.ip_network("0.0.0.0/8")
        )
    )


def _parts(url: str) -> tuple[str, str, int]:
    # Reuse the shared normalized shape/port/userinfo checks. Only the final
    # destination decision is model-specific; input downloads never use it.
    parsed = validate_public_http_url_shape(url, host_validator=lambda host: None)
    host = str(parsed.hostname).rstrip(".")
    if host in _METADATA_HOSTS:
        raise PrivateNetworkHostError("Model metadata destinations are forbidden")
    try:
        address = _address(host)
    except ValueError:
        pass
    else:
        if _hard_deny(address):
            raise PrivateNetworkHostError("Model metadata/unsafe address is forbidden")
    return parsed.scheme, host, parsed.port or (443 if parsed.scheme == "https" else 80)


class ModelNetworkPolicy:
    def __init__(self) -> None:
        endpoints = get_model_private_endpoints()
        self.origins: frozenset[tuple[str, str, int]] = frozenset(
            _parts(url) for url in endpoints
        )
        for url in endpoints:
            parsed = urlsplit(url)
            if (
                "*" in str(parsed.hostname)
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "Private model endpoints must be exact origins, without path/query/fragment"
                )
        self.networks = tuple(
            ipaddress.ip_network(value) for value in get_model_private_cidrs()
        )
        for network in self.networks:
            if network.is_global or _hard_deny(network.network_address):
                raise ValueError(
                    "Private model CIDRs must be private/loopback ranges, not metadata or unsafe ranges"
                )

    def destination(self, url: str) -> tuple[str, int]:
        origin = _parts(url)
        if origin not in self.origins:
            reject_private_network_host(origin[1])
        try:
            literal = _address(origin[1])
        except ValueError:
            pass
        else:
            self._validate_addresses(origin in self.origins, [str(literal)])
        return origin[1], origin[2]

    def _validate_addresses(self, approved: bool, addresses: list[str]) -> None:
        if not addresses:
            raise PrivateNetworkHostError("Model host did not resolve")
        for value in addresses:
            address = _address(value)
            if _hard_deny(address):
                raise PrivateNetworkHostError(
                    "Model metadata/unsafe address is forbidden"
                )
            if address.is_global:
                continue
            if not approved or not any(
                address in network
                for network in self.networks
                if address.version == network.version
            ):
                raise PrivateNetworkHostError(
                    "Private model destination is not approved by deployment policy"
                )

    def validate_resolved(self, url: str, addresses: list[str]) -> None:
        self.destination(url)
        self._validate_addresses(_parts(url) in self.origins, addresses)

    def resolve_sync(self, url: str) -> list[str]:
        host, port = self.destination(url)
        addresses = [
            str(record[4][0])
            for record in socket.getaddrinfo(
                host, port, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        ]
        self._validate_addresses(_parts(url) in self.origins, addresses)
        return addresses

    async def resolve(self, url: str) -> list[str]:
        return await asyncio.to_thread(self.resolve_sync, url)
