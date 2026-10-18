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
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def _address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if isinstance(address, ipaddress.IPv6Address) and address in _NAT64:
        # Classify the embedded destination but retain the original IPv6 for
        # connections. Only the well-known /96 has this unambiguous layout.
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
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
        variable = "XAGENT_MODEL_PRIVATE_ENDPOINTS"
        try:
            endpoints = get_model_private_endpoints()
            origins = set()
            for url in endpoints:
                origin = _parts(url)
                parsed = urlsplit(url)
                if (
                    "*" in str(parsed.hostname)
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError("Private model endpoints must be exact origins")
                origins.add(origin)
            self.origins = frozenset(origins)
            variable = "XAGENT_MODEL_PRIVATE_CIDRS"
            self.networks = tuple(
                ipaddress.ip_network(value) for value in get_model_private_cidrs()
            )
            for network in self.networks:
                if network.is_global or _hard_deny(network.network_address):
                    raise ValueError("Private model CIDRs must be safe private ranges")
        except ValueError:
            # Invalid operator configuration is terminal too, not an SDK
            # connection failure worth retrying. Do not echo configured URLs.
            raise PrivateNetworkHostError(
                f"Invalid deployment model network policy configuration: {variable}"
            ) from None

    def destination(self, url: str) -> tuple[str, int]:
        return self._destination(_parts(url))

    def _destination(self, origin: tuple[str, str, int]) -> tuple[str, int]:
        try:
            literal = _address(origin[1])
        except ValueError:
            if origin not in self.origins:
                reject_private_network_host(origin[1])
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
        origin = _parts(url)
        self._destination(origin)
        self._validate_addresses(origin in self.origins, addresses)

    def resolve_sync(self, url: str) -> list[str]:
        origin = _parts(url)
        host, port = self._destination(origin)
        # Preserve a caller's DNS root label: dropping it can make the resolver
        # search unrelated deployment domains for an otherwise exact origin.
        parsed = urlsplit(url)
        host = str(parsed.hostname)
        addresses = [
            str(record[4][0])
            for record in socket.getaddrinfo(
                host, port, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        ]
        self._validate_addresses(origin in self.origins, addresses)
        return addresses

    async def resolve(self, url: str) -> list[str]:
        return await asyncio.to_thread(self.resolve_sync, url)
