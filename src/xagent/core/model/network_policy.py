"""Default-public outbound policy with explicit, deployment-owned private exceptions."""

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

from ...config import (
    MODEL_PRIVATE_CIDRS,
    MODEL_PRIVATE_ENDPOINTS,
    get_model_private_cidrs,
    get_model_private_endpoints,
)
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


def _parts(url: str, label: str) -> tuple[str, str, int]:
    # Reuse the shared normalized shape/port/userinfo checks. Only the final
    # destination decision is policy-specific; input downloads never use it.
    parsed = validate_public_http_url_shape(url, host_validator=lambda host: None)
    host = str(parsed.hostname).rstrip(".")
    if host in _METADATA_HOSTS:
        raise PrivateNetworkHostError(f"{label} metadata destinations are forbidden")
    try:
        address = _address(host)
    except ValueError:
        pass
    else:
        if _hard_deny(address):
            raise PrivateNetworkHostError(
                f"{label} metadata/unsafe address is forbidden"
            )
    return parsed.scheme, host, parsed.port or (443 if parsed.scheme == "https" else 80)


class PrivateEndpointPolicy:
    label: str
    noun: str
    endpoints_variable: str
    cidrs_variable: str

    def private_endpoints(self) -> list[str]:
        raise NotImplementedError

    def private_cidrs(self) -> list[str]:
        raise NotImplementedError

    def __init__(self) -> None:
        variable = self.endpoints_variable
        try:
            endpoints = self.private_endpoints()
            origins = set()
            for url in endpoints:
                origin = _parts(url, self.label)
                parsed = urlsplit(url)
                if (
                    "*" in str(parsed.hostname)
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError("Private endpoints must be exact origins")
                origins.add(origin)
            self.origins = frozenset(origins)
            variable = self.cidrs_variable
            self.networks = tuple(
                ipaddress.ip_network(value) for value in self.private_cidrs()
            )
            for network in self.networks:
                if network.is_global or _hard_deny(network.network_address):
                    raise ValueError("Private CIDRs must be safe private ranges")
        except ValueError:
            # Invalid operator configuration is terminal too, not an SDK
            # connection failure worth retrying. Do not echo configured URLs.
            raise PrivateNetworkHostError(
                f"Invalid deployment {self.noun} network policy "
                f"configuration: {variable}"
            ) from None

    def approves(self, url: str) -> bool:
        try:
            return _parts(url, self.label) in self.origins
        except ValueError:
            return False

    def destination(self, url: str) -> tuple[str, int]:
        return self._destination(_parts(url, self.label))

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
            raise PrivateNetworkHostError(f"{self.label} host did not resolve")
        for value in addresses:
            address = _address(value)
            if _hard_deny(address):
                raise PrivateNetworkHostError(
                    f"{self.label} metadata/unsafe address is forbidden"
                )
            if address.is_global:
                continue
            if not approved or not any(
                address in network
                for network in self.networks
                if address.version == network.version
            ):
                raise PrivateNetworkHostError(
                    f"Private {self.noun} destination is not approved "
                    "by deployment policy"
                )

    def validate_resolved(self, url: str, addresses: list[str]) -> None:
        origin = _parts(url, self.label)
        self._destination(origin)
        self._validate_addresses(origin in self.origins, addresses)

    def resolve_sync(self, url: str) -> list[str]:
        origin = _parts(url, self.label)
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


class ModelNetworkPolicy(PrivateEndpointPolicy):
    label = "Model"
    noun = "model"
    endpoints_variable = MODEL_PRIVATE_ENDPOINTS
    cidrs_variable = MODEL_PRIVATE_CIDRS

    def private_endpoints(self) -> list[str]:
        return get_model_private_endpoints()

    def private_cidrs(self) -> list[str]:
        return get_model_private_cidrs()
