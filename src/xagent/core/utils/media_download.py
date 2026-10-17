"""Bounded, public-network-only downloads of untrusted provider media URLs."""

import asyncio
import ipaddress
import math
import os
import socket
import ssl
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import aiohttp
from aiohttp.abc import ResolveResult
from aiohttp.resolver import ThreadedResolver

from .security import (
    _MAX_REDIRECTS,
    build_ca_bundle_ssl_context,
    reject_private_network_host,
)


@lru_cache(maxsize=1)
def _media_ssl_context(_ca_file: str | None, _ca_dir: str | None) -> ssl.SSLContext:
    # The arguments invalidate the cache when the configured CA paths change.
    return build_ca_bundle_ssl_context()


def resolve_workspace_media_path(path: Path, workspace_root: Path) -> Path:
    root = workspace_root.resolve()
    expanded = path.expanduser()
    resolved = (expanded if expanded.is_absolute() else root / expanded).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Access denied: local path is outside the workspace")
    return resolved


class PublicMediaResolver(ThreadedResolver):
    """Validate the addresses returned to the connection, not a DNS preflight."""

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[ResolveResult]:
        reject_private_network_host(host)
        addresses = await super().resolve(host, port, family)
        for address in addresses:
            reject_private_network_host(address["host"])
        return addresses


def _validate_media_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Media URL must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Media URL must not contain embedded credentials")
    # aiohttp bypasses its resolver for IP literals, so check them here too.
    host = parsed.hostname.rstrip(".")
    # aiohttp skips its resolver for decimal/shortened IPv4 spellings too.
    # The socket layer may interpret them as private addresses; never let
    # these ambiguous literals reach a connect without resolver validation.
    if ":" in host or host.replace(".", "").isdigit():
        try:
            ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("Media URL contains a non-canonical IP address") from exc
    reject_private_network_host(host)
    if parsed.port == 0:
        raise ValueError("Media URL contains an invalid port")


async def download_public_media(
    url: str,
    destination: Path,
    *,
    timeout: float,
    max_bytes: int,
    media_kind: str,
) -> None:
    """Stream atomically to disk with checked DNS, redirects, size and total time.

    Proxy environment variables are deliberately ignored. This client is only
    for provider-result downloads, not configured private model API endpoints.
    """
    if max_bytes <= 0 or timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("Media download limits must be positive")
    _validate_media_url(url)
    resolver = PublicMediaResolver()
    temporary: Path | None = None
    try:
        async with asyncio.timeout(timeout):
            connector = aiohttp.TCPConnector(
                resolver=resolver,
                use_dns_cache=False,
                ssl=await asyncio.to_thread(
                    _media_ssl_context,
                    os.getenv("SSL_CERT_FILE"),
                    os.getenv("SSL_CERT_DIR"),
                ),
            )
            async with aiohttp.ClientSession(
                connector=connector,
                trust_env=False,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as session:
                for hop in range(_MAX_REDIRECTS + 1):
                    _validate_media_url(url)
                    async with session.get(url, allow_redirects=False) as response:
                        if response.status in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location")
                            if not location or hop == _MAX_REDIRECTS:
                                raise ValueError("Invalid or excessive media redirects")
                            target = urljoin(url, location)
                            if (
                                urlsplit(url).scheme == "https"
                                and urlsplit(target).scheme == "http"
                            ):
                                raise ValueError(
                                    "Media redirects must not downgrade HTTPS"
                                )
                            url = target
                            continue
                        if response.status != 200:
                            raise RuntimeError(
                                f"Failed to download {media_kind}: HTTP {response.status}"
                            )
                        length = response.headers.get("Content-Length")
                        if length is not None and not 0 <= int(length) <= max_bytes:
                            raise ValueError("Media exceeds maximum download size")
                        with tempfile.NamedTemporaryFile(
                            dir=destination.parent, prefix=".media-", delete=False
                        ) as output:
                            temporary = Path(output.name)
                            downloaded = 0
                            async for chunk in response.content.iter_chunked(
                                1024 * 1024
                            ):
                                downloaded += len(chunk)
                                if downloaded > max_bytes:
                                    raise ValueError(
                                        "Media exceeds maximum download size"
                                    )
                                await asyncio.to_thread(output.write, chunk)
                            if not downloaded:
                                raise ValueError("Media download was empty")
                        await asyncio.to_thread(os.replace, temporary, destination)
                        temporary = None
                        return
    finally:
        await resolver.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
