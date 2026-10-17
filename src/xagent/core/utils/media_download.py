"""Bounded, public-network-only downloads of untrusted provider media URLs."""

import asyncio
import math
import os
import socket
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import aiohttp
from aiohttp.abc import ResolveResult
from aiohttp.resolver import ThreadedResolver

from .security import build_ca_bundle_ssl_context, reject_private_network_host


class PublicMediaResolver(ThreadedResolver):
    """Validate the addresses returned to the connection, not a DNS preflight."""

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[ResolveResult]:
        reject_private_network_host(host)
        addresses = await super().resolve(host, port, family)
        if not addresses:
            raise ValueError("Media host did not resolve to an address")
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
    reject_private_network_host(parsed.hostname)
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
                ssl=build_ca_bundle_ssl_context(),
            )
            async with aiohttp.ClientSession(
                connector=connector,
                trust_env=False,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as session:
                for hop in range(6):
                    _validate_media_url(url)
                    async with session.get(url, allow_redirects=False) as response:
                        if response.status in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location")
                            if not location or hop == 5:
                                raise ValueError("Invalid or excessive media redirects")
                            url = urljoin(url, location)
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
                            async for chunk in response.content.iter_chunked(64 * 1024):
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
