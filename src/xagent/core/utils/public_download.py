"""Bounded public-only downloads for untrusted, caller-supplied input URLs."""

import asyncio
import math

import httpx

from .security import PublicHttpResponse, fetch_public_http_bytes


class PublicDownloadError(ValueError):
    """A download failure whose public message cannot expose a secret URL."""


async def download_public_input(
    url: str, *, max_bytes: int, timeout: float
) -> PublicHttpResponse:
    """Use DNS pinning or an explicitly trusted policy-enforcing egress proxy.

    The deadline includes DNS, every redirect and streaming the identity body.
    No caller/model API credentials are attached. Errors deliberately omit
    raw URLs, redirect Locations and third-party response text.
    """
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes <= 0
        or timeout <= 0
        or not math.isfinite(timeout)
    ):
        raise ValueError("Public download limits must be positive and finite")
    # Import lazily: web_content owns the existing deployment proxy policy.
    from ..tools.core.web_content import (
        get_trusted_proxy_url,
        public_fetch_client,
    )

    try:
        async with asyncio.timeout(timeout):
            proxy_url = get_trusted_proxy_url()
            async with public_fetch_client(proxy_url, timeout=timeout) as client:
                return await fetch_public_http_bytes(
                    client,
                    url,
                    max_content_bytes=max_bytes,
                    timeout=timeout,
                    resource_name="remote input",
                    require_non_empty=True,
                    via_proxy=bool(proxy_url),
                    allow_https_downgrade=False,
                    require_identity_encoding=True,
                    required_status_code=200,
                )
    except (TimeoutError, httpx.TimeoutException):
        raise PublicDownloadError("Remote input download timed out") from None
    except httpx.HTTPStatusError as exc:
        raise PublicDownloadError(
            f"Remote input returned HTTP {exc.response.status_code}"
        ) from None
    except (httpx.RequestError, OSError):
        raise PublicDownloadError("Remote input connection failed") from None
    except (ValueError, httpx.InvalidURL):
        raise PublicDownloadError(
            "Remote input violated the public-network, redirect or size policy"
        ) from None
