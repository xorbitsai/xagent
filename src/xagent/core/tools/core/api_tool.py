"""
API Tool - HTTP Client for making arbitrary API calls

Supports various HTTP methods, authentication, headers, and error handling.
"""

import base64
import json
import logging
import os
from typing import Any, Dict, Mapping, Optional, Union
from urllib.parse import ParseResult, parse_qs, urlencode, urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

# Fixed or tenant-subdomain API hosts already covered by a dedicated,
# credentialed MCP connector (see src/xagent/web/tools/mcp/). A caller who
# reaches this generic, credential-less tool for one of these - typically an
# agent that couldn't find a write tool on the real connector and fell back
# to hand-rolling the REST call - has no way to attach that connector's OAuth
# token here, so the request is guaranteed to 401/403 regardless of whether
# the connector itself is healthy. That failure was previously indistinguishable
# from an actually-broken connection (see the HubSpot deal-write incident this
# guard was added for: reads kept working the whole time via the real
# connector, but the raw fallback calls' 401s got diagnosed as "HubSpot
# connection broken"). Entries are the domain as it would be matched by
# _hostname_matches_connector_domain (exact host or "*.<domain>"), not a
# literal example hostname - deputy.com/zendesk.com/salesforce.com/
# myshopify.com/posthog.com/mixpanel.com are multi-tenant services where the
# real host is a customer-specific subdomain.
#
# Google, Microsoft Graph, and Meta Graph are each only PARTIALLY covered:
# xagent's connectors wrap specific product APIs, not every API a customer
# could reach on that host. Google's per-product hosts below (Ads, Analytics,
# Search Console, Gmail, Docs, Sheets, Slides) are each dedicated to one
# product, so listing them is precise. Calendar and Drive are deliberately
# NOT listed: both share the general-purpose "www.googleapis.com" host with
# many unrelated Google APIs xagent has no connector for, and guarding that
# host would block those unrelated APIs too. graph.microsoft.com (Outlook /
# OneDrive / Teams) and graph.facebook.com (Facebook / Instagram - also the
# WhatsApp Business Cloud API's host, which xagent does not implement) can't
# be narrowed the same way since Microsoft/Meta multiplex many products onto
# one host with no separate hostname per product; the warning message below
# says so explicitly rather than claiming full coverage for either.
_KNOWN_CONNECTOR_DOMAINS: tuple[tuple[str, str], ...] = (
    ("hubapi.com", "HubSpot"),
    ("slack.com", "Slack"),
    ("api.stripe.com", "Stripe"),
    ("api.github.com", "GitHub"),
    ("graph.microsoft.com", "Microsoft Graph (Outlook/OneDrive/Teams)"),
    ("graph.facebook.com", "Meta Graph (Facebook/Instagram)"),
    ("api.linkedin.com", "LinkedIn"),
    ("api.intercom.io", "Intercom"),
    ("api.linear.app", "Linear"),
    ("api.zoom.us", "Zoom"),
    ("googleads.googleapis.com", "Google Ads"),
    ("analyticsdata.googleapis.com", "Google Analytics"),
    ("analyticsadmin.googleapis.com", "Google Analytics"),
    ("searchconsole.googleapis.com", "Google Search Console"),
    ("gmail.googleapis.com", "Gmail"),
    ("docs.googleapis.com", "Google Docs"),
    ("sheets.googleapis.com", "Google Sheets"),
    ("slides.googleapis.com", "Google Slides"),
    ("api.atlassian.com", "Jira"),
    ("api.chartmogul.com", "ChartMogul"),
    ("api.employmenthero.com", "Employment Hero"),
    ("api.myob.com", "MYOB"),
    ("zendesk.com", "Zendesk"),
    ("deputy.com", "Deputy"),
    ("myshopify.com", "Shopify"),
    ("posthog.com", "PostHog"),
    ("mixpanel.com", "Mixpanel"),
    ("salesforce.com", "Salesforce"),
    ("api.xero.com", "Xero"),
)

# Domains that match one of _KNOWN_CONNECTOR_DOMAINS by suffix but are
# actually a different, self-authenticating product the guard must not
# touch: hooks.slack.com (Slack incoming webhooks) carries its secret in the
# URL path itself, not in a header/param/token, and is unrelated to the
# credentialed Slack Web API at slack.com/api that the guard is meant to
# protect.
_SELF_AUTHENTICATING_SUBDOMAINS = frozenset({"hooks.slack.com"})


def _hostname_matches_connector_domain(hostname: str, domain: str) -> bool:
    # Deferred import: xagent.web.oauth_provider_quirks.host_matches_suffix
    # is the one shared implementation of "exact host or dot-anchored
    # subdomain" (its own docstring says so, and auth.py's Deputy/Employment
    # Hero endpoint checks already use it) - reusing it here instead of a
    # second copy avoids the two silently diverging, which already happened
    # once (this module's trailing-dot handling lives in the caller,
    # match_known_connector_domain, not duplicated into this function).
    # core doesn't import from web at module load time elsewhere in this
    # codebase (see e.g. workspace.py's deferred `from ..web...` imports),
    # so this follows that same lazy-import precedent rather than adding a
    # new top-level core->web dependency.
    from ....web.oauth_provider_quirks import host_matches_suffix

    return host_matches_suffix(hostname.lower(), domain.lower())


def match_known_connector_domain(hostname: str) -> Optional[str]:
    """Return the connector name if ``hostname`` belongs to one of
    ``_KNOWN_CONNECTOR_DOMAINS``, else ``None``."""
    # A trailing "." is a valid DNS root-label marker (api.hubapi.com. is the
    # same host as api.hubapi.com) that a caller could legitimately send;
    # without stripping it, that spelling would silently skip this check.
    hostname = hostname.lower().rstrip(".")
    if hostname in _SELF_AUTHENTICATING_SUBDOMAINS:
        return None
    for domain, label in _KNOWN_CONNECTOR_DOMAINS:
        if _hostname_matches_connector_domain(hostname, domain):
            return label
    return None


# Substrings of a header name that indicate the caller already attached
# their own credential - not just the standard "Authorization" header, since
# several of the services in _KNOWN_CONNECTOR_DOMAINS use a service-specific
# one instead (e.g. Shopify's "X-Shopify-Access-Token"). Deliberately does
# NOT include the bare "auth" or "signature" substrings: "auth" alone also
# matches non-credential headers like "X-Author" or the HTTP/2 ":authority"
# pseudo-header, and "signature" matches non-credential headers like
# GitHub's outbound "X-Hub-Signature-256" webhook-verification header - both
# would silently defeat the guard below for a call that has no real
# credential. "authorization" alone already covers the real "Authorization"
# case (it's a superset match of that word), so nothing is lost by dropping
# the broader "auth".
_AUTH_HEADER_NAME_MARKERS = (
    "authorization",
    "token",
    "api-key",
    "apikey",
)

# Query parameter names (case-insensitive) commonly used to carry an API key
# instead of a header - e.g. Google APIs' "key" parameter.
_AUTH_QUERY_PARAM_NAMES = frozenset(
    {"key", "api_key", "api-key", "apikey", "token", "access_token"}
)

# auth_type values _prepare_headers()/call_api() actually know how to turn
# into a real credential on the outgoing request. auth_token alone (without
# one of these) is inert - _prepare_headers only attaches an Authorization
# header when both auth_type and auth_token are set, and api_key_query is
# handled separately in call_api's query-param step - so counting a bare
# auth_token as "has a credential" would be wrong: the request that
# actually goes out would carry none.
_EFFECTIVE_AUTH_TYPES = frozenset({"bearer", "basic", "api_key", "api_key_query"})


def _is_blank(value: Any) -> bool:
    # str(None) is the non-empty text "None", so None must be special-cased
    # here rather than folded into the str(value).strip() check below - every
    # caller treats "blank" as "nothing was actually provided", and every one
    # of these Optional fields defaults to None when the caller omits it.
    if value is None:
        return True
    return not str(value).strip()


def _has_effective_auth_token(
    auth_type: Optional[str], auth_token: Optional[str]
) -> bool:
    if _is_blank(auth_token) or not auth_type:
        return False
    return auth_type.lower() in _EFFECTIVE_AUTH_TYPES


def _has_auth_header(headers: Optional[Mapping[str, str]]) -> bool:
    if not headers:
        return False
    for key, value in headers.items():
        if _is_blank(value):
            continue
        key_lower = str(key).lower()
        if any(marker in key_lower for marker in _AUTH_HEADER_NAME_MARKERS):
            return True
    return False


def _has_auth_query_param(
    parsed_url: ParseResult, params: Optional[Mapping[str, Any]]
) -> bool:
    if params:
        for key, value in params.items():
            if str(key).lower() in _AUTH_QUERY_PARAM_NAMES and not _is_blank(value):
                return True
    # parse_qs defaults to keep_blank_values=False, so a blank value here
    # (e.g. "?key=") is already dropped for us - nothing extra to check.
    query_keys = {key.lower() for key in parse_qs(parsed_url.query)}
    return not _AUTH_QUERY_PARAM_NAMES.isdisjoint(query_keys)


def _has_url_embedded_credentials(parsed_url: ParseResult) -> bool:
    """Whether the URL carries HTTP Basic-Auth userinfo (``user:pass@host``)
    directly, rather than in a header or query parameter."""
    return bool(parsed_url.username or parsed_url.password)


def _sanitize_final_url(url: str) -> str:
    """Strip query string and userinfo from a response URL, keeping only
    scheme/host/port/path.

    ``_make_request`` reports the actual post-redirect response URL so the
    connector-domain hint can match the host that really answered - but the
    raw URL can itself carry a credential: an api_key_query auth_token gets
    merged into the query string before the request is made, and a caller
    can embed Basic-Auth userinfo directly (``user:pass@host``). The only
    thing any consumer of this value needs is which host answered, so
    everything else is dropped here rather than trusting every call site
    that ever reads this field to remember not to log or forward it whole.
    """
    parsed = urlparse(url)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


def has_auth_credentials(
    url: str,
    headers: Optional[Mapping[str, str]],
    params: Optional[Mapping[str, Any]],
    auth_type: Optional[str],
    auth_token: Optional[str],
) -> bool:
    """Whether the caller appears to already have their own credential for
    this call, by any of the mechanisms api_call or a typical direct REST
    call supports: auth_token paired with a supported auth_type, a
    non-blank auth-looking header, a non-blank auth-looking query
    parameter (in either `params` or the URL itself), or a credential
    embedded directly in the URL (Basic-Auth userinfo).

    A name-shaped match alone isn't enough for the header/param cases - an
    empty value (e.g. {"Authorization": ""}) or a name that merely
    resembles a credential without being one (e.g. "X-Idempotency-Token",
    a request-deduplication id, not a secret) both look credential-shaped
    by name. That imprecision is why this is used only to decide whether
    to append an informational hint to an ALREADY-received 401/403 (see
    known_connector_domain_hint / adapters/vibe/api_tool.py), never to
    block a request outright: worst case here is a missed hint, not a
    request that should have succeeded being refused.
    """
    parsed_url = urlparse(url)
    return bool(
        _has_effective_auth_token(auth_type, auth_token)
        or _has_auth_header(headers)
        or _has_auth_query_param(parsed_url, params)
        or _has_url_embedded_credentials(parsed_url)
    )


def known_connector_domain_hint(connector_label: str) -> str:
    """A self-contained sentence (no leading/trailing whitespace) appended to
    an ALREADY-received 401/403 from a domain covered by a dedicated MCP
    connector, when the request carried no credential this tool recognizes
    (see has_auth_credentials). Purely informational - it never gates
    whether the request is made; a public, credential-less endpoint on the
    same host (e.g. GitHub's public repo reads) still succeeds normally and
    never sees this text. Callers append it after existing error text - see
    append_known_connector_domain_hint, which handles the punctuation/
    spacing needed to avoid a run-on sentence.
    """
    return (
        f"This looks like a {connector_label} API endpoint, and this call "
        f"carried no credential in a form this tool recognizes "
        f"(auth_type/auth_token, a credential-shaped header, or a "
        f"credential-shaped query parameter), which would explain a "
        f"401/403 here - if the dedicated {connector_label} connector "
        "tools cover this request, prefer those (they carry the account's "
        "own OAuth credential); otherwise pass your own credential via one "
        "of those mechanisms."
    )


def append_known_connector_domain_hint(
    existing_error: Optional[str], connector_label: str
) -> str:
    """Combine `existing_error` (e.g. "HTTP 401", or the size-limit error
    text `_make_request` also uses on a 401/403) with
    `known_connector_domain_hint`'s sentence, adding a period + space
    between them when `existing_error` doesn't already end in sentence
    punctuation - otherwise the two read as one run-on sentence (e.g.
    "...download aborted This looks like a HubSpot API endpoint...").
    """
    hint = known_connector_domain_hint(connector_label)
    existing_error = (existing_error or "").strip()
    if not existing_error:
        return hint
    if existing_error[-1] not in ".!?":
        existing_error += "."
    return f"{existing_error} {hint}"


class APIClientCore:
    """Core API client for making HTTP requests"""

    def __init__(
        self,
        default_timeout: int = 30,
        max_response_size: int = 10 * 1024 * 1024,  # 10MB
        default_retry_count: int = 3,
    ):
        """
        Initialize API client.

        Args:
            default_timeout: Default request timeout in seconds
            max_response_size: Maximum response size in bytes
            default_retry_count: Default number of retries on failure
        """
        self.default_timeout = default_timeout
        self.max_response_size = max_response_size
        self.default_retry_count = default_retry_count

    async def call_api(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Union[Dict[str, Any], str]] = None,
        auth_type: Optional[str] = None,
        auth_token: Optional[str] = None,
        api_key_param: str = "api_key",
        timeout: Optional[int] = None,
        retry_count: Optional[int] = None,
        allow_redirects: bool = True,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to an API.

        Args:
            url: Target URL
            method: HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)
            headers: Request headers
            params: Query parameters
            body: Request body (dict for JSON, str for raw)
            auth_type: Authentication type ('bearer', 'basic', 'api_key', 'api_key_query')
            auth_token: Authentication token/credentials (for 'basic' auth, use "username:password" format)
            api_key_param: Parameter name for API key when using 'api_key_query' auth
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
            allow_redirects: Whether to follow redirects

        Returns:
            Dictionary with success status, status_code, headers, body, and error
        """
        logger.info(
            f"🌐 API Call: {method} {url}"
            + (f" (auth: {auth_type})" if auth_type else "")
        )

        # Validate URL
        if not self._is_valid_url(url):
            return {
                "success": False,
                "status_code": 0,
                "headers": {},
                "body": None,
                "error": f"Invalid URL: {url}",
            }

        # Prepare request
        method = method.upper()
        timeout = timeout or self.default_timeout
        retry_count = (
            retry_count if retry_count is not None else self.default_retry_count
        )
        # Ensure retry_count is non-negative to avoid empty range
        retry_count = max(0, retry_count)

        # Handle API key in query parameters. Case-insensitive to match
        # _prepare_headers' auth_type.lower() convention for bearer/basic/
        # api_key below, and _has_effective_auth_token's - a caller sending
        # auth_type="API_KEY_QUERY" would otherwise get no credential
        # attached here while has_auth_credentials (which does lower()
        # first) wrongly reports one was, suppressing the connector hint
        # for exactly the resulting unauthenticated 401/403.
        request_params: Dict[str, Any] = dict(params) if params else {}
        if auth_type and auth_type.lower() == "api_key_query" and auth_token:
            request_params[api_key_param] = auth_token

        # Merge params directly into the URL string (rather than passing
        # them separately to httpx) to prevent httpx from stripping any
        # query string already present in `url` - so every param ends up
        # merged into `url` here, and `_make_request` never receives a
        # non-None `params` of its own.
        #
        # httpx.URL(...) can raise for a URL _is_valid_url's cheap
        # scheme/netloc check lets through but httpx itself rejects (e.g. a
        # non-numeric port, "http://example.com:abc/path") - only reachable
        # once request_params is non-empty, since a request with no query
        # params at all never reaches this line. Caught here so call_api
        # keeps its documented dict-return contract instead of letting an
        # exception escape uncaught before the retry loop's own try/except
        # ever gets a chance to handle it.
        if request_params:
            try:
                url = str(httpx.URL(url).copy_merge_params(request_params))
            except Exception as e:
                return {
                    "success": False,
                    "status_code": 0,
                    "headers": {},
                    "body": None,
                    "error": f"Invalid URL: {url} ({e})",
                }

        # Prepare headers
        request_headers = self._prepare_headers(headers, auth_type, auth_token, body)

        # Prepare body
        request_body = self._prepare_body(body, request_headers)

        # Get proxy configuration
        proxy_url = self._get_proxy_url()

        # Attempt request with retries
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                result = await self._make_request(
                    url=url,
                    method=method,
                    headers=request_headers,
                    # Always None: every param was already merged into
                    # `url` above, precisely so it isn't passed here too.
                    params=None,
                    data=request_body,
                    timeout=timeout,
                    proxy_url=proxy_url,
                    allow_redirects=allow_redirects,
                )
                logger.info(
                    f"✅ API Call successful: {method} {url} -> {result['status_code']}"
                )
                return result

            except Exception as e:
                last_error = e
                if attempt < retry_count:
                    logger.warning(
                        f"⚠️ API Call failed (attempt {attempt + 1}/{retry_count + 1}): {str(e)}"
                    )
                else:
                    logger.error(
                        f"❌ API Call failed after {retry_count + 1} attempts: {str(e)}"
                    )

        # All retries failed
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "body": None,
            "error": f"Request failed after {retry_count + 1} attempts: {str(last_error)}",
        }

    async def _make_request(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]],
        data: Optional[Union[str, bytes]],
        timeout: int,
        proxy_url: Optional[str],
        allow_redirects: bool,
    ) -> Dict[str, Any]:
        """Make the actual HTTP request with streaming to limit download size"""
        client_kwargs: Dict[str, Any] = {"timeout": timeout}
        if proxy_url:
            client_kwargs["proxy"] = proxy_url
            logger.debug(f"   Using proxy: {proxy_url}")

        async with httpx.AsyncClient(**client_kwargs) as client:
            # Use streaming to limit download size
            async with client.stream(
                method=method,
                url=url,
                headers=headers,
                params=params,
                content=data,
                follow_redirects=allow_redirects,
            ) as response:
                # Whether the request httpx actually sent - after any
                # redirect - still carried a credential. Derived from the
                # real sent request/headers rather than the caller's
                # original args, because httpx strips the Authorization
                # header (and a Location without a query string drops any
                # api_key_query credential) on a cross-origin redirect: a
                # request that started out credentialed can still reach a
                # known connector host with no credential attached, and the
                # adapter's hint decision needs to reflect that, not the
                # original request's credential.
                final_request_has_credential = has_auth_credentials(
                    str(response.request.url),
                    response.request.headers,
                    None,
                    None,
                    None,
                )

                # Check response size by reading only up to max_response_size
                response_size = 0
                content_chunks = []
                async for chunk in response.aiter_bytes():
                    response_size += len(chunk)
                    if response_size > self.max_response_size:
                        logger.warning(
                            f"⚠️ Response size exceeds limit ({self.max_response_size} bytes), aborting download"
                        )
                        return {
                            "success": False,
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "body": None,
                            "error": f"Response too large (exceeds {self.max_response_size} bytes), download aborted",
                            # The URL that actually produced this response,
                            # which can differ from the request's starting
                            # URL after a redirect - see final_url below.
                            "final_url": _sanitize_final_url(str(response.url)),
                            "final_request_has_credential": final_request_has_credential,
                        }
                    content_chunks.append(chunk)

                response_content = b"".join(content_chunks)

                # Parse response body
                body = self._parse_response_body_from_content(
                    response_content, dict(response.headers)
                )

                return {
                    "success": 200 <= response.status_code < 300,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": body,
                    "error": None
                    if 200 <= response.status_code < 300
                    else f"HTTP {response.status_code}",
                    # Internal-only field (not part of APICallResult's public
                    # schema - callers use dict.get so an extra key is
                    # harmless). httpx.Response.url is the URL of the final
                    # request in the redirect chain, which can be a
                    # different host than the one the caller originally
                    # asked for; api_call's known-connector-domain hint
                    # needs this to know which host actually answered.
                    # Sanitized (see _sanitize_final_url) because the raw
                    # URL can carry a credential: an api_key_query
                    # auth_token is merged into the query string before the
                    # request runs, and a caller can embed Basic-Auth
                    # userinfo directly in the URL.
                    "final_url": _sanitize_final_url(str(response.url)),
                    "final_request_has_credential": final_request_has_credential,
                }

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and scheme"""
        try:
            result = urlparse(url)
            if result.scheme not in ("http", "https"):
                logger.error(f"❌ Invalid URL scheme: {result.scheme}")
                return False
            if not result.netloc:
                logger.error("❌ Invalid URL: missing network location")
                return False
            return True
        except Exception as e:
            logger.error(f"❌ URL parsing failed: {str(e)}")
            return False

    def _prepare_headers(
        self,
        headers: Optional[Dict[str, str]],
        auth_type: Optional[str],
        auth_token: Optional[str],
        body: Optional[Union[Dict[str, Any], str]],
    ) -> Dict[str, str]:
        """Prepare request headers with authentication"""
        request_headers = {}

        # Default headers
        request_headers["User-Agent"] = "Xagent-API-Tool/1.0"
        request_headers["Accept"] = "application/json"

        # Add custom headers
        if headers:
            request_headers.update(headers)

        # Add authentication
        if auth_type and auth_token:
            auth_type_lower = auth_type.lower()
            if auth_type_lower == "bearer":
                request_headers["Authorization"] = f"Bearer {auth_token}"
            elif auth_type_lower == "basic":
                credentials = base64.b64encode(auth_token.encode()).decode()
                request_headers["Authorization"] = f"Basic {credentials}"
            elif auth_type_lower == "api_key":
                # Default to X-API-Key header
                if "x-api-key" not in [h.lower() for h in request_headers.keys()]:
                    request_headers["X-API-Key"] = auth_token

        # Set Content-Type for body
        if body and "content-type" not in [h.lower() for h in request_headers.keys()]:
            if isinstance(body, dict):
                request_headers["Content-Type"] = "application/json"

        return request_headers

    def _prepare_body(
        self, body: Optional[Union[Dict[str, Any], str]], headers: Dict[str, str]
    ) -> Optional[Union[str, bytes]]:
        """Prepare request body"""
        if body is None:
            return None

        if isinstance(body, dict):
            # Check content type case-insensitively
            content_type = ""
            for key, value in headers.items():
                if key.lower() == "content-type":
                    content_type = value
                    break

            if "application/json" in content_type:
                return json.dumps(body)
            else:
                # For dict with non-JSON content type, convert to form data
                # Use urlencode to properly escape special characters
                return urlencode(body)
        elif isinstance(body, str):
            return body
        else:
            return str(body)

    def _parse_response_body_from_content(
        self, content: bytes, headers: Dict[str, str]
    ) -> Any:
        """Parse response body from raw content based on content type"""
        if not content:
            return None

        content_type = headers.get("content-type", "").lower()

        # Try to determine encoding from content-type header
        encoding = "utf-8"
        if "charset=" in content_type:
            try:
                encoding = content_type.split("charset=")[-1].split(";")[0].strip()
            except Exception:
                pass

        try:
            if "application/json" in content_type:
                return json.loads(content.decode(encoding))
            else:
                # Return text for non-JSON responses
                return content.decode(encoding)
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse response body: {str(e)}")
            # Try to return text as fallback with ignored errors
            try:
                return content.decode(encoding, errors="ignore")
            except Exception:
                return None

    def _get_proxy_url(self) -> Optional[str]:
        """Get proxy URL from environment variables"""
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        return https_proxy or http_proxy


# Convenience function for direct usage
async def call_api(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Union[Dict[str, Any], str]] = None,
    auth_type: Optional[str] = None,
    auth_token: Optional[str] = None,
    api_key_param: str = "api_key",
    timeout: Optional[int] = None,
    retry_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Make an HTTP request to an API.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)
        headers: Request headers
        params: Query parameters
        body: Request body (dict for JSON, str for raw)
        auth_type: Authentication type ('bearer', 'basic', 'api_key', 'api_key_query')
        auth_token: Authentication token/credentials (for 'basic' auth, use "username:password" format)
        api_key_param: Parameter name for API key when using 'api_key_query' auth
        timeout: Request timeout in seconds
        retry_count: Number of retries on failure

    Returns:
        Dictionary with success status, status_code, headers, body, and error

    Example:
        >>> # GET request
        >>> result = await call_api("https://api.example.com/data")

        >>> # POST request with JSON body
        >>> result = await call_api(
        ...     "https://api.example.com/users",
        ...     method="POST",
        ...     body={"name": "John", "email": "john@example.com"},
        ...     auth_type="bearer",
        ...     auth_token="your-token"
        ... )

        >>> # GET with query parameters
        >>> result = await call_api(
        ...     "https://api.example.com/search",
        ...     params={"q": "test", "limit": 10}
        ... )
    """
    client = APIClientCore()
    return await client.call_api(
        url=url,
        method=method,
        headers=headers,
        params=params,
        body=body,
        auth_type=auth_type,
        auth_token=auth_token,
        api_key_param=api_key_param,
        timeout=timeout,
        retry_count=retry_count,
    )
