"""Rate limiting for public trigger callbacks and trigger CRUD APIs.

Backed by the ``limits`` package: Redis storage when ``XAGENT_REDIS_URL`` is
configured (shared limits across workers), in-memory storage otherwise
(per-process limits, fine for development and single-process deployments).

Rate limiting runs before per-request audit writes so hostile callback
traffic cannot amplify database writes; rate-limited requests leave a
deduplicated rate_limited audit trail (at most one row per source per
window) via should_audit_rate_limited.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import threading

from limits import RateLimitItem, parse
from limits.storage import MemoryStorage, storage_from_string
from limits.strategies import MovingWindowRateLimiter

from ...config import (
    get_redis_url,
    get_trigger_callback_ip_rate_limit,
    get_trigger_callback_rate_limit,
    get_trigger_crud_rate_limit,
    get_trusted_proxy_hops,
)

logger = logging.getLogger(__name__)

_CALLBACK_NAMESPACE = "trigger-callback"
_CALLBACK_IP_NAMESPACE = "trigger-callback-ip"
_CRUD_NAMESPACE = "trigger-crud"
_RATE_LIMITED_AUDIT_NAMESPACE = "trigger-callback-429-audit"
_RATE_LIMITED_AUDIT_IP_NAMESPACE = "trigger-callback-429-audit-ip"

# One rate_limited audit row per source per this window: enough signal to see
# that a caller is being throttled without turning sustained 429 traffic into
# sustained database writes.
_RATE_LIMITED_AUDIT_WINDOW = "1/5minutes"

_WORKER_COUNT_ENV_VARS = ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS")


class TriggerRateLimiter:
    """Moving-window limiter over Redis or in-process memory."""

    def __init__(self) -> None:
        redis_url = get_redis_url()
        if redis_url:
            self.storage = storage_from_string(redis_url)
            self.backend = "redis"
        else:
            self.storage = MemoryStorage()
            self.backend = "memory"
        self._limiter = MovingWindowRateLimiter(self.storage)
        self._callback_limit = _parse_rate(
            get_trigger_callback_rate_limit(), fallback="120/minute"
        )
        self._callback_ip_limit = _parse_rate(
            get_trigger_callback_ip_rate_limit(), fallback="600/minute"
        )
        self._crud_limit = _parse_rate(
            get_trigger_crud_rate_limit(), fallback="60/minute"
        )
        self._rate_limited_audit_window = parse(_RATE_LIMITED_AUDIT_WINDOW)

    def hit_callback(self, callback_id: str, remote_ip: str | None) -> bool:
        """Count one callback request; False when the limit is exceeded.

        Two buckets must both admit: per callback id + IP (protects one
        trigger from one noisy caller) and per IP alone. Without the IP-only
        bucket, an attacker rotating random callback ids would get a fresh
        bucket per request and amplify audit writes without bound.
        """
        ip_key = remote_ip or "unknown"
        if not self._limiter.hit(
            self._callback_ip_limit, _CALLBACK_IP_NAMESPACE, ip_key
        ):
            return False
        key = f"{callback_id}:{ip_key}"
        return self._limiter.hit(self._callback_limit, _CALLBACK_NAMESPACE, key)

    def hit_crud(self, user_id: int) -> bool:
        """Count one trigger CRUD request; False when the limit is exceeded."""
        return self._limiter.hit(self._crud_limit, _CRUD_NAMESPACE, str(user_id))

    def should_audit_rate_limited(
        self, callback_id: str, remote_ip: str | None
    ) -> bool:
        """Admit at most one rate_limited audit write per source per window.

        The IP-only gate is checked first so an attacker rotating garbage
        callback ids cannot mint a fresh dedup bucket per request: whatever
        the callback id, one IP gets at most one audited 429 per window.
        """
        ip_key = remote_ip or "unknown"
        if not self._limiter.hit(
            self._rate_limited_audit_window, _RATE_LIMITED_AUDIT_IP_NAMESPACE, ip_key
        ):
            return False
        key = f"{callback_id}:{ip_key}"
        return self._limiter.hit(
            self._rate_limited_audit_window, _RATE_LIMITED_AUDIT_NAMESPACE, key
        )


def _parse_rate(value: str, *, fallback: str) -> RateLimitItem:
    try:
        return parse(value)
    except ValueError:
        logger.warning(
            "Invalid trigger rate limit %r; falling back to %s", value, fallback
        )
        return parse(fallback)


_lock = threading.Lock()
_limiter: TriggerRateLimiter | None = None


def get_trigger_rate_limiter() -> TriggerRateLimiter:
    global _limiter
    if _limiter is None:
        with _lock:
            if _limiter is None:
                _limiter = TriggerRateLimiter()
    return _limiter


def reset_trigger_rate_limiter() -> None:
    """Drop the cached limiter so new env configuration takes effect (tests)."""
    global _limiter
    with _lock:
        _limiter = None


def check_callback_rate_limit(callback_id: str, remote_ip: str | None) -> bool:
    return get_trigger_rate_limiter().hit_callback(callback_id, remote_ip)


def should_audit_rate_limited_callback(callback_id: str, remote_ip: str | None) -> bool:
    return get_trigger_rate_limiter().should_audit_rate_limited(callback_id, remote_ip)


def check_trigger_crud_rate_limit(user_id: int) -> bool:
    return get_trigger_rate_limiter().hit_crud(user_id)


def _configured_worker_count() -> int:
    for env_var in _WORKER_COUNT_ENV_VARS:
        value = (os.getenv(env_var) or "").strip()
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                continue
    return 1


def warn_if_rate_limits_are_per_process() -> None:
    """Startup check: multi-process deployments need Redis for shared limits.

    Never fails startup; in-memory limiting degrades to per-process limits.
    """
    if get_redis_url():
        return
    workers = _configured_worker_count()
    if workers > 1:
        logger.warning(
            "Trigger rate limiting is using in-memory storage with %s worker "
            "processes: limits apply per process, not globally. Configure "
            "XAGENT_REDIS_URL for shared rate limits.",
            workers,
        )


def remote_ip_from_request(request: object) -> str | None:
    """Derive the caller IP, honoring trusted reverse-proxy hops.

    With ``XAGENT_TRUSTED_PROXY_HOPS=N`` (N>0), the client address is taken
    from ``X-Forwarded-For`` counting N trusted entries from the right;
    otherwise the raw socket peer address is used. The returned value is
    always a bare IP: a port appended by the proxy is stripped, and anything
    not IP-shaped falls back to the peer address.
    """
    client = getattr(request, "client", None)
    peer_host = getattr(client, "host", None)
    peer_ip = str(peer_host) if peer_host is not None else None

    hops = get_trusted_proxy_hops()
    if hops <= 0:
        return peer_ip

    headers = getattr(request, "headers", {})
    forwarded_for = headers.get("x-forwarded-for") if headers else None
    if not forwarded_for:
        return peer_ip

    entries = [
        entry.strip() for entry in str(forwarded_for).split(",") if entry.strip()
    ]
    if not entries:
        return peer_ip
    # The rightmost entry was appended by the nearest trusted proxy; walking
    # `hops` entries from the right lands on the last untrusted client.
    index = max(0, len(entries) - hops)
    candidate = entries[index]
    # Only return an actually IP-shaped value. The selected entry is
    # client-controlled whenever `hops` over-counts the real proxy chain, and
    # callers use this as rate-limit key material and persist it (#1108's
    # `widget_client_ip`), so an unvalidated header could inject an unbounded
    # arbitrary string. Fall back to the peer address rather than trusting it.
    client_ip = _client_ip_from_forwarded_entry(candidate)
    if client_ip is None:
        logger.warning(
            "Ignoring non-IP X-Forwarded-For entry %r; using peer address",
            candidate[:64],
        )
        return peer_ip
    return client_ip


def _client_ip_from_forwarded_entry(candidate: str) -> str | None:
    """Return the bare IP an X-Forwarded-For entry denotes, else None.

    The entry is returned as written, not canonicalized: equivalent spellings
    of one IPv6 address still key separate rate-limit buckets, exactly as they
    did before ports were handled.

    Bare IPs are the convention (nginx, ALB, most CDNs), but some proxies
    (Azure App Service / Application Gateway among them) append the client's
    source port as ``ip:port`` or ``[ipv6]:port``. Rejecting those outright
    would silently collapse every client behind such a proxy onto the peer
    address — one shared rate-limit bucket and, for widgets, one shared run
    quota (#1185). The port is dropped rather than kept so buckets stay keyed
    per client rather than per ephemeral connection.
    """
    # ipaddress.ip_address() accepts an IPv6 zone-id suffix, and the zone-id
    # itself is unbounded and unvalidated: `::1%` + 4000 characters parses and
    # round-trips. A zone-id names a local interface, so it is meaningless for
    # a remote client address and no proxy emits one — but leaving it accepted
    # would defeat the bound this validation exists to enforce, since the
    # result is persisted and used as Redis key material.
    if "%" in candidate:
        return None
    host, port = _split_host_port(candidate)
    if host is None:
        return None
    if port is not None and not _is_valid_port(port):
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return None
    return host


def _split_host_port(candidate: str) -> tuple[str | None, str | None]:
    """Split an entry into (host, port), with port None when absent.

    Returns ``(None, None)`` for structurally malformed bracket forms. A bare
    IPv6 literal is never split: ``2001:db8::1:8080`` is a valid address whose
    last group only looks like a port, so splitting on the final colon would
    silently rewrite it into a *different* address. Only the unambiguous forms
    carry a port: a bracketed host, and a host with exactly one colon (IPv4).
    Brackets are not checked for IPv6 specifically — RFC 3986 reserves them for
    it, and a bracketed IPv4 still yields the same string its bare form would,
    so it lands in the same bucket.
    """
    if candidate.startswith("["):
        closing = candidate.find("]")
        if closing < 0:
            return None, None
        host = candidate[1:closing]
        rest = candidate[closing + 1 :]
        if not rest:
            return host, None
        if not rest.startswith(":"):
            return None, None
        return host, rest[1:]
    if candidate.count(":") == 1:
        host, _, port = candidate.partition(":")
        return host, port
    return candidate, None


def _is_valid_port(port: str) -> bool:
    # The length bound has to come first: int() raises on digit strings longer
    # than sys.get_int_max_str_digits() (4300 by default), so a header of the
    # form `1.2.3.4:99…9` would otherwise escape as an unhandled ValueError.
    # isascii() matters too: str.isdigit() also accepts non-ASCII digit
    # characters (superscripts among them) that int() then rejects.
    return len(port) <= 5 and port.isascii() and port.isdigit() and int(port) <= 65535
