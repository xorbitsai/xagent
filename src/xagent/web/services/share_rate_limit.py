"""Rate limiting and run quotas for the public share channels (#973).

A parallel of :mod:`trigger_rate_limit` for the unauthenticated share
surfaces (``/api/share/*`` + the share websocket). It reuses the same
``limits`` substrate — Redis storage when ``XAGENT_REDIS_URL`` is configured
(shared limits across workers), in-memory otherwise (per-process, fine for
dev / single-process) — and the shared :func:`remote_ip_from_request` helper.

Two distinct concerns share the substrate:

* **Request rate limits** on the public endpoints (auth, task-create, upload)
  and per-turn on the websocket — cheap early throttles returning ``False``
  when a bucket is exceeded so the caller can raise 429 / reject the turn.
* **Per-share run quota** — a rolling window bounding the owner-billed runs a
  single share link (and a single guest within it) can start, so one public
  link cannot exhaust the owner's team quota. Rolling, not cumulative, so a
  busy-but-legitimate link self-clears instead of being permanently bricked.

Kept deliberately separate from :class:`TriggerRateLimiter`: no shared util is
extracted yet, only the stable ``remote_ip_from_request`` helper is imported.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable
from typing import TypeVar

from limits import RateLimitItem, parse
from limits.storage import MemoryStorage, storage_from_string
from limits.strategies import MovingWindowRateLimiter

from ...config import (
    get_redis_url,
    get_share_auth_ip_rate_limit,
    get_share_auth_rate_limit,
    get_share_run_guest_quota,
    get_share_run_quota,
    get_share_task_create_rate_limit,
    get_share_task_create_token_rate_limit,
    get_share_upload_rate_limit,
    get_share_ws_connect_ip_rate_limit,
    get_share_ws_turn_rate_limit,
    get_widget_upload_ip_rate_limit,
    get_widget_upload_rate_limit,
)

# remote_ip_from_request is stable, shared reverse-proxy-aware infra; import it
# rather than duplicate the XAGENT_TRUSTED_PROXY_HOPS parsing.
from .trigger_rate_limit import remote_ip_from_request

logger = logging.getLogger(__name__)

__all__ = [
    "ShareRateLimiter",
    "get_share_rate_limiter",
    "reset_share_rate_limiter",
    "remote_ip_from_request",
]

_AUTH_TOKEN_NAMESPACE = "share-auth"
_AUTH_IP_NAMESPACE = "share-auth-ip"
_TASK_CREATE_GUEST_NAMESPACE = "share-task-create"
_TASK_CREATE_TOKEN_NAMESPACE = "share-task-create-token"
_WS_TURN_NAMESPACE = "share-ws-turn"
_WS_CONNECT_IP_NAMESPACE = "share-ws-connect-ip"
_UPLOAD_NAMESPACE = "share-upload"
_WIDGET_UPLOAD_ENTITY_NAMESPACE = "widget-upload"
_WIDGET_UPLOAD_IP_NAMESPACE = "widget-upload-ip"
_RUN_SHARE_NAMESPACE = "share-run"
_RUN_GUEST_NAMESPACE = "share-run-guest"


def _parse_rate(value: str, *, fallback: str) -> RateLimitItem:
    try:
        return parse(value)
    except ValueError:
        logger.warning(
            "Invalid share rate limit %r; falling back to %s", value, fallback
        )
        return parse(fallback)


_F = TypeVar("_F", bound=Callable[..., bool])


def _fail_open(method: _F) -> _F:
    """Make an ``allow_*`` gate fail open on storage/backend errors.

    Rate limiting is a non-blocking guard, not a correctness gate: when the
    Redis backend is down or misconfigured the limiter calls raise, and on a
    public share surface that would 500 the auth/upload/create endpoints and
    tear down live websockets. A transient infra blip must not lock every
    visitor out, so any exception is logged and treated as "admit" — matching
    the fail-open run gate at the ``execute_task`` chokepoint. (A genuine
    over-limit still returns ``False`` normally; only raised errors admit.)
    """

    @functools.wraps(method)
    def wrapper(self: "ShareRateLimiter", *args: object, **kwargs: object) -> bool:
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            logger.warning(
                "Share rate limiter (%s) failed open: %s",
                method.__name__,
                exc,
                exc_info=True,
            )
            return True

    return wrapper  # type: ignore[return-value]


class ShareRateLimiter:
    """Moving-window limiter over Redis or in-process memory for share channels."""

    def __init__(self) -> None:
        redis_url = get_redis_url()
        if redis_url:
            self.storage = storage_from_string(redis_url)
            self.backend = "redis"
        else:
            self.storage = MemoryStorage()
            self.backend = "memory"
        self._limiter = MovingWindowRateLimiter(self.storage)
        self._auth_token_limit = _parse_rate(
            get_share_auth_rate_limit(), fallback="60/minute"
        )
        self._auth_ip_limit = _parse_rate(
            get_share_auth_ip_rate_limit(), fallback="300/minute"
        )
        self._task_create_guest_limit = _parse_rate(
            get_share_task_create_rate_limit(), fallback="30/minute"
        )
        self._task_create_token_limit = _parse_rate(
            get_share_task_create_token_rate_limit(), fallback="120/minute"
        )
        self._ws_turn_limit = _parse_rate(
            get_share_ws_turn_rate_limit(), fallback="60/minute"
        )
        self._ws_connect_ip_limit = _parse_rate(
            get_share_ws_connect_ip_rate_limit(), fallback="120/minute"
        )
        self._upload_limit = _parse_rate(
            get_share_upload_rate_limit(), fallback="60/minute"
        )
        self._widget_upload_entity_limit = _parse_rate(
            get_widget_upload_rate_limit(), fallback="240/minute"
        )
        self._widget_upload_ip_limit = _parse_rate(
            get_widget_upload_ip_rate_limit(), fallback="60/minute"
        )
        self._run_share_limit = _parse_rate(get_share_run_quota(), fallback="500/day")
        self._run_guest_limit = _parse_rate(
            get_share_run_guest_quota(), fallback="60/hour"
        )

    @_fail_open
    def allow_auth(self, share_token: str, remote_ip: str | None) -> bool:
        """Count one auth attempt; False when a bucket is exceeded.

        Two buckets must both admit: per caller IP (across all links) and per
        share token. No ``guest_id`` exists yet at auth time.
        """
        ip_key = remote_ip or "unknown"
        if not self._limiter.hit(self._auth_ip_limit, _AUTH_IP_NAMESPACE, ip_key):
            return False
        return self._limiter.hit(
            self._auth_token_limit, _AUTH_TOKEN_NAMESPACE, share_token or "unknown"
        )

    @_fail_open
    def allow_task_create(self, share_token: str, guest_id: str) -> bool:
        """Count one task-create; False when a bucket is exceeded.

        Per share token first (stops guest_id rotation bypassing the guest
        bucket), then per guest (the tighter, owner-cost-bearing bucket).
        """
        if not self._limiter.hit(
            self._task_create_token_limit,
            _TASK_CREATE_TOKEN_NAMESPACE,
            share_token or "unknown",
        ):
            return False
        return self._limiter.hit(
            self._task_create_guest_limit,
            _TASK_CREATE_GUEST_NAMESPACE,
            guest_id or "unknown",
        )

    @_fail_open
    def allow_ws_turn(self, guest_id: str) -> bool:
        """Count one websocket turn for a guest; False when exceeded."""
        return self._limiter.hit(
            self._ws_turn_limit, _WS_TURN_NAMESPACE, guest_id or "unknown"
        )

    @_fail_open
    def allow_ws_connect(self, remote_ip: str | None) -> bool:
        """Count one share websocket connection attempt for an IP.

        Keyed per IP because this gate runs *pre-auth* (no guest_id or share
        token exists yet): accept-before-auth means every attempt — valid or
        garbage token — otherwise completes a full 101 upgrade before being
        rejected, so the handshake itself needs a budget (#993 F5).
        """
        return self._limiter.hit(
            self._ws_connect_ip_limit, _WS_CONNECT_IP_NAMESPACE, remote_ip or "unknown"
        )

    @_fail_open
    def allow_upload(self, guest_id: str) -> bool:
        """Count one share upload for a guest; False when exceeded."""
        return self._limiter.hit(
            self._upload_limit, _UPLOAD_NAMESPACE, guest_id or "unknown"
        )

    @_fail_open
    def allow_widget_upload(self, entity_key: str, remote_ip: str | None) -> bool:
        """Count one widget upload; False when a bucket is exceeded.

        Keyed on the widget entity (``agent:<id>`` / ``workforce:<id>``) plus
        the caller IP — NOT the widget ``guest_id``, which unlike the share
        path is client-supplied and therefore rotatable at will. Per-IP is
        the tighter bucket (bounds one abuser); per-entity is the loose
        backstop bounding total orphan/storage production through one widget.
        """
        if not self._limiter.hit(
            self._widget_upload_ip_limit,
            _WIDGET_UPLOAD_IP_NAMESPACE,
            remote_ip or "unknown",
        ):
            return False
        return self._limiter.hit(
            self._widget_upload_entity_limit,
            _WIDGET_UPLOAD_ENTITY_NAMESPACE,
            entity_key or "unknown",
        )

    @_fail_open
    def allow_run(self, share_key: str, guest_id: str) -> bool:
        """Count one owner-billed share run; False when a quota is exceeded.

        ``share_key`` identifies the link (e.g. ``"agent:42"`` /
        ``"workforce:7"``). Both the per-share daily quota and the shorter
        per-guest window must admit. Unlike the request throttles, this bounds
        real billing, so both buckets are tested non-destructively first and
        only consumed when both admit — a denial by either never burns a slot
        in the other. (The test→hit gap is not atomic: every request that
        passes ``test()`` before the racing ``hit()`` calls land is admitted,
        so the overshoot bound is the number of requests racing within that
        window across all workers, not a fixed margin. Acceptable for a soft
        quota, and it fails toward allowing.)
        """
        share_key = share_key or "unknown"
        guest_id = guest_id or "unknown"
        if not self._limiter.test(
            self._run_share_limit, _RUN_SHARE_NAMESPACE, share_key
        ):
            return False
        if not self._limiter.test(
            self._run_guest_limit, _RUN_GUEST_NAMESPACE, guest_id
        ):
            return False
        self._limiter.hit(self._run_share_limit, _RUN_SHARE_NAMESPACE, share_key)
        self._limiter.hit(self._run_guest_limit, _RUN_GUEST_NAMESPACE, guest_id)
        return True


_lock = threading.Lock()
_limiter: ShareRateLimiter | None = None


def get_share_rate_limiter() -> ShareRateLimiter:
    global _limiter
    if _limiter is None:
        with _lock:
            if _limiter is None:
                _limiter = ShareRateLimiter()
    return _limiter


def reset_share_rate_limiter() -> None:
    """Drop the cached limiter so new env configuration takes effect (tests)."""
    global _limiter
    with _lock:
        _limiter = None
