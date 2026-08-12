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
    get_widget_auth_ip_rate_limit,
    get_widget_auth_rate_limit,
    get_widget_run_ip_quota,
    get_widget_run_quota,
    get_widget_task_create_ip_rate_limit,
    get_widget_task_create_rate_limit,
    get_widget_upload_ip_rate_limit,
    get_widget_upload_rate_limit,
    get_widget_ws_connect_ip_rate_limit,
    get_widget_ws_turn_ip_rate_limit,
    get_widget_ws_turn_rate_limit,
)

# remote_ip_from_request is stable, shared reverse-proxy-aware infra; import it
# rather than duplicate the XAGENT_TRUSTED_PROXY_HOPS parsing.
from .trigger_rate_limit import remote_ip_from_request

logger = logging.getLogger(__name__)

__all__ = [
    "ShareRateLimiter",
    "entity_rate_limit_key",
    "get_share_rate_limiter",
    "reset_share_rate_limiter",
    "remote_ip_from_request",
]


def entity_rate_limit_key(agent_id: int | None, workforce_id: int | None) -> str | None:
    """Rate-limit/quota key for an agent or workforce entity.

    The single formatter behind every entity-keyed bucket (``"agent:<id>"``
    / ``"workforce:<id>"``), shared by the widget upload/turn gates and the
    share run quota so the shape can never drift between call sites.
    Workforce wins when both ids are set (callers guarantee at most one is;
    on the share path ``ShareChatAccessContext.__post_init__`` enforces that
    structurally, #1225).
    Returns ``None`` when neither id is set; the request-throttle ``allow_*``
    gates degrade that to their shared ``"unknown"`` bucket rather than
    admitting freely, while the run-quota chokepoint (chat.py) instead admits
    a task it cannot attribute — matching the original share-gate behaviour of
    never blocking a run on a missing marker.
    """
    if workforce_id is not None:
        return f"workforce:{workforce_id}"
    if agent_id is not None:
        return f"agent:{agent_id}"
    return None


_AUTH_TOKEN_NAMESPACE = "share-auth"
_AUTH_IP_NAMESPACE = "share-auth-ip"
_TASK_CREATE_GUEST_NAMESPACE = "share-task-create"
_TASK_CREATE_TOKEN_NAMESPACE = "share-task-create-token"
_WS_TURN_NAMESPACE = "share-ws-turn"
_WS_CONNECT_IP_NAMESPACE = "share-ws-connect-ip"
_UPLOAD_NAMESPACE = "share-upload"
_WIDGET_UPLOAD_ENTITY_NAMESPACE = "widget-upload"
_WIDGET_UPLOAD_IP_NAMESPACE = "widget-upload-ip"
_WIDGET_WS_TURN_ENTITY_NAMESPACE = "widget-ws-turn"
_WIDGET_WS_TURN_IP_NAMESPACE = "widget-ws-turn-ip"
_WIDGET_WS_CONNECT_IP_NAMESPACE = "widget-ws-connect-ip"
_WIDGET_AUTH_ENTITY_NAMESPACE = "widget-auth"
_WIDGET_AUTH_IP_NAMESPACE = "widget-auth-ip"
_WIDGET_TASK_CREATE_ENTITY_NAMESPACE = "widget-task-create"
_WIDGET_TASK_CREATE_IP_NAMESPACE = "widget-task-create-ip"
_RUN_SHARE_NAMESPACE = "share-run"
_RUN_GUEST_NAMESPACE = "share-run-guest"
_WIDGET_RUN_NAMESPACE = "widget-run"
_WIDGET_RUN_IP_NAMESPACE = "widget-run-ip"


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


_D = TypeVar("_D", bound=Callable[..., "str | None"])


def _fail_open_denial(method: _D) -> _D:
    """:func:`_fail_open` for gates that return a denial reason, not a bool.

    Same contract — a raising storage backend must admit rather than 500 a
    public surface — expressed as ``None`` ("no bucket refused") instead of
    ``True``.
    """

    @functools.wraps(method)
    def wrapper(
        self: "ShareRateLimiter", *args: object, **kwargs: object
    ) -> str | None:
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            logger.warning(
                "Share rate limiter (%s) failed open: %s",
                method.__name__,
                exc,
                exc_info=True,
            )
            return None

    return wrapper  # type: ignore[return-value]


class ShareRateLimiter:
    """Moving-window limiter over Redis or in-process memory for share channels."""

    def __init__(self) -> None:
        redis_url = get_redis_url()
        if redis_url:
            # Degrade to in-process memory if the Redis URL is unusable
            # (invalid scheme, unreachable at build time) rather than letting
            # storage_from_string raise: this constructor runs lazily on the
            # first request to a public endpoint, outside the per-call
            # @_fail_open boundary, so an exception here would 500 the auth /
            # task-create / embed-ticket surfaces instead of failing open. A
            # process-local limiter still throttles (per-worker, not shared),
            # which is strictly better than no gate. Mirrors the hot-path
            # cache's degrade-to-memory fallback.
            try:
                self.storage = storage_from_string(redis_url)
                self.backend = "redis"
            except Exception as exc:
                logger.warning(
                    "Share rate limiter: Redis storage %r unusable (%s); "
                    "falling back to in-process memory",
                    redis_url,
                    exc,
                    exc_info=True,
                )
                self.storage = MemoryStorage()
                self.backend = "memory"
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
        self._widget_ws_connect_ip_limit = _parse_rate(
            get_widget_ws_connect_ip_rate_limit(), fallback="120/minute"
        )
        self._widget_ws_turn_ip_limit = _parse_rate(
            get_widget_ws_turn_ip_rate_limit(), fallback="60/minute"
        )
        self._widget_ws_turn_entity_limit = _parse_rate(
            get_widget_ws_turn_rate_limit(), fallback="240/minute"
        )
        self._widget_auth_entity_limit = _parse_rate(
            get_widget_auth_rate_limit(), fallback="1200/minute"
        )
        self._widget_auth_ip_limit = _parse_rate(
            get_widget_auth_ip_rate_limit(), fallback="300/minute"
        )
        self._widget_task_create_entity_limit = _parse_rate(
            get_widget_task_create_rate_limit(), fallback="240/minute"
        )
        self._widget_task_create_ip_limit = _parse_rate(
            get_widget_task_create_ip_rate_limit(), fallback="60/minute"
        )
        self._run_share_limit = _parse_rate(get_share_run_quota(), fallback="500/day")
        self._run_guest_limit = _parse_rate(
            get_share_run_guest_quota(), fallback="60/hour"
        )
        self._widget_run_limit = _parse_rate(get_widget_run_quota(), fallback="500/day")
        self._widget_run_ip_limit = _parse_rate(
            get_widget_run_ip_quota(), fallback="120/hour"
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

    def _denial_from_ip_and_entity(
        self,
        ip_limit: RateLimitItem,
        ip_namespace: str,
        ip_key: str,
        entity_limit: RateLimitItem,
        entity_namespace: str,
        entity_key: str,
    ) -> str | None:
        """Paired-bucket admission reporting *which* bucket refused.

        ``None`` admits; ``"ip"`` / ``"entity"`` name the bucket that refused,
        so a caller can tell a per-caller throttle (the visitor should retry)
        apart from an owner-budget exhaustion (the visitor cannot act on it).
        Keys arrive pre-normalized. See :meth:`_admit_ip_and_entity` for the
        non-destructive test→hit contract this implements.
        """
        if not self._limiter.test(ip_limit, ip_namespace, ip_key):
            return "ip"
        if not self._limiter.test(entity_limit, entity_namespace, entity_key):
            return "entity"
        self._limiter.hit(ip_limit, ip_namespace, ip_key)
        self._limiter.hit(entity_limit, entity_namespace, entity_key)
        return None

    def _admit_ip_and_entity(
        self,
        ip_limit: RateLimitItem,
        ip_namespace: str,
        remote_ip: str | None,
        entity_limit: RateLimitItem,
        entity_namespace: str,
        entity_key: str | None,
    ) -> bool:
        """Admit one request through a paired IP (tight) + entity (loose) gate.

        The shared shape behind the widget gates: per-IP bounds one abuser
        (the widget ``guest_id`` is client-supplied, so IP is the only
        per-abuser key available), per-entity is the loose backstop across
        all callers of one widget. Both buckets are tested non-destructively
        and only consumed when both admit — a denial by either never burns a
        slot in the other (the :meth:`allow_run` shape). The test→hit gap is
        not atomic, so requests racing within it can overshoot slightly;
        acceptable for soft throttles, and it fails toward allowing.
        """
        ip_key = remote_ip or "unknown"
        entity = entity_key or "unknown"
        if not self._limiter.test(ip_limit, ip_namespace, ip_key):
            return False
        if not self._limiter.test(entity_limit, entity_namespace, entity):
            return False
        self._limiter.hit(ip_limit, ip_namespace, ip_key)
        self._limiter.hit(entity_limit, entity_namespace, entity)
        return True

    @_fail_open
    def allow_widget_ws_connect(self, remote_ip: str | None) -> bool:
        """Count one widget websocket connection attempt for an IP (#1056).

        The widget mirror of :meth:`allow_ws_connect`, in its own bucket so
        probes against one public channel cannot consume the other's budget.
        Keyed per IP because the gate runs pre-auth (no guest or entity is
        known yet).
        """
        return self._limiter.hit(
            self._widget_ws_connect_ip_limit,
            _WIDGET_WS_CONNECT_IP_NAMESPACE,
            remote_ip or "unknown",
        )

    @_fail_open
    def allow_widget_ws_turn(
        self, entity_key: str | None, remote_ip: str | None
    ) -> bool:
        """Count one widget websocket turn; False when a bucket is exceeded.

        Keyed on the widget entity (``agent:<id>`` / ``workforce:<id>``) plus
        the caller IP — NOT the widget ``guest_id``, which unlike the share
        path is client-supplied and therefore rotatable at will (the same
        reasoning as :meth:`allow_widget_upload`).
        """
        return self._admit_ip_and_entity(
            self._widget_ws_turn_ip_limit,
            _WIDGET_WS_TURN_IP_NAMESPACE,
            remote_ip,
            self._widget_ws_turn_entity_limit,
            _WIDGET_WS_TURN_ENTITY_NAMESPACE,
            entity_key,
        )

    @_fail_open
    def allow_widget_auth(self, entity_key: str, remote_ip: str | None) -> bool:
        """Count one widget auth / embed-ticket attempt; False when exceeded (#1108).

        Uses the tight-IP / loose-entity pairing every other widget gate uses,
        NOT the share-``auth`` loose-IP / tight-credential shape. The auth and
        embed-ticket endpoints fire on *every* widget page load, and the entity
        key is shared by all of one widget's visitors, so a tight per-entity
        bucket would 429 ordinary visitors on a busy embed. Instead the per-IP
        bucket is the per-visitor / per-abuser bound (visitors have distinct
        IPs), and the per-entity bucket is a loose aggregate backstop across
        all callers of one widget.

        ``entity_key`` must be *stable*, derived without DB work — the widget
        key, or the owner entity decoded from the embed ticket's signed claims
        (pure crypto) — NOT the raw ticket, which the embedded flow mints fresh
        per page load (a raw-ticket bucket would never accumulate). Both buckets
        are tested non-destructively and only consumed when both admit, so an
        entity-backstop denial never burns the caller's per-IP allowance for
        *other* widgets (the :meth:`allow_widget_upload` shape).
        """
        return self._admit_ip_and_entity(
            self._widget_auth_ip_limit,
            _WIDGET_AUTH_IP_NAMESPACE,
            remote_ip,
            self._widget_auth_entity_limit,
            _WIDGET_AUTH_ENTITY_NAMESPACE,
            entity_key,
        )

    @_fail_open
    def allow_upload(self, guest_id: str) -> bool:
        """Count one share upload for a guest; False when exceeded."""
        return self._limiter.hit(
            self._upload_limit, _UPLOAD_NAMESPACE, guest_id or "unknown"
        )

    @_fail_open
    def allow_widget_upload(
        self, entity_key: str | None, remote_ip: str | None
    ) -> bool:
        """Count one widget upload; False when a bucket is exceeded.

        Keyed on the widget entity (``agent:<id>`` / ``workforce:<id>``) plus
        the caller IP — NOT the widget ``guest_id``, which unlike the share
        path is client-supplied and therefore rotatable at will.
        """
        return self._admit_ip_and_entity(
            self._widget_upload_ip_limit,
            _WIDGET_UPLOAD_IP_NAMESPACE,
            remote_ip,
            self._widget_upload_entity_limit,
            _WIDGET_UPLOAD_ENTITY_NAMESPACE,
            entity_key,
        )

    @_fail_open
    def allow_widget_task_create(
        self, entity_key: str | None, remote_ip: str | None
    ) -> bool:
        """Count one widget task-create; False when a bucket is exceeded (#1108).

        The widget mirror of :meth:`allow_task_create`. Keyed on the widget
        entity (``agent:<id>`` / ``workforce:<id>``) plus the caller IP — NOT
        the widget ``guest_id``, which unlike the share path is client-supplied
        and therefore rotatable at will (the same reasoning as
        :meth:`allow_widget_upload`). Task-create is the costly surface (each
        spawns an owner-billed run), so the per-IP bucket is the tight
        per-abuser gate and the per-entity bucket the loose backstop.
        """
        return self._admit_ip_and_entity(
            self._widget_task_create_ip_limit,
            _WIDGET_TASK_CREATE_IP_NAMESPACE,
            remote_ip,
            self._widget_task_create_entity_limit,
            _WIDGET_TASK_CREATE_ENTITY_NAMESPACE,
            entity_key,
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

    @_fail_open_denial
    def widget_run_denial_reason(
        self, entity_key: str, client_ip: str | None
    ) -> str | None:
        """Count one owner-billed widget run; name the bucket that refused it.

        Returns ``None`` when admitted, ``"ip"`` when the per-caller sub-quota
        refused (the visitor can retry later), or ``"entity"`` when the
        widget's own budget is exhausted (only the owner can act) — the caller
        needs the distinction to show copy the reader can act on.

        The widget mirror of :meth:`allow_run`, in its own buckets so a
        popular/abused widget cannot drain the owner's whole team quota. Keyed
        on the widget entity (``agent:<id>`` / ``workforce:<id>``) plus the IP
        the server observed when the task was created (stamped into
        ``agent_config``; never client-supplied) — NOT the widget ``guest_id``,
        which is client-supplied and rotatable at will.

        The IP sub-quota is scoped ``entity|ip``, NOT bare IP: this gate is
        charged per *turn*, so a bare-IP bucket would make one NAT/CGNAT egress
        share a single budget across every widget on the instance, and one busy
        embed would lock that whole network out of unrelated widgets. Scoping
        to the widget keeps the per-abuser bound while confining its blast
        radius to the entity whose budget it protects. Its window is
        deliberately far tighter than the per-minute burst gates (widget WS
        turn / task-create, both 60/minute per IP): those bound *rate*, this
        bounds one caller's share of a rolling owner-billed *budget*, so it is
        sized as a fraction of the entity quota rather than to match them.

        Both buckets use the non-destructive test→hit pairing, so an IP-window
        denial never burns an entity slot. The sole caller (the ``chat.py``
        chokepoint) short-circuits before calling when it cannot form an
        ``entity_key``, so this takes a non-empty ``str`` — no ``"unknown"``
        fallback. ``client_ip`` is ``None`` for tasks created before the marker
        existed; those are bounded by the entity quota alone rather than
        collapsing every legacy task into one shared IP bucket. Rolling, not
        cumulative, so a busy-but-legitimate widget self-clears rather than
        being bricked.
        """
        if client_ip is None:
            admitted = self._limiter.hit(
                self._widget_run_limit, _WIDGET_RUN_NAMESPACE, entity_key
            )
            return None if admitted else "entity"
        return self._denial_from_ip_and_entity(
            self._widget_run_ip_limit,
            _WIDGET_RUN_IP_NAMESPACE,
            f"{entity_key}|{client_ip}",
            self._widget_run_limit,
            _WIDGET_RUN_NAMESPACE,
            entity_key,
        )


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
