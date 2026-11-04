"""Widget embedding-origin normalization and allowlist enforcement."""

import logging
from typing import Literal
from urllib.parse import urlparse

from fastapi import HTTPException

__all__ = [
    "domain_allowed",
    "normalize_widget_allowed_domain",
    "origin_to_domain",
    "require_domain_allowed",
]

logger = logging.getLogger(__name__)

_MalformedPolicyReason = Literal[
    "not_list",
    "non_string_entry",
    "blank_entry",
]


def origin_to_domain(origin: str) -> str:
    """Normalize an origin or referer value to a lowercased host[:port]."""
    if not origin:
        return ""
    parsed = urlparse(origin)
    return (parsed.netloc or parsed.path).lower()


def normalize_widget_allowed_domain(value: str) -> str:
    """Return the canonical form used to detect blanks and match domains."""
    return value.strip().lower()


def _normalize_allowed_domains(
    allowed_domains: object,
) -> tuple[list[str] | None, _MalformedPolicyReason | None]:
    """Validate and normalize the complete persisted allowlist.

    JSON columns do not enforce their declared application-level shape. Treat
    any non-list container, non-string element, or blank entry as an invalid
    policy rather than coercing or skipping it and potentially broadening
    access.
    """
    if not isinstance(allowed_domains, list):
        return None, "not_list"

    normalized_domains: list[str] = []
    for domain in allowed_domains:
        if type(domain) is not str:
            return None, "non_string_entry"
        normalized_domain = normalize_widget_allowed_domain(domain)
        if not normalized_domain:
            return None, "blank_entry"
        normalized_domains.append(normalized_domain)
    return normalized_domains, None


def _evaluate_domain_policy(
    origin_domain: str, allowed_domains: object
) -> tuple[bool, _MalformedPolicyReason | None]:
    normalized_domains, malformed_reason = _normalize_allowed_domains(allowed_domains)
    if malformed_reason is not None:
        return False, malformed_reason
    assert normalized_domains is not None

    for normalized_domain in normalized_domains:
        if (
            normalized_domain == "*"
            or normalized_domain == origin_domain
            or (origin_domain and origin_domain.endswith("." + normalized_domain))
        ):
            return True, None
    return False, None


def domain_allowed(origin_domain: str, allowed_domains: object) -> bool:
    """Return whether a normalized ``origin_domain`` matches allowlist entries.

    ``origin_domain`` must already be normalized with :func:`origin_to_domain`.
    Allowlist entries retain their legacy surrounding-whitespace and
    case-insensitive handling inside this matcher. A malformed persisted
    allowlist denies access in full.
    """
    allowed, _malformed_reason = _evaluate_domain_policy(origin_domain, allowed_domains)
    return allowed


def require_domain_allowed(
    origin_domain: str,
    allowed_domains: object,
    *,
    owner_type: str,
    owner_id: int,
) -> None:
    """Enforce a normalized origin against stored widget allowlist entries.

    ``origin_domain`` must be the result of :func:`origin_to_domain`.
    """
    allowed, malformed_reason = _evaluate_domain_policy(origin_domain, allowed_domains)
    if malformed_reason is not None:
        logger.warning(
            "Rejected malformed widget allowed-domains policy: "
            "owner_type=%s owner_id=%s reason=%s",
            owner_type,
            owner_id,
            malformed_reason,
        )
    if not allowed:
        raise HTTPException(
            status_code=403, detail=f"Domain not allowed: {origin_domain}"
        )
