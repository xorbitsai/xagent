"""Provider-specific OAuth refresh error classification.

RFC 6749's ``invalid_grant`` is deliberately too broad to classify as a dead
refresh token by itself: it also covers a token issued to another client.
Keep provider-specific evidence here so every refresh path applies the same
strict rule without teaching generic OAuth code about SDK exception shapes.
"""

from __future__ import annotations

from collections.abc import Mapping

_GOOGLE_REVOKED_REFRESH_DESCRIPTION = "token has been expired or revoked"


def _normalized_error_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold().rstrip(".").strip()
    return normalized or None


def is_google_revoked_refresh_error(
    *,
    error_code: object,
    error_description: object,
    error_subtype: object = None,
) -> bool:
    """Return whether Google explicitly says this refresh token is dead.

    ``invalid_grant`` alone is not sufficient: an OAuth client credential
    rotation can produce the same code for every still-live user grant.
    Google's revoked/expired response carries a stable description; require
    both fields and exclude a subtype so organization reauthentication
    failures such as ``invalid_rapt`` remain distinguishable.
    """

    return bool(
        _normalized_error_text(error_code) == "invalid_grant"
        and _normalized_error_text(error_description)
        == _GOOGLE_REVOKED_REFRESH_DESCRIPTION
        and _normalized_error_text(error_subtype) is None
    )


def google_refresh_error_mapping_from_exception(
    exc: BaseException,
) -> Mapping[str, object] | None:
    """Extract google-auth's structured refresh response from an exception.

    ``google.auth.exceptions.RefreshError`` stores the decoded response body
    as one of its positional arguments. Avoid parsing ``str(exc)``: exception
    messages are presentation text and can change independently of the JSON
    response contract.
    """

    for value in reversed(exc.args):
        if isinstance(value, Mapping):
            return value
    return None


def is_google_revoked_refresh_exception(exc: BaseException) -> bool:
    """Classify a structured google-auth refresh exception."""

    details = google_refresh_error_mapping_from_exception(exc)
    if details is None:
        return False
    return is_google_revoked_refresh_error(
        error_code=details.get("error"),
        error_description=details.get("error_description"),
        error_subtype=details.get("error_subtype"),
    )
