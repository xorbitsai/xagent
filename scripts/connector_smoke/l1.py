"""L1 configuration smoke: pure assertions over recorded HTTP responses.

Nothing here performs I/O, so every rule is unit-testable against recorded
responses. Failure reasons never echo a token, a ticket, a client ID, or a
redirect URL; they name at most an origin, a status, and the title of the
server's error page.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

# Gmail sits behind the Google restricted-scope gate and can be refused while
# the provider itself is fine, so it only represents Google as a last resort.
_AVOID_AS_REPRESENTATIVE = frozenset({"gmail"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REASON_DETAIL = 200


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class OAuthTarget:
    provider: str
    app_id: str
    needs_account: bool = False


def format_result(result: CheckResult) -> str:
    status = "PASS" if result.ok else "FAIL"
    line = f"{status} {result.name}"
    return f"{line}: {result.reason}" if result.reason else line


def expected_app_ids(
    registry_rows: Iterable[Mapping[str, Any]],
    *,
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
) -> set[str]:
    """Builtin apps the registry shows by default, adjusted per environment.

    ``exclude`` names apps this deployment hides; each must be one the
    registry shows by default, so a typo cannot silently shrink the check.
    ``include`` names apps this deployment shows beyond that default (an app
    the registry hides, or an operator-created one).
    """
    default_visible = {
        str(row["app_id"])
        for row in registry_rows
        if row.get("app_id") and row.get("is_visible_in_connector")
    }
    excluded = set(exclude)
    included = set(include)
    both = sorted(excluded & included)
    if both:
        raise ValueError(f"apps both included and excluded: {', '.join(both)}")
    unknown = sorted(excluded - default_visible)
    if unknown:
        raise ValueError(
            "excluded apps are not shown by the builtin registry by default: "
            + ", ".join(unknown)
        )
    return (default_visible - excluded) | included


def check_catalog(
    apps: Iterable[Mapping[str, Any]], expected: set[str]
) -> list[CheckResult]:
    """One result per expected app, plus one failure per unexpected app."""
    listed = {str(app["id"]): app for app in apps if app.get("id")}
    results: list[CheckResult] = []
    for app_id in sorted(expected | set(listed)):
        name = f"catalog:{app_id}"
        app = listed.get(app_id)
        if app_id not in expected:
            results.append(CheckResult(name, False, "unexpected app in the catalog"))
        elif app is None:
            results.append(
                CheckResult(name, False, "missing from the catalog (absent or hidden)")
            )
        elif app.get("is_visible_in_connector") is False:
            results.append(CheckResult(name, False, "listed but marked hidden"))
        elif app.get("auth_type") in (None, "", "unconnectable"):
            results.append(
                CheckResult(
                    name,
                    False,
                    f"classified as {app.get('auth_type') or 'unknown'}, "
                    "not connectable",
                )
            )
        else:
            results.append(CheckResult(name, True))
    return results


def select_oauth_targets(
    apps: Iterable[Mapping[str, Any]], preferred: Iterable[str] = ()
) -> list[OAuthTarget]:
    """One launch per builtin OAuth provider, through a visible catalog app.

    The launch always names an app: providers whose apps need an app-scoped
    grant refuse a bare provider launch. Apps in ``preferred`` (the expected
    registry apps) represent their provider before operator-created ones,
    whose extra scopes could be refused for reasons unrelated to the
    provider's configuration.
    """
    preferred = set(preferred)
    by_provider: dict[str, list[Mapping[str, Any]]] = {}
    for app in apps:
        provider = app.get("provider")
        if (
            app.get("auth_type") != "builtin_oauth"
            or not provider
            or not app.get("id")
            or app.get("is_visible_in_connector") is False
        ):
            continue
        by_provider.setdefault(str(provider), []).append(app)

    targets = []
    for provider in sorted(by_provider):
        candidates = sorted(
            by_provider[provider],
            key=lambda app: (
                str(app["id"]) in _AVOID_AS_REPRESENTATIVE,
                str(app["id"]) not in preferred,
                str(app["id"]),
            ),
        )
        app = candidates[0]
        targets.append(
            OAuthTarget(
                provider=provider,
                app_id=str(app["id"]),
                needs_account=bool(app.get("oauth_account_input")),
            )
        )
    return targets


def normalize_origin(value: object) -> str | None:
    """``scheme://host[:port]`` in canonical form, or None if not an origin."""
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value.strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in _DEFAULT_PORTS or not host or parts.username or parts.password:
        return None
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def check_callback_origin(
    provider: str,
    *,
    status_code: int,
    payload: object,
    expected_origin: str,
) -> CheckResult:
    """The provider's configured callback must live on this environment.

    Read from the launch start, so it is checked even when the client ID is
    missing and the redirect is never built.
    """
    name = f"oauth:{provider}:callback"
    if status_code != 200 or not isinstance(payload, Mapping):
        return CheckResult(
            name,
            False,
            f"launch start refused: HTTP {status_code}{_detail_suffix(payload)}",
        )
    actual = normalize_origin(payload.get("callback_origin"))
    if actual is None:
        return CheckResult(name, False, "launch start returned no callback origin")
    if actual != normalize_origin(expected_origin):
        return CheckResult(
            name, False, f"callback origin is {actual}, expected {expected_origin}"
        )
    return CheckResult(name, True)


def check_authorize_redirect(
    provider: str,
    *,
    status_code: int,
    location: str | None,
    body: str,
    expected_origin: str,
) -> CheckResult:
    """The redeemed launch must redirect to the provider with this
    environment's client ID and callback."""
    name = f"oauth:{provider}:authorize"
    if status_code not in _REDIRECT_STATUSES or not location:
        return CheckResult(
            name,
            False,
            f"no provider redirect: HTTP {status_code}{_detail_suffix(body)}",
        )
    try:
        query = parse_qs(urlsplit(location).query)
    except ValueError:
        return CheckResult(name, False, "provider redirect is not a valid URL")

    problems = []
    if not "".join(query.get("client_id", [])).strip():
        problems.append("client_id is empty")
    redirect_uris = query.get("redirect_uri", [])
    expected_path = f"/api/auth/{provider}/callback"
    if len(redirect_uris) != 1:
        problems.append("redirect_uri is missing")
    else:
        redirect_uri = redirect_uris[0]
        origin = normalize_origin(redirect_uri)
        if origin != normalize_origin(expected_origin):
            problems.append(
                f"redirect_uri is on {origin or 'an invalid origin'}, "
                f"expected {expected_origin}"
            )
        if urlsplit(redirect_uri).path != expected_path:
            problems.append(f"redirect_uri path is not {expected_path}")
    if problems:
        return CheckResult(name, False, "; ".join(problems))
    return CheckResult(name, True)


def _detail_suffix(body: object) -> str:
    detail = _error_detail(body)
    return f" ({detail})" if detail else ""


def _error_detail(body: object) -> str:
    """A short, value-free summary of an error response body.

    FastAPI errors carry a JSON ``detail``; the OAuth launch's own error pages
    carry an ``<h1>`` title and a ``<p>`` that names a missing variable.
    """
    if isinstance(body, str):
        text = body
        try:
            body = json.loads(text)
        except ValueError:
            return _html_summary(text)
    if isinstance(body, Mapping) and isinstance(body.get("detail"), str):
        return _clip(body["detail"])
    return ""


def _html_summary(body: str) -> str:
    parts = []
    for tag in ("h1", "p"):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", body, re.IGNORECASE | re.DOTALL)
        if match:
            parts.append(html.unescape(re.sub(r"<[^>]+>", "", match.group(1))))
    return _clip(" - ".join(part for part in parts if part.strip()))


def _clip(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > _MAX_REASON_DETAIL:
        return text[: _MAX_REASON_DETAIL - 3] + "..."
    return text
