"""Check that each vendor still accepts the platform's OAuth clients (#3580).

Usage: ``python -m xagent.web.oauth_client_verification [--exclude NAME]...``,
run in the backend container like ``oauth_readiness``.

``oauth_readiness`` proves a client ID and secret are present. This command
proves the vendor still accepts them: for every builtin OAuth provider it
sends one token request with the deployment's real client credential and a
deliberately invalid refresh token, and classifies the answer. Every remote
MCP connector with a pre-registered client is listed too; none of them can be
checked this way today (see ``MCP_UNVERIFIABLE``).

- ``client_rejected``: the vendor rejected the client (``invalid_client`` or
  the vendor's equivalent). Every user of that connector is affected.
- ``client_ok``: the vendor accepted the client and rejected the token.
- ``inconclusive``: anything else (a 5xx, a timeout, an answer that does not
  tell the two apart).
- ``unverifiable``: this provider's token endpoint cannot tell the two apart,
  or depends on the customer's account; the line says why.

A builtin provider's request is built by ``build_oauth_refresh_request``, the
same function a refresh uses, so the command checks exactly what a refresh
sends. Microsoft is the exception: its token endpoint checks a refresh
token's format before the client, so its secret is checked with a
client-credentials request against the app's home tenant, named by
``MICROSOFT_TENANT_ID``; without it Microsoft is ``unverifiable``. Each credential is resolved the way a refresh resolves it (the
``oauth_providers`` row first, then the environment variable). One request per
credential per run, no retries, no user data, no consent screen.

Output follows ``oauth_readiness``: one line per credential with where each
field resolved from, never a value. Exit 1 when any credential is
``client_rejected``; ``inconclusive`` lines are listed and do not fail the run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TextIO

Verdict = Literal["client_ok", "client_rejected", "inconclusive", "unverifiable"]

# Sent where the refresh sends the user's token. Never a real token.
INVALID_TOKEN = "xagent-client-verification-invalid-token"
_CODE_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


@dataclass(frozen=True)
class VendorAnswer:
    """What the command keeps of a token endpoint's answer."""

    status: int
    # The vendor's error code, normalized: the top-level OAuth ``error``,
    # ``meta:<code>`` for Meta's nested OAuthException, or the first
    # ``errors[].code`` of an Intercom-style error list.
    error: str | None
    # Lowercased description, only used to tell two answers apart.
    description: str


Rule = Callable[[VendorAnswer], Verdict]


def _by_code(*, rejected: Iterable[str], ok: Iterable[str]) -> Rule:
    rejected_codes, ok_codes = frozenset(rejected), frozenset(ok)

    def rule(answer: VendorAnswer) -> Verdict:
        if answer.status >= 500:
            return "inconclusive"
        if answer.error in rejected_codes:
            return "client_rejected"
        if answer.error in ok_codes:
            return "client_ok"
        return "inconclusive"

    return rule


def _intercom(answer: VendorAnswer) -> Verdict:
    # Only the answer measured for a made-up client on 2026-11-09:
    # 404 {"errors": [{"code": "not_found", "message": "Client not found"}]}.
    # How it answers a known client with an invalid token is not known yet.
    if (
        answer.status == 404
        and answer.error == "not_found"
        and "client not found" in answer.description
    ):
        return "client_rejected"
    return "inconclusive"


def _meta(answer: VendorAnswer) -> Verdict:
    # OAuthException 101 "Invalid Client ID"; 1 is "Error validating client
    # secret" but also Meta's generic "An unknown error occurred", so it needs
    # the message too. 190 is an invalid access token.
    if answer.status >= 500:
        return "inconclusive"
    if answer.error == "meta:101" or (
        answer.error == "meta:1" and "client secret" in answer.description
    ):
        return "client_rejected"
    if answer.error == "meta:190":
        return "client_ok"
    return "inconclusive"


# RFC 6749 section 5.2: invalid_client when client authentication fails,
# invalid_grant when the refresh token is invalid. The rejected answers were
# measured on 2026-11-09 with a made-up client; the client_ok answers follow
# each vendor's documentation until a run with a real client confirms them.
_STANDARD = _by_code(rejected={"invalid_client"}, ok={"invalid_grant"})
RULES: dict[str, Rule] = {
    "google": _by_code(
        rejected={"invalid_client", "unauthorized_client"}, ok={"invalid_grant"}
    ),
    "meta": _meta,
    "zoom": _STANDARD,
    "salesforce": _by_code(
        rejected={"invalid_client_id", "invalid_client"}, ok={"invalid_grant"}
    ),
    "linear": _STANDARD,
    "xero": _STANDARD,
    "myob": _STANDARD,
    "employment-hero": _STANDARD,
    "intercom": _intercom,
    # GitHub answers 200 with these codes; a made-up client got a bare 404.
    "github": _by_code(
        rejected={"incorrect_client_credentials"}, ok={"bad_refresh_token"}
    ),
    # A made-up client got 403 unauthorized_client "refresh_token is
    # invalid", which names neither side clearly, so it stays inconclusive.
    "jira": _STANDARD,
}
_TOKEN_FIRST = (
    "the token endpoint rejects an invalid refresh token before it checks "
    "the client (measured 2026-11-09)"
)
UNVERIFIABLE: dict[str, str] = {
    "linkedin": _TOKEN_FIRST,
    "slack": _TOKEN_FIRST,
    "zendesk": "the token endpoint is on the customer's own subdomain",
    "deputy": "the token endpoint is on the customer's own install",
}
# Pre-registered MCP clients, by credential provider. HubSpot answered a
# made-up client with 400 invalid_request "missing or invalid refresh token"
# (BAD_REFRESH_TOKEN), so it checks the token first too (measured 2026-11-09).
MCP_UNVERIFIABLE: dict[str, str] = {"hubspot": _TOKEN_FIRST}

# Microsoft checks a refresh token's format before the client (AADSTS9002313,
# measured 2026-11-09), so its secret is checked with a client-credentials
# request against the app's home tenant instead. The provider rows sign in
# through ``common``; only this request needs the tenant, which comes from
# the deployment's configuration next to MICROSOFT_CLIENT_ID, per region.
MICROSOFT_TENANT_ENV = "MICROSOFT_TENANT_ID"
MICROSOFT_TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
# Asks for an app-only token with whatever application permissions the app
# already has; the token is never kept.
MICROSOFT_SCOPE = "https://graph.microsoft.com/.default"
# The app registration's Directory (tenant) ID. ``common`` or a domain name is
# not the app's own tenant, and anything else must not reach the URL.
_TENANT_ID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_MICROSOFT_NO_TENANT = (
    "refresh tokens are format-checked before the client; set "
    f"{MICROSOFT_TENANT_ENV} to check the secret against the app's home tenant"
)

# A real refresh carries a refresh token the vendor issued, so an answer that
# rejects the client means the client, also for providers the verification
# above cannot check. These are the refresh's own codes, not the probe RULES:
# on a real refresh Google's unauthorized_client means the refresh token was
# issued to another client (a grant from before a client switch), which is
# that grant's problem, not the current client's. Slack's oauth.v2.access
# names its own codes.
_DEFAULT_CLIENT_REJECTED_CODES = frozenset({"invalid_client"})
_REFRESH_CLIENT_REJECTED_CODES: dict[str, frozenset[str]] = {
    "github": frozenset({"incorrect_client_credentials"}),
    "salesforce": frozenset({"invalid_client_id", "invalid_client"}),
    "slack": frozenset({"invalid_client", "invalid_client_id", "bad_client_secret"}),
}
# Vendors whose rejection is not a bare error code.
_REFRESH_RULES: dict[str, Rule] = {"meta": _meta, "intercom": _intercom}


class UnknownExclusionError(ValueError):
    """``--exclude`` names something that is neither a provider nor an MCP client."""


@dataclass(frozen=True)
class VerificationResult:
    kind: Literal["oauth-client", "mcp-client"]
    name: str
    verdict: Verdict
    detail: str


def _code(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        value = str(value)
    if isinstance(value, str) and _CODE_PATTERN.fullmatch(value):
        return value
    return None


def vendor_answer(status: int, payload: object) -> VendorAnswer:
    """Normalize one token endpoint answer; keeps no token or secret."""
    error: str | None = None
    description = ""
    if isinstance(payload, Mapping):
        raw_error = payload.get("error")
        if isinstance(raw_error, Mapping):  # Meta's nested OAuthException
            code = _code(raw_error.get("code"))
            error = f"meta:{code}" if code else None
            description = str(raw_error.get("message") or "")
        else:
            error = _code(raw_error)
            description = str(payload.get("error_description") or "")
        errors = payload.get("errors")
        if error is None and isinstance(errors, list) and errors:
            first = errors[0] if isinstance(errors[0], Mapping) else {}
            error = _code(first.get("code"))
            description = str(first.get("message") or "")
    return VendorAnswer(status, error, description.lower()[:200])


def is_client_rejected(provider: str, status: int, payload: object) -> bool:
    """Whether a failed refresh's answer says the vendor rejected the client.

    Used by the refresh paths (#3580): such a failure affects every user of
    the provider's connectors, unlike a dead grant or a transient error.
    """
    answer = vendor_answer(status, payload)
    if answer.status >= 500:
        return False
    name = provider.lower()
    rule = _REFRESH_RULES.get(name)
    if rule is not None:
        return rule(answer) == "client_rejected"
    return answer.error in _REFRESH_CLIENT_REJECTED_CODES.get(
        name, _DEFAULT_CLIENT_REJECTED_CODES
    )


def client_rejected_reason(provider: str) -> str:
    """The session reason for a rejected client: ``client_rejected:<provider>``,
    or a bare ``client_rejected`` when the name is not a valid suffix (an
    admin-created provider may be named ``my_crm``).
    """
    from ..core.tools.adapters.vibe.config import (
        CLIENT_REJECTED_REASON,
        is_client_rejected_reason,
    )

    reason = f"{CLIENT_REJECTED_REASON}:{provider.lower()}"
    return reason if is_client_rejected_reason(reason) else CLIENT_REJECTED_REASON


def _answer_detail(answer: VendorAnswer) -> str:
    return f"vendor answered {answer.status} {answer.error or '-'}"


async def _ask_vendor(send: Callable[[], Any]) -> VendorAnswer | str:
    """One request; a string is why there is no answer.

    ``send`` resolves the credential, builds the request, creates the client
    and sends it. Any failure on the way (a timeout, a refused egress policy,
    a proxy setting the client refuses) is reported for this credential only,
    so one failing credential never hides the other results.
    """
    try:
        response = await send()
    except Exception as exc:
        return f"no answer ({type(exc).__name__})"
    try:
        payload = response.json()
    except Exception:
        payload = None
    return vendor_answer(response.status_code, payload)


def _microsoft_tenant() -> str:
    return (os.environ.get(MICROSOFT_TENANT_ENV) or "").strip()


_AADSTS = re.compile(r"aadsts(\d{1,9})")


def _microsoft(answer: VendorAnswer) -> Verdict:
    """Classify a client-credentials answer from the app's home tenant."""
    if answer.status == 200 and answer.error is None:
        return "client_ok"  # an app-only token, never kept
    if answer.status >= 500:
        return "inconclusive"
    # AADSTS7000215 (wrong secret), 7000222 (expired secret) and the like.
    if answer.error == "invalid_client":
        return "client_rejected"
    # AADSTS700016 (no such app in the tenant), 90002 (no such tenant) and the
    # rest more likely mean the tenant setting is wrong than that Microsoft
    # rejected the platform's client.
    return "inconclusive"


def _microsoft_detail(answer: VendorAnswer) -> str:
    if answer.status == 200 and answer.error is None:
        return "vendor issued an app-only token (discarded)"
    detail = _answer_detail(answer)
    code = _AADSTS.match(answer.description)
    return f"{detail} (AADSTS{code.group(1)})" if code else detail


async def _verify_microsoft(
    provider: str, row: Any | None, sources: str
) -> VerificationResult:
    tenant = _microsoft_tenant()
    if not _TENANT_ID.fullmatch(tenant):
        return VerificationResult(
            "oauth-client",
            provider,
            "inconclusive",
            f"{sources}; {MICROSOFT_TENANT_ENV} is not a directory (tenant) ID",
        )

    from .api.auth import _resolve_oauth_client_per_field
    from .tools.config import _oauth_refresh_http_client

    async def ask() -> Any:
        client_id, client_secret = _resolve_oauth_client_per_field(provider, row)
        async with _oauth_refresh_http_client() as client:
            return await client.post(
                MICROSOFT_TOKEN_URL.format(tenant=tenant),
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": MICROSOFT_SCOPE,
                },
                headers={"Accept": "application/json"},
            )

    answer = await _ask_vendor(ask)
    if isinstance(answer, str):
        return VerificationResult(
            "oauth-client", provider, "inconclusive", f"{sources}; {answer}"
        )
    return VerificationResult(
        "oauth-client",
        provider,
        _microsoft(answer),
        f"{sources}; {_microsoft_detail(answer)}",
    )


async def verify_builtin_provider(provider: str, row: Any | None) -> VerificationResult:
    """Verify one builtin provider's client with an invalid refresh token."""
    from .oauth_readiness import check_provider

    readiness = check_provider(provider, row)
    sources = f"client_id={readiness.client_id} client_secret={readiness.client_secret}"
    # Rules are keyed like the refresh's own provider checks: lowercased.
    name = provider.lower()
    if name in UNVERIFIABLE:
        return VerificationResult(
            "oauth-client", provider, "unverifiable", UNVERIFIABLE[name]
        )
    if name == "microsoft" and not _microsoft_tenant():
        return VerificationResult(
            "oauth-client", provider, "unverifiable", _MICROSOFT_NO_TENANT
        )
    rule = RULES.get(name)
    if rule is None and name != "microsoft":
        return VerificationResult(
            "oauth-client", provider, "unverifiable", "no verification rule"
        )
    if not readiness.ready:
        return VerificationResult(
            "oauth-client",
            provider,
            "inconclusive",
            f"{sources}; not configured, see oauth_readiness",
        )
    if rule is None:
        return await _verify_microsoft(provider, row, sources)

    from .api.auth import _resolve_oauth_client_per_field
    from .tools.config import (
        _oauth_refresh_http_client,
        build_oauth_refresh_request,
        send_oauth_refresh_request,
    )

    async def ask() -> Any:
        client_id, client_secret = _resolve_oauth_client_per_field(provider, row)
        request = await build_oauth_refresh_request(
            provider,
            row,
            client_id=client_id,
            client_secret=client_secret,
            access_token=INVALID_TOKEN,
            refresh_token=INVALID_TOKEN,
            instance_url=None,
            user_id="client-verification",
        )
        async with _oauth_refresh_http_client() as client:
            return await send_oauth_refresh_request(client, request)

    answer = await _ask_vendor(ask)
    if isinstance(answer, str):
        return VerificationResult(
            "oauth-client", provider, "inconclusive", f"{sources}; {answer}"
        )
    return VerificationResult(
        "oauth-client", provider, rule(answer), f"{sources}; {_answer_detail(answer)}"
    )


def _mcp_credential_provider(app: Mapping[str, Any]) -> str | None:
    """The credential provider of an ``mcp_oauth`` catalog app, if it has one."""
    auth = (app.get("launch_config") or {}).get("auth") or {}
    credential_provider = auth.get("credential_provider")
    if auth.get("type") != "mcp_oauth" or not isinstance(credential_provider, str):
        return None
    return credential_provider


def verify_mcp_client(app: Mapping[str, Any]) -> VerificationResult | None:
    """One catalog MCP connector's pre-registered client, if it has one.

    No pre-registered client can be checked with an invalid token today, so
    each one is reported with the reason and nothing is sent.
    """
    credential_provider = _mcp_credential_provider(app)
    if credential_provider is None:
        return None
    reason = MCP_UNVERIFIABLE.get(credential_provider, "no verification rule")
    return VerificationResult("mcp-client", credential_provider, "unverifiable", reason)


def format_result(result: VerificationResult) -> str:
    return f"{result.verdict.upper()} {result.kind}:{result.name}: {result.detail}"


async def verify_all(
    rows: Iterable[Any],
    *,
    builtin_providers: Sequence[str],
    mcp_apps: Iterable[Mapping[str, Any]],
    exclude: set[str],
) -> list[VerificationResult]:
    rows_by_provider = {str(row.provider_name): row for row in rows}
    providers = list(dict.fromkeys([*builtin_providers, *sorted(rows_by_provider)]))
    mcp_list = list(mcp_apps)
    known = set(providers) | {
        name for app in mcp_list if (name := _mcp_credential_provider(app))
    }
    unknown = sorted(exclude - known)
    if unknown:
        raise UnknownExclusionError(f"unknown name to exclude: {', '.join(unknown)}")
    results = [
        await verify_builtin_provider(provider, rows_by_provider.get(provider))
        for provider in providers
        if provider not in exclude
    ]
    for app in mcp_list:
        result = verify_mcp_client(app)
        if result is not None and result.name not in exclude:
            results.append(result)
    return results


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    out = sys.stdout if out is None else out
    parser = argparse.ArgumentParser(
        prog="python -m xagent.web.oauth_client_verification",
        description="Check that each vendor still accepts the platform's OAuth "
        "clients. Prints no values.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="NAME",
        help="OAuth provider or MCP credential provider name to skip; "
        "repeatable or comma-separated",
    )
    args = parser.parse_args(argv)

    from ..core.utils.encryption import get_cipher
    from .builtin_mcp_registry import (
        get_builtin_oauth_provider_rows,
        get_builtin_public_mcp_app_rows,
    )
    from .oauth_readiness import _load_provider_rows, _parse_exclude

    try:
        get_cipher()
    except ValueError as exc:
        print(f"FAIL encryption-key: {type(exc).__name__}", file=out)
        return 1
    try:
        rows = _load_provider_rows()
    except Exception as exc:
        # The class name only: a driver error can carry the database host.
        print(f"FAIL database: {type(exc).__name__}", file=out)
        return 1
    try:
        results = asyncio.run(
            verify_all(
                rows,
                builtin_providers=[
                    str(row["provider_name"])
                    for row in get_builtin_oauth_provider_rows()
                ],
                mcp_apps=get_builtin_public_mcp_app_rows(),
                exclude=_parse_exclude(args.exclude),
            )
        )
    except UnknownExclusionError as exc:
        parser.error(str(exc))

    for result in results:
        print(format_result(result), file=out)
    counts = {
        verdict: 0
        for verdict in ("client_ok", "client_rejected", "inconclusive", "unverifiable")
    }
    for result in results:
        counts[result.verdict] += 1
    print(
        f"{len(results)} credentials: "
        + ", ".join(f"{count} {verdict}" for verdict, count in counts.items()),
        file=out,
    )
    return 1 if counts["client_rejected"] else 0


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    sys.exit(main())
