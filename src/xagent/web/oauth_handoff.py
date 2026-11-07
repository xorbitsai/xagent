"""Opt-in regional OAuth navigation, with a hosting-owned routing adapter.

A ticket minted on one public entry is redeemed on the authorization entry,
where the user usually has no session. The ticket is therefore a bearer
credential that is not bound to the initiating browser, so redemption takes
two steps: the first GET renders a confirmation page naming the target account
and sets a SameSite=Strict nonce cookie; only the follow-up GET carrying that
nonce consumes the ticket. A ticket URL opened from someone else's link stops
at a page naming *their* account, and a cross-site navigation straight to the
second step lacks the cookie.
"""

import hashlib
import html
import re
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, NoReturn, Protocol, cast
from urllib.parse import parse_qs, urlencode, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import (
    get_dual_entry_oauth_enabled,
    get_oauth_authorization_entry_origin,
    get_public_entries,
    normalize_public_origin,
)
from .models.database import get_db
from .models.user import User
from .public_entries import request_public_entry

HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
# Long enough to read the confirmation page; the ticket stays single-use.
TICKET_TTL = timedelta(seconds=120)
CONFIRM_COOKIE_PREFIX = "oauth_launch_confirm_"
# The confirmation page loads no script and cannot be framed or posted from.
CONFIRM_PAGE_HEADERS = {
    **HEADERS,
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'none'"
    ),
    "X-Frame-Options": "DENY",
}


@dataclass(frozen=True)
class OAuthHandoffContext:
    product: str
    environment: str
    region: str
    authorization_entry_origin: str


class OAuthHandoffAdapter(Protocol):
    def context(self, request: Request) -> OAuthHandoffContext:
        """Return authoritative deployment identity, never browser-supplied region."""
        ...

    def set_route_cookie(
        self,
        response: Response,
        *,
        state: str,
        kind: Literal["builtin", "mcp"],
        provider: str | None,
    ) -> None:
        """Set the existing state-scoped routing proof, never selected_region."""
        ...


def handoff_context(
    request: Request,
) -> tuple[OAuthHandoffAdapter, OAuthHandoffContext]:
    adapter = getattr(request.app.state, "oauth_handoff_adapter", None)
    if not get_dual_entry_oauth_enabled() or adapter is None:
        raise HTTPException(503, "Regional OAuth unavailable", headers=HEADERS)
    context = adapter.context(request)
    if (
        not context.product
        or not context.environment
        or not re.fullmatch(r"[a-z0-9-]+", context.region)
        or context.authorization_entry_origin != get_oauth_authorization_entry_origin()
    ):
        raise HTTPException(
            503, "Invalid regional OAuth configuration", headers=HEADERS
        )
    return adapter, context


def regional_launch_requested(request: Request) -> bool:
    return (
        get_dual_entry_oauth_enabled()
        and request.headers.get("x-xagent-oauth-launch-version") == "2"
    )


def validate_builtin_resource(db: Session, provider: str, app_id: str | None) -> None:
    """Apply the existing catalog visibility gate before consuming navigation proof."""
    from .api.mcp import _reject_hidden_catalog_app
    from .mcp_apps import get_app_by_id, requires_app_scoped_oauth_grant

    if not app_id and requires_app_scoped_oauth_grant(provider):
        raise HTTPException(400, "Use this connector's catalog entry", headers=HEADERS)
    if app_id:
        app = get_app_by_id(db, app_id)
        if app:
            _reject_hidden_catalog_app(app)
            if app.get("provider") != provider:
                raise HTTPException(400, "Connector provider mismatch", headers=HEADERS)


def issue_handoff(
    request: Request,
    user: User,
    *,
    kind: Literal["builtin", "mcp"],
    resource: dict[str, Any],
) -> dict[str, str]:
    from .api.auth import create_access_token
    from .auth_sessions import user_session_claims

    _, context = handoff_context(request)
    entry = request_public_entry(request)
    claims = {
        **user_session_claims(user),
        "type": "regional_oauth_launch",
        "purpose": kind,
        "product": context.product,
        "environment": context.environment,
        "region": context.region,
        "target_origin": context.authorization_entry_origin,
        "return_origin": entry.origin,
        "resource": resource,
        "jti": secrets.token_urlsafe(24),
    }
    ticket = create_access_token(claims, expires_delta=TICKET_TTL)
    return {
        "launch_mode": "regional_bootstrap",
        "authorization_url": context.authorization_entry_origin
        + "/api/oauth/launch/"
        + context.region
        + "?"
        + urlencode({"ticket": ticket}),
    }


def _confirm_cookie_name(claims: dict[str, Any]) -> str:
    """One cookie per ticket so concurrent launches do not overwrite each other."""
    digest = hashlib.sha256(str(claims["jti"]).encode()).hexdigest()[:16]
    return CONFIRM_COOKIE_PREFIX + digest


def _confirmation_page(
    request: Request,
    claims: dict[str, Any],
    *,
    account: str,
    target: str,
    cookie_path: str,
    secure: bool,
) -> HTMLResponse:
    """Render the step-one page; nothing is consumed or committed here."""
    nonce = secrets.token_urlsafe(24)
    # Relative to the current path, so only the query changes.
    href = "?" + urlencode({"ticket": request.query_params["ticket"], "confirm": nonce})
    target_text, account_text = html.escape(target), html.escape(account)
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Confirm connection</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:32rem;margin:4rem auto;
padding:0 1rem;line-height:1.5}}
a{{display:inline-block;margin-top:1rem;padding:.6rem 1.2rem;background:#111;
color:#fff;border-radius:.4rem;text-decoration:none}}
</style></head>
<body><h1>Connect {target_text}?</h1>
<p>This connects <strong>{target_text}</strong> to the account
<strong>{account_text}</strong>.</p>
<p>If this is not your account, close this window.</p>
<a href="{html.escape(href)}">Continue</a></body></html>"""
    response = HTMLResponse(body, headers=CONFIRM_PAGE_HEADERS)
    response.set_cookie(
        _confirm_cookie_name(claims),
        nonce,
        max_age=int(TICKET_TTL.total_seconds()),
        path=cookie_path,
        secure=secure,
        httponly=True,
        samesite="strict",
    )
    return response


router = APIRouter()


@router.get("/api/oauth/launch/{region}")
def redeem_handoff(
    region: str, request: Request, db: Session = Depends(get_db)
) -> Response:
    from .api import auth, mcp
    from .models.mcp import MCPServer
    from .models.mcp_oauth import MCPOAuthClient, MCPOAuthFlowState
    from .models.oauth_provider import OAuthProvider
    from .services.oauth_launch import consume_oauth_launch_ticket

    def reject() -> NoReturn:
        raise HTTPException(
            401, "Invalid or expired authorization ticket", headers=HEADERS
        )

    adapter, context = handoff_context(request)
    hosts = request.headers.getlist("x-forwarded-host") or request.headers.getlist(
        "host"
    )
    try:
        actual_origin = (
            normalize_public_origin(
                urlsplit(context.authorization_entry_origin).scheme + "://" + hosts[0]
            )
            if len(hosts) == 1
            else None
        )
    except ValueError:
        actual_origin = None
    if (
        actual_origin != context.authorization_entry_origin
        or region != context.region
        or list(request.query_params.keys()) not in (["ticket"], ["ticket", "confirm"])
        or any(len(request.query_params.getlist(k)) != 1 for k in request.query_params)
        or request_public_entry(request).origin != context.authorization_entry_origin
    ):
        reject()
    try:
        claims = auth.verify_token(request.query_params["ticket"])
    except (TypeError, ValueError, OverflowError):
        reject()
    if (
        not claims
        or claims.get("type") != "regional_oauth_launch"
        or claims.get("purpose") not in ("builtin", "mcp")
        or claims.get("product") != context.product
        or claims.get("environment") != context.environment
        or claims.get("region") != region
        or claims.get("target_origin") != context.authorization_entry_origin
        or claims.get("return_origin") not in {e.origin for e in get_public_entries()}
        or type(claims.get("user_id")) is not int
        or not 0 < claims["user_id"] <= 2**31 - 1
        or not isinstance(claims.get("jti"), str)
    ):
        reject()
    assert claims is not None
    user = db.get(User, claims["user_id"])
    if (
        user is None
        or user.username != claims.get("sub")
        or not auth.matches_user_session(user, claims)
    ):
        reject()
    resource = claims.get("resource")
    if not isinstance(resource, dict):
        reject()
    kind = claims["purpose"]
    provider = None
    if kind == "builtin":
        provider = resource.get("provider")
        if not isinstance(provider, str):
            reject()
        db_provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == provider)
            .first()
        )
        if db_provider is None:
            reject()
        try:
            options = auth.OAuthLaunchRequest.model_validate(resource.get("options"))
        except ValueError:
            reject()
        if options.redirect != claims["return_origin"]:
            reject()
        validate_builtin_resource(db, provider, options.app_id)
        target = str(db_provider.name)
    else:
        flow = (
            db.query(MCPOAuthFlowState)
            .filter(MCPOAuthFlowState.state == resource.get("state"))
            .first()
        )
        if (
            flow is None
            or flow.user_id != user.id
            or flow.mcp_oauth_client_id != resource.get("client_id")
            or flow.mcp_server_id != resource.get("server_id")
            or flow.resource_owner_key != mcp._default_resource_owner_key(int(user.id))
            or flow.redirect_after != "/auth/mcp-complete"
            or mcp._mcp_oauth_flow_state_error(db, flow) is not None
        ):
            reject()
        client = db.get(MCPOAuthClient, flow.mcp_oauth_client_id)
        if client is None:
            reject()
        server = db.get(MCPServer, flow.mcp_server_id)
        target = str(server.name) if server is not None else "MCP server"

    cookie_path = f"/api/oauth/launch/{region}"
    secure = urlsplit(context.authorization_entry_origin).scheme == "https"
    confirm = request.query_params.get("confirm")
    if confirm is None:
        return _confirmation_page(
            request,
            claims,
            account=str(user.email or user.username),
            target=target,
            cookie_path=cookie_path,
            secure=secure,
        )
    cookie_name = _confirm_cookie_name(claims)
    expected = request.cookies.get(cookie_name)
    if not expected or not secrets.compare_digest(expected.encode(), confirm.encode()):
        reject()
    try:
        consumed = consume_oauth_launch_ticket(db, claims, purpose=kind)
    except SQLAlchemyError:
        raise HTTPException(
            503, "Authorization temporarily unavailable", headers=HEADERS
        ) from None
    if not consumed:
        reject()
    db.expire_all()
    user = db.get(User, claims["user_id"])
    if user is None or not auth.matches_user_session(user, claims):
        reject()
    response: Response
    if kind == "builtin":
        assert provider is not None
        response = auth._generic_oauth_login(
            provider,
            token=None,
            app_id=options.app_id,
            redirect=options.redirect,
            db=db,
            db_provider=db_provider,
            trusted_user_id=int(user.id),
            trusted_session_epoch=cast(str | None, user.session_epoch),
            resource_owner_key=None,
            actor_flow_nonce=None,
            account=options.account,
        )
        if isinstance(response, RedirectResponse):
            states = parse_qs(urlsplit(response.headers["location"]).query).get(
                "state", []
            )
            if len(states) != 1:
                reject()
            adapter.set_route_cookie(
                response, state=states[0], kind="builtin", provider=provider
            )
            response.status_code = 303
    else:
        assert flow is not None and client is not None
        if mcp._mcp_oauth_flow_state_error(db, flow) is not None:
            reject()
        params = {
            "response_type": "code",
            "client_id": str(client.client_id),
            "redirect_uri": str(client.redirect_uri),
            "state": str(flow.state),
            "code_challenge": mcp._pkce_code_challenge(
                mcp.decrypt_value(str(flow.code_verifier))
            ),
            "code_challenge_method": "S256",
            "resource": str(flow.resource),
        }
        if flow.scope:
            params["scope"] = str(flow.scope)
        response = RedirectResponse(
            mcp._oauth_authorization_url(str(client.authorization_endpoint), params),
            status_code=303,
        )
        mcp._set_mcp_oauth_state_cookie(response, str(flow.state))
        adapter.set_route_cookie(
            response, state=str(flow.state), kind="mcp", provider=None
        )
    response.delete_cookie(
        cookie_name, path=cookie_path, secure=secure, httponly=True, samesite="strict"
    )
    response.headers.update(HEADERS)
    return response
