"""Interactive OAuth authorization-code + PKCE orchestration for connect/callback."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin

import httpx

from .....utils.encryption import get_cipher


def new_pkce() -> Tuple[str, str]:
    """Return (code_verifier, code_challenge) using S256."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def encode_state(user_id: int, mcpserver_id: int) -> str:
    """Opaque, tamper-proof state (Fernet-encrypted JSON)."""
    payload = json.dumps(
        {"u": user_id, "s": mcpserver_id, "n": secrets.token_urlsafe(8)}
    )
    return get_cipher().encrypt(payload.encode()).decode()


def decode_state(token: str) -> Tuple[int, int, str]:
    from cryptography.fernet import InvalidToken

    try:
        raw = get_cipher().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("invalid state") from exc
    try:
        data = json.loads(raw)
        return int(data["u"]), int(data["s"]), str(data["n"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("malformed state payload") from exc


async def _get_json(client: httpx.AsyncClient, url: str) -> Optional[Dict[str, Any]]:
    resp = await client.get(url)
    if resp.status_code == 200:
        return resp.json()
    return None


async def discover_auth_server(
    server_url: str, client: httpx.AsyncClient
) -> Dict[str, Any]:
    """Discover the authorization server metadata for a protected MCP resource."""
    prm = await _get_json(
        client, urljoin(server_url, "/.well-known/oauth-protected-resource")
    )
    issuer = None
    if prm and prm.get("authorization_servers"):
        issuer = prm["authorization_servers"][0]
    issuer = issuer or server_url

    for suffix in (
        "/.well-known/oauth-authorization-server",
        "/.well-known/openid-configuration",
    ):
        meta = await _get_json(client, urljoin(issuer + "/", suffix.lstrip("/")))
        if meta and meta.get("token_endpoint"):
            return meta
    raise ValueError(f"could not discover auth server metadata for {server_url}")


async def register_client_dcr(
    as_meta: Dict[str, Any],
    redirect_uris: List[str],
    client: httpx.AsyncClient,
) -> Dict[str, Any]:
    """Dynamic Client Registration (RFC 7591)."""
    endpoint = as_meta.get("registration_endpoint")
    if not endpoint:
        raise ValueError("server does not support dynamic client registration")
    resp = await client.post(
        endpoint,
        json={
            "client_name": "xagent",
            "redirect_uris": redirect_uris,
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post",
        },
    )
    resp.raise_for_status()
    return resp.json()


def build_authorization_url(
    as_meta: Dict[str, Any],
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
    scope: Optional[str] = None,
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if scope:
        params["scope"] = scope
    authorization_endpoint = as_meta.get("authorization_endpoint")
    if not authorization_endpoint:
        raise ValueError("authorization server metadata has no authorization_endpoint")
    return f"{authorization_endpoint}?{urlencode(params)}"


async def exchange_code_for_tokens(
    as_meta: Dict[str, Any],
    client_id: str,
    client_secret: Optional[str],
    code: str,
    code_verifier: str,
    redirect_uri: str,
    client: httpx.AsyncClient,
) -> Dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    if client_secret:
        data["client_secret"] = client_secret
    resp = await client.post(as_meta["token_endpoint"], data=data)
    resp.raise_for_status()
    return resp.json()
