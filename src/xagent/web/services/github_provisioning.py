"""Repository webhook provisioning for GitHub triggers.

A GitHub trigger owns exactly one repository webhook, created with the
trigger owner's connected GitHub account (a ``UserOAuth`` row); no separate
credential store exists. The hook id is recorded on the trigger as
``resource_id = "<owner>/<repo>#<hook id>"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ...core.utils import public_requests as requests
from ..models.user_oauth import (
    COMPOSIO_TOKEN_TYPE,
    UserOAuth,
    is_undecryptable_oauth_token,
)
from .user_oauth import get_scoped_user_oauth_account, scoped_user_oauth_query

logger = logging.getLogger(__name__)

GITHUB_OAUTH_PROVIDER = "github"
GITHUB_API_ROOT = "https://api.github.com"
GITHUB_REQUEST_TIMEOUT_SECONDS = 20
_ERROR_TEXT_LIMIT = 300

GITHUB_RECONNECT_REQUIRED_ERROR = (
    "GitHub OAuth credentials are unavailable; reconnect the GitHub account"
)
GITHUB_NO_ACCOUNT_ERROR = "No connected GitHub account; connect GitHub first"


class GithubProvisioningError(Exception):
    """A hook operation failed; the message is safe to show the trigger owner."""


@dataclass(frozen=True)
class GithubHook:
    hook_id: int


def split_binding(resource_id: str | None) -> tuple[str | None, int | None]:
    """Split ``owner/repo#hook`` into the repository and the hook id."""
    if not resource_id:
        return None, None
    repository, _, hook = str(resource_id).partition("#")
    hook_id = int(hook) if hook.isdecimal() else None
    return (repository or None), hook_id


def join_binding(repository: str, hook_id: int) -> str:
    return f"{repository}#{hook_id}"


def resolve_github_account(
    db: Session, *, user_id: int, oauth_account_id: int | None
) -> UserOAuth:
    """Return the owner's ordinary GitHub grant, or raise with a safe message."""
    if oauth_account_id is not None:
        account = get_scoped_user_oauth_account(
            db,
            user_id=int(user_id),
            account_id=int(oauth_account_id),
            resource_owner_key=None,
        )
        if account is None or account.provider != GITHUB_OAUTH_PROVIDER:
            raise GithubProvisioningError("Selected account is not a GitHub account")
    else:
        account = (
            scoped_user_oauth_query(db, user_id=int(user_id), resource_owner_key=None)
            .filter(UserOAuth.provider == GITHUB_OAUTH_PROVIDER)
            .order_by(UserOAuth.id.desc())
            .first()
        )
        if account is None:
            raise GithubProvisioningError(GITHUB_NO_ACCOUNT_ERROR)
    if str(account.token_type or "") == COMPOSIO_TOKEN_TYPE:
        raise GithubProvisioningError(
            "This GitHub connection is managed by Composio and cannot manage "
            "webhooks; connect GitHub directly"
        )
    if is_undecryptable_oauth_token(account.access_token) or not account.access_token:
        raise GithubProvisioningError(GITHUB_RECONNECT_REQUIRED_ERROR)
    return account


async def ensure_fresh_token(db: Session, account: UserOAuth) -> str:
    """Refresh an expiring grant (GitHub OAuth Apps normally never expire)."""
    expires_at = account.expires_at
    if expires_at is not None:
        aware = (
            expires_at.replace(tzinfo=timezone.utc)
            if expires_at.tzinfo is None
            else expires_at
        )
        if aware <= datetime.now(timezone.utc):
            from ..tools.config import (
                _OAuthRefreshClientRejected,
                _OAuthRefreshPermanentlyInvalid,
                refresh_oauth_token_if_needed,
            )

            try:
                refreshed = await refresh_oauth_token_if_needed(
                    db, account, GITHUB_OAUTH_PROVIDER
                )
            except _OAuthRefreshPermanentlyInvalid as exc:
                db.rollback()
                raise GithubProvisioningError(
                    "The GitHub authorization was revoked or expired; "
                    "reconnect the GitHub account"
                ) from exc
            except _OAuthRefreshClientRejected as exc:
                db.rollback()
                raise GithubProvisioningError(
                    "GitHub rejected this deployment's OAuth client while "
                    "refreshing the token; contact the administrator"
                ) from exc
            if not refreshed:
                raise GithubProvisioningError(GITHUB_RECONNECT_REQUIRED_ERROR)
            # GitHub invalidates the consumed refresh token, so the rotated
            # pair must be durable before any later step can roll back.
            db.commit()
    token = str(account.access_token or "")
    if not token:
        raise GithubProvisioningError(GITHUB_RECONNECT_REQUIRED_ERROR)
    return token


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _hook_body(callback_url: str, secret: str, events: list[str]) -> dict[str, Any]:
    return {
        "name": "web",
        "active": True,
        "events": events,
        "config": {
            "url": callback_url,
            "content_type": "json",
            "secret": secret,
            "insecure_ssl": "0",
        },
    }


def _raise_for(response: Any, repository: str, action: str) -> None:
    status = int(response.status_code)
    if status == 401:
        raise GithubProvisioningError(GITHUB_RECONNECT_REQUIRED_ERROR)
    if status in (403, 404):
        raise GithubProvisioningError(
            f"GitHub refused to {action} for {repository} (HTTP {status}); the "
            "connected account needs admin access to the repository, and the "
            "repository must exist"
        )
    detail = ""
    try:
        message = response.json().get("message")
        detail = f": {str(message)[:_ERROR_TEXT_LIMIT]}" if message else ""
    except Exception:  # noqa: BLE001 - body is advisory only
        pass
    raise GithubProvisioningError(
        f"GitHub could not {action} for {repository} (HTTP {status}){detail}"
    )


def create_hook(
    token: str, repository: str, *, callback_url: str, secret: str, events: list[str]
) -> GithubHook:
    """Create the repository webhook (blocking HTTP)."""
    try:
        response = requests.post(
            f"{GITHUB_API_ROOT}/repos/{repository}/hooks",
            headers=_headers(token),
            json=_hook_body(callback_url, secret, events),
            timeout=GITHUB_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise GithubProvisioningError(
            f"GitHub is unreachable ({type(exc).__name__}); will retry"
        ) from exc
    if int(response.status_code) != 201:
        _raise_for(response, repository, "create the webhook")
    return GithubHook(hook_id=int(response.json()["id"]))


def update_hook(
    token: str,
    repository: str,
    hook_id: int,
    *,
    callback_url: str,
    secret: str,
    events: list[str],
) -> GithubHook | None:
    """Re-assert the hook's URL/secret/events. None when the hook is gone."""
    try:
        response = requests.patch(
            f"{GITHUB_API_ROOT}/repos/{repository}/hooks/{hook_id}",
            headers=_headers(token),
            json=_hook_body(callback_url, secret, events),
            timeout=GITHUB_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise GithubProvisioningError(
            f"GitHub is unreachable ({type(exc).__name__}); will retry"
        ) from exc
    if int(response.status_code) == 404:
        return None
    if int(response.status_code) != 200:
        _raise_for(response, repository, "update the webhook")
    return GithubHook(hook_id=hook_id)


def delete_hook(token: str, repository: str, hook_id: int) -> None:
    """Delete the hook; an already-missing hook is success."""
    try:
        response = requests.delete(
            f"{GITHUB_API_ROOT}/repos/{repository}/hooks/{hook_id}",
            headers=_headers(token),
            timeout=GITHUB_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise GithubProvisioningError(
            f"GitHub is unreachable ({type(exc).__name__})"
        ) from exc
    if int(response.status_code) in (204, 404):
        return
    _raise_for(response, repository, "delete the webhook")
