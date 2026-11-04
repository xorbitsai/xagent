"""Shared Google Drive Picker configuration resolution."""

import logging
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Optional, cast

from sqlalchemy.orm import Session

from ...core.utils.encryption import decrypt_value
from ..builtin_mcp_registry import GOOGLE_DRIVE_FILE_SCOPE, GOOGLE_DRIVE_SCOPE
from ..models.oauth_provider import OAuthProvider

logger = logging.getLogger(__name__)

# Every Drive scope shares this prefix ("drive", "drive.file", "drive.readonly",
# "drive.metadata.readonly", ...).
GOOGLE_DRIVE_SCOPE_PREFIX = GOOGLE_DRIVE_SCOPE
GOOGLE_DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
# Scopes that already expose every file in the account, which makes choosing
# individual files with the Picker pointless.
GOOGLE_FULL_DRIVE_SCOPES = frozenset({GOOGLE_DRIVE_SCOPE, GOOGLE_DRIVE_READONLY_SCOPE})
# Identity scopes that may accompany ``drive.file`` on a token handed to a
# browser when the caller asks for a minimal grant.
GOOGLE_PICKER_COMPANION_SCOPES = frozenset(
    {
        "openid",
        "email",
        "profile",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    }
)

GoogleDrivePickerScopeState = Literal[
    "drive_file", "full_drive", "drive_missing", "other"
]

# (app id, OAuth client id) pairs already reported as mismatched, so that one
# misconfiguration is logged once per process rather than on every check.
_REPORTED_APP_ID_MISMATCHES: set[tuple[str, str]] = set()
_REPORTED_APP_ID_MISMATCHES_LOCK = threading.Lock()


@dataclass(frozen=True)
class GooglePickerConfig:
    developer_key: str
    app_id: str


def _granted_scope_set(scopes: str | Iterable[str] | None) -> set[str]:
    if scopes is None:
        return set()
    if isinstance(scopes, str):
        # ``UserOAuth.scope`` stores the granted scopes space separated.
        return set(scopes.split())
    return {
        scope.strip() for scope in scopes if isinstance(scope, str) and scope.strip()
    }


def classify_google_drive_picker_scopes(
    scopes: str | Iterable[str] | None, *, minimal: bool = False
) -> GoogleDrivePickerScopeState:
    """Classify a Google grant for use with the Drive Picker.

    ``scopes`` is either the stored space-separated scope string or an
    iterable of scopes (for example ``Credentials.scopes``).

    * ``drive_missing``: no Drive scope was granted at all.
    * ``full_drive``: the Drive scopes include full Drive access
      (``drive`` or ``drive.readonly``), so there is nothing to pick.
    * ``drive_file``: the Drive scopes are exactly ``{drive.file}``; the Picker
      is how existing files become visible to the app.
    * ``other``: any other mix of Drive scopes.

    With ``minimal=True`` every non-Drive scope must also be one of
    ``GOOGLE_PICKER_COMPANION_SCOPES``; otherwise the grant is ``other``
    (unless it has no Drive scope, which stays ``drive_missing``). Use this
    when the token will be handed to a browser and must carry nothing beyond
    per-file Drive access and basic identity.
    """
    granted = _granted_scope_set(scopes)
    drive_scopes = {
        scope for scope in granted if scope.startswith(GOOGLE_DRIVE_SCOPE_PREFIX)
    }
    if not drive_scopes:
        return "drive_missing"
    if minimal and not (granted - drive_scopes) <= GOOGLE_PICKER_COMPANION_SCOPES:
        return "other"
    if drive_scopes & GOOGLE_FULL_DRIVE_SCOPES:
        return "full_drive"
    if drive_scopes == {GOOGLE_DRIVE_FILE_SCOPE}:
        return "drive_file"
    return "other"


def _google_oauth_client_id(db: Session) -> Optional[str]:
    provider = (
        db.query(OAuthProvider).filter(OAuthProvider.provider_name == "google").first()
    )
    if not provider:
        return os.environ.get("GOOGLE_CLIENT_ID") or None
    client_id = decrypt_value(cast(str, provider.client_id))
    return client_id or os.environ.get("GOOGLE_CLIENT_ID") or None


def google_picker_app_id_matches_client(
    app_id: Optional[str], oauth_client_id: Optional[str]
) -> bool:
    """Return whether the Picker app id belongs to the OAuth client's project.

    Google records the files chosen in the Picker against the Cloud project
    named by ``app_id``. When that is not the project owning the OAuth client
    that issued the access token, the Picker still opens and accepts a
    selection, but the app cannot see the chosen files afterwards.

    This returns ``False`` only when ``app_id`` is set and differs from the
    numeric project prefix of ``oauth_client_id``. The first time a process
    sees a given mismatched pair it logs a warning. A missing app id, or a
    client id without a numeric prefix, cannot be compared and is treated as
    a match.
    """
    explicit_app_id = app_id.strip() if isinstance(app_id, str) else ""
    if not explicit_app_id:
        return True
    client_id = oauth_client_id.strip() if isinstance(oauth_client_id, str) else ""
    client_project = client_id.split("-", 1)[0]
    if not client_project.isdigit() or explicit_app_id == client_project:
        return True
    with _REPORTED_APP_ID_MISMATCHES_LOCK:
        reported = (explicit_app_id, client_id) in _REPORTED_APP_ID_MISMATCHES
        _REPORTED_APP_ID_MISMATCHES.add((explicit_app_id, client_id))
    if reported:
        return False
    logger.warning(
        "Google Picker app id %s does not match the Google OAuth client's "
        "project number %s; files chosen in the Picker would be granted to "
        "another Cloud project.",
        explicit_app_id,
        client_project,
    )
    return False


def get_google_picker_config(
    db: Session, *, oauth_client_id: Optional[str] = None
) -> GooglePickerConfig | None:
    """Resolve the browser-safe Picker key and numeric Cloud project ID.

    The developer key is intentionally separate from ``GOOGLE_API_KEY``. The
    latter is used by server-side services and must never be sent to a browser.
    """
    developer_key = os.environ.get("GOOGLE_PICKER_API_KEY", "").strip()
    app_id = os.environ.get("GOOGLE_PICKER_APP_ID", "").strip()
    if not app_id:
        client_id = oauth_client_id or _google_oauth_client_id(db)
        if client_id:
            candidate = client_id.split("-", 1)[0]
            if candidate.isdigit():
                app_id = candidate
    if not developer_key or not app_id:
        return None
    return GooglePickerConfig(developer_key=developer_key, app_id=app_id)
