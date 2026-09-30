"""Shared Google Drive Picker configuration resolution."""

import os
from dataclasses import dataclass
from typing import Optional, cast

from sqlalchemy.orm import Session

from ...core.utils.encryption import decrypt_value
from ..models.oauth_provider import OAuthProvider


@dataclass(frozen=True)
class GooglePickerConfig:
    developer_key: str
    app_id: str


def _google_oauth_client_id(db: Session) -> Optional[str]:
    provider = (
        db.query(OAuthProvider).filter(OAuthProvider.provider_name == "google").first()
    )
    if not provider:
        return os.environ.get("GOOGLE_CLIENT_ID") or None
    client_id = decrypt_value(cast(str, provider.client_id))
    return client_id or os.environ.get("GOOGLE_CLIENT_ID") or None


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
