"""Tests for cloud-storage API metadata contracts."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi import Response

from xagent.web.api.cloud_storage import (
    _google_credentials_expiry,
    get_google_credentials,
    get_google_drive_picker_config,
    list_google_drive_files,
)


@pytest.mark.asyncio
async def test_google_drive_listing_preserves_resource_keys() -> None:
    list_calls: list[dict[str, object]] = []

    class _ListRequest:
        def execute(self):
            return {
                "files": [
                    {
                        "id": "slides-linked",
                        "name": "Linked Slides",
                        "mimeType": "application/vnd.google-apps.presentation",
                        "resourceKey": "link-resource-key",
                    },
                    {
                        "id": "slides-unlinked",
                        "name": "Unlinked Slides",
                        "mimeType": "application/vnd.google-apps.presentation",
                    },
                ]
            }

    class _FilesResource:
        def list(self, **kwargs):
            list_calls.append(kwargs)
            return _ListRequest()

    class _DriveService:
        def files(self):
            return _FilesResource()

    with (
        patch(
            "xagent.web.api.cloud_storage.get_google_credentials",
            return_value=object(),
        ),
        patch(
            "xagent.web.api.cloud_storage.build",
            return_value=_DriveService(),
        ),
    ):
        files = await list_google_drive_files(
            db=MagicMock(),
            user=SimpleNamespace(id=1),
        )

    assert list_calls[0]["fields"] == (
        "nextPageToken, files(id, name, mimeType, size, modifiedTime, resourceKey)"
    )
    assert files[0]["resourceKey"] == "link-resource-key"
    assert files[1]["resourceKey"] is None


@pytest.mark.asyncio
async def test_google_drive_picker_config_returns_access_token_without_refresh_secret(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GOOGLE_PICKER_API_KEY", "picker-api-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_PICKER_APP_ID", raising=False)

    response = Response()
    with (
        patch(
            "xagent.web.api.cloud_storage.get_google_credentials",
            return_value=SimpleNamespace(
                token="short-lived-access-token",
                scopes={"https://www.googleapis.com/auth/drive.file"},
            ),
        ),
        patch(
            "xagent.web.api.cloud_storage.get_google_oauth_config",
            return_value=("1234567890-client.apps.googleusercontent.com", "secret"),
        ),
    ):
        result = await get_google_drive_picker_config(
            db=MagicMock(),
            user=SimpleNamespace(id=1),
            response=response,
        )

    assert result == {
        "access_token": "short-lived-access-token",
        "developer_key": "picker-api-key",
        "app_id": "1234567890",
    }
    assert "refresh_token" not in result
    assert "client_secret" not in result
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_google_drive_picker_config_rejects_legacy_full_drive_scope(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GOOGLE_PICKER_API_KEY", "picker-api-key")
    monkeypatch.setenv("GOOGLE_PICKER_APP_ID", "1234567890")

    with (
        patch(
            "xagent.web.api.cloud_storage.get_google_credentials",
            return_value=SimpleNamespace(
                token="access-token",
                scopes={"https://www.googleapis.com/auth/drive"},
            ),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await get_google_drive_picker_config(
            db=MagicMock(),
            user=SimpleNamespace(id=1),
            response=Response(),
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_google_drive_picker_config_does_not_fallback_to_server_api_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "server-side-key")
    monkeypatch.delenv("GOOGLE_PICKER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_PICKER_APP_ID", raising=False)

    with (
        patch(
            "xagent.web.api.cloud_storage.get_google_credentials",
            return_value=SimpleNamespace(
                token="access-token",
                scopes={"https://www.googleapis.com/auth/drive.file"},
            ),
        ),
        patch(
            "xagent.web.api.cloud_storage.get_google_oauth_config",
            return_value=("1234567890-client.apps.googleusercontent.com", "secret"),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await get_google_drive_picker_config(
            db=MagicMock(),
            user=SimpleNamespace(id=1),
            response=Response(),
        )

    assert getattr(exc_info.value, "status_code", None) == 503


def test_google_credentials_expiry_is_normalized_to_naive_utc() -> None:
    aware = datetime(2026, 9, 24, 12, 34, tzinfo=timezone.utc)
    assert _google_credentials_expiry(aware) == aware.replace(tzinfo=None)


def test_get_google_credentials_passes_expiry_to_google_auth() -> None:
    expires_at = datetime(2026, 9, 24, 12, 34, tzinfo=timezone.utc)
    account = SimpleNamespace(
        access_token="access-token",
        refresh_token="refresh-token",
        scope="https://www.googleapis.com/auth/drive.file",
        expires_at=expires_at,
    )
    query = MagicMock()
    query.filter.return_value = query
    query.first.return_value = account
    fake_credentials = SimpleNamespace(expired=False, refresh_token="refresh-token")

    with (
        patch(
            "xagent.web.api.cloud_storage.scoped_user_oauth_query",
            return_value=query,
        ),
        patch(
            "xagent.web.api.cloud_storage.get_google_oauth_config",
            return_value=("client-id", "client-secret"),
        ),
        patch(
            "xagent.web.api.cloud_storage.Credentials",
            return_value=fake_credentials,
        ) as credentials_class,
    ):
        get_google_credentials(user_id=1, db=MagicMock())

    assert credentials_class.call_args.kwargs["expiry"] == expires_at.replace(
        tzinfo=None
    )


@pytest.mark.asyncio
async def test_google_drive_picker_config_requires_picker_credentials(
    monkeypatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_PICKER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_PICKER_APP_ID", raising=False)

    with (
        patch(
            "xagent.web.api.cloud_storage.get_google_credentials",
            return_value=SimpleNamespace(
                token="access-token",
                scopes={"https://www.googleapis.com/auth/drive.file"},
            ),
        ),
        patch(
            "xagent.web.api.cloud_storage.get_google_oauth_config",
            return_value=(None, None),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await get_google_drive_picker_config(
            db=MagicMock(),
            user=SimpleNamespace(id=1),
            response=Response(),
        )

    assert getattr(exc_info.value, "status_code", None) == 503
