"""Tests for cloud-storage API metadata contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.api.cloud_storage import (
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
    monkeypatch.setenv("GOOGLE_API_KEY", "picker-api-key")
    monkeypatch.delenv("GOOGLE_PICKER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_PICKER_APP_ID", raising=False)

    with (
        patch(
            "xagent.web.api.cloud_storage.get_google_credentials",
            return_value=SimpleNamespace(token="short-lived-access-token"),
        ),
        patch(
            "xagent.web.api.cloud_storage.get_google_oauth_config",
            return_value=("1234567890-client.apps.googleusercontent.com", "secret"),
        ),
    ):
        result = await get_google_drive_picker_config(
            db=MagicMock(),
            user=SimpleNamespace(id=1),
        )

    assert result == {
        "access_token": "short-lived-access-token",
        "developer_key": "picker-api-key",
        "app_id": "1234567890",
    }
    assert "refresh_token" not in result
    assert "client_secret" not in result


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
            return_value=SimpleNamespace(token="access-token"),
        ),
        patch(
            "xagent.web.api.cloud_storage.get_google_oauth_config",
            return_value=(None, None),
        ),
        pytest.raises(Exception) as exc_info,
    ):
        await get_google_drive_picker_config(
            db=MagicMock(),
            user=SimpleNamespace(id=1),
        )

    assert getattr(exc_info.value, "status_code", None) == 503
