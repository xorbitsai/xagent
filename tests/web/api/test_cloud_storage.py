"""Tests for cloud-storage API metadata contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.api.cloud_storage import list_google_drive_files


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
