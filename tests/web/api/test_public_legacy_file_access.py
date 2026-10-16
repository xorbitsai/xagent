import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from xagent.web.api.files import public_download_file, public_preview_file
from xagent.web.api import websocket as websocket_api


def test_public_download_rejects_legacy_paths():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(public_download_file("user_1/private/report.pdf", db=object()))

    assert exc_info.value.status_code == 404


@pytest.mark.parametrize("requesting_user_id,allowed", [(1, True), (2, False)])
def test_registered_legacy_file_requires_its_owner(
    monkeypatch, tmp_path, requesting_user_id, allowed
):
    target = tmp_path / "report.pdf"
    monkeypatch.setattr(
        websocket_api,
        "_resolve_legacy_preview_storage_path",
        lambda path: (target, "user_1/report.pdf"),
    )
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
        user_id=1, file_id="registered-file"
    )
    factory = MagicMock()
    factory.return_value.__enter__.return_value = session
    monkeypatch.setattr(websocket_api, "get_session_local", lambda: factory)
    if allowed:
        assert (
            websocket_api._register_legacy_preview_isolated(
                "user_1/report.pdf", requesting_user_id
            )
            == "registered-file"
        )
    else:
        with pytest.raises(websocket_api._LegacyPreviewRegistrationError):
            websocket_api._register_legacy_preview_isolated(
                "user_1/report.pdf", requesting_user_id
            )


def test_public_preview_rejects_legacy_paths():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(public_preview_file("user_1/private/report.pdf", db=object()))

    assert exc_info.value.status_code == 404
