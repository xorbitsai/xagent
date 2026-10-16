import asyncio

import pytest
from fastapi import HTTPException

from xagent.web.api.files import public_download_file, public_preview_file


def test_public_download_rejects_legacy_paths():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(public_download_file("user_1/private/report.pdf", db=object()))

    assert exc_info.value.status_code == 404


def test_public_preview_rejects_legacy_paths():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(public_preview_file("user_1/private/report.pdf", db=object()))

    assert exc_info.value.status_code == 404
