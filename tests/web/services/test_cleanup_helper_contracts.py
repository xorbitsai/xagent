"""Claim races preserve the shared file helper's missing-file contract."""

from tests.web.services.test_complete_upload_cleanup import (
    lifecycle as cleanup_lifecycle_fixture,
)
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.managed_file_ref import ensure_uploaded_file_local_path

lifecycle = cleanup_lifecycle_fixture


def test_claimed_after_lookup_returns_no_local_path(lifecycle):
    sessions, source, *_ = lifecycle
    with sessions() as db:
        stale = db.query(UploadedFile).one()
        with sessions.begin() as claimant:
            claimant.query(UploadedFile).one().storage_status = "compensating"
        assert ensure_uploaded_file_local_path(stale) is None
    assert source.exists()
