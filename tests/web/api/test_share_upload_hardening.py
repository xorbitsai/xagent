"""Public upload hardening: storage gate + orphan source marker (#973, PR3)."""

from __future__ import annotations

import io
from collections.abc import Iterator

import pytest

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.orphan_upload_gc import TASKLESS_SHARE_UPLOAD_SOURCE
from xagent.web.services.quota_hooks import set_storage_gate_hook

from .conftest import _admin_headers, _direct_db_session, _setup_admin, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _user_id() -> int:
    _setup_admin()
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == "admin").one().id)
    finally:
        db.close()


def _create_workforce(name: str) -> str:
    headers = _admin_headers()
    owner = _user_id()

    def _agent(agent_name: str, tok: str) -> int:
        db = _direct_db_session()
        try:
            a = Agent(
                user_id=owner,
                name=agent_name,
                description="d",
                instructions="i",
                execution_mode="balanced",
                status=AgentStatus.PUBLISHED,
                share_enabled=True,
                share_token=tok,
            )
            db.add(a)
            db.commit()
            db.refresh(a)
            return int(a.id)
        finally:
            db.close()

    resp = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "upload hardening",
            "manager_agent_id": _agent(f"{name} Mgr", f"{name}-mgr"),
            "workers": [
                {
                    "source_type": "existing",
                    "agent_id": _agent(f"{name} Wrk", f"{name}-wrk"),
                    "alias": "w1",
                    "assignment_instructions": "go",
                    "enabled": True,
                    "sort_order": 1,
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    wf_id = int(resp.json()["id"])
    published = client.post(f"/api/workforces/{wf_id}/publish", headers=headers)
    assert published.status_code == 200, published.text
    share = client.post(f"/api/workforces/{wf_id}/share-link", headers=headers)
    assert share.status_code == 200, share.text
    return str(share.json()["share_token"])


def _guest_headers(token: str) -> dict[str, str]:
    resp = client.post("/api/share/auth", json={"share_token": token})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _taskless_upload(headers: dict[str, str]):
    return client.post(
        "/api/share/files/upload",
        headers=headers,
        data={"task_type": "task"},
        files={"file": ("brief.txt", io.BytesIO(b"brief"), "text/plain")},
    )


@pytest.fixture()
def _clear_storage_gate() -> Iterator[None]:
    """Restore the process-wide storage-gate hook after the test."""
    yield
    set_storage_gate_hook(None)


def test_taskless_share_upload_stamps_source_marker() -> None:
    token = _create_workforce("Marker WF")
    guest = _guest_headers(token)

    resp = _taskless_upload(guest)
    assert resp.status_code == 200, resp.text
    file_id = resp.json()["file_id"]

    db = _direct_db_session()
    try:
        row = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
        assert row.task_id is None
        assert row.upload_source == TASKLESS_SHARE_UPLOAD_SOURCE
    finally:
        db.close()


def test_share_upload_blocked_by_storage_gate_returns_402(
    _clear_storage_gate: None,
) -> None:
    token = _create_workforce("Gate WF")
    guest = _guest_headers(token)

    set_storage_gate_hook(lambda _db, _user_id: "Owner storage limit reached")

    resp = _taskless_upload(guest)
    assert resp.status_code == 402, resp.text
    assert resp.json()["detail"] == "Owner storage limit reached"

    # No orphan row is created when the gate refuses the upload.
    db = _direct_db_session()
    try:
        assert (
            db.query(UploadedFile)
            .filter(UploadedFile.upload_source == TASKLESS_SHARE_UPLOAD_SOURCE)
            .count()
            == 0
        )
    finally:
        db.close()
