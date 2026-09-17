"""A team-collection ingest job must be owned by the acting member.

Regression for the team-KB upload 404: the endpoint swaps ``_user`` to the
team storage tenant so the worker writes into the shared collection, but the
background job the browser polls under ``/api/jobs/{id}`` must stay owned by
the real member. Otherwise the member's own poll is rejected 404 and the UI
reports a failure for an ingest that actually succeeded.

The two dimensions were each covered before and never crossed: the ingest-job
tests only ever used a personally owned collection, where the swap is a no-op
and the two ids coincide, and the team-scope tests never create a job.
"""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import xagent.web.api.jobs as jobs_module
import xagent.web.api.kb as kb_module
from xagent.web.api.jobs import jobs_router
from xagent.web.api.kb import _EffectiveKnowledgeBaseUser, kb_router
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.background_job import (
    BackgroundJob,
    BackgroundJobStatus,
    BackgroundJobType,
)
from xagent.web.models.database import get_db
from xagent.web.models.user import User
from xagent.web.services.knowledge_base_team_scope import KnowledgeBaseAccess

ACTOR_ID = 101
STORAGE_ID = 999


def _actor() -> User:
    return User(id=ACTOR_ID, username="member", email="m@example.com", is_admin=False)


def _mock_db() -> MagicMock:
    """Pin the one query the ingest paths make, as the sibling ingest tests do.

    Left unpinned, ``UploadedFile ... .first()`` answers a truthy MagicMock and
    the document path silently takes its "file already registered" branch.
    """
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    return db


def _client(actor: User) -> TestClient:
    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_current_user] = lambda: actor
    app.dependency_overrides[get_db] = _mock_db
    return TestClient(app)


def _capture_owner(
    captured: dict[str, int], job_id: str, job_type: BackgroundJobType
) -> Callable[..., BackgroundJob]:
    """Record both ids the endpoint assigns, and hand back a usable job row."""

    def capture_create(
        db: object, *, user_id: int, payload: dict[str, Any], **kwargs: object
    ) -> BackgroundJob:
        captured["job_user_id"] = user_id
        captured["payload_user_id"] = int(payload["user_id"])
        return BackgroundJob(
            id=job_id,
            user_id=user_id,
            job_type=job_type.value,
            queue="kb_ingest",
            status=BackgroundJobStatus.ENQUEUED.value,
            payload=payload,
            attempts=0,
            max_attempts=3,
        )

    return capture_create


async def _passthrough(db: object, job: BackgroundJob) -> BackgroundJob:
    return job


@pytest.mark.parametrize(
    ("owner_id", "expected_status"),
    [(ACTOR_ID, 200), (STORAGE_ID, 404)],
    ids=["owned-by-actor", "owned-by-storage-tenant"],
)
def test_member_polls_their_own_job_but_not_a_tenant_owned_one(
    owner_id: int, expected_status: int
) -> None:
    """The 404 -> 200 transition itself, through the real ``_authorize_job``.

    The ingest tests above pin which id the endpoint assigns; this pins what
    that choice costs the member at the endpoint the browser actually polls.
    ``owner_id=STORAGE_ID`` is the pre-fix state and reproduces the reported
    failure verbatim.
    """
    job = BackgroundJob(
        id="job-1",
        user_id=owner_id,
        job_type=BackgroundJobType.KB_INGEST_DOCUMENT.value,
        queue="kb_ingest",
        status=BackgroundJobStatus.ENQUEUED.value,
        payload={"user_id": STORAGE_ID},
        attempts=0,
        max_attempts=3,
    )
    app = FastAPI()
    app.include_router(jobs_router)
    app.dependency_overrides[get_current_user] = _actor
    app.dependency_overrides[get_db] = _mock_db

    with patch.object(jobs_module, "get_background_job", return_value=job):
        response = TestClient(app).get("/api/jobs/job-1")

    assert response.status_code == expected_status


def test_web_ingest_job_owner_is_actor_not_team_storage_tenant() -> None:
    captured: dict[str, int] = {}
    actor = _actor()
    team_access = KnowledgeBaseAccess(
        name="team-handbook", storage_user_id=STORAGE_ID, team_owned=True
    )

    with (
        patch.object(
            kb_module,
            "_effective_knowledge_base_user",
            return_value=(_EffectiveKnowledgeBaseUser(actor, STORAGE_ID), team_access),
        ),
        patch.object(
            kb_module, "_ensure_background_job_queue_available_async", new=AsyncMock()
        ),
        patch.object(kb_module, "get_collection_sync", side_effect=ValueError),
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=MagicMock(save_collection_config=AsyncMock()),
        ),
        patch.object(
            kb_module,
            "get_non_terminal_background_job_by_idempotency_key",
            return_value=None,
        ),
        patch.object(
            kb_module,
            "create_background_job",
            side_effect=_capture_owner(
                captured, "job-1", BackgroundJobType.KB_INGEST_WEB
            ),
        ),
        patch.object(
            kb_module, "_enqueue_background_job_or_503_async", side_effect=_passthrough
        ),
    ):
        response = _client(actor).post(
            "/api/kb/ingest-web/jobs",
            data={"collection": "team-handbook", "start_url": "https://example.com"},
        )

    assert response.status_code == 202, response.text
    # Owned by the real member, so their own /api/jobs/{id} poll authorizes.
    assert captured["job_user_id"] == ACTOR_ID
    # The id the browser reads back and polls with.
    assert response.json()["user_id"] == ACTOR_ID
    # The worker still writes into the team storage tenant.
    assert captured["payload_user_id"] == STORAGE_ID


def test_document_ingest_job_owner_is_actor_not_team_storage_tenant(
    tmp_path: Path,
) -> None:
    """The path the reported 404 came from: a file upload into a team KB."""
    captured: dict[str, int] = {}
    actor = _actor()
    team_access = KnowledgeBaseAccess(
        name="team-handbook", storage_user_id=STORAGE_ID, team_owned=True
    )
    staged = tmp_path / "staged.txt"

    with (
        patch.object(
            kb_module,
            "_effective_knowledge_base_user",
            return_value=(_EffectiveKnowledgeBaseUser(actor, STORAGE_ID), team_access),
        ),
        patch.object(kb_module, "_ensure_collection_access", new=AsyncMock()),
        patch.object(
            kb_module, "_ensure_background_job_queue_available_async", new=AsyncMock()
        ),
        patch.object(kb_module, "get_collection_sync", side_effect=ValueError),
        patch.object(
            kb_module, "get_upload_path", return_value=str(tmp_path / "target.txt")
        ),
        patch.object(
            kb_module, "_build_background_ingest_staging_path", return_value=staged
        ),
        patch.object(
            kb_module,
            "_copy_upload_file_to_path",
            return_value=SimpleNamespace(total_size=3, sha256="abc"),
        ),
        patch.object(
            kb_module,
            "get_non_terminal_background_job_by_idempotency_key",
            return_value=None,
        ),
        patch.object(
            kb_module,
            "create_background_job",
            side_effect=_capture_owner(
                captured, "job-2", BackgroundJobType.KB_INGEST_DOCUMENT
            ),
        ),
        patch.object(kb_module, "admit_kb_ingest_target") as admit,
        patch.object(kb_module, "_cleanup_background_ingest_staging_file"),
        patch.object(
            kb_module, "_enqueue_background_job_or_503_async", side_effect=_passthrough
        ),
    ):
        response = _client(actor).post(
            "/api/kb/ingest/jobs",
            data={"collection": "team-handbook"},
            files={"file": ("notes.txt", b"abc", "text/plain")},
        )

    assert response.status_code == 202, response.text
    assert captured["job_user_id"] == ACTOR_ID
    assert response.json()["user_id"] == ACTOR_ID
    assert captured["payload_user_id"] == STORAGE_ID
    # The per-target concurrency lock must stay tenant-keyed, or two members
    # writing the same file would each take their own lock.
    assert admit.call_args.kwargs["user_id"] == STORAGE_ID
