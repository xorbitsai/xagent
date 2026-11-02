"""Knowledge-base routes bill provider usage to the acting member.

A team collection swaps ``_user`` to the team's storage tenant, so the id the
route hands to storage is not the member who asked. Usage must follow the
member. Each route here also crosses a different boundary the usage has to
survive: the search runs on the request's own context, the upload hops to a
``run_in_executor`` thread, and the web crawl hops to one and starts a new
event loop inside it.

The ingestion is stubbed to bill one embedding call and then fail, which also
covers the other half of the contract: usage recorded before the work raises
is still delivered, once.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import xagent.web.api.kb as kb_module
from xagent.core.model.chat.token_context import MediaCallType, add_media_usage
from xagent.web.api.kb import _EffectiveKnowledgeBaseUser, kb_router
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import get_db
from xagent.web.models.user import User
from xagent.web.services import quota_hooks
from xagent.web.services.knowledge_base_team_scope import KnowledgeBaseAccess

ACTOR_ID = 101
STORAGE_ID = 999


def _mock_db() -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    return db


@pytest.fixture
def hook():
    calls: list = []
    quota_hooks.set_standalone_usage_hook(
        lambda user_id, details, source: calls.append((user_id, details, source))
    )
    yield calls
    quota_hooks.set_standalone_usage_hook(None)


@pytest.fixture
def client():
    actor = User(id=ACTOR_ID, username="member", email="m@example.com", is_admin=False)
    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_current_user] = lambda: actor
    app.dependency_overrides[get_db] = _mock_db
    team = KnowledgeBaseAccess(
        name="team-kb", storage_user_id=STORAGE_ID, team_owned=True
    )
    with (
        patch.object(
            kb_module,
            "_effective_knowledge_base_user",
            return_value=(_EffectiveKnowledgeBaseUser(actor, STORAGE_ID), team),
        ),
        patch.object(kb_module, "_ensure_collection_access", new=AsyncMock()),
        patch.object(kb_module, "get_collection_sync", side_effect=ValueError),
    ):
        yield TestClient(app, raise_server_exceptions=False)


def _bill_then_fail(threads: list[int]):
    def _run(*_args: Any, **kwargs: Any) -> Any:
        threads.append(threading.get_ident())
        # The storage side still gets the tenant.
        assert kwargs["user_id"] == STORAGE_ID
        add_media_usage(call_type=MediaCallType.EMBEDDING, quantity=3, model_id="e")
        raise RuntimeError("provider returned garbage")

    return _run


def _assert_billed_to_actor(calls: list, source: str) -> None:
    [(user_id, [row], delivered_source)] = calls
    assert (user_id, delivered_source) == (ACTOR_ID, source)
    assert (row["call_type"], row["quantity"]) == ("embedding", 3.0)


def test_search_bills_the_member_not_the_team_tenant(client, hook):
    threads: list[int] = []
    with patch.object(
        kb_module, "run_document_search", side_effect=_bill_then_fail(threads)
    ):
        client.post(
            "/api/kb/search",
            data={"collection": "team-kb", "query_text": "q"},
        )

    _assert_billed_to_actor(hook, "kb_search")


def test_upload_ingest_carries_the_scope_into_the_executor(client, hook, tmp_path):
    threads: list[int] = []
    with (
        patch.object(
            kb_module,
            "run_document_ingestion_with_outcome",
            side_effect=_bill_then_fail(threads),
        ),
        patch.object(
            kb_module, "get_upload_path", return_value=str(tmp_path / "f.txt")
        ),
        patch.object(
            kb_module,
            "_upsert_uploaded_file_record",
            return_value=MagicMock(file_id="file-1"),
        ),
    ):
        client.post(
            "/api/kb/ingest",
            data={"collection": "team-kb"},
            files={"file": ("notes.txt", b"abc", "text/plain")},
        )

    assert threads and threads[0] != threading.get_ident()
    _assert_billed_to_actor(hook, "kb_ingest")


def test_web_ingest_carries_the_scope_into_the_executor_loop(client, hook):
    threads: list[int] = []
    bill = _bill_then_fail(threads)

    async def run_web(**kwargs: Any) -> Any:
        return bill(**kwargs)

    with patch.object(kb_module, "run_web_ingestion_with_outcome", side_effect=run_web):
        client.post(
            "/api/kb/ingest-web",
            data={"collection": "team-kb", "start_url": "https://example.com"},
        )

    assert threads, "the crawl never ran"
    _assert_billed_to_actor(hook, "kb_ingest_web")


@pytest.fixture
def refusing_gate():
    calls: list = []

    def gate(user_id: int, source: str) -> str:
        calls.append((user_id, source))
        return "Out of AI credits"

    quota_hooks.set_standalone_gate_hook(gate)
    yield calls
    quota_hooks.set_standalone_gate_hook(None)


def _assert_refused(response: Any, gate_calls: list, source: str) -> None:
    assert response.status_code == 402
    detail = response.json()["detail"]
    assert (detail["code"], detail["source"]) == ("standalone_usage_refused", source)
    # The member is checked, as the member is billed.
    assert gate_calls[0] == (ACTOR_ID, source)


def test_search_refused_by_the_gate_never_searches(client, hook, refusing_gate):
    with patch.object(kb_module, "run_document_search") as search:
        response = client.post(
            "/api/kb/search",
            data={"collection": "team-kb", "query_text": "q"},
        )

    _assert_refused(response, refusing_gate, "kb_search")
    search.assert_not_called()
    assert hook == []


def test_upload_ingest_refused_by_the_gate_writes_nothing(
    client, hook, refusing_gate, tmp_path
):
    target = tmp_path / "f.txt"
    with (
        patch.object(kb_module, "run_document_ingestion_with_outcome") as ingest,
        patch.object(kb_module, "get_upload_path", return_value=str(target)),
        patch.object(kb_module, "_upsert_uploaded_file_record") as upsert,
    ):
        response = client.post(
            "/api/kb/ingest",
            data={"collection": "team-kb"},
            files={"file": ("notes.txt", b"abc", "text/plain")},
        )

    _assert_refused(response, refusing_gate, "kb_ingest")
    assert refusing_gate == [(ACTOR_ID, "kb_ingest")]
    ingest.assert_not_called()
    upsert.assert_not_called()
    assert not target.exists()


def test_ingest_job_refused_by_the_gate_is_never_queued(
    client, refusing_gate, tmp_path
):
    with (
        patch.object(
            kb_module, "get_upload_path", return_value=str(tmp_path / "f.txt")
        ),
        patch.object(
            kb_module, "_ensure_background_job_queue_available_async", new=AsyncMock()
        ),
        patch.object(kb_module, "_enqueue_background_job_or_503_async") as enqueue,
    ):
        response = client.post(
            "/api/kb/ingest/jobs",
            data={"collection": "team-kb"},
            files={"file": ("notes.txt", b"abc", "text/plain")},
        )

    _assert_refused(response, refusing_gate, "kb_ingest_job")
    enqueue.assert_not_called()


def test_web_ingest_refused_by_the_gate_never_crawls(client, hook, refusing_gate):
    with patch.object(kb_module, "run_web_ingestion_with_outcome") as crawl:
        response = client.post(
            "/api/kb/ingest-web",
            data={"collection": "team-kb", "start_url": "https://example.com"},
        )

    _assert_refused(response, refusing_gate, "kb_ingest_web")
    crawl.assert_not_called()
    assert hook == []
