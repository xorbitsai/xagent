"""Tests for the /api/kb/collections listing handler.

Focus: the team-owner scans must run concurrently, and each scan must honour
the configurable per-scan timeout instead of a hardcoded value.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionDocumentMetadata,
    CollectionInfo,
    ListCollectionsResult,
)
from xagent.web.api import kb as kb_api
from xagent.web.services.knowledge_base_team_scope import KnowledgeBaseAccess

OWNER_IDS = (2, 3, 4, 5, 6)
SCAN_DELAY_SECONDS = 0.2


def _collection(name: str) -> CollectionInfo:
    """Build a collection that never triggers the document-metadata fallback scan."""
    return CollectionInfo(
        name=name,
        documents=1,
        document_names=[f"{name}.txt"],
        document_metadata=[CollectionDocumentMetadata(filename=f"{name}.txt")],
    )


def _result(*names: str) -> ListCollectionsResult:
    return ListCollectionsResult(
        status="success",
        collections=[_collection(name) for name in names],
        total_count=len(names),
        message="ok",
    )


class _ScanRecorder:
    """Async ``list_collections`` stub that records observed concurrency."""

    def __init__(self, delay: float = SCAN_DELAY_SECONDS) -> None:
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[int] = []

    async def __call__(
        self, user_id: int, is_admin: bool = False
    ) -> ListCollectionsResult:
        self.calls.append(user_id)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        if user_id == 1:
            return _result("personal_kb")
        return _result(f"team_kb_{user_id}")


@pytest.fixture
def team_listing_env(monkeypatch):
    recorder = _ScanRecorder()
    monkeypatch.setattr(kb_api, "list_collections", recorder)
    monkeypatch.setattr(
        kb_api,
        "visible_team_knowledge_bases",
        lambda _db, _user_id: [
            KnowledgeBaseAccess(
                name=f"team_kb_{owner_id}",
                storage_user_id=owner_id,
                team_owned=True,
                can_edit=owner_id % 2 == 0,
                can_delete=False,
            )
            for owner_id in OWNER_IDS
        ],
    )
    return recorder


@pytest.mark.asyncio
async def test_team_owner_scans_run_concurrently(team_listing_env):
    """Each distinct team-KB owner scan must be awaited concurrently, not serially."""
    recorder = team_listing_env
    user = SimpleNamespace(id=1, is_admin=False)

    started = time.perf_counter()
    await kb_api.list_collections_api(_user=user, db=None)
    elapsed = time.perf_counter() - started

    assert recorder.max_in_flight == len(OWNER_IDS)
    serial_cost = SCAN_DELAY_SECONDS * (len(OWNER_IDS) + 1)
    assert elapsed < serial_cost * 0.6, (
        f"team owner scans look serial: {elapsed:.3f}s "
        f"(serial cost would be ~{serial_cost:.3f}s)"
    )


@pytest.mark.asyncio
async def test_team_owner_scan_results_are_merged_per_owner(team_listing_env):
    """Concurrency must not change the per-owner merge semantics."""
    user = SimpleNamespace(id=1, is_admin=False)

    result = await kb_api.list_collections_api(_user=user, db=None)

    by_name = {collection.name: collection for collection in result.collections}
    assert result.total_count == len(by_name)
    assert by_name["personal_kb"].ownership == "personal"
    assert by_name["personal_kb"].storage_user_id == 1
    assert by_name["personal_kb"].can_edit is True
    assert by_name["personal_kb"].can_delete is True

    for owner_id in OWNER_IDS:
        collection = by_name[f"team_kb_{owner_id}"]
        assert collection.ownership == "team"
        assert collection.storage_user_id == owner_id
        assert collection.can_edit is (owner_id % 2 == 0)
        assert collection.can_delete is False


@pytest.mark.asyncio
async def test_scan_timeout_comes_from_config(monkeypatch):
    """The per-scan deadline must be configurable, not a hardcoded constant."""
    monkeypatch.setenv("XAGENT_KB_COLLECTIONS_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(
        kb_api, "visible_team_knowledge_bases", lambda _db, _user_id: []
    )

    async def _slow_scan(user_id: int, is_admin: bool = False):
        await asyncio.sleep(5)
        return _result("never")

    monkeypatch.setattr(kb_api, "list_collections", _slow_scan)
    user = SimpleNamespace(id=1, is_admin=False)

    started = time.perf_counter()
    with pytest.raises(kb_api.HTTPException) as excinfo:
        await kb_api.list_collections_api(_user=user, db=None)
    elapsed = time.perf_counter() - started

    assert excinfo.value.status_code == 503
    assert elapsed < 3, f"configured 1s timeout was not honoured ({elapsed:.3f}s)"
