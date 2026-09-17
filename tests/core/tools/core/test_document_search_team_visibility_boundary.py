"""Regression coverage for the document-search team visibility boundary."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool

from xagent.core.tools.core import document_search
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    ListCollectionsResult,
)
from xagent.web.services import knowledge_base_team_scope as kb_scope
from xagent.web.services.knowledge_base_team_scope import (
    KnowledgeBaseAccess,
    KnowledgeBaseVisibilityHook,
)


@pytest.fixture(autouse=True)
def _isolate_team_knowledge_base_visibility_hook():
    """Keep this suite from changing unrelated application hooks.

    Goes through the module's own snapshot/restore primitive rather than
    monkeypatching ``_visibility_hook`` alone: a direct monkeypatch of that
    one attribute does not save or restore the team-keyed slot, so a test
    in this suite that installs a team-keyed hook (or a future test that
    does) would leak it into whatever runs next. The primitive saves every
    slot the module has, known or not yet added.
    """

    with kb_scope.snapshot_knowledge_base_team_hooks():
        kb_scope.set_knowledge_base_team_hooks()
        yield


@pytest.fixture()
def install_visibility_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[KnowledgeBaseVisibilityHook], None]:
    def install(hook: KnowledgeBaseVisibilityHook) -> None:
        # A direct attribute monkeypatch, not the reset-all setter: this
        # fixture must not disturb an access/lifecycle hook another part of
        # the same test installed. The autouse fixture above is what keeps
        # this suite isolated from the rest of the module's state, including
        # the team-keyed slot this fixture never touches.
        monkeypatch.setattr(kb_scope, "_visibility_hook", hook)

    return install


def _collections_result(*collections: CollectionInfo) -> ListCollectionsResult:
    return ListCollectionsResult(
        status="success",
        collections=list(collections),
        total_count=len(collections),
        message="ok",
    )


def _install_collection_listing(monkeypatch: pytest.MonkeyPatch) -> CollectionInfo:
    personal = CollectionInfo(name="personal")
    shared = CollectionInfo(name="shared")

    async def list_for_user(
        user_id: int | None = None, is_admin: bool = False
    ) -> ListCollectionsResult:
        del is_admin
        if user_id == 2:
            return _collections_result(shared)
        return _collections_result(personal)

    monkeypatch.setattr(document_search, "list_collections", list_for_user)
    return personal


def _one_slot_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=2.0,
    )


async def _wait_for(event: threading.Event) -> None:
    completed = await asyncio.to_thread(event.wait, 5)
    if not completed:
        raise TimeoutError("thread event did not settle")


async def _await_without_leaking(task: asyncio.Task[object]) -> None:
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


@pytest.mark.asyncio
async def test_team_visibility_hook_waits_off_loop_and_releases_pool(
    monkeypatch: pytest.MonkeyPatch,
    install_visibility_hook: Callable[[KnowledgeBaseVisibilityHook], None],
) -> None:
    """A blocked hook checkout must leave the loop responsive until release."""
    _install_collection_listing(monkeypatch)
    engine = _one_slot_engine()
    held_connection = engine.connect()
    hook_started = threading.Event()
    hook_closed = threading.Event()
    ticker_advanced = threading.Event()
    observed: dict[str, int] = {}
    loop_thread_id = threading.get_ident()

    def visibility_hook(db: Session | None, user_id: int) -> list[KnowledgeBaseAccess]:
        assert db is None
        assert user_id == 1
        observed["hook_thread_id"] = threading.get_ident()
        hook_started.set()
        try:
            with Session(engine) as session:
                observed["sql_thread_id"] = threading.get_ident()
                session.execute(text("SELECT 1"))
        finally:
            hook_closed.set()
        return [KnowledgeBaseAccess(name="shared", storage_user_id=2)]

    async def tick_after_hook_starts() -> None:
        await _wait_for(hook_started)
        await asyncio.sleep(0)
        ticker_advanced.set()

    install_visibility_hook(visibility_hook)
    ticker = asyncio.create_task(tick_after_hook_starts())
    listing = asyncio.create_task(
        document_search._list_visible_collections(user_id=1, is_admin=False)
    )
    try:
        await _wait_for(hook_started)
        await _wait_for(ticker_advanced)
        assert not listing.done()
        assert engine.pool.checkedout() == 1

        held_connection.close()
        result = await listing
        await _wait_for(hook_closed)

        assert observed["hook_thread_id"] != loop_thread_id
        assert observed["sql_thread_id"] != loop_thread_id
        assert [collection.name for collection in result.collections] == [
            "personal",
            "shared",
        ]
        shared = next(
            collection
            for collection in result.collections
            if collection.name == "shared"
        )
        assert shared.ownership == "team"
        assert shared.storage_user_id == 2
        assert engine.pool.checkedout() == 0
    finally:
        if not held_connection.closed:
            held_connection.close()
        await _await_without_leaking(listing)
        await _await_without_leaking(ticker)
        engine.dispose()


@pytest.mark.asyncio
async def test_cancellation_drains_team_visibility_session_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
    install_visibility_hook: Callable[[KnowledgeBaseVisibilityHook], None],
) -> None:
    """Cancellation must wait for the hook-owned Session to close."""
    _install_collection_listing(monkeypatch)
    engine = _one_slot_engine()
    held_connection = engine.connect()
    hook_started = threading.Event()
    hook_closed = threading.Event()

    def visibility_hook(db: Session | None, user_id: int) -> list[KnowledgeBaseAccess]:
        assert db is None
        assert user_id == 1
        hook_started.set()
        try:
            with Session(engine) as session:
                session.execute(text("SELECT 1"))
        finally:
            hook_closed.set()
        return []

    install_visibility_hook(visibility_hook)
    listing = asyncio.create_task(
        document_search._list_visible_collections(user_id=1, is_admin=False)
    )
    try:
        await _wait_for(hook_started)
        listing.cancel()
        await asyncio.sleep(0)
        assert not listing.done()
        assert not hook_closed.is_set()
        assert engine.pool.checkedout() == 1

        held_connection.close()
        await _wait_for(hook_closed)
        with pytest.raises(asyncio.CancelledError):
            await listing
        assert engine.pool.checkedout() == 0
    finally:
        if not held_connection.closed:
            held_connection.close()
        await _await_without_leaking(listing)
        engine.dispose()


@pytest.mark.asyncio
async def test_team_visibility_hook_error_preserves_identity_and_closes_session(
    monkeypatch: pytest.MonkeyPatch,
    install_visibility_hook: Callable[[KnowledgeBaseVisibilityHook], None],
) -> None:
    """The hook error is not wrapped and its worker-owned Session is released."""
    _install_collection_listing(monkeypatch)
    engine = _one_slot_engine()
    hook_error = RuntimeError("team visibility failed")
    hook_closed = threading.Event()

    def visibility_hook(db: Session | None, user_id: int) -> list[KnowledgeBaseAccess]:
        assert db is None
        assert user_id == 1
        try:
            with Session(engine) as session:
                session.execute(text("SELECT 1"))
                raise hook_error
        finally:
            hook_closed.set()

    install_visibility_hook(visibility_hook)
    try:
        with pytest.raises(RuntimeError) as raised:
            await document_search._list_visible_collections(user_id=1, is_admin=False)
        assert raised.value is hook_error
        await _wait_for(hook_closed)
        assert engine.pool.checkedout() == 0
    finally:
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "is_admin", "install_hook"),
    [
        (None, False, True),
        (1, True, True),
        (1, False, False),
    ],
    ids=["anonymous", "admin", "no-hook"],
)
async def test_visibility_bypasses_keep_personal_collections_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    install_visibility_hook: Callable[[KnowledgeBaseVisibilityHook], None],
    user_id: int | None,
    is_admin: bool,
    install_hook: bool,
) -> None:
    """Anonymous, admin, and unconfigured hook paths keep the personal result."""
    personal = _install_collection_listing(monkeypatch)
    hook_calls: list[tuple[Session | None, int]] = []

    def visibility_hook(
        db: Session | None, hooked_user_id: int
    ) -> list[KnowledgeBaseAccess]:
        hook_calls.append((db, hooked_user_id))
        return [KnowledgeBaseAccess(name="shared", storage_user_id=2)]

    if install_hook:
        install_visibility_hook(visibility_hook)

    result = await document_search._list_visible_collections(
        user_id=user_id, is_admin=is_admin
    )

    assert result.collections == [personal]
    assert result.total_count == 1
    assert hook_calls == []


def test_visibility_test_installation_preserves_unrelated_hooks(
    monkeypatch: pytest.MonkeyPatch,
    install_visibility_hook: Callable[[KnowledgeBaseVisibilityHook], None],
) -> None:
    """The visibility fixture must not reset access or lifecycle hooks."""

    def access_hook(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return None

    def visibility_hook(
        _db: Session | None, _user_id: int
    ) -> list[KnowledgeBaseAccess]:
        return []

    monkeypatch.setattr(kb_scope, "_access_hook", access_hook)
    install_visibility_hook(visibility_hook)

    assert kb_scope._access_hook is access_hook


def test_no_visibility_hook_bypasses_saturated_default_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone collection listing must not queue guaranteed-empty work."""

    personal = _install_collection_listing(monkeypatch)
    worker_started = threading.Event()
    release_worker = threading.Event()

    def occupy_default_executor() -> None:
        worker_started.set()
        release_worker.wait()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop.set_default_executor(executor)
            occupied = loop.run_in_executor(None, occupy_default_executor)
            while not worker_started.is_set():
                await asyncio.sleep(0)

            listing = asyncio.create_task(
                document_search._list_visible_collections(
                    user_id=1,
                    is_admin=False,
                )
            )
            try:
                done, _pending = await asyncio.wait({listing}, timeout=1.0)
                assert listing in done
                result = listing.result()
                assert result.collections == [personal]
                assert result.total_count == 1
            finally:
                release_worker.set()
                await occupied
                await _await_without_leaking(listing)

    asyncio.run(scenario())
